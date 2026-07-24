"""
Выживает ли гребёнка тяжёлого дрона в лог-мел признаке? Модели здесь нет.

Идея. Берём полевую запись, считаем её спектр двумя способами:
  P      — как есть, с гребёнкой;
  P_flat — та же спектральная огибающая, но гребёнка стёрта медианным фильтром
           по частоте с окном шире 1.5*f0.
Оба прогоняем через один и тот же мел-банк. Если log-mel(P) и log-mel(P_flat)
почти совпадают, значит признак физически не содержит информации о гребёнке —
никакое обучение её оттуда не достанет. Разница в дБ и есть "видимость".

Так проверяются гипотезы (a) разрешение по частоте и (c) раскладка мел-полос
без единого прогона обучения.

    python3 evalx/feat_visibility.py
"""

import os
import glob
import wave
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR, WIN, HOP = 16000, 8000, 128

CONFIGS = [
    # n_fft, n_mels, fmin, fmax
    (512,  64, 20, 8000),
    (2048, 64, 20, 8000),
    (2048, 64, 20, 2000),
    (2048, 96, 20, 8000),
    (2048, 128, 20, 8000),
    (4096, 64, 20, 2000),
    (2048, 64, 20, 1000),
]


def mel_filterbank(n_mels, n_fft, fmin, fmax, sr=SR):
    """Та же математика, что в train.mel_filterbank."""
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], max(bins[i + 1], bins[i] + 1), max(bins[i + 2], bins[i + 1] + 2)
        r = min(r, fb.shape[1])
        c = min(c, r - 1)
        fb[i, l:c] = np.linspace(0, 1, c - l, endpoint=False)
        fb[i, c:r] = np.linspace(1, 0, r - c, endpoint=False)
    return fb, pts[1:-1]


def stft_power(w, n_fft):
    """Мощность STFT, эквивалент torch.stft(center=True, hann)."""
    pad = n_fft // 2
    x = np.pad(w, pad, mode="reflect")
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (len(x) - n_fft) // HOP
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, n_fft), strides=(x.strides[0] * HOP, x.strides[0])).copy()
    S = np.fft.rfft(frames * win, axis=1)
    return (S.real ** 2 + S.imag ** 2).T          # [F, T]


def median_flatten(P, ker):
    """Стирает гребёнку: медиана по частоте окном ker бинов."""
    ker = max(3, ker | 1)
    pad = ker // 2
    Q = np.pad(P, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(P)
    # скользящая медиана через stride-трюк (F невелико)
    view = np.lib.stride_tricks.as_strided(
        Q, shape=(P.shape[0], ker, P.shape[1]),
        strides=(Q.strides[0], Q.strides[0], Q.strides[1]))
    np.median(view, axis=1, out=out)
    return out


def field_windows(hop=WIN // 2):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "field", "drone_video*.wav"))):
        w = wave.open(p)
        raw = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        out[os.path.basename(p)] = np.stack(
            [raw[i:i + WIN] for i in range(0, len(raw) - WIN, hop)])
    return out


def visibility(W, f0, n_fft, n_mels, fmin, fmax, n_win=40):
    fb, centers = mel_filterbank(n_mels, n_fft, fmin, fmax)
    sel = np.linspace(0, len(W) - 1, min(n_win, len(W))).astype(int)
    ker = int(round(1.5 * f0 / (SR / n_fft)))
    d_all, d_lo = [], []
    lo_bands = centers < 300.0
    for i in sel:
        w = W[i].astype(np.float32) / 32768.0
        w /= (np.abs(w).max() + 1e-9)
        P = stft_power(w, n_fft)
        Pf = median_flatten(P, ker)
        A = 10 * np.log10(fb @ P + 1e-8)
        B = 10 * np.log10(fb @ Pf + 1e-8)
        d = np.abs(A - B).mean(1)                  # средняя по времени, [n_mels]
        d_all.append(d.mean())
        if lo_bands.any():
            d_lo.append(d[lo_bands].max())
    return (float(np.mean(d_all)), float(np.mean(d_lo)) if d_lo else float("nan"),
            int(lo_bands.sum()), centers)


def comb(f0, weights=(1.0, 0.5, 0.35, 0.2, 0.15, 0.1, 0.08, 0.05), n=WIN):
    t = np.arange(n, dtype=np.float32) / SR
    x = np.zeros(n, np.float32)
    for k, w in enumerate(weights, 1):
        if f0 * k < SR / 2:
            x += w * np.sin(2 * np.pi * f0 * k * t + np.random.rand() * 6.283)
    return x / (np.abs(x).max() + 1e-9)


def hum_confusion():
    """Различает ли признак наводку ЛЭП 50 Гц и гребёнку дрона 62/78 Гц?

    Если косинус близок к 1, то аугментация 'подмешай гул' учит сеть давить
    ровно ту область признака, где живёт тяжёлый дрон: в этом представлении
    они один и тот же вектор.
    """
    np.random.seed(0)
    sigs = {"hum50": comb(50.0, (1.0, 0.5, 0.35, 0.2)),
            "drone62": comb(62.0), "drone78": comb(78.0), "drone200": comb(200.0)}
    print("\n--- различимость наводки ЛЭП и гребёнки дрона в лог-мел (косинус, "
          "только полосы < 300 Гц) ---")
    print(f"{'конфиг':34s}{'полос<300':>10}{'cos(hum,62)':>13}{'cos(hum,78)':>13}"
          f"{'cos(hum,200)':>14}")
    for n_fft, n_mels, fmin, fmax in CONFIGS:
        fb, centers = mel_filterbank(n_mels, n_fft, fmin, fmax)
        lo = centers < 300.0
        V = {}
        for nm, x in sigs.items():
            A = 10 * np.log10(fb @ stft_power(x, n_fft) + 1e-8)
            v = A.mean(1)[lo]
            V[nm] = v - v.mean()
        cs = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        print(f"{f'n_fft={n_fft} n_mels={n_mels} fmax={fmax}':34s}{int(lo.sum()):10d}"
              f"{cs(V['hum50'], V['drone62']):13.3f}{cs(V['hum50'], V['drone78']):13.3f}"
              f"{cs(V['hum50'], V['drone200']):14.3f}")


def main():
    fields = field_windows()
    f0s = {"drone_video.wav": 78.0, "drone_video2.wav": 62.0}   # из f0_survey.py

    print("Видимость гребёнки в лог-мел признаке (дБ). Больше — лучше.")
    print("  vis_all  — средняя по всем полосам разница |log-mel(с гребёнкой) - log-mel(без)|")
    print("  vis_low  — максимум той же разницы среди полос с центром < 300 Гц")
    print("  n<300    — сколько мел-полос попадает ниже 300 Гц")
    print("  n<f0*4   — сколько полос ниже четвёртой гармоники (там вся энергия цели)\n")

    hdr = f"{'конфиг':34s}{'n<300':>7}{'шаг STFT':>10}"
    for nm in fields:
        hdr += f"{nm[:14]:>18}"
    print(hdr)
    print("-" * len(hdr))
    for n_fft, n_mels, fmin, fmax in CONFIGS:
        cells = []
        nlo = None
        for nm, W in fields.items():
            va, vl, nlo, centers = visibility(W, f0s[nm], n_fft, n_mels, fmin, fmax)
            cells.append(f"{va:8.2f}/{vl:7.2f}")
        name = f"n_fft={n_fft} n_mels={n_mels} fmax={fmax}"
        print(f"{name:34s}{nlo:7d}{SR/n_fft:9.2f}м" + "".join(f"{c:>18}" for c in cells))

    print("\n(в каждой ячейке: vis_all / vis_low)")

    print("\n--- центры мел-полос ниже 400 Гц ---")
    for n_fft, n_mels, fmin, fmax in [(2048, 64, 20, 8000), (2048, 64, 20, 2000),
                                      (2048, 128, 20, 8000), (2048, 64, 20, 1000)]:
        _, centers = mel_filterbank(n_mels, n_fft, fmin, fmax)
        c = centers[centers < 400]
        print(f"n_mels={n_mels} fmax={fmax}: {len(c)} полос: " +
              " ".join(f"{v:.0f}" for v in c))

    hum_confusion()


if __name__ == "__main__":
    main()
