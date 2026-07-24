"""
Спектрограммы и усреднённые спектры полевых записей.

Нужно, чтобы глазами увидеть то, что пока выведено косвенно: где именно сидит
гармоническая гребёнка грузового дрона и насколько она совпадает с полосой,
которую аугментация гула ЛЭП объявляет помехой (50/100/150/200 Гц).

    python3 spectrum.py                       # все drone_video*.wav
    python3 spectrum.py --fmax 2000           # шире по частоте
    python3 spectrum.py --file запись.wav
"""

import os
import sys
import glob
import wave
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 16000
HUM = [50, 100, 150, 200]           # частоты, которые аугментация учит игнорировать


def read(path):
    w = wave.open(path)
    if w.getnchannels() != 1 or w.getframerate() != SR:
        sys.exit(f"{path}: нужен моно {SR} Гц")
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def spectrogram(x, n_fft, hop):
    n = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft)
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=1)).T, np.fft.rfftfreq(n_fft, 1 / SR)


def find_harmonics(freqs, spec, fmin=40, fmax=400, n=6):
    """Основная частота по гребёнке: перебираем кандидатов, суммируем энергию
    в их гармониках, берём лучший. Устойчивее, чем брать самый громкий пик —
    у дрона основная нередко тише второй гармоники."""
    band = (freqs >= fmin) & (freqs <= fmax)
    best, best_score = None, -1
    for f0 in freqs[band]:
        if f0 < fmin:
            continue
        score = sum(spec[np.argmin(np.abs(freqs - f0 * k))] for k in range(1, n + 1)
                    if f0 * k < freqs[-1])
        if score > best_score:
            best, best_score = f0, score
    return best


def plot(path, args):
    x = read(path)
    name = os.path.basename(path)

    spec, freqs = spectrogram(x, args.n_fft, args.hop)
    db = 20 * np.log10(spec + 1e-10)
    dur = len(x) / SR
    sel = freqs <= args.fmax

    avg = spec.mean(axis=1)
    f0 = find_harmonics(freqs, avg)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5),
                           gridspec_kw={"width_ratios": [2, 1]})

    m = db[sel]
    ax[0].imshow(m, origin="lower", aspect="auto", cmap="magma",
                 extent=[0, dur, 0, args.fmax],
                 vmin=np.percentile(m, 40), vmax=np.percentile(m, 99.8))
    for h in HUM:
        if h <= args.fmax:
            ax[0].axhline(h, color="#4da6ff", lw=0.8, ls=":", alpha=.8)
    ax[0].set(xlabel="время, с", ylabel="частота, Гц",
              title=f"{name} — спектрограмма (n_fft={args.n_fft}, шаг {SR/args.n_fft:.1f} Гц)")

    ax[1].plot(avg[sel], freqs[sel], lw=.9, color="#222")
    for h in HUM:
        if h <= args.fmax:
            ax[1].axhline(h, color="#4da6ff", lw=1, ls=":",
                          label="гул ЛЭП 50/100/150/200 Гц" if h == 50 else None)
    if f0:
        for k in range(1, 9):
            if f0 * k <= args.fmax:
                ax[1].axhline(f0 * k, color="#e8590c", lw=1, ls="--", alpha=.75,
                              label=f"гребёнка дрона, основная {f0:.0f} Гц" if k == 1 else None)
    ax[1].set(xlabel="средняя амплитуда", ylim=(0, args.fmax),
              title="усреднённый спектр")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].grid(alpha=.25)

    fig.tight_layout()
    out = os.path.join(ROOT, "field", f"spectrum_{os.path.splitext(name)[0]}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)

    coll = [k for k in range(1, 9) if f0 and any(abs(f0 * k - h) < 6 for h in HUM)]
    print(f"{name}: основная {f0:.1f} Гц   -> {out}")
    if coll:
        print(f"   гармоники {', '.join(f'{f0*k:.0f} Гц' for k in coll)} "
              f"совпадают с линиями гула, которые аугментация учит игнорировать")


def selfcheck():
    """Поиск основной частоты должен находить гребёнку, а не самый громкий пик."""
    t = np.arange(SR) / SR
    # основная 70 Гц тише своей второй гармоники — наивный argmax ошибётся
    x = 0.3 * np.sin(2 * np.pi * 70 * t) + 1.0 * np.sin(2 * np.pi * 140 * t) \
        + 0.5 * np.sin(2 * np.pi * 210 * t) + 0.4 * np.sin(2 * np.pi * 280 * t)
    spec, freqs = spectrogram(x, 4096, 1024)
    f0 = find_harmonics(freqs, spec.mean(axis=1))
    assert abs(f0 - 70) < 3, f0
    assert freqs[np.argmax(spec.mean(axis=1))] > 100      # громче всего 140 Гц
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--fmax", type=float, default=800)
    ap.add_argument("--n-fft", dest="n_fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=512)
    args = ap.parse_args()

    files = [args.file] if args.file else sorted(glob.glob(os.path.join(ROOT, "field", "drone_video*.wav")))
    if not files:
        sys.exit("нет файлов field/drone_video*.wav")
    for f in files:
        plot(f, args)


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
