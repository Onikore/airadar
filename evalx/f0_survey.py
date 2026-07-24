"""
Разведка: какая основная частота (f0) гармонической гребёнки живёт в данных.

Никакой модели, только сигнальная обработка. Отвечает на два вопроса:

  1. Есть ли в DADS вообще окна с f0 < 100 Гц (то есть тяжёлые дроны)?
     Если есть — из них можно нарезать страту и мерить recall по f0-полосам,
     не дожидаясь новых полевых записей.
  2. Какие трудные негативы имеют гребёнку в той же полосе, что грузовой дрон?
     Это готовый пул "самых трудных негативов" для ненасыщающейся FAR.

Оценка f0 — гармоническая сумма логарифма спектра (HPS в лог-домене) по
кандидатам 40..400 Гц. salience = насколько гребёнка выступает над фоном:
разница между максимумом гармонической суммы и её медианой по кандидатам.

CPU-only, читает кэши последовательно, GPU не трогает.
"""

import os
import sys
import glob
import wave
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR, WIN = 16000, 8000
NFFT = 32768                      # zero-pad: шаг 0.49 Гц (окно 0.5 с даёт 2 Гц реально)
F0_LO, F0_HI, F0_STEP = 40.0, 400.0, 0.5
NHARM = 8


def _cand_index():
    f0 = np.arange(F0_LO, F0_HI + 1e-9, F0_STEP)
    k = np.arange(1, NHARM + 1)
    idx = np.rint(f0[:, None] * k[None, :] * NFFT / SR).astype(np.int64)
    idx = np.clip(idx, 0, NFFT // 2)
    return f0, idx


F0_CAND, CAND_IDX = _cand_index()
WINDOW = np.hanning(WIN).astype(np.float32)


def f0_salience(W, bs=256):
    """W: [N, WIN] int16 -> (f0_hat [N], salience [N] в дБ, band_lo_frac [N]).

    band_lo_frac — доля энергии ниже 300 Гц, грубый признак "низкочастотная цель".
    """
    f0s, sal, blo = [], [], []
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    lo = freqs < 300.0
    band = (freqs >= 40.0) & (freqs <= 4000.0)
    for i in range(0, len(W), bs):
        w = np.ascontiguousarray(W[i:i + bs]).astype(np.float32) / 32768.0
        w -= w.mean(1, keepdims=True)
        w /= (np.abs(w).max(1, keepdims=True) + 1e-9)
        P = np.abs(np.fft.rfft(w * WINDOW, n=NFFT, axis=1)) ** 2 + 1e-12
        L = 10.0 * np.log10(P)
        S = L[:, CAND_IDX].mean(2)                  # [B, n_cand] гармоническая сумма
        j = S.argmax(1)
        f0s.append(F0_CAND[j])
        sal.append(S.max(1) - np.median(S, axis=1))
        blo.append(P[:, lo].sum(1) / (P[:, band].sum(1) + 1e-12))
    return np.concatenate(f0s), np.concatenate(sal), np.concatenate(blo)


def field_windows(pattern="field/drone_video*.wav", hop=WIN // 2):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        w = wave.open(p)
        raw = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        out[os.path.basename(p)] = np.stack(
            [raw[i:i + WIN] for i in range(0, len(raw) - WIN, hop)])
    return out


def hist(name, f0, sal, sal_min=6.0):
    edges = [40, 60, 80, 100, 130, 170, 220, 300, 400.1]
    ok = sal >= sal_min
    print(f"\n{name}: n={len(f0)}, с выраженной гребёнкой (salience>={sal_min} дБ): "
          f"{ok.sum()} ({100*ok.mean():.1f}%)")
    print("   f0-полоса     всего   с гребёнкой")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f0 >= a) & (f0 < b)
        print(f"   {a:3.0f}-{b:3.0f} Гц   {m.sum():7d}   {(m & ok).sum():7d}")


def sample(idx, n, rng):
    if len(idx) <= n:
        return np.sort(idx)
    return np.sort(rng.choice(idx, n, replace=False))


def main():
    rng = np.random.default_rng(0)

    print("=== полевые записи (проверка оценщика: ожидаем ~59 и ~78 Гц) ===")
    for nm, W in field_windows().items():
        f0, sal, blo = f0_salience(W)
        print(f"\n{nm}: n={len(W)}  медиана f0={np.median(f0):.1f} Гц  "
              f"медиана salience={np.median(sal):.1f} дБ  "
              f"медиана доли энергии <300 Гц={np.median(blo):.2f}")
        q = np.percentile(f0, [10, 25, 50, 75, 90])
        print("   f0 перцентили 10/25/50/75/90: " + " ".join(f"{v:.1f}" for v in q))
        hist("   по окнам", f0, sal)

    d = os.path.join(ROOT, "cache_dads")
    m = np.load(os.path.join(d, "meta.npz"))
    y = m["y"].astype(int)
    X = np.memmap(os.path.join(d, "windows.bin"), dtype=np.int16,
                  mode="r", shape=(int(m["n"]), int(m["win"])))
    n_take = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    for lab, nm in ((1, "DADS дрон"), (0, "DADS фон")):
        idx = sample(np.flatnonzero(y == lab), n_take, rng)
        f0, sal, blo = f0_salience(X[idx])
        hist(f"=== {nm} (выборка) ===", f0, sal)
        np.savez(os.path.join(ROOT, "evalx", f"f0_dads_{lab}.npz"),
                 idx=idx, f0=f0, sal=sal, blo=blo)

    dh = os.path.join(ROOT, "cache_hard")
    mh = np.load(os.path.join(dh, "meta.npz"))
    cat = mh["cat"]
    Xh = np.memmap(os.path.join(dh, "windows.bin"), dtype=np.int16,
                   mode="r", shape=(int(mh["n"]), int(mh["win"])))
    hard = set(mh["hard"].tolist())
    ih = sample(np.flatnonzero(np.isin(cat, list(hard))), n_take, rng)
    f0h, salh, bloh = f0_salience(Xh[ih])
    hist("=== трудные негативы (выборка) ===", f0h, salh)
    np.savez(os.path.join(ROOT, "evalx", "f0_hard.npz"),
             idx=ih, f0=f0h, sal=salh, blo=bloh, cat=cat[ih])

    print("\n--- негативы с гребёнкой в полосе грузового дрона (50-110 Гц, sal>=6 дБ) ---")
    sel = (f0h >= 50) & (f0h <= 110) & (salh >= 6.0)
    print(f"всего: {sel.sum()} из {len(f0h)} ({100*sel.mean():.1f}%)")
    u, c = np.unique(cat[ih][sel], return_counts=True)
    for k, v in sorted(zip(u, c), key=lambda t: -t[1])[:12]:
        tot = (cat[ih] == k).sum()
        print(f"   {k:20s} {v:5d} из {tot:5d}  ({100*v/max(tot,1):.0f}%)")


if __name__ == "__main__":
    main()
