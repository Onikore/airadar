"""
Доверительные интервалы для recall на полевых записях.

Вопрос, ради которого написано: 24.1% против 18.8% — это разница между
моделями или шум измерения? Окна перекрываются вдвое и идут подряд по одной
записи, поэтому обычная биномиальная формула врёт: соседние окна почти
одинаковы. Считаем блочный бутстрап (moving block bootstrap) с блоками разной
длины — блок в L окон эквивалентен допущению, что независимы куски записи
длиной L*0.25 с.

Заодно считается метрика, у которой есть разрешение даже когда recall = 0:
  auc_fh — AUC "окна полевой записи против трудных негативов";
  pct    — медианный перцентиль полевого окна в распределении трудных негативов.
Recall при FAR 1% упирается в пол (0.0%) и перестаёт различать модели,
эти две — нет.

Работает на CPU, чтобы не мешать идущему обучению.

    CUDA_VISIBLE_DEVICES= python3 evalx/field_ci.py
"""

import os
import sys
import glob
import wave
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

torch.set_num_threads(3)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SR, WIN = 16000, 8000
N_HARD = 6000            # выборка удержанных трудных негативов для порога
N_BOOT = 4000
BLOCKS = [1, 4, 12, 40]  # длина блока в окнах: 0.25 / 1 / 3 / 10 с

MODELS = [
    ("dronenet_fft512.pt", 512),
    ("dronenet_fft2048_hum08.pt", 2048),
    ("dronenet_fft2048_hum025.pt", 2048),
]


def load_module(n_fft):
    """train.py с подменённым n_fft (как в compare_models.py)."""
    src = open(os.path.join(ROOT, "train.py")).read()
    old = "N_FFT, HOP, N_MELS = 2048, 128, 64"
    assert old in src, "не нашёл строку N_FFT в train.py"
    src = src.replace(old, f"N_FFT, HOP, N_MELS = {n_fft}, 128, 64")
    mod = type(sys)(f"train_{n_fft}")
    mod.__file__ = os.path.join(ROOT, "train.py")
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def field_windows(hop=WIN // 2):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "field", "drone_video*.wav"))):
        w = wave.open(p)
        raw = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        out[os.path.basename(p)] = np.stack(
            [raw[i:i + WIN] for i in range(0, len(raw) - WIN, hop)])
    return out


def moving_block_boot(hits, L, n_boot=N_BOOT, rng=None):
    """CI для доли по зависимому ряду. hits: 0/1 по окнам подряд."""
    rng = rng or np.random.default_rng(0)
    n = len(hits)
    L = min(L, n)
    starts = np.arange(n - L + 1)
    nb = int(np.ceil(n / L))
    pick = rng.choice(starts, size=(n_boot, nb))
    idx = (pick[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = hits[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def kofm(hits, k=4, m=6):
    """Последовательное решение: тревога, если k из m последних окон сработали."""
    if len(hits) < m:
        return np.zeros(0, bool)
    c = np.convolve(hits.astype(int), np.ones(m, int), mode="valid")
    return c >= k


def main():
    fields = field_windows()
    rng = np.random.default_rng(0)
    results = {}

    for fname, n_fft in MODELS:
        path = os.path.join(ROOT, "models", fname)
        if not os.path.exists(path):
            print(f"нет файла {fname}, пропуск")
            continue
        t = load_module(n_fft)
        model = t.DroneNet()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        lm = t.LogMel()

        X, cat, _, hva = t.load_hard()
        sel = hva[np.isin(cat[hva], list(t.HARD_CATS))]
        if len(sel) > N_HARD:
            sel = np.sort(rng.choice(sel, N_HARD, replace=False))
        print(f"\n=== {fname} (n_fft={n_fft}) — трудных негативов: {len(sel)} ===",
              flush=True)
        ph = t._probs(model, lm, X[sel], bs=128)
        thr = {far: float(np.quantile(ph, 1 - far)) for far in (0.05, 0.01, 0.001)}

        for nm, W in fields.items():
            pf = t._probs(model, lm, W, bs=128)
            rec = {}
            for far, th in thr.items():
                hits = (pf > th)
                lo, hi = {}, {}
                for L in BLOCKS:
                    a, b = moving_block_boot(hits.astype(float), L, rng=rng)
                    lo[L], hi[L] = a, b
                rec[far] = (hits.mean(), lo, hi, kofm(hits).mean())
            auc_fh = float((pf[:, None] > ph[None, :]).mean()
                           + 0.5 * (pf[:, None] == ph[None, :]).mean())
            pct = float(np.mean([(ph < v).mean() for v in pf]))
            results[(fname, nm)] = (rec, auc_fh, pct, pf, ph)

            print(f"\n  {nm}  (n={len(W)} окон, {len(W)*0.25+0.25:.0f} с)")
            print(f"    AUC(поле против трудных) = {auc_fh:.4f}   "
                  f"средний перцентиль окна = {pct*100:.1f}%   "
                  f"медиана p = {np.median(pf):.4f}")
            for far in (0.05, 0.01, 0.001):
                r, lo, hi, seq = rec[far]
                ci = "  ".join(f"L={L}: [{lo[L]*100:4.1f}, {hi[L]*100:5.1f}]"
                               for L in BLOCKS)
                print(f"    FAR {far*100:4.1f}%: recall {r*100:5.1f}%  "
                      f"тревога 4из6 {seq*100:5.1f}%   95% CI: {ci}")

    # --- сравнение моделей на одной записи ---
    print("\n=== разница между моделями, блочный бутстрап L=12 (3 с) ===")
    names = [m[0] for m in MODELS if (m[0], list(fields)[0]) in results]
    if len(names) == 2:
        for nm in fields:
            a = results[(names[0], nm)]
            b = results[(names[1], nm)]
            print(f"\n  {nm}")
            for far in (0.05, 0.01):
                ra, rb = a[0][far][0], b[0][far][0]
                # бутстрап разницы: те же блоки, парно по окнам
                pfa, pha = a[3], a[4]
                pfb, phb = b[3], b[4]
                ha = (pfa > np.quantile(pha, 1 - far)).astype(float)
                hb = (pfb > np.quantile(phb, 1 - far)).astype(float)
                n, L = len(ha), 12
                starts = np.arange(n - L + 1)
                nb = int(np.ceil(n / L))
                pick = rng.choice(starts, size=(N_BOOT, nb))
                idx = (pick[:, :, None] + np.arange(L)[None, None, :]).reshape(N_BOOT, -1)[:, :n]
                d = ha[idx].mean(1) - hb[idx].mean(1)
                lo, hi = np.percentile(d, [2.5, 97.5])
                sign = "значима" if lo * hi > 0 else "НЕ значима"
                print(f"    FAR {far*100:4.1f}%: {names[0]} {ra*100:5.1f}%  "
                      f"против {names[1]} {rb*100:5.1f}%   "
                      f"разница {(ra-rb)*100:+5.1f} пп, CI [{lo*100:+.1f}, {hi*100:+.1f}] "
                      f"-> {sign}")


if __name__ == "__main__":
    main()
