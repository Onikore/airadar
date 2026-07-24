"""
Диагностика утечки. AUC 0.99 на первой эпохе обычно значит, что модель нашла
короткий путь, а не выучила звук дрона.

Три независимые проверки:
  1. Точные дубликаты окон между train и val.
  2. Ближайший сосед: если у окна из val есть почти идентичное окно в train —
     это один и тот же кусок записи по обе стороны сплита.
  3. Baseline на тупых признаках (громкость, ZCR, спектральный центроид).
     Эти признаки физически не могут отличить дрон от газонокосилки.
     Высокий AUC здесь = в датасете есть шорткат.

Всё на CPU, GPU не трогает — можно гонять параллельно с обучением.
"""

import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
SR, N = 16000, 12000          # окон на класс в выборке
rng = np.random.default_rng(0)


def load():
    d = os.path.join(ROOT, "cache_dads")
    m = np.load(os.path.join(d, "meta.npz"))
    X = np.memmap(os.path.join(d, "windows.bin"), dtype=np.int16,
                  mode="r", shape=(int(m["n"]), int(m["win"])))
    return X, m["y"].astype(int), m["group"]


def dumb_features(W):
    """Признаки, не несущие информации о том, дрон это или нет."""
    w = W.astype(np.float32) / 32768.0
    w = w / (np.abs(w).max(1, keepdims=True) + 1e-8)
    spec = np.abs(np.fft.rfft(w * np.hanning(w.shape[1]), axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(w.shape[1], 1 / SR)
    p = spec / spec.sum(1, keepdims=True)
    return np.column_stack([
        np.log(np.sqrt((w ** 2).mean(1)) + 1e-9),                  # громкость
        (np.diff(np.signbit(w), axis=1) != 0).mean(1),             # zero-crossing rate
        (p * freqs).sum(1),                                        # спектральный центроид
        np.exp(np.log(p).mean(1)) / p.mean(1),                     # спектральная плоскостность
        np.sqrt((p * (freqs - (p * freqs).sum(1, keepdims=True)) ** 2).sum(1)),  # разброс
    ])


def melmean(W, nb=48):
    """Грубый отпечаток окна: усреднённый по времени лог-спектр в 48 полосах."""
    w = W.astype(np.float32) / 32768.0
    w = w / (np.abs(w).max(1, keepdims=True) + 1e-8)
    spec = np.abs(np.fft.rfft(w, axis=1))
    edges = np.geomspace(20, SR / 2, nb + 1)
    idx = np.clip((edges / (SR / 2) * (spec.shape[1] - 1)).astype(int), 0, spec.shape[1] - 1)
    f = np.stack([np.log(spec[:, idx[i]:max(idx[i + 1], idx[i] + 1)].mean(1) + 1e-9)
                  for i in range(nb)], axis=1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)


def main():
    X, y, g = load()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(X, y, g))
    print(f"групп всего: {len(np.unique(g))}  train: {len(np.unique(g[tr]))}  val: {len(np.unique(g[va]))}")

    tr_s = np.sort(rng.choice(tr, min(N * 2, len(tr)), replace=False))
    va_s = np.sort(rng.choice(va, min(N * 2, len(va)), replace=False))
    Wtr, Wva = np.ascontiguousarray(X[tr_s]), np.ascontiguousarray(X[va_s])
    ytr, yva = y[tr_s], y[va_s]

    # 1. точные дубликаты
    htr = {hash(b.tobytes()) for b in Wtr}
    dup = sum(hash(b.tobytes()) in htr for b in Wva)
    print(f"\n1. точных дубликатов val в train: {dup}/{len(Wva)} ({100*dup/len(Wva):.2f}%)")

    # 2. ближайший сосед
    Ftr, Fva = melmean(Wtr), melmean(Wva)
    best = np.empty(len(Fva), np.float32)
    for i in range(0, len(Fva), 512):
        best[i:i + 512] = (Fva[i:i + 512] @ Ftr.T).max(1)
    print("\n2. косинусная близость val к ближайшему окну train:")
    for q in (50, 90, 99):
        print(f"     p{q}: {np.percentile(best, q):.4f}")
    print(f"     доля > 0.999: {(best > 0.999).mean()*100:.1f}%   > 0.99: {(best > 0.99).mean()*100:.1f}%")

    # 3. baseline на тупых признаках
    Atr, Ava = dumb_features(Wtr), dumb_features(Wva)
    mu, sd = Atr.mean(0), Atr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000).fit((Atr - mu) / sd, ytr)
    auc = roc_auc_score(yva, clf.predict_proba((Ava - mu) / sd)[:, 1])
    print(f"\n3. AUC на 5 тупых признаках (громкость/ZCR/центроид/плоскостность/разброс): {auc:.4f}")
    if auc > 0.85:
        print("   в датасете есть шорткат — дрон отличим без анализа гармоник.")
    elif auc > 0.7:
        print("   частичный шорткат, но не решающий.")
    else:
        print("   шортката нет, задача требует спектральных деталей.")


if __name__ == "__main__":
    main()
