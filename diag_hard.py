"""
Две проверки поверх таблицы ложных срабатываний.

1. Пересечение cache_hard с обучающей выборкой DADS.
   Негативы DADS взяты из ESC-50 и UrbanSound8K — из тех же источников, что
   cache_hard. Если те же файлы были в train, то FAR 1.8% на бензопиле измерен
   на обучающих данных и поле не предсказывает. Нарезка в обоих кэшах
   детерминированная (с нулевого отсчёта, тот же шаг, та же нормализация),
   поэтому совпавшие окна должны совпасть байт в байт.

2. Baseline на тупых признаках против тех же трудных негативов.
   diag_leak показал AUC 0.9844 на пяти признаках, не различающих гармоники.
   Если этот baseline срабатывает на бензопиле заметно чаще, чем CNN, значит
   сеть всё же выучила что-то сверх шортката.

CPU-only.
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression

from diag_leak import dumb_features, load as load_dads

ROOT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def load_hard():
    d = os.path.join(ROOT, "cache_hard")
    m = np.load(os.path.join(d, "meta.npz"))
    X = np.memmap(os.path.join(d, "windows.bin"), dtype=np.int16,
                  mode="r", shape=(int(m["n"]), int(m["win"])))
    return X, m["cat"], set(m["hard"].tolist())


def main():
    Xd, yd, gd = load_dads()
    Xh, cat, hard = load_hard()
    # Тот же замороженный сплит, что при обучении — иначе "1. пересечение с
    # train" проверяло бы утечку в разбиении, которого в обучении не существует,
    # и результат ничего не говорил бы про реальную таблицу FAR.
    from train import load_split
    tr, va = np.flatnonzero(load_split() == 0), np.flatnonzero(load_split() == 1)

    # --- 1. пересечение по точному совпадению окон ---
    print("хеширую DADS train...", flush=True)
    tr = np.sort(tr)
    h_tr = set()
    for i in range(0, len(tr), 20000):
        for b in Xd[tr[i:i + 20000]]:
            h_tr.add(hash(b.tobytes()))

    print("хеширую cache_hard...", flush=True)
    hits = np.zeros(len(Xh), bool)
    for i in range(0, len(Xh), 20000):
        chunk = Xh[i:i + 20000]
        hits[i:i + len(chunk)] = [hash(b.tobytes()) in h_tr for b in chunk]

    print(f"\n1. окон cache_hard, найденных в DADS train: "
          f"{hits.sum()}/{len(Xh)} ({100*hits.mean():.1f}%)")
    ih = np.isin(cat, list(hard))
    print(f"   среди трудных: {hits[ih].sum()}/{ih.sum()} ({100*hits[ih].mean():.1f}%)")
    for c in sorted(hard):
        m = cat == c
        if m.sum():
            print(f"     {c:20s} {100*hits[m].mean():5.1f}%  n={m.sum()}")
    if hits[ih].mean() > 0.3:
        print("\n   таблица FAR измерена в основном на обучающих данных — поле не предсказывает.")
    elif hits[ih].mean() < 0.05:
        print("\n   пересечение мало, таблица FAR честная.")

    # --- 2. тупой baseline против тех же негативов ---
    print("\nучу baseline на тупых признаках...", flush=True)
    s = np.sort(rng.choice(tr, 24000, replace=False))
    A, ytr = dumb_features(np.ascontiguousarray(Xd[s])), yd[s]
    mu, sd = A.mean(0), A.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000).fit((A - mu) / sd, ytr)

    # порог baseline подбираем так, чтобы FAR на val DADS был 1% — как у CNN
    va_s = np.sort(rng.choice(va, 24000, replace=False))
    pv = clf.predict_proba((dumb_features(np.ascontiguousarray(Xd[va_s])) - mu) / sd)[:, 1]
    thr = np.quantile(pv[yd[va_s] == 0], 0.99)

    ph = clf.predict_proba((dumb_features(np.ascontiguousarray(Xh)) - mu) / sd)[:, 1]
    print(f"\n2. FAR тупого baseline при том же FAR 1% на DADS val (порог {thr:.4f}):")
    rows = sorted(((c, (ph[cat == c] > thr).mean()) for c in sorted(hard)), key=lambda r: -r[1])
    for c, fa in rows:
        print(f"     {c:20s} {fa*100:5.1f}%")
    print(f"\n   средняя FAR baseline на трудных: {(ph[ih] > thr).mean()*100:.1f}%")
    print("   сравни с 1.8% у CNN. Если baseline заметно хуже — сеть выучила гармоники.")


if __name__ == "__main__":
    main()
