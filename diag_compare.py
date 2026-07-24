"""
Сравнение CNN и тупого baseline при ОДИНАКОВОМ recall.

diag_hard сравнивал их при одинаковом FAR на val DADS, что ничего не говорит:
модели ловят разную долю дронов, поэтому их FAR на бензопиле несопоставимы.
Здесь порог каждой модели подбирается так, чтобы она ловила заданную долю
дронов, и только после этого сравнивается частота ложных срабатываний на
трудных негативах.

Это и есть вопрос, который решает судьбу проекта: даёт ли свёрточная сеть
что-то сверх пяти тривиальных признаков.
"""

import os
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression

from train import LogMel, DroneNet, DEV, ROOT
from diag_leak import dumb_features, load as load_dads
from diag_hard import load_hard
from eval import predict

RECALLS = [0.80, 0.90, 0.95, 0.98]
rng = np.random.default_rng(0)


def far_at_recall(p_drone, p_neg, recalls):
    """Порог по доле пойманных дронов -> частота ложных на негативах."""
    out = []
    for r in recalls:
        thr = np.quantile(p_drone, 1 - r)
        out.append((r, thr, float((p_neg > thr).mean())))
    return out


def main():
    Xd, yd, gd = load_dads()
    Xh, cat, hard = load_hard()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
                  .split(Xd, yd, gd))
    tr, va = np.sort(tr), np.sort(va)
    ih = np.isin(cat, list(hard))
    Xva = np.ascontiguousarray(Xd[va])      # ~0.8 ГБ, читаем с диска один раз

    # ---- CNN ----
    ckpt = torch.load(os.path.join(ROOT, "models", "dronenet.pt"), map_location=DEV, weights_only=False)
    model = DroneNet().to(DEV)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logmel = LogMel().to(DEV)
    print("CNN на val DADS...", flush=True)
    pc_va = predict(model, logmel, Xva)
    print("CNN на трудных негативах...", flush=True)
    pc_h = predict(model, logmel, Xh)

    # ---- тупой baseline ----
    print("baseline...", flush=True)
    s = np.sort(rng.choice(tr, 24000, replace=False))
    A = dumb_features(np.ascontiguousarray(Xd[s]))
    mu, sd = A.mean(0), A.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000).fit((A - mu) / sd, yd[s])
    f = lambda W: clf.predict_proba((dumb_features(np.ascontiguousarray(W)) - mu) / sd)[:, 1]
    pb_va, pb_h = f(Xva), f(Xh)

    drone = yd[va] == 1
    neg = yd[va] == 0

    print(f"\n{'recall':>7} | {'CNN: FAR трудные':>17} {'FAR val':>8} | "
          f"{'baseline: FAR трудные':>22} {'FAR val':>8}")
    print("-" * 72)
    for r in RECALLS:
        tc = np.quantile(pc_va[drone], 1 - r)
        tb = np.quantile(pb_va[drone], 1 - r)
        print(f"{r*100:6.0f}% | {(pc_h[ih] > tc).mean()*100:16.1f}% "
              f"{(pc_va[neg] > tc).mean()*100:7.1f}% | "
              f"{(pb_h[ih] > tb).mean()*100:21.1f}% {(pb_va[neg] > tb).mean()*100:7.1f}%")

    # разбивка по категориям при recall 90%
    tc = np.quantile(pc_va[drone], 0.10)
    tb = np.quantile(pb_va[drone], 0.10)
    print(f"\nпри recall 90% — по категориям ({'CNN':>6} / {'baseline':>8}):")
    rows = sorted(((c, (pc_h[cat == c] > tc).mean(), (pb_h[cat == c] > tb).mean())
                   for c in sorted(hard)), key=lambda r: -r[1])
    for c, a, b in rows:
        flag = "  <-- CNN хуже" if a > b + 0.03 else ""
        print(f"   {c:20s} {a*100:6.1f}% / {b*100:7.1f}%{flag}")


if __name__ == "__main__":
    main()
