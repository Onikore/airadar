"""
Оценка детектора в операционных терминах.

Accuracy для этой задачи бесполезна: классы несбалансированы, а цена ошибок
несимметрична. В поле важно другое — сколько дронов ловим при том уровне
ложных тревог, который оператор готов терпеть. Система, которая срабатывает
на каждый порыв ветра, будет выключена в первый же день.

Поэтому считаем recall при фиксированном FAR и подбираем рабочий порог.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve

from train import LogMel, DroneNet, load_cache, DEV, ROOT

FAR_TARGETS = [0.001, 0.01, 0.05]      # доля ложных тревог на окно 0.5 с


def predict(model, logmel, X, bs=512):
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            wav = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(DEV).float() / 32768.0
            wav = wav / (wav.abs().amax(1, keepdim=True) + 1e-8)
            probs.append(torch.sigmoid(model(logmel(wav).unsqueeze(1))).cpu().numpy())
    return np.concatenate(probs)


def report(name, y, p):
    if len(np.unique(y)) < 2:
        print(f"\n[{name}] пропущено — один класс")
        return {}
    auc = roc_auc_score(y, p)
    fpr, tpr, thr = roc_curve(y, p)
    print(f"\n[{name}]  n={len(y)}  дрон={int(y.sum())}  AUC={auc:.4f}")
    out = {"auc": float(auc), "n": int(len(y)), "operating_points": {}}
    for far in FAR_TARGETS:
        i = int(np.searchsorted(fpr, far, side="right") - 1)
        i = max(i, 0)
        print(f"  FAR {far*100:5.1f}%  ->  recall {tpr[i]*100:5.1f}%   порог {thr[i]:.4f}")
        out["operating_points"][str(far)] = {"recall": float(tpr[i]), "threshold": float(thr[i])}
    # порог с лучшим балансом (Youden J) — разумный дефолт, если FAR не задан
    j = int(np.argmax(tpr - fpr))
    print(f"  баланс (Youden)  recall {tpr[j]*100:5.1f}%  FAR {fpr[j]*100:5.1f}%  порог {thr[j]:.4f}")
    out["balanced"] = {"recall": float(tpr[j]), "far": float(fpr[j]), "threshold": float(thr[j])}
    return out


def hard_negatives(model, logmel, thr, thr_name):
    """Доля ложных срабатываний по категориям шума.

    Общий AUC на DADS ничего не говорит о поле: там негативы — лай и звонки.
    Здесь спрашиваем то, что решает судьбу системы — молчит ли детектор на
    бензопиле, генераторе, вертолёте и ветре. Всё, что выше ~20%, означает
    непрерывный вой сирены на реальной точке.
    """
    from train import load_hard

    # Только удержанная половина: вторая ушла в обучение, на ней таблица
    # измеряла бы ложные срабатывания на обучающих данных.
    X, cat, _, hva = load_hard()
    if X is None:
        print("\ncache_hard нет — сначала prep_hard.py")
        return
    m = np.load(os.path.join(ROOT, "cache_hard", "meta.npz"))
    X, cat = X[hva], cat[hva]
    print(f"\n(удержанная половина cache_hard: {len(cat)} окон)")
    p = predict(model, logmel, X)

    print(f"\n=== ложные срабатывания при пороге {thr_name} ({thr:.4f}) ===")
    hard = set(m["hard"].tolist())
    rows = [(c, (p[cat == c] > thr).mean(), int((cat == c).sum())) for c in np.unique(cat)]
    rows.sort(key=lambda r: -r[1])

    print("\n  трудные (механический шум и погода):")
    for c, fa, n in rows:
        if c in hard:
            print(f"    {c:20s} {fa*100:5.1f}%   n={n}")
    print("\n  худшие из остальных:")
    for c, fa, n in [r for r in rows if r[0] not in hard][:6]:
        print(f"    {c:20s} {fa*100:5.1f}%   n={n}")

    h = np.isin(cat, list(hard))
    print(f"\n  средняя FAR на трудных: {(p[h] > thr).mean()*100:.1f}%")
    print(f"  средняя FAR на прочих:  {(p[~h] > thr).mean()*100:.1f}%")
    return {c: float(fa) for c, fa, _ in rows}


def main():
    ckpt = torch.load(os.path.join(ROOT, "models", "dronenet.pt"), map_location=DEV, weights_only=False)
    model = DroneNet().to(DEV)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logmel = LogMel().to(DEV)

    X, y, groups, synth = load_cache()
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
                 .split(X, y, groups))
    va = np.sort(va)
    Xva, yva, sva = np.ascontiguousarray(X[va]), y[va], synth[va]

    p = predict(model, logmel, Xva)

    res = {}
    res["all"] = report("всё вместе", yva, p)
    # реальные записи дрона против всего фона — то, что поедет в поле
    real = (yva == 0) | (~sva)
    res["real_only"] = report("только реальные записи дрона", yva[real], p[real])
    # синтетика отдельно: если тут AUC сильно выше — модель ловит артефакты Griffin-Lim
    syn = (yva == 0) | sva
    res["synth_only"] = report("только синтетика (Griffin-Lim)", yva[syn], p[syn])

    gap = res["synth_only"].get("auc", 0) - res["real_only"].get("auc", 0)
    if res["synth_only"]:
        print(f"\nразрыв синтетика-реальность: {gap:+.4f}")
        if gap > 0.05:
            print("  модель частично ловит артефакты генерации, а не дрон.")

    op = res["all"].get("operating_points", {}).get("0.01")
    if op:
        res["hard_negatives"] = hard_negatives(model, logmel, op["threshold"], "FAR 1%")

    with open(os.path.join(ROOT, "eval.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\nсохранено: eval.json")


if __name__ == "__main__":
    main()
