"""Разводит ли новый CQT-признак гул ЛЭП и гребёнку тяжёлого дрона?

Тот же метод, что уже применён в проекте (evalx/feat_visibility.py,
hum_confusion): синтетические гул и гребёнка дрона проецируются в признак,
косинус между ними на полосах < 300 Гц. На старом mel-признаке (n_fft=2048,
64 mel) косинус hum/drone62 был 0.78 (docs/metrics-plan.md §0.3) — это и
объясняло провал на грузовом дроне: аугментация "подмешай гул" душила ровно
то подпространство признака, где живёт тяжёлая машина.
"""

import sys
import numpy as np
import torch
from airadar.features.cqt import LogCQT, BINS_PER_OCTAVE

SR = 16000


def comb(f0, sr=SR, n=64000, weights=(1.0, 0.5, 0.35, 0.2, 0.15, 0.1, 0.08, 0.05), seed=0):
    """Синтетическая гребёнка с гармониками, случайные фазы (детерминировано seed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / sr
    x = np.zeros(n, np.float32)
    for k, w in enumerate(weights, 1):
        if f0 * k < sr / 2:
            x += w * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    return x / (np.abs(x).max() + 1e-9)


def cosine_hum_drone(seed=0):
    """Косинус между гулом ЛЭП и гребёнками дрона на полосах < 300 Гц.

    Гул генерируется той же формулой, что в train.Augment (50 Гц + 3
    гармоники затухающих весов), гребёнки дрона — на f0 из реальных
    полевых записей (62, 78 Гц) и типичного малого квадрокоптера (200 Гц).
    """
    cqt = LogCQT()
    lo = cqt.frequencies < 300.0
    assert lo.any(), "нет ни одного бина ниже 300 Гц — проверить FMIN/N_BINS"

    sigs = {
        "hum": comb(50.0, weights=(1.0, 0.5, 0.35, 0.2), seed=seed),
        "drone62": comb(62.0, seed=seed + 1),
        "drone78": comb(78.0, seed=seed + 2),
        "drone200": comb(200.0, seed=seed + 3),
    }
    vecs = {}
    for name, x in sigs.items():
        ch0 = cqt(torch.from_numpy(x)).squeeze(0).numpy()   # [183, T]
        v = ch0.mean(axis=1)[lo]                             # усредняем по времени
        vecs[name] = v - v.mean()                             # центрируем, как в исходном методе

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    return {
        "n_bands_lt_300": int(lo.sum()),
        "hum_vs_62": cos(vecs["hum"], vecs["drone62"]),
        "hum_vs_78": cos(vecs["hum"], vecs["drone78"]),
        "hum_vs_200": cos(vecs["hum"], vecs["drone200"]),
    }


def selfcheck():
    # comb() детерминирован по seed снаружи — здесь просто форма и энергия
    x = comb(78.0)
    assert x.shape == (64000,)
    assert np.abs(x).max() <= 1.0 + 1e-6

    # разные f0 дают разные сигналы
    assert not np.allclose(comb(50.0), comb(78.0))

    res = cosine_hum_drone()
    assert res["n_bands_lt_300"] > 8, res   # больше, чем 8 полос старого mel-банка
    for k in ("hum_vs_62", "hum_vs_78", "hum_vs_200"):
        assert -1.0 - 1e-6 <= res[k] <= 1.0 + 1e-6, (k, res[k])

    print("feat_visibility selfcheck ok")


def report():
    res = cosine_hum_drone()
    print(f"полос < 300 Гц: {res['n_bands_lt_300']} (было 8 на старом mel-банке)")
    print(f"cos(гул, дрон 62 Гц):  {res['hum_vs_62']:.3f}  (было 0.78-0.90)")
    print(f"cos(гул, дрон 78 Гц):  {res['hum_vs_78']:.3f}")
    print(f"cos(гул, дрон 200 Гц): {res['hum_vs_200']:.3f}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    elif "--report" in sys.argv:
        report()
    else:
        sys.exit("--selfcheck или --report")
