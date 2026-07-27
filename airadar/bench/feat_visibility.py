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


def median_flatten(spec, ker):
    """Стирает гребёнку медианным фильтром по частотной оси (не по времени)."""
    ker = max(3, ker | 1)
    pad = ker // 2
    padded = np.pad(spec, ((pad, pad), (0, 0)), mode="edge")
    view = np.lib.stride_tricks.as_strided(
        padded, shape=(spec.shape[0], ker, spec.shape[1]),
        strides=(padded.strides[0], padded.strides[0], padded.strides[1]))
    return np.median(view, axis=1)


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

    # median_flatten: на плоском (константном) спектре ничего не меняет
    flat = np.full((20, 5), 3.0, np.float32)
    assert np.allclose(median_flatten(flat, ker=5), flat)

    # на спектре с одиночным пиком по частоте — пик стирается медианой
    spec = np.zeros((21, 3), np.float32)
    spec[10] = 10.0                            # пик в середине частотной оси
    out = median_flatten(spec, ker=7)
    assert out[10, 0] < 5.0, out[10, 0]        # пик подавлен
    assert np.allclose(out[0], spec[0])        # края почти не тронуты (edge-паддинг)

    print("feat_visibility selfcheck ok")


def visibility_field(f0_by_name=None):
    """Видимость гребёнки на полевых записях: |ch0(с гребёнкой) - ch0(без)| в дБ.

    f0 записей взяты из evalx/f0_survey.py (задокументировано в README):
    drone_video1.wav ~78 Гц, drone_video2.wav ~62 Гц.
    """
    from airadar.bench.corpus import field_records

    f0_by_name = f0_by_name or {"drone_video1.wav": 78.0, "drone_video2.wav": 62.0}
    cqt = LogCQT()
    out = {}
    for name, wav in field_records().items():
        f0 = f0_by_name.get(name)
        if f0 is None:
            continue
        ch0 = cqt(torch.from_numpy(wav.astype(np.float32))).squeeze(0).numpy()  # [183, T]
        # Ширина медианного окна — диапазон от f0 до 2.5*f0 (запас "1.5*f0"
        # поверх самой f0, как в исходном evalx/feat_visibility.py), в бинах
        # ЛОГ-оси. На линейной STFT-оси ширина в бинах зависела от f0 (там
        # ker = 1.5*f0 / (SR/n_fft)); на лог-оси она НЕ зависит от f0 —
        # ровно то свойство self-similarity, ради которого выбран constant-Q:
        # соотношение f0 -> 2.5*f0 занимает одно и то же число бинов на
        # любой частоте.
        ker_bins = max(3, int(round(BINS_PER_OCTAVE * np.log2(2.5))))   # ~32 бина
        flat = median_flatten(ch0, ker_bins)
        diff_db = np.abs(ch0 - flat).mean(axis=1) * 10.0 / np.log(10.0)  # нат -> дБ
        lo = cqt.frequencies < 300.0
        out[name] = {
            "vis_all_db": float(diff_db.mean()),
            "vis_low_db": float(diff_db[lo].max()) if lo.any() else float("nan"),
            "f0": f0,
        }
    return out


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
