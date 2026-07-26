"""Recall по полосам основной частоты, отчёт по худшей полосе.

Все агрегаты проекта — средние по распределению, в котором интересующий
случай составляет проценты: 86% окон дрона в DADS это квадрокоптеры с
f0 > 100 Гц с близкой дистанции. Среднее по такому пулу обязано насытиться.
Худшая полоса — нет: в низкой страте порядка 21 600 окон, у неё свой узкий
доверительный интервал, и видно, растёт ли она отдельно от остальных.
"""

import os
import sys
import numpy as np

F0_BANDS = ((40.0, 80.0), (80.0, 120.0), (120.0, 200.0), (200.0, 300.0), (300.0, 1e9))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F0_DRONE_NPZ = os.path.join(ROOT, "evalx", "f0_dads_1.npz")


def _name(b):
    lo, hi = F0_BANDS[b]
    return f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f"{lo:.0f}+"


def band_of(f0):
    """Индекс полосы для каждой оценки f0. -1 — вне диапазона 40..inf."""
    f0 = np.asarray(f0, np.float64)
    out = np.full(len(f0), -1, np.int64)
    for b, (lo, hi) in enumerate(F0_BANDS):
        out[(f0 >= lo) & (f0 < hi)] = b
    return out


def recall_by_band(logits, y, f0, thr):
    """Recall отдельно по каждой f0-полосе. Пустые полосы не отчитываются."""
    logits, y = np.asarray(logits), np.asarray(y)
    b = band_of(f0)
    out = {}
    for k in range(len(F0_BANDS)):
        sel = (b == k) & (y == 1)
        if sel.sum() == 0:
            continue
        out[_name(k)] = float((logits[sel] >= thr).mean())
    return out


def worst_band(rec):
    """Отчётная величина — худшая полоса, а не средняя по полосам."""
    if not rec:
        return "", float("nan")
    k = min(rec, key=rec.get)
    return k, rec[k]


def load_f0_estimates(min_salience=6.0):
    """Оценки f0 по окнам ДРОНА из cache_dads.

    Возвращает (idx, f0) — индексы окон в кэше и основную частоту.
    idx обязателен: f0_survey считает выборку, а не весь кэш, и без индексов
    оценки не сопоставить с логитами — страта тихо съедет на другие окна.

    Отсекаются окна со слабой гребёнкой: оценщик f0 на них возвращает шум,
    и такая страта измеряла бы качество оценщика, а не детектора.
    """
    if not os.path.exists(F0_DRONE_NPZ):
        raise FileNotFoundError(
            f"нет {F0_DRONE_NPZ} — сначала: python evalx/f0_survey.py 3000")
    d = np.load(F0_DRONE_NPZ)
    missing = {"idx", "f0", "sal"} - set(d.files)
    if missing:
        raise KeyError(f"{F0_DRONE_NPZ}: нет ключей {missing}, есть {list(d.files)}")
    keep = d["sal"] >= min_salience
    return d["idx"][keep].astype(np.int64), d["f0"][keep].astype(np.float64)


def selfcheck():
    f0 = np.array([50.0, 90.0, 150.0, 250.0, 500.0, 30.0, 0.0])
    assert list(band_of(f0)) == [0, 1, 2, 3, 4, -1, -1], list(band_of(f0))
    # границы принадлежат верхней полосе
    assert band_of(np.array([80.0]))[0] == 1
    assert band_of(np.array([300.0]))[0] == 4

    # recall по полосам: в полосе 0 угадано 1 из 2, в полосе 2 — 2 из 2
    lg = np.array([1.0, -1.0, 5.0, 5.0], np.float32)
    y = np.array([1, 1, 1, 1])
    f = np.array([50.0, 50.0, 150.0, 150.0])
    rec = recall_by_band(lg, y, f, thr=0.0)
    assert abs(rec["40-80"] - 0.5) < 1e-9, rec
    assert abs(rec["120-200"] - 1.0) < 1e-9, rec
    assert "80-120" not in rec, "пустые полосы не должны попадать в отчёт"

    name, val = worst_band(rec)
    assert name == "40-80" and abs(val - 0.5) < 1e-9, (name, val)

    print("strata selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
