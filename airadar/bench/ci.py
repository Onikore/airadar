"""Блочный бутстрап: доверительные интервалы для рядов с автокорреляцией.

Окна перекрываются вдвое и идут подряд по одной записи, поэтому соседние
оценки почти одинаковы. Обычный биномиальный интервал в такой ситуации врёт
вдвое в сторону оптимизма — именно поэтому пять экспериментов проекта
сравнили шум и приняли его за сигнал. Блок в L окон означает допущение, что
независимы куски записи длиной L*hop секунд.
"""

import sys
import numpy as np


def selfcheck():
    rng = np.random.default_rng(0)
    x = rng.random(2000) < 0.5                      # iid, истинное среднее 0.5

    lo1, hi1 = ci(block_bootstrap(x, np.mean, block=1, seed=1))
    assert lo1 < 0.5 < hi1, (lo1, hi1)
    assert 0.03 < hi1 - lo1 < 0.06, hi1 - lo1       # ~ +-2.2% для n=2000

    # на iid-данных крупный блок не должен систематически сужать интервал,
    # а на коррелированных обязан его расширить
    y = np.repeat(rng.random(50) < 0.5, 40)         # блоки по 40 одинаковых
    w1 = np.subtract(*ci(block_bootstrap(y, np.mean, block=1, seed=2))[::-1])
    w40 = np.subtract(*ci(block_bootstrap(y, np.mean, block=40, seed=2))[::-1])
    assert w40 > 2 * w1, (w1, w40)

    # ресэмплировка не меняет длину выборки
    assert len(block_bootstrap(x, np.mean, n_boot=17, block=12)) == 17

    # парная разность: одинаковые ряды -> CI накрывает ноль и он узкий
    lo, hi = paired_diff_ci(x, x.copy(), np.mean, block=12)
    assert lo == hi == 0.0, (lo, hi)

    print("ci selfcheck ok")


def _resample_idx(n, block, n_boot, rng):
    """Индексы moving block bootstrap: nb блоков подряд, обрезка до n."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=(n_boot, nb))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    return idx.reshape(n_boot, -1)[:, :n] % n


def block_bootstrap(x, stat, n_boot=4000, block=12, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = _resample_idx(len(x), block, n_boot, rng)
    return np.array([stat(x[i]) for i in idx], np.float64)


def ci(samples, level=0.95):
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [a, 1.0 - a])
    return float(lo), float(hi)


def paired_diff_ci(xa, xb, stat, block=12, n_boot=4000, seed=0):
    """CI разности stat(a) - stat(b) при ОБЩИХ индексах ресэмплировки.

    Общие индексы обязательны: ряды посчитаны на одних и тех же окнах одной
    записи, и большая часть разброса у них общая. Независимая ресэмплировка
    раздула бы интервал и спрятала реальную разницу между моделями.
    """
    xa, xb = np.asarray(xa), np.asarray(xb)
    assert len(xa) == len(xb), "ряды должны быть по одним и тем же окнам"
    rng = np.random.default_rng(seed)
    idx = _resample_idx(len(xa), block, n_boot, rng)
    d = np.array([stat(xa[i]) - stat(xb[i]) for i in idx], np.float64)
    return ci(d)


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
