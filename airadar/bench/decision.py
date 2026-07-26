"""Решающий слой: от ряда логитов к событиям и к FA/час.

Правило "k из m" удалено. Измерено, что при равном пооконном FAR оно recall
снижает (24.1% -> 12.4%): бинаризация до накопления выбрасывает величину
оценки. Здесь накапливается сам логит, а гистерезис убирает дребезг тревоги.

Порог задаётся там, где принимается решение — по бюджету ложных тревог в
час на фоне, а не по доле окон. Порог, откалиброванный по доле окон
лабораторного корпуса, в лесу соврёт в разы.
"""

import sys
import numpy as np


def selfcheck():
    hop = 0.25

    # сглаживание: константа остаётся собой, одиночный выброс подавляется
    c = np.full(20, 3.0, np.float32)
    assert np.allclose(smooth(c, hop, tau_s=2.0), 3.0)
    spike = np.zeros(20, np.float32); spike[10] = 10.0
    assert smooth(spike, hop, tau_s=2.0).max() < 2.0

    # гистерезис: включается по on, держится до off, не дребезжит
    x = np.array([0, 2, 0.6, 0.6, 0, 0], np.float32)
    assert list(hysteresis(x, on=1.0, off=0.5)) == [False, True, True, True, False, False]

    # события: слипшиеся тревоги — одно событие, мёртвое время склеивает
    al = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], bool)
    assert list(events(al, hop_s=hop, dead_s=0.0)) == [1, 5]
    assert list(events(al, hop_s=hop, dead_s=30.0)) == [1]

    # FA/час: 2 события на 1 часе фона
    n = int(3600 / hop)
    bg = np.zeros(n, np.float32); bg[100] = 10.0; bg[n // 2] = 10.0
    assert abs(fa_per_hour(bg, hop, on=1.0, off=0.5, dead_s=0.0) - 2.0) < 1e-6

    # подбор порога под бюджет: результат обязан дать FA не выше бюджета
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, n).astype(np.float32)
    thr = threshold_for_fa(noise, hop, target_fa=1.0, dead_s=30.0)
    assert fa_per_hour(noise, hop, thr, thr - 1.0, dead_s=30.0) <= 1.0 + 1e-9

    # время до тревоги
    y = np.array([0, 0, 5, 5], np.float32)
    assert abs(time_to_alarm(y, hop, on=1.0, off=0.5) - 0.5) < 1e-9
    assert time_to_alarm(np.zeros(4, np.float32), hop, 1.0, 0.5) == np.inf
    assert detected(y, hop, 1.0, 0.5) and not detected(np.zeros(4, np.float32), hop, 1.0, 0.5)

    print("decision selfcheck ok")


def smooth(logits, hop_s, tau_s=2.0):
    """Экспоненциальное сглаживание логита.

    Сглаживается логит, а не бинарное решение: величина оценки несёт
    информацию, и её сохранение — самая дешёвая прибавка к дальности из
    доступных.
    """
    x = np.asarray(logits, np.float32)
    if len(x) == 0:
        return x
    a = float(np.exp(-hop_s / tau_s))
    out = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = a * acc + (1.0 - a) * x[i]
        out[i] = acc
    return out


def hysteresis(x, on, off):
    """Триггер Шмитта: включение по on, выключение по off < on."""
    assert off <= on, "порог выключения должен быть не выше порога включения"
    x = np.asarray(x)
    out = np.zeros(len(x), bool)
    state = False
    for i, v in enumerate(x):
        state = v >= on if not state else v >= off
        out[i] = state
    return out


def events(alarm, hop_s, dead_s=30.0):
    """Начала событий. Слипшееся — одно; после события мёртвое время.

    Мёртвое время моделирует интерфейс оператора: тревога, повторившаяся
    через две секунды, для него не вторая тревога, а та же самая.
    """
    alarm = np.asarray(alarm, bool)
    dead = int(round(dead_s / hop_s))
    out, last = [], -10**9
    prev = False
    for i, v in enumerate(alarm):
        if v and not prev and i - last > dead:
            out.append(i)
            last = i
        prev = v
    return np.array(out, np.int64)


def fa_per_hour(logits, hop_s, on, off, dead_s=30.0, mask=None):
    """Ложных тревог в час. mask=False выбрасывает окно из знаменателя тоже."""
    x = np.asarray(logits, np.float32)
    if mask is not None:
        x = x[np.asarray(mask, bool)]
    if len(x) == 0:
        return 0.0
    hours = len(x) * hop_s / 3600.0
    return len(events(hysteresis(x, on, off), hop_s, dead_s)) / hours


def threshold_for_fa(logits_bg, hop_s, target_fa, off_delta=1.0,
                     dead_s=30.0, mask=None, n_grid=512):
    """Наименьший порог, при котором FA/час не превышает бюджет.

    Перебор по сетке значений самого ряда, а не бисекция: функция FA(порог)
    ступенчата, и бисекция на ступеньках останавливается где попало.
    """
    x = np.asarray(logits_bg, np.float32)
    if mask is not None:
        x = x[np.asarray(mask, bool)]
    if len(x) == 0:
        return float("inf")
    grid = np.unique(np.quantile(x, np.linspace(0.5, 1.0, n_grid)))
    for thr in grid:                      # от низкого к высокому
        if fa_per_hour(x, hop_s, thr, thr - off_delta, dead_s) <= target_fa:
            return float(thr)
    return float(grid[-1] + off_delta)


def detected(logits, hop_s, on, off):
    return bool(hysteresis(np.asarray(logits, np.float32), on, off).any())


def time_to_alarm(logits, hop_s, on, off):
    al = hysteresis(np.asarray(logits, np.float32), on, off)
    idx = np.flatnonzero(al)
    return float(idx[0] * hop_s) if len(idx) else float("inf")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
