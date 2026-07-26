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

    # регрессия: длинная непрерывная тревога (дольше dead_s) с провалом в один
    # отсчёт внутри неё — по-прежнему ОДНО событие. Мёртвый счётчик обязан
    # отслеживать последний АКТИВНЫЙ отсчёт целиком, а не только старт события:
    # иначе разрыв в 0.25 с внутри 30-секундной тревоги считается новой тревогой.
    dead = int(round(30.0 / hop))                             # 120 отсчётов
    run1 = dead + 20                                          # непрерывная тревога дольше dead_s
    long_alarm = np.ones(run1 + 10, bool)
    long_alarm[run1] = False                                  # провал на 1 отсчёт сразу после run1
    assert list(events(long_alarm, hop_s=hop, dead_s=30.0)) == [0]

    # для контроля: настоящий разрыв длиннее dead_s должен давать ДВА события,
    # чтобы фикс не начал всё подряд склеивать в одно
    gap_alarm = np.concatenate([np.ones(11, bool), np.zeros(dead + 10, bool), np.ones(5, bool)])
    assert list(events(gap_alarm, hop_s=hop, dead_s=30.0)) == [0, 11 + dead + 10]

    # FA/час: 2 события на 1 часе фона
    n = int(3600 / hop)
    bg = np.zeros(n, np.float32); bg[100] = 10.0; bg[n // 2] = 10.0
    assert abs(fa_per_hour(bg, hop, on=1.0, off=0.5, dead_s=0.0) - 2.0) < 1e-6

    # маска в fa_per_hour: событие внутри вырезанной области не считается,
    # и сама вырезанная область выкидывается из часов в знаменателе тоже
    mask_bg = np.ones(n, bool)
    hole = slice(n // 2 - 5, n // 2 + 5)
    mask_bg[hole] = False                       # вырезаем область со вторым выбросом целиком
    hours_masked = (n - (hole.stop - hole.start)) * hop / 3600.0
    fa_masked = fa_per_hour(bg, hop, on=1.0, off=0.5, dead_s=0.0, mask=mask_bg)
    assert abs(fa_masked - 1.0 / hours_masked) < 1e-6            # остался только первый выброс

    # подбор порога под бюджет: результат обязан дать FA не выше бюджета
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, n).astype(np.float32)
    thr = threshold_for_fa(noise, hop, target_fa=1.0, dead_s=30.0)
    assert fa_per_hour(noise, hop, thr, thr - 1.0, dead_s=30.0) <= 1.0 + 1e-9

    # маска в threshold_for_fa: подбор обязан вестись только по квантилям
    # немаскированных окон. Портим вторую половину ряда огромными выбросами
    # и маскируем именно её: если бы сетка квантилей считалась по полному
    # ряду (а маска применялась бы только внутри fa_per_hour), выбросы
    # сдвинули бы саму сетку и результат разошёлся бы с расчётом на заведомо
    # урезанном ряде
    spiky = noise.copy()
    spiky[np.arange(7000, 14000, 200)] = 50.0
    quiet_mask = np.ones(n, bool); quiet_mask[7000:] = False
    thr_masked = threshold_for_fa(spiky, hop, target_fa=1.0, dead_s=30.0, mask=quiet_mask)
    assert thr_masked == threshold_for_fa(spiky[quiet_mask], hop, target_fa=1.0, dead_s=30.0)

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
    """Триггер Шмитта: включение по on, выключение по off <= on."""
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

    Отсчёт мёртвого времени ведётся от последнего АКТИВНОГО отсчёта тревоги,
    а не от старта последнего принятого события: тревога, которая сама по
    себе длится дольше dead_s, не должна давать второе событие из-за провала
    в один отсчёт где-то в середине.
    """
    alarm = np.asarray(alarm, bool)
    dead = int(round(dead_s / hop_s))
    out, last_active = [], -10**9
    prev = False
    for i, v in enumerate(alarm):
        if v:
            if not prev and i - last_active > dead:
                out.append(i)
            last_active = i
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
    """Наименьший порог из верхней половины распределения оценок, при котором
    FA/час не превышает бюджет.

    Перебор по сетке значений самого ряда, а не бисекция: функция FA(порог)
    ступенчата, и бисекция на ступеньках останавливается где попало.

    Сетка сознательно берётся только по квантилям [0.5, 1.0], а не от
    минимума: порог ниже медианы держит hysteresis во включённом состоянии
    почти всё время, events() склеивает это в одно событие, и функция вернёт
    вырожденный «всегда тревога» порог как якобы наименьший подходящий. Для
    реалистичного бюджета (единицы тревог в час на часовом фоне) настоящий
    ответ и так лежит намного выше медианы, так что это ограничение ничего
    не стоит на практике — но гарантия действует только выше медианы: если
    для конкретных данных валидный порог лежит ниже неё, эта функция его не
    найдёт.
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
