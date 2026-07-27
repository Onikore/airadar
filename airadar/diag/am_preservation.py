"""Актуален ли предел Найквиста 3.9 Гц (при шаге кадра 128 мс) для реальной
амплитудной модуляции лопастей?

Не пытается ничего исправить — только измеряет и печатает факт, как D0 в
этапе 0. Решение (короче ли делать hop, заводить ли отдельную "быструю"
ветку для АМ) — за планом этапа 3, не за этой диагностикой.
"""

import sys
import numpy as np

HOP_S = 0.128
NYQUIST_HZ = 1.0 / (2.0 * HOP_S)   # 3.90625 Гц — предел по теореме Найквиста


def am_rate_hz(ch0_band, hop_s=HOP_S):
    """Доминирующая частота модуляции огибающей через автокорреляцию.

    ch0_band — одна полоса ch0 во времени, [T]. Ищем пик автокорреляции
    (кроме нулевого лага) и переводим лаг в кадрах в частоту в Гц.
    """
    x = np.asarray(ch0_band, dtype=np.float64)
    x = x - x.mean()
    if np.allclose(x, 0):
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-12
    # первый локальный максимум после лага 0, не считая сам лаг 0
    peak_lag = None
    for lag in range(1, len(ac) - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.1:
            peak_lag = lag
            break
    if peak_lag is None:
        return 0.0
    return 1.0 / (peak_lag * hop_s)


def selfcheck():
    sr_frames = 1.0 / HOP_S   # 7.8125 "кадров в секунду"

    # синусоида-огибающая на известной частоте, ниже предела Найквиста —
    # автокорреляционный метод обязан её найти с разумной точностью
    t = np.arange(200) * HOP_S
    known_hz = 1.0                                   # заведомо ниже 3.9 Гц
    band = 1.0 + 0.5 * np.sin(2 * np.pi * known_hz * t)
    est = am_rate_hz(band, hop_s=HOP_S)
    assert abs(est - known_hz) < 0.3, est

    # чистый шум без периодичности — оценка не обязана совпасть ни с чем
    # конкретным, но обязана вернуть конечное число, не падать
    noise = np.random.default_rng(0).normal(1.0, 0.1, 200)
    est_noise = am_rate_hz(noise, hop_s=HOP_S)
    assert np.isfinite(est_noise)

    print(f"предел Найквиста при hop={HOP_S}с: {NYQUIST_HZ:.3f} Гц")
    print("am_preservation selfcheck ok")


def main():
    from airadar.features.cqt import LogCQT
    from airadar.bench.corpus import field_records
    import torch

    cqt = LogCQT()
    print(f"предел Найквиста при hop={HOP_S}с: {NYQUIST_HZ:.3f} Гц\n")
    for name, wav in field_records().items():
        ch0 = cqt(torch.from_numpy(wav.astype(np.float32))).squeeze(0).numpy()  # [183, T]
        # три полосы в диапазоне лопастного шума малых аппаратов: 500 Гц,
        # 2 кГц, 4 кГц — ищем ближайший бин к каждой
        freqs = cqt.frequencies
        for target in (500.0, 2000.0, 4000.0):
            idx = int(np.argmin(np.abs(freqs - target)))
            rate = am_rate_hz(ch0[idx])
            over = "ВЫШЕ предела" if rate > NYQUIST_HZ else "в пределах"
            print(f"{name}  полоса {freqs[idx]:.0f} Гц: "
                  f"АМ ~{rate:.2f} Гц ({over} Найквиста)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
