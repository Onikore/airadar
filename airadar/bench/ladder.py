"""Лестница деградации: главный скаляр проекта.

Вместо "recall при фиксированных условиях" измеряется условие, при котором
детектор ломается. Метрика не насыщается конструктивно: стало лучше — кривая
сдвинулась влево, и сдвиг виден. Единица физическая: 6 дБ примерно вдвое по
дальности в свободном поле.

Фон для подмешивания обязан быть удержанным и не тем, что в noise_pool
обучения, иначе лестница измерит запоминание фона, а не обнаружение цели.
"""

import sys
import numpy as np

SNR_GRID = np.arange(20.0, -15.1, -2.5)


def selfcheck():
    rng = np.random.default_rng(0)

    # подмешивание: фактический SNR совпадает с заказанным
    t = rng.normal(0, 1, 16000).astype(np.float32)
    n = rng.normal(0, 1, 16000).astype(np.float32)
    for want in (10.0, 0.0, -10.0):
        m = mix_at_snr(t, n, want, rng)
        got = 10 * np.log10(np.mean(t ** 2) / np.mean((m - t) ** 2))
        assert abs(got - want) < 0.2, (want, got)

    # сетка идёт сверху вниз и содержит 15 ступеней
    assert len(SNR_GRID) == 15 and SNR_GRID[0] == 20.0 and SNR_GRID[-1] == -15.0

    # snr50 на идеальной ступеньке: переход между 5.0 и 2.5 -> ровно посередине
    p = np.where(SNR_GRID >= 5.0, 1.0, 0.0)
    assert abs(snr50(SNR_GRID, p) - 3.75) < 1e-6, snr50(SNR_GRID, p)

    # линейный спад: 0.5 достигается там, где кривая её пересекает
    p2 = np.clip(0.5 + (SNR_GRID - 0.0) / 20.0, 0.0, 1.0)
    assert abs(snr50(SNR_GRID, p2) - 0.0) < 0.5, snr50(SNR_GRID, p2)

    # кривая, никогда не падающая до 0.5 — метрика не определена, не выдумываем
    assert np.isnan(snr50(SNR_GRID, np.ones_like(SNR_GRID)))
    assert np.isnan(snr50(SNR_GRID, np.zeros_like(SNR_GRID)))

    # Синтетический скорер: доля энергии в полосе цели. Именно доля, а не
    # абсолютный уровень — при падении SNR подмешивается БОЛЬШЕ шума, полная
    # энергия растёт, и скорер по уровню дал бы кривую, растущую вниз по
    # лестнице. Цель — тон 200 Гц, фон — белый шум.
    class BandScorer:
        hop_s, context_s = 0.25, 0.5

        def score(self, audio):
            from airadar.bench.scorer import n_scores
            n = n_scores(len(audio), self.context_s, self.hop_s)
            ctx, hop = 8000, 4000
            f = np.fft.rfftfreq(ctx, 1.0 / 16000)
            band = (f > 190) & (f < 210)
            wnd = np.hanning(ctx)
            out = np.empty(n, np.float32)
            for i in range(n):
                sp = np.abs(np.fft.rfft(audio[i * hop:i * hop + ctx] * wnd)) ** 2
                out[i] = 10 * np.log10((sp[band].sum() + 1e-12)
                                       / (sp.sum() + 1e-12))
            return out

    rng2 = np.random.default_rng(7)
    tt = np.arange(16000 * 4) / 16000.0
    tgt = np.sin(2 * np.pi * 200 * tt).astype(np.float32)
    pool = [rng2.normal(0, 1.0, 16000 * 4).astype(np.float32) for _ in range(4)]
    curve = p_detect_curve(BandScorer(), tgt, pool, on=-10.0, off=-11.0,
                           n_rep=4, seed=3)
    assert curve[0] >= curve[-1], curve          # вниз по SNR не растёт
    assert curve[0] == 1.0 and curve[-1] == 0.0, curve
    assert 0.0 <= curve.min() and curve.max() <= 1.0

    print("ladder selfcheck ok")


def mix_at_snr(target, noise, snr_db, rng):
    """Подмешать фон к цели так, чтобы получился заданный SNR.

    Фон при необходимости зацикливается и берётся со случайного сдвига:
    иначе на всех ступенях лестницы окажется один и тот же кусок фона, и
    разброс между ступенями будет измерять его, а не модель.
    """
    t = np.asarray(target, np.float32)
    n = np.asarray(noise, np.float32)
    if len(n) < len(t):
        n = np.tile(n, int(np.ceil(len(t) / len(n))))
    off = int(rng.integers(0, len(n)))
    n = np.roll(n, off)[:len(t)]
    tp = float(np.mean(t ** 2)) + 1e-12
    np_ = float(np.mean(n ** 2)) + 1e-12
    scale = np.sqrt(tp / (np_ * 10.0 ** (snr_db / 10.0)))
    return t + n * scale


def snr50(snrs, pdet):
    """SNR, при котором вероятность обнаружения падает до 0.5.

    Кривая по построению убывает с падением SNR. Ищем первое пересечение
    уровня 0.5 сверху вниз и интерполируем линейно. Если пересечения нет —
    возвращаем nan: метрика не определена, и выдумывать её нельзя, иначе
    получится ещё одно число, которое сравнивает шум.
    """
    s = np.asarray(snrs, np.float64)
    p = np.asarray(pdet, np.float64)
    for i in range(len(s) - 1):
        if p[i] >= 0.5 > p[i + 1]:
            w = (p[i] - 0.5) / (p[i] - p[i + 1])
            return float(s[i] + w * (s[i + 1] - s[i]))
    return float("nan")


def p_detect_curve(scorer, target, noise_pool, on, off, snrs=SNR_GRID,
                   n_rep=8, seed=0, tau_s=2.0):
    """Доля повторов, в которых цель обнаружена, на каждой ступени SNR.

    n_rep повторов с разным куском фона: одна и та же цель на разных фонах
    при одинаковом SNR даёт заметно разные оценки, и без усреднения кривая
    измеряет выбор фона.
    """
    from airadar.bench.decision import smooth, detected
    rng = np.random.default_rng(seed)
    out = np.empty(len(snrs), np.float64)
    for j, snr in enumerate(snrs):
        hits = 0
        for _ in range(n_rep):
            noise = noise_pool[int(rng.integers(0, len(noise_pool)))]
            mixed = mix_at_snr(target, noise, float(snr), rng)
            lg = smooth(scorer.score(mixed), scorer.hop_s, tau_s)
            hits += int(detected(lg, scorer.hop_s, on, off))
        out[j] = hits / n_rep
    return out


def snr50_ci(snrs, pdet_boot):
    """CI по бутстрап-выборке кривых [n_boot, len(snrs)]."""
    from airadar.bench.ci import ci
    vals = np.array([snr50(snrs, p) for p in pdet_boot], np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return float("nan"), float("nan")
    return ci(vals)


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
