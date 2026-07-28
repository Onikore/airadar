"""Смешивание позитива и фона при заданном SNR (§4.1/§4.2). Пиковая
нормализация убрана: раньше `wav /= abs(wav).max()` привязывала масштаб
ВСЕГО примера к самой громкой отдельной помехе внутри окна (щелчок,
всплеск) — сеть заново училась абсолютной громкости на каждом примере
вместо относительной. Теперь уровень несёт ch1 (вычитание фона, этап 2)
и random_gain (аугментация масштаба, не нормализация — значения могут
выйти за [-1,1], это осознанно, см. план)."""
import sys
import numpy as np


def snr_scale(signal, background, snr_db):
    sig_p = np.mean(np.asarray(signal, dtype=np.float64) ** 2) + 1e-12
    bg_p = np.mean(np.asarray(background, dtype=np.float64) ** 2) + 1e-12
    return float(np.sqrt(sig_p / (bg_p * 10 ** (snr_db / 10.0))))


def mix_at_snr(signal, background, snr_db):
    """signal, background: [N] float32, одинаковой длины -> signal +
    масштабированный background, дающий заданный SNR относительно
    ВСЕГО signal (для короткого позитива на офсете внутри канвы см.
    snr_scale + airadar/train/sampler.py — там сигнал и локальный фон
    короче полной канвы, а масштаб потом применяется к канве целиком)."""
    scale = snr_scale(signal, background, snr_db)
    return (np.asarray(signal, dtype=np.float32)
            + np.asarray(background, dtype=np.float32) * scale)


def random_gain(wav, rng, lo=-6.0, hi=6.0):
    gain_db = rng.uniform(lo, hi)
    return (np.asarray(wav, dtype=np.float32) * 10 ** (gain_db / 20.0)).astype(np.float32)


def place_at_offset(short, canvas_len, rng):
    """short: [n] float32, n <= canvas_len -> (canvas [canvas_len] float32
    с short на случайном офсете поверх нулей, offset int)."""
    short = np.asarray(short, dtype=np.float32)
    n = len(short)
    assert n <= canvas_len, (n, canvas_len)
    offset = int(rng.integers(0, canvas_len - n + 1))
    canvas = np.zeros(canvas_len, dtype=np.float32)
    canvas[offset:offset + n] = short
    return canvas, offset


def selfcheck():
    rng = np.random.default_rng(0)

    sig = np.ones(1000, dtype=np.float32)
    bg = np.ones(1000, dtype=np.float32) * 2.0
    scale = snr_scale(sig, bg, 0.0)   # 0 дБ -> равные мощности после масштаба
    scaled_bg_power = np.mean((bg * scale) ** 2)
    assert abs(scaled_bg_power - np.mean(sig ** 2)) < 1e-6, scaled_bg_power

    mixed = mix_at_snr(sig, bg, 0.0)
    assert mixed.shape == sig.shape
    assert np.allclose(mixed, sig + bg * scale)

    # выше SNR -> меньше вклад фона
    scale_hi = snr_scale(sig, bg, 20.0)
    scale_lo = snr_scale(sig, bg, -15.0)
    assert scale_hi < scale_lo

    g = random_gain(sig, rng, lo=-6.0, hi=6.0)
    ratio_db = 20 * np.log10(np.abs(g[0]) / np.abs(sig[0]))
    assert -6.0 - 1e-3 <= ratio_db <= 6.0 + 1e-3, ratio_db

    short = np.arange(1, 11, dtype=np.float32)
    canvas, offset = place_at_offset(short, 100, rng)
    assert canvas.shape == (100,)
    assert np.array_equal(canvas[offset:offset + 10], short)
    assert canvas[:offset].sum() == 0 and canvas[offset + 10:].sum() == 0

    print("mixing selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
