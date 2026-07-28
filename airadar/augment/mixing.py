"""Смешивание позитива и фона при заданном SNR (§4.1/§4.2).

`normalize_rms` добавлена по факту реального бага, найденного систематической
отладкой на этапе 3г: без НИКАКОЙ нормализации `mix_at_snr` складывает сигнал
и фон (позитив = сигнал+фон, негатив = только фон целиком) — сумма
систематически громче одного слагаемого, и позитивы оказались медианно
вдвое громче негативов в тренировочных данных (измерено: peak/rms
позитивов и негативов на 150+150 реальных примерах). Модель, обученная
15 эпох на всём датасете, научилась использовать ГРОМКОСТЬ как признак
класса: одна и та же полевая запись меняла вердикт с "не дрон" (-5.8) на
"дрон" (+2.5) при простом усилении в 10-20 раз, без изменения содержимого.
Пиковая нормализация была убрана намеренно (щелчок внутри окна не должен
диктовать масштаб всего примера) — RMS от этого не страдает: громкий
короткий импульс почти не двигает RMS многосекундного окна, в отличие от
пика. `normalize_rms` — финальный шаг сборки примера (после сложения
сигнала и фона, после гула), она убирает системную разницу громкости
между классами, `random_gain` после неё — управляемая, симметричная для
обоих классов случайность масштаба, а не источник побочной корреляции."""
import sys
import numpy as np


def normalize_rms(wav, target_rms=0.05, eps=1e-8):
    """wav: [...,N] float32 -> смещено по громкости так, чтобы RMS по
    последней оси стал target_rms. Работает и на одном примере [N], и на
    батче [B,N] (используется DroneNet2Scorer на батче окон, sampler.py —
    на одном примере). Тишина (rms < eps) не усиливается — умножение на
    огромный масштаб дало бы усиленный шум квантования, а не сигнал."""
    wav = np.asarray(wav, dtype=np.float32)
    rms = np.sqrt(np.mean(wav.astype(np.float64) ** 2, axis=-1, keepdims=True))
    scale = np.where(rms < eps, 1.0, target_rms / np.maximum(rms, eps))
    return (wav * scale).astype(np.float32)


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

    # normalize_rms: сам RMS выходит на target, форма/фаза не портятся
    tone = np.sin(2 * np.pi * 100.0 * np.arange(4000, dtype=np.float32) / 16000) * 3.7
    normed = normalize_rms(tone, target_rms=0.05)
    assert abs(np.sqrt(np.mean(normed.astype(np.float64) ** 2)) - 0.05) < 1e-4
    # то же самое, только слабее -- после нормализации совпадает с точностью
    # до знака (оба положительные масштабы, direct proportional)
    quiet = tone * 0.01
    normed_quiet = normalize_rms(quiet, target_rms=0.05)
    assert np.allclose(normed, normed_quiet, atol=1e-4), \
        "два разномасштабных, но одинаковых по форме сигнала обязаны совпасть после нормализации"

    # тишина не взрывается делением на почти-ноль
    silence = np.zeros(4000, dtype=np.float32)
    normed_silence = normalize_rms(silence, target_rms=0.05)
    assert np.allclose(normed_silence, 0.0)
    assert np.isfinite(normed_silence).all()

    # батч [B,N]: нормализация по последней оси независимо на каждую строку
    batch = np.stack([tone, quiet, silence])
    normed_batch = normalize_rms(batch, target_rms=0.05)
    assert normed_batch.shape == batch.shape
    assert np.allclose(normed_batch[0], normed_batch[1], atol=1e-4)
    assert np.allclose(normed_batch[2], 0.0)

    print("mixing selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
