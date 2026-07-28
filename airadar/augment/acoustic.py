"""Затухание верхов (air absorption) — без изменений по сути (§4.2),
перенесено на новый CQT-признак. Циклический сдвиг и SpecAugment — тоже
без изменений, перенесены на новую форму [2, F, T] (было [1, M, T]).

Затухание в старом LogMel применялось на СЫРОЙ мощности спектра до
логарифма: power *= exp(-k*f/1000)^2. Новый фронтенд (этап 2) отдаёт уже
log(power) — то же действие в лог-домене: log(power*att^2) = log(power) +
2*log(att) = log(power) - 2*k*f/1000. Применяется ТОЛЬКО к ch0 (см.
докстринг задачи в плане): наклон, постоянный по времени внутри примера,
и так вычитается при вычислении ch1.
"""
import sys
import numpy as np
import torch


def cyclic_shift(wav, rng):
    """wav: [N] float32 -> циклически сдвинутый на случайный офсет.
    Модель не должна цепляться за абсолютную позицию события в окне."""
    wav = np.asarray(wav, dtype=np.float32)
    shift = int(rng.integers(0, len(wav)))
    return np.roll(wav, shift)


def apply_air_absorption(ch0, freqs, k):
    """ch0: [B, F, T] лог-мощность (обычно канал 0 выхода Frontend).
    freqs: [F] Гц (LogCQT.frequencies). k: [B] коэффициент затухания
    (0 = нет затухания, air_k_max = сильное) -> [B, F, T]."""
    freqs_t = torch.as_tensor(freqs, dtype=ch0.dtype, device=ch0.device)   # [F]
    k_t = torch.as_tensor(k, dtype=ch0.dtype, device=ch0.device)
    tilt = (2.0 * k_t / 1000.0)[:, None, None]                            # [B,1,1]
    return ch0 - tilt * freqs_t[None, :, None]


def spec_augment(feat, rng, n_masks=2, max_frac=1.0 / 6.0):
    """feat: [2, F, T] (ch0, ch1) -> с замаскированными полосами частот и
    времени. Маска ОДИНАКОВА для обоих каналов: физически "эта часть
    записи пропала", не "пропала только в одном представлении"."""
    feat = feat.clone()
    _, F, T = feat.shape
    for _ in range(n_masks):
        f = int(rng.integers(0, max(1, int(F * max_frac)) + 1))
        f0 = int(rng.integers(0, max(1, F - f + 1)))
        feat[:, f0:f0 + f, :] = 0.0
        t = int(rng.integers(0, max(1, int(T * max_frac)) + 1))
        t0 = int(rng.integers(0, max(1, T - t + 1)))
        feat[:, :, t0:t0 + t] = 0.0
    return feat


def selfcheck():
    rng = np.random.default_rng(0)

    wav = np.arange(100, dtype=np.float32)
    shifted = cyclic_shift(wav, rng)
    assert shifted.shape == wav.shape
    assert set(shifted.tolist()) == set(wav.tolist())   # те же значения, другой порядок
    assert not np.array_equal(shifted, wav) or len(set(rng.integers(0, 100, 5))) == 1

    B, F, T = 3, 183, 32
    freqs = np.linspace(40.0, 8000.0, F).astype(np.float32)
    ch0 = torch.zeros(B, F, T)
    k = torch.tensor([0.0, 1.0, 2.5])
    out = apply_air_absorption(ch0, freqs, k)
    assert out.shape == (B, F, T)
    assert torch.allclose(out[0], ch0[0])           # k=0 -> без изменений
    # эффект растёт с частотой: высокий бин ослаблен больше низкого при k>0
    assert (out[1, -1, 0] < out[1, 0, 0]).item()
    assert (out[1, -1, 0] > out[2, -1, 0]).item()    # k=2.5 ослабляет сильнее k=1.0

    feat = torch.ones(2, 183, 32)
    masked = spec_augment(feat, rng)
    assert masked.shape == feat.shape
    zero_ch0 = (masked[0] == 0)
    zero_ch1 = (masked[1] == 0)
    assert torch.equal(zero_ch0, zero_ch1)          # маска одинакова на обоих каналах
    assert zero_ch0.any()                            # хоть что-то замаскировано
    assert not zero_ch0.all()                         # не всё замаскировано

    print("acoustic selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
