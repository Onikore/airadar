"""f0-сдвиг позитивов (§4.1): ресемплинг с коэффициентом r ∈ [0.35, 1.5]
размазывает f0 обучающей массы по диапазону 40-400 Гц. На лог-оси это
ровно трансляция (тот же принцип, что делает harmonic stacking
f0-независимым, см. airadar/features/harmonic.py).

Порядок обязателен (§4.1): сдвигается ЧИСТЫЙ позитив, ДО подмешивания
фона (airadar/augment/mixing.py, airadar/train/sampler.py) — иначе вместе
с целью сдвинется и фон, которому сдвигаться не с чего.
"""
import sys
from math import gcd
import numpy as np
from scipy.signal import resample_poly


def sample_r(rng, cfg=None):
    from airadar.config import AugCfg
    cfg = cfg or AugCfg()
    return float(rng.uniform(cfg.pitch_r_lo, cfg.pitch_r_hi))


def pitch_shift(wav, r):
    """wav: [N] float32 -> [round(N/r)] float32. r<1 понижает f0 и
    удлиняет клип (r=0.35: 200Гц->70Гц, 0.6с->1.71с); r>1 — наоборот."""
    up = 10000
    down = round(10000 * r)
    g = gcd(up, down)
    up, down = up // g, down // g
    return resample_poly(np.asarray(wav, dtype=np.float32), up, down).astype(np.float32)


def selfcheck():
    sr = 16000
    t = np.arange(round(0.6 * sr), dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)   # f0=200Гц, квадрокоптер

    shifted = pitch_shift(tone, 0.35)
    # длительность: 0.6с/0.35 = 1.714с (§4.1, пример из спецификации)
    assert abs(len(shifted) / sr - 1.714) < 0.02, len(shifted) / sr

    # f0 реально сдвинулась на r: ищем пик спектра, ожидаем ~70 Гц
    spec = np.abs(np.fft.rfft(shifted * np.hanning(len(shifted))))
    freqs = np.fft.rfftfreq(len(shifted), 1 / sr)
    f0_hat = freqs[spec.argmax()]
    assert abs(f0_hat - 70.0) < 3.0, f0_hat

    # r=1.0 -> длина не меняется (с точностью до округления рационального
    # приближения gcd(10000,10000)=10000 -> up=down=1)
    unchanged = pitch_shift(tone, 1.0)
    assert len(unchanged) == len(tone), (len(unchanged), len(tone))

    # sample_r: диапазон соблюдён, детерминирован при фиксированном seed
    rng = np.random.default_rng(0)
    rs = [sample_r(rng) for _ in range(200)]
    assert all(0.35 <= r <= 1.5 for r in rs)
    assert min(rs) < 0.5 and max(rs) > 1.3   # диапазон реально используется целиком

    print("pitch selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
