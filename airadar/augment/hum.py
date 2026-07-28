"""Гул ЛЭП: 50 Гц + гармоники (§4.2). Амплитуда до 0.8 (было 0.25 — при
разрешении CQT 1.172 Гц на 40 Гц гул отделим от гребёнки дрона, топить
его сильнее незачем, см. этап 2, feat_visibility). Расстройка 49.8-50.2
Гц — реальная сеть не держит частоту идеально ровно 50.0.

Гармонический состав и случайная фаза перенесены из train.py:Augment без
изменений — §4.2 просит изменить амплитуду и добавить расстройку, не
переизобретать сам гул.
"""
import sys
import numpy as np

_HARMONICS = ((1, 1.0), (2, 0.5), (3, 0.35), (4, 0.2))


def make_hum(n_samples, sr, rng, f0_lo=49.8, f0_hi=50.2):
    f0 = rng.uniform(f0_lo, f0_hi)
    t = np.arange(n_samples, dtype=np.float64) / sr
    hum = np.zeros(n_samples, dtype=np.float64)
    for k, w in _HARMONICS:
        phase = rng.uniform(0, 2 * np.pi)
        hum += w * np.sin(2 * np.pi * f0 * k * t + phase)
    return hum.astype(np.float32)


def add_hum(wav, rng, amp_max=0.8, sr=16000):
    wav = np.asarray(wav, dtype=np.float32)
    hum = make_hum(len(wav), sr, rng)
    amp = rng.uniform(0.0, amp_max)
    scale = amp * (np.abs(wav).max() + 1e-8)
    return wav + hum * scale


def selfcheck():
    sr = 16000
    rng = np.random.default_rng(0)
    hum = make_hum(sr, sr, rng)   # 1с
    assert hum.shape == (sr,)
    assert np.isfinite(hum).all()

    # пик спектра рядом с 50 Гц (внутри расстройки 49.8-50.2)
    spec = np.abs(np.fft.rfft(hum * np.hanning(len(hum))))
    freqs = np.fft.rfftfreq(len(hum), 1 / sr)
    f0_hat = freqs[spec.argmax()]
    assert 49.0 <= f0_hat <= 51.0, f0_hat   # разрешение FFT на 1с ~1Гц, допуск шире расстройки

    # расстройка реально варьируется между вызовами, не зафиксирована на
    # 50.0 — на 1с окне разрешение FFT ровно 1Гц (>= ширины расстройки
    # 0.4Гц), все черновики округлились бы в один и тот же бин. Нужно
    # окно длиннее: 20с -> разрешение 0.05Гц, восьмикратный запас
    dur_long = 20 * sr
    freqs_long = np.fft.rfftfreq(dur_long, 1 / sr)
    f0s = []
    for _ in range(20):
        h = make_hum(dur_long, sr, np.random.default_rng())
        s = np.abs(np.fft.rfft(h * np.hanning(len(h))))
        f0s.append(round(freqs_long[s.argmax()], 2))
    assert len(set(f0s)) > 1, ("расстройка должна варьироваться", f0s)

    wav = np.sin(2 * np.pi * 200.0 * np.arange(sr, dtype=np.float32) / sr)
    out = add_hum(wav, rng, amp_max=0.8, sr=sr)
    assert out.shape == wav.shape
    assert np.isfinite(out).all()
    assert not np.allclose(out, wav)   # гул реально что-то добавил

    print("hum selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
