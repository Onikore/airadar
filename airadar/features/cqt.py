"""Constant-Q через nnAudio: ch0 признака.

Три ручных STFT (16384/4096/1024 отсчётов) плюс разрежённая band-матрица
из первой версии спецификации решали задачу, которую constant-Q закрывает
нативно: Q=f/Δf постоянно ⇒ эффективное окно ∝ 1/f само по себе, без сшивки
тиров. Параметры ниже — не выбор по вкусу, а измеренный факт (спецификация
§2, 2026-07-27): при bins_per_octave=24 шаг на 40 Гц равен 1.172 Гц — ровно
требование `f·(2^(1/24)−1)` на нижнем крае диапазона.
"""

import sys
import numpy as np
import torch

SR = 16000
HOP_S = 0.128
FMIN, FMAX = 40.0, 8000.0
BINS_PER_OCTAVE = 24
N_BINS = 183                     # round(24 * log2(8000/40)) — см. §1.1 спецификации
HOP_LENGTH = round(HOP_S * SR)   # 2048, ровно 128 мс при 16 кГц


class LogCQT(torch.nn.Module):
    """ch0 признака: лог-мощность на сетке constant-Q, 183 бина.

    trainable=False — фронтенд не участвует в обратном проходе, как и
    нынешний ручной LogMel (его mel-банк тоже не обучается).

    cfg=None -> airadar.config.FeatureCfg() по умолчанию, что воспроизводит
    прежнее поведение (глобалы этого модуля остаются для обратной
    совместимости — их использует, например, airadar/bench/feat_visibility.py
    и airadar/features/harmonic.py по прямому импорту, менять их нельзя).
    """

    def __init__(self, cfg=None):
        super().__init__()
        from airadar.config import FeatureCfg
        self.cfg = cfg or FeatureCfg()
        from nnAudio.features import CQT2010v2
        self._cqt = CQT2010v2(
            sr=self.cfg.sr, hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin, fmax=self.cfg.fmax,
            n_bins=self.cfg.n_bins, bins_per_octave=self.cfg.bins_per_octave,
            output_format="Magnitude", trainable=False, verbose=False,
        )

    @property
    def frequencies(self):
        return np.asarray(self._cqt.frequencies)

    def forward(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mag = self._cqt(wav)                    # [B, N_BINS, T], линейная магнитуда
        power = mag.pow(2)
        return torch.log(power + 1e-8)


def selfcheck():
    cqt = LogCQT()

    # 4.0 с — окно модели. Число кадров ИЗМЕРЕНО для nnAudio==0.3.4, не
    # вычислено по 4.0/0.128=31.25 (паддинг библиотеки нетривиален).
    wav4 = torch.zeros(1, round(4.0 * SR))
    out4 = cqt(wav4)
    assert out4.shape == (1, N_BINS, 32), out4.shape

    # без батч-измерения тоже должно работать (приводится к [1, ...])
    out4_flat = cqt(torch.zeros(round(4.0 * SR)))
    assert out4_flat.shape == (1, N_BINS, 32), out4_flat.shape

    # 12.0 с — полный вход обучающего примера (8с истории + 4с модели)
    wav12 = torch.zeros(1, round(12.0 * SR))
    out12 = cqt(wav12)
    assert out12.shape == (1, N_BINS, 94), out12.shape   # измерено, не 32+63

    # выход — лог-мощность: конечен, не NaN, на тишине не -inf благодаря eps
    assert torch.isfinite(out4).all()

    # частоты бинов: 183 значения, первый — ровно FMIN, монотонно растут
    freqs = cqt.frequencies
    assert len(freqs) == N_BINS
    assert abs(freqs[0] - FMIN) < 0.01, freqs[0]
    assert np.all(np.diff(freqs) > 0), "частоты бинов должны монотонно расти"
    # разрешение на нижнем крае — измеренный факт спецификации §2
    assert abs((freqs[1] - freqs[0]) - 1.172) < 0.01, freqs[1] - freqs[0]

    print("cqt selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
