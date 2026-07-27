"""Сборка [ch0, ch1] — единственная точка входа фронтенда.

Обрезка до окна модели (последние 32 кадра) НЕ встроена в forward: разным
вызывающим кодам (валидации этого плана, будущему обучению) нужна разная
длина — валидации интересна вся последовательность, обучению — только
последние 4с. Здесь только вычисление признака; политику длины окна
задаёт вызывающий код через last_model_frames().
"""

import sys
import torch

from airadar.features.cqt import LogCQT, N_BINS
from airadar.features.background import rolling_percentile_causal, BG_WINDOW_FRAMES, BG_QUANTILE

MODEL_FRAMES = 32   # 4.0 с — контекст модели, см. airadar.features.cqt


class Frontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._logcqt = LogCQT()

    def forward(self, wav):
        ch0 = self._logcqt(wav)                                       # [B, F, T]
        ch1 = ch0 - rolling_percentile_causal(ch0, BG_WINDOW_FRAMES, BG_QUANTILE)
        return torch.stack([ch0, ch1], dim=1)                         # [B, 2, F, T]

    @staticmethod
    def last_model_frames(x):
        T = x.shape[-1]
        if T < MODEL_FRAMES:
            raise ValueError(
                f"нужно minimum {MODEL_FRAMES} кадров контекста модели, "
                f"получено {T} — короткий клип должен быть подмешан в фон "
                f"вызывающим кодом (MIL), а не обрезан молча")
        return x[..., -MODEL_FRAMES:]


def selfcheck():
    fe = Frontend()

    wav12 = torch.zeros(1, round(12.0 * 16000))
    out = fe(wav12)
    assert out.shape == (1, 2, N_BINS, 94), out.shape   # 12с -> 94 кадра, см. Task 1

    # ch0 канал (индекс 0) обязан совпасть с LogCQT напрямую — сборка не
    # должна ничего менять в ch0, только добавлять ch1
    ch0_direct = LogCQT()(wav12)
    assert torch.allclose(out[:, 0], ch0_direct, atol=1e-5)

    # ch1 = ch0 - перцентиль(ch0): на тишине (всё константа) ch1 обязан
    # быть везде нулём — перцентиль константного ряда равен самой константе
    assert torch.allclose(out[:, 1], torch.zeros_like(out[:, 1]), atol=1e-4)

    # last_model_frames: обрезка до последних 32 кадров, без побочных
    # эффектов на исходном тензоре
    trimmed = Frontend.last_model_frames(out)
    assert trimmed.shape == (1, 2, N_BINS, MODEL_FRAMES)
    assert torch.allclose(trimmed, out[..., -MODEL_FRAMES:])

    # T < MODEL_FRAMES — короткий клип (например, 0.6с DADS даёт 5 кадров,
    # см. Task 1) — обрезка обязана падать, а не молча отдавать что есть:
    # вызывающий код (будущее обучение, MIL-подмешивание) должен явно
    # решить, что делать с короткими клипами, а не получить тихо усечённый
    # тензор неожиданной формы
    short = torch.zeros(1, 2, N_BINS, 5)
    try:
        Frontend.last_model_frames(short)
    except ValueError as e:
        assert "32" in str(e) and "5" in str(e), str(e)
    else:
        raise AssertionError("last_model_frames должен падать на T < MODEL_FRAMES")

    print("frontend selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
