"""Harmonic stacking (§3.1): для f0-бина i собираются бины
i + round(24·log2 k), k=1..8 — чистое индексирование, вычислений ноль.

Смысл: на лог-частотной оси гармоника k эквидистантна от основной частоты
НЕЗАВИСИМО от самой f0 (self-similarity лог-оси) — тот же принцип, что делает
f0-сдвиг (§4.1) чистой трансляцией. round(24·log2(k)) — это смещение в
бинах между гармониками k и 1 на сетке 24 бина/октаву; для f0-бина i индекс
k-й гармоники — i + это смещение, всегда, вне зависимости от i.

F0_HI поднят с 400 до 800 Гц (2026-07-28, по факту, не заранее): реальная
FPV-запись (гонка/фристайл, близкий пролёт) дала измеренную стабильную
доминанту ~390-400 Гц — ровно на старой верхней границе диапазона поиска.
Модель на ней путалась (свой f0_hat давал 140-180 Гц вместо истинных
~390 Гц) — не архитектурный сбой, а недообученность у самой границы:
исходный диапазон 40-400 Гц был выбран под тяжёлые дроны (низкий f0,
причина всей переработки, §1), а не под быстрые FPV с мелким пропеллером
на высоких оборотах — те физически лежат выше. Проверено: 800 Гц всё ещё
укладывается в сетку CQT с запасом (_MAX_INDEX ниже, 175 < 183 бинов).
"""
import sys
import numpy as np
import torch

from airadar.features.cqt import BINS_PER_OCTAVE, N_BINS

F0_LO, F0_HI = 40.0, 800.0
N_HARMONICS = 8
N_F0_BINS = round(BINS_PER_OCTAVE * np.log2(F0_HI / F0_LO))   # 104, см. докстринг

_OFFSETS = [round(BINS_PER_OCTAVE * np.log2(k)) for k in range(1, N_HARMONICS + 1)]
_MAX_INDEX = (N_F0_BINS - 1) + max(_OFFSETS)
assert _MAX_INDEX < N_BINS, (
    f"harmonic stacking выходит за сетку CQT: индекс {_MAX_INDEX} >= {N_BINS}")


def harmonic_stack(x):
    """x: [B, C, N_BINS, T] (лог-CQT, любое число каналов C) ->
    [B, C*N_HARMONICS, N_F0_BINS, T].

    Канал (c, k) на f0-бине i — это x[:, c, i + _OFFSETS[k], :]."""
    B, C, F, T = x.shape
    assert F == N_BINS, (F, N_BINS)
    idx = torch.tensor(
        [[i + off for off in _OFFSETS] for i in range(N_F0_BINS)],
        device=x.device, dtype=torch.long)          # [N_F0_BINS, N_HARMONICS]
    gathered = x[:, :, idx, :]                       # [B, C, N_F0_BINS, N_HARMONICS, T]
    gathered = gathered.permute(0, 1, 3, 2, 4)        # [B, C, N_HARMONICS, N_F0_BINS, T]
    return gathered.reshape(B, C * N_HARMONICS, N_F0_BINS, T)


def selfcheck():
    assert N_F0_BINS == 104, N_F0_BINS   # 24*log2(800/40), см. докстринг про поднятый F0_HI
    assert _OFFSETS == [0, 24, 38, 48, 56, 62, 67, 72], _OFFSETS

    # уникальное значение на (channel, freq_bin) паре -> из выхода можно
    # однозначно восстановить, какой исходный бин был взят на gather
    B, C, T = 1, 2, 4
    x = torch.zeros(B, C, N_BINS, T)
    for c in range(C):
        for f in range(N_BINS):
            x[0, c, f, :] = c * 1000 + f            # кодирует (c, f), время не участвует

    out = harmonic_stack(x)
    assert out.shape == (B, C * N_HARMONICS, N_F0_BINS, T), out.shape

    # проверка адресации: канал (c=1, k=5 индекс 4 -> offset 56), f0-бин i=10
    # обязан читать исходный бин f = 10 + 56 = 66, значение 1*1000+66=1066
    c, k_idx, i = 1, 4, 10
    out_channel = c * N_HARMONICS + k_idx
    expected = c * 1000 + (i + _OFFSETS[k_idx])
    assert out[0, out_channel, i, 0].item() == expected, \
        (out[0, out_channel, i, 0].item(), expected)

    # k=1 (offset 0) на f0-бине i обязан совпасть с исходным ch0 на том же бине
    for i in (0, 40, 103):
        assert out[0, 0, i, 0].item() == 0 * 1000 + i

    # градиент доходит до исходного тензора через gather (не detached)
    x2 = torch.randn(1, 2, N_BINS, T, requires_grad=True)
    harmonic_stack(x2).sum().backward()
    assert x2.grad is not None and torch.any(x2.grad != 0)

    print("harmonic selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
