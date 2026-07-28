"""Ветка A (гармоническая) и ветка B (текстурная), §3.1/§3.2.

Обе схлопывают частотную ось в покадровое свидетельство через logsumexp —
не max (чувствителен к одному пику, шумно) и не mean (топит слабую
гребёнку в широкой полосе). logsumexp — гладкая аппроксимация max,
дифференцируема всюду, и на практике ведёт себя как "мягкий OR" по бинам:
если хотя бы один бин на кадре t горячий, кадр горячий.

Обе ветки stride=1 по обеим осям — размер по времени T сохраняется точно,
это нужно для поэлементного слияния логитов веток (airadar/models/dronenet2.py).
"""
import sys
import torch
import torch.nn as nn

from airadar.features.harmonic import N_F0_BINS, N_HARMONICS


class BranchA(nn.Module):
    """Вход: [B, C*N_HARMONICS, N_F0_BINS, T] (после harmonic_stack).
    C=2 (ch0, ch1) -> вход 16 каналов."""

    def __init__(self, in_channels=2 * N_HARMONICS, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, stacked):
        evidence = self.conv(stacked).squeeze(1)         # [B, N_F0_BINS, T]
        frame_logit = torch.logsumexp(evidence, dim=1)    # [B, T]
        f0_idx = evidence.argmax(dim=1)                   # [B, T], диагностика
        return frame_logit, evidence, f0_idx


class BranchB(nn.Module):
    """Вход: [B, 2, N_BINS, T] — полная сетка CQT, оба канала (ch0, ch1).

    Первый слой — более широкое ядро по частоте (7 против 5 у ветки A):
    ловит широкополосный лопастной шум малых FPV, где гребёнка слабее
    относительно шума ротора (§3.2), а не узкую гармонику."""

    def __init__(self, in_channels=2, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        evidence = self.conv(x).squeeze(1)                # [B, N_BINS, T]
        frame_logit = torch.logsumexp(evidence, dim=1)     # [B, T]
        return frame_logit


def selfcheck():
    B, T = 2, 32
    a = BranchA()
    stacked = torch.randn(B, 2 * N_HARMONICS, N_F0_BINS, T)
    frame_logit, evidence, f0_idx = a(stacked)
    assert frame_logit.shape == (B, T), frame_logit.shape
    assert evidence.shape == (B, N_F0_BINS, T), evidence.shape
    assert f0_idx.shape == (B, T), f0_idx.shape
    assert f0_idx.dtype == torch.long
    assert (f0_idx >= 0).all() and (f0_idx < N_F0_BINS).all()

    b = BranchB()
    x = torch.randn(B, 2, 183, T)
    frame_logit_b = b(x)
    assert frame_logit_b.shape == (B, T), frame_logit_b.shape

    # T сохраняется точно на других длинах, не только 32 — Frontend может
    # отдать больше кадров (12с -> 94), ветки обязаны не падать
    for T2 in (5, 94):
        stacked2 = torch.randn(1, 2 * N_HARMONICS, N_F0_BINS, T2)
        fl, ev, fi = a(stacked2)
        assert fl.shape == (1, T2) and ev.shape == (1, N_F0_BINS, T2)
        x2 = torch.randn(1, 2, 183, T2)
        assert b(x2).shape == (1, T2)

    # градиент доходит до входа обеих веток
    stacked.requires_grad_(True)
    a(stacked)[0].sum().backward()
    assert stacked.grad is not None and torch.any(stacked.grad != 0)

    x.requires_grad_(True)
    b(x).sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0)

    print("branches selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
