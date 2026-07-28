"""Вспомогательная голова f0/salience (§3.4). Не продукт — в рантайме не
используется, только при обучении как регуляризация.

Без своих обучаемых параметров: f0 — softmax-взвешенное ожидание по карте
evidence ветки A (дифференцируемо), salience — max минус медиана по f0-
бинам, то же определение, что в airadar/data/f0label.py, чтобы предсказание
и метка (манифест) были в одной системе координат.
"""
import sys
import torch
import torch.nn as nn

from airadar.features.harmonic import N_F0_BINS, F0_LO, F0_HI
from airadar.features.cqt import BINS_PER_OCTAVE


class AuxHead(nn.Module):
    def __init__(self):
        super().__init__()
        f0_grid = F0_LO * 2.0 ** (torch.arange(N_F0_BINS, dtype=torch.float32)
                                   / BINS_PER_OCTAVE)
        self.register_buffer("f0_grid", f0_grid)   # [N_F0_BINS], Гц

    def forward(self, evidence):
        # evidence: [B, N_F0_BINS, T] от BranchA
        p = torch.softmax(evidence, dim=1)                        # [B,N_F0_BINS,T]
        f0_hat = (p * self.f0_grid[None, :, None]).sum(dim=1)     # [B,T], Гц
        salience_hat = evidence.max(dim=1).values - evidence.median(dim=1).values
        return f0_hat, salience_hat


def selfcheck():
    B, T = 2, 32
    head = AuxHead()
    assert head.f0_grid.shape == (N_F0_BINS,)
    assert abs(head.f0_grid[0].item() - F0_LO) < 0.1
    assert head.f0_grid[-1].item() <= F0_HI * 1.05   # верхний край около F0_HI

    evidence = torch.randn(B, N_F0_BINS, T)
    f0_hat, salience_hat = head(evidence)
    assert f0_hat.shape == (B, T) and salience_hat.shape == (B, T)
    # f0_hat -- выпуклая комбинация f0_grid, обязана лежать в [F0_LO, F0_HI]
    assert (f0_hat >= F0_LO - 1e-3).all() and (f0_hat <= F0_HI + 1e-3).all()

    # выраженный пик на конкретном f0-бине -> f0_hat близко к этому бину
    evidence_peaked = torch.full((1, N_F0_BINS, 1), -10.0)
    evidence_peaked[0, 40, 0] = 10.0
    f0_hat_p, _ = head(evidence_peaked)
    assert abs(f0_hat_p.item() - head.f0_grid[40].item()) < 5.0, f0_hat_p.item()

    # градиент доходит до evidence
    evidence.requires_grad_(True)
    f0_hat2, sal2 = head(evidence)
    (f0_hat2.sum() + sal2.sum()).backward()
    assert evidence.grad is not None and torch.any(evidence.grad != 0)

    print("heads selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
