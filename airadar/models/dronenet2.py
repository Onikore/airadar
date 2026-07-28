"""Сборка модели целиком (§3): ветка A + ветка B + слияние + attention-MIL
+ вспомогательная голова. Единственная точка входа модели — как
Frontend для признака (этап 2).

Размер (§3.5): 400 000-800 000 параметров. При branch_hidden=128 (ModelCfg
по умолчанию) — посчитано вручную при написании этого плана: BranchA
~277k, BranchB ~252k, слияние+MIL ~116 -> ~529.5k, в требуемом диапазоне.
selfcheck проверяет это кодом на реальной модели, не по расчёту на бумаге.
"""
import sys
import torch
import torch.nn as nn

from airadar.config import ModelCfg
from airadar.features.harmonic import harmonic_stack
from airadar.models.branches import BranchA, BranchB
from airadar.models.mil import AttentionMIL
from airadar.models.heads import AuxHead


class DroneNet2(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelCfg()
        self.branch_a = BranchA(hidden=self.cfg.branch_hidden)
        self.branch_b = BranchB(hidden=self.cfg.branch_hidden)
        self.fuse = nn.Linear(2, 1)
        self.mil = AttentionMIL(in_dim=2, hidden=self.cfg.mil_hidden)
        self.aux = AuxHead()

    def forward(self, feat):
        # feat: [B, 2, 183, T] — выход Frontend.last_model_frames (T=32 в
        # обычном режиме; ветки и MIL не требуют конкретного T).
        stacked = harmonic_stack(feat)                    # [B, 16, 80, T]
        frame_logit_a, evidence_a, f0_idx = self.branch_a(stacked)
        frame_logit_b = self.branch_b(feat)                # [B, T]

        frame_feat = torch.stack([frame_logit_a, frame_logit_b], dim=-1)  # [B,T,2]
        frame_value = self.fuse(frame_feat).squeeze(-1)                   # [B,T]
        clip_logit, attn = self.mil(frame_feat, frame_value)

        f0_hat, salience_hat = self.aux(evidence_a)

        return {
            "clip_logit": clip_logit,       # [B] — основной выход (логит вероятности дрона)
            "attn": attn,                   # [B,T] — веса MIL, диагностика
            "f0_hat": f0_hat,               # [B,T], Гц — вспом. голова
            "salience_hat": salience_hat,   # [B,T], дБ — вспом. голова
            "f0_idx": f0_idx,               # [B,T] long — трек f0 ветки A, диагностика
        }


def selfcheck():
    B, T = 2, 32
    model = DroneNet2()
    feat = torch.randn(B, 2, 183, T)
    out = model(feat)

    assert set(out) == {"clip_logit", "attn", "f0_hat", "salience_hat", "f0_idx"}
    assert out["clip_logit"].shape == (B,)
    assert out["attn"].shape == (B, T)
    assert out["f0_hat"].shape == (B, T)
    assert out["salience_hat"].shape == (B, T)
    assert out["f0_idx"].shape == (B, T)

    assert torch.allclose(out["attn"].sum(dim=-1), torch.ones(B), atol=1e-4)
    assert (out["f0_hat"] >= 40.0 - 1e-3).all() and (out["f0_hat"] <= 400.0 + 1e-3).all()

    # не падает на других T (94 -- полный выход Frontend на 12с, не только
    # обрезанные 32 кадра модели)
    feat94 = torch.randn(1, 2, 183, 94)
    out94 = model(feat94)
    assert out94["clip_logit"].shape == (1,)
    assert out94["attn"].shape == (1, 94)

    # градиент из clip_logit доходит до входа -- весь граф связан, ни одна
    # ветка не оторвана (частый класс багов: диагностический выход
    # случайно отделён detach()'ем или недифференцируемой операцией на
    # магистральном пути)
    feat.requires_grad_(True)
    model(feat)["clip_logit"].sum().backward()
    assert feat.grad is not None and torch.any(feat.grad != 0)

    # размер модели (§3.5): 400k-800k параметров
    n_params = sum(p.numel() for p in model.parameters())
    assert 400_000 <= n_params <= 800_000, n_params
    print(f"параметров: {n_params}")

    print("dronenet2 selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
