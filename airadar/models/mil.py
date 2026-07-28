"""Attention-MIL (§3.3): клип-логит — обучаемая взвешенная сумма покадровых
логитов. Заменяет правило "k из m" полностью (правило удалено из системы).

Формула — gated attention (Ilse et al., 2018): score_t = w^T(tanh(V h_t) *
sigmoid(U h_t)), attn = softmax(score) по кадрам, clip_logit = Σ attn_t *
value_t. Gate (tanh * sigmoid) даёт сети возможность подавлять кадр, а не
только взвешивать положительно — нужно, когда цель слышна не всё время.
"""
import sys
import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.U = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, frame_feat, frame_value):
        # frame_feat: [B,T,in_dim], frame_value: [B,T]
        gate = torch.tanh(self.V(frame_feat)) * torch.sigmoid(self.U(frame_feat))
        score = self.w(gate).squeeze(-1)              # [B,T]
        attn = torch.softmax(score, dim=-1)            # [B,T], сумма=1 по кадрам
        clip_logit = (attn * frame_value).sum(dim=-1)  # [B]
        return clip_logit, attn


def selfcheck():
    B, T, D = 3, 32, 2
    mil = AttentionMIL(in_dim=D, hidden=16)
    feat = torch.randn(B, T, D)
    value = torch.randn(B, T)

    clip_logit, attn = mil(feat, value)
    assert clip_logit.shape == (B,), clip_logit.shape
    assert attn.shape == (B, T), attn.shape
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (attn >= 0).all()                     # softmax -> неотрицательно

    # clip_logit обязан лежать в выпуклой оболочке value (attn суммируется в
    # 1 и неотрицателен) — простая проверка на здравый смысл формулы
    assert (clip_logit >= value.min(dim=-1).values - 1e-4).all()
    assert (clip_logit <= value.max(dim=-1).values + 1e-4).all()

    # градиент течёт в обе стороны (feat и value)
    feat.requires_grad_(True)
    value.requires_grad_(True)
    clip_logit2, _ = mil(feat, value)
    clip_logit2.sum().backward()
    assert feat.grad is not None and torch.any(feat.grad != 0)
    assert value.grad is not None and torch.any(value.grad != 0)

    print("mil selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
