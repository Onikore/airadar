"""
Эксперимент: DroneNetV2 (depthwise-separable + inverted bottleneck + SE +
residual) против текущего DroneNet, на тех же данных и признаках.

Инфраструктура (LogMel, Augment, загрузчики, цикл обучения) не дублируется —
импортируется из train.py и параметризуется через model_cls/out_name. Это
держит сравнение честным: меняется только архитектура, n_fft/гул/данные те же,
что у эталонного models/dronenet.pt (n_fft=2048, гул 0.25, AUC_fh 0.780 на
трудной полевой записи — см. README).

Важные ограничения по сравнению с исходным предложением:
- SEBlock — это внимание по КАНАЛАМ (AdaptiveAvgPool2d(1) схлопывает и частоту,
  и время), не по mel-полосам. Он не может "давить полосу 50 Гц" — такое
  требует отдельного частотного внимания, которого здесь нет. Оставлено как
  задокументированное ограничение, не как то, что блок якобы умеет.
- Без AMP/torch.compile/cudnn.benchmark: это отдельные оптимизации скорости,
  не архитектуры, и вносят лишнюю переменную в контролируемое сравнение.
"""

import sys
import torch
import torch.nn as nn

import train
from train import DEV


class SEBlock(nn.Module):
    """Squeeze-and-Excitation — внимание по каналам (не по mel-полосам)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(x).view(b, c, 1, 1)


class ConvNeXtBlock(nn.Module):
    """Inverted bottleneck: depthwise 5x5 -> расширение 2x -> SE -> сжатие, residual."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1)
        self.act = nn.SiLU()
        self.se = SEBlock(2 * dim)
        self.pwconv2 = nn.Conv2d(2 * dim, dim, kernel_size=1)

    def forward(self, x):
        r = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pwconv2(x)
        return r + x


class DroneNetV2(nn.Module):
    """Depthwise-separable CNN с SE-вниманием и residual-связями."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
        )
        self.stage1 = nn.Sequential(ConvNeXtBlock(32), ConvNeXtBlock(32), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            ConvNeXtBlock(64), ConvNeXtBlock(64), nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            ConvNeXtBlock(128), ConvNeXtBlock(128), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x).squeeze(1)  # (B,), не (B,1) — иначе BCEWithLogitsLoss сломается broadcasting-ом


def selfcheck():
    m = DroneNetV2().to(DEV)
    x = torch.randn(4, 1, train.N_MELS, 63, device=DEV)
    y = m(x)
    assert y.shape == (4,), y.shape
    assert torch.isfinite(y).all()
    n = sum(p.numel() for p in m.parameters())
    print(f"параметров: {n/1e3:.0f}k")
    assert 100_000 < n < 400_000, f"вне ожидаемого диапазона: {n}"
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
        train.main(epochs=epochs, model_cls=DroneNetV2, out_name="dronenet_v2.pt")
