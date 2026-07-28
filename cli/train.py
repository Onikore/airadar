"""CLI: обучение DroneNet2 на реальном манифесте.

    python cli/train.py --limit 200 --epochs 2     # смоук-тест, минуты
    python cli/train.py --epochs 15 --run-name dronenet2_seed0 --save-every-epoch --seed 0
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from airadar.train.loop import main

MANIFEST_PATH = os.path.join(ROOT, "data", "manifest.parquet")
CLIPS_PATH = os.path.join(ROOT, "data", "clips.bin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--run-name", default="dronenet2")
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="фиксирует torch/numpy RNG перед стартом — для 2 seed §9")
    a = ap.parse_args()

    if a.seed is not None:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)

    main(MANIFEST_PATH, CLIPS_PATH, epochs=a.epochs, bs=a.bs, lr=a.lr,
        out_dir=a.out_dir, limit=a.limit, run_name=a.run_name,
        save_every_epoch=a.save_every_epoch)
