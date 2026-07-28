"""CLI: обучение DroneNet2 на реальном манифесте.

    python cli/train.py --limit 200 --epochs 2     # смоук-тест, минуты
    python cli/train.py --epochs 30                # полный прогон
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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
    a = ap.parse_args()
    main(MANIFEST_PATH, CLIPS_PATH, epochs=a.epochs, bs=a.bs, lr=a.lr,
        out_dir=a.out_dir, limit=a.limit)
