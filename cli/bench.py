"""Прогон харнеса по одному чекпоинту.

    CUDA_VISIBLE_DEVICES= python cli/bench.py \
        --model models/dronenet_local.pt --name dronenet_local
    python cli/bench.py \
        --model models/dronenet2_last.pt --name dronenet2_smoke --arch dronenet2
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from airadar.bench.scorer import LegacyScorer, DroneNet2Scorer
from airadar.bench.report import run_bench, write_report

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arch", choices=["legacy", "dronenet2"], default="legacy")
ap.add_argument("--hop-s", type=float, default=1.0,
                help="только для --arch dronenet2 — шаг между окнами бенча")
a = ap.parse_args()

if a.arch == "legacy":
    scorer = LegacyScorer(a.model, a.device)
else:
    scorer = DroneNet2Scorer(a.model, a.device, hop_s=a.hop_s)

rep = run_bench(scorer, a.name, a.seed, model_path=a.model)
jp, mp = write_report(rep)

# Отчёт написан на диск (файлы уже в UTF-8) до этой точки — печать в консоль
# не должна ронять прогон. На Windows sys.stdout по умолчанию открыт в
# кодовой странице консоли (cp1251/cp866), которая не знает символов вроде
# '→' и роняет print с UnicodeEncodeError уже ПОСЛЕ того, как отчёт записан.
# Явно переключаем поток на UTF-8, а всё, что всё-таки не отобразится
# (легаси-консоль без UTF-8), заменяем, а не падаем.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(open(mp, encoding="utf-8").read())
print(f"записано: {jp}  {mp}")
