"""Прогон харнеса по одному чекпоинту.

    CUDA_VISIBLE_DEVICES= python cli/bench.py \
        --model models/dronenet_local.pt --name dronenet_local
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from airadar.bench.scorer import LegacyScorer
from airadar.bench.report import run_bench, write_report

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rep = run_bench(LegacyScorer(a.model, a.device), a.name, a.seed,
                model_path=a.model)
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
