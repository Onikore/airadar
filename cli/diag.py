"""Диагностики, не требующие обучения.

    python cli/diag.py dads-contiguity --shard 0 --n 300
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["dads-contiguity"])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()

import numpy as np
import hf_sources
from huggingface_hub import hf_hub_download
from airadar.diag.dads_contiguity import scan_shard, verdict

# Шард уже лежит в локальном кэше HF от предыдущего prep_hf.py (39 файлов
# train-NNNNN-of-00039.parquet, подтверждено на диске). local_files_only=True
# резолвит путь прямо из кэша без единого сетевого запроса, а значит и без
# hub.token() — тот безусловно требует токен даже для уже скачанных файлов
# (hf_sources.shards/local_shard идут через него), что здесь не нужно.
repo, _pat = hf_sources.REPOS[hf_sources.SRC_DADS]
n_shards = hf_sources.N_SHARDS[hf_sources.SRC_DADS]
rel = f"data/train-{a.shard:05d}-of-{n_shards:05d}.parquet"
path = hf_hub_download(repo, rel, repo_type="dataset", local_files_only=True)
adj, ctl = scan_shard(path, a.n)
print(f"шард {rel}: пар соседних {len(adj)}, контрольных {len(ctl)}")
if len(adj) == 0 or len(ctl) == 0:
    # Шарды DADS не перемешаны по меткам: 0-2 сплошь no-drone, 3 переходный,
    # 4+ сплошь drone (проверено по всем 39 файлам кэша). Пустой шард — не
    # баг скрипта, а факт раскладки данных, и его надо напечатать, а не
    # уронить процесс в IndexError на np.quantile(пустой массив).
    print("D0: в шарде нет позитивных клипов (label==1) — вердикт не считается")
    sys.exit(0)
print(f"скачок соседних:  медиана {np.median(adj):.2f}  p90 {np.quantile(adj, .9):.2f}")
print(f"скачок контроля:  медиана {np.median(ctl):.2f}  p90 {np.quantile(ctl, .9):.2f}")
ok, msg = verdict(adj, ctl)
print(f"\nD0: {msg}")
