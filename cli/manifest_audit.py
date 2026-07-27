"""Аудит собранного манифеста.

    python cli/manifest_audit.py
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pyarrow.parquet as pq
from airadar.bench.manifest_audit import audit

path = os.path.join(ROOT, "data", "manifest.parquet")
clips_bin = os.path.join(ROOT, "data", "clips.bin")
table = pq.read_table(path)

# Размер clips.bin — вход проверки замощения. Без него манифест сверяется
# только сам с собой, и усечённый файл клипов остаётся невидимым; отсутствие
# файла при существующем манифесте — уже дефект, а не повод проверять меньше.
if not os.path.exists(clips_bin):
    sys.exit(f"манифест есть, а {clips_bin} нет — хранилище клипов потеряно")
rep = audit(table, clips_bin_bytes=os.path.getsize(clips_bin))
print(json.dumps(rep, ensure_ascii=False, indent=2))
sys.exit(0 if rep["ok"] else 1)
