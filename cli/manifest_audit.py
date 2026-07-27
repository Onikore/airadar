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
table = pq.read_table(path)
rep = audit(table)
print(json.dumps(rep, ensure_ascii=False, indent=2))
sys.exit(0 if rep["ok"] else 1)
