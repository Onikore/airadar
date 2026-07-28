"""CLI: заполняет f0_med, salience, lf_energy во ВСЕХ строках манифеста.

CPU-only (HPS не использует GPU), без обращения к HF — только локальные
data/manifest.parquet и data/clips.bin, уже собранные cli/build_manifest.py.
Перезаписывает manifest.parquet на месте: это производный, не исходный
файл (data/ в .gitignore, clips.bin не трогается).

    python cli/label_manifest_f0.py            # весь манифест
    python cli/label_manifest_f0.py --limit 500  # проверка на куске
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pyarrow.parquet as pq
import pyarrow as pa

from airadar.data.clips import ClipReader
from airadar.data.f0label import label_row

MANIFEST_PATH = os.path.join(ROOT, "data", "manifest.parquet")
CLIPS_PATH = os.path.join(ROOT, "data", "clips.bin")


def main(limit=None):
    table = pq.read_table(MANIFEST_PATH)
    offsets = table.column("offset").to_pylist()
    n_samples = table.column("n_samples").to_pylist()
    n = table.num_rows if limit is None else min(limit, table.num_rows)

    f0s = [None] * table.num_rows
    sals = [None] * table.num_rows
    lfs = [None] * table.num_rows
    skipped = 0
    with ClipReader(CLIPS_PATH) as reader:
        for i in range(n):
            got = label_row(reader, offsets[i], n_samples[i])
            if got is None:
                skipped += 1
                continue
            f0s[i], sals[i], lfs[i] = got
            if i % 10000 == 0:
                print(f"[{i}/{n}] labeled, {skipped} skipped (too short)")

    table = table.set_column(table.schema.get_field_index("f0_med"), "f0_med",
                              pa.array(f0s, type=pa.float32()))
    table = table.set_column(table.schema.get_field_index("salience"), "salience",
                              pa.array(sals, type=pa.float32()))
    table = table.set_column(table.schema.get_field_index("lf_energy"), "lf_energy",
                              pa.array(lfs, type=pa.float32()))
    pq.write_table(table, MANIFEST_PATH)
    print(f"готово: {n} строк обработано, {skipped} пропущено (короче WIN)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(limit=a.limit)
