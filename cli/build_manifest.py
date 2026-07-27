"""Полная сборка манифеста и хранилища клипов из четырёх источников HF.

    python cli/build_manifest.py --limit 1     # по одному шарду источника, проверка
    python cli/build_manifest.py               # полный прогон — часы, IO-связанный

Чекпоинт по шардам, как в prep_hf.py: строки манифеста каждого шарда сразу
пишутся в свой parquet-файл части (data/parts/<ключ>.parquet), а в JSON-
чекпоинте хранятся только список готовых ключей и next_clip_id — не сами
строки. Иначе на полном прогоне (десятки тысяч строк, 85 шардов) JSON
чекпоинта пришлось бы целиком перезаписывать после каждого шарда, и это
стало бы тяжелее с каждым шардом. Финальная сборка — конкатенация всех
частей одним проходом в конце, а не накопление в памяти по ходу.
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pyarrow as pa
import pyarrow.parquet as pq
import hf_sources as S
from airadar.data.clips import ClipWriter
from airadar.data.build import ingest_shard, _key
from airadar.data.manifest import rows_to_table, MANIFEST_VERSION
from airadar.data.split import apply_split

OUT_DIR = os.path.join(ROOT, "data")
PARTS_DIR = os.path.join(OUT_DIR, "parts")
CLIPS_BIN = os.path.join(OUT_DIR, "clips.bin")
MANIFEST_PARQUET = os.path.join(OUT_DIR, "manifest.parquet")
CHECKPOINT_JSON = os.path.join(OUT_DIR, "build_checkpoint.json")
ORDER = (S.SRC_ESC, S.SRC_URBAN, S.SRC_DAS, S.SRC_DADS)   # мелкие источники первыми


def _load_checkpoint():
    if os.path.exists(CHECKPOINT_JSON):
        with open(CHECKPOINT_JSON) as f:
            return json.load(f)
    return {"done": [], "next_clip_id": 0}


def _save_checkpoint(ck):
    with open(CHECKPOINT_JSON, "w") as f:
        json.dump(ck, f)


def _ingest_all(limit=None):
    """Скачивает и пишет недостающие шарды. Возвращает next_clip_id."""
    os.makedirs(PARTS_DIR, exist_ok=True)
    ck = _load_checkpoint()
    done = set(ck["done"])
    writer = ClipWriter(CLIPS_BIN, mode="ab" if done else "wb")
    next_id = ck["next_clip_id"]

    for src in ORDER:
        rels = S.shards(src)
        if limit is not None:
            rels = rels[:limit]
        for i, rel in enumerate(rels):
            key = _key(src, i)
            if key in done:
                continue
            print(f"[{S.NAMES[src]} {i+1}/{len(rels)}] {rel}")
            path = S.local_shard(src, rel)
            new_rows, next_id = ingest_shard(S.read_shard(src, path), writer, next_id)
            if new_rows:
                pq.write_table(rows_to_table(new_rows),
                               os.path.join(PARTS_DIR, f"{key}.parquet"))
            done.add(key)
            _save_checkpoint({"done": sorted(done), "next_clip_id": next_id})

    writer.close()
    return next_id


def _assemble():
    """Склеивает все части в один манифест и назначает сплит один раз."""
    parts = sorted(os.path.join(PARTS_DIR, f) for f in os.listdir(PARTS_DIR)
                   if f.endswith(".parquet"))
    if not parts:
        sys.exit("нет ни одной части — сборка не выполнялась")
    # concat_tables — в pyarrow, а не в pyarrow.parquet (в отличие от read_table/write_table)
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    table = apply_split(table)
    pq.write_table(table, MANIFEST_PARQUET)
    print(f"манифест: {table.num_rows} строк из {len(parts)} частей -> {MANIFEST_PARQUET}")
    print(f"клипы: -> {CLIPS_BIN}")


def main(limit=None):
    _ingest_all(limit=limit)
    _assemble()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="шардов на источник (для проверки, не для полного прогона)")
    a = ap.parse_args()
    main(limit=a.limit)
