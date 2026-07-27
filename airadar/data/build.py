"""Адаптер: hf_sources.Rec -> (клип в хранилище, строка манифеста).

hf_sources.read_shard уже отдаёт клипы целиком, не окна — windows() режет
их позже, при обучении. Поэтому здесь нет реконструкции: один Rec -> одна
строка манифеста, без склейки соседних индексов. Находка D0 (этап 0)
показала, что для DADS такая склейка была бы всё равно неверна — соседние
индексы не смежны физически.
"""

import sys
import numpy as np
from collections import namedtuple

from airadar.data.manifest import make_row, validate_row
from airadar.data.clips import ClipWriter

FakeRec = namedtuple("Rec", "audio group label cat src")


def selfcheck():
    import tempfile, os
    recs = [
        FakeRec(np.ones(100, np.float32), group=5, label=1, cat=None, src=0),
        FakeRec(np.zeros(50, np.float32), group=7, label=0, cat="wind", src=3),
        # SRC_DAS отдельно: без него _domain могла бы перепутать ветки
        # das_rig_/dads_block_ местами, а selfcheck этого бы не заметил
        FakeRec(np.full(30, 2.0, np.float32), group=9, label=1, cat=None, src=1),
    ]
    with tempfile.TemporaryDirectory() as d:
        w = ClipWriter(os.path.join(d, "clips.bin"))
        rows, next_id = ingest_shard(iter(recs), w, next_clip_id=100)
        w.close()

        assert next_id == 103, next_id
        assert len(rows) == 3
        assert rows[0]["clip_id"] == 100 and rows[1]["clip_id"] == 101
        assert rows[2]["clip_id"] == 102
        assert rows[0]["offset"] == 0 and rows[0]["n_samples"] == 100
        assert rows[1]["offset"] == 100 and rows[1]["n_samples"] == 50   # встык
        assert rows[2]["offset"] == 150 and rows[2]["n_samples"] == 30   # встык
        assert rows[0]["label"] == 1 and rows[0]["category"] is None
        assert rows[1]["label"] == 0 and rows[1]["category"] == "wind"
        assert rows[2]["label"] == 1 and rows[2]["category"] is None
        assert rows[0]["domain"] == "dads_block_5", rows[0]["domain"]
        assert rows[1]["domain"] == "scene_7", rows[1]["domain"]
        assert rows[2]["domain"] == "das_rig_9", rows[2]["domain"]
        assert rows[0]["synth"] is False
        assert rows[1]["synth"] is False
        assert rows[2]["synth"] is False

        for r in rows:
            validate_row(r)                # строки обязаны проходить схему

    # пустой генератор — пустой результат, next_clip_id не меняется
    with tempfile.TemporaryDirectory() as d:
        w = ClipWriter(os.path.join(d, "c.bin"))
        rows2, next_id2 = ingest_shard(iter([]), w, next_clip_id=0)
        w.close()
        assert rows2 == [] and next_id2 == 0

    assert _key(0, 3) == "0_0003"

    print("build selfcheck ok")


SRC_DADS, SRC_DAS, SRC_URBAN, SRC_ESC = 0, 1, 2, 3   # см. hf_sources.py


def _domain(rec):
    if rec.src == SRC_DAS:
        return f"das_rig_{rec.group}"
    if rec.src == SRC_DADS:
        return f"dads_block_{rec.group}"
    return f"scene_{rec.group}"


def rec_to_row(rec, clip_id, offset, n_samples):
    return make_row(
        clip_id=clip_id, src=rec.src, offset=offset, n_samples=n_samples,
        label=rec.label, label_conf=1.0, group_id=rec.group,
        domain=_domain(rec), category=rec.cat, synth=False,
    )


def ingest_shard(rec_iter, writer, next_clip_id):
    rows = []
    cid = next_clip_id
    for rec in rec_iter:
        offset, n = writer.write(rec.audio.astype(np.float32))
        rows.append(rec_to_row(rec, cid, offset, n))
        cid += 1
    return rows, cid


def _key(src, i):
    return f"{src}_{i:04d}"


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
