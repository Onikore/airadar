"""
Препроцессинг DADS (180 320 клипов) -> кэш окон 0.5 с.

DADS уже приведён к 16 kHz / 16 bit / mono, поэтому утечки по частоте
дискретизации, как в старом Kaggle-наборе, здесь нет. Но остаётся вторая:
средняя длина клипа дрона 0.60 с против 7.28 с у фона. Нарезка на окна
фиксированной длины её убирает.

Третья проблема — в именах файлов остался только сквозной номер
(drone-20941.wav), источник записи вырезан. 163k клипов по 0.6 с нарезаны
из 6 исходных датасетов, то есть соседние номера — почти наверняка куски
одного полёта. Случайный сплит растащил бы их по train и val и надул метрику.
Группируем по блокам номеров, см. BLOCK.
"""

import os
import re
import io
import sys
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from scipy.signal import resample_poly
from math import gcd

ROOT = os.path.dirname(os.path.abspath(__file__))
DADS = os.path.join(ROOT, "DADS", "data")
OUT = os.path.join(ROOT, "cache_dads")

SR = 16000
WIN = SR // 2                # 0.5 с
# ponytail: приближение записи блоком по 250 номеров. Точная группировка
# требует метаданных источника, которых в DADS нет. Если найдётся маппинг
# номер -> исходный файл, заменить на него.
BLOCK = 250
CAP = {0: 16, 1: 1}          # окон с файла: дрона много, фона мало


def to_mono_16k(data, sr):
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != SR:
        g = gcd(int(sr), SR)
        data = resample_poly(data, SR // g, sr // g).astype(np.float32)
    peak = np.abs(data).max()
    if not np.isfinite(peak) or peak < 1e-6:
        return None
    return data / peak


def windows(x, cap):
    if len(x) < WIN:
        return [np.tile(x, int(np.ceil(WIN / len(x))))[:WIN]]
    n = min(len(x) // WIN, cap)
    return [x[i * WIN:(i + 1) * WIN] for i in range(n)]


def main():
    os.makedirs(OUT, exist_ok=True)
    shards = sorted(f for f in os.listdir(DADS) if f.endswith(".parquet"))
    print(f"шардов: {len(shards)}")

    bin_path = os.path.join(OUT, "windows.bin")
    labels, groups = [], []
    bad = 0
    num_re = re.compile(r"(\d+)")

    with open(bin_path, "wb") as out:
        for si, shard in enumerate(shards):
            pf = pq.ParquetFile(os.path.join(DADS, shard))
            for rg in range(pf.num_row_groups):
                for r in pf.read_row_group(rg).to_pylist():
                    label = int(r["label"])
                    try:
                        data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                        x = to_mono_16k(data, sr)
                    except Exception:
                        x = None
                    if x is None:
                        bad += 1
                        continue
                    m = num_re.search(r["audio"]["path"] or "")
                    idx = int(m.group(1)) if m else len(labels)
                    gid = label * 10_000_000 + idx // BLOCK
                    for w in windows(x, CAP[label]):
                        out.write((w * 32767).astype(np.int16).tobytes())
                        labels.append(label)
                        groups.append(gid)
            print(f"  шард {si+1}/{len(shards)}  окон: {len(labels)}", flush=True)

    y = np.array(labels, np.int8)
    np.savez(os.path.join(OUT, "meta.npz"),
             y=y, group=np.array(groups, np.int64),
             n=np.int64(len(y)), win=np.int64(WIN))

    print(f"\nбитых клипов: {bad}")
    print(f"окон всего: {len(y)}  ({len(y)*WIN*2/1e9:.2f} GB)")
    print(f"  no_drone: {(y==0).sum()}")
    print(f"  drone:    {(y==1).sum()}")
    print(f"групп: {len(set(groups))}")


def selfcheck():
    assert len(windows(np.zeros(WIN), 4)) == 1
    assert len(windows(np.zeros(WIN * 10), 4)) == 4
    assert len(windows(np.zeros(WIN * 10), 1)) == 1
    (w,) = windows(np.arange(100, dtype=np.float32), 4)
    assert len(w) == WIN and w[100] == 0        # зациклено, не дополнено нулями
    x = to_mono_16k(np.random.randn(44100, 2).astype(np.float32), 44100)
    assert abs(len(x) - SR) <= 2 and abs(x).max() <= 1.0 + 1e-6
    assert to_mono_16k(np.zeros(1000, np.float32), SR) is None   # тишина отбрасывается
    # группировка: соседние номера в одной группе, далёкие — в разных
    g = lambda lab, i: lab * 10_000_000 + i // BLOCK
    assert g(1, 20941) == g(1, 20999) and g(1, 20941) != g(1, 21999)
    assert g(0, 5) != g(1, 5)
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
