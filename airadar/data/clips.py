"""Хранилище клипов: конкатенация float32 подряд, без разделителей.

Кэш окон хранил 50% перекрытия — половина диска дублировалась. Здесь клип
пишется один раз целиком; смещение и длина — в манифесте, а не в отдельном
индексном файле: манифест и есть индекс, второй источник истины не нужен.
"""

import os
import sys
import numpy as np


def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clips.bin")
        w = ClipWriter(path)
        a = np.arange(10, dtype=np.float32)
        b = np.arange(100, 105, dtype=np.float32)
        off_a, n_a = w.write(a)
        off_b, n_b = w.write(b)
        w.close()

        assert off_a == 0 and n_a == 10
        assert off_b == 10 and n_b == 5, (off_b, n_b)   # встык, без разрывов

        r = ClipReader(path)
        assert np.array_equal(r.read(off_a, n_a), a)
        assert np.array_equal(r.read(off_b, n_b), b)
        del r   # на Windows memmap держит файл открытым, пока жив объект —
                # иначе следующий open("ab") и удаление tempdir могут упасть

        # запись пустого клипа не должна ломать смещения следующего
        off_empty, n_empty = w2_offset_check(path)
        assert off_empty == 15 and n_empty == 0

    # неверный dtype на входе — явная ошибка, не молчаливое приведение
    with tempfile.TemporaryDirectory() as d:
        w = ClipWriter(os.path.join(d, "c.bin"))
        try:
            w.write(np.arange(5, dtype=np.int16))
        except ValueError:
            pass
        else:
            raise AssertionError("write должен требовать float32")
        finally:
            w.close()

    print("clips selfcheck ok")


def w2_offset_check(path):
    w = ClipWriter(path, mode="ab")
    off, n = w.write(np.zeros(0, dtype=np.float32))
    w.close()
    return off, n


class ClipWriter:
    """Пишет клипы подряд, без разделителей. offset — в ОТСЧЁТАХ, не байтах."""

    def __init__(self, bin_path, mode="wb"):
        # self._f.tell() сразу после open("ab") ненадёжен: CPython не
        # гарантирует, что буферизованный поток отражает реальную позицию
        # конца файла до первой записи. os.path.getsize — то же самое,
        # но без этой двусмысленности.
        existing = os.path.getsize(bin_path) if (mode == "ab" and os.path.exists(bin_path)) else 0
        self._f = open(bin_path, mode)
        self._pos = existing // 4           # float32 = 4 байта

    def write(self, audio):
        audio = np.asarray(audio)
        if audio.dtype != np.float32:
            raise ValueError(f"клип должен быть float32, получено {audio.dtype}")
        offset = self._pos
        self._f.write(audio.tobytes())
        self._pos += len(audio)
        return offset, len(audio)

    def close(self):
        self._f.close()


class ClipReader:
    """Читает клип по (offset, n_samples) через memmap — без чтения всего файла."""

    def __init__(self, bin_path):
        self._mm = np.memmap(bin_path, dtype=np.float32, mode="r")

    def read(self, offset, n_samples):
        return np.array(self._mm[offset:offset + n_samples])


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
