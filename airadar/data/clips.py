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
        # flush обязан довести записанное до файла, не дожидаясь close:
        # на нём держится обещание чекпоинта о размере clips.bin
        w.flush()
        assert os.path.getsize(path) == 15 * 4, os.path.getsize(path)
        # и довести именно до диска, а не до страничного кэша: чекпоинт
        # пишется с fsync, и пережить его данные обязаны, а не наоборот
        import inspect
        assert "fsync" in inspect.getsource(ClipWriter.flush), \
            "ClipWriter.flush обязан вызывать os.fsync"
        w.close()

        assert off_a == 0 and n_a == 10
        assert off_b == 10 and n_b == 5, (off_b, n_b)   # встык, без разрывов

        with ClipReader(path) as r:
            assert np.array_equal(r.read(off_a, n_a), a)
            assert np.array_equal(r.read(off_b, n_b), b)

            # чтение за пределами файла — явная ошибка, а не тихая усечка
            try:
                r.read(off_b, n_b + 1000)
            except ValueError:
                pass
            else:
                raise AssertionError("read должен проверять границы файла")
        # r.close() вызван через __exit__ — memmap освобождён по-настоящему,
        # а не по случайному моменту сборки мусора (важно на Windows, где
        # незакрытый memmap держит файл залоченным)

        # запись пустого клипа не должна ломать смещения следующего
        off_empty, n_empty = w2_offset_check(path)
        assert off_empty == 15 and n_empty == 0

    # неверный dtype на входе — явная ошибка, не молчаливое приведение
    with tempfile.TemporaryDirectory() as d:
        with ClipWriter(os.path.join(d, "c.bin")) as w:
            try:
                w.write(np.arange(5, dtype=np.int16))
            except ValueError:
                pass
            else:
                raise AssertionError("write должен требовать float32")
        # выход из with обязан закрыть файл даже после ошибки внутри блока
        assert w._f.closed, "ClipWriter.__exit__ не закрыл файл"

    # файл, оборванный на не кратном 4 размере, дописывать нельзя: иначе все
    # последующие клипы уедут на 1-3 байта относительно манифеста
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "torn.bin")
        with open(path, "wb") as f:
            f.write(b"\x00" * 10)              # 2.5 отсчёта float32
        try:
            ClipWriter(path, mode="ab")
        except AssertionError as e:
            assert "не кратен 4" in str(e), str(e)
        else:
            raise AssertionError("ClipWriter должен ловить оборванный clips.bin")

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
        # Оборванная запись (убитый процесс, кончившееся место) оставляет файл
        # на границе не кратной 4 байтам. Целочисленное деление ниже это
        # проглотит, и КАЖДЫЙ следующий клип уедет на 1-3 байта относительно
        # того, что о нём говорит манифест, — молча, до самого обучения.
        assert existing % 4 == 0, (
            f"clips.bin повреждён: размер {existing} не кратен 4 байтам (float32), "
            f"дописывать в него нельзя — предыдущая запись оборвалась")
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

    def flush(self):
        # Нужен перед сохранением чекпоинта: чекпоинт утверждает, сколько
        # отсчётов лежит в clips.bin, и это утверждение обязано быть верным в
        # момент записи. Иначе убитый процесс оставит файл короче обещанного.
        #
        # fsync здесь обязателен, а не избыточен. Чекпоинт пишется с fsync;
        # если clips.bin доходит только до страничного кэша, то при пропаже
        # питания переживёт обещание, а не данные, — и получится clips.bin
        # КОРОЧЕ обещанного, единственный случай, который _reconcile считает
        # невосстановимым. Цена — один fsync на шард против его же скачивания
        # и декодирования.
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        self._f.close()

    # Писатель открыт на всё время сборки (часы). Обрыв сети посреди прогона —
    # штатный сценарий, а не исключительный: без контекстного менеджера файл
    # закрывался бы только сборщиком мусора, в неопределённый момент.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ClipReader:
    """Читает клип по (offset, n_samples) через memmap — без чтения всего файла.

    На Windows memmap держит файл залоченным, пока жив объект — используйте
    `close()` или контекстный менеджер, а не полагайтесь на сборку мусора,
    иначе последующий open() того же файла на запись/удаление может упасть.
    """

    def __init__(self, bin_path):
        self._mm = np.memmap(bin_path, dtype=np.float32, mode="r")

    def read(self, offset, n_samples):
        end = offset + n_samples
        if end > len(self._mm):
            raise ValueError(
                f"клип [{offset}:{end}) выходит за пределы файла "
                f"(в файле {len(self._mm)} отсчётов) — манифест рассинхронизирован с clips.bin"
            )
        return np.array(self._mm[offset:end])

    def close(self):
        # у np.memmap нет явного close() — сброс ссылки и есть штатный способ
        # освободить mmap; без него на Windows файл остаётся залоченным
        self._mm = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
