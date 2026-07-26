"""Чтение четырёх источников HF в единый вид Rec.

У каждого источника своя схема parquet и своё правило группировки. Смысл
группировки один: куски одной физической записи не должны расползтись по
train и val, иначе метрика надувается. Правила разной точности — у DADS имена
обрезаны и остаётся приближение блоками, у остальных ключ точный.

Схемы разведаны чтением футеров parquet 2026-07-26, подробности и обоснования —
docs/superpowers/plans/2026-07-26-colab-dataset.md, раздел «Разведанные факты».
"""

import io
import re
import sys
from collections import namedtuple
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
SRC_DADS, SRC_DAS, SRC_URBAN, SRC_ESC = 0, 1, 2, 3
OFFSET = 10 ** 8                 # пространство групп на источник
LABEL_OFFSET = 10 ** 7           # и на метку внутри источника
BLOCK = 250                      # блок номеров DADS, как в prep_dads.py

REPOS = {
    SRC_DADS:  ("geronimobasso/drone-audio-detection-samples", "data/*.parquet"),
    SRC_DAS:   ("ahlab-drone-project/DroneAudioSet",           "drone-only/*.parquet"),
    SRC_URBAN: ("danavery/urbansound8K",                       "data/*.parquet"),
    SRC_ESC:   ("ashraq/esc50",                                "data/*.parquet"),
}

NAMES = {SRC_DADS: "DADS", SRC_DAS: "DroneAudioSet",
         SRC_URBAN: "UrbanSound8K", SRC_ESC: "ESC-50"}

# Сколько шардов ожидаем. Расхождение означает, что датасет на HF
# перезалили — тогда разведанные схемы могли устареть, и молча продолжать
# нельзя. Проверяется в ноутбуке до обработки 18 ГБ.
N_SHARDS = {SRC_DADS: 39, SRC_DAS: 28, SRC_URBAN: 16, SRC_ESC: 2}

Rec = namedtuple("Rec", "audio group label cat src")

_NUM = re.compile(r"(\d+)")
# drone-only-recordings/drone1-only/mic-dist-25cm/throttle-low/mic1_soundskrit-File3.wav
_DAS = re.compile(r"/(drone\d+)-only/.*?/[^/]*?-?(File\d+|silence)\.wav$")


def _gid(src, label, key):
    """Ключ группы: источник, метка, номер записи.

    Метка входит в ключ обязательно. В DADS нумерация своя у каждого класса —
    drone-100.wav и no-drone-100.wav это разные записи, — и без метки они
    склеиваются в одну группу. Дальше assign_split раскладывает страты дрона и
    фона независимо, одна и та же группа уезжает в train и в val, и утечка
    готова. Так и было: «группа 0 разорвана».
    """
    key = int(key)
    assert 0 <= key < LABEL_OFFSET, f"номер записи {key} не влезает в ключ"
    return src * OFFSET + int(label) * LABEL_OFFSET + key


def dads_group(idx, label):
    return _gid(SRC_DADS, label, idx // BLOCK)


def urban_group(fs_id):
    return _gid(SRC_URBAN, 0, fs_id)


def esc_group(src_file):
    return _gid(SRC_ESC, 0, src_file)


def das_group(file_path):
    """Ключ = (дрон, номер файла), возвращает (ключ, тишина ли).

    Микрофон и дистанция намеренно отброшены: mic1_soundskrit-File3 и
    mic2_8array-down-File3 — одна запись с двух микрофонов. В разных сплитах
    они дадут ту самую утечку, от которой блоки по 250 защищают DADS.

    Записи *-silence.wav — тот же тракт без дрона. Это негативы, и группа у
    них отдельная от полётов того же дрона.
    """
    m = _DAS.search(file_path.replace("\\", "/"))
    if not m:
        raise ValueError(f"не разобран путь DroneAudioSet: {file_path}")
    drone, tail = m.group(1), m.group(2)
    silence = tail == "silence"
    dnum = int(_NUM.search(drone).group(1))
    fnum = 0 if silence else int(_NUM.search(tail).group(1))
    return _gid(SRC_DAS, 0 if silence else 1, dnum * 1000 + fnum), silence


def to_mono_16k(data, sr):
    """Моно, 16 кГц, пик нормирован в 1.0. None, если в окне тишина."""
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != SR:
        g = gcd(int(sr), SR)
        data = resample_poly(data, SR // g, sr // g).astype(np.float32)
    peak = np.abs(data).max() if data.size else 0.0
    if not np.isfinite(peak) or peak < 1e-6:
        return None
    return data / peak


def shards(src):
    from huggingface_hub import HfFileSystem
    import hub
    repo, pat = REPOS[src]
    fs = HfFileSystem(token=hub.token())
    return sorted(fs.glob(f"datasets/{repo}/{pat}"))


def _open(path):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    import hub
    fs = HfFileSystem(token=hub.token())
    return pq.ParquetFile(fs.open(path, "rb"))


def _decode(blob):
    try:
        data, sr = sf.read(io.BytesIO(blob), dtype="float32")
    except Exception:
        return None
    return to_mono_16k(data, sr)


def read_shard(src, path):
    """Генератор Rec из одного шарда. Читает построчно по row-group, чтобы
    не держать весь шард в памяти: у UrbanSound8K это 440 МБ."""
    pf = _open(path)
    if src == SRC_DAS:
        yield from _read_das(pf)
        return
    cols = {SRC_DADS:  ["audio", "label"],
            SRC_URBAN: ["audio", "fsID", "class"],
            SRC_ESC:   ["audio", "src_file", "category"]}[src]
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=cols).to_pylist():
            x = _decode(r["audio"]["bytes"])
            if x is None:
                continue
            if src == SRC_DADS:
                m = _NUM.search(r["audio"]["path"] or "")
                idx = int(m.group(1)) if m else 0
                lab = int(r["label"])
                yield Rec(x, dads_group(idx, lab), lab, None, src)
            elif src == SRC_URBAN:
                yield Rec(x, urban_group(r["fsID"]), 0, r["class"], src)
            else:
                yield Rec(x, esc_group(r["src_file"]), 0, r["category"], src)


def das_channel0(list_array, i):
    """Нулевой канал записи i из колонки audio.array.

    Форма — [отсчёт][канал], отсчёты во внешнем измерении: mic1_soundskrit даёт
    1 канал, оба 8array — 8. Разбирать надо срезом, а не через ListScalar:
    у скаляра .values отдаёт всю дочернюю колонку row-group целиком, а не срез
    своей записи, и получается каша из шести записей.

    Берём именно нулевой канал, а не среднее по восьми. Среднее по решётке
    микрофонов — это ненамеренная пространственная фильтрация: приходящее не
    по нормали частично гасится, и высокие частоты режутся тем сильнее, чем
    больше угол. В эксплуатации микрофон один, и обучать надо на том, что он
    услышит. Дублировать восемь коррелированных каналов как отдельные примеры
    тоже смысла нет — это один и тот же пролёт с разницей в сантиметры.
    """
    outer = list_array.slice(i, 1).flatten()          # внутренние списки этой записи
    if len(outer) == 0:
        return None
    lens = np.diff(outer.offsets.to_numpy())
    n_ch = int(lens[0])
    if n_ch < 1 or lens.min() != lens.max():
        return None                                   # рваная запись, пропускаем
    flat = outer.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(-1, n_ch)[:, 0]


def _read_das(pf):
    """audio.array — вложенные списки float64, а не закодированные байты.

    to_pylist() раздул бы 19 млн значений самой длинной записи в питоновские
    float по 28 байт (полгигабайта на запись) и убил бы сессию на T4. Достаём
    плоский буфер Arrow напрямую.
    """
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        paths = t.column("file_path").to_pylist()
        audio = t.column("audio").combine_chunks()
        arrays, rates = audio.field("array"), audio.field("sampling_rate").to_pylist()
        for i, fp in enumerate(paths):
            ch0 = das_channel0(arrays, i)
            if ch0 is None:
                continue
            x = to_mono_16k(ch0, int(rates[i]))
            if x is None:
                continue
            g, silence = das_group(fp)
            yield Rec(x, g, 0 if silence else 1,
                      "drone_rig_silence" if silence else None, SRC_DAS)


def selfcheck():
    """В сеть не ходит. Сетевой дымовой прогон запускается отдельно."""
    base = "drone-only-recordings/{d}-only/mic-dist-{c}cm/throttle-{t}/{m}-File{n}.wav"

    # группа = (дрон, номер файла); микрофон и дистанция не влияют
    a, a_sil = das_group(base.format(d="drone1", c=25, t="low", m="mic1_soundskrit", n=3))
    b, _ = das_group(base.format(d="drone1", c=50, t="high", m="mic2_8array-down", n=3))
    assert a == b, "один полёт с разных микрофонов должен дать одну группу"
    c, _ = das_group(base.format(d="drone2", c=25, t="low", m="mic1_soundskrit", n=3))
    assert a != c, "разные дроны — разные группы"
    d, _ = das_group(base.format(d="drone1", c=25, t="low", m="mic1_soundskrit", n=4))
    assert a != d, "разные полёты — разные группы"
    assert a_sil is False

    # тишина распознаётся, не смешивается с полётом и не идёт в позитивы
    sil, is_sil = das_group(
        "drone-only-recordings/drone1-only/mic-dist-25cm/throttle-low/mic1_soundskrit-silence.wav")
    assert is_sil is True and sil != a
    sil2, _ = das_group(
        "drone-only-recordings/drone1-only/mic-dist-50cm/throttle-high/mic2_8array-down-silence.wav")
    assert sil == sil2, "тишина одного дрона — одна группа"
    sil3, _ = das_group(
        "drone-only-recordings/drone2-only/mic-dist-25cm/throttle-low/mic1_soundskrit-silence.wav")
    assert sil != sil3, "тишина разных дронов — разные группы"

    # пространства групп источников не пересекаются
    assert a // OFFSET == SRC_DAS
    assert dads_group(20941, 1) // OFFSET == SRC_DADS
    assert urban_group(100032) // OFFSET == SRC_URBAN
    assert esc_group(7973) // OFFSET == SRC_ESC
    # блок 250: соседние номера в одной группе, далёкие — в разных
    assert dads_group(20941, 1) == dads_group(20999, 1) != dads_group(21999, 1)

    # ГЛАВНОЕ про DADS: нумерация своя у каждого класса, поэтому drone-100.wav
    # и no-drone-100.wav обязаны дать РАЗНЫЕ группы. Без метки в ключе они
    # склеивались, и assign_split разрывал такую группу между train и val.
    assert dads_group(100, 1) != dads_group(100, 0), \
        "метка должна входить в ключ группы DADS"
    assert dads_group(0, 1) != dads_group(0, 0)
    # и это не должно ломать разделение источников
    assert dads_group(100, 0) // OFFSET == dads_group(100, 1) // OFFSET == SRC_DADS

    # тишина и полёты DroneAudioSet тоже в разных пространствах меток
    flight, _ = das_group(base.format(d="drone1", c=25, t="low", m="mic1_soundskrit", n=3))
    sil_g, _ = das_group(
        "drone-only-recordings/drone1-only/mic-dist-25cm/throttle-low/mic1_soundskrit-silence.wav")
    assert (flight // LABEL_OFFSET) % 10 != (sil_g // LABEL_OFFSET) % 10

    # номер записи, не влезающий в ключ, должен падать, а не молча переполняться
    try:
        _gid(SRC_ESC, 0, LABEL_OFFSET)
    except AssertionError:
        pass
    else:
        raise AssertionError("переполнение ключа должно быть замечено")

    # битый путь не проглатывается молча
    try:
        das_group("совсем/не/тот/путь.wav")
    except ValueError:
        pass
    else:
        raise AssertionError("неразобранный путь должен падать, а не давать группу")

    # конвертация
    x = to_mono_16k(np.random.randn(44100, 2).astype(np.float32), 44100)
    assert abs(len(x) - SR) <= 2 and abs(x).max() <= 1.0 + 1e-6
    assert to_mono_16k(np.zeros(1000, np.float32), SR) is None       # тишина отбрасывается
    y = to_mono_16k(np.random.randn(SR).astype(np.float32), SR)
    assert len(y) == SR                                              # без ресемплинга длина цела

    # многоканальный float64 из DroneAudioSet сводится в моно
    z = to_mono_16k(np.random.randn(1000, 8).astype(np.float64), SR)
    assert z.dtype == np.float32 and z.ndim == 1 and len(z) == 1000

    two = np.stack([np.ones(100), -np.ones(100)], axis=1)
    assert to_mono_16k(two, SR) is None, "противофаза схлопывается в тишину"

    # разбор audio.array DroneAudioSet: [отсчёт][канал], берём нулевой канал.
    # Собираем две записи с разным числом каналов в одной колонке — ровно как
    # в настоящем шарде, где soundskrit даёт 1 канал, а 8array восемь.
    import pyarrow as pa
    rec0 = [[10.0 + c for c in range(8)] for _ in range(5)]      # 5 отсчётов x 8 каналов
    rec1 = [[20.0] for _ in range(3)]                            # 3 отсчёта x 1 канал
    col = pa.array([rec0, rec1], type=pa.list_(pa.list_(pa.float64())))

    c0 = das_channel0(col, 0)
    assert c0.shape == (5,), c0.shape
    assert (c0 == 10.0).all(), f"должен быть нулевой канал, а не среднее: {c0}"
    c1 = das_channel0(col, 1)
    assert c1.shape == (3,) and (c1 == 20.0).all()

    # срез не должен захватывать соседнюю запись: 5 отсчётов, не 8
    assert len(das_channel0(col, 0)) == 5, "ListScalar.values отдаёт всю колонку — нужен slice"

    # пустая и рваная записи пропускаются, а не падают
    empty = pa.array([[]], type=pa.list_(pa.list_(pa.float64())))
    assert das_channel0(empty, 0) is None
    ragged = pa.array([[[1.0, 2.0], [3.0]]], type=pa.list_(pa.list_(pa.float64())))
    assert das_channel0(ragged, 0) is None

    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        for k, (repo, pat) in REPOS.items():
            print(f"{NAMES[k]:16s} {repo}  ({N_SHARDS[k]} шардов)")
