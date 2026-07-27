# Этап 1: манифест + clip store — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** заменить кэш нарезанных 0.5-секундных окон (`cache_dads/`, `cache_hard/`) манифестом первого класса и хранилищем целых клипов, чтобы (а) баги вроде перевёрнутого `CAP` или потери метки в ключе группы были видны запросом, а не находились постфактум на полном прогоне, и (б) смена контекста обучения (0.5 → 4.0 с) не требовала пересборки кэша.

**Architecture:** `hf_sources.read_shard()` уже отдаёт клипы целиком (`Rec.audio` — не нарезан на окна; нарезка происходит только в `prep_hf.windows()`, ниже по потоку). Значит манифест строится прямо из `Rec`, без промежуточной реконструкции: один проход по шарду — одна запись в манифесте плюс один клип в хранилище. Сплит назначается один раз при сборке и хранится колонкой; `train.py` в будущем читает `WHERE split='train'`, а не пересчитывает разбиение.

**Tech Stack:** Python 3.10, numpy, pyarrow (parquet), уже используемые `hf_sources.py`/`hub.py`. Новых зависимостей не вводится.

## Global Constraints

- `SR = 16000` Гц, моно, float32 в клипах (как отдаёт `to_mono_16k`), пик нормирован в 1.0 на уровне источника — хранилище это не трогает.
- Конвенция проекта: каждый модуль со своей логикой имеет `selfcheck()`, работающий **без сети и без данных**, запускаемый как `python -m airadar.data.<модуль> --selfcheck`. Сборщики без собственной логики (оркестрация в `cli/`) `selfcheck` не имеют.
- Комментарии объясняют «почему», а не «что». Язык — русский.
- Целевой размер модуля — до ~200 строк.
- `hub.py`/`hf_sources.py` не трогаются: сетевой доступ и токен-логика уже написаны и протестированы (в том числе `Task 9` этапа 0 против них уже сверялась). Этот план их только вызывает.
- Формат `manifest.parquet` и `clips.bin`/`clips.idx` версионируется (`MANIFEST_VERSION`), как `PREP_VERSION` в `prep_hf.py` — правка схемы обязана поднимать версию, а не молча смешивать несовместимые записи.
- Полный прогон сборки по всем ~85 шардам источников — сетевой, IO-связанный процесс, повторяющий по масштабу `prep_hf.collect()` (часы, не минуты). Он **не** часть автоматического исполнения этого плана — см. Task 6 и раздел «Что этот план не делает».

## Ссылки

- Спецификация: [docs/superpowers/specs/2026-07-26-architecture-redesign-design.md](../specs/2026-07-26-architecture-redesign-design.md), §5
- Существующий код для переиспользования: `hf_sources.py` (`Rec`, `read_shard`, `shards`, `local_shard`, `to_mono_16k`), `prep_hf.py` (`windows`, `assign_split`, `coverage_report`, `CAP`, `HARD`, `_key`, `_todo`, `CHECKPOINT_EVERY`-паттерн чекпоинта)
- Находка D0 (этап 0): соседние индексы DADS **не смежны** — клипы DADS остаются отдельными записями по ~0.6 с, реконструкция в длинные дорожки невозможна. См. `docs/superpowers/specs/2026-07-26-architecture-redesign-design.md` §5.4.
- Фактические ключи `cache_dads/meta.npz` (сверено на диске): `y (int8), group (int64), src (int8), split (int8), synth (bool), n, win`.

## Структура файлов

| файл | ответственность |
|---|---|
| `airadar/data/__init__.py` | пустой |
| `airadar/data/manifest.py` | схема строки манифеста, чистые функции построения/проверки |
| `airadar/data/clips.py` | хранилище клипов: запись, чтение по индексу, память |
| `airadar/data/build.py` | адаптер шард → (клипы в хранилище, строки манифеста), чекпоинт |
| `airadar/data/split.py` | назначение сплита на манифест (адаптация `assign_split`) |
| `airadar/bench/manifest_audit.py` | санити-проверки манифеста (дубли групп между сплитами, покрытие категорий, доля меток) |
| `cli/build_manifest.py` | оркестрация полной сборки по всем источникам |
| `cli/manifest_audit.py` | тонкая обёртка над `manifest_audit.py` |

---

### Task 1: схема манифеста и чистые функции построения

**Files:**
- Create: `airadar/data/__init__.py` (пустой), `airadar/data/manifest.py`

**Interfaces:**
- Consumes: ничего (работает с обычными Python-объектами, не с `hf_sources.Rec` напрямую — см. Task 3 про адаптер)
- Produces:
  - `MANIFEST_VERSION: int = 1`
  - `SCHEMA: dict[str, str]` — имя колонки → тип pyarrow-совместимой строки (`"int64"`, `"float32"`, `"bool"`, `"string"`), используется и для валидации, и для построения пустой таблицы
  - `make_row(clip_id: int, src: int, offset: int, n_samples: int, label: int, label_conf: float, group_id: int, domain: str, category: str | None, synth: bool = False) -> dict` — одна строка манифеста с прочими полями (`split`, `prep_version`, признаковые `f0_med`/`salience`/`lf_energy`) как `None`/`NaN`-заглушками, заполняемыми позже отдельными шагами
  - `validate_row(row: dict) -> None` — бросает `ValueError` с именем поля при несовпадении типа или пропуске обязательного поля
  - `rows_to_table(rows: list[dict]) -> "pyarrow.Table"` — сборка в готовую к записи таблицу с типами из `SCHEMA`

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Схема манифеста: один клип — одна строка.

Манифест — это единственный источник знаний о данных. Четыре бага одной
прошлой сессии (перевёрнутый CAP, потерянная метка в ключе группы, страта
по окну вместо группы, drone_rig_silence) были невидимы, потому что жили в
императивном коде и проявлялись только на полном прогоне. Здесь то же самое
— колонка таблицы, и неверное значение видно запросом `SELECT`, а не
всплывает через час обучения на насыщенной метрике.
"""

import sys

MANIFEST_VERSION = 1

SCHEMA = {
    "clip_id": "int64", "src": "int8", "offset": "int64", "n_samples": "int64",
    "label": "int8", "label_conf": "float32", "group_id": "int64",
    "domain": "string", "category": "string", "synth": "bool",
    "f0_med": "float32", "salience": "float32", "lf_energy": "float32",
    "split": "int8", "prep_version": "int64",
}


def selfcheck():
    row = make_row(clip_id=1, src=0, offset=0, n_samples=9600,
                   label=1, label_conf=1.0, group_id=42, domain="rig1",
                   category=None)
    assert set(row) == set(SCHEMA), set(SCHEMA) - set(row)
    assert row["clip_id"] == 1 and row["n_samples"] == 9600
    assert row["synth"] is False               # значение по умолчанию
    assert row["split"] is None                 # заполняется позже (Task 4)
    assert row["prep_version"] == MANIFEST_VERSION

    validate_row(row)                            # не должно бросать

    bad = dict(row)
    bad["clip_id"] = "не число"
    try:
        validate_row(bad)
    except ValueError as e:
        assert "clip_id" in str(e)
    else:
        raise AssertionError("validate_row должен ловить неверный тип")

    missing = dict(row)
    del missing["group_id"]
    try:
        validate_row(missing)
    except ValueError as e:
        assert "group_id" in str(e)
    else:
        raise AssertionError("validate_row должен ловить пропущенное поле")

    t = rows_to_table([row, dict(row, clip_id=2)])
    assert t.num_rows == 2
    assert t.column("clip_id").to_pylist() == [1, 2]
    assert str(t.schema.field("clip_id").type) == "int64"
    assert str(t.schema.field("category").type) == "string"

    print("manifest selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.data.manifest --selfcheck`
Expected: FAIL, `NameError: name 'make_row' is not defined`

- [ ] **Step 3: Реализовать**

```python
def make_row(clip_id, src, offset, n_samples, label, label_conf, group_id,
            domain, category, synth=False):
    return {
        "clip_id": int(clip_id), "src": int(src), "offset": int(offset),
        "n_samples": int(n_samples), "label": int(label),
        "label_conf": float(label_conf), "group_id": int(group_id),
        "domain": str(domain), "category": category, "synth": bool(synth),
        "f0_med": None, "salience": None, "lf_energy": None,
        "split": None, "prep_version": MANIFEST_VERSION,
    }


_PY_TYPES = {
    "int64": int, "int8": int, "float32": (float, type(None)),
    "bool": bool, "string": (str, type(None)),
}


def validate_row(row):
    for name, kind in SCHEMA.items():
        if name not in row:
            raise ValueError(f"в строке манифеста нет поля {name!r}")
        val = row[name]
        want = _PY_TYPES[kind]
        if val is None and kind in ("int64", "int8"):
            raise ValueError(f"{name!r}: обязательное целочисленное поле пусто")
        if val is not None and not isinstance(val, want):
            raise ValueError(f"{name!r}: ожидался {want}, получено {type(val)}")


def rows_to_table(rows):
    import pyarrow as pa
    cols = {name: [r[name] for r in rows] for name in SCHEMA}
    types = {"int64": pa.int64(), "int8": pa.int8(), "float32": pa.float32(),
             "bool": pa.bool_(), "string": pa.string()}
    fields = [pa.field(name, types[kind]) for name, kind in SCHEMA.items()]
    arrays = [pa.array(cols[name], type=types[kind])
             for name, kind in SCHEMA.items()]
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.data.manifest --selfcheck`
Expected: PASS, `manifest selfcheck ok`

- [ ] **Step 5: Коммит**

```bash
git add airadar/data/__init__.py airadar/data/manifest.py
git commit -m "data: схема манифеста, одна строка на клип

Манифест — единственный источник знаний о данных. Прошлые баги (CAP
перевёрнут, метка потеряна в ключе группы) были невидимы в императивном
коде; здесь неверное значение видно запросом, а не всплывает через час
обучения на насыщенной метрике."
```

---

### Task 2: хранилище клипов

**Files:**
- Create: `airadar/data/clips.py`

**Interfaces:**
- Consumes: ничего
- Produces:
  - `class ClipWriter` — `__init__(self, bin_path: str)`, `write(self, audio: np.ndarray) -> tuple[int, int]` (возвращает `(offset, n_samples)` в отсчётах float32), `close(self)`
  - `class ClipReader` — `__init__(self, bin_path: str)`, `read(self, offset: int, n_samples: int) -> np.ndarray` (float32, память через `np.memmap`, копия не делается до среза)
  - Формат: `clips.bin` — конкатенация float32 клипов подряд, без разделителей; смещение и длина хранятся в манифесте (`offset`, `n_samples`), отдельного `.idx` файла не заводится — манифест это и есть индекс, дублировать незачем

- [ ] **Step 1: Написать падающий selfcheck**

```python
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


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.data.clips --selfcheck`
Expected: FAIL, `NameError: name 'ClipWriter' is not defined`

- [ ] **Step 3: Реализовать**

```python
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
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.data.clips --selfcheck`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add airadar/data/clips.py
git commit -m "data: хранилище клипов — конкатенация float32, манифест это индекс

Кэш окон дублировал 50% данных на перекрытии. Здесь клип пишется один раз
целиком; смещение и длина живут в манифесте, отдельного .idx нет — второй
источник истины не нужен."
```

---

### Task 3: адаптер шард → клипы + строки манифеста

**Files:**
- Create: `airadar/data/build.py`

**Interfaces:**
- Consumes: `hf_sources.Rec`, `hf_sources.read_shard`, `hf_sources.shards`, `hf_sources.local_shard`, `hf_sources.NAMES`; `airadar.data.manifest.make_row`; `airadar.data.clips.ClipWriter`
- Produces:
  - `rec_to_row(rec, clip_id: int, offset: int, n_samples: int) -> dict` — чистая функция, домен вычисляется из `rec.src`/`rec.group` (см. Step 3)
  - `ingest_shard(rec_iter, writer: ClipWriter, next_clip_id: int) -> tuple[list[dict], int]` — потребляет генератор `Rec`, пишет каждый клип в `writer`, возвращает `(строки_манифеста, следующий_свободный_clip_id)`. Чистая функция относительно сети — сеть уже отработала в `rec_iter`.
  - `_key(src: int, i: int) -> str` — переиспользует формат `prep_hf._key`, тот же паттерн чекпоинта по шардам

**Замечание для реализатора:** `hf_sources.Rec.audio` — уже готовый float32-клип (не окно). `rec.cat is not None` означает «негатив с категорией» (см. `prep_hf._dest`) — из этого берётся `category`; для позитивов `category=None`. `domain` — приближение источника записи: для DAS это дрон-риг (`rec.group // 1000` восстанавливает `dnum` по формуле из `hf_sources.das_group`, но проще и надёжнее хранить `domain=f"das_rig_{rec.group}"` — весь `group` уже уникально определяет полёт); для DADS домен неизвестен по находке D0 (клипы не смежны) — `domain=f"dads_block_{rec.group}"`; для URBAN/ESC — `domain=f"scene_{rec.group}"`. Синтетика (`synth`) всегда `False` — генератор Griffin-Lim был только в старом Kaggle-наборе, в HF-источниках его нет (проверено: `prep_hf.py` всегда пишет `synth=np.zeros(...)`).

- [ ] **Step 1: Написать падающий selfcheck**

```python
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
    ]
    with tempfile.TemporaryDirectory() as d:
        w = ClipWriter(os.path.join(d, "clips.bin"))
        rows, next_id = ingest_shard(iter(recs), w, next_clip_id=100)
        w.close()

        assert next_id == 102, next_id
        assert len(rows) == 2
        assert rows[0]["clip_id"] == 100 and rows[1]["clip_id"] == 101
        assert rows[0]["offset"] == 0 and rows[0]["n_samples"] == 100
        assert rows[1]["offset"] == 100 and rows[1]["n_samples"] == 50   # встык
        assert rows[0]["label"] == 1 and rows[0]["category"] is None
        assert rows[1]["label"] == 0 and rows[1]["category"] == "wind"
        assert rows[0]["domain"] == "dads_block_5", rows[0]["domain"]
        assert rows[1]["domain"] == "scene_7", rows[1]["domain"]
        assert rows[0]["synth"] is False

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


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.data.build --selfcheck`
Expected: FAIL, `NameError: name 'ingest_shard' is not defined`

- [ ] **Step 3: Реализовать**

```python
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
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.data.build --selfcheck`
Expected: PASS

- [ ] **Step 5: Проверить на одном настоящем шарде (сеть, но дёшево)**

Небольшой смоук-тест — не часть `selfcheck` (тот обязан работать без сети),
отдельная ручная проверка перед тем, как доверять адаптеру полный прогон:

```bash
python -c "
import sys; sys.path.insert(0, '.')
import hf_sources as S
from airadar.data.build import ingest_shard
from airadar.data.clips import ClipWriter
import tempfile, os

with tempfile.TemporaryDirectory() as d:
    rel = S.shards(S.SRC_ESC)[0]           # ESC-50 — самый маленький источник
    path = S.local_shard(S.SRC_ESC, rel)
    w = ClipWriter(os.path.join(d, 'clips.bin'))
    rows, next_id = ingest_shard(S.read_shard(S.SRC_ESC, path), w, 0)
    w.close()
    print('строк манифеста:', len(rows), 'next_id:', next_id)
    print('пример строки:', rows[0])
    assert all(r['n_samples'] > 0 for r in rows)
    assert len({r['clip_id'] for r in rows}) == len(rows), 'дублирующиеся clip_id'
"
```

Expected: несколько сотен строк (ESC-50, 1 из 2 шардов), `clip_id` без
повторов, `n_samples` у всех положительный.

- [ ] **Step 6: Коммит**

```bash
git add airadar/data/build.py
git commit -m "data: адаптер шард -> (клипы, строки манифеста)

hf_sources.read_shard уже отдаёт клипы целиком — реконструкции не нужно.
Для DADS домен помечен как dads_block_N: находка D0 (этап 0) показала, что
соседние индексы физически не смежны, склеивать их в манифесте было бы
неверно."
```

---

### Task 4: назначение сплита на манифест

**Files:**
- Create: `airadar/data/split.py`

**Interfaces:**
- Consumes: `pyarrow.Table` из `manifest.rows_to_table`
- Produces:
  - `TRAIN, VAL, TEST = 0, 1, 2`
  - `assign_split(group_id: np.ndarray, strata: np.ndarray, frac: tuple = (0.75, 0.15, 0.10), seed: int = 0) -> np.ndarray` — тонкая обёртка над `prep_hf.assign_split` (импортируется, не копируется — эта функция уже проверена в проекте на реальных данных, дублировать её было бы риском разъехаться)
  - `apply_split(table: "pyarrow.Table") -> "pyarrow.Table"` — берёт `group_id` и составную страту `(src, label, category)` из таблицы, вызывает `assign_split`, возвращает таблицу с заполненной колонкой `split`

**Замечание для реализатора:** `prep_hf.assign_split` уже существует, протестирован и решает ровно эту задачу (группы разного размера, страта — свойство группы, а не окна). Здесь не переписывается — импортируется как `from prep_hf import assign_split as _assign_split`. Единственная новая часть — сборка составной страты и запись результата в колонку `pyarrow.Table`, а не в отдельный массив.

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Назначение сплита один раз, на уровне манифеста.

train.py ранее сам пересчитывал разбиение при каждом обучении — это и был
источник несравнимости прогонов. Здесь сплит — колонка манифеста, вычисляемая
один раз при сборке; дальше это WHERE split='train', а не логика.

Сама раскладка групп по сплитам не переписана: prep_hf.assign_split уже
решает эту задачу (группы разного размера, страта — свойство группы) и
проверена на реальных данных. Здесь только сборка страты и запись в колонку.
"""

import sys
import numpy as np

TRAIN, VAL, TEST = 0, 1, 2


def selfcheck():
    import pyarrow as pa
    t = pa.table({
        "group_id": [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        "src": [0] * 15, "label": [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "category": pa.array([None] * 15, type=pa.string()),
        "split": pa.array([None] * 15, type=pa.int8()),
    })
    out = apply_split(t)
    sp = out.column("split").to_pylist()
    assert all(s in (TRAIN, VAL, TEST) for s in sp), "все строки должны получить сплит"

    # группа физически не может оказаться в двух сплитах
    gid = out.column("group_id").to_pylist()
    for g in set(gid):
        vals = {sp[i] for i in range(len(gid)) if gid[i] == g}
        assert len(vals) == 1, f"группа {g} разорвана между сплитами: {vals}"

    # детерминированность по seed
    out2 = apply_split(t)
    assert out.column("split").to_pylist() == out2.column("split").to_pylist()

    print("split selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.data.split --selfcheck`
Expected: FAIL, `NameError: name 'apply_split' is not defined`

- [ ] **Step 3: Реализовать**

```python
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
from prep_hf import assign_split as _assign_split  # noqa: E402


def _strata(table):
    src = table.column("src").to_pylist()
    label = table.column("label").to_pylist()
    cat = table.column("category").to_pylist()
    return np.array([f"{s}_{l}_{c}" for s, l, c in zip(src, label, cat)], object)


def apply_split(table, frac=(0.75, 0.15, 0.10), seed=0):
    group_id = np.asarray(table.column("group_id").to_pylist())
    split = _assign_split(group_id, _strata(table), frac=frac, seed=seed)
    return table.set_column(
        table.schema.get_field_index("split"), "split",
        __import__("pyarrow").array(split, type=__import__("pyarrow").int8()))
```

Примечание: импорт `prep_hf` через явную вставку пути к корню репозитория —
`prep_hf.py` лежит в корне, не в пакете `airadar`, и обычный `import prep_hf`
не найдёт его при запуске `python -m airadar.data.split` из корня. Если
реализатор найдёт способ чище (например, `sys.path` уже настроен окружением
проекта) — использовать его, конструкция выше рабочая, но не единственно
верная.

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.data.split --selfcheck`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add airadar/data/split.py
git commit -m "data: сплит — колонка манифеста, назначается один раз

Переиспользует prep_hf.assign_split (уже проверен на реальных данных, не
дублируется). Новое — сборка составной страты (src, label, category) и
запись результата в колонку таблицы вместо пересчёта при каждом обучении."
```

---

### Task 5: аудит манифеста

**Files:**
- Create: `airadar/bench/manifest_audit.py`

**Interfaces:**
- Consumes: `pyarrow.Table` со схемой из `airadar.data.manifest.SCHEMA`
- Produces:
  - `check_group_not_split(table) -> list[int]` — список `group_id`, встретившихся более чем в одном значении `split` (в норме пусто)
  - `check_category_coverage(table, splits=(1, 2)) -> dict[str, list[int]]` — для каждой запрошенной части сплита список категорий, отсутствующих в ней целиком (перенос `prep_hf.coverage_report` на манифест)
  - `check_label_balance(table) -> dict[int, dict[int, int]]` — `{split: {label: count}}`
  - `audit(table) -> dict` — сводка всех трёх проверок плюс `"n_rows"`, `"n_groups"`, `"ok": bool` (`True`, если `check_group_not_split` пуст — единственная проверка, чей провал структурно невозможен, если только не сломан код, остальные две могут законно быть непустыми на малых стратах)

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Санити-проверки манифеста как запросы, а не как ручной разбор постфактум.

Три прошлых бага (CAP перевёрнут, метка потеряна в ключе группы, страта по
окну вместо группы) обнаруживались только на полном прогоне обучения, часы
спустя. check_group_not_split здесь — прямая защита от второго из них: если
он не пуст, манифест собран неверно, и это видно за секунды, не за час.
"""

import sys


def selfcheck():
    import pyarrow as pa

    good = pa.table({
        "group_id": [1, 1, 2, 2, 3], "split": [0, 0, 1, 1, 2],
        "label": [1, 1, 0, 0, 1],
        "category": pa.array(["wind", "wind", None, None, "rain"], type=pa.string()),
    })
    assert check_group_not_split(good) == []

    bad = pa.table({
        "group_id": [1, 1, 2], "split": [0, 1, 1],       # группа 1 разорвана
        "label": [1, 1, 0],
        "category": pa.array([None, None, None], type=pa.string()),
    })
    broken = check_group_not_split(bad)
    assert broken == [1], broken

    cov = check_category_coverage(good, splits=(1, 2))
    assert 1 in cov and "rain" in cov[1], cov          # rain нет в part=1 (val)
    assert 2 in cov and "wind" in cov[2], cov           # wind нет в part=2 (test)

    bal = check_label_balance(good)
    assert bal[0] == {1: 2}, bal
    assert bal[1] == {0: 2}, bal

    rep_good = audit(good)
    assert rep_good["ok"] is True
    assert rep_good["n_rows"] == 5 and rep_good["n_groups"] == 3

    rep_bad = audit(bad)
    assert rep_bad["ok"] is False

    print("manifest_audit selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.manifest_audit --selfcheck`
Expected: FAIL, `NameError: name 'check_group_not_split' is not defined`

- [ ] **Step 3: Реализовать**

```python
def check_group_not_split(table):
    gid = table.column("group_id").to_pylist()
    sp = table.column("split").to_pylist()
    by_group = {}
    for g, s in zip(gid, sp):
        by_group.setdefault(g, set()).add(s)
    return sorted(g for g, vals in by_group.items() if len(vals) > 1)


def check_category_coverage(table, splits=(1, 2)):
    sp = table.column("split").to_pylist()
    cat = table.column("category").to_pylist()
    all_cats = {c for c in cat if c is not None}
    out = {}
    for part in splits:
        present = {c for c, s in zip(cat, sp) if s == part and c is not None}
        missing = sorted(all_cats - present)
        if missing:
            out[part] = missing
    return out


def check_label_balance(table):
    sp = table.column("split").to_pylist()
    lab = table.column("label").to_pylist()
    out = {}
    for s, l in zip(sp, lab):
        out.setdefault(s, {}).setdefault(l, 0)
        out[s][l] += 1
    return out


def audit(table):
    broken = check_group_not_split(table)
    return {
        "n_rows": table.num_rows,
        "n_groups": len(set(table.column("group_id").to_pylist())),
        "groups_split_across_parts": broken,
        "missing_categories": check_category_coverage(table),
        "label_balance": check_label_balance(table),
        "ok": len(broken) == 0,
    }
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.manifest_audit --selfcheck`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add airadar/bench/manifest_audit.py
git commit -m "bench: аудит манифеста — запрос вместо ручного разбора постфактум

check_group_not_split — прямая защита от бага 'группа разорвана между
train/val', который в прошлой сессии проявлялся только через assertion в
ноутбуке после часа обучения."
```

---

### Task 6: оркестрация полной сборки (CLI, не автопрогон)

**Files:**
- Create: `cli/build_manifest.py`, `cli/manifest_audit.py`

**Interfaces:**
- Consumes: всё из Task 1–5, плюс `hf_sources.shards`, `hf_sources.local_shard`, `hf_sources.read_shard`, `hf_sources.NAMES`
- Produces: CLI `python cli/build_manifest.py --limit N` (собирает первые N шардов каждого источника — для проверки) и `python cli/build_manifest.py` (полный прогон); пишет `data/clips.bin` и `data/manifest.parquet`; CLI `python cli/manifest_audit.py` — печатает `audit()` по собранному манифесту

**Важно:** этот таск производит оркестрирующий код и проверяет его на
`--limit 1` (по одному шарду от каждого источника — минуты, не часы).
**Полный прогон без `--limit` не запускается как часть этого плана** — это
IO-связанный процесс на весь объём источников (~85 шардов, тот же порядок
величины, что `prep_hf.collect()`, который в прошлой сессии шёл далеко не
секунды). Решение о запуске полной сборки — отдельное, осознанное действие
после того, как `--limit 1` подтвердит корректность на малом объёме.

- [ ] **Step 1: Написать `cli/build_manifest.py`**

```python
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
    table = pq.concat_tables([pq.read_table(p) for p in parts])
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
```

- [ ] **Step 2: Написать `cli/manifest_audit.py`**

```python
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
```

- [ ] **Step 3: Прогнать на малом объёме — по одному шарду источника**

Run:
```bash
python cli/build_manifest.py --limit 1 2>&1 | tee logs/build_manifest_smoke.log
python cli/manifest_audit.py
```

Expected: `build_manifest.py` печатает по строке на источник (4 строки —
ESC, URBAN, DAS, DADS, по одному шарду каждого), пишет `data/manifest.parquet`
и `data/clips.bin`, оба ненулевого размера. `manifest_audit.py` печатает JSON
с `"ok": true` (на четырёх шардах группы не могут физически разорваться
между сплитами — Task 4 это гарантирует по построению) и завершается кодом 0.

Если `ok: false` — не подгонять данные под ожидание, а разбираться: это
ровно тот случай, ради которого аудит заведён.

- [ ] **Step 4: Проверить, что повторный запуск не пересчитывает готовое**

Run:
```bash
python cli/build_manifest.py --limit 1 2>&1 | tee -a logs/build_manifest_smoke.log
```

Expected: не печатает ни одной строки `[источник i/n]` — все четыре шарда
уже в `done` из чекпоинта, `main` проходит все циклы без единого нового
`ingest_shard`. Это проверяет чекпоинт-логику без необходимости прерывать
процесс вручную.

- [ ] **Step 5: Убрать смоук-артефакты перед коммитом кода**

Малые тестовые `data/manifest.parquet`, `data/clips.bin`, `data/parts/`,
`data/build_checkpoint.json` от `--limit 1` — не то же самое, что итог
полной сборки; коммитить их как реальный манифест было бы обманом. Удалить
перед коммитом:

```bash
rm -rf data/manifest.parquet data/clips.bin data/parts data/build_checkpoint.json
```

- [ ] **Step 6: Коммит**

```bash
git add cli/build_manifest.py cli/manifest_audit.py logs/build_manifest_smoke.log
git commit -m "cli: оркестрация сборки манифеста, чекпоинт по шардам

Проверено на --limit 1 (по одному шарду каждого источника) и на повторном
запуске поверх готового чекпоинта. Полный прогон без --limit — отдельное,
осознанное действие: это IO-связанный процесс на весь объём источников, не
часть автоматического исполнения этого плана."
```

---

## Проверка перед закрытием этапа

- [ ] `python -m airadar.data.manifest --selfcheck` — PASS
- [ ] `python -m airadar.data.clips --selfcheck` — PASS
- [ ] `python -m airadar.data.build --selfcheck` — PASS
- [ ] `python -m airadar.data.split --selfcheck` — PASS
- [ ] `python -m airadar.bench.manifest_audit --selfcheck` — PASS
- [ ] `cli/selfcheck.py` подхватывает пять новых модулей (проверить, что `EXEMPT`/`MIN_CHECKS` в `cli/selfcheck.py` не блокируют новые пакеты — при необходимости поднять `MIN_CHECKS`)
- [ ] `python cli/build_manifest.py --limit 1` + `python cli/manifest_audit.py` — оба зелёные на реальных данных
- [ ] смоук-артефакты `data/*` от `--limit 1` не закоммичены

## Что этот план сознательно не делает

- **Не запускает полную сборку по всем ~85 шардам.** Это отдельное решение
  после проверки на `--limit 1` — см. Task 6.
- **Не считает признаковые колонки манифеста** (`f0_med`, `salience`,
  `lf_energy` из спецификации §5.2) — они остаются `None` после Task 1–6.
  Заполнение — отдельный проход поверх готового `clips.bin`
  (`evalx/f0_survey.py`-подобная логика, адаптированная на весь корпус, а не
  на выборку), имеет смысл только после того, как манифест на реальных
  данных подтверждён аудитом.
- **Не строит псевдо-источники кластеризацией** (спецификация §5.5,
  20–40 кластеров по логмел-профилю и f0, домен-холдаут по кластеру) — это
  отдельный, более исследовательский шаг поверх готового манифеста и
  признаковых колонок, не инфраструктура.
- **Не переносит `train.py`/`eval.py`/`detect.py` на манифест.** Они
  продолжают читать `cache_dads/`/`cache_hard/` до отдельного шага миграции
  — переключать обучение раньше, чем манифест пройдёт аудит на полном
  объёме данных, было бы преждевременно.
