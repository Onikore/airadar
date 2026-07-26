# Расширенный датасет и обучение в Colab — план реализации

> **Для агентов:** ОБЯЗАТЕЛЬНЫЙ СУБ-НАВЫК: `superpowers:subagent-driven-development`
> (рекомендуется) либо `superpowers:executing-plans`. Шаги отмечены чекбоксами
> (`- [ ]`) для отслеживания.

**Цель:** собрать кэш окон из четырёх источников HF с замороженным сплитом
75/15/10 по группам и перенести обучение в Colab так, чтобы логи и чекпоинты
были доступны для разбора через HF-коннектор.

**Архитектура:** приватный dataset-репозиторий `Onikore/airadar-hub` работает
общим диском между Colab и рабочей машиной. `prep_hf.py` один раз собирает кэш
и кладёт его туда; `notebooks/02_train.ipynb` тянет кэш за минуты и пушит
`train.log` и `last.pt` после каждой эпохи.

**Стек:** Python 3.10, numpy, scipy, soundfile, pyarrow, huggingface_hub,
scikit-learn, torch (CPU локально, CUDA в Colab).

Спецификация: [2026-07-26-colab-dataset-design.md](../specs/2026-07-26-colab-dataset-design.md)

## Глобальные ограничения

- **Тестов на pytest в проекте нет.** Конвенция — функция `selfcheck()` в
  каждом модуле, вызываемая через `python <файл>.py --selfcheck`. Новый код
  следует ей же. Не вводить pytest.
- **Комментарии и вывод — по-русски**, как во всём проекте. Комментарий
  объясняет «почему», а не «что»: см. `train.py:26-31`, `prep_dads.py:1-13`.
- **Секреты в файлы не попадают.** Токен HF читается из переменной окружения
  `HF_TOKEN` либо из Colab Secrets. Ни один коммит не содержит `hf_...`.
- **Целевой формат окна не меняется:** 16 кГц, моно, `WIN = 8000` отсчётов
  (0.5 с), `int16` в `windows.bin`. Так устроены `cache_dads` и `cache_hard`.
- **Существующие контракты сохраняются:** `load_cache()` возвращает
  `(X, y, group, synth)`, `load_hard()` возвращает `(X, cat, train_idx, val_idx)`.
  Новые данные отдаются новыми функциями, а не расширением кортежей.
- **Репозиторий HF:** `Onikore/airadar-hub`, тип `dataset`, приватный.
- Локальная машина без GPU: `DEV` в `train.py` сам падает на `"cpu"`.

---

## Структура файлов

| файл | ответственность |
|---|---|
| `hub.py` (новый) | только ввод-вывод к HF Hub: `push`, `pull`, `exists`, JSON-хелперы |
| `hf_sources.py` (новый) | только чтение источников: четыре генератора, отдающие единый `Rec` |
| `prep_hf.py` (новый) | только сборка кэша: нарезка окон, группировка, сплит, манифест |
| `train.py` (правка) | `load_split`, `load_hard_test`, `--resume`, лог в jsonl |
| `eval.py` (правка) | читает замороженный сплит вместо `GroupShuffleSplit` |
| `diag_leak.py` (правка) | то же — иначе диагностика утечки меряет не тот сплит |
| `notebooks/01_prep.ipynb` (новый) | оркестрация prep в Colab |
| `notebooks/02_train.ipynb` (новый) | оркестрация обучения в Colab |

Разделение `hf_sources.py` и `prep_hf.py` намеренное: у источников четыре
разных схемы parquet и четыре разных правила группировки, у сборщика — одна
логика нарезки и сплита. Слив их в один файл дал бы ~450 строк с двумя
несвязанными осями изменения.

---

## Разведанные факты об источниках

Проверено чтением футеров parquet 2026-07-26. **Не перепроверять, не угадывать.**

### `geronimobasso/drone-audio-detection-samples` (DADS)
39 шардов, 180 320 строк. Схема: `audio {bytes, path}`, `label` (0/1).
Аудио — закодированные байты wav, читаются `soundfile.read(io.BytesIO(...))`.
Уже 16 кГц моно. Обработка полностью повторяет существующий `prep_dads.py`.

### `ashraq/esc50` (ESC-50)
2 шарда, 2000 строк. Схема:
```
filename string | fold int64 | target int64 | category string
esc10 bool | src_file int64 | take string | audio struct<bytes, path>
```
`category` — готовая строка (`chainsaw`, `helicopter`, …), парсить имя файла
не нужно. `src_file` — идентификатор исходной записи Freesound, точный ключ
группировки: клипы из одной записи не должны расползтись по сплитам.

### `danavery/urbansound8K` (UrbanSound8K)
16 шардов, ~546 строк каждый, 8732 всего. Схема:
```
audio struct<bytes, path> | slice_file_name string | fsID int64
start double | end double | salience int64 | fold int64
classID int64 | class string
```
`class` — готовая строка. `fsID` — исходная запись, точный ключ группировки.

### `ahlab-drone-project/DroneAudioSet`, конфиг `drone-only`
28 шардов по 6 строк, 168 записей. Схема:
```
file_path string
audio struct<array: list<list<double>>, sampling_rate: int64, path: string>
data_type string
```

**Четыре особенности, каждая ломает наивную реализацию:**

1. **`audio.array` — не байты, а вложенные списки float64 формы
   `[отсчёт][канал]`** (отсчёты во внешнем измерении, не каналы).
   `mic1_soundskrit` даёт 1 канал, оба `8array` — 8. Всего 1 725,1 млн
   значений на 168 записей, до 19,4 млн в одной. Вызов `.to_pylist()` раздует
   это в питоновские float по 28 байт — полгигабайта на запись и падение по
   памяти на T4.

   Разбирать **срезом**, не через `ListScalar`: у скаляра `.values` отдаёт
   всю дочернюю колонку row-group целиком, а не срез своей записи, и
   получается каша из шести записей (проверено, ошибка тихая — падает
   позже на reshape). Верный путь:
   `list_array.slice(i, 1).flatten().flatten().to_numpy(zero_copy_only=False)`.

   Берётся **нулевой канал**, а не среднее по восьми: среднее по решётке
   микрофонов — ненамеренная пространственная фильтрация, гасящая приходящее
   не по нормали, а в эксплуатации микрофон один.
2. **`sampling_rate` = 16000** — ресемплинг не нужен, в отличие от остальных.
3. **24 записи из 168 (14%) — тишина**, имя оканчивается на `-silence.wav`.
   Это записи того же тракта без дрона. Пометить их `label=1` значит учить
   сеть, что тишина — дрон. Идут в негативы, причём это негативы высшего
   сорта: тот же микрофон, та же комната, тот же тракт.
4. **Один полёт записан несколькими микрофонами.** Путь устроен как
   `drone-only-recordings/{drone}/mic-dist-{25,50}cm/throttle-{low,high}/{mic}-File{N}.wav`,
   где `mic` ∈ {`mic1_soundskrit`, `mic2_8array-down`, `mic3_8array-up`}.
   `mic1_soundskrit-File3` и `mic2_8array-down-File3` — **одна и та же
   запись с двух микрофонов**. Группировать по `(drone, File{N})`,
   игнорируя микрофон и дистанцию, иначе один полёт попадёт и в train, и в
   val, и метрика надуется.

Инвентарь: `drone1-only` 84 / `drone2-only` 84; `mic-dist-25cm` 84 /
`mic-dist-50cm` 84; `throttle-high` 84 / `throttle-low` 84; по 56 записей на
каждый из трёх микрофонов. Длина записи 30–152 с.

**Объём.** 1 725,1 млн значений — это `отсчёты × каналы`, а не отсчёты.
По замеренному шарду `train_001` (7,30 млн отсчётов на 6 записей, из них
13,5% тишина) на все 28 шардов выходит ~204 млн отсчётов, то есть **3,5 часа**
звука: ~22 тыс. окон дрона и ~3,5 тыс. окон тишины. Это добавка 13% к 164 тыс.
окон дрона из DADS, а не второй столп набора. Ценность — две новые машины и
тишина того же тракта, не объём.

---

## Task 1: Локальный инструментарий и зелёная база

Прежде чем что-то менять, нужно уметь запускать существующие проверки. Сейчас
`torch` не установлен, и ни один `--selfcheck` из README не проходит.

**Файлы:**
- Изменить: `README.md` (раздел «Окружение»)

**Интерфейсы:**
- Производит: рабочее локальное окружение; ничего программного.

- [ ] **Шаг 1: Установить недостающие пакеты**

Только CPU-сборка torch — GPU локально нет, а полное колесо CUDA весит 2,5 ГБ.

```bash
pip install pyarrow huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- [ ] **Шаг 2: Убедиться, что база зелёная**

```bash
for f in train.py train2.py eval.py detect.py web.py spectrum.py prep_dads.py prep_hard.py; do
    echo "--- $f"; python "$f" --selfcheck
done
```

Ожидается: `selfcheck ok` от каждого. `detect.py` и `web.py` могут ругнуться
на отсутствие звукового устройства — это нормально, важно отсутствие
`ImportError` и `AssertionError`.

- [ ] **Шаг 3: Зафиксировать окружение в README**

В разделе «Окружение» дописать, что для работы без GPU достаточно CPU-сборки
torch, и привести команду из шага 1.

- [ ] **Шаг 4: Коммит**

```bash
git add README.md
git commit -m "Окружение: CPU-сборка torch для работы без GPU"
```

---

## Task 2: `hub.py` — обёртка над HF Hub

**Файлы:**
- Создать: `hub.py`

**Интерфейсы:**
- Потребляет: `huggingface_hub`, переменную окружения `HF_TOKEN`.
- Производит:
  - `REPO: str = "Onikore/airadar-hub"`
  - `token() -> str` — из `HF_TOKEN`, иначе из Colab Secrets, иначе `RuntimeError`
  - `ensure_repo() -> None` — создаёт приватный dataset-репо, если его нет
  - `push(local: str, remote: str) -> None` — файл или каталог
  - `pull(remote: str, local: str) -> str` — файл или каталог, возвращает путь
  - `exists(remote: str) -> bool`
  - `read_json(remote: str) -> dict | None` — `None`, если файла нет
  - `write_json(obj: dict, remote: str) -> None`

- [ ] **Шаг 1: Написать падающую проверку**

Создать `hub.py` только с функцией `selfcheck()`. Проверки не ходят в сеть —
сеть проверяется отдельным шагом вручную.

```python
def selfcheck():
    import tempfile, json, os
    # токен обязателен и не подставляется молча
    old = os.environ.pop("HF_TOKEN", None)
    try:
        try:
            token()
        except RuntimeError:
            pass
        else:
            raise AssertionError("без HF_TOKEN должен быть RuntimeError")
    finally:
        if old is not None:
            os.environ["HF_TOKEN"] = old

    os.environ["HF_TOKEN"] = "hf_" + "x" * 10
    assert token().startswith("hf_")

    # нормализация путей: и Windows-разделитель, и лишние слэши
    assert _norm("cache\\meta.npz") == "cache/meta.npz"
    assert _norm("/cache//meta.npz/") == "cache/meta.npz"

    # сериализация json детерминирована и читаема
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.json")
        _dump_json({"b": 1, "a": [2, 3]}, p)
        assert json.load(open(p, encoding="utf-8")) == {"b": 1, "a": [2, 3]}
        assert "\n" in open(p, encoding="utf-8").read()   # с отступами, не одной строкой
    print("selfcheck ok")
```

- [ ] **Шаг 2: Запустить и убедиться, что падает**

```bash
python hub.py --selfcheck
```

Ожидается: `NameError: name 'token' is not defined`.

- [ ] **Шаг 3: Реализовать**

```python
"""Репозиторий HF как общий диск между Colab и рабочей машиной.

Colab теряет диск при обрыве сессии, а прогресс обучения надо видеть снаружи.
Всё, что должно пережить сессию — кэш окон, логи, чекпоинты — лежит здесь.
"""

import os
import sys
import json

REPO = "Onikore/airadar-hub"
REPO_TYPE = "dataset"


def token():
    t = os.environ.get("HF_TOKEN")
    if not t:
        try:                                  # в Colab токен живёт в Secrets
            from google.colab import userdata
            t = userdata.get("HF_TOKEN")
        except Exception:
            t = None
    if not t:
        raise RuntimeError(
            "нет токена HF: задайте переменную окружения HF_TOKEN "
            "или добавьте секрет HF_TOKEN в Colab (значок ключа слева)")
    return t


def _norm(p):
    """Пути в HF всегда через прямой слэш и без ведущего."""
    return "/".join(x for x in str(p).replace("\\", "/").split("/") if x)


def _dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=token())


def ensure_repo():
    _api().create_repo(REPO, repo_type=REPO_TYPE, private=True, exist_ok=True)


def exists(remote):
    from huggingface_hub import HfApi
    try:
        files = HfApi(token=token()).list_repo_files(REPO, repo_type=REPO_TYPE)
    except Exception:
        return False
    r = _norm(remote)
    return r in files or any(f.startswith(r + "/") for f in files)


def push(local, remote):
    api, r = _api(), _norm(remote)
    if os.path.isdir(local):
        api.upload_folder(folder_path=local, path_in_repo=r,
                          repo_id=REPO, repo_type=REPO_TYPE)
    else:
        api.upload_file(path_or_fileobj=local, path_in_repo=r,
                        repo_id=REPO, repo_type=REPO_TYPE)


def pull(remote, local):
    from huggingface_hub import hf_hub_download, snapshot_download
    r = _norm(remote)
    if local and os.path.splitext(r)[1] == "":
        return snapshot_download(REPO, repo_type=REPO_TYPE, token=token(),
                                 allow_patterns=f"{r}/*", local_dir=local)
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    p = hf_hub_download(REPO, r, repo_type=REPO_TYPE, token=token())
    import shutil
    shutil.copyfile(p, local)
    return local


def read_json(remote):
    if not exists(remote):
        return None
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, _norm(remote), repo_type=REPO_TYPE, token=token())
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, remote):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tmp.json")
        _dump_json(obj, p)
        push(p, remote)
```

Дописать в конец файла:

```python
if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else print(f"репозиторий: {REPO}")
```

- [ ] **Шаг 4: Запустить проверку**

```bash
python hub.py --selfcheck
```

Ожидается: `selfcheck ok`.

- [ ] **Шаг 5: Проверить живой круг по сети**

Единственный сетевой тест во всём плане. Требует настоящий токен.

```bash
HF_TOKEN=<токен> python -c "
import hub, tempfile, os
hub.ensure_repo()
hub.write_json({'проверка': 1}, 'runs/_smoke.json')
assert hub.exists('runs/_smoke.json')
assert hub.read_json('runs/_smoke.json') == {'проверка': 1}
assert hub.read_json('runs/нет-такого.json') is None
print('круг замкнулся')
"
```

Ожидается: `круг замкнулся`. Если `create_repo` вернёт 403 — у токена нет
`repo.write`, перевыпустить.

- [ ] **Шаг 6: Коммит**

```bash
git add hub.py
git commit -m "hub.py: репозиторий HF как общий диск с Colab"
```

---

## Task 3: `hf_sources.py` — чтение четырёх источников

**Файлы:**
- Создать: `hf_sources.py`

**Интерфейсы:**
- Потребляет: `pyarrow.parquet`, `huggingface_hub.HfFileSystem`, `soundfile`,
  `scipy.signal.resample_poly`.
- Производит:
  - `SR = 16000`
  - `Rec = namedtuple("Rec", "audio group label cat src")` — `audio` это
    `np.float32` моно 16 кГц, пик нормирован в 1.0; `group` `int`;
    `label` `int` 0/1; `cat` `str | None`; `src` `int` 0..3
  - `SRC_DADS, SRC_DAS, SRC_URBAN, SRC_ESC = 0, 1, 2, 3`
  - `shards(src: int) -> list[str]` — пути parquet на `hf://`
  - `read_shard(src: int, path: str) -> Iterator[Rec]`
  - `to_mono_16k(data, sr) -> np.ndarray | None`
  - `das_group(file_path: str) -> tuple[int, bool]` — `(ключ группы, тишина ли)`

- [ ] **Шаг 1: Написать падающие проверки**

Ключевая проверка — группировка DroneAudioSet: один полёт с трёх микрофонов и
двух дистанций обязан дать один ключ.

```python
def selfcheck():
    import numpy as np

    # --- DroneAudioSet: группа = (дрон, номер файла), микрофон и дистанция не влияют
    base = "drone-only-recordings/{d}-only/mic-dist-{c}cm/throttle-{t}/{m}-File{n}.wav"
    a = das_group(base.format(d="drone1", c=25, t="low",  m="mic1_soundskrit", n=3))
    b = das_group(base.format(d="drone1", c=50, t="high", m="mic2_8array-down", n=3))
    assert a[0] == b[0], "один полёт с разных микрофонов должен дать одну группу"
    c = das_group(base.format(d="drone2", c=25, t="low",  m="mic1_soundskrit", n=3))
    assert a[0] != c[0], "разные дроны — разные группы"
    d = das_group(base.format(d="drone1", c=25, t="low",  m="mic1_soundskrit", n=4))
    assert a[0] != d[0], "разные полёты — разные группы"

    # --- тишина распознаётся и не идёт в позитивы
    sil = das_group("drone-only-recordings/drone1-only/mic-dist-25cm/throttle-low/mic1_soundskrit-silence.wav")
    assert sil[1] is True and a[1] is False
    assert sil[0] != a[0], "тишина — отдельная группа, не смешивается с полётом"

    # --- пространства групп источников не пересекаются
    assert das_group(base.format(d="drone1", c=25, t="low", m="mic1_soundskrit", n=3))[0] // OFFSET == SRC_DAS
    assert dads_group(20941) // OFFSET == SRC_DADS
    assert urban_group(100032) // OFFSET == SRC_URBAN
    assert esc_group(7973) // OFFSET == SRC_ESC
    assert dads_group(20941) == dads_group(20999) != dads_group(21999)   # блок 250

    # --- конвертация
    x = to_mono_16k(np.random.randn(44100, 2).astype(np.float32), 44100)
    assert abs(len(x) - SR) <= 2 and abs(x).max() <= 1.0 + 1e-6
    assert to_mono_16k(np.zeros(1000, np.float32), SR) is None          # тишина отбрасывается
    y = to_mono_16k(np.random.randn(SR).astype(np.float32), SR)
    assert len(y) == SR                                                  # без ресемплинга длина цела

    # --- многоканальный float64 из DroneAudioSet сводится в моно
    z = to_mono_16k(np.random.randn(1000, 8).astype(np.float64), SR)
    assert z.dtype == np.float32 and z.ndim == 1 and len(z) == 1000
    print("selfcheck ok")
```

- [ ] **Шаг 2: Запустить и убедиться, что падает**

```bash
python hf_sources.py --selfcheck
```

Ожидается: `NameError: name 'das_group' is not defined`.

- [ ] **Шаг 3: Реализовать**

```python
"""Чтение четырёх источников HF в единый вид Rec.

У каждого источника своя схема parquet и своё правило группировки. Смысл
группировки один: куски одной физической записи не должны расползтись по
train и val, иначе метрика надувается. Правила разной точности — см.
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
BLOCK = 250                      # блок номеров DADS, как в prep_dads.py

REPOS = {
    SRC_DADS:  ("geronimobasso/drone-audio-detection-samples", "data/*.parquet"),
    SRC_DAS:   ("ahlab-drone-project/DroneAudioSet",           "drone-only/*.parquet"),
    SRC_URBAN: ("danavery/urbansound8K",                       "data/*.parquet"),
    SRC_ESC:   ("ashraq/esc50",                                "data/*.parquet"),
}

Rec = namedtuple("Rec", "audio group label cat src")

_NUM = re.compile(r"(\d+)")
# drone-only-recordings/drone1-only/mic-dist-25cm/throttle-low/mic1_soundskrit-File3.wav
_DAS = re.compile(r"/(drone\d+)-only/.*?/(File\d+|silence)\.wav$")


def dads_group(idx):
    return SRC_DADS * OFFSET + idx // BLOCK


def urban_group(fs_id):
    return SRC_URBAN * OFFSET + int(fs_id)


def esc_group(src_file):
    return SRC_ESC * OFFSET + int(src_file)


def das_group(file_path):
    """Ключ = (дрон, номер файла). Микрофон и дистанция намеренно отброшены:
    mic1_soundskrit-File3 и mic2_8array-down-File3 — одна запись с двух
    микрофонов, в разных сплитах они дадут утечку."""
    m = _DAS.search(file_path.replace("\\", "/"))
    if not m:
        raise ValueError(f"не разобран путь DroneAudioSet: {file_path}")
    drone, tail = m.group(1), m.group(2)
    silence = tail == "silence"
    dnum = int(_NUM.search(drone).group(1))
    fnum = 0 if silence else int(_NUM.search(tail).group(1))
    # тишина каждого дрона — своя группа, отдельная от полётов
    key = dnum * 1000 + (500 if silence else fnum)
    return SRC_DAS * OFFSET + key, silence


def to_mono_16k(data, sr):
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
                yield Rec(x, dads_group(idx), int(r["label"]), None, src)
            elif src == SRC_URBAN:
                yield Rec(x, urban_group(r["fsID"]), 0, r["class"], src)
            else:
                yield Rec(x, esc_group(r["src_file"]), 0, r["category"], src)


def _read_das(pf):
    """audio.array — вложенные списки float64, а не байты. to_pylist() раздул бы
    10,3 млн значений в питоновские float по 28 байт (1,25 ГБ на запись), поэтому
    достаём плоский буфер Arrow напрямую и режем по числу каналов."""
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        paths = t.column("file_path").to_pylist()
        audio = t.column("audio").combine_chunks()
        arrays = audio.field("array")
        rates = audio.field("sampling_rate").to_pylist()
        for i, fp in enumerate(paths):
            chans = arrays[i]
            n_ch = len(chans)
            if n_ch == 0:
                continue
            flat = chans.values.to_numpy(zero_copy_only=False)
            x = flat.reshape(n_ch, -1).mean(axis=0) if n_ch > 1 else flat
            x = to_mono_16k(x, int(rates[i]))
            if x is None:
                continue
            g, silence = das_group(fp)
            yield Rec(x, g, 0 if silence else 1,
                      "drone_rig_silence" if silence else None, SRC_DAS)
```

Дописать в конец:

```python
if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else print(
        "\n".join(f"{k}: {v[0]}" for k, v in REPOS.items()))
```

- [ ] **Шаг 4: Запустить проверку**

```bash
python hf_sources.py --selfcheck
```

Ожидается: `selfcheck ok`.

- [ ] **Шаг 5: Дымовой прогон по сети, один шард на источник**

Проверяет, что схемы разобраны верно. Скачивает по одному шарду (самый
тяжёлый — UrbanSound8K, ~440 МБ), поэтому запускать один раз.

```bash
HF_TOKEN=<токен> python -c "
import hf_sources as S, itertools, numpy as np
for src in (S.SRC_ESC, S.SRC_DAS, S.SRC_URBAN, S.SRC_DADS):
    sh = S.shards(src)
    recs = list(itertools.islice(S.read_shard(src, sh[0]), 3))
    print(f'src={src}  шардов={len(sh)}  прочитано={len(recs)}')
    for r in recs:
        assert r.audio.dtype == np.float32 and r.audio.ndim == 1
        assert 0.99 <= np.abs(r.audio).max() <= 1.01
        print(f'   group={r.group} label={r.label} cat={r.cat} сек={len(r.audio)/S.SR:.1f}')
"
```

Ожидается: по три записи с каждого источника, `label=1` у DADS-дрона и у
полётов DroneAudioSet, `label=0` у тишины, `cat` заполнен у ESC-50 и
UrbanSound8K. Число шардов: 39 / 28 / 16 / 2.

- [ ] **Шаг 6: Коммит**

```bash
git add hf_sources.py
git commit -m "hf_sources.py: чтение DADS, DroneAudioSet, UrbanSound8K и ESC-50"
```

---

## Task 4: `prep_hf.py` — сборка кэша и замороженный сплит

**Файлы:**
- Создать: `prep_hf.py`

**Интерфейсы:**
- Потребляет: `hf_sources.Rec`, `hf_sources.read_shard`, `hf_sources.shards`,
  `hub.push`, `hub.read_json`, `hub.write_json`.
- Производит:
  - `WIN = 8000`
  - `CAP` — предел окон с одной записи по источникам
  - `windows(x, cap) -> list[np.ndarray]`
  - `assign_split(groups, y, src, frac=(0.75, 0.15, 0.10), seed=0) -> np.ndarray[int8]`
  - `main()` — собирает `cache_dads/` и `cache_hard/`, заливает на HF

- [ ] **Шаг 1: Написать падающие проверки**

Главное, что нужно доказать: ни одна группа не пересекает границу сплита, и
все три части содержат все источники.

```python
def selfcheck():
    import numpy as np

    # --- нарезка окон (поведение как в prep_dads.py)
    assert len(windows(np.zeros(WIN), 4)) == 1
    assert len(windows(np.zeros(WIN * 10), 4)) == 4
    assert len(windows(np.zeros(WIN * 10), 1)) == 1
    (w,) = windows(np.arange(100, dtype=np.float32), 4)
    assert len(w) == WIN and w[100] == 0        # зациклено, а не дополнено нулями

    # --- сплит: группа целиком в одной части
    rng = np.random.default_rng(0)
    g = rng.integers(0, 300, 5000)
    y = (g % 2).astype(np.int8)
    src = (g % 4).astype(np.int8)
    sp = assign_split(g, y, src)
    assert sp.dtype == np.int8 and len(sp) == len(g)
    for gid in np.unique(g):
        assert len(np.unique(sp[g == gid])) == 1, f"группа {gid} разорвана между сплитами"

    # --- пропорции близки к заданным
    for part, want in enumerate((0.75, 0.15, 0.10)):
        got = (sp == part).mean()
        assert abs(got - want) < 0.10, f"часть {part}: {got:.2f} против {want}"

    # --- каждый источник и каждый класс представлены во всех трёх частях
    for part in (0, 1, 2):
        assert set(np.unique(src[sp == part])) == set(np.unique(src)), \
            f"часть {part} потеряла источник"
        assert set(np.unique(y[sp == part])) == set(np.unique(y)), \
            f"часть {part} потеряла класс"

    # --- детерминированность
    assert (assign_split(g, y, src) == sp).all()

    # --- манифест: докачка не переделывает готовое
    m = {"done": ["a.parquet"], "n": 10}
    assert _todo(["a.parquet", "b.parquet"], m) == ["b.parquet"]
    assert _todo(["a.parquet"], m) == []
    assert _todo(["a.parquet", "b.parquet"], None) == ["a.parquet", "b.parquet"]
    print("selfcheck ok")
```

- [ ] **Шаг 2: Запустить и убедиться, что падает**

```bash
python prep_hf.py --selfcheck
```

Ожидается: `NameError: name 'windows' is not defined`.

- [ ] **Шаг 3: Реализовать нарезку и сплит**

```python
"""Сборка кэша окон из четырёх источников HF с замороженным сплитом.

Отличие от prep_dads.py: сплит считается здесь и кладётся в meta.npz, а не
пересчитывается при каждом обучении. При четырёх источниках и кэше, живущем
на HF, сплит обязан быть одинаковым между сессиями — иначе два прогона
несравнимы, а именно этим проект и болел (см. NEXT_STEPS.md, шаг 0).

Третья часть (test) не трогается до финальной проверки. Сейчас в проекте
удержанной части нет вообще: auc_hard одновременно отбирает чекпоинт и
служит отчётным числом.
"""

import os
import sys
import json
import numpy as np

import hub
import hf_sources as S
from hf_sources import SRC_DADS, SRC_DAS, SRC_URBAN, SRC_ESC

ROOT = os.path.dirname(os.path.abspath(__file__))
WIN = 8000                       # 0.5 с при 16 кГц

# Окон с одной записи. У DADS клипы дрона по 0.6 с, фона по 7.3 с — отсюда
# перекос 16 к 1, он выравнивает вклад классов. DroneAudioSet ограничен, чтобы
# 3,5 часа лабораторной записи двух машин не перевесили 164k окон DADS.
# При нынешних данных кэп 700 не срабатывает ни на чём (самая длинная запись
# 152 с даёт 304 окна) — он страховка на случай перезаливки датасета.
CAP = {
    (SRC_DADS, 1): 16, (SRC_DADS, 0): 1,
    (SRC_DAS, 1): 700, (SRC_DAS, 0): 700,
    (SRC_URBAN, 0): 4, (SRC_ESC, 0): 5,
}

FRAC = (0.75, 0.15, 0.10)


def windows(x, cap):
    if len(x) < WIN:
        return [np.tile(x, int(np.ceil(WIN / len(x))))[:WIN]]
    n = min(len(x) // WIN, cap)
    return [x[i * WIN:(i + 1) * WIN] for i in range(n)]


def assign_split(groups, y, src, frac=FRAC, seed=0):
    """0 = train, 1 = val, 2 = test. Делим группы, не окна.

    Раскладываем группы по частям внутри каждой пары (источник, класс), а не
    по всему набору сразу: иначе маленький источник целиком уедет в одну часть
    и val перестанет его измерять. ESC-50 это ровно 2000 клипов против 180k
    у DADS, случайное деление всего набора легко оставило бы его без val.
    """
    groups = np.asarray(groups)
    split = np.full(len(groups), -1, np.int8)
    rng = np.random.default_rng(seed)

    for s in np.unique(src):
        for lab in np.unique(y):
            sel = (src == s) & (y == lab)
            if not sel.any():
                continue
            gs = np.unique(groups[sel])
            rng.shuffle(gs)
            n = len(gs)
            # хотя бы по одной группе в val и test, если групп хватает
            n_va = max(1, int(round(n * frac[1]))) if n >= 3 else 0
            n_te = max(1, int(round(n * frac[2]))) if n >= 3 else 0
            n_va = min(n_va, n - 1)
            n_te = min(n_te, n - 1 - n_va)
            parts = np.concatenate([
                np.zeros(n - n_va - n_te, np.int8),
                np.ones(n_va, np.int8),
                np.full(n_te, 2, np.int8)])
            table = dict(zip(gs.tolist(), parts.tolist()))
            idx = np.flatnonzero(sel)
            split[idx] = [table[g] for g in groups[idx].tolist()]

    assert (split >= 0).all(), "остались нераспределённые окна"
    return split


def _todo(all_shards, manifest):
    done = set((manifest or {}).get("done", []))
    return [s for s in all_shards if s not in done]
```

- [ ] **Шаг 4: Запустить проверку**

```bash
python prep_hf.py --selfcheck
```

Ожидается: `selfcheck ok`.

- [ ] **Шаг 5: Коммит промежуточного результата**

```bash
git add prep_hf.py
git commit -m "prep_hf.py: нарезка окон и замороженный сплит по группам"
```

- [ ] **Шаг 6: Реализовать сборку с манифестом**

Дописать в `prep_hf.py`. Позитивы и негативы с категориями пишутся в два
разных кэша, чтобы сохранить контракты `load_cache` и `load_hard`.

```python
# Куда какой источник попадает. UrbanSound8K и ESC-50 несут категорию, значит
# идут в cache_hard — на них строится таблица ложных срабатываний. Тишина
# DroneAudioSet тоже: это негатив с категорией, причём того же тракта, что и
# полёты, то есть самый честный негатив в наборе.
def _dest(rec):
    if rec.cat is not None:
        return "cache_hard"
    return "cache_dads"


def _process(src, out_dirs, state):
    """Обрабатывает шарды источника, дописывая в открытые файлы. Манифест
    обновляется после каждого шарда: сессия Colab рвётся, и переделывать
    полтора часа из-за обрыва на последнем шарде недопустимо."""
    todo = _todo(S.shards(src), state["manifest"])
    for i, path in enumerate(todo):
        for rec in S.read_shard(src, path):
            d = _dest(rec)
            for w in windows(rec.audio, CAP[(rec.src, rec.label)]):
                state[d]["bin"].write((w * 32767).astype(np.int16).tobytes())
                state[d]["y"].append(rec.label)
                state[d]["group"].append(rec.group)
                state[d]["src"].append(rec.src)
                state[d]["cat"].append(rec.cat or "")
        state["manifest"].setdefault("done", []).append(path)
        state["manifest"]["counts"] = {k: len(state[k]["y"]) for k in out_dirs}
        hub.write_json(state["manifest"], "manifest.json")
        print(f"  src={src} шард {i+1}/{len(todo)}  "
              f"dads={len(state['cache_dads']['y'])} "
              f"hard={len(state['cache_hard']['y'])}", flush=True)


def main(upload=True):
    out_dirs = ("cache_dads", "cache_hard")
    state = {"manifest": hub.read_json("manifest.json") or {"done": []}}
    for d in out_dirs:
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
        mode = "ab" if state["manifest"].get("done") else "wb"
        state[d] = {"bin": open(os.path.join(ROOT, d, "windows.bin"), mode),
                    "y": [], "group": [], "src": [], "cat": []}

    for src in (SRC_ESC, SRC_URBAN, SRC_DAS, SRC_DADS):   # мелкие первыми: быстрее падает на ошибке схемы
        _process(src, out_dirs, state)

    for d in out_dirs:
        state[d]["bin"].close()
        y = np.array(state[d]["y"], np.int8)
        g = np.array(state[d]["group"], np.int64)
        s = np.array(state[d]["src"], np.int8)
        cat = np.array(state[d]["cat"])
        split = assign_split(g, y, s)
        meta = dict(y=y, group=g, src=s, split=split,
                    synth=np.zeros(len(y), bool),
                    n=np.int64(len(y)), win=np.int64(WIN))
        if d == "cache_hard":
            meta["cat"] = cat
            meta["hard"] = np.array(sorted(HARD))
        np.savez(os.path.join(ROOT, d, "meta.npz"), **meta)
        print(f"\n{d}: окон {len(y)}  ({len(y)*WIN*2/1e9:.2f} ГБ)  групп {len(np.unique(g))}")
        for part, nm in enumerate(("train", "val", "test")):
            print(f"  {nm}: {(split==part).sum()}  дрон {int(y[(split==part)&(y==1)].sum())}")
        if upload:
            hub.push(os.path.join(ROOT, d), f"cache/{d}")


HARD = {
    "chainsaw", "helicopter", "airplane", "engine", "train", "vacuum_cleaner",
    "washing_machine", "hand_saw", "wind", "rain", "thunderstorm", "crackling_fire",
    "air_conditioner", "drilling", "engine_idling", "jackhammer",
    "drone_rig_silence",
}
```

Дописать в конец файла:

```python
if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
```

- [ ] **Шаг 7: Дополнить проверку новой константой**

В `selfcheck()` добавить, что расширенный `HARD` не содержит опечаток
относительно категорий, которые реально приходят из источников:

```python
    # HARD не должен содержать категорий, которых нет ни в одном источнике
    known = {
        "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep",
        "crow", "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
        "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
        "crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps",
        "laughing", "brushing_teeth", "snoring", "drinking", "door_knock",
        "mouse_click", "keyboard", "door_wood_creaks", "can_opening",
        "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick",
        "glass_breaking", "helicopter", "chainsaw", "siren", "car_horn", "engine",
        "train", "church_bells", "airplane", "fireworks", "hand_saw",
        "air_conditioner", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "street_music",
        "drone_rig_silence",
    }
    assert HARD <= known, f"опечатка в HARD: {HARD - known}"
```

- [ ] **Шаг 8: Запустить проверку**

```bash
python prep_hf.py --selfcheck
```

Ожидается: `selfcheck ok`. Если упадёт на `HARD <= known` — в `HARD` осталась
категория из старого `prep_hard.py`, где ESC-50 читался из имён файлов и
`door_creaks` писался иначе, чем `door_wood_creaks` в колонке `category`.

- [ ] **Шаг 9: Коммит**

```bash
git add prep_hf.py
git commit -m "prep_hf.py: сборка двух кэшей с манифестом и докачкой"
```

---

## Task 5: Чтение замороженного сплита в `train.py`, `eval.py`, `diag_leak.py`

**Файлы:**
- Изменить: `train.py:209-244` (`load_hard`, `load_cache`), `train.py:273-286` (`main`)
- Изменить: `eval.py:100-105`
- Изменить: `diag_leak.py:63-66`

**Интерфейсы:**
- Потребляет: `split` из `meta.npz`, созданный `prep_hf.assign_split`.
- Производит:
  - `load_split(name="cache_dads") -> np.ndarray[int8]` — с откатом на
    `GroupShuffleSplit`, если `split` в meta нет
  - `load_hard_test() -> tuple[np.ndarray, np.ndarray]` — `(X, cat)` удержанной части
  - `load_hard()` и `load_cache()` — сигнатуры **не меняются**

- [ ] **Шаг 1: Написать падающую проверку**

Дописать в `selfcheck()` файла `train.py`. Проверка работает на синтетическом
кэше во временном каталоге — настоящий кэш локально отсутствует.

```python
    # --- замороженный сплит читается из meta, а не пересчитывается
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        n = 400
        want = (np.arange(n) % 3).astype(np.int8)
        np.savez(os.path.join(d, "meta.npz"),
                 y=(np.arange(n) % 2).astype(np.int8),
                 group=np.arange(n) // 4, src=np.zeros(n, np.int8),
                 split=want, synth=np.zeros(n, bool),
                 n=np.int64(n), win=np.int64(8000))
        np.memmap(os.path.join(d, "windows.bin"), dtype=np.int16,
                  mode="w+", shape=(n, 8000)).flush()
        got = load_split(d)
        assert (got == want).all(), "сплит должен читаться из meta без изменений"

        # откат на GroupShuffleSplit, если поля split нет (старые кэши)
        m = dict(np.load(os.path.join(d, "meta.npz"), allow_pickle=True))
        del m["split"]
        np.savez(os.path.join(d, "meta.npz"), **m)
        old = load_split(d)
        assert set(np.unique(old)) <= {0, 1}, "откат даёт только train и val"
        assert 0.10 < (old == 1).mean() < 0.20
```

- [ ] **Шаг 2: Запустить и убедиться, что падает**

```bash
python train.py --selfcheck
```

Ожидается: `NameError: name 'load_split' is not defined`.

- [ ] **Шаг 3: Добавить `load_split` и `load_hard_test` в `train.py`**

Вставить после `load_cache` (после строки 244):

```python
def load_split(name="cache_dads"):
    """Сплит берётся из meta.npz, где его заморозил prep_hf.py.

    0 = train, 1 = val, 2 = test. Пересчитывать его при каждом обучении, как
    было раньше, нельзя: при четырёх источниках любое изменение порядка данных
    молча переставит границу и сделает два прогона несравнимыми.

    Старые кэши (cache_dads от prep_dads.py) поля split не имеют — для них
    остаётся прежнее деление, чтобы ранее полученные числа воспроизводились.
    """
    d = name if os.path.isdir(name) else os.path.join(ROOT, name)
    m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    if "split" in m.files:
        return m["split"].astype(np.int8)
    n = int(m["n"])
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
                  .split(np.arange(n), m["y"], m["group"]))
    s = np.zeros(n, np.int8)
    s[va] = 1
    return s


def load_hard_test():
    """Удержанная треть трудных негативов. Не вызывать до финальной проверки:
    всё, что попадает в отбор чекпоинта, перестаёт быть честной метрикой."""
    d = os.path.join(ROOT, "cache_hard")
    if not os.path.exists(os.path.join(d, "meta.npz")):
        return None, None
    m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    X = np.memmap(os.path.join(d, "windows.bin"), dtype=np.int16,
                  mode="r", shape=(int(m["n"]), int(m["win"])))
    te = np.flatnonzero(load_split("cache_hard") == 2)
    return X[te], m["cat"][te]
```

- [ ] **Шаг 4: Переключить `load_hard` на замороженный сплит**

Заменить тело `load_hard` (строки 209-229) так, чтобы возвращаемый кортеж
остался четырёхэлементным:

```python
def load_hard(train_frac=0.5):
    """Трудные негативы (бензопила, ветер, двигатель) с категориями.

    Возвращает (X, cat, train_idx, val_idx) — форма прежняя, её ждут eval.py,
    compare_models.py и evalx/field_ci.py. Изменилось происхождение индексов:
    раньше делили пополам по порядку окон внутри категории, теперь берём
    замороженный сплит по группам. Прежнее деление рвало исходную запись между
    обучением и оценкой, если её окна попадали по разные стороны среза.

    Аргумент train_frac сохранён для совместимости и игнорируется, когда в
    meta есть split.
    """
    d = os.path.join(ROOT, "cache_hard")
    if not os.path.exists(os.path.join(d, "meta.npz")):
        return None, None, None, None
    m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    X = np.memmap(os.path.join(d, "windows.bin"), dtype=np.int16,
                  mode="r", shape=(int(m["n"]), int(m["win"])))
    cat = m["cat"]
    if "split" in m.files:
        s = m["split"]
        return X, cat, np.flatnonzero(s == 0), np.flatnonzero(s == 1)
    tr, va = [], []
    for c in np.unique(cat):
        idx = np.flatnonzero(cat == c)
        cut = int(len(idx) * train_frac)
        tr.append(idx[:cut]); va.append(idx[cut:])
    return X, cat, np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))
```

- [ ] **Шаг 5: Переключить `main` в `train.py`**

Заменить строки 274-275:

```python
    X, y, groups, synth = load_cache()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
                  .split(X, y, groups))
```

на:

```python
    X, y, groups, synth = load_cache()
    sp = load_split()
    tr, va = np.flatnonzero(sp == 0), np.flatnonzero(sp == 1)
    if (sp == 2).any():
        print(f"удержано в test и не используется: {(sp == 2).sum()} окон")
```

- [ ] **Шаг 6: Переключить `eval.py`**

Заменить строки 101-104:

```python
    X, y, groups, synth = load_cache()
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
                 .split(X, y, groups))
    va = np.sort(va)
```

на:

```python
    X, y, groups, synth = load_cache()
    va = np.flatnonzero(load_split() == 1)
```

и в строке 19 добавить `load_split` к импорту из `train`, а импорт
`GroupShuffleSplit` (строка 16) удалить.

- [ ] **Шаг 7: Переключить `diag_leak.py`**

Строка 65 меряет утечку между train и val. Если сплит здесь свой, диагностика
проверяет разбиение, на котором никто не учится. Заменить строки 64-66:

```python
    X, y, g = load()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(X, y, g))
    print(f"групп всего: {len(np.unique(g))}  train: {len(np.unique(g[tr]))}  val: {len(np.unique(g[va]))}")
```

на:

```python
    X, y, g = load()
    from train import load_split
    sp = load_split()
    tr, va = np.flatnonzero(sp == 0), np.flatnonzero(sp == 1)
    print(f"групп всего: {len(np.unique(g))}  train: {len(np.unique(g[tr]))}  "
          f"val: {len(np.unique(g[va]))}  (сплит замороженный, из meta.npz)")
```

Импорт `GroupShuffleSplit` в строке 18 удалить.

- [ ] **Шаг 8: Запустить проверки**

```bash
python train.py --selfcheck && python eval.py --selfcheck 2>/dev/null || python -c "import eval; print('eval импортируется')"
python -c "import diag_leak; print('diag_leak импортируется')"
```

Ожидается: `selfcheck ok` от `train.py`, отсутствие `ImportError` у остальных.

- [ ] **Шаг 9: Коммит**

```bash
git add train.py eval.py diag_leak.py
git commit -m "Замороженный сплит из meta.npz вместо пересчёта GroupShuffleSplit"
```

---

## Task 6: Возобновляемое обучение и лог, читаемый снаружи

**Файлы:**
- Изменить: `train.py:273-385` (`main`)

**Интерфейсы:**
- Потребляет: `hub.push`, `hub.pull`, `hub.exists`.
- Производит:
  - `main(..., run="dronenet", resume=False)` — новые аргументы
  - На HF после каждой эпохи: `runs/<run>/train.log`, `runs/<run>/metrics.jsonl`,
    `runs/<run>/last.pt`, `runs/<run>/best.pt`

- [ ] **Шаг 1: Написать падающую проверку**

Дописать в `selfcheck()` файла `train.py`:

```python
    # --- строка метрик сериализуема в jsonl и содержит номер эпохи
    row = _metrics_row(3, 0.42, {"auc": 0.99, "auc_hard": 0.87,
                                 "rec_field": {"a.wav": 0.24}})
    import json as _json
    back = _json.loads(_json.dumps(row, ensure_ascii=False))
    assert back["ep"] == 3 and back["loss"] == 0.42
    assert back["auc_hard"] == 0.87 and back["rec_field"]["a.wav"] == 0.24
    assert "\n" not in _json.dumps(row)          # ровно одна строка на эпоху
```

- [ ] **Шаг 2: Запустить и убедиться, что падает**

```bash
python train.py --selfcheck
```

Ожидается: `NameError: name '_metrics_row' is not defined`.

- [ ] **Шаг 3: Реализовать сериализацию метрик**

Вставить перед `def main(` в `train.py`:

```python
def _metrics_row(ep, loss, m):
    """Одна строка jsonl на эпоху. Текстовый лог читает человек, jsonl —
    инструменты: прогон идёт в Colab, и разбирать его приходится по файлу,
    вытянутому с HF, а не по экрану."""
    row = {"ep": int(ep), "loss": float(loss)}
    for k in ("auc", "auc_real", "auc_hard", "far_hard@r90"):
        if k in m:
            row[k] = float(m[k])
    if "rec_field" in m:
        row["rec_field"] = {str(k): float(v) for k, v in m["rec_field"].items()}
    return row
```

- [ ] **Шаг 4: Запустить проверку**

```bash
python train.py --selfcheck
```

Ожидается: `selfcheck ok`.

- [ ] **Шаг 5: Добавить возобновление и выгрузку в `main`**

Изменить сигнатуру (строка 273):

```python
def main(epochs=30, bs=256, lr=3e-4, use_synth=False, model_cls=None,
         out_name="dronenet.pt", run=None, resume=False):
```

После создания `sched` и `lossf` (после строки 325) вставить:

```python
    run = run or os.path.splitext(out_name)[0]
    remote = f"runs/{run}"
    start_ep = 0
    if resume:
        import hub
        if hub.exists(f"{remote}/last.pt"):
            p = hub.pull(f"{remote}/last.pt", os.path.join(ROOT, "models", f"{run}_last.pt"))
            ck = torch.load(p, map_location=DEV, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            best = ck.get("best", 0.0)
            start_ep = ck["ep"]
            torch.set_rng_state(ck["rng"].cpu())
            np.random.set_state(tuple(ck["np_rng"]))
            print(f"продолжаем с эпохи {start_ep + 1}, лучшая метрика {best:.4f}")
        else:
            print(f"на HF нет {remote}/last.pt — начинаем с нуля")
```

Заменить заголовок цикла (строка 331):

```python
    for ep in range(epochs):
```

на:

```python
    for ep in range(start_ep, epochs):
```

- [ ] **Шаг 6: Выгружать состояние после каждой эпохи**

Сразу после `print(f"ep{ep+1:02d} ...")` (после строки 385) вставить:

```python
        # Пуш после каждой эпохи, а не в конце: бесплатный Colab рвёт сессию,
        # и потерять восемь эпох из-за обрыва на девятой недопустимо.
        import hub
        line = f"ep{ep+1:02d} loss {tot/n:.4f} auc {m['auc']:.4f} auc_hard {m.get('auc_hard', float('nan')):.4f}\n"
        log_p = os.path.join(ROOT, "logs", f"{run}.log")
        jsonl_p = os.path.join(ROOT, "logs", f"{run}.jsonl")
        os.makedirs(os.path.dirname(log_p), exist_ok=True)
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(line)
        with open(jsonl_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(_metrics_row(ep + 1, tot / n, m), ensure_ascii=False) + "\n")
        last_p = os.path.join(ROOT, "models", f"{run}_last.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "ep": ep + 1, "best": best,
                    "rng": torch.get_rng_state(),
                    "np_rng": np.random.get_state()}, last_p)
        try:
            hub.push(log_p, f"{remote}/train.log")
            hub.push(jsonl_p, f"{remote}/metrics.jsonl")
            hub.push(last_p, f"{remote}/last.pt")
            if tag:
                hub.push(os.path.join(ROOT, "models", out_name), f"{remote}/best.pt")
        except Exception as e:
            print(f"  выгрузка на HF не удалась ({e}); обучение продолжается")
```

Добавить `import json` в начало файла (после `import sys`, строка 15).

- [ ] **Шаг 7: Запустить проверку**

```bash
python train.py --selfcheck
```

Ожидается: `selfcheck ok`. Полный прогон локально невозможен — кэша нет,
проверяется в Task 8 в Colab.

- [ ] **Шаг 8: Коммит**

```bash
git add train.py
git commit -m "Обучение: возобновление с HF и выгрузка лога после каждой эпохи"
```

---

## Task 7: `notebooks/01_prep.ipynb`

**Файлы:**
- Создать: `notebooks/01_prep.ipynb`

**Интерфейсы:**
- Потребляет: `prep_hf.main`, `hub.ensure_repo`.
- Производит: `cache/cache_dads/` и `cache/cache_hard/` на HF, `manifest.json`.

- [ ] **Шаг 1: Собрать ноутбук**

Ноутбук пишется как `.py` со скриптом-конвертером — редактировать json
вручную неудобно и легко сломать. Создать `notebooks/01_prep.py`:

```python
# %% [markdown]
# # Сборка кэша окон (запускается один раз, ~1.5–2 ч)
#
# Перед запуском: значок ключа слева -> добавить секрет `HF_TOKEN`
# с правом записи, включить доступ для этого ноутбука.
#
# Обрыв сессии не страшен: прогресс пишется в `manifest.json` на HF,
# повторный запуск продолжит с места обрыва.

# %%
!pip install -q pyarrow soundfile huggingface_hub

# %%
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

!git clone -q https://github.com/Onikore/airadar.git /content/airadar || true
%cd /content/airadar

# %%
# Свободное место: кэш ~6,7 ГБ плюс шарды источников в кэше huggingface_hub.
!df -h /content | tail -1
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# %%
import hub
hub.ensure_repo()
print("репозиторий:", hub.REPO)

# %%
# Схемы источников — печатаются до обработки, чтобы не молотить 18 ГБ,
# если структура датасета на HF изменилась с момента написания кода.
import hf_sources as S
for src in (S.SRC_ESC, S.SRC_URBAN, S.SRC_DAS, S.SRC_DADS):
    sh = S.shards(src)
    print(f"src={src}  {S.REPOS[src][0]}  шардов={len(sh)}")
assert len(S.shards(S.SRC_DADS)) == 39, "число шардов DADS изменилось"
assert len(S.shards(S.SRC_DAS)) == 28
assert len(S.shards(S.SRC_URBAN)) == 16
assert len(S.shards(S.SRC_ESC)) == 2

# %%
import prep_hf
prep_hf.main(upload=True)

# %%
# Полевые записи кладутся отдельно: они не публичные и приходят от оператора.
# Загрузить их в /content/airadar/field/ и выполнить эту ячейку.
import os, hub
if os.path.isdir("field") and os.listdir("field"):
    hub.push("field", "field")
    print("полевые записи выгружены:", os.listdir("field"))
else:
    print("field/ пуст — шаг пропущен, обучение пойдёт без recall_поле")
```

- [ ] **Шаг 2: Сконвертировать в ipynb**

```bash
pip install jupytext
jupytext --to notebook notebooks/01_prep.py -o notebooks/01_prep.ipynb
```

- [ ] **Шаг 3: Проверить, что ноутбук читается**

```bash
python -c "
import json
nb = json.load(open('notebooks/01_prep.ipynb', encoding='utf-8'))
print('ячеек:', len(nb['cells']))
assert len(nb['cells']) >= 7
assert not any('hf_' + 'TZPK' in ''.join(c['source']) for c in nb['cells']), 'токен в ноутбуке'
print('ok')
"
```

Ожидается: `ячеек: 8`, затем `ok`.

- [ ] **Шаг 4: Коммит**

```bash
git add notebooks/01_prep.py notebooks/01_prep.ipynb
git commit -m "Ноутбук сборки кэша в Colab"
```

---

## Task 8: `notebooks/02_train.ipynb` и живой прогон

**Файлы:**
- Создать: `notebooks/02_train.py`, `notebooks/02_train.ipynb`
- Изменить: `RUNBOOK.md`

**Интерфейсы:**
- Потребляет: `hub.pull`, `train.main`.
- Производит: работающий цикл обучения в Colab с выгрузкой на HF.

- [ ] **Шаг 1: Собрать ноутбук**

Создать `notebooks/02_train.py`:

```python
# %% [markdown]
# # Обучение (кэш тянется с HF, ~5 мин)
#
# Сессия оборвалась — просто запустить ноутбук заново: `resume=True`
# подхватит `last.pt` с HF и продолжит с прерванной эпохи.

# %%
!pip install -q pyarrow soundfile huggingface_hub

# %%
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
!git clone -q https://github.com/Onikore/airadar.git /content/airadar || true
%cd /content/airadar
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# %%
import hub, os
for d in ("cache_dads", "cache_hard"):
    if not os.path.exists(f"{d}/meta.npz"):
        hub.pull(f"cache/{d}", ".")
if not os.path.isdir("field"):
    try:
        hub.pull("field", ".")
    except Exception:
        print("полевых записей на HF нет — recall_поле считаться не будет")
!du -sh cache_dads cache_hard 2>/dev/null

# %%
# Состав кэша и сплита — глазами, до запуска обучения.
import numpy as np
for d in ("cache_dads", "cache_hard"):
    m = np.load(f"{d}/meta.npz", allow_pickle=True)
    sp, y = m["split"], m["y"]
    print(f"\n{d}: окон {int(m['n'])}  групп {len(np.unique(m['group']))}")
    for p, nm in enumerate(("train", "val", "test")):
        print(f"  {nm:5s} {int((sp==p).sum()):>7}  дрон {int(y[(sp==p)].sum()):>7}")
    for gid in np.unique(m["group"]):
        assert len(np.unique(sp[m["group"] == gid])) == 1, f"группа {gid} разорвана"
print("\nни одна группа не пересекает границу сплита")

# %%
import train
train.main(epochs=12, bs=256, run="dronenet_hf", resume=True)

# %%
# Метрики прогона — те же строки, что лежат на HF.
import json
for line in open("logs/dronenet_hf.jsonl", encoding="utf-8"):
    print(json.loads(line))
```

- [ ] **Шаг 2: Сконвертировать и проверить**

```bash
jupytext --to notebook notebooks/02_train.py -o notebooks/02_train.ipynb
python -c "
import json
nb = json.load(open('notebooks/02_train.ipynb', encoding='utf-8'))
assert len(nb['cells']) >= 6
assert not any('hf_' + 'TZPK' in ''.join(c['source']) for c in nb['cells'])
print('ячеек:', len(nb['cells']), 'ok')
"
```

- [ ] **Шаг 3: Опубликовать код на GitHub**

Ноутбуки клонируют `https://github.com/Onikore/airadar.git` (публичный, ветка
`master`). Без публикации Colab получит версию кода до всех правок этого плана.

```bash
git push origin master
git log --oneline origin/master -1
```

Ожидается: в выводе последний коммит этого плана, не `ce07091`.

- [ ] **Шаг 4: Живой прогон в Colab**

Это единственная сквозная проверка всей цепочки, и выполняет её человек.

1. Открыть `notebooks/01_prep.ipynb` в Colab, добавить секрет `HF_TOKEN`,
   выполнить целиком. Ожидается печать состава кэша: `cache_dads` порядка
   355-375 тысяч окон, `cache_hard` порядка 44-53 тысяч, каждая часть сплита
   содержит все источники.
2. Открыть `notebooks/02_train.ipynb`, выполнить целиком. Ожидается:
   ячейка проверки сплита печатает «ни одна группа не пересекает границу»,
   обучение идёт, после первой же эпохи на HF появляется
   `runs/dronenet_hf/train.log`.
3. Прервать сессию на 3-4 эпохе и запустить ноутбук заново. Ожидается строка
   `продолжаем с эпохи N`, где N — номер прерванной.

- [ ] **Шаг 5: Обновить RUNBOOK**

В `RUNBOOK.md` добавить раздел «Обучение в Colab» перед «Живой детектор»:
ссылки на оба ноутбука, требование секрета `HF_TOKEN`, указание, что
повторный запуск `02_train` продолжает прерванный прогон, и команда чтения
лога с HF без Colab:

```bash
HF_TOKEN=<токен> python -c "
import hub, json
p = hub.pull('runs/dronenet_hf/metrics.jsonl', 'logs/remote.jsonl')
for l in open(p, encoding='utf-8'): print(json.loads(l))
"
```

- [ ] **Шаг 6: Коммит**

```bash
git add notebooks/02_train.py notebooks/02_train.ipynb RUNBOOK.md
git commit -m "Ноутбук обучения в Colab с возобновлением, раздел в RUNBOOK"
```

---

## Самопроверка плана

**Покрытие спецификации.** Источники — Task 3. Кэп DroneAudioSet — Task 4
(`CAP`). Отказ от `source-only` — состав `REPOS` в Task 3. Заморозка сплита —
Task 4 (`assign_split`) и Task 5 (`load_split`). Три части — Task 4 (`FRAC`),
удержанная часть недоступна обычным путём (Task 5, `load_hard_test`).
Группировка по четырём правилам — Task 3. Формат `meta.npz` — Task 4.
Правки `train.py`, `eval.py` — Task 5. `hub.py` — Task 2. Ноутбуки — Task 7 и
8. Устойчивость к обрыву: prep — Task 4 (манифест), train — Task 6
(`resume`). Полевые записи — Task 7, последняя ячейка. Проверка квоты HF —
Task 7, ячейка `df -h` и живой прогон.

**Отклонение от спецификации.** Спецификация утверждает, что `diag_*.py`
правок не требуют. `diag_leak.py:65` вошёл в Task 5: его единственная задача —
измерить утечку между train и val, и на собственном сплите он измеряет
разбиение, на котором никто не учится. Остальные `diag_*` не тронуты.

**Добавление против спецификации.** Тишина DroneAudioSet (24 записи из 168)
разведкой обнаружена уже после утверждения дизайна. Идёт в `cache_hard`
категорией `drone_rig_silence`, добавлена в `HARD` (Task 4).

**Согласованность имён.** `load_split(name)` — Task 5, вызывается в Task 5
(`train.main`, `eval.main`, `diag_leak.main`) и Task 6. `assign_split(groups,
y, src, frac, seed)` — Task 4, вызывается там же. `Rec(audio, group, label,
cat, src)` — Task 3, разбирается в Task 4 (`_process`, `_dest`).
`hub.push(local, remote)` — Task 2, вызывается в Task 4, 6, 7.
`hub.pull(remote, local)` — Task 2, вызывается в Task 6, 8.
`_metrics_row(ep, loss, m)` — Task 6. `das_group` возвращает пару
`(ключ, тишина)` — Task 3, распаковывается в `_read_das` там же.

**Незакрытый риск.** Числа окон в Task 8 («355-375 тысяч», «44-53 тысяч») —
оценка по инвентарю источников, а не измерение. Расхождение вдвое означает
ошибку в `CAP` или в разборе схемы, и это повод остановиться, а не
подкручивать константу.
