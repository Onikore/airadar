# Модель (гармоническая укладка + MIL) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Собрать саму модель `DroneNet2` (гармоническая ветка A + текстурная
ветка B + attention-MIL агрегация + вспомогательные головы f0/salience) как
самостоятельный, тестируемый `nn.Module`, без обучающего цикла — плюс
`airadar/config.py` (инвариант «фронтенд из FeatureCfg») и заполнение
`f0_med/salience/lf_energy` в реальном манифесте.

**Architecture:** Вход модели — `[B, 2, 183, 32]` (выход
`Frontend.last_model_frames`, этап 2). Ветка A делает harmonic stacking по
индексам (без вычислений) на f0-диапазоне 40–400 Гц и свёртку по карте
`p(f0,t)`; ветка B — обычная свёртка по полной сетке. Обе схлопываются в
покадровые логиты (`logsumexp`), сливаются обучаемой линейной комбинацией,
агрегируются в клип-логит через gated-attention MIL (Ilse et al. 2018).
Вспомогательные головы f0/salience читаются с карты ветки A без своих
параметров — форсируют её быть тем, чем она должна быть.

**Tech Stack:** PyTorch (`torch.nn`), NumPy, PyArrow (манифест), продолжает
`airadar/features/{cqt,frontend}.py` этапа 2.

## Global Constraints

- Признак фронтенда (этап 2, не меняется): `SR=16000, HOP_S=0.128,
  FMIN=40.0, FMAX=8000.0, BINS_PER_OCTAVE=24, N_BINS=183, MODEL_FRAMES=32`.
- f0-диапазон гармонической ветки: 40–400 Гц = 3.32 октавы = **80 бинов**
  на лог-оси (спецификация §3.1).
- Harmonic stacking: для f0-бина `i` берутся бины `i + round(24·log2 k)`,
  `k = 1..8` — чистое индексирование (§3.1).
- Вход ветки A: `[2·8, 80, 32]` (§3.1). Вход ветки B: `[2, 183, 32]` (§3.2).
- Клип-логит — обучаемая взвешенная сумма покадровых логитов по 32 кадрам
  (attention-MIL, §3.3). Правило «k из m» не используется нигде в новом коде.
- Вспомогательные головы f0/salience — не продукт, только регуляризация при
  обучении; в рантайме не используются (§3.4).
- Размер модели: **400 000–800 000** параметров (§3.5).
- Каждый модуль имеет `--selfcheck` (§8, конвенция проекта).
- Целевой размер модуля — до ~200 строк (§8).
- Инвариант §8: фронтенд конструируется из `FeatureCfg`, а не из глобалов
  модуля — рассинхронизация train/inference признака должна быть невозможна
  структурно, а не только по соглашению.

---

### Task 1: `airadar/config.py` + `FeatureCfg` в `LogCQT`/`Frontend`

**Files:**
- Create: `airadar/config.py`
- Modify: `airadar/features/cqt.py` (класс `LogCQT`, метод `__init__`)
- Modify: `airadar/features/frontend.py` (класс `Frontend`, метод `__init__`
  и `forward`; `last_model_frames` НЕ трогать — он уже explicitly не часть
  признака, см. докстринг файла)

**Interfaces:**
- Produces: `airadar.config.FeatureCfg` (frozen dataclass: `sr, hop_s, fmin,
  fmax, bins_per_octave, n_bins, model_frames, bg_window_frames,
  bg_quantile`, свойство `hop_length`), `airadar.config.ModelCfg` (frozen
  dataclass: `f0_lo, f0_hi, n_harmonics, branch_hidden, mil_hidden`).
- Consumes: ничего нового — `LogCQT`/`Frontend` уже существуют (этап 2).

Значения по умолчанию `FeatureCfg` обязаны воспроизводить нынешние модульные
константы `airadar/features/cqt.py` (`SR=16000, HOP_S=0.128, FMIN=40.0,
FMAX=8000.0, BINS_PER_OCTAVE=24, N_BINS=183`) и
`airadar/features/frontend.py`/`background.py`
(`MODEL_FRAMES=32, BG_WINDOW_FRAMES=63, BG_QUANTILE=0.20`) — иначе
существующие вызовы `LogCQT()`/`Frontend()` без аргументов изменят
поведение задним числом, а это уже смерженный, отревьюженный код этапа 2.

- [ ] **Step 1: Написать `airadar/config.py`**

```python
"""Конфигурация признака и модели как данные, а не глобалы модуля.

Инвариант §8 спецификации: фронтенд конструируется из FeatureCfg, лежащего
в чекпоинте, а не из констант airadar.features.cqt/frontend напрямую.
Раньше (detect.py, архивный) рассинхрон train/inference признака (win/sr)
был возможен и только предупреждался, не блокировался. FeatureCfg —
единственный источник параметров фронтенда; сериализуется вместе с весами
в train/checkpoint.py.

Значения по умолчанию здесь ОБЯЗАНЫ совпадать с модульными константами
airadar/features/cqt.py и airadar/features/frontend.py, background.py —
иначе LogCQT()/Frontend() без аргументов (уже смерженный код этапа 2)
изменят поведение задним числом.
"""
import sys
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class FeatureCfg:
    sr: int = 16000
    hop_s: float = 0.128
    fmin: float = 40.0
    fmax: float = 8000.0
    bins_per_octave: int = 24
    n_bins: int = 183
    model_frames: int = 32
    bg_window_frames: int = 63
    bg_quantile: float = 0.20

    @property
    def hop_length(self):
        return round(self.hop_s * self.sr)


@dataclass(frozen=True)
class ModelCfg:
    """Только то, что реально параметризует DroneNet2 (airadar/models/
    dronenet2.py). f0-диапазон (40-400 Гц) и число гармоник (8) — не здесь:
    они зашиты в airadar/features/harmonic.py как производные сетки CQT
    (bins_per_octave), а не независимые настройки — гармонический индекс
    round(24*log2 k) осмыслен только при фиксированном bins_per_octave
    FeatureCfg, дублировать его в ModelCfg значило бы завести два источника
    истины для одного и того же числа."""
    branch_hidden: int = 128
    mil_hidden: int = 16


def selfcheck():
    cfg = FeatureCfg()
    assert cfg.hop_length == 2048, cfg.hop_length   # совпадает с cqt.HOP_LENGTH
    d = asdict(cfg)
    assert d["sr"] == 16000 and d["n_bins"] == 183

    cfg2 = FeatureCfg(sr=8000)
    assert cfg2.hop_length == round(0.128 * 8000)

    # frozen — конфиг разделяется между обучением и инференсом, случайная
    # мутация одного не должна быть возможна в принципе
    try:
        cfg.sr = 44100
    except Exception:
        pass
    else:
        raise AssertionError("FeatureCfg обязан быть frozen")

    mc = ModelCfg()
    assert mc.branch_hidden == 128 and mc.mil_hidden == 16

    print("config selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Проверить, что модуль импортируется и селфчек проходит**

Run: `python airadar/config.py --selfcheck`
Expected: `config selfcheck ok`

- [ ] **Step 3: Завести `FeatureCfg` в `LogCQT` (`airadar/features/cqt.py`)**

Заменить `__init__` `LogCQT` (сейчас использует глобалы `SR, HOP_LENGTH,
FMIN, FMAX, N_BINS, BINS_PER_OCTAVE` напрямую):

```python
class LogCQT(torch.nn.Module):
    """ch0 признака: лог-мощность на сетке constant-Q, 183 бина.

    cfg=None -> airadar.config.FeatureCfg() по умолчанию, что воспроизводит
    прежнее поведение (глобалы этого модуля остаются для обратной
    совместимости — их использует, например, airadar/bench/feat_visibility.py
    и harmonic.py по прямому импорту, менять их нельзя).

    trainable=False — фронтенд не участвует в обратном проходе, как и
    нынешний ручной LogMel (его mel-банк тоже не обучается).
    """

    def __init__(self, cfg=None):
        super().__init__()
        from airadar.config import FeatureCfg
        self.cfg = cfg or FeatureCfg()
        from nnAudio.features import CQT2010v2
        self._cqt = CQT2010v2(
            sr=self.cfg.sr, hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin, fmax=self.cfg.fmax,
            n_bins=self.cfg.n_bins, bins_per_octave=self.cfg.bins_per_octave,
            output_format="Magnitude", trainable=False, verbose=False,
        )
```

`forward`/`frequencies` не меняются — они не читают глобалы напрямую.

- [ ] **Step 4: Проверить, что старый селфчек `cqt.py` не сломался**

Run: `python airadar/features/cqt.py --selfcheck`
Expected: `cqt selfcheck ok` (без изменений в поведении — `LogCQT()` без
аргумента строит тот же CQT, что и раньше)

- [ ] **Step 5: Завести `FeatureCfg` в `Frontend` (`airadar/features/frontend.py`)**

Заменить `__init__` и `forward` (`last_model_frames` и `MODEL_FRAMES` НЕ
трогать — докстринг файла явно отделяет политику длины окна от вычисления
признака, эта задача — только про сетку признака):

```python
class Frontend(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        from airadar.config import FeatureCfg
        self.cfg = cfg or FeatureCfg()
        self._logcqt = LogCQT(self.cfg)

    def forward(self, wav):
        ch0 = self._logcqt(wav)                                       # [B, F, T]
        ch1 = ch0 - rolling_percentile_causal(
            ch0, self.cfg.bg_window_frames, self.cfg.bg_quantile)
        return torch.stack([ch0, ch1], dim=1)                         # [B, 2, F, T]
```

- [ ] **Step 6: Проверить, что старый селфчек `frontend.py` не сломался**

Run: `python airadar/features/frontend.py --selfcheck`
Expected: `frontend selfcheck ok`

- [ ] **Step 7: Проверить сборку с нестандартным `FeatureCfg` вручную**

```bash
python -c "
from airadar.config import FeatureCfg
from airadar.features.frontend import Frontend
import torch
cfg = FeatureCfg(sr=16000, hop_s=0.064)   # вдвое чаще кадры
fe = Frontend(cfg)
out = fe(torch.zeros(1, round(4.0*16000)))
print(out.shape)   # ожидается вдвое больше кадров, чем стандартные 32
assert out.shape[-1] > 32
print('custom FeatureCfg ok')
"
```
Expected: `custom FeatureCfg ok`, число кадров > 32 (подтверждает, что
`hop_s` реально доходит до `CQT2010v2`, а не игнорируется).

- [ ] **Step 8: Прогнать общий свип селфчеков**

Run: `python cli/selfcheck.py`
Expected: все модули зелёные, счётчик не уменьшился.

- [ ] **Step 9: Commit**

```bash
git add airadar/config.py airadar/features/cqt.py airadar/features/frontend.py
git commit -m "config: FeatureCfg как источник параметров LogCQT/Frontend"
```

---

### Task 2: `airadar/data/f0label.py` + `cli/label_manifest_f0.py`

**Files:**
- Create: `airadar/data/f0label.py`
- Create: `cli/label_manifest_f0.py`

**Interfaces:**
- Consumes: `airadar.data.clips.ClipReader` (этап 1, `read(offset,
  n_samples) -> np.ndarray[float32]`), реальный `data/manifest.parquet` +
  `data/clips.bin` (уже собраны — 191196 строк, `cli/manifest_audit.py`
  зелёный).
- Produces: `airadar.data.f0label.f0_salience_lfenergy(w) -> (f0_hz: float,
  salience_db: float, lf_energy: float)`, `airadar.data.f0label.label_row(
  reader, offset, n_samples) -> tuple|None`. Колонки `f0_med, salience,
  lf_energy` манифеста (уже в схеме `airadar/data/manifest.py`, сейчас
  всегда `None` — эта задача единственная их заполняет).

Манифест (§5.2) резервирует `f0_med, salience, lf_energy` как «посчитано
один раз», но этап 1 (manifest-clipstore) их не заполнял — колонки есть,
значения `None`. Вспомогательным головам (§3.4) нужны реальные метки f0 на
клип, иначе их не на чем обучать. Алгоритм не изобретается заново — это
прямой перенос HPS-оценщика (гармоническая сумма лог-спектра) из
`evalx/f0_survey.py`, но источником становится `ClipReader` по строкам
манифеста вместо старого кэша окон `cache_dads`. Как и в `evalx/f0_survey.py`,
считается по ВСЕМ строкам (не только `label=1`): трудные негативы с гребёнкой
в полосе 50–110 Гц — готовый пул самых трудных негативов (§5.4), это
работало и на старом кэше, и работает так же на новом.

- [ ] **Step 1: Написать `airadar/data/f0label.py`**

```python
"""f0/salience/lf_energy по манифесту: перенос evalx/f0_survey.py (HPS в
лог-домене) на строки манифеста через ClipReader, вместо старого кэша окон
cache_dads. Алгоритм не меняется — меняется только источник аудио.

Одно значение на клип, не на окно: центральное окно длиной WIN=8000
отсчётов (0.5 с, тот же WIN, что в evalx/f0_survey.py). Клипы короче
WIN — f0_med/salience/lf_energy остаются None. У реальных источников такое
не встречалось (D0, этап 0: минимальная длина позитива DADS 0.6с=9600>WIN),
но проверка есть, чтобы не падать, а не потому что ожидается сработать.
"""
import sys
import numpy as np

SR = 16000
WIN = 8000
NFFT = 32768
F0_LO, F0_HI, F0_STEP = 40.0, 400.0, 0.5
NHARM = 8


def _cand_index():
    f0 = np.arange(F0_LO, F0_HI + 1e-9, F0_STEP)
    k = np.arange(1, NHARM + 1)
    idx = np.rint(f0[:, None] * k[None, :] * NFFT / SR).astype(np.int64)
    idx = np.clip(idx, 0, NFFT // 2)
    return f0, idx


F0_CAND, CAND_IDX = _cand_index()
_WINDOW = np.hanning(WIN).astype(np.float32)
_FREQS = np.fft.rfftfreq(NFFT, 1 / SR)
_LO_MASK = _FREQS < 300.0
_BAND_MASK = (_FREQS >= 40.0) & (_FREQS <= 4000.0)


def f0_salience_lfenergy(w):
    """w: [WIN] float32, пик нормирован (см. airadar/data/clips.py) ->
    (f0_hz, salience_db, lf_energy). lf_energy — доля энергии ниже 300 Гц
    в полосе 40-4000 Гц, тот же признак, что evalx/f0_survey.py:blo."""
    x = np.asarray(w, dtype=np.float32)
    x = x - x.mean()
    P = np.abs(np.fft.rfft(x * _WINDOW, n=NFFT)) ** 2 + 1e-12
    L = 10.0 * np.log10(P)
    S = L[CAND_IDX].mean(axis=1)                 # [n_cand] гармоническая сумма
    j = int(S.argmax())
    f0 = float(F0_CAND[j])
    salience = float(S.max() - np.median(S))
    lf_energy = float(P[_LO_MASK].sum() / (P[_BAND_MASK].sum() + 1e-12))
    return f0, salience, lf_energy


def label_row(reader, offset, n_samples):
    """Возвращает (f0, salience, lf_energy) для строки манифеста, или None,
    если клип короче WIN — центральное окно взять неоткуда."""
    if n_samples < WIN:
        return None
    start = offset + (n_samples - WIN) // 2
    w = reader.read(start, WIN)
    return f0_salience_lfenergy(w)


def selfcheck():
    import tempfile
    import os
    from airadar.data.clips import ClipWriter, ClipReader

    t = np.arange(WIN, dtype=np.float32) / SR
    tone = 0.5 * np.sin(2 * np.pi * 120.0 * t).astype(np.float32)   # f0=120Гц
    for k in range(2, NHARM + 1):
        tone += (0.5 / k) * np.sin(2 * np.pi * 120.0 * k * t).astype(np.float32)

    f0, sal, lf = f0_salience_lfenergy(tone)
    assert abs(f0 - 120.0) < 1.0, f0            # оценщик находит гармонику
    assert sal > 6.0, sal                       # выраженная гребёнка

    silence = np.zeros(WIN, dtype=np.float32)
    _, sal_s, _ = f0_salience_lfenergy(silence)
    assert np.isfinite(sal_s)                   # не падает на тишине

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clips.bin")
        with ClipWriter(path) as w:
            off_long, n_long = w.write(np.tile(tone, 3))               # длиннее WIN
            off_short, n_short = w.write(np.zeros(100, dtype=np.float32))  # короче WIN
        with ClipReader(path) as r:
            got = label_row(r, off_long, n_long)
            assert got is not None
            assert abs(got[0] - 120.0) < 1.0, got

            assert label_row(r, off_short, n_short) is None   # короткий клип пропущен

    print("f0label selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать селфчек**

Run: `python airadar/data/f0label.py --selfcheck`
Expected: `f0label selfcheck ok`

- [ ] **Step 3: Написать `cli/label_manifest_f0.py`**

```python
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
```

- [ ] **Step 4: Проверить на куске реального манифеста**

Run: `python cli/label_manifest_f0.py --limit 500`
Expected: печатает `готово: 500 строк обработано, N пропущено`, без ошибок.
Реальный `data/manifest.parquet` (191196 строк, `cli/manifest_audit.py`
зелёный) уже собран — можно проверять сразу на нём, не на синтетике.

- [ ] **Step 5: Проверить, что колонки реально записались**

```bash
python -c "
import pyarrow.parquet as pq
t = pq.read_table('data/manifest.parquet')
f0 = t.column('f0_med').to_pylist()
non_null = sum(1 for v in f0[:500] if v is not None)
print(f'{non_null}/500 строк получили f0_med')
assert non_null > 400, non_null   # почти все клипы длиннее WIN=8000
print('f0_med populated ok')
"
```
Expected: `f0_med populated ok`

- [ ] **Step 6: Запустить на всём манифесте (полный прогон, CPU, без GPU)**

Run: `python cli/label_manifest_f0.py`
Expected: `готово: 191196 строк обработано, N пропущено` — минуты, не часы
(191k окон по 0.5с, FFT на CPU). Список: если `N пропущено` заметно
отличается от нуля — записать это как наблюдение в резюме задачи, не
блокирующее (короткие клипы существуют и это не ошибка).

- [ ] **Step 7: Commit**

```bash
git add airadar/data/f0label.py cli/label_manifest_f0.py
git commit -m "data: f0_med/salience/lf_energy в манифесте — перенос HPS-оценщика с cache_dads на clips.bin"
```

Файл `data/manifest.parquet` не коммитится (в `.gitignore`), коммитятся
только код и логи прогона, если решите их сохранить.

---

### Task 3: `airadar/features/harmonic.py` — harmonic stacking

**Files:**
- Create: `airadar/features/harmonic.py`

**Interfaces:**
- Consumes: `airadar.features.cqt.BINS_PER_OCTAVE` (=24), `airadar.features
  .cqt.N_BINS` (=183) — уже существующие модульные константы этапа 2.
- Produces: `airadar.features.harmonic.N_F0_BINS` (=80),
  `airadar.features.harmonic.harmonic_stack(x: Tensor[B,C,N_BINS,T]) ->
  Tensor[B, C*8, N_F0_BINS, T]`.

Индексы гармоник (`round(24·log2 k)` для `k=1..8`): `[0, 24, 38, 48, 56, 62,
67, 72]`. Максимальный используемый индекс в исходной 183-бинной сетке —
`(80-1) + 72 = 151 < 183`: гармоники физически не выходят за сетку CQT ни
при каком f0-бине диапазона 40–400 Гц (проверено вручную при написании
этого плана; Step 2 ниже проверяет то же самое кодом).

- [ ] **Step 1: Написать `airadar/features/harmonic.py`**

```python
"""Harmonic stacking (§3.1): для f0-бина i собираются бины
i + round(24·log2 k), k=1..8 — чистое индексирование, вычислений ноль.

Смысл: на лог-частотной оси гармоника k эквидистантна от основной частоты
НЕЗАВИСИМО от самой f0 (self-similarity лог-оси) — тот же принцип, что делает
f0-сдвиг (§4.1) чистой трансляцией. round(24·log2(k)) — это смещение в
бинах между гармониками k и 1 на сетке 24 бина/октаву; для f0-бина i индекс
k-й гармоники — i + это смещение, всегда, вне зависимости от i.
"""
import sys
import numpy as np
import torch

from airadar.features.cqt import BINS_PER_OCTAVE, N_BINS

F0_LO, F0_HI = 40.0, 400.0
N_HARMONICS = 8
N_F0_BINS = round(BINS_PER_OCTAVE * np.log2(F0_HI / F0_LO))   # 80, см. §3.1

_OFFSETS = [round(BINS_PER_OCTAVE * np.log2(k)) for k in range(1, N_HARMONICS + 1)]
_MAX_INDEX = (N_F0_BINS - 1) + max(_OFFSETS)
assert _MAX_INDEX < N_BINS, (
    f"harmonic stacking выходит за сетку CQT: индекс {_MAX_INDEX} >= {N_BINS}")


def harmonic_stack(x):
    """x: [B, C, N_BINS, T] (лог-CQT, любое число каналов C) ->
    [B, C*N_HARMONICS, N_F0_BINS, T].

    Канал (c, k) на f0-бине i — это x[:, c, i + _OFFSETS[k], :]."""
    B, C, F, T = x.shape
    assert F == N_BINS, (F, N_BINS)
    idx = torch.tensor(
        [[i + off for off in _OFFSETS] for i in range(N_F0_BINS)],
        device=x.device, dtype=torch.long)          # [N_F0_BINS, N_HARMONICS]
    gathered = x[:, :, idx, :]                       # [B, C, N_F0_BINS, N_HARMONICS, T]
    gathered = gathered.permute(0, 1, 3, 2, 4)        # [B, C, N_HARMONICS, N_F0_BINS, T]
    return gathered.reshape(B, C * N_HARMONICS, N_F0_BINS, T)


def selfcheck():
    assert N_F0_BINS == 80, N_F0_BINS
    assert _OFFSETS == [0, 24, 38, 48, 56, 62, 67, 72], _OFFSETS

    # уникальное значение на (channel, freq_bin) паре -> из выхода можно
    # однозначно восстановить, какой исходный бин был взят на gather
    B, C, T = 1, 2, 4
    x = torch.zeros(B, C, N_BINS, T)
    for c in range(C):
        for f in range(N_BINS):
            x[0, c, f, :] = c * 1000 + f            # кодирует (c, f), время не участвует

    out = harmonic_stack(x)
    assert out.shape == (B, C * N_HARMONICS, N_F0_BINS, T), out.shape

    # проверка адресации: канал (c=1, k=5 индекс 4 -> offset 56), f0-бин i=10
    # обязан читать исходный бин f = 10 + 56 = 66, значение 1*1000+66=1066
    c, k_idx, i = 1, 4, 10
    out_channel = c * N_HARMONICS + k_idx
    expected = c * 1000 + (i + _OFFSETS[k_idx])
    assert out[0, out_channel, i, 0].item() == expected, \
        (out[0, out_channel, i, 0].item(), expected)

    # k=1 (offset 0) на f0-бине i обязан совпасть с исходным ch0 на том же бине
    for i in (0, 40, 79):
        assert out[0, 0, i, 0].item() == 0 * 1000 + i

    # градиент доходит до исходного тензора через gather (не detached)
    x2 = torch.randn(1, 2, N_BINS, T, requires_grad=True)
    harmonic_stack(x2).sum().backward()
    assert x2.grad is not None and torch.any(x2.grad != 0)

    print("harmonic selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать селфчек**

Run: `python airadar/features/harmonic.py --selfcheck`
Expected: `harmonic selfcheck ok` (включая проверку `_MAX_INDEX < N_BINS` на
уровне импорта модуля — если бы она провалилась, модуль не импортировался
бы вовсе).

- [ ] **Step 3: Commit**

```bash
git add airadar/features/harmonic.py
git commit -m "features: harmonic stacking (ветка A) — индексирование по round(24*log2 k)"
```

---

### Task 4: `airadar/models/branches.py` — BranchA, BranchB

**Files:**
- Create: `airadar/models/branches.py`
- Create: `airadar/models/__init__.py` (пустой, как `airadar/features/__init__.py`)

**Interfaces:**
- Consumes: `airadar.features.harmonic.N_F0_BINS` (=80),
  `airadar.features.harmonic.N_HARMONICS` (=8), `airadar.config.ModelCfg`
  (`branch_hidden`, по умолчанию 128).
- Produces: `airadar.models.branches.BranchA` — `forward(stacked:
  Tensor[B, 16, 80, T]) -> (frame_logit: Tensor[B,T], evidence:
  Tensor[B,80,T], f0_idx: Tensor[B,T] long)`. `airadar.models.branches.
  BranchB` — `forward(x: Tensor[B,2,183,T]) -> frame_logit: Tensor[B,T]`.

Свёртки без страйда (`stride=1`, `padding=(k//2, 1)`) — размер по времени
`T` сохраняется точно на всех слоях обеих веток, это нужно Task 5/6 для
слияния покадровых логитов веток A и B поэлементно.

- [ ] **Step 1: Написать `airadar/models/__init__.py`**

Пустой файл (как `airadar/features/__init__.py`, `airadar/data/__init__.py`).

- [ ] **Step 2: Написать `airadar/models/branches.py`**

```python
"""Ветка A (гармоническая) и ветка B (текстурная), §3.1/§3.2.

Обе схлопывают частотную ось в покадровое свидетельство через logsumexp —
не max (чувствителен к одному пику, шумно) и не mean (топит слабую
гребёнку в широкой полосе). logsumexp — гладкая аппроксимация max,
дифференцируема всюду, и на практике ведёт себя как "мягкий OR" по бинам:
если хотя бы один бин на кадре t горячий, кадр горячий.

Обе ветки stride=1 по обеим осям — размер по времени T сохраняется точно,
это нужно для поэлементного слияния логитов веток (airadar/models/dronenet2.py).
"""
import sys
import torch
import torch.nn as nn

from airadar.features.harmonic import N_F0_BINS, N_HARMONICS


class BranchA(nn.Module):
    """Вход: [B, C*N_HARMONICS, N_F0_BINS, T] (после harmonic_stack).
    C=2 (ch0, ch1) -> вход 16 каналов."""

    def __init__(self, in_channels=2 * N_HARMONICS, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, stacked):
        evidence = self.conv(stacked).squeeze(1)         # [B, N_F0_BINS, T]
        frame_logit = torch.logsumexp(evidence, dim=1)    # [B, T]
        f0_idx = evidence.argmax(dim=1)                   # [B, T], диагностика
        return frame_logit, evidence, f0_idx


class BranchB(nn.Module):
    """Вход: [B, 2, N_BINS, T] — полная сетка CQT, оба канала (ch0, ch1).

    Первый слой — более широкое ядро по частоте (7 против 5 у ветки A):
    ловит широкополосный лопастной шум малых FPV, где гребёнка слабее
    относительно шума ротора (§3.2), а не узкую гармонику."""

    def __init__(self, in_channels=2, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        evidence = self.conv(x).squeeze(1)                # [B, N_BINS, T]
        frame_logit = torch.logsumexp(evidence, dim=1)     # [B, T]
        return frame_logit


def selfcheck():
    B, T = 2, 32
    a = BranchA()
    stacked = torch.randn(B, 2 * N_HARMONICS, N_F0_BINS, T)
    frame_logit, evidence, f0_idx = a(stacked)
    assert frame_logit.shape == (B, T), frame_logit.shape
    assert evidence.shape == (B, N_F0_BINS, T), evidence.shape
    assert f0_idx.shape == (B, T), f0_idx.shape
    assert f0_idx.dtype == torch.long
    assert (f0_idx >= 0).all() and (f0_idx < N_F0_BINS).all()

    b = BranchB()
    x = torch.randn(B, 2, 183, T)
    frame_logit_b = b(x)
    assert frame_logit_b.shape == (B, T), frame_logit_b.shape

    # T сохраняется точно на других длинах, не только 32 — Frontend может
    # отдать больше кадров (12с -> 94), ветки обязаны не падать
    for T2 in (5, 94):
        stacked2 = torch.randn(1, 2 * N_HARMONICS, N_F0_BINS, T2)
        fl, ev, fi = a(stacked2)
        assert fl.shape == (1, T2) and ev.shape == (1, N_F0_BINS, T2)
        x2 = torch.randn(1, 2, 183, T2)
        assert b(x2).shape == (1, T2)

    # градиент доходит до входа обеих веток
    stacked.requires_grad_(True)
    a(stacked)[0].sum().backward()
    assert stacked.grad is not None and torch.any(stacked.grad != 0)

    x.requires_grad_(True)
    b(x).sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0)

    print("branches selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 3: Прогнать селфчек**

Run: `python airadar/models/branches.py --selfcheck`
Expected: `branches selfcheck ok`

- [ ] **Step 4: Commit**

```bash
git add airadar/models/__init__.py airadar/models/branches.py
git commit -m "models: BranchA (гармоническая) и BranchB (текстурная), logsumexp по частоте"
```

---

### Task 5: `airadar/models/mil.py` — attention-MIL агрегация

**Files:**
- Create: `airadar/models/mil.py`

**Interfaces:**
- Produces: `airadar.models.mil.AttentionMIL` — `__init__(in_dim: int,
  hidden: int = 16)`, `forward(frame_feat: Tensor[B,T,in_dim], frame_value:
  Tensor[B,T]) -> (clip_logit: Tensor[B], attn: Tensor[B,T])`.
- Consumes: ничего из предыдущих задач напрямую (принимает произвольные
  `frame_feat`/`frame_value` — используется в Task 6 с `in_dim=2`).

Формула — gated attention (Ilse, Tomczak, Welling, "Attention-based Deep
Multiple Instance Learning", 2018): `score_t = w^T (tanh(V h_t) ⊙
sigmoid(U h_t))`, `attn = softmax(score)` по кадрам, `clip_logit = Σ_t
attn_t · value_t`. Это стандартная, проверенная параметризация MIL-пулинга,
а не изобретение с нуля — спецификация (§3.3) описывает поведение
("обучаемая взвешенная сумма покадровых логитов"), не формулу; выбрана
эта, потому что gate (`tanh · sigmoid`) даёт сети механизм подавлять кадры,
а не только их взвешивать положительно, что важно, когда цель слышна не
всё время (ровно случай, описанный в §3.3).

- [ ] **Step 1: Написать `airadar/models/mil.py`**

```python
"""Attention-MIL (§3.3): клип-логит — обучаемая взвешенная сумма покадровых
логитов. Заменяет правило "k из m" полностью (правило удалено из системы).

Формула — gated attention (Ilse et al., 2018): score_t = w^T(tanh(V h_t) *
sigmoid(U h_t)), attn = softmax(score) по кадрам, clip_logit = Σ attn_t *
value_t. Gate (tanh * sigmoid) даёт сети возможность подавлять кадр, а не
только взвешивать положительно — нужно, когда цель слышна не всё время.
"""
import sys
import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.U = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, frame_feat, frame_value):
        # frame_feat: [B,T,in_dim], frame_value: [B,T]
        gate = torch.tanh(self.V(frame_feat)) * torch.sigmoid(self.U(frame_feat))
        score = self.w(gate).squeeze(-1)              # [B,T]
        attn = torch.softmax(score, dim=-1)            # [B,T], сумма=1 по кадрам
        clip_logit = (attn * frame_value).sum(dim=-1)  # [B]
        return clip_logit, attn


def selfcheck():
    B, T, D = 3, 32, 2
    mil = AttentionMIL(in_dim=D, hidden=16)
    feat = torch.randn(B, T, D)
    value = torch.randn(B, T)

    clip_logit, attn = mil(feat, value)
    assert clip_logit.shape == (B,), clip_logit.shape
    assert attn.shape == (B, T), attn.shape
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (attn >= 0).all()                     # softmax -> неотрицательно

    # clip_logit обязан лежать в выпуклой оболочке value (attn суммируется в
    # 1 и неотрицателен) — простая проверка на здравый смысл формулы
    assert (clip_logit >= value.min(dim=-1).values - 1e-4).all()
    assert (clip_logit <= value.max(dim=-1).values + 1e-4).all()

    # если один кадр несёт всё "свидетельство" (остальные -100), attn должен
    # сосредоточиться на нём почти полностью после обучения -- здесь только
    # проверяем, что градиент течёт в обе стороны (feat и value), а не что
    # attn уже выучен (веса случайные)
    feat.requires_grad_(True)
    value.requires_grad_(True)
    clip_logit2, _ = mil(feat, value)
    clip_logit2.sum().backward()
    assert feat.grad is not None and torch.any(feat.grad != 0)
    assert value.grad is not None and torch.any(value.grad != 0)

    print("mil selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать селфчек**

Run: `python airadar/models/mil.py --selfcheck`
Expected: `mil selfcheck ok`

- [ ] **Step 3: Commit**

```bash
git add airadar/models/mil.py
git commit -m "models: attention-MIL (gated, Ilse 2018) — замена правилу k из m"
```

---

### Task 6: `airadar/models/heads.py` + `airadar/models/dronenet2.py` — сборка модели

**Files:**
- Create: `airadar/models/heads.py`
- Create: `airadar/models/dronenet2.py`

**Interfaces:**
- Consumes: `airadar.features.harmonic.{harmonic_stack, N_F0_BINS}`,
  `airadar.models.branches.{BranchA, BranchB}`, `airadar.models.mil.
  AttentionMIL`, `airadar.config.ModelCfg`.
- Produces: `airadar.models.heads.AuxHead` — `forward(evidence:
  Tensor[B,N_F0_BINS,T]) -> (f0_hat: Tensor[B,T] Hz, salience_hat:
  Tensor[B,T] dB)`. `airadar.models.dronenet2.DroneNet2` —
  `forward(feat: Tensor[B,2,183,T]) -> dict` с ключами `clip_logit [B]`,
  `attn [B,T]`, `f0_hat [B,T]`, `salience_hat [B,T]`, `f0_idx [B,T]`.

`AuxHead` не имеет обучаемых параметров: f0 — ожидание по распределению
`softmax(evidence)` над сеткой частот f0-бинов (дифференцируемо), salience —
`max - median` по кандидатам, буквально то же определение, что в
`evalx/f0_survey.py`/`airadar/data/f0label.py` (Task 2), чтобы метка и
предсказание были в одной системе координат. Это заставляет саму карту
`evidence` ветки A быть тем, чем она должна быть (§3.4: «регуляризация,
заставляющая пользоваться гребёнкой»), не добавляя веса, которые могли бы
научиться считать что-то postороннее.

- [ ] **Step 1: Написать `airadar/models/heads.py`**

```python
"""Вспомогательная голова f0/salience (§3.4). Не продукт — в рантайме не
используется, только при обучении как регуляризация.

Без своих обучаемых параметров: f0 — softmax-взвешенное ожидание по карте
evidence ветки A (дифференцируемо), salience — max минус медиана по f0-
бинам, то же определение, что в airadar/data/f0label.py, чтобы предсказание
и метка (манифест, Task 2) были в одной системе координат.
"""
import sys
import torch
import torch.nn as nn

from airadar.features.harmonic import N_F0_BINS, F0_LO, F0_HI
from airadar.features.cqt import BINS_PER_OCTAVE


class AuxHead(nn.Module):
    def __init__(self):
        super().__init__()
        f0_grid = F0_LO * 2.0 ** (torch.arange(N_F0_BINS, dtype=torch.float32)
                                   / BINS_PER_OCTAVE)
        self.register_buffer("f0_grid", f0_grid)   # [N_F0_BINS], Гц

    def forward(self, evidence):
        # evidence: [B, N_F0_BINS, T] от BranchA
        p = torch.softmax(evidence, dim=1)                        # [B,N_F0_BINS,T]
        f0_hat = (p * self.f0_grid[None, :, None]).sum(dim=1)     # [B,T], Гц
        salience_hat = evidence.max(dim=1).values - evidence.median(dim=1).values
        return f0_hat, salience_hat


def selfcheck():
    B, T = 2, 32
    head = AuxHead()
    assert head.f0_grid.shape == (N_F0_BINS,)
    assert abs(head.f0_grid[0].item() - F0_LO) < 0.1
    assert head.f0_grid[-1].item() <= F0_HI * 1.05   # верхний край около F0_HI

    evidence = torch.randn(B, N_F0_BINS, T)
    f0_hat, salience_hat = head(evidence)
    assert f0_hat.shape == (B, T) and salience_hat.shape == (B, T)
    # f0_hat -- выпуклая комбинация f0_grid, обязана лежать в [F0_LO, F0_HI]
    assert (f0_hat >= F0_LO - 1e-3).all() and (f0_hat <= F0_HI + 1e-3).all()

    # выраженный пик на конкретном f0-бине -> f0_hat близко к этому бину
    evidence_peaked = torch.full((1, N_F0_BINS, 1), -10.0)
    evidence_peaked[0, 40, 0] = 10.0
    f0_hat_p, _ = head(evidence_peaked)
    assert abs(f0_hat_p.item() - head.f0_grid[40].item()) < 5.0, f0_hat_p.item()

    # градиент доходит до evidence
    evidence.requires_grad_(True)
    f0_hat2, sal2 = head(evidence)
    (f0_hat2.sum() + sal2.sum()).backward()
    assert evidence.grad is not None and torch.any(evidence.grad != 0)

    print("heads selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать селфчек `heads.py`**

Run: `python airadar/models/heads.py --selfcheck`
Expected: `heads selfcheck ok`

- [ ] **Step 3: Написать `airadar/models/dronenet2.py`**

```python
"""Сборка модели целиком (§3): ветка A + ветка B + слияние + attention-MIL
+ вспомогательная голова. Единственная точка входа модели — как
Frontend для признака (этап 2).

Размер (§3.5): 400 000-800 000 параметров. При branch_hidden=128 (ModelCfg
по умолчанию) — посчитано вручную при написании этого плана: BranchA
~277k, BranchB ~252k, слияние+MIL ~116 -> ~529.5k, в требуемом диапазоне.
Step 4 ниже проверяет это кодом на реальной модели, не по расчёту на бумаге.
"""
import sys
import torch
import torch.nn as nn

from airadar.config import ModelCfg
from airadar.features.harmonic import harmonic_stack
from airadar.models.branches import BranchA, BranchB
from airadar.models.mil import AttentionMIL
from airadar.models.heads import AuxHead


class DroneNet2(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelCfg()
        self.branch_a = BranchA(hidden=self.cfg.branch_hidden)
        self.branch_b = BranchB(hidden=self.cfg.branch_hidden)
        self.fuse = nn.Linear(2, 1)
        self.mil = AttentionMIL(in_dim=2, hidden=self.cfg.mil_hidden)
        self.aux = AuxHead()

    def forward(self, feat):
        # feat: [B, 2, 183, T] — выход Frontend.last_model_frames (T=32 в
        # обычном режиме; ветки и MIL не требуют конкретного T).
        stacked = harmonic_stack(feat)                    # [B, 16, 80, T]
        frame_logit_a, evidence_a, f0_idx = self.branch_a(stacked)
        frame_logit_b = self.branch_b(feat)                # [B, T]

        frame_feat = torch.stack([frame_logit_a, frame_logit_b], dim=-1)  # [B,T,2]
        frame_value = self.fuse(frame_feat).squeeze(-1)                   # [B,T]
        clip_logit, attn = self.mil(frame_feat, frame_value)

        f0_hat, salience_hat = self.aux(evidence_a)

        return {
            "clip_logit": clip_logit,       # [B] — основной выход (логит вероятности дрона)
            "attn": attn,                   # [B,T] — веса MIL, диагностика
            "f0_hat": f0_hat,               # [B,T], Гц — вспом. голова
            "salience_hat": salience_hat,   # [B,T], дБ — вспом. голова
            "f0_idx": f0_idx,               # [B,T] long — трек f0 ветки A, диагностика
        }


def selfcheck():
    B, T = 2, 32
    model = DroneNet2()
    feat = torch.randn(B, 2, 183, T)
    out = model(feat)

    assert set(out) == {"clip_logit", "attn", "f0_hat", "salience_hat", "f0_idx"}
    assert out["clip_logit"].shape == (B,)
    assert out["attn"].shape == (B, T)
    assert out["f0_hat"].shape == (B, T)
    assert out["salience_hat"].shape == (B, T)
    assert out["f0_idx"].shape == (B, T)

    assert torch.allclose(out["attn"].sum(dim=-1), torch.ones(B), atol=1e-4)
    assert (out["f0_hat"] >= 40.0 - 1e-3).all() and (out["f0_hat"] <= 400.0 + 1e-3).all()

    # не падает на других T (94 -- полный выход Frontend на 12с, не только
    # обрезанные 32 кадра модели)
    feat94 = torch.randn(1, 2, 183, 94)
    out94 = model(feat94)
    assert out94["clip_logit"].shape == (1,)
    assert out94["attn"].shape == (1, 94)

    # градиент из clip_logit доходит до входа -- весь граф связан, ни одна
    # ветка не оторвана (частый класс багов: диагностический выход
    # случайно отделён detach()'ем или недифференцируемой операцией на
    # магистральном пути)
    feat.requires_grad_(True)
    model(feat)["clip_logit"].sum().backward()
    assert feat.grad is not None and torch.any(feat.grad != 0)

    # размер модели (§3.5): 400k-800k параметров
    n_params = sum(p.numel() for p in model.parameters())
    assert 400_000 <= n_params <= 800_000, n_params
    print(f"параметров: {n_params}")

    print("dronenet2 selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 4: Прогнать селфчек, проверить реальный размер модели**

Run: `python airadar/models/dronenet2.py --selfcheck`
Expected: `параметров: <N>` с `400000 <= N <= 800000` (ожидается ~529500 по
ручному расчёту в докстринге — если реальное число сильно разойдётся,
проверить, не разъехались ли `padding`/`kernel_size` в `branches.py`), затем
`dronenet2 selfcheck ok`.

- [ ] **Step 5: Прогнать общий свип селфчеков, поднять `MIN_CHECKS`**

Run: `python cli/selfcheck.py`
Expected: `прогнано 25/25 модулей, освобождено 1`.

Реальное текущее число модулей с `selfcheck` на момент написания этого
плана (до Task 1) — 18 (девять в `bench/`, четыре в `data/`, два в
`diag/`, три в `features/`; проверено `grep -rl "^def selfcheck" airadar/`,
а не докстрингом `cli/selfcheck.py` — его текст устарел ещё после этапа 2:
по-прежнему пишет "восемь в bench/... тринадцать всего", хотя `MIN_CHECKS`
всегда был нижней границей, не точным числом). Этот план добавляет 7
новых модулей с `selfcheck`: `config.py` (Task 1), `f0label.py` (Task 2,
пятый в `data/`), `harmonic.py` (Task 3, четвёртый в `features/`),
`branches.py`, `mil.py`, `heads.py`, `dronenet2.py` (Task 4-6, четыре в
`models/` — у `models/__init__.py` своего `selfcheck` нет, как и у прочих
`__init__.py` пакета). Итого 18 + 7 = 25.

Поднять `MIN_CHECKS` в `cli/selfcheck.py` с `13` до `25`, переписать
комментарий над ним: «девять в bench/, пять в data/, два в diag/, четыре в
features/, четыре в models/, один airadar/config.py — двадцать пять».

Затем перезапустить:

Run: `python cli/selfcheck.py`
Expected: `прогнано 25/25 модулей, освобождено 1`, код возврата 0.

- [ ] **Step 6: Commit**

```bash
git add airadar/models/heads.py airadar/models/dronenet2.py cli/selfcheck.py
git commit -m "models: DroneNet2 — сборка ветка A + ветка B + attention-MIL + вспом. голова"
```

---

## Что эта задача НЕ делает (сознательно, не забыто)

- **Обучающий цикл, чекпоинт, аугментация (§4).** `DroneNet2` — чистая
  архитектура, проверенная на случайном входе. Обучение (`train/loop.py`,
  `train/checkpoint.py`, `augment/{pitch,acoustic,hum,mixing}.py`, сборка
  тренировочного примера из манифеста с коротким/длинным контекстом,
  MIL-подмешивание коротких DADS-позитивов в фон, 2 seed, сравнение с
  `dronenet_local.pt` по правилу §6.2) — отдельный план, следующий за этим.
- **Реальные потери (loss).** `clip_logit` — логит, готовый для
  `BCEWithLogitsLoss`; `f0_hat`/`salience_hat` — готовы для маскированной
  регрессии там, где `f0_med`/`salience` манифеста не `None` (после Task 2).
  Веса вспомогательных потерь, порядок обучения — решения обучающего цикла.
  `f0label.py` также используется бенчем на новых данных (worst f0-band,
  §6.1) вместо старого `evalx/f0_dads_*.npz`, но это тоже интеграция
  следующего плана, не этого.
