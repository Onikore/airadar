# Аугментация + сборка обучающего примера — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Основная аугментация §4 спецификации (f0-сдвиг фиксированным
`r`, гул, SNR-подмешивание, гейн, циклический сдвиг, SpecAugment,
затухание верхов — не дрейф f0/доплер и не синтетическая гребёнка, см.
раздел «что не делает» в конце плана) как независимые тестируемые функции
на массивах, плюс `airadar/train/sampler.py`, который собирает ОДИН
обучающий пример (сырое аудио + метка) из строки манифеста и `clips.bin`
(этап 1) — единственная точка, где решается, что делать с коротким
позитивом DADS (86% данных, D0: не смежны физически) против длинного
непрерывного позитивом DAS.

**Architecture:** Аугментация разбита на слой сырого аудио (f0-сдвиг,
гул, SNR-подмешивание, циклический сдвиг, гейн — `airadar/augment/`) и
слой признака (SpecAugment, затухание верхов — применяются ПОСЛЕ
`Frontend`, этап 2, потому что действуют на лог-мощность CQT, не на
звук). `sampler.py` — оркестратор слоя сырого аудио: по строке манифеста
(`label`, `offset`, `n_samples`) решает режим сборки (длинный клип
целиком / клип средней длины как есть / короткий клип на случайном
офсете внутри фона) и применяет f0-сдвиг + SNR + гул + гейн + циклический
сдвиг в правильном порядке.

**Tech Stack:** NumPy, SciPy (`resample_poly`, уже используется в
`hf_sources.py`). Продолжает `airadar/data/{manifest,clips}.py` (этап 1),
`airadar/config.py` (этап 3а).

## Global Constraints

- Порядок аугментации позитива (§4.1, обязателен): сначала f0-сдвиг
  чистого сигнала, потом подмешивание НЕсдвинутого реального фона.
- f0-сдвиг: `r ∈ [0.35, 1.5]` (§4.1). Понижение f0 удлиняет клип
  (`r=0.35`: 200→70 Гц, 0.6с→1.71с) — побочный эффект, не баг.
- SNR: `-15…+20 дБ`, записывается в пример как поле (§4.2, было `-6…+15`).
- Пиковая нормализация убрана; уровень несёт `ch1` (этап 2) + случайный
  гейн (§4.2).
- Гул ЛЭП: амплитуда до `0.8` (было `0.25`), расстройка `49.8–50.2 Гц`,
  «только гул» — отдельный трудный негатив (§4.2).
- Циклический сдвиг, SpecAugment — без изменений по сути, перенесены на
  новую сетку признака (§4.2).
- Затухание верхов `exp(-k·f·d)` — без изменений (§4.2), см. Task 4 про
  точку применения на новом фронтенде.
- Каждый модуль имеет `--selfcheck` (§8, конвенция проекта).
- Целевой размер модуля — до ~200 строк (§8).
- `config.py` — единственный источник диапазонов аугментации, чекпоинт
  (этап 3в) сериализует его целиком (§8, инвариант, продолжение этапа 3а).

---

### Task 1: `AugCfg`/`TrainCfg` в `airadar/config.py`

**Files:**
- Modify: `airadar/config.py` (добавить `AugCfg`, `TrainCfg`, дополнить `selfcheck`)

**Interfaces:**
- Produces: `airadar.config.AugCfg` (frozen dataclass: `pitch_r_lo,
  pitch_r_hi, pitch_prob, snr_db_lo, snr_db_hi, gain_db_lo, gain_db_hi,
  hum_amp_max, hum_f0_lo, hum_f0_hi, hum_prob, hum_only_prob, air_k_max,
  spec_mask_n, spec_mask_frac`), `airadar.config.TrainCfg` (frozen
  dataclass: `model_samples=64000, target_samples=192000`).

Диапазоны — значения из спецификации §4 (см. Global Constraints).
`hum_prob`/`hum_only_prob`/`pitch_prob` спецификацией численно не заданы
(она описывает эффект, не частоту применения) — выбраны как разумные
инженерные значения (`hum_prob=0.3`, `hum_only_prob=0.05`,
`pitch_prob=0.7`) и явно помечены в докстринге как решение этого плана, а
не перенос из спецификации, чтобы не выглядеть заимствованным числом,
которого нет в тексте §4.

- [ ] **Step 1: Дописать `AugCfg`/`TrainCfg` в `airadar/config.py`**

Добавить после `ModelCfg`:

```python
@dataclass(frozen=True)
class AugCfg:
    """Диапазоны аугментации (§4). pitch_prob/hum_prob/hum_only_prob —
    инженерное решение этого плана (спецификация задаёт диапазоны и
    эффекты, не частоту применения)."""
    pitch_r_lo: float = 0.35
    pitch_r_hi: float = 1.5
    pitch_prob: float = 0.7          # доля позитивов со сдвинутым f0
    snr_db_lo: float = -15.0
    snr_db_hi: float = 20.0
    gain_db_lo: float = -6.0
    gain_db_hi: float = 6.0
    hum_amp_max: float = 0.8
    hum_f0_lo: float = 49.8
    hum_f0_hi: float = 50.2
    hum_prob: float = 0.3            # доля примеров с подмешанным гулом
    hum_only_prob: float = 0.05      # доля НЕГАТИВОВ, заменяемых на чистый гул
    air_k_max: float = 2.5
    spec_mask_n: int = 2
    spec_mask_frac: float = 1.0 / 6.0


@dataclass(frozen=True)
class TrainCfg:
    """Длины окна сборки примера, отсчёты при 16 кГц. target_samples —
    8с истории (BG_WINDOW_FRAMES) + 4с окна модели (MODEL_FRAMES), см.
    airadar/features/frontend.py, этап 2. model_samples — минимум, ниже
    которого Frontend.last_model_frames не наберёт MODEL_FRAMES кадров."""
    model_samples: int = 64000    # 4.0с
    target_samples: int = 192000  # 12.0с
```

- [ ] **Step 2: Дополнить `selfcheck` в `airadar/config.py`**

Добавить перед `print("config selfcheck ok")`:

```python
    ac = AugCfg()
    assert ac.pitch_r_lo == 0.35 and ac.pitch_r_hi == 1.5
    assert ac.snr_db_lo == -15.0 and ac.snr_db_hi == 20.0
    assert ac.hum_amp_max == 0.8

    tc = TrainCfg()
    assert tc.model_samples == 64000 and tc.target_samples == 192000
```

- [ ] **Step 3: Прогнать селфчек**

Run: `python airadar/config.py --selfcheck`
Expected: `config selfcheck ok`

- [ ] **Step 4: Commit**

```bash
git add airadar/config.py
git commit -m "config: AugCfg/TrainCfg — диапазоны аугментации и длины окна сборки примера"
```

---

### Task 2: `airadar/augment/pitch.py` — f0-сдвиг

**Files:**
- Create: `airadar/augment/__init__.py` (пустой)
- Create: `airadar/augment/pitch.py`

**Interfaces:**
- Produces: `airadar.augment.pitch.sample_r(rng, cfg=None) -> float`,
  `airadar.augment.pitch.pitch_shift(wav: np.ndarray[N] float32, r: float,
  sr: int = 16000) -> np.ndarray[N'] float32`.

Ресемплинг `resample_poly(wav, up, down)` даёт `len(wav)·up/down`
отсчётов. Чтобы f0 умножилась на `r` (а длительность — на `1/r`, см.
Global Constraints), нужно `up/down = 1/r`: дробь `up=10000,
down=round(10000·r)`, сокращённая через `gcd` — та же схема, что
`hf_sources.to_mono_16k` уже использует для смены частоты дискретизации.

- [ ] **Step 1: Написать `airadar/augment/__init__.py`**

Пустой файл.

- [ ] **Step 2: Написать `airadar/augment/pitch.py`**

```python
"""f0-сдвиг позитивов (§4.1): ресемплинг с коэффициентом r ∈ [0.35, 1.5]
размазывает f0 обучающей массы по диапазону 40-400 Гц. На лог-оси это
ровно трансляция (тот же принцип, что делает harmonic stacking
f0-независимым, см. airadar/features/harmonic.py).

Порядок обязателен (§4.1): сдвигается ЧИСТЫЙ позитив, ДО подмешивания
фона (airadar/augment/mixing.py, airadar/train/sampler.py) — иначе вместе
с целью сдвинется и фон, которому сдвигаться не с чего.
"""
import sys
from math import gcd
import numpy as np
from scipy.signal import resample_poly


def sample_r(rng, cfg=None):
    from airadar.config import AugCfg
    cfg = cfg or AugCfg()
    return float(rng.uniform(cfg.pitch_r_lo, cfg.pitch_r_hi))


def pitch_shift(wav, r):
    """wav: [N] float32 -> [round(N/r)] float32. r<1 понижает f0 и
    удлиняет клип (r=0.35: 200Гц->70Гц, 0.6с->1.71с); r>1 — наоборот."""
    up = 10000
    down = round(10000 * r)
    g = gcd(up, down)
    up, down = up // g, down // g
    return resample_poly(np.asarray(wav, dtype=np.float32), up, down).astype(np.float32)


def selfcheck():
    sr = 16000
    t = np.arange(round(0.6 * sr), dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)   # f0=200Гц, квадрокоптер

    shifted = pitch_shift(tone, 0.35)
    # длительность: 0.6с/0.35 = 1.714с (§4.1, пример из спецификации)
    assert abs(len(shifted) / sr - 1.714) < 0.02, len(shifted) / sr

    # f0 реально сдвинулась на r: ищем пик спектра, ожидаем ~70 Гц
    spec = np.abs(np.fft.rfft(shifted * np.hanning(len(shifted))))
    freqs = np.fft.rfftfreq(len(shifted), 1 / sr)
    f0_hat = freqs[spec.argmax()]
    assert abs(f0_hat - 70.0) < 3.0, f0_hat

    # r=1.0 -> длина не меняется (с точностью до округления рационального
    # приближения gcd(10000,10000)=10000 -> up=down=1)
    unchanged = pitch_shift(tone, 1.0)
    assert len(unchanged) == len(tone), (len(unchanged), len(tone))

    # sample_r: диапазон соблюдён, детерминирован при фиксированном seed
    rng = np.random.default_rng(0)
    rs = [sample_r(rng) for _ in range(200)]
    assert all(0.35 <= r <= 1.5 for r in rs)
    assert min(rs) < 0.5 and max(rs) > 1.3   # диапазон реально используется целиком

    print("pitch selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 3: Прогнать селфчек**

Run: `python -m airadar.augment.pitch --selfcheck`
Expected: `pitch selfcheck ok`

- [ ] **Step 4: Commit**

```bash
git add airadar/augment/__init__.py airadar/augment/pitch.py
git commit -m "augment: f0-сдвиг (pitch_shift) — ресемплинг по коэффициенту r"
```

---

### Task 3: `airadar/augment/hum.py` — гул ЛЭП

**Files:**
- Create: `airadar/augment/hum.py`

**Interfaces:**
- Produces: `airadar.augment.hum.make_hum(n_samples: int, sr: int, rng) ->
  np.ndarray[n_samples] float32` (амплитуда ~[-1,1] до масштабирования),
  `airadar.augment.hum.add_hum(wav: np.ndarray[N] float32, rng, amp_max:
  float = 0.8, sr: int = 16000) -> np.ndarray[N] float32`.

Гармонический состав (`1.0, 0.5, 0.35, 0.2` на гармониках `1..4`) и
случайная фаза на гармонику — перенесены из `train.py:Augment.forward`
без изменений (это не то, что §4.2 просит менять). Что меняется — верхняя
граница амплитуды (`0.25`→`0.8`) и добавленная расстройка `49.8–50.2 Гц`
вместо ровно `50.0`.

- [ ] **Step 1: Написать `airadar/augment/hum.py`**

```python
"""Гул ЛЭП: 50 Гц + гармоники (§4.2). Амплитуда до 0.8 (было 0.25 — при
разрешении CQT 1.172 Гц на 40 Гц гул отделим от гребёнки дрона, топить
его сильнее незачем, см. этап 2, feat_visibility). Расстройка 49.8-50.2
Гц — реальная сеть не держит частоту идеально ровно 50.0.

Гармонический состав и случайная фаза перенесены из train.py:Augment без
изменений — §4.2 просит изменить амплитуду и добавить расстройку, не
переизобретать сам гул.
"""
import sys
import numpy as np

_HARMONICS = ((1, 1.0), (2, 0.5), (3, 0.35), (4, 0.2))


def make_hum(n_samples, sr, rng, f0_lo=49.8, f0_hi=50.2):
    f0 = rng.uniform(f0_lo, f0_hi)
    t = np.arange(n_samples, dtype=np.float64) / sr
    hum = np.zeros(n_samples, dtype=np.float64)
    for k, w in _HARMONICS:
        phase = rng.uniform(0, 2 * np.pi)
        hum += w * np.sin(2 * np.pi * f0 * k * t + phase)
    return hum.astype(np.float32)


def add_hum(wav, rng, amp_max=0.8, sr=16000):
    wav = np.asarray(wav, dtype=np.float32)
    hum = make_hum(len(wav), sr, rng)
    amp = rng.uniform(0.0, amp_max)
    scale = amp * (np.abs(wav).max() + 1e-8)
    return wav + hum * scale


def selfcheck():
    sr = 16000
    rng = np.random.default_rng(0)
    hum = make_hum(sr, sr, rng)   # 1с
    assert hum.shape == (sr,)
    assert np.isfinite(hum).all()

    # пик спектра рядом с 50 Гц (внутри расстройки 49.8-50.2)
    spec = np.abs(np.fft.rfft(hum * np.hanning(len(hum))))
    freqs = np.fft.rfftfreq(len(hum), 1 / sr)
    f0_hat = freqs[spec.argmax()]
    assert 49.0 <= f0_hat <= 51.0, f0_hat   # разрешение FFT на 1с ~1Гц, допуск шире расстройки

    # расстройка реально варьируется между вызовами, не зафиксирована на
    # 50.0 — на 1с окне разрешение FFT ровно 1Гц (>= ширины расстройки
    # 0.4Гц), все черновики округлились бы в один и тот же бин. Нужно
    # окно длиннее: 20с -> разрешение 0.05Гц, восьмикратный запас
    dur_long = 20 * sr
    freqs_long = np.fft.rfftfreq(dur_long, 1 / sr)
    f0s = []
    for _ in range(20):
        h = make_hum(dur_long, sr, np.random.default_rng())
        s = np.abs(np.fft.rfft(h * np.hanning(len(h))))
        f0s.append(round(freqs_long[s.argmax()], 2))
    assert len(set(f0s)) > 1, ("расстройка должна варьироваться", f0s)

    wav = np.sin(2 * np.pi * 200.0 * np.arange(sr, dtype=np.float32) / sr)
    out = add_hum(wav, rng, amp_max=0.8, sr=sr)
    assert out.shape == wav.shape
    assert np.isfinite(out).all()
    assert not np.allclose(out, wav)   # гул реально что-то добавил

    print("hum selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать селфчек**

Run: `python -m airadar.augment.hum --selfcheck`
Expected: `hum selfcheck ok`

- [ ] **Step 3: Commit**

```bash
git add airadar/augment/hum.py
git commit -m "augment: гул ЛЭП — амплитуда 0.8, расстройка 49.8-50.2 Гц"
```

---

### Task 4: `airadar/augment/acoustic.py` — затухание верхов, циклический сдвиг, SpecAugment

**Files:**
- Create: `airadar/augment/acoustic.py`

**Interfaces:**
- Consumes: ничего из предыдущих задач (независимый модуль).
- Produces: `airadar.augment.acoustic.cyclic_shift(wav: np.ndarray[N]
  float32, rng) -> np.ndarray[N] float32` (аудио, до фронтенда).
  `airadar.augment.acoustic.apply_air_absorption(ch0: Tensor[B,F,T],
  freqs: array[F], k: Tensor[B]) -> Tensor[B,F,T]` (признак, после
  `Frontend`). `airadar.augment.acoustic.spec_augment(feat: Tensor[2,F,T],
  rng, n_masks=2, max_frac=1/6) -> Tensor[2,F,T]` (признак, один пример,
  маска одинакова для обоих каналов).

`apply_air_absorption` действует ТОЛЬКО на `ch0`, не на весь `[2,F,T]`
тензор: старый `LogMel` затухание тоже применял до вычитания фона
(которого в старом фронтенде не было). На новом фронтенде наклон,
постоянный по времени внутри примера, вычитается сам при вычислении
`ch1 = ch0 - каузальный_перцентиль(ch0)` — значит эффект остаётся только
в `ch0`, что и требуется: `ch1` создан убирать именно такую стационарную
широкополосную окраску (см. риск R3 спецификации), а `ch0` — канал,
которому положено видеть сырую картину целиком.

- [ ] **Step 1: Написать `airadar/augment/acoustic.py`**

```python
"""Затухание верхов (air absorption) — без изменений по сути (§4.2),
перенесено на новый CQT-признак. Циклический сдвиг и SpecAugment — тоже
без изменений, перенесены на новую форму [2, F, T] (было [1, M, T]).

Затухание в старом LogMel применялось на СЫРОЙ мощности спектра до
логарифма: power *= exp(-k*f/1000)^2. Новый фронтенд (этап 2) отдаёт уже
log(power) — то же действие в лог-домене: log(power*att^2) = log(power) +
2*log(att) = log(power) - 2*k*f/1000. Применяется ТОЛЬКО к ch0 (см.
докстринг задачи в плане): наклон, постоянный по времени внутри примера,
и так вычитается при вычислении ch1.
"""
import sys
import numpy as np
import torch


def cyclic_shift(wav, rng):
    """wav: [N] float32 -> циклически сдвинутый на случайный офсет.
    Модель не должна цепляться за абсолютную позицию события в окне."""
    wav = np.asarray(wav, dtype=np.float32)
    shift = int(rng.integers(0, len(wav)))
    return np.roll(wav, shift)


def apply_air_absorption(ch0, freqs, k):
    """ch0: [B, F, T] лог-мощность (обычноканал 0 выхода Frontend).
    freqs: [F] Гц (LogCQT.frequencies). k: [B] коэффициент затухания
    (0 = нет затухания, air_k_max = сильное) -> [B, F, T]."""
    freqs_t = torch.as_tensor(freqs, dtype=ch0.dtype, device=ch0.device)   # [F]
    k_t = torch.as_tensor(k, dtype=ch0.dtype, device=ch0.device)
    tilt = (2.0 * k_t / 1000.0)[:, None, None]                            # [B,1,1]
    return ch0 - tilt * freqs_t[None, :, None]


def spec_augment(feat, rng, n_masks=2, max_frac=1.0 / 6.0):
    """feat: [2, F, T] (ch0, ch1) -> с замаскированными полосами частот и
    времени. Маска ОДИНАКОВА для обоих каналов: физически "эта часть
    записи пропала", не "пропала только в одном представлении"."""
    feat = feat.clone()
    _, F, T = feat.shape
    for _ in range(n_masks):
        f = int(rng.integers(0, max(1, int(F * max_frac)) + 1))
        f0 = int(rng.integers(0, max(1, F - f + 1)))
        feat[:, f0:f0 + f, :] = 0.0
        t = int(rng.integers(0, max(1, int(T * max_frac)) + 1))
        t0 = int(rng.integers(0, max(1, T - t + 1)))
        feat[:, :, t0:t0 + t] = 0.0
    return feat


def selfcheck():
    rng = np.random.default_rng(0)

    wav = np.arange(100, dtype=np.float32)
    shifted = cyclic_shift(wav, rng)
    assert shifted.shape == wav.shape
    assert set(shifted.tolist()) == set(wav.tolist())   # те же значения, другой порядок
    assert not np.array_equal(shifted, wav) or len(set(rng.integers(0, 100, 5))) == 1

    B, F, T = 3, 183, 32
    freqs = np.linspace(40.0, 8000.0, F).astype(np.float32)
    ch0 = torch.zeros(B, F, T)
    k = torch.tensor([0.0, 1.0, 2.5])
    out = apply_air_absorption(ch0, freqs, k)
    assert out.shape == (B, F, T)
    assert torch.allclose(out[0], ch0[0])           # k=0 -> без изменений
    # эффект растёт с частотой: высокий бин ослаблен больше низкого при k>0
    assert (out[1, -1, 0] < out[1, 0, 0]).item()
    assert (out[1, -1, 0] > out[2, -1, 0]).item()    # k=2.5 ослабляет сильнее k=1.0

    feat = torch.ones(2, 183, 32)
    masked = spec_augment(feat, rng)
    assert masked.shape == feat.shape
    zero_ch0 = (masked[0] == 0)
    zero_ch1 = (masked[1] == 0)
    assert torch.equal(zero_ch0, zero_ch1)          # маска одинакова на обоих каналах
    assert zero_ch0.any()                            # хоть что-то замаскировано
    assert not zero_ch0.all()                         # не всё замаскировано

    print("acoustic selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать селфчек**

Run: `python -m airadar.augment.acoustic --selfcheck`
Expected: `acoustic selfcheck ok`

- [ ] **Step 3: Commit**

```bash
git add airadar/augment/acoustic.py
git commit -m "augment: затухание верхов (ch0 only), циклический сдвиг, SpecAugment на [2,F,T]"
```

---

### Task 5: `airadar/augment/mixing.py` — SNR-подмешивание, гейн, размещение на офсете

**Files:**
- Create: `airadar/augment/mixing.py`

**Interfaces:**
- Produces: `airadar.augment.mixing.snr_scale(signal: np.ndarray,
  background: np.ndarray, snr_db: float) -> float`,
  `airadar.augment.mixing.mix_at_snr(signal: np.ndarray[N] float32,
  background: np.ndarray[N] float32, snr_db: float) -> np.ndarray[N]
  float32`, `airadar.augment.mixing.random_gain(wav: np.ndarray[N]
  float32, rng, lo=-6.0, hi=6.0) -> np.ndarray[N] float32`,
  `airadar.augment.mixing.place_at_offset(short: np.ndarray[n] float32,
  canvas_len: int, rng) -> (canvas: np.ndarray[canvas_len] float32,
  offset: int)`.

`snr_scale` вынесена из `mix_at_snr` отдельной функцией: сборщику примера
(Task 6) нужно посчитать масштаб фона по ЛОКАЛЬНОМУ участку (короткий
позитив против фона ровно под ним), а применить этот же масштаб ко ВСЕЙ
канве фона — иначе «SNR» короткого события, вложенного в основном нулевую
канву, считался бы относительно средней мощности почти-тишины и не имел
бы смысла как реальный SNR момента, когда дрон слышен.

- [ ] **Step 1: Написать `airadar/augment/mixing.py`**

```python
"""Смешивание позитива и фона при заданном SNR (§4.1/§4.2). Пиковая
нормализация убрана: раньше `wav /= abs(wav).max()` привязывала масштаб
ВСЕГО примера к самой громкой отдельной помехе внутри окна (щелчок,
всплеск) — сеть заново училась абсолютной громкости на каждом примере
вместо относительной. Теперь уровень несёт ch1 (вычитание фона, этап 2)
и random_gain (аугментация масштаба, не нормализация — значения могут
выйти за [-1,1], это осознанно, см. план)."""
import sys
import numpy as np


def snr_scale(signal, background, snr_db):
    sig_p = np.mean(np.asarray(signal, dtype=np.float64) ** 2) + 1e-12
    bg_p = np.mean(np.asarray(background, dtype=np.float64) ** 2) + 1e-12
    return float(np.sqrt(sig_p / (bg_p * 10 ** (snr_db / 10.0))))


def mix_at_snr(signal, background, snr_db):
    """signal, background: [N] float32, одинаковой длины -> signal +
    масштабированный background, дающий заданный SNR относительно
    ВСЕГО signal (для короткого позитива на офсете внутри канвы см.
    snr_scale + airadar/train/sampler.py — там сигнал и локальный фон
    короче полной канвы, а масштаб потом применяется к канве целиком)."""
    scale = snr_scale(signal, background, snr_db)
    return (np.asarray(signal, dtype=np.float32)
            + np.asarray(background, dtype=np.float32) * scale)


def random_gain(wav, rng, lo=-6.0, hi=6.0):
    gain_db = rng.uniform(lo, hi)
    return (np.asarray(wav, dtype=np.float32) * 10 ** (gain_db / 20.0)).astype(np.float32)


def place_at_offset(short, canvas_len, rng):
    """short: [n] float32, n <= canvas_len -> (canvas [canvas_len] float32
    с short на случайном офсете поверх нулей, offset int)."""
    short = np.asarray(short, dtype=np.float32)
    n = len(short)
    assert n <= canvas_len, (n, canvas_len)
    offset = int(rng.integers(0, canvas_len - n + 1))
    canvas = np.zeros(canvas_len, dtype=np.float32)
    canvas[offset:offset + n] = short
    return canvas, offset


def selfcheck():
    rng = np.random.default_rng(0)

    sig = np.ones(1000, dtype=np.float32)
    bg = np.ones(1000, dtype=np.float32) * 2.0
    scale = snr_scale(sig, bg, 0.0)   # 0 дБ -> равные мощности после масштаба
    scaled_bg_power = np.mean((bg * scale) ** 2)
    assert abs(scaled_bg_power - np.mean(sig ** 2)) < 1e-6, scaled_bg_power

    mixed = mix_at_snr(sig, bg, 0.0)
    assert mixed.shape == sig.shape
    assert np.allclose(mixed, sig + bg * scale)

    # выше SNR -> меньше вклад фона
    scale_hi = snr_scale(sig, bg, 20.0)
    scale_lo = snr_scale(sig, bg, -15.0)
    assert scale_hi < scale_lo

    g = random_gain(sig, rng, lo=-6.0, hi=6.0)
    ratio_db = 20 * np.log10(np.abs(g[0]) / np.abs(sig[0]))
    assert -6.0 - 1e-3 <= ratio_db <= 6.0 + 1e-3, ratio_db

    short = np.arange(1, 11, dtype=np.float32)
    canvas, offset = place_at_offset(short, 100, rng)
    assert canvas.shape == (100,)
    assert np.array_equal(canvas[offset:offset + 10], short)
    assert canvas[:offset].sum() == 0 and canvas[offset + 10:].sum() == 0

    print("mixing selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать селфчек**

Run: `python -m airadar.augment.mixing --selfcheck`
Expected: `mixing selfcheck ok`

- [ ] **Step 3: Commit**

```bash
git add airadar/augment/mixing.py
git commit -m "augment: SNR-подмешивание (snr_scale/mix_at_snr), random_gain, place_at_offset"
```

---

### Task 6: `airadar/train/sampler.py` — сборка обучающего примера

**Files:**
- Create: `airadar/train/__init__.py` (пустой)
- Create: `airadar/train/sampler.py`

**Interfaces:**
- Consumes: `airadar.data.clips.ClipReader`, `airadar.config.{AugCfg,
  TrainCfg}`, `airadar.augment.pitch.{sample_r, pitch_shift}`,
  `airadar.augment.hum.add_hum`, `airadar.augment.mixing.{snr_scale,
  mix_at_snr, random_gain, place_at_offset}`,
  `airadar.augment.acoustic.cyclic_shift`.
- Produces: `airadar.train.sampler.draw_background(neg_pool:
  np.ndarray[K,2] int64, reader: ClipReader, length: int, rng) ->
  np.ndarray[length] float32`, `airadar.train.sampler.assemble_example(
  row: dict, reader: ClipReader, neg_pool: np.ndarray[K,2] int64, rng,
  aug_cfg=None, train_cfg=None) -> (wav: np.ndarray[N] float32, label:
  int, meta: dict)`.

`row` — словарь с как минимум `offset, n_samples, label` (строка
манифеста, `airadar/data/manifest.py:SCHEMA`). `neg_pool` — заранее
отфильтрованный (`label=0`, `split=train`) массив `(offset, n_samples)`
из манифеста; собирается ОДИН раз при старте обучения вызывающим кодом
(этап 3в, `train/loop.py`), не на каждый вызов `assemble_example` — иначе
каждый пример требовал бы полного скана манифеста.

Инвариант: `assemble_example` всегда возвращает `len(wav) >=
train_cfg.model_samples` (64000) — короче не бывает ни при каком входе,
это гарантирует, что `Frontend.last_model_frames` (этап 2) никогда не
упадёт на реальном обучающем примере, только на намеренно некорректном
входе (её собственный селфчек это уже проверяет).

- [ ] **Step 1: Написать `airadar/train/__init__.py`**

Пустой файл.

- [ ] **Step 2: Написать `airadar/train/sampler.py`**

```python
"""Сборка одного обучающего примера из строки манифеста (этап 1) —
единственное место, где решается, что делать с коротким позитивом DADS
(86% данных, D0: клипы не смежны физически, см. спецификация §5.4)
против длинного непрерывного позитива DAS.

Три режима по длине клипа:
  - длинный (>= target_samples, 12с): случайное окно нужной длины прямо
    из клипа — контекста достаточно для полного 8с+4с окна модели.
  - средний (>= model_samples, 4с, но короче 12с): клип целиком — короче
    идеала, но Frontend/last_model_frames справляется (даёт меньше
    "истории" для ch1, не меньше кадров модели).
  - короткий (< model_samples): клип кладётся на случайный офсет внутри
    канвы длиной model_samples, остальное — фон при случайном SNR.
    Деградационная ветка D0 (этап 0): "позитив 0.6с кладётся в случайное
    место 4-секундного фона" — применена буквально.

Для негативов (label=0) режимы длины те же, но без вложения на офсет
(негатив негативен целиком, вкладывать некуда) — только выбор окна
нужной длины из его собственного аудио, с зацикливанием, если аудио
короче model_samples.
"""
import sys
import numpy as np

from airadar.augment.pitch import sample_r, pitch_shift
from airadar.augment.hum import add_hum
from airadar.augment.mixing import snr_scale, mix_at_snr, random_gain, place_at_offset
from airadar.augment.acoustic import cyclic_shift


def draw_background(neg_pool, reader, length, rng):
    """neg_pool: [K,2] int64 (offset, n_samples) строк label=0 -> [length]
    float32, случайный негатив нужной длины (зацикленный, если короче)."""
    i = int(rng.integers(0, len(neg_pool)))
    offset, n = int(neg_pool[i, 0]), int(neg_pool[i, 1])
    audio = reader.read(offset, n)
    if n >= length:
        start = int(rng.integers(0, n - length + 1))
        return audio[start:start + length].astype(np.float32)
    reps = -(-length // n)   # ceil division — зацикливание, шов не сглаживается
    return np.tile(audio, reps)[:length].astype(np.float32)


def _own_window(audio, length, rng):
    """audio: [n] float32 -> [length] float32: случайное окно, если
    audio длиннее length, иначе зацикленное audio."""
    n = len(audio)
    if n >= length:
        start = int(rng.integers(0, n - length + 1))
        return audio[start:start + length].astype(np.float32)
    reps = -(-length // n)
    return np.tile(audio, reps)[:length].astype(np.float32)


def assemble_example(row, reader, neg_pool, rng, aug_cfg=None, train_cfg=None):
    from airadar.config import AugCfg, TrainCfg
    aug_cfg = aug_cfg or AugCfg()
    train_cfg = train_cfg or TrainCfg()

    label = int(row["label"])
    meta = {"snr_db": None, "pitch_r": None, "mode": None, "hum_added": False}

    if label == 1:
        pos = reader.read(row["offset"], row["n_samples"]).astype(np.float32)
        if rng.random() < aug_cfg.pitch_prob:
            r = sample_r(rng, aug_cfg)
            pos = pitch_shift(pos, r)
            meta["pitch_r"] = r
        n = len(pos)

        if n >= train_cfg.target_samples:
            meta["mode"] = "long"
            start = int(rng.integers(0, n - train_cfg.target_samples + 1))
            signal = pos[start:start + train_cfg.target_samples]
            bg = draw_background(neg_pool, reader, len(signal), rng)
            snr_db = float(rng.uniform(aug_cfg.snr_db_lo, aug_cfg.snr_db_hi))
            wav = mix_at_snr(signal, bg, snr_db)
            meta["snr_db"] = snr_db
        elif n >= train_cfg.model_samples:
            meta["mode"] = "medium"
            signal = pos
            bg = draw_background(neg_pool, reader, len(signal), rng)
            snr_db = float(rng.uniform(aug_cfg.snr_db_lo, aug_cfg.snr_db_hi))
            wav = mix_at_snr(signal, bg, snr_db)
            meta["snr_db"] = snr_db
        else:
            meta["mode"] = "short"
            canvas_len = train_cfg.model_samples
            bg_full = draw_background(neg_pool, reader, canvas_len, rng)
            canvas, offset = place_at_offset(pos, canvas_len, rng)
            meta["offset"] = offset
            snr_db = float(rng.uniform(aug_cfg.snr_db_lo, aug_cfg.snr_db_hi))
            # масштаб фона считается ПО ЛОКАЛЬНОМУ участку (позитив против
            # фона ровно под ним), применяется ко всей канве фона —
            # см. докстринг airadar/augment/mixing.py:snr_scale
            local_bg = bg_full[offset:offset + len(pos)]
            scale = snr_scale(pos, local_bg, snr_db)
            wav = canvas + bg_full * scale
            meta["snr_db"] = snr_db
    else:
        if rng.random() < aug_cfg.hum_only_prob:
            meta["mode"] = "hum_only"
            wav = np.zeros(train_cfg.model_samples, dtype=np.float32)
        else:
            meta["mode"] = "negative"
            neg = reader.read(row["offset"], row["n_samples"]).astype(np.float32)
            # тот же трёхуровневый выбор длины, что у позитивов выше: клип
            # средней длины используется целиком, а не обрезается до
            # model_samples — иначе доступный контекст терялся бы без причины
            if row["n_samples"] >= train_cfg.target_samples:
                length = train_cfg.target_samples
            elif row["n_samples"] >= train_cfg.model_samples:
                length = row["n_samples"]
            else:
                length = train_cfg.model_samples
            wav = _own_window(neg, length, rng)

    # универсальная пост-обработка сырого аудио, для обеих меток
    wav = cyclic_shift(wav, rng)
    if meta["mode"] == "hum_only" or rng.random() < aug_cfg.hum_prob:
        wav = add_hum(wav, rng, amp_max=aug_cfg.hum_amp_max)
        meta["hum_added"] = True
    wav = random_gain(wav, rng, lo=aug_cfg.gain_db_lo, hi=aug_cfg.gain_db_hi)

    assert len(wav) >= train_cfg.model_samples, (len(wav), train_cfg.model_samples)
    return wav, label, meta


def selfcheck():
    import tempfile
    import os
    from airadar.data.clips import ClipWriter, ClipReader
    from airadar.config import AugCfg, TrainCfg

    sr = 16000
    tc = TrainCfg()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clips.bin")
        with ClipWriter(path) as w:
            # длинный позитив (12с+) — режим "long"
            long_pos = np.random.default_rng(1).standard_normal(tc.target_samples + 1000).astype(np.float32) * 0.1
            off_long, n_long = w.write(long_pos)
            # средний позитив (между model_samples и target_samples)
            med_pos = np.random.default_rng(2).standard_normal(tc.model_samples + 5000).astype(np.float32) * 0.1
            off_med, n_med = w.write(med_pos)
            # короткий позитив (0.6с, как DADS) — режим "short"
            short_pos = np.random.default_rng(3).standard_normal(round(0.6 * sr)).astype(np.float32) * 0.1
            off_short, n_short = w.write(short_pos)
            # негативы для neg_pool — разной длины
            neg_offsets = []
            for i in range(20):
                neg = np.random.default_rng(100 + i).standard_normal(tc.model_samples).astype(np.float32) * 0.05
                o, n = w.write(neg)
                neg_offsets.append((o, n))

        neg_pool = np.array(neg_offsets, dtype=np.int64)

        with ClipReader(path) as reader:
            rng = np.random.default_rng(42)
            # pitch_prob=0 здесь: длина после сдвига f0 умножается на 1/r,
            # r в [0.35, 1.5] может перекинуть клип через границу
            # длинный/средний/короткий — эти два случая проверяют именно
            # выбор режима ПО ДЛИНЕ, сдвиг проверен отдельно в pitch.py
            no_pitch = AugCfg(pitch_prob=0.0)

            row_long = {"offset": off_long, "n_samples": n_long, "label": 1}
            wav, label, meta = assemble_example(row_long, reader, neg_pool, rng,
                                                aug_cfg=no_pitch)
            assert label == 1 and meta["mode"] == "long"
            assert len(wav) >= tc.model_samples
            assert np.isfinite(wav).all()

            row_med = {"offset": off_med, "n_samples": n_med, "label": 1}
            wav, label, meta = assemble_example(row_med, reader, neg_pool, rng,
                                                aug_cfg=no_pitch)
            assert meta["mode"] == "medium"
            assert len(wav) >= tc.model_samples

            row_short = {"offset": off_short, "n_samples": n_short, "label": 1}
            wav, label, meta = assemble_example(row_short, reader, neg_pool, rng)
            assert meta["mode"] == "short"
            assert len(wav) == tc.model_samples   # короткий позитив -> ровно окно модели
            assert 0 <= meta["offset"] <= tc.model_samples - n_short

            row_neg = {"offset": neg_offsets[0][0], "n_samples": neg_offsets[0][1], "label": 0}
            wav, label, meta = assemble_example(row_neg, reader, neg_pool, rng)
            assert label == 0
            assert len(wav) >= tc.model_samples

            # hum_only реально срабатывает при hum_only_prob=1.0
            cfg_always_hum = AugCfg(hum_only_prob=1.0)
            wav, label, meta = assemble_example(row_neg, reader, neg_pool, rng,
                                                aug_cfg=cfg_always_hum)
            assert meta["mode"] == "hum_only" and meta["hum_added"] is True

            # детерминированность: тот же seed -> тот же результат
            rng_a = np.random.default_rng(7)
            rng_b = np.random.default_rng(7)
            wav_a, _, _ = assemble_example(row_short, reader, neg_pool, rng_a)
            wav_b, _, _ = assemble_example(row_short, reader, neg_pool, rng_b)
            assert np.array_equal(wav_a, wav_b)

    print("sampler selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 3: Прогнать селфчек**

Run: `python -m airadar.train.sampler --selfcheck`
Expected: `sampler selfcheck ok`

- [ ] **Step 4: Проверить на реальном манифесте (не только на синтетике)**

```bash
python -c "
import numpy as np
import pyarrow.parquet as pq
from airadar.data.clips import ClipReader
from airadar.train.sampler import assemble_example

t = pq.read_table('data/manifest.parquet')
tr = t.filter((t.column('split').isin([0])))
neg = tr.filter(tr.column('label') == 0)
neg_pool = np.stack([neg.column('offset').to_numpy(),
                     neg.column('n_samples').to_numpy()], axis=1).astype(np.int64)
pos = tr.filter(tr.column('label') == 1)

rng = np.random.default_rng(0)
modes = {}
with ClipReader('data/clips.bin') as reader:
    for i in range(200):
        row = {'offset': int(pos.column('offset')[i].as_py()),
               'n_samples': int(pos.column('n_samples')[i].as_py()),
               'label': 1}
        wav, label, meta = assemble_example(row, reader, neg_pool, rng)
        assert np.isfinite(wav).all(), (i, meta)
        modes[meta['mode']] = modes.get(meta['mode'], 0) + 1
print('режимы на 200 реальных позитивах:', modes)
print('sampler on real manifest ok')
"
```
Expected: `sampler on real manifest ok`, печатает распределение режимов
(ожидается в основном `short` — 86% позитивов DADS короче 4с, D0).

- [ ] **Step 5: Прогнать общий свип селфчеков**

Run: `python cli/selfcheck.py`
Expected: `прогнано N/N модулей, освобождено 1` — этот план добавил 5
новых модулей с `selfcheck` (`pitch`, `hum`, `acoustic`, `mixing`,
`sampler`) к прежним 25 (этап 3а) -> 30. Поднять `MIN_CHECKS` в
`cli/selfcheck.py` с `25` до `30`, комментарий — «плюс пять в augment/, один
в train/».

- [ ] **Step 6: Commit**

```bash
git add airadar/train/__init__.py airadar/train/sampler.py cli/selfcheck.py
git commit -m "train: sampler.py — сборка обучающего примера (long/medium/short) из манифеста"
```

---

## Что эта задача НЕ делает (сознательно, не забыто)

- **Обучающий цикл, чекпоинт, реальный прогон, сравнение с базой (§6.2,
  §6.3, 2 seed).** `assemble_example` производит `(wav, label, meta)` —
  готово для батчинга и подачи в `Frontend` (этап 2) → `DroneNet2` (этап
  3а). `train/loop.py`, `train/checkpoint.py`, `cli/train.py`, DataLoader
  с балансировкой батча (реальный манифест сильно несбалансирован в
  пользу дрона: train label=1 122857 против label=0 21016), маскированные
  вспомогательные потери по `f0_med`/`salience` манифеста — отдельный
  план, следующий за этим.
- **Балансировка батча.** Эта задача не решает, как часто каждая метка
  встречается в батче — только как выглядит ОДИН пример каждой метки.
  Решение о частоте — за `train/loop.py`.
- **Дрейф f0 / доплер (§4.2, строка «дрейф f0 (медленная модуляция
  коэффициента ресемплинга), доплер»).** Эта задача реализует f0-сдвиг
  ОДНИМ фиксированным `r` на весь пример (Task 2/6) — то, что спецификация
  называет основным приёмом §4.1. Медленная модуляция `r` во времени —
  это уже не единый вызов `resample_poly`, а кусочный/фазовый ресемплинг
  (переменный шаг), заметно более сложная DSP-задача. Сознательно
  отложено как отдельное расширение `pitch.py`, а не втиснуто в этот план
  наспех — зафиксировано здесь, чтобы не потеряться, а не выброшено.
- **Синтетическая гребёнка с рандомизированной физикой как
  дополнительный позитив (§4.2, последняя строка таблицы).** Требует
  отдельного генератора синтетического дрон-подобного сигнала
  (варьируемый спад гармоник, джиттер оборотов) плюс интеграцию с полем
  `synth` манифеста (уже зарезервировано схемой, `airadar/data/manifest.py`)
  и с исключением синтетики из eval — сопоставимый по объёму со всем этим
  планом кусок работы, не расширение `sampler.py`. Отдельная задача.
