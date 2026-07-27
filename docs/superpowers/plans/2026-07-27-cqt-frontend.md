# Этап 2: фронтенд constant-Q через nnAudio — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** реализовать признак проекта — двухканальный constant-Q спектр
(`ch0` лог-мощность, `ch1` с вычтенным скользящим фоном) через `nnAudio`, и
подтвердить измерением, а не предположением, что он действительно разводит
гул ЛЭП и гребёнку тяжёлого дрона, которые в старом mel-признаке были
неотличимы (косинус 0.90).

**Architecture:** `nnAudio.features.CQT2010v2` с `bins_per_octave=24` даёт
всю сетку 40 Гц–8 кГц одним слоем (эмпирически проверено в спецификации
§2 — переменное разрешение по частоте у constant-Q встроено, ручная сшивка
трёх STFT не нужна). Поверх — каузальный скользящий 20-й перцентиль по
логарифму мощности как `ch1`. Валидация — тот же метод, что уже использован
в проекте (`evalx/feat_visibility.py`: синтетические гул и гребёнка,
косинус в признаковом пространстве), применённый к новому фронтенду и
сверенный с задокументированным старым числом.

**Tech Stack:** Python 3.10, `torch` (уже используется), `nnAudio==0.3.4`
(новая зависимость, решение записано в спецификации §2.2), numpy.

## Global Constraints

- `SR = 16000` Гц, моно, float32 в [-1, 1] — как везде в проекте.
- Параметры CQT зафиксированы измерением (спецификация §2, дата 2026-07-27):
  `hop_length=2048` (128 мс), `fmin=40.0`, `fmax=8000.0`, `n_bins=183`,
  `bins_per_octave=24`, `output_format='Magnitude'`, `trainable=False`
  (фронтенд — фиксированное преобразование, как нынешний ручной `LogMel`,
  не участвует в обратном проходе).
- Числа кадров — **измеренные факты для `nnAudio==0.3.4`, не вычисленные
  по формуле** (паддинг библиотеки не тривиален: 4.0 с даёт 32 кадра, хотя
  `4.0/0.128=31.25`): 4.0 с → **32** кадра, 8.0 с → **63** кадра, 12.0 с →
  **94** кадра (не сумма 32+63 — тоже эффект паддинга). Апгрейд `nnAudio`
  обязан перепроверить эти числа заново, не полагаться на них как на
  математический вывод.
- Контекст модели — **последние 32 кадра** (4.0 с) любой посчитанной
  последовательности. Полный вход для обучающего примера — **до 12.0 с**
  (8 с истории фона + 4 с окна модели), короче — можно, паддинга не
  требуется (см. спецификация §2.1, «каузальная оценка расширяющимся
  окном»).
- Конвенция проекта: каждый модуль со своей логикой имеет `selfcheck()`,
  работающий **без сети** (сеть не нужна вообще — `nnAudio` ничего не
  скачивает), запускаемый как `python -m airadar.features.<модуль>
  --selfcheck`. CPU достаточно — GPU не требуется для selfcheck.
- Комментарии объясняют «почему», а не «что». Язык — русский.
- Целевой размер модуля — до ~200 строк.
- `train.py`, `hf_sources.py`, `hub.py` не трогаются — этот план не
  переключает обучение на новый фронтенд, он его строит и проверяет.

## Ссылки

- Спецификация: [docs/superpowers/specs/2026-07-26-architecture-redesign-design.md](../specs/2026-07-26-architecture-redesign-design.md), §2 (включая §2.1–§2.3, переписанные 2026-07-27 по итогам измерения)
- Методология валидации, переиспользуемая: `evalx/feat_visibility.py` (`comb()` — синтетическая гребёнка, `hum_confusion()` — метод сравнения по косинусу)
- Задокументированное старое число для сравнения: `docs/metrics-plan.md` §0.3 — косинус гул/дрон 0.78–0.90 на старом mel-признаке
- Существующий, уже проверенный загрузчик полевых записей: `airadar/bench/corpus.py`, `field_records(pattern="field/drone_video*.wav") -> dict[str, np.ndarray]`

## Структура файлов

| файл | ответственность |
|---|---|
| `airadar/features/__init__.py` | пустой |
| `airadar/features/cqt.py` | обёртка над `CQT2010v2`, `ch0` (лог-мощность) |
| `airadar/features/background.py` | каузальный скользящий перцентиль — `ch1` |
| `airadar/features/frontend.py` | сборка `[ch0, ch1]` — единственная точка входа |
| `airadar/bench/feat_visibility.py` | синтетическая и полевая проверка разводимости гула/дрона в новом признаке |
| `airadar/diag/am_preservation.py` | проверка гипотезы про амплитудную модуляцию лопастей и шаг кадра |

---

### Task 1: CQT-обёртка и `ch0`

**Files:**
- Create: `airadar/features/__init__.py` (пустой), `airadar/features/cqt.py`

**Interfaces:**
- Consumes: `nnAudio.features.CQT2010v2`
- Produces:
  - `SR = 16000`, `HOP_S = 0.128`, `FMIN = 40.0`, `FMAX = 8000.0`, `BINS_PER_OCTAVE = 24`, `N_BINS = 183`
  - `class LogCQT(torch.nn.Module)` — `__init__(self)`, `forward(self, wav: torch.Tensor) -> torch.Tensor` — `wav` формы `[B, N]` или `[N]` (приводится к батчу), возвращает лог-мощность `[B, 183, T]`, `T` — сколько кадров даёт вход (не фиксировано внутри модуля)
  - `LogCQT.frequencies -> np.ndarray` — центры бинов в Гц, длина 183 (для последующего поиска «полосы < 300 Гц» в Task 5)

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Constant-Q через nnAudio: ch0 признака.

Три ручных STFT (16384/4096/1024 отсчётов) плюс разрежённая band-матрица
из первой версии спецификации решали задачу, которую constant-Q закрывает
нативно: Q=f/Δf постоянно ⇒ эффективное окно ∝ 1/f само по себе, без сшивки
тиров. Параметры ниже — не выбор по вкусу, а измеренный факт (спецификация
§2, 2026-07-27): при bins_per_octave=24 шаг на 40 Гц равен 1.172 Гц — ровно
требование `f·(2^(1/24)−1)` на нижнем крае диапазона.
"""

import sys
import numpy as np
import torch

SR = 16000
HOP_S = 0.128
FMIN, FMAX = 40.0, 8000.0
BINS_PER_OCTAVE = 24
N_BINS = 183                     # round(24 * log2(8000/40)) — см. §1.1 спецификации
HOP_LENGTH = round(HOP_S * SR)   # 2048, ровно 128 мс при 16 кГц


def selfcheck():
    cqt = LogCQT()

    # 4.0 с — окно модели. Число кадров ИЗМЕРЕНО для nnAudio==0.3.4, не
    # вычислено по 4.0/0.128=31.25 (паддинг библиотеки нетривиален).
    wav4 = torch.zeros(1, round(4.0 * SR))
    out4 = cqt(wav4)
    assert out4.shape == (1, N_BINS, 32), out4.shape

    # без батч-измерения тоже должно работать (приводится к [1, ...])
    out4_flat = cqt(torch.zeros(round(4.0 * SR)))
    assert out4_flat.shape == (1, N_BINS, 32), out4_flat.shape

    # 12.0 с — полный вход обучающего примера (8с истории + 4с модели)
    wav12 = torch.zeros(1, round(12.0 * SR))
    out12 = cqt(wav12)
    assert out12.shape == (1, N_BINS, 94), out12.shape   # измерено, не 32+63

    # выход — лог-мощность: конечен, не NaN, на тишине не -inf благодаря eps
    assert torch.isfinite(out4).all()

    # частоты бинов: 183 значения, первый — ровно FMIN, монотонно растут
    freqs = cqt.frequencies
    assert len(freqs) == N_BINS
    assert abs(freqs[0] - FMIN) < 0.01, freqs[0]
    assert np.all(np.diff(freqs) > 0), "частоты бинов должны монотонно расти"
    # разрешение на нижнем крае — измеренный факт спецификации §2
    assert abs((freqs[1] - freqs[0]) - 1.172) < 0.01, freqs[1] - freqs[0]

    print("cqt selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.features.cqt --selfcheck`
Expected: FAIL, `NameError: name 'LogCQT' is not defined`

- [ ] **Step 3: Реализовать**

```python
class LogCQT(torch.nn.Module):
    """ch0 признака: лог-мощность на сетке constant-Q, 183 бина.

    trainable=False — фронтенд не участвует в обратном проходе, как и
    нынешний ручной LogMel (его mel-банк тоже не обучается).
    """

    def __init__(self):
        super().__init__()
        from nnAudio.features import CQT2010v2
        self._cqt = CQT2010v2(
            sr=SR, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX,
            n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
            output_format="Magnitude", trainable=False, verbose=False,
        )

    @property
    def frequencies(self):
        return np.asarray(self._cqt.frequencies)

    def forward(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mag = self._cqt(wav)                    # [B, N_BINS, T], линейная магнитуда
        power = mag.pow(2)
        return torch.log(power + 1e-8)
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.features.cqt --selfcheck`
Expected: PASS, `cqt selfcheck ok`

- [ ] **Step 5: Коммит**

```bash
git add airadar/features/__init__.py airadar/features/cqt.py
git commit -m "features: ch0 — лог-мощность через nnAudio CQT2010v2

bins_per_octave=24 даёт всю сетку 40 Гц-8 кГц одним слоем: у constant-Q
переменное эффективное окно по частоте встроено (Q=f/Δf постоянно), ручная
сшивка трёх STFT из первой версии спецификации избыточна. Числа кадров —
измеренный факт для nnAudio==0.3.4, не вычислены по формуле."
```

---

### Task 2: каузальный скользящий перцентиль — `ch1`

**Files:**
- Create: `airadar/features/background.py`

**Interfaces:**
- Consumes: ничего (работает с `torch.Tensor`, не знает про CQT)
- Produces:
  - `rolling_percentile_causal(x: torch.Tensor, window: int, q: float) -> torch.Tensor` — `x` формы `[..., T]`, возвращает то же форму; для кадра `t` берёт перцентиль `q` (0..1) по срезу `x[..., max(0, t-window+1):t+1]`
  - `BG_WINDOW_FRAMES = 63` (8.0 с при шаге 128 мс — измеренное число кадров, см. Task 1)
  - `BG_QUANTILE = 0.20`

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""ch1: ch0 минус каузальный скользящий 20-й перцентиль по полосе.

Каузально и расширяющимся окном — решение спецификации §2.1: на кадре t
перцентиль берётся по всей доступной истории вплоть до BG_WINDOW_FRAMES
кадров назад, а не по фиксированному окну, требующему паддинга. Так
короткие клипы (86% позитивов DADS короче 8 с — находка D0) не нуждаются
в отдельной логике: расширяющееся окно корректно определено с нулевой
предыстории — это тот же режим "детектор только что включился", что и в
реальном поле.
"""

import sys
import torch

BG_WINDOW_FRAMES = 63      # 8.0 с при шаге 128 мс, см. airadar.features.cqt
BG_QUANTILE = 0.20


def selfcheck():
    # монотонно растущий ряд: перцентиль на каждом шаге не может превышать
    # текущее значение (20-й перцентиль растущего ряда — где-то в начале
    # окна, всегда <= x[t])
    x = torch.arange(20, dtype=torch.float32).unsqueeze(0)   # [1, 20]
    p = rolling_percentile_causal(x, window=8, q=BG_QUANTILE)
    assert p.shape == x.shape
    assert (p <= x + 1e-6).all()

    # константный ряд: перцентиль равен самой константе на любом кадре,
    # включая самый первый (окно длиной 1 — единственное значение)
    c = torch.full((1, 10), 5.0)
    pc = rolling_percentile_causal(c, window=8, q=BG_QUANTILE)
    assert torch.allclose(pc, c)

    # расширяющееся окно: до кадра `window` перцентиль считается по всему,
    # что уже накопилось, а не только по последним `window` кадрам —
    # проверяем на ряде с выбросом в начале, который окно ещё не должно
    # "забыть" на кадре 2
    x2 = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]])
    p2 = rolling_percentile_causal(x2, window=3, q=0.5)
    # кадр 1: история [100, 0] -> медиана 50; кадр 2: [100,0,0] -> медиана 0
    # (окно длиной 3 с кадра 2 уже включает начальный выброс полностью)
    assert abs(p2[0, 1].item() - 50.0) < 1e-4, p2[0, 1]
    assert abs(p2[0, 2].item() - 0.0) < 1e-4, p2[0, 2]

    # окно останавливается на ширине `window`, забывает более старые кадры:
    # длинный ряд нулей после одного выброса, окно=3 — после трёх шагов
    # выброс должен полностью выпасть из окна
    x3 = torch.tensor([[100.0] + [0.0] * 10])
    p3 = rolling_percentile_causal(x3, window=3, q=1.0)   # максимум в окне
    assert p3[0, 3].item() == 0.0, p3[0, 3]   # кадр 3: окно [0,0,0], выброса уже нет

    # работает на многомерном входе (батч x полосы x время) поэлементно по
    # ведущим измерениям — ровно то, что понадобится для ch0 формы [B,F,T]
    x4 = torch.rand(2, 5, 16)
    p4 = rolling_percentile_causal(x4, window=BG_WINDOW_FRAMES, q=BG_QUANTILE)
    assert p4.shape == x4.shape

    print("background selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.features.background --selfcheck`
Expected: FAIL, `NameError: name 'rolling_percentile_causal' is not defined`

- [ ] **Step 3: Реализовать**

```python
def rolling_percentile_causal(x, window, q):
    """Каузальный перцентиль расширяющимся, затем скользящим окном.

    Цикл по времени, не векторизовано — T для одного клипа мало (максимум
    94 кадра на 12 с по этому плану), а сборка тренировочного батча идёт
    на CPU при подготовке признаков, не в горячем цикле обучения. Если
    трассировка на GPU в реальном времени этапа 3 потребует иного —
    отдельная задача оптимизации, не эта.
    """
    T = x.shape[-1]
    out = torch.empty_like(x)
    for t in range(T):
        lo = max(0, t - window + 1)
        out[..., t] = torch.quantile(x[..., lo:t + 1], q, dim=-1)
    return out
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.features.background --selfcheck`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add airadar/features/background.py
git commit -m "features: ch1 — каузальный скользящий перцентиль расширяющимся окном

Решение спецификации §2.1: перцентиль на кадре t берётся по всей доступной
истории, не по фиксированному окну. Короткие клипы (86% позитивов DADS
короче 8с) не требуют паддинга — расширяющееся окно корректно определено
с нулевой предыстории."
```

---

### Task 3: сборка каналов — единственная точка входа

**Files:**
- Create: `airadar/features/frontend.py`

**Interfaces:**
- Consumes: `airadar.features.cqt.LogCQT`, `airadar.features.background.{rolling_percentile_causal, BG_WINDOW_FRAMES, BG_QUANTILE}`
- Produces:
  - `MODEL_FRAMES = 32` (4.0 с — контекст модели, измеренное число кадров)
  - `class Frontend(torch.nn.Module)` — `__init__(self)`, `forward(self, wav: torch.Tensor) -> torch.Tensor` — вход `[B, N]` или `[N]`, выход `[B, 2, 183, T]` (оба канала, все кадры входа — обрезка до последних `MODEL_FRAMES` НЕ делается здесь, это решение вызывающего кода, см. докстринг)
  - `Frontend.last_model_frames(x: torch.Tensor) -> torch.Tensor` — статический хелпер, берёт последние `MODEL_FRAMES` кадров по последней оси `[..., T] -> [..., MODEL_FRAMES]`, бросает, если `T < MODEL_FRAMES`

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Сборка [ch0, ch1] — единственная точка входа фронтенда.

Обрезка до окна модели (последние 32 кадра) НЕ встроена в forward: разным
вызывающим кодам (валидации этого плана, будущему обучению) нужна разная
длина — валидации интересна вся последовательность, обучению — только
последние 4с. Здесь только вычисление признака; политику длины окна
задаёт вызывающий код через last_model_frames().
"""

import sys
import torch

from airadar.features.cqt import LogCQT, N_BINS
from airadar.features.background import rolling_percentile_causal, BG_WINDOW_FRAMES, BG_QUANTILE

MODEL_FRAMES = 32   # 4.0 с — контекст модели, см. airadar.features.cqt


def selfcheck():
    fe = Frontend()

    wav12 = torch.zeros(1, round(12.0 * 16000))
    out = fe(wav12)
    assert out.shape == (1, 2, N_BINS, 94), out.shape   # 12с -> 94 кадра, см. Task 1

    # ch0 канал (индекс 0) обязан совпасть с LogCQT напрямую — сборка не
    # должна ничего менять в ch0, только добавлять ch1
    ch0_direct = LogCQT()(wav12)
    assert torch.allclose(out[:, 0], ch0_direct, atol=1e-5)

    # ch1 = ch0 - перцентиль(ch0): на тишине (всё константа) ch1 обязан
    # быть везде нулём — перцентиль константного ряда равен самой константе
    assert torch.allclose(out[:, 1], torch.zeros_like(out[:, 1]), atol=1e-4)

    # last_model_frames: обрезка до последних 32 кадров, без побочных
    # эффектов на исходном тензоре
    trimmed = Frontend.last_model_frames(out)
    assert trimmed.shape == (1, 2, N_BINS, MODEL_FRAMES)
    assert torch.allclose(trimmed, out[..., -MODEL_FRAMES:])

    # T < MODEL_FRAMES — короткий клип (например, 0.6с DADS даёт 5 кадров,
    # см. Task 1) — обрезка обязана падать, а не молча отдавать что есть:
    # вызывающий код (будущее обучение, MIL-подмешивание) должен явно
    # решить, что делать с короткими клипами, а не получить тихо усечённый
    # тензор неожиданной формы
    short = torch.zeros(1, 2, N_BINS, 5)
    try:
        Frontend.last_model_frames(short)
    except ValueError as e:
        assert "32" in str(e) and "5" in str(e), str(e)
    else:
        raise AssertionError("last_model_frames должен падать на T < MODEL_FRAMES")

    print("frontend selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.features.frontend --selfcheck`
Expected: FAIL, `NameError: name 'Frontend' is not defined`

- [ ] **Step 3: Реализовать**

```python
class Frontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._logcqt = LogCQT()

    def forward(self, wav):
        ch0 = self._logcqt(wav)                                       # [B, F, T]
        ch1 = ch0 - rolling_percentile_causal(ch0, BG_WINDOW_FRAMES, BG_QUANTILE)
        return torch.stack([ch0, ch1], dim=1)                         # [B, 2, F, T]

    @staticmethod
    def last_model_frames(x):
        T = x.shape[-1]
        if T < MODEL_FRAMES:
            raise ValueError(
                f"нужно minimum {MODEL_FRAMES} кадров контекста модели, "
                f"получено {T} — короткий клип должен быть подмешан в фон "
                f"вызывающим кодом (MIL), а не обрезан молча")
        return x[..., -MODEL_FRAMES:]
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.features.frontend --selfcheck`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add airadar/features/frontend.py
git commit -m "features: сборка [ch0,ch1] — единственная точка входа

Обрезка до окна модели не встроена в forward: валидации этого плана нужна
вся последовательность, будущему обучению — последние 32 кадра. Политику
длины задаёт вызывающий код через last_model_frames(), которая явно падает
на клипах короче контекста модели, а не молча усекает."
```

---

### Task 4: синтетическая проверка — разводит ли новый признак гул и гребёнку

**Files:**
- Create: `airadar/bench/feat_visibility.py`

**Interfaces:**
- Consumes: `airadar.features.frontend.Frontend`, `airadar.features.cqt.LogCQT`
- Produces:
  - `comb(f0: float, sr: int = 16000, n: int = 64000, weights=(1.0,0.5,0.35,0.2,0.15,0.1,0.08,0.05)) -> np.ndarray` — синтетическая гребёнка, портирована из `evalx/feat_visibility.py`
  - `cosine_hum_drone() -> dict[str, float]` — считает `ch0` для синтетических гула (50 Гц) и гребёнок дрона (62, 78, 200 Гц) на полосе < 300 Гц, возвращает косинусы `hum_vs_62`, `hum_vs_78`, `hum_vs_200`

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Разводит ли новый CQT-признак гул ЛЭП и гребёнку тяжёлого дрона?

Тот же метод, что уже применён в проекте (evalx/feat_visibility.py,
hum_confusion): синтетические гул и гребёнка дрона проецируются в признак,
косинус между ними на полосах < 300 Гц. На старом mel-признаке (n_fft=2048,
64 mel) косинус hum/drone62 был 0.78 (docs/metrics-plan.md §0.3) — это и
объясняло провал на грузовом дроне: аугментация "подмешай гул" душила ровно
то подпространство признака, где живёт тяжёлая машина.
"""

import sys
import numpy as np


def selfcheck():
    # comb() детерминирован по seed снаружи — здесь просто форма и энергия
    x = comb(78.0)
    assert x.shape == (64000,)
    assert np.abs(x).max() <= 1.0 + 1e-6

    # разные f0 дают разные сигналы
    assert not np.allclose(comb(50.0), comb(78.0))

    print("feat_visibility selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    elif "--report" in sys.argv:
        report()
    else:
        sys.exit("--selfcheck или --report")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.feat_visibility --selfcheck`
Expected: FAIL, `NameError: name 'comb' is not defined`

- [ ] **Step 3: Реализовать `comb()` и проверку по selfcheck**

```python
import torch
from airadar.features.cqt import LogCQT, BINS_PER_OCTAVE

SR = 16000


def comb(f0, sr=SR, n=64000, weights=(1.0, 0.5, 0.35, 0.2, 0.15, 0.1, 0.08, 0.05), seed=0):
    """Синтетическая гребёнка с гармониками, случайные фазы (детерминировано seed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / sr
    x = np.zeros(n, np.float32)
    for k, w in enumerate(weights, 1):
        if f0 * k < sr / 2:
            x += w * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    return x / (np.abs(x).max() + 1e-9)
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.feat_visibility --selfcheck`
Expected: PASS

- [ ] **Step 5: Написать падающую проверку косинуса (новый TDD-цикл в этом же файле)**

Дописать перед `if __name__ == "__main__":`:

```python
def cosine_hum_drone(seed=0):
    """Косинус между гулом ЛЭП и гребёнками дрона на полосах < 300 Гц.

    Гул генерируется той же формулой, что в train.Augment (50 Гц + 3
    гармоники затухающих весов), гребёнки дрона — на f0 из реальных
    полевых записей (62, 78 Гц) и типичного малого квадрокоптера (200 Гц).
    """
    cqt = LogCQT()
    lo = cqt.frequencies < 300.0
    assert lo.any(), "нет ни одного бина ниже 300 Гц — проверить FMIN/N_BINS"

    sigs = {
        "hum": comb(50.0, weights=(1.0, 0.5, 0.35, 0.2), seed=seed),
        "drone62": comb(62.0, seed=seed + 1),
        "drone78": comb(78.0, seed=seed + 2),
        "drone200": comb(200.0, seed=seed + 3),
    }
    vecs = {}
    for name, x in sigs.items():
        ch0 = cqt(torch.from_numpy(x)).squeeze(0).numpy()   # [183, T]
        v = ch0.mean(axis=1)[lo]                             # усредняем по времени
        vecs[name] = v - v.mean()                             # центрируем, как в исходном методе

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    return {
        "n_bands_lt_300": int(lo.sum()),
        "hum_vs_62": cos(vecs["hum"], vecs["drone62"]),
        "hum_vs_78": cos(vecs["hum"], vecs["drone78"]),
        "hum_vs_200": cos(vecs["hum"], vecs["drone200"]),
    }
```

И добавить в `selfcheck()`:

```python
    res = cosine_hum_drone()
    assert res["n_bands_lt_300"] > 8, res   # больше, чем 8 полос старого mel-банка
    for k in ("hum_vs_62", "hum_vs_78", "hum_vs_200"):
        assert -1.0 - 1e-6 <= res[k] <= 1.0 + 1e-6, (k, res[k])
```

- [ ] **Step 6: Прогнать, убедиться что проходит, и получить реальное число**

Run:
```bash
python -m airadar.bench.feat_visibility --selfcheck
python -c "
import sys; sys.path.insert(0, '.')
from airadar.bench.feat_visibility import cosine_hum_drone
print(cosine_hum_drone())
"
```

Expected: `selfcheck` проходит. Второй вызов печатает реальные косинусы —
записать их в отчёт (Step 8) и сравнить со старым числом (0.78–0.90). Если
косинус НЕ упал существенно ниже 0.78 — это реальный, важный результат
(гипотеза о разрешении не объясняет всю коллизию), а не повод подгонять
код: сообщить как есть.

- [ ] **Step 7: Добавить `report()` для человекочитаемого вывода**

```python
def report():
    res = cosine_hum_drone()
    print(f"полос < 300 Гц: {res['n_bands_lt_300']} (было 8 на старом mel-банке)")
    print(f"cos(гул, дрон 62 Гц):  {res['hum_vs_62']:.3f}  (было 0.78-0.90)")
    print(f"cos(гул, дрон 78 Гц):  {res['hum_vs_78']:.3f}")
    print(f"cos(гул, дрон 200 Гц): {res['hum_vs_200']:.3f}")
```

- [ ] **Step 8: Прогнать `--report`, зафиксировать числа в коммите**

Run: `python -m airadar.bench.feat_visibility --report 2>&1 | tee logs/feat_visibility_cqt.log`

- [ ] **Step 9: Коммит**

```bash
git add airadar/bench/feat_visibility.py logs/feat_visibility_cqt.log
git commit -m "bench: синтетическая проверка разводимости гула и гребёнки в CQT-признаке

Тот же метод, что evalx/feat_visibility.py (hum_confusion), применён к
новому фронтенду. Старое число для сравнения — косинус 0.78-0.90 на mel.
Результат — в logs/feat_visibility_cqt.log, не подогнан под ожидание."
```

---

### Task 5: проверка на реальных полевых записях

**Files:**
- Create: modify `airadar/bench/feat_visibility.py`

**Interfaces:**
- Consumes: `airadar.bench.corpus.field_records`, `airadar.features.cqt.LogCQT`
- Produces:
  - `median_flatten(spec: np.ndarray, ker: int) -> np.ndarray` — стирает гребёнку медианным фильтром по частотной оси, портировано из `evalx/feat_visibility.py`
  - `visibility_field() -> dict[str, dict]` — по каждой полевой записи: видимость гребёнки в дБ (разница `ch0` до/после стирания медианным фильтром)

**Замечание для реализатора:** `airadar.bench.corpus.field_records()` уже
существует и проверен (этап 0) — не переписывать загрузку WAV заново.

- [ ] **Step 1: Написать падающий selfcheck**

Дописать в `airadar/bench/feat_visibility.py`, в `selfcheck()`:

```python
    # median_flatten: на плоском (константном) спектре ничего не меняет
    flat = np.full((20, 5), 3.0, np.float32)
    assert np.allclose(median_flatten(flat, ker=5), flat)

    # на спектре с одиночным пиком по частоте — пик стирается медианой
    spec = np.zeros((21, 3), np.float32)
    spec[10] = 10.0                            # пик в середине частотной оси
    out = median_flatten(spec, ker=7)
    assert out[10, 0] < 5.0, out[10, 0]        # пик подавлен
    assert np.allclose(out[0], spec[0])        # края почти не тронуты (edge-паддинг)
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.feat_visibility --selfcheck`
Expected: FAIL, `NameError: name 'median_flatten' is not defined`

- [ ] **Step 3: Реализовать**

```python
def median_flatten(spec, ker):
    """Стирает гребёнку медианным фильтром по частотной оси (не по времени)."""
    ker = max(3, ker | 1)
    pad = ker // 2
    padded = np.pad(spec, ((pad, pad), (0, 0)), mode="edge")
    view = np.lib.stride_tricks.as_strided(
        padded, shape=(spec.shape[0], ker, spec.shape[1]),
        strides=(padded.strides[0], padded.strides[0], padded.strides[1]))
    return np.median(view, axis=1)
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.feat_visibility --selfcheck`
Expected: PASS

- [ ] **Step 5: Реализовать `visibility_field()` (проверяется на реальных данных, не в selfcheck)**

```python
def visibility_field(f0_by_name=None):
    """Видимость гребёнки на полевых записях: |ch0(с гребёнкой) - ch0(без)| в дБ.

    f0 записей взяты из evalx/f0_survey.py (задокументировано в README):
    drone_video1.wav ~78 Гц, drone_video2.wav ~62 Гц.
    """
    from airadar.bench.corpus import field_records

    f0_by_name = f0_by_name or {"drone_video1.wav": 78.0, "drone_video2.wav": 62.0}
    cqt = LogCQT()
    out = {}
    for name, wav in field_records().items():
        f0 = f0_by_name.get(name)
        if f0 is None:
            continue
        ch0 = cqt(torch.from_numpy(wav.astype(np.float32))).squeeze(0).numpy()  # [183, T]
        # Ширина медианного окна — диапазон от f0 до 2.5*f0 (запас "1.5*f0"
        # поверх самой f0, как в исходном evalx/feat_visibility.py), в бинах
        # ЛОГ-оси. На линейной STFT-оси ширина в бинах зависела от f0 (там
        # ker = 1.5*f0 / (SR/n_fft)); на лог-оси она НЕ зависит от f0 —
        # ровно то свойство self-similarity, ради которого выбран constant-Q:
        # соотношение f0 -> 2.5*f0 занимает одно и то же число бинов на
        # любой частоте.
        ker_bins = max(3, int(round(BINS_PER_OCTAVE * np.log2(2.5))))   # ~32 бина
        flat = median_flatten(ch0, ker_bins)
        diff_db = np.abs(ch0 - flat).mean(axis=1) * 10.0 / np.log(10.0)  # нат -> дБ
        lo = cqt.frequencies < 300.0
        out[name] = {
            "vis_all_db": float(diff_db.mean()),
            "vis_low_db": float(diff_db[lo].max()) if lo.any() else float("nan"),
            "f0": f0,
        }
    return out
```

- [ ] **Step 6: Прогнать на реальных полевых записях, получить числа**

Run:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from airadar.bench.feat_visibility import visibility_field
import json
print(json.dumps(visibility_field(), indent=2, ensure_ascii=False))
" | tee -a logs/feat_visibility_cqt.log
```

Expected: словарь с `vis_all_db`/`vis_low_db` по каждой записи. Сравнить
со старым числом из README (n_fft=2048, 64 mel: `drone_video` 13.9 дБ,
`drone_video2` 4.8 дБ). Записать оба числа как есть, не подгонять.

- [ ] **Step 7: Коммит**

```bash
git add airadar/bench/feat_visibility.py logs/feat_visibility_cqt.log
git commit -m "bench: видимость гребёнки на реальных полевых записях в CQT-признаке

Сравнение со старым числом README (mel n_fft=2048: 13.9/4.8 дБ). Метод —
median_flatten по частотной оси, портирован из evalx/feat_visibility.py."
```

---

### Task 6: диагностика — переживает ли амплитудная модуляция шаг кадра 128 мс

**Files:**
- Create: `airadar/diag/am_preservation.py`

**Interfaces:**
- Consumes: `airadar.features.cqt.LogCQT`, `airadar.bench.corpus.field_records`
- Produces:
  - `am_rate_hz(ch0_band: np.ndarray, hop_s: float = 0.128) -> float` — доминирующая частота модуляции огибающей одной полосы `ch0` по времени (через автокорреляцию), в Гц
  - CLI `python -m airadar.diag.am_preservation`

**Замечание для реализатора.** Открытый вопрос спецификации §2 (после
таблицы): исходный трёхтирный дизайн держал верхний тир на коротком окне
ради амплитудной модуляции лопастей, но частота КАДРОВ всё равно задаётся
`hop_s=128` мс — по теореме Найквиста любая модуляция быстрее `1/(2*0.128)
≈ 3.9 Гц` не резолвится в последовательности кадров независимо от длины
анализирующего окна. Эта задача **измеряет**, какая скорость модуляции
реально присутствует в полевых записях, чтобы понять, актуален ли предел
3.9 Гц для настоящих БПЛА (лопастная частота малого квадрокоптера обычно
выше — десятки герц), а не спорит с математикой Найквиста.

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Актуален ли предел Найквиста 3.9 Гц (при шаге кадра 128 мс) для реальной
амплитудной модуляции лопастей?

Не пытается ничего исправить — только измеряет и печатает факт, как D0 в
этапе 0. Решение (короче ли делать hop, заводить ли отдельную "быструю"
ветку для АМ) — за планом этапа 3, не за этой диагностикой.
"""

import sys
import numpy as np

HOP_S = 0.128
NYQUIST_HZ = 1.0 / (2.0 * HOP_S)   # 3.90625 Гц — предел по теореме Найквиста


def selfcheck():
    sr_frames = 1.0 / HOP_S   # 7.8125 "кадров в секунду"

    # синусоида-огибающая на известной частоте, ниже предела Найквиста —
    # автокорреляционный метод обязан её найти с разумной точностью
    t = np.arange(200) * HOP_S
    known_hz = 1.0                                   # заведомо ниже 3.9 Гц
    band = 1.0 + 0.5 * np.sin(2 * np.pi * known_hz * t)
    est = am_rate_hz(band, hop_s=HOP_S)
    assert abs(est - known_hz) < 0.3, est

    # чистый шум без периодичности — оценка не обязана совпасть ни с чем
    # конкретным, но обязана вернуть конечное число, не падать
    noise = np.random.default_rng(0).normal(1.0, 0.1, 200)
    est_noise = am_rate_hz(noise, hop_s=HOP_S)
    assert np.isfinite(est_noise)

    print(f"предел Найквиста при hop={HOP_S}с: {NYQUIST_HZ:.3f} Гц")
    print("am_preservation selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.diag.am_preservation --selfcheck`
Expected: FAIL, `NameError: name 'am_rate_hz' is not defined`

- [ ] **Step 3: Реализовать**

```python
def am_rate_hz(ch0_band, hop_s=HOP_S):
    """Доминирующая частота модуляции огибающей через автокорреляцию.

    ch0_band — одна полоса ch0 во времени, [T]. Ищем пик автокорреляции
    (кроме нулевого лага) и переводим лаг в кадрах в частоту в Гц.
    """
    x = np.asarray(ch0_band, dtype=np.float64)
    x = x - x.mean()
    if np.allclose(x, 0):
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-12
    # первый локальный максимум после лага 0, не считая сам лаг 0
    peak_lag = None
    for lag in range(1, len(ac) - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.1:
            peak_lag = lag
            break
    if peak_lag is None:
        return 0.0
    return 1.0 / (peak_lag * hop_s)
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.diag.am_preservation --selfcheck`
Expected: PASS

- [ ] **Step 5: Реализовать `main()` — измерение на реальных полевых записях**

```python
def main():
    from airadar.features.cqt import LogCQT
    from airadar.bench.corpus import field_records
    import torch

    cqt = LogCQT()
    print(f"предел Найквиста при hop={HOP_S}с: {NYQUIST_HZ:.3f} Гц\n")
    for name, wav in field_records().items():
        ch0 = cqt(torch.from_numpy(wav.astype(np.float32))).squeeze(0).numpy()  # [183, T]
        # три полосы в диапазоне лопастного шума малых аппаратов: 500 Гц,
        # 2 кГц, 4 кГц — ищем ближайший бин к каждой
        freqs = cqt.frequencies
        for target in (500.0, 2000.0, 4000.0):
            idx = int(np.argmin(np.abs(freqs - target)))
            rate = am_rate_hz(ch0[idx])
            over = "ВЫШЕ предела" if rate > NYQUIST_HZ else "в пределах"
            print(f"{name}  полоса {freqs[idx]:.0f} Гц: "
                  f"АМ ~{rate:.2f} Гц ({over} Найквиста)")
```

- [ ] **Step 6: Прогнать на реальных данных**

Run: `python -m airadar.diag.am_preservation 2>&1 | tee logs/am_preservation.log`

Expected: таблица частот АМ по трём полосам на каждой из двух полевых
записей. Если частоты систематически выше 3.9 Гц — предел Найквиста реален
и актуален (подтверждает риск, поднятый в спецификации §2); если ниже —
беспокойство было напрасным на этих конкретных записях. Записать как есть,
не подгонять формулировку под ожидаемый результат.

- [ ] **Step 7: Коммит**

```bash
git add airadar/diag/am_preservation.py logs/am_preservation.log
git commit -m "diag: амплитудная модуляция лопастей против предела Найквиста hop=128мс

Не чинит ничего — измеряет, актуален ли предел 3.9 Гц для реальных полевых
записей, поднятый как открытый вопрос в спецификации §2. Результат в
logs/am_preservation.log, решение (короче ли hop) — за планом этапа 3."
```

---

## Проверка перед закрытием этапа

- [ ] `python cli/selfcheck.py` — все модули, включая шесть новых, зелёные
- [ ] `logs/feat_visibility_cqt.log` содержит реальные числа косинуса и видимости (синтетика + поле), сравнённые со старыми (0.78–0.90 и 13.9/4.8 дБ)
- [ ] `logs/am_preservation.log` содержит измеренную частоту АМ на реальных полевых записях
- [ ] `nnAudio==0.3.4` — версия, на которой измерены числа кадров (94/63/32) в Global Constraints; если план исполняется на другой версии, числа нужно перепроверить, не полагаться на них

## Что этот план сознательно не делает

- **Не переключает `train.py`/`detect.py`/`eval.py` на новый фронтенд.**
  Это отдельная миграция после того, как признак подтверждён валидацией
  этого плана, а не до.
- **Не считает harmonic stacking** (спецификация §3.1, ветка A) — это
  часть модели (этап 3), не фронтенда.
- **Не оптимизирует `rolling_percentile_causal` под батчевое GPU-обучение**
  (Task 2, комментарий в коде) — цикл по времени пригоден для оффлайн
  подготовки признаков и валидации, но не для горячего цикла обучения на
  масштабе полного корпуса; при необходимости — отдельная задача этапа 3.
- **Не отвечает на вопрос «нужен ли отдельный короткий hop для АМ»**
  (Task 6) — измеряет факт, решение принимается при написании плана
  этапа 3.
- **Не добавляет `nnAudio` в README/RUNBOOK-инструкции по установке** — они
  документируют текущий рабочий пайплайн; зависимость попадёт туда, когда
  реальный код обучения станет её импортировать (миграция, не этот план).
