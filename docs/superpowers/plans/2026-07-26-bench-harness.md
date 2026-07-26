# Этап 0: измерительный харнесс и диагностика D0 — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** построить замороженный измерительный харнесс `airadar/bench/`, получить базовые цифры для `models/dronenet_local.pt` и ответить на вопрос D0 (смежны ли соседние клипы DADS) — до того, как написана хоть строка новой модели.

**Architecture:** харнесс не знает о моделях. Всё общение идёт через протокол `Scorer`: «дай непрерывное аудио — верни ряд **логитов** с известным шагом». Нынешняя модель заворачивается в `LegacyScorer`, будущая — в свой адаптер, и обе меряются одним кодом. Метрики строятся снизу вверх: блочный бутстрап → решающий слой (сглаживание, события, FA/час) → лестница SNR50 → страты и перенос порога → отчёт.

**Tech Stack:** Python 3.10, numpy, torch 2.11+cu128, scipy, pyarrow, soundfile. Новых зависимостей не вводится.

## Global Constraints

- `SR = 16000` Гц, моно, int16 на диске, float32 в расчётах.
- Все метрики возвращают **логиты**, не вероятности. `sigmoid` в float32 насыщается в 1.0 и уничтожает ранжирование в верхнем хвосте — именно там живёт рабочая точка.
- Ни один модуль не импортирует `train.py` кроме `airadar/bench/scorer.py`. Это единственная точка связи со старым кодом.
- Конвенция проекта сохраняется: каждый модуль **с собственной логикой** имеет `selfcheck()`, работающий **без данных и без GPU**, запускаемый как `python -m airadar.bench.<модуль> --selfcheck`. Исключение — сборщики и обёртки, у которых своей логики нет: `report.py` (только склейка чужих результатов) и всё в `cli/`. Проверять там нечего, и пустой `selfcheck` был бы тестом, который ничего не утверждает.
- Комментарии объясняют «почему», а не «что». Язык комментариев и сообщений — русский, как во всём проекте.
- Целевой размер модуля — до ~200 строк.
- Полевые записи `field/drone_video*.wav` — финальный холдаут (спецификация §6.4). В этом этапе они читаются, но ни один порог и ни один гиперпараметр по ним не подбирается.
- Все скрипты харнесса должны работать на CPU: `CUDA_VISIBLE_DEVICES= python -m ...`.

## Ссылки

- Спецификация: [docs/superpowers/specs/2026-07-26-architecture-redesign-design.md](../specs/2026-07-26-architecture-redesign-design.md)
- Обоснование метрик: [docs/metrics-plan.md](../../metrics-plan.md)
- Существующие прототипы для переноса: `evalx/field_ci.py` (блочный бутстрап, `auc_fh`), `evalx/f0_survey.py` (оценка f0, сохраняет `evalx/f0_*.npz`)

## Структура файлов

| файл | ответственность |
|---|---|
| `airadar/__init__.py`, `airadar/bench/__init__.py` | пакет |
| `airadar/bench/scorer.py` | протокол `Scorer` + `LegacyScorer` — единственный мост к `train.py` |
| `airadar/bench/ci.py` | блочный бутстрап, обобщённый по статистике |
| `airadar/bench/corpus.py` | загрузка полевых записей, трудных негативов, сборка непрерывного фона |
| `airadar/bench/decision.py` | сглаживание логита, гистерезис, события, FA/час, поиск порога под бюджет |
| `airadar/bench/ladder.py` | подмешивание по лестнице SNR, психометрическая кривая, SNR50 |
| `airadar/bench/strata.py` | recall по f0-полосам, худшая полоса |
| `airadar/bench/transfer.py` | ошибка переноса порога между корпусами |
| `airadar/bench/report.py` | сборка JSON + markdown отчёта |
| `airadar/diag/dads_contiguity.py` | D0 |
| `cli/bench.py`, `cli/diag.py`, `cli/selfcheck.py` | тонкие обёртки |

---

### Task 1: протокол Scorer и адаптер нынешней модели

**Files:**
- Create: `airadar/__init__.py` (пустой), `airadar/bench/__init__.py` (пустой)
- Create: `airadar/bench/scorer.py`

**Interfaces:**
- Consumes: `train.LogMel`, `train.DroneNet`, `train.DEV` из существующего `train.py`
- Produces:
  - `class Scorer(Protocol)` с атрибутами `hop_s: float`, `context_s: float` и методом `score(audio: np.ndarray) -> np.ndarray`
  - `n_scores(n_samples: int, context_s: float, hop_s: float, sr: int = 16000) -> int`
  - `frame_times(n: int, context_s: float, hop_s: float) -> np.ndarray` — время **центра** каждого окна в секундах
  - `class LegacyScorer` с `hop_s = 0.25`, `context_s = 0.5`, конструктор `LegacyScorer(ckpt_path: str, device: str = "cpu")`

- [ ] **Step 1: Написать падающий selfcheck**

Создать `airadar/bench/scorer.py` только с блоком проверок (реализации ещё нет):

```python
"""Единственный мост между харнесом и моделями.

Харнес не должен знать, что внутри модели. Он знает одно: дай непрерывное
аудио — получи ряд логитов с известным шагом. Тогда нынешняя 0.5-секундная
модель и будущая 4-секундная меряются одним и тем же кодом, а сравнение
между ними осмысленно.

Возвращаются логиты, а не вероятности: sigmoid в float32 упирается в 1.0 и
стирает порядок в верхнем хвосте распределения — ровно там, где стоит
рабочая точка.
"""

import sys
import numpy as np

SR = 16000


def selfcheck():
    # длина ряда оценок: первое окно занимает context, дальше шаг hop
    assert n_scores(8000, 0.5, 0.25) == 1          # ровно одно окно
    assert n_scores(12000, 0.5, 0.25) == 2         # 0.75 с -> окна в 0, 0.25
    assert n_scores(4000, 0.5, 0.25) == 0          # короче контекста
    assert n_scores(64000, 4.0, 0.128) == 1        # ровно 4 с

    # центры окон: первое окно [0, 0.5) -> центр 0.25
    t = frame_times(3, 0.5, 0.25)
    assert np.allclose(t, [0.25, 0.5, 0.75]), t

    print("scorer selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.scorer --selfcheck`
Expected: FAIL, `NameError: name 'n_scores' is not defined`

- [ ] **Step 3: Реализовать чистые функции**

Вставить перед `selfcheck()`:

```python
def n_scores(n_samples, context_s, hop_s, sr=SR):
    """Сколько оценок даст ряд из n_samples отсчётов."""
    ctx, hop = int(round(context_s * sr)), int(round(hop_s * sr))
    if n_samples < ctx:
        return 0
    return 1 + (n_samples - ctx) // hop


def frame_times(n, context_s, hop_s):
    """Время центра каждого окна. Нужно, чтобы сшивать оценку с разметкой."""
    return context_s / 2.0 + np.arange(n) * hop_s
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.scorer --selfcheck`
Expected: PASS, `scorer selfcheck ok`

- [ ] **Step 5: Добавить падающую проверку протокола и адаптера**

Дописать в `selfcheck()` перед `print`:

```python
    # адаптер должен считать на синтетике без чекпоинта: подменяем модель
    class Fake:
        hop_s, context_s = 0.25, 0.5

        def score(self, audio):
            n = n_scores(len(audio), self.context_s, self.hop_s)
            return np.zeros(n, np.float32)

    s = Fake()
    assert check_scorer(s, n_samples=12000) == 2

    # рассинхронизация шага и контекста должна ловиться, а не молча врать
    class Broken(Fake):
        def score(self, audio):
            return np.zeros(99, np.float32)

    try:
        check_scorer(Broken(), n_samples=12000)
    except AssertionError:
        pass
    else:
        raise AssertionError("check_scorer не поймал неверную длину ряда")
```

- [ ] **Step 6: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.scorer --selfcheck`
Expected: FAIL, `NameError: name 'check_scorer' is not defined`

- [ ] **Step 7: Реализовать протокол, проверку и LegacyScorer**

```python
from typing import Protocol


class Scorer(Protocol):
    """Контракт, который обязан выполнять любой детектор.

    hop_s     — шаг между соседними оценками
    context_s — сколько секунд аудио нужно на одну оценку
    score     — float32 [T] -> float32 [n_scores(T)], ЛОГИТЫ
    """

    hop_s: float
    context_s: float

    def score(self, audio: np.ndarray) -> np.ndarray: ...


def check_scorer(s, n_samples=SR * 4):
    """Проверяет, что скорер держит собственный контракт по длине ряда.

    Заведено потому, что рассинхронизация шага и длины ряда молча сдвигает
    всю разметку по времени, и обнаруживается это уже в метрике, где выглядит
    как «модель хуже», а не как баг.
    """
    audio = np.zeros(n_samples, np.float32)
    out = s.score(audio)
    want = n_scores(n_samples, s.context_s, s.hop_s)
    assert out.ndim == 1, f"score вернул {out.ndim} измерений, нужен 1"
    assert len(out) == want, f"score вернул {len(out)} оценок, ожидалось {want}"
    assert out.dtype == np.float32, f"score вернул {out.dtype}, нужен float32"
    return len(out)


class LegacyScorer:
    """Нынешняя DroneNet за фасадом Scorer. Базовая линия для сравнений.

    Повторяет предобработку detect.py дословно: окно 0.5 с, шаг 0.25 с,
    пиковая нормализация. Отклоняться нельзя — иначе базовая цифра будет
    измерять не ту модель, что стоит в проекте.
    """

    hop_s, context_s = 0.25, 0.5

    def __init__(self, ckpt_path, device="cpu"):
        import torch
        from train import LogMel, DroneNet
        self._torch = torch
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = DroneNet().to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.logmel = LogMel().to(device)
        self.device = device

    def score(self, audio, bs=512):
        torch = self._torch
        ctx = int(self.context_s * SR)
        hop = int(self.hop_s * SR)
        n = n_scores(len(audio), self.context_s, self.hop_s)
        if n == 0:
            return np.zeros(0, np.float32)
        win = np.stack([audio[i * hop:i * hop + ctx] for i in range(n)])
        out = np.empty(n, np.float32)
        with torch.no_grad():
            for i in range(0, n, bs):
                x = torch.from_numpy(win[i:i + bs]).to(self.device).float()
                x = x / (x.abs().amax(1, keepdim=True) + 1e-8)
                out[i:i + bs] = self.model(self.logmel(x).unsqueeze(1)).cpu().numpy()
        return out
```

- [ ] **Step 8: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.scorer --selfcheck`
Expected: PASS

- [ ] **Step 9: Проверить адаптер на настоящем чекпоинте**

Run:
```bash
CUDA_VISIBLE_DEVICES= python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from airadar.bench.scorer import LegacyScorer, check_scorer
s = LegacyScorer('models/dronenet_local.pt')
print('оценок:', check_scorer(s, 16000*4))
print('логиты на тишине:', s.score(np.zeros(16000*4, np.float32))[:3])
"
```
Expected: `оценок: 15`, три конечных числа (не `nan`, не `inf`).

- [ ] **Step 10: Коммит**

```bash
git add airadar/__init__.py airadar/bench/__init__.py airadar/bench/scorer.py
git commit -m "bench: протокол Scorer и адаптер нынешней DroneNet

Харнес общается с моделями через один контракт: непрерывное аудио -> ряд
логитов с известным шагом. Логиты, а не вероятности: sigmoid в float32
упирается в 1.0 и стирает порядок в верхнем хвосте, где стоит рабочая точка."
```

---

### Task 2: блочный бутстрап

**Files:**
- Create: `airadar/bench/ci.py`
- Reference: `evalx/field_ci.py` (существующий прототип, обобщается)

**Interfaces:**
- Consumes: ничего
- Produces:
  - `block_bootstrap(x: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int = 4000, block: int = 12, seed: int = 0) -> np.ndarray` — распределение статистики, `[n_boot]`
  - `ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]`
  - `paired_diff_ci(xa, xb, stat, block=12, n_boot=4000, seed=0) -> tuple[float, float]` — CI **разности** при общей ресэмплировке индексов

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Блочный бутстрап: доверительные интервалы для рядов с автокорреляцией.

Окна перекрываются вдвое и идут подряд по одной записи, поэтому соседние
оценки почти одинаковы. Обычный биномиальный интервал в такой ситуации врёт
вдвое в сторону оптимизма — именно поэтому пять экспериментов проекта
сравнили шум и приняли его за сигнал. Блок в L окон означает допущение, что
независимы куски записи длиной L*hop секунд.
"""

import sys
import numpy as np


def selfcheck():
    rng = np.random.default_rng(0)
    x = rng.random(2000) < 0.5                      # iid, истинное среднее 0.5

    lo1, hi1 = ci(block_bootstrap(x, np.mean, block=1, seed=1))
    assert lo1 < 0.5 < hi1, (lo1, hi1)
    assert 0.03 < hi1 - lo1 < 0.06, hi1 - lo1       # ~ +-2.2% для n=2000

    # на iid-данных крупный блок не должен систематически сужать интервал,
    # а на коррелированных обязан его расширить
    y = np.repeat(rng.random(50) < 0.5, 40)         # блоки по 40 одинаковых
    w1 = np.subtract(*ci(block_bootstrap(y, np.mean, block=1, seed=2))[::-1])
    w40 = np.subtract(*ci(block_bootstrap(y, np.mean, block=40, seed=2))[::-1])
    assert w40 > 2 * w1, (w1, w40)

    # ресэмплировка не меняет длину выборки
    assert len(block_bootstrap(x, np.mean, n_boot=17, block=12)) == 17

    # парная разность: одинаковые ряды -> CI накрывает ноль и он узкий
    lo, hi = paired_diff_ci(x, x.copy(), np.mean, block=12)
    assert lo == hi == 0.0, (lo, hi)

    print("ci selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.ci --selfcheck`
Expected: FAIL, `NameError: name 'ci' is not defined`

- [ ] **Step 3: Реализовать**

```python
def _resample_idx(n, block, n_boot, rng):
    """Индексы moving block bootstrap: nb блоков подряд, обрезка до n."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=(n_boot, nb))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    return idx.reshape(n_boot, -1)[:, :n] % n


def block_bootstrap(x, stat, n_boot=4000, block=12, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = _resample_idx(len(x), block, n_boot, rng)
    return np.array([stat(x[i]) for i in idx], np.float64)


def ci(samples, level=0.95):
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [a, 1.0 - a])
    return float(lo), float(hi)


def paired_diff_ci(xa, xb, stat, block=12, n_boot=4000, seed=0):
    """CI разности stat(a) - stat(b) при ОБЩИХ индексах ресэмплировки.

    Общие индексы обязательны: ряды посчитаны на одних и тех же окнах одной
    записи, и большая часть разброса у них общая. Независимая ресэмплировка
    раздула бы интервал и спрятала реальную разницу между моделями.
    """
    xa, xb = np.asarray(xa), np.asarray(xb)
    assert len(xa) == len(xb), "ряды должны быть по одним и тем же окнам"
    rng = np.random.default_rng(seed)
    idx = _resample_idx(len(xa), block, n_boot, rng)
    d = np.array([stat(xa[i]) - stat(xb[i]) for i in idx], np.float64)
    return ci(d)
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.ci --selfcheck`
Expected: PASS, `ci selfcheck ok`

- [ ] **Step 5: Коммит**

```bash
git add airadar/bench/ci.py
git commit -m "bench: блочный бутстрап, обобщённый по статистике

Парная разность считается на общих индексах ресэмплировки: ряды посчитаны на
одних окнах одной записи, общий разброс надо сокращать, иначе интервал
раздувается и прячет реальную разницу между моделями."
```

---

### Task 3: корпуса

**Files:**
- Create: `airadar/bench/corpus.py`

**Interfaces:**
- Consumes: `airadar.bench.scorer.SR`
- Produces:
  - `read_wav_mono16k(path: str) -> np.ndarray` — float32 в [-1, 1], бросает `ValueError` при неверном формате
  - `field_records(pattern: str = "field/drone_video*.wav") -> dict[str, np.ndarray]`
  - `load_cache(name: str) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]` — `(X, y, split, cat)`; `X` формы `[N, 8000]` int16
  - `hard_holdout(cat_filter: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — удержанные (`split != 0`) окна `cache_hard`, их категории и номера групп
  - `regroup(X: np.ndarray, group: np.ndarray) -> list[np.ndarray]` — восстановленные исходные клипы
  - `stitch(clips: list[np.ndarray], xfade_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]` — непрерывная дорожка и границы стыков в отсчётах
  - `seam_mask(n: int, seams: np.ndarray, context_s: float, hop_s: float) -> np.ndarray` — bool `[n]`, `True` для окон, **не** пересекающих стык

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Загрузка корпусов и сборка непрерывного фона.

Кэши состоят из нарезанных окон, а метрика FA/час требует непрерывности:
событие определено на дорожке, а не на мешке окон. Клипы склеиваются обратно
с кроссфейдом 50 мс, а окна, пересекающие стык, исключаются из подсчёта —
скачок уровня на стыке читается детектором как событие и завышает FA/час.
"""

import os
import sys
import glob
import wave
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SR = 16000


def selfcheck():
    # склейка: длина = сумма минус перекрытия, стыки на своих местах
    a = np.ones(1600, np.float32)
    b = np.full(1600, 2.0, np.float32)
    xf = 800                                   # 0.05 с при 16 кГц
    out, seams = stitch([a, b, a], xfade_s=0.05)
    assert len(out) == 3 * 1600 - 2 * xf, len(out)
    assert list(seams) == [1600 - xf, 2 * 1600 - 2 * xf], seams
    # кроссфейд монотонно переводит 1.0 в 2.0, разрывов нет
    seg = out[seams[0]:seams[0] + xf]
    assert seg[0] < seg[-1] and np.all(np.diff(seg) >= -1e-6)

    # маска стыков: окно 0.5 с шагом 0.25 с, стык на 1.0 с (отсчёт 16000)
    m = seam_mask(6, np.array([16000]), context_s=0.5, hop_s=0.25)
    # окна начинаются в 0.00 0.25 0.50 0.75 1.00 1.25 и длятся 0.5 с.
    # Стык в 1.0 с пересекает только окно, начинающееся в 0.75 (оно идёт
    # до 1.25). Окно с 0.50 кончается ровно на стыке и его НЕ пересекает —
    # границы полуоткрыты, иначе маска выбрасывала бы вдвое больше нужного.
    assert list(m) == [True, True, True, False, True, True], list(m)

    # короткий кроссфейд не должен превышать длину клипа
    try:
        stitch([np.ones(100, np.float32)] * 2, xfade_s=0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("stitch не поймал клип короче кроссфейда")

    # восстановление исходных клипов: соседние окна одной группы смежны
    # по построению (prep_hf.windows режет подряд и без перекрытия), поэтому
    # склеиваются встык, без кроссфейда, и стык внутри группы не возникает
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    g = np.array([7, 7, 7, 9, 9, 7])
    tr = regroup(X, g)
    assert len(tr) == 2, len(tr)
    assert list(tr[0]) == list(range(0, 12)) + list(range(20, 24)), tr[0]
    assert list(tr[1]) == list(range(12, 20)), tr[1]

    print("corpus selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.corpus --selfcheck`
Expected: FAIL, `NameError: name 'stitch' is not defined`

- [ ] **Step 3: Реализовать склейку и маску**

```python
def stitch(clips, xfade_s=0.05):
    """Склейка клипов в непрерывную дорожку с кроссфейдом.

    Возвращает (дорожка, позиции стыков). Позиция стыка — начало зоны
    кроссфейда: именно там уровень меняется, и именно эту зону надо
    вырезать из подсчёта событий.
    """
    xf = int(round(xfade_s * SR))
    for c in clips:
        if len(c) < 2 * xf:
            raise ValueError(f"клип {len(c)} отсчётов короче двух кроссфейдов {2*xf}")
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    out = clips[0].astype(np.float32).copy()
    seams = []
    for c in clips[1:]:
        c = c.astype(np.float32)
        seams.append(len(out) - xf)
        out[-xf:] = out[-xf:] * (1.0 - ramp) + c[:xf] * ramp
        out = np.concatenate([out, c[xf:]])
    return out, np.array(seams, np.int64)


def seam_mask(n, seams, context_s, hop_s, sr=SR):
    """True для окон, НЕ пересекающих ни один стык."""
    ctx, hop = int(round(context_s * sr)), int(round(hop_s * sr))
    starts = np.arange(n) * hop
    ok = np.ones(n, bool)
    for s in np.atleast_1d(seams):
        ok &= ~((starts < s) & (starts + ctx > s))
    return ok
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.corpus --selfcheck`
Expected: PASS

- [ ] **Step 5: Добавить загрузчики (без нового selfcheck — они требуют данных)**

```python
def read_wav_mono16k(path):
    w = wave.open(path)
    if w.getframerate() != SR or w.getnchannels() != 1:
        raise ValueError(f"{path}: нужен моно {SR} Гц, "
                         f"а тут {w.getnchannels()} кан. {w.getframerate()} Гц")
    raw = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return raw.astype(np.float32) / 32768.0


def field_records(pattern="field/drone_video*.wav"):
    """Полевые записи целиком, НЕ нарезанные.

    Не усредняются между собой: у них разная основная частота (78.0 и
    60.5 Гц), и общее число спрятало бы, что одна пропускается целиком.
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        out[os.path.basename(p)] = read_wav_mono16k(p)
    if not out:
        raise FileNotFoundError(f"нет записей по шаблону {pattern}")
    return out


def load_cache(name):
    """Кэш окон: (X memmap [N,8000] int16, y, split, cat)."""
    d = os.path.join(ROOT, name)
    meta = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    n, win = int(meta["n"]), int(meta["win"])
    X = np.memmap(os.path.join(d, "windows.bin"), np.int16, "r", shape=(n, win))
    cat = meta["cat"] if "cat" in meta.files else np.full(n, "", object)
    return X, meta["y"], meta["split"], cat


def hard_holdout(cat_filter=None):
    """Удержанные трудные негативы. split != 0 — то, чего не было в обучении."""
    X, y, split, cat, group = load_cache("cache_hard")
    sel = np.flatnonzero(split != 0)
    if cat_filter is not None:
        sel = sel[np.isin(cat[sel], list(cat_filter))]
    sel = np.sort(sel)
    return (np.ascontiguousarray(X[sel]).astype(np.float32) / 32768.0,
            cat[sel], group[sel])


def regroup(X, group):
    """Восстановить исходные клипы из окон кэша.

    prep_hf.windows режет клип на подряд идущие непересекающиеся куски,
    поэтому окна одной группы склеиваются ВСТЫК и дают ровно исходное аудио.
    Это принципиально: если склеивать 0.5-секундные окна с кроссфейдом, стык
    возникает каждые 0.45 с, а при контексте 0.5 с его задевает каждое окно —
    считать FA/час становится не на чем.
    """
    order = np.argsort(group, kind="stable")
    out, i = [], 0
    while i < len(order):
        j = i
        while j < len(order) and group[order[j]] == group[order[i]]:
            j += 1
        out.append(np.concatenate([X[k] for k in order[i:j]]))
        i = j
    return out
```

Соответственно `load_cache` возвращает пять массивов, а не четыре — поправить
его сигнатуру и распаковку:

```python
def load_cache(name):
    """Кэш окон: (X memmap [N,8000] int16, y, split, cat, group).

    Ключи meta.npz сверены с диском 2026-07-26:
    y, group, src, split, synth, n, win (+ cat, hard в cache_hard).
    """
    d = os.path.join(ROOT, name)
    meta = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    n, win = int(meta["n"]), int(meta["win"])
    X = np.memmap(os.path.join(d, "windows.bin"), np.int16, "r", shape=(n, win))
    cat = meta["cat"] if "cat" in meta.files else np.full(n, "", "<U16")
    return X, meta["y"], meta["split"], cat, meta["group"]
```

- [ ] **Step 6: Проверить загрузчики на настоящих данных**

Run:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from airadar.bench.corpus import field_records, hard_holdout, regroup
f = field_records()
print({k: round(len(v)/16000, 1) for k, v in f.items()})
X, c, g = hard_holdout()
import numpy as np
print('удержанных трудных окон:', len(X), 'категорий:', len(np.unique(c)))
tr = regroup(X, g)
d = np.array([len(t)/16000 for t in tr])
print('клипов после regroup:', len(tr), 'медиана длины, с:', round(float(np.median(d)), 2))
"
```
Expected: две записи по ~48 и ~39 с; ~9000 удержанных окон; после `regroup` — клипы медианной длины около 2 с (UrbanSound8K, `CAP=4`) или 2.5 с (ESC-50, `CAP=5`).

Если медиана вышла 0.5 с, значит группы не объединяют окна одного клипа — остановиться и разобраться с ключом группы в `prep_hf`, а не продолжать: без восстановленных клипов метрика FA/час не имеет знаменателя.

- [ ] **Step 7: Коммит**

```bash
git add airadar/bench/corpus.py
git commit -m "bench: корпуса, склейка непрерывного фона, маска стыков

FA/час определена на дорожке, а не на мешке окон. Клипы склеиваются с
кроссфейдом 50 мс, окна через стык исключаются: скачок уровня на стыке
детектор читает как событие и завышает FA/час."
```

---

### Task 4: решающий слой и FA/час

**Files:**
- Create: `airadar/bench/decision.py`

**Interfaces:**
- Consumes: ничего
- Produces:
  - `smooth(logits: np.ndarray, hop_s: float, tau_s: float = 2.0) -> np.ndarray`
  - `hysteresis(x: np.ndarray, on: float, off: float) -> np.ndarray` — bool
  - `events(alarm: np.ndarray, hop_s: float, dead_s: float = 30.0) -> np.ndarray` — индексы начала событий
  - `fa_per_hour(logits, hop_s, on, off, dead_s=30.0, mask=None) -> float`
  - `threshold_for_fa(logits_bg, hop_s, target_fa, off_delta=1.0, dead_s=30.0, mask=None) -> float`
  - `detected(logits, hop_s, on, off) -> bool` — сработала ли тревога хоть раз
  - `time_to_alarm(logits, hop_s, on, off) -> float` — секунды до первой тревоги, `inf` если её нет

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Решающий слой: от ряда логитов к событиям и к FA/час.

Правило "k из m" удалено. Измерено, что при равном пооконном FAR оно recall
снижает (24.1% -> 12.4%): бинаризация до накопления выбрасывает величину
оценки. Здесь накапливается сам логит, а гистерезис убирает дребезг тревоги.

Порог задаётся там, где принимается решение — по бюджету ложных тревог в
час на фоне, а не по доле окон. Порог, откалиброванный по доле окон
лабораторного корпуса, в лесу соврёт в разы.
"""

import sys
import numpy as np


def selfcheck():
    hop = 0.25

    # сглаживание: константа остаётся собой, одиночный выброс подавляется
    c = np.full(20, 3.0, np.float32)
    assert np.allclose(smooth(c, hop, tau_s=2.0), 3.0)
    spike = np.zeros(20, np.float32); spike[10] = 10.0
    assert smooth(spike, hop, tau_s=2.0).max() < 2.0

    # гистерезис: включается по on, держится до off, не дребезжит
    x = np.array([0, 2, 0.6, 0.6, 0, 0], np.float32)
    assert list(hysteresis(x, on=1.0, off=0.5)) == [False, True, True, True, False, False]

    # события: слипшиеся тревоги — одно событие, мёртвое время склеивает
    al = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], bool)
    assert list(events(al, hop_s=hop, dead_s=0.0)) == [1, 5]
    assert list(events(al, hop_s=hop, dead_s=30.0)) == [1]

    # FA/час: 2 события на 1 часе фона
    n = int(3600 / hop)
    bg = np.zeros(n, np.float32); bg[100] = 10.0; bg[n // 2] = 10.0
    assert abs(fa_per_hour(bg, hop, on=1.0, off=0.5, dead_s=0.0) - 2.0) < 1e-6

    # подбор порога под бюджет: результат обязан дать FA не выше бюджета
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, n).astype(np.float32)
    thr = threshold_for_fa(noise, hop, target_fa=1.0, dead_s=30.0)
    assert fa_per_hour(noise, hop, thr, thr - 1.0, dead_s=30.0) <= 1.0 + 1e-9

    # время до тревоги
    y = np.array([0, 0, 5, 5], np.float32)
    assert abs(time_to_alarm(y, hop, on=1.0, off=0.5) - 0.5) < 1e-9
    assert time_to_alarm(np.zeros(4, np.float32), hop, 1.0, 0.5) == np.inf
    assert detected(y, hop, 1.0, 0.5) and not detected(np.zeros(4, np.float32), hop, 1.0, 0.5)

    print("decision selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.decision --selfcheck`
Expected: FAIL, `NameError: name 'smooth' is not defined`

- [ ] **Step 3: Реализовать**

```python
def smooth(logits, hop_s, tau_s=2.0):
    """Экспоненциальное сглаживание логита.

    Сглаживается логит, а не бинарное решение: величина оценки несёт
    информацию, и её сохранение — самая дешёвая прибавка к дальности из
    доступных.
    """
    x = np.asarray(logits, np.float32)
    if len(x) == 0:
        return x
    a = float(np.exp(-hop_s / tau_s))
    out = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = a * acc + (1.0 - a) * x[i]
        out[i] = acc
    return out


def hysteresis(x, on, off):
    """Триггер Шмитта: включение по on, выключение по off < on."""
    assert off <= on, "порог выключения должен быть не выше порога включения"
    x = np.asarray(x)
    out = np.zeros(len(x), bool)
    state = False
    for i, v in enumerate(x):
        state = v >= on if not state else v >= off
        out[i] = state
    return out


def events(alarm, hop_s, dead_s=30.0):
    """Начала событий. Слипшееся — одно; после события мёртвое время.

    Мёртвое время моделирует интерфейс оператора: тревога, повторившаяся
    через две секунды, для него не вторая тревога, а та же самая.
    """
    alarm = np.asarray(alarm, bool)
    dead = int(round(dead_s / hop_s))
    out, last = [], -10**9
    prev = False
    for i, v in enumerate(alarm):
        if v and not prev and i - last > dead:
            out.append(i)
            last = i
        prev = v
    return np.array(out, np.int64)


def fa_per_hour(logits, hop_s, on, off, dead_s=30.0, mask=None):
    """Ложных тревог в час. mask=False выбрасывает окно из знаменателя тоже."""
    x = np.asarray(logits, np.float32)
    if mask is not None:
        x = x[np.asarray(mask, bool)]
    if len(x) == 0:
        return 0.0
    hours = len(x) * hop_s / 3600.0
    return len(events(hysteresis(x, on, off), hop_s, dead_s)) / hours


def threshold_for_fa(logits_bg, hop_s, target_fa, off_delta=1.0,
                     dead_s=30.0, mask=None, n_grid=512):
    """Наименьший порог, при котором FA/час не превышает бюджет.

    Перебор по сетке значений самого ряда, а не бисекция: функция FA(порог)
    ступенчата, и бисекция на ступеньках останавливается где попало.
    """
    x = np.asarray(logits_bg, np.float32)
    if mask is not None:
        x = x[np.asarray(mask, bool)]
    if len(x) == 0:
        return float("inf")
    grid = np.unique(np.quantile(x, np.linspace(0.5, 1.0, n_grid)))
    for thr in grid:                      # от низкого к высокому
        if fa_per_hour(x, hop_s, thr, thr - off_delta, dead_s) <= target_fa:
            return float(thr)
    return float(grid[-1] + off_delta)


def detected(logits, hop_s, on, off):
    return bool(hysteresis(np.asarray(logits, np.float32), on, off).any())


def time_to_alarm(logits, hop_s, on, off):
    al = hysteresis(np.asarray(logits, np.float32), on, off)
    idx = np.flatnonzero(al)
    return float(idx[0] * hop_s) if len(idx) else float("inf")
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.decision --selfcheck`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add airadar/bench/decision.py
git commit -m "bench: решающий слой — сглаживание логита, гистерезис, FA/час

Правило k из m удалено: измерено, что при равном пооконном FAR оно recall
снижает (24.1 -> 12.4%). Порог подбирается по бюджету тревог в час на фоне,
а не по доле окон: порог по доле окон лабораторного корпуса в лесу врёт."
```

---

### Task 5: лестница деградации и SNR50

**Files:**
- Create: `airadar/bench/ladder.py`

**Interfaces:**
- Consumes: `airadar.bench.decision.{smooth, detected, threshold_for_fa}`, `airadar.bench.ci.{block_bootstrap, ci}`
- Produces:
  - `mix_at_snr(target: np.ndarray, noise: np.ndarray, snr_db: float, rng) -> np.ndarray`
  - `SNR_GRID: np.ndarray` — от +20 до −15 дБ шагом 2.5 (15 ступеней)
  - `p_detect_curve(scorer, target, noise_pool, on, off, snrs=SNR_GRID, n_rep=8, seed=0) -> np.ndarray` — `[len(snrs)]`
  - `snr50(snrs: np.ndarray, pdet: np.ndarray) -> float` — линейная интерполяция по убывающей кривой, `nan` если 0.5 не пересекается
  - `snr50_ci(snrs, pdet_boot: np.ndarray) -> tuple[float, float]`

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Лестница деградации: главный скаляр проекта.

Вместо "recall при фиксированных условиях" измеряется условие, при котором
детектор ломается. Метрика не насыщается конструктивно: стало лучше — кривая
сдвинулась влево, и сдвиг виден. Единица физическая: 6 дБ примерно вдвое по
дальности в свободном поле.

Фон для подмешивания обязан быть удержанным и не тем, что в noise_pool
обучения, иначе лестница измерит запоминание фона, а не обнаружение цели.
"""

import sys
import numpy as np

SNR_GRID = np.arange(20.0, -15.1, -2.5)


def selfcheck():
    rng = np.random.default_rng(0)

    # подмешивание: фактический SNR совпадает с заказанным
    t = rng.normal(0, 1, 16000).astype(np.float32)
    n = rng.normal(0, 1, 16000).astype(np.float32)
    for want in (10.0, 0.0, -10.0):
        m = mix_at_snr(t, n, want, rng)
        got = 10 * np.log10(np.mean(t ** 2) / np.mean((m - t) ** 2))
        assert abs(got - want) < 0.2, (want, got)

    # сетка идёт сверху вниз и содержит 15 ступеней
    assert len(SNR_GRID) == 15 and SNR_GRID[0] == 20.0 and SNR_GRID[-1] == -15.0

    # snr50 на идеальной ступеньке: переход между 5.0 и 2.5 -> ровно посередине
    p = np.where(SNR_GRID >= 5.0, 1.0, 0.0)
    assert abs(snr50(SNR_GRID, p) - 3.75) < 1e-6, snr50(SNR_GRID, p)

    # линейный спад: 0.5 достигается там, где кривая её пересекает
    p2 = np.clip(0.5 + (SNR_GRID - 0.0) / 20.0, 0.0, 1.0)
    assert abs(snr50(SNR_GRID, p2) - 0.0) < 0.5, snr50(SNR_GRID, p2)

    # кривая, никогда не падающая до 0.5 — метрика не определена, не выдумываем
    assert np.isnan(snr50(SNR_GRID, np.ones_like(SNR_GRID)))
    assert np.isnan(snr50(SNR_GRID, np.zeros_like(SNR_GRID)))

    print("ladder selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.ladder --selfcheck`
Expected: FAIL, `NameError: name 'mix_at_snr' is not defined`

- [ ] **Step 3: Реализовать смешивание и SNR50**

```python
def mix_at_snr(target, noise, snr_db, rng):
    """Подмешать фон к цели так, чтобы получился заданный SNR.

    Фон при необходимости зацикливается и берётся со случайного сдвига:
    иначе на всех ступенях лестницы окажется один и тот же кусок фона, и
    разброс между ступенями будет измерять его, а не модель.
    """
    t = np.asarray(target, np.float32)
    n = np.asarray(noise, np.float32)
    if len(n) < len(t):
        n = np.tile(n, int(np.ceil(len(t) / len(n))))
    off = int(rng.integers(0, len(n)))
    n = np.roll(n, off)[:len(t)]
    tp = float(np.mean(t ** 2)) + 1e-12
    np_ = float(np.mean(n ** 2)) + 1e-12
    scale = np.sqrt(tp / (np_ * 10.0 ** (snr_db / 10.0)))
    return t + n * scale


def snr50(snrs, pdet):
    """SNR, при котором вероятность обнаружения падает до 0.5.

    Кривая по построению убывает с падением SNR. Ищем первое пересечение
    уровня 0.5 сверху вниз и интерполируем линейно. Если пересечения нет —
    возвращаем nan: метрика не определена, и выдумывать её нельзя, иначе
    получится ещё одно число, которое сравнивает шум.
    """
    s = np.asarray(snrs, np.float64)
    p = np.asarray(pdet, np.float64)
    for i in range(len(s) - 1):
        if p[i] >= 0.5 > p[i + 1]:
            w = (p[i] - 0.5) / (p[i] - p[i + 1])
            return float(s[i] + w * (s[i + 1] - s[i]))
    return float("nan")


def snr50_ci(snrs, pdet_boot):
    """CI по бутстрап-выборке кривых [n_boot, len(snrs)]."""
    from airadar.bench.ci import ci
    vals = np.array([snr50(snrs, p) for p in pdet_boot], np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return float("nan"), float("nan")
    return ci(vals)
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.ladder --selfcheck`
Expected: PASS

- [ ] **Step 5: Добавить падающую проверку кривой на синтетическом скорере**

Дописать в `selfcheck()` перед `print`:

```python
    # Синтетический скорер: доля энергии в полосе цели. Именно доля, а не
    # абсолютный уровень — при падении SNR подмешивается БОЛЬШЕ шума, полная
    # энергия растёт, и скорер по уровню дал бы кривую, растущую вниз по
    # лестнице. Цель — тон 200 Гц, фон — белый шум.
    class BandScorer:
        hop_s, context_s = 0.25, 0.5

        def score(self, audio):
            from airadar.bench.scorer import n_scores
            n = n_scores(len(audio), self.context_s, self.hop_s)
            ctx, hop = 8000, 4000
            f = np.fft.rfftfreq(ctx, 1.0 / 16000)
            band = (f > 190) & (f < 210)
            wnd = np.hanning(ctx)
            out = np.empty(n, np.float32)
            for i in range(n):
                sp = np.abs(np.fft.rfft(audio[i * hop:i * hop + ctx] * wnd)) ** 2
                out[i] = 10 * np.log10((sp[band].sum() + 1e-12)
                                       / (sp.sum() + 1e-12))
            return out

    rng2 = np.random.default_rng(7)
    tt = np.arange(16000 * 4) / 16000.0
    tgt = np.sin(2 * np.pi * 200 * tt).astype(np.float32)
    pool = [rng2.normal(0, 1.0, 16000 * 4).astype(np.float32) for _ in range(4)]
    curve = p_detect_curve(BandScorer(), tgt, pool, on=-10.0, off=-11.0,
                           n_rep=4, seed=3)
    assert curve[0] >= curve[-1], curve          # вниз по SNR не растёт
    assert curve[0] == 1.0 and curve[-1] == 0.0, curve
    assert 0.0 <= curve.min() and curve.max() <= 1.0
```

- [ ] **Step 6: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.ladder --selfcheck`
Expected: FAIL, `NameError: name 'p_detect_curve' is not defined`

- [ ] **Step 7: Реализовать кривую**

```python
def p_detect_curve(scorer, target, noise_pool, on, off, snrs=SNR_GRID,
                   n_rep=8, seed=0, tau_s=2.0):
    """Доля повторов, в которых цель обнаружена, на каждой ступени SNR.

    n_rep повторов с разным куском фона: одна и та же цель на разных фонах
    при одинаковом SNR даёт заметно разные оценки, и без усреднения кривая
    измеряет выбор фона.
    """
    from airadar.bench.decision import smooth, detected
    rng = np.random.default_rng(seed)
    out = np.empty(len(snrs), np.float64)
    for j, snr in enumerate(snrs):
        hits = 0
        for _ in range(n_rep):
            noise = noise_pool[int(rng.integers(0, len(noise_pool)))]
            mixed = mix_at_snr(target, noise, float(snr), rng)
            lg = smooth(scorer.score(mixed), scorer.hop_s, tau_s)
            hits += int(detected(lg, scorer.hop_s, on, off))
        out[j] = hits / n_rep
    return out
```

- [ ] **Step 8: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.ladder --selfcheck`
Expected: PASS, `ladder selfcheck ok`

- [ ] **Step 9: Коммит**

```bash
git add airadar/bench/ladder.py
git commit -m "bench: лестница деградации и SNR50

Измеряется условие отказа, а не доля при фиксированной сложности. Метрика не
насыщается конструктивно, единица физическая: 6 дБ примерно вдвое по
дальности. Пересечения 0.5 нет — возвращается nan, а не выдуманное число."
```

---

### Task 6: f0-страты

**Files:**
- Create: `airadar/bench/strata.py`

**Interfaces:**
- Consumes: `evalx/f0_*.npz` (результат `evalx/f0_survey.py`), `airadar.bench.corpus.load_cache`
- Produces:
  - `F0_BANDS: tuple[tuple[float, float], ...]` = `((40, 80), (80, 120), (120, 200), (200, 300), (300, 1e9))`
  - `band_of(f0: np.ndarray) -> np.ndarray` — индекс полосы `[N]`, `-1` вне диапазона
  - `recall_by_band(logits, y, f0, thr) -> dict[str, float]`
  - `worst_band(rec: dict[str, float]) -> tuple[str, float]`
  - `load_f0_estimates() -> tuple[np.ndarray, np.ndarray]` — `(idx, f0)`: индексы окон в `cache_dads` и оценки f0 для них

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Recall по полосам основной частоты, отчёт по худшей полосе.

Все агрегаты проекта — средние по распределению, в котором интересующий
случай составляет проценты: 86% окон дрона в DADS это квадрокоптеры с
f0 > 100 Гц с близкой дистанции. Среднее по такому пулу обязано насытиться.
Худшая полоса — нет: в низкой страте порядка 21 600 окон, у неё свой узкий
доверительный интервал, и видно, растёт ли она отдельно от остальных.
"""

import sys
import numpy as np

F0_BANDS = ((40.0, 80.0), (80.0, 120.0), (120.0, 200.0), (200.0, 300.0), (300.0, 1e9))


def selfcheck():
    f0 = np.array([50.0, 90.0, 150.0, 250.0, 500.0, 30.0, 0.0])
    assert list(band_of(f0)) == [0, 1, 2, 3, 4, -1, -1], list(band_of(f0))
    # границы принадлежат верхней полосе
    assert band_of(np.array([80.0]))[0] == 1
    assert band_of(np.array([300.0]))[0] == 4

    # recall по полосам: в полосе 0 угадано 1 из 2, в полосе 2 — 2 из 2
    lg = np.array([1.0, -1.0, 5.0, 5.0], np.float32)
    y = np.array([1, 1, 1, 1])
    f = np.array([50.0, 50.0, 150.0, 150.0])
    rec = recall_by_band(lg, y, f, thr=0.0)
    assert abs(rec["40-80"] - 0.5) < 1e-9, rec
    assert abs(rec["120-200"] - 1.0) < 1e-9, rec
    assert "80-120" not in rec, "пустые полосы не должны попадать в отчёт"

    name, val = worst_band(rec)
    assert name == "40-80" and abs(val - 0.5) < 1e-9, (name, val)

    print("strata selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.strata --selfcheck`
Expected: FAIL, `NameError: name 'band_of' is not defined`

- [ ] **Step 3: Реализовать**

```python
def _name(b):
    lo, hi = F0_BANDS[b]
    return f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f"{lo:.0f}+"


def band_of(f0):
    """Индекс полосы для каждой оценки f0. -1 — вне диапазона 40..inf."""
    f0 = np.asarray(f0, np.float64)
    out = np.full(len(f0), -1, np.int64)
    for b, (lo, hi) in enumerate(F0_BANDS):
        out[(f0 >= lo) & (f0 < hi)] = b
    return out


def recall_by_band(logits, y, f0, thr):
    """Recall отдельно по каждой f0-полосе. Пустые полосы не отчитываются."""
    logits, y = np.asarray(logits), np.asarray(y)
    b = band_of(f0)
    out = {}
    for k in range(len(F0_BANDS)):
        sel = (b == k) & (y == 1)
        if sel.sum() == 0:
            continue
        out[_name(k)] = float((logits[sel] >= thr).mean())
    return out


def worst_band(rec):
    """Отчётная величина — худшая полоса, а не средняя по полосам."""
    if not rec:
        return "", float("nan")
    k = min(rec, key=rec.get)
    return k, rec[k]
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.strata --selfcheck`
Expected: PASS

- [ ] **Step 5: Убедиться, что оценки f0 на диске есть**

Run:
```bash
ls evalx/f0_*.npz 2>/dev/null || OMP_NUM_THREADS=4 python evalx/f0_survey.py 3000
python -c "
import numpy as np, glob
p = sorted(glob.glob('evalx/f0_*.npz'))[0]
d = np.load(p)
print(p, list(d.files), {k: d[k].shape for k in d.files})
"
```
Expected: массивы `f0`, `salience` и т.п.

- [ ] **Step 6: Реализовать загрузчик оценок f0**

Формат сверен с `evalx/f0_survey.py:117` на 2026-07-26: файл
`evalx/f0_dads_1.npz` (суффикс — метка, `1` = дрон) содержит ключи
`idx, f0, sal, blo`. Файл `f0_dads_0.npz` — это **фон**, и брать первый по
алфавиту нельзя: `sorted(glob)[0]` попадёт именно на него.

```python
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F0_DRONE_NPZ = os.path.join(ROOT, "evalx", "f0_dads_1.npz")


def load_f0_estimates(min_salience=6.0):
    """Оценки f0 по окнам ДРОНА из cache_dads.

    Возвращает (idx, f0) — индексы окон в кэше и основную частоту.
    idx обязателен: f0_survey считает выборку, а не весь кэш, и без индексов
    оценки не сопоставить с логитами — страта тихо съедет на другие окна.

    Отсекаются окна со слабой гребёнкой: оценщик f0 на них возвращает шум,
    и такая страта измеряла бы качество оценщика, а не детектора.
    """
    if not os.path.exists(F0_DRONE_NPZ):
        raise FileNotFoundError(
            f"нет {F0_DRONE_NPZ} — сначала: python evalx/f0_survey.py 3000")
    d = np.load(F0_DRONE_NPZ)
    missing = {"idx", "f0", "sal"} - set(d.files)
    if missing:
        raise KeyError(f"{F0_DRONE_NPZ}: нет ключей {missing}, есть {list(d.files)}")
    keep = d["sal"] >= min_salience
    return d["idx"][keep].astype(np.int64), d["f0"][keep].astype(np.float64)
```

- [ ] **Step 7: Прогнать selfcheck ещё раз**

Run: `python -m airadar.bench.strata --selfcheck`
Expected: PASS (загрузчик данных не трогается selfcheck-ом, он требует файлов)

- [ ] **Step 8: Коммит**

```bash
git add airadar/bench/strata.py
git commit -m "bench: recall по f0-полосам, отчёт по худшей полосе

Среднее по пулу, где 86% — квадрокоптеры с близкой дистанции, обязано
насытиться. Худшая полоса не насыщается: в низкой страте ~21600 окон, у неё
свой узкий интервал, и видно, растёт ли она отдельно от остальных."
```

---

### Task 7: ошибка переноса порога

**Files:**
- Create: `airadar/bench/transfer.py`

**Interfaces:**
- Consumes: ничего
- Produces:
  - `threshold_at_far(logits_neg: np.ndarray, far: float) -> float`
  - `transfer_error(logits_a: np.ndarray, logits_b: np.ndarray, far: float) -> dict` — ключи `threshold`, `far_nominal`, `far_actual`, `ratio`
  - `drift(logits_a: np.ndarray, logits_b: np.ndarray, q: float = 0.99) -> float` — сдвиг квантиля в единицах разброса корпуса A

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""Насколько соврёт порог, откалиброванный в лаборатории, приехав в лес.

Отношение факт/номинал прямо отвечает на вопрос, который иначе выясняется
только на месте. Метрика не насыщается: это мера сдвига распределений, а не
качества модели. Если дрейф больше зазора между дроном и негативом, никакой
фиксированный порог не работает в принципе — и тогда адаптивный порог не
улучшение, а необходимость.
"""

import sys
import numpy as np


def selfcheck():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 100000).astype(np.float32)

    # порог по FAR 1% на стандартной нормали ~ 2.326
    thr = threshold_at_far(a, 0.01)
    assert abs(thr - 2.326) < 0.05, thr

    # тот же корпус — перенос идеален
    r = transfer_error(a, a.copy(), 0.01)
    assert abs(r["ratio"] - 1.0) < 0.05, r

    # корпус B сдвинут на +1 сигму: фактический FAR обязан вырасти в разы
    b = a + 1.0
    r2 = transfer_error(a, b, 0.01)
    assert r2["far_actual"] > 0.05, r2
    assert r2["ratio"] > 5.0, r2

    # дрейф квантиля в единицах разброса A
    assert abs(drift(a, b, q=0.99) - 1.0) < 0.05, drift(a, b, q=0.99)
    assert abs(drift(a, a.copy(), q=0.99)) < 0.02

    print("transfer selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.bench.transfer --selfcheck`
Expected: FAIL, `NameError: name 'threshold_at_far' is not defined`

- [ ] **Step 3: Реализовать**

```python
def threshold_at_far(logits_neg, far):
    """Порог, дающий заданную долю ложных на корпусе негативов."""
    return float(np.quantile(np.asarray(logits_neg, np.float64), 1.0 - far))


def transfer_error(logits_a, logits_b, far):
    """Порог откалиброван на A, применён к B. Отношение факт/номинал."""
    thr = threshold_at_far(logits_a, far)
    actual = float((np.asarray(logits_b) >= thr).mean())
    return {"threshold": thr, "far_nominal": float(far),
            "far_actual": actual, "ratio": actual / far if far > 0 else float("inf")}


def drift(logits_a, logits_b, q=0.99):
    """Сдвиг квантиля B относительно A, выраженный в стандартных отклонениях A.

    Единица выбрана так, чтобы величину можно было сравнивать с зазором
    между дроном и негативом, измеренным на том же корпусе.
    """
    a = np.asarray(logits_a, np.float64)
    s = float(a.std()) + 1e-12
    return float((np.quantile(np.asarray(logits_b, np.float64), q)
                  - np.quantile(a, q)) / s)
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.bench.transfer --selfcheck`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add airadar/bench/transfer.py
git commit -m "bench: ошибка переноса порога и дрейф распределения негативов

Отношение факт/номинал отвечает на вопрос, который иначе выясняется только на
месте. Если дрейф больше зазора дрон/негатив, фиксированный порог не работает
в принципе, и адаптивный порог перестаёт быть улучшением."
```

---

### Task 8: отчёт и базовая цифра

**Files:**
- Create: `airadar/bench/report.py`
- Create: `cli/bench.py`
- Create: `cli/selfcheck.py`

**Interfaces:**
- Consumes: всё из Task 1–7
- Produces:
  - `corpus.hard_categories() -> np.ndarray` — 16 трудных категорий из `meta["hard"]` (дописывается в `airadar/bench/corpus.py`)
  - `run_bench(scorer, name: str, seed: int = 0) -> dict` — полный отчёт
  - `write_report(rep: dict, out_dir: str = "bench_out") -> tuple[str, str]` — пути к JSON и markdown
  - CLI: `python cli/bench.py --model models/dronenet_local.pt --name dronenet_local`
  - CLI: `python cli/selfcheck.py` — прогоняет `selfcheck()` всех модулей `airadar/`

- [ ] **Step 1: Написать cli/selfcheck.py и убедиться, что он зелёный**

```python
"""Прогон всех selfcheck пакета одной командой.

Заведено потому, что модулей стало много, и проверка "всё ли ещё цело"
не должна требовать помнить список.
"""

import os
import sys
import pkgutil
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import airadar

fail = 0
for mod in pkgutil.walk_packages(airadar.__path__, "airadar."):
    m = importlib.import_module(mod.name)
    fn = getattr(m, "selfcheck", None)
    if fn is None:
        continue
    try:
        fn()
    except Exception as e:
        print(f"ПРОВАЛ {mod.name}: {e}")
        fail += 1
sys.exit(1 if fail else 0)
```

Run: `python cli/selfcheck.py`
Expected: по строке `... selfcheck ok` на каждый модуль, код возврата 0.

- [ ] **Step 2: Коммит**

```bash
git add cli/selfcheck.py
git commit -m "cli: прогон всех selfcheck пакета одной командой"
```

- [ ] **Step 2b: Дописать `hard_categories()` в `airadar/bench/corpus.py`**

```python
def hard_categories():
    """16 трудных категорий (механический гул и погода) из meta["hard"].

    Список зафиксирован при сборке кэша, а не продублирован здесь: копия
    разъехалась бы с данными при первой же пересборке, и метрика молча
    начала бы считаться по другому пулу.
    """
    meta = np.load(os.path.join(ROOT, "cache_hard", "meta.npz"),
                   allow_pickle=True)
    return meta["hard"]
```

Проверить: `python -c "import sys; sys.path.insert(0,'.'); from airadar.bench.corpus import hard_categories; print(len(hard_categories()), sorted(hard_categories())[:4])"` — ожидается `16` и первые категории по алфавиту.

- [ ] **Step 3: Написать report.py**

```python
"""Сборка отчёта по одному чекпоинту.

Один вызов — один JSON и один markdown. Все числа идут с доверительными
интервалами: число без интервала сравнивать между прогонами нельзя, это
установлено измерением (epoch-to-epoch разброс полевого recall ~9 пп при
монотонно растущем auc_hard).
"""

import os
import json
import numpy as np

from airadar.bench import corpus, decision, ladder, transfer, strata
from airadar.bench.ci import block_bootstrap, ci

FA_BUDGET = 1.0          # тревог в час: оператор терпит одну, десять — выключит
NOMINAL_FAR = 0.01
OFF_DELTA = 1.0


def run_bench(scorer, name, seed=0):
    rep = {"name": name, "hop_s": scorer.hop_s, "context_s": scorer.context_s}

    # 1. непрерывный трудный фон: рабочая точка и FA/час.
    #    Сначала regroup восстанавливает исходные клипы (окна одной группы
    #    смежны встык), и только потом клипы сшиваются кроссфейдом. Склейка
    #    напрямую из 0.5-секундных окон дала бы стык каждые 0.45 с, и при
    #    контексте 0.5 с его задевало бы каждое окно.
    # Фильтр по трудным категориям обязателен. В cache_hard 58 категорий, а
    # трудных (механический гул и погода) — 16, список лежит в meta["hard"].
    # Без фильтра рабочая точка считалась бы по пулу, где собственно моторных
    # и винтовых звуков единицы процентов, а остальное — лай, плач и стройка.
    # Такое среднее насыщается по построению, и именно на это указывает
    # docs/metrics-plan.md §0.5.
    hard, cats, grp = corpus.hard_holdout(cat_filter=corpus.hard_categories())
    clips = corpus.regroup(hard, grp)
    track, seams = corpus.stitch(clips)
    mask = corpus.seam_mask(_n(scorer, len(track)), seams,
                            scorer.context_s, scorer.hop_s)
    if mask.sum() < 0.2 * len(mask):
        raise RuntimeError(
            f"стыки съели {100*(1-mask.mean()):.0f}% окон при контексте "
            f"{scorer.context_s} с — нужен непрерывный корпус из исходников "
            f"(этап 4), а не склейка нарезанного кэша")
    lg_bg = decision.smooth(scorer.score(track), scorer.hop_s)
    on = decision.threshold_for_fa(lg_bg, scorer.hop_s, FA_BUDGET,
                                   OFF_DELTA, mask=mask)
    rep["operating_point"] = {
        "fa_budget_per_hour": FA_BUDGET, "on": on, "off": on - OFF_DELTA,
        "fa_actual": decision.fa_per_hour(lg_bg, scorer.hop_s, on,
                                          on - OFF_DELTA, mask=mask),
        "background_hours": float(len(track) / corpus.SR / 3600.0),
    }

    # 2. лестница SNR50 по каждой полевой записи отдельно.
    #    Усреднять по записям нельзя: у них разная основная частота, и
    #    среднее спрятало бы, что одна пропускается целиком.
    pool = [clips[i] for i in np.linspace(0, len(clips) - 1, 64).astype(int)]
    rep["snr50"] = {}
    for nm, audio in corpus.field_records().items():
        curve = ladder.p_detect_curve(scorer, audio, pool, on, on - OFF_DELTA,
                                      seed=seed)
        rep["snr50"][nm] = {
            "curve": [float(v) for v in curve],
            "snrs": [float(v) for v in ladder.SNR_GRID],
            "snr50_db": ladder.snr50(ladder.SNR_GRID, curve),
        }

    # 3. auc_fh и медианный перцентиль с блочным CI
    lg_hard = scorer.score(track)[mask]
    rep["field"] = {}
    for nm, audio in corpus.field_records().items():
        lg_f = scorer.score(audio)
        rep["field"][nm] = {
            "n_windows": int(len(lg_f)),
            "auc_fh": _auc(lg_f, lg_hard),
            "auc_fh_ci": ci(block_bootstrap(lg_f, lambda v: _auc(v, lg_hard),
                                            n_boot=400, block=12, seed=seed)),
            "median_pct": float(np.mean(lg_hard[None, :] < np.median(lg_f))),
        }

    # 4. перенос порога: фон DADS (лёгкий, лабораторный) -> трудные негативы
    Xd, yd, spd, _cd, gd = corpus.load_cache("cache_dads")
    sel = np.sort(np.flatnonzero((yd == 0) & (spd != 0))[:20000])
    Xn = np.ascontiguousarray(Xd[sel]).astype(np.float32) / 32768.0
    lg_dads = np.concatenate([scorer.score(t)
                              for t in corpus.regroup(Xn, gd[sel])])
    rep["transfer"] = transfer.transfer_error(lg_dads, lg_hard, NOMINAL_FAR)
    rep["transfer"]["drift_p99"] = transfer.drift(lg_dads, lg_hard)

    # 5. recall по f0-полосам, отчётная величина — худшая полоса.
    #    Порог берётся из рабочей точки FA/час, а не из перцентиля лёгких
    #    позитивов: порог, откалиброванный по квадрокоптеру с близкой
    #    дистанции, к тяжёлому дрону отношения не имеет.
    try:
        idx, f0 = strata.load_f0_estimates()
    except (FileNotFoundError, KeyError) as e:
        rep["strata"] = {"error": str(e)}
    else:
        keep = np.isin(idx, np.flatnonzero((yd == 1) & (spd != 0)))
        idx, f0 = idx[keep], f0[keep]
        order = np.argsort(idx)
        idx, f0 = idx[order], f0[order]
        Xp = np.ascontiguousarray(Xd[idx]).astype(np.float32) / 32768.0
        lg_pos = score_windows(scorer, Xp)
        rec = strata.recall_by_band(lg_pos, np.ones(len(lg_pos), int), f0, on)
        name, val = strata.worst_band(rec)
        rep["strata"] = {"by_band": rec, "worst_band": name, "worst_recall": val,
                         "n_windows": int(len(lg_pos))}
    return rep


def _n(scorer, n_samples):
    from airadar.bench.scorer import n_scores
    return n_scores(n_samples, scorer.context_s, scorer.hop_s)


def score_windows(scorer, X):
    """По одной оценке на изолированное окно [N, win].

    Если контекст скорера длиннее окна, окно зацикливается до контекста.
    Это искусственно, и для f0-страт допустимо только потому, что все модели
    получают одну и ту же обработку, а сравниваются они между собой. Для
    будущего 4-секундного скорера страту надо будет пересобрать на клипах —
    отмечено в спецификации как ограничение этапа 0.
    """
    win = X.shape[1]
    ctx = int(round(scorer.context_s * corpus.SR))
    out = np.empty(len(X), np.float32)
    for i, w in enumerate(X):
        a = w if win >= ctx else np.tile(w, int(np.ceil(ctx / win)))[:ctx]
        s = scorer.score(a.astype(np.float32))
        out[i] = s[len(s) // 2]
    return out


def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def write_report(rep, out_dir="bench_out"):
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, f"{rep['name']}.json")
    mp = os.path.join(out_dir, f"{rep['name']}.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(_markdown(rep))
    return jp, mp


def _markdown(rep):
    op = rep["operating_point"]
    out = [f"# {rep['name']}", "",
           f"Рабочая точка: бюджет {op['fa_budget_per_hour']} тревог/час, "
           f"фактически {op['fa_actual']:.2f} на {op['background_hours']:.2f} ч фона.",
           f"Порог включения {op['on']:.3f}, выключения {op['off']:.3f}.", "",
           "| запись | SNR50, дБ | auc_fh | 95% CI auc_fh |", "|---|---|---|---|"]
    for nm in rep["snr50"]:
        s = rep["snr50"][nm]["snr50_db"]
        f_ = rep["field"][nm]
        lo, hi = f_["auc_fh_ci"]
        out.append(f"| {nm} | {s:.1f} | {f_['auc_fh']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    t = rep["transfer"]
    out += ["", f"Перенос порога DADS→трудные: номинал {t['far_nominal']:.3f}, "
                f"факт {t['far_actual']:.3f}, отношение **{t['ratio']:.1f}×**, "
                f"дрейф p99 {t['drift_p99']:.2f} σ."]
    st = rep.get("strata", {})
    if "by_band" in st:
        out += ["", "| f0-полоса, Гц | recall |", "|---|---|"]
        out += [f"| {k} | {v:.3f} |" for k, v in sorted(st["by_band"].items())]
        out += ["", f"Худшая полоса: **{st['worst_band']} Гц, "
                    f"recall {st['worst_recall']:.3f}** "
                    f"({st['n_windows']} окон). Это и есть отчётная величина — "
                    f"среднее по полосам скрывает именно тяжёлые машины."]
    else:
        out += ["", f"Страты не посчитаны: {st.get('error', 'нет данных')}"]
    return "\n".join(out) + "\n"
```

- [ ] **Step 4: Написать cli/bench.py**

```python
"""Прогон харнеса по одному чекпоинту.

    CUDA_VISIBLE_DEVICES= python cli/bench.py \
        --model models/dronenet_local.pt --name dronenet_local
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from airadar.bench.scorer import LegacyScorer
from airadar.bench.report import run_bench, write_report

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rep = run_bench(LegacyScorer(a.model, a.device), a.name, a.seed)
jp, mp = write_report(rep)
print(open(mp, encoding="utf-8").read())
print(f"записано: {jp}  {mp}")
```

- [ ] **Step 5: Прогнать на нынешнем чекпоинте — получить базовую цифру**

Run:
```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python cli/bench.py \
    --model models/dronenet_local.pt --name dronenet_local 2>&1 | tee logs/bench_baseline.log
```
Expected: markdown-таблица с SNR50 по каждой полевой записи, `auc_fh` с интервалами (ожидаемо около 0.94 и 0.76 по прошлым измерениям), отношение переноса порога заметно больше 1.

Если `snr50_db` вышел `nan` на обеих записях — кривая не пересекает 0.5, и это сам по себе результат: при бюджете 1 тревога/час нынешняя модель не обнаруживает полевые записи ни на одной ступени. Записать это как базовую цифру, не подкручивая бюджет ради красивого числа.

- [ ] **Step 6: Коммит**

```bash
git add airadar/bench/report.py cli/bench.py bench_out/dronenet_local.json bench_out/dronenet_local.md logs/bench_baseline.log
git commit -m "bench: отчёт и базовая цифра для dronenet_local

Первое измерение по протоколу спецификации: рабочая точка задана бюджетом
тревог в час на непрерывном фоне, а не долей окон; SNR50 считается по каждой
полевой записи отдельно; все числа с блочными интервалами."
```

---

### Task 9: D0 — смежны ли клипы DADS

**Files:**
- Create: `airadar/diag/__init__.py` (пустой), `airadar/diag/dads_contiguity.py`
- Create: `cli/diag.py`

**Interfaces:**
- Consumes: `hf_sources.{local_shard, shards, SRC_DADS, to_mono_16k}`
- Produces:
  - `seam_jump(a: np.ndarray, b: np.ndarray) -> float` — скачок на стыке в единицах типичного межотсчётного перепада `a`
  - `verdict(adj: np.ndarray, ctl: np.ndarray) -> tuple[bool, str]` — решение и формулировка
  - CLI: `python cli/diag.py dads-contiguity --shard 0 --n 300`

**Замечание для реализатора:** `hf_sources.read_shard` возвращает `Rec`, который **не содержит** номер клипа — он расходуется внутри `dads_group` и теряется. Для D0 нужен собственный проход по parquet, сохраняющий `idx` из `r["audio"]["path"]`.

- [ ] **Step 1: Написать падающий selfcheck**

```python
"""D0: являются ли соседние индексы DADS кусками одной записи?

От ответа зависит доступный контекст. Клипы DADS длиной 0.6 с; если соседние
номера смежны, 27 часов позитивов восстанавливаются в непрерывные дорожки и
накопление по 4 с даёт заявленные 9 дБ. Если нет — контекст для 86% позитивов
ограничен, выигрыш падает примерно до 5 дБ, и обучение идёт через MIL с
позитивом, положенным в случайное место фона.

Статистика: скачок на стыке в единицах типичного межотсчётного перепада.
У смежного аудио стык ничем не отличается от любой другой точки, отношение
около 1. У несвязанных клипов скачок порядка полного размаха сигнала, а он
для низкочастотно-доминированного звука много больше межотсчётного перепада.
"""

import sys
import numpy as np


def selfcheck():
    # смежные куски одного синуса: стык неотличим от внутренней точки
    t = np.arange(4000) / 16000.0
    x = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    a, b = x[:2000], x[2000:]
    assert seam_jump(a, b) < 3.0, seam_jump(a, b)

    # несвязанные куски: другая фаза даёт скачок много больше межотсчётного.
    # Инверсия (-x) для контроля не годится: обе последовательности проходят
    # через ноль в одной точке, и скачок выходит обманчиво малым.
    c = np.sin(2 * np.pi * 200 * t + 1.5).astype(np.float32)[:2000]
    assert seam_jump(a, c) > 10.0, seam_jump(a, c)

    # решение принимается по разделению распределений, а не по одной паре
    adj = np.full(200, 1.2)
    ctl = np.full(200, 40.0)
    ok, msg = verdict(adj, ctl)
    assert ok and "смежны" in msg, msg
    ok2, msg2 = verdict(ctl, ctl)
    assert not ok2, msg2

    print("dads_contiguity selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать, убедиться что падает**

Run: `python -m airadar.diag.dads_contiguity --selfcheck`
Expected: FAIL, `NameError: name 'seam_jump' is not defined`

- [ ] **Step 3: Реализовать статистику и решение**

```python
def seam_jump(a, b):
    """Скачок на стыке a|b в единицах типичного межотсчётного перепада a."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = np.median(np.abs(np.diff(a))) + 1e-9
    return float(abs(b[0] - a[-1]) / scale)


def verdict(adj, ctl):
    """Решение по разделению двух распределений скачков.

    Порог намеренно грубый: нас интересует разница на порядок, а не на
    проценты. Промежуточный результат — тоже результат, и он должен
    называться промежуточным, а не округляться в удобную сторону.
    """
    adj, ctl = np.asarray(adj, np.float64), np.asarray(ctl, np.float64)
    ma, mc = float(np.median(adj)), float(np.median(ctl))
    ratio = mc / (ma + 1e-9)
    if ratio > 5.0 and ma < 5.0:
        return True, (f"соседние клипы смежны: медиана скачка {ma:.2f} против "
                      f"{mc:.2f} у контроля (в {ratio:.1f} раз)")
    if ratio < 1.5:
        return False, (f"соседние клипы НЕ смежны: медиана скачка {ma:.2f}, "
                       f"контроль {mc:.2f} — распределения совпадают")
    return False, (f"промежуточный результат: медиана {ma:.2f}, контроль "
                   f"{mc:.2f} (в {ratio:.1f} раз). Смежна лишь часть пар — "
                   f"нужен разбор по группам, не обобщать")
```

- [ ] **Step 4: Прогнать, убедиться что проходит**

Run: `python -m airadar.diag.dads_contiguity --selfcheck`
Expected: PASS

- [ ] **Step 5: Реализовать проход по шарду**

```python
import os
import re

_NUM = re.compile(r"(\d+)")


def scan_shard(local_path, n_pairs=300, seed=0):
    """Собирает скачки для соседних пар и для контрольных случайных пар."""
    import pyarrow.parquet as pq
    import io
    import soundfile as sf

    pf = pq.ParquetFile(local_path)
    clips = {}
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=["audio", "label"]).to_pylist():
            if int(r["label"]) != 1:
                continue
            m = _NUM.search(r["audio"]["path"] or "")
            if not m:
                continue
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if sr != 16000 or x.ndim != 1:
                continue
            clips[int(m.group(1))] = x
            if len(clips) > n_pairs * 4:
                break
        if len(clips) > n_pairs * 4:
            break

    keys = sorted(clips)
    adj = [seam_jump(clips[k], clips[k + 1])
           for k in keys[:-1] if k + 1 in clips][:n_pairs]
    rng = np.random.default_rng(seed)
    ctl = []
    for _ in range(len(adj)):
        i, j = rng.integers(0, len(keys), 2)
        if abs(keys[i] - keys[j]) > 10:
            ctl.append(seam_jump(clips[keys[i]], clips[keys[j]]))
    return np.array(adj), np.array(ctl)
```

- [ ] **Step 6: Написать cli/diag.py и запустить D0**

```python
"""Диагностики, не требующие обучения.

    python cli/diag.py dads-contiguity --shard 0 --n 300
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["dads-contiguity"])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()

import numpy as np
import hf_sources
from airadar.diag.dads_contiguity import scan_shard, verdict

rel = hf_sources.shards(hf_sources.SRC_DADS)[a.shard]
path = hf_sources.local_shard(hf_sources.SRC_DADS, rel)
adj, ctl = scan_shard(path, a.n)
print(f"шард {rel}: пар соседних {len(adj)}, контрольных {len(ctl)}")
print(f"скачок соседних:  медиана {np.median(adj):.2f}  p90 {np.quantile(adj, .9):.2f}")
print(f"скачок контроля:  медиана {np.median(ctl):.2f}  p90 {np.quantile(ctl, .9):.2f}")
ok, msg = verdict(adj, ctl)
print(f"\nD0: {msg}")
```

Run:
```bash
python cli/diag.py dads-contiguity --shard 0 --n 300 2>&1 | tee logs/d0_contiguity.log
```
Expected: одна из трёх формулировок `verdict`. Прогнать **на трёх разных шардах** (0, 10, 25) — если вердикты расходятся, это само по себе находка, и она идёт в отчёт как «смежность зависит от источника».

- [ ] **Step 7: Записать результат в спецификацию**

Дописать в `docs/superpowers/specs/2026-07-26-architecture-redesign-design.md`, §5.4, абзац с фактическим ответом D0, датой и номерами прогнанных шардов. Это меняет ожидаемую величину выигрыша от накопления и, следовательно, план этапа 3.

- [ ] **Step 8: Коммит**

```bash
git add airadar/diag/ cli/diag.py logs/d0_contiguity.log \
        docs/superpowers/specs/2026-07-26-architecture-redesign-design.md
git commit -m "diag: D0 — смежность соседних клипов DADS

От ответа зависит доступный контекст: смежны — 27 часов позитивов
восстанавливаются в дорожки и накопление даёт 9 дБ; нет — контекст для 86%
позитивов ограничен и обучение идёт через MIL с позитивом в случайном месте
фона. Результат записан в спецификацию, §5.4."
```

---

## Проверка перед закрытием этапа

- [ ] `python cli/selfcheck.py` — код возврата 0, все модули зелёные
- [ ] `bench_out/dronenet_local.json` существует и содержит непустые `snr50`, `field`, `transfer`, `strata` (без ключа `error`)
- [ ] `logs/d0_contiguity.log` содержит вердикт по трём шардам
- [ ] §5.4 спецификации дополнена фактическим ответом D0
- [ ] полевые записи не использовались для подбора ни одного порога и ни одного гиперпараметра

## Что этот этап сознательно не делает

- Не трогает `train.py`, `prep_hf.py`, `detect.py`, `eval.py` — старый конвейер остаётся рабочим до этапа 1.
- Не строит непрерывный фон из UrbanSound8K/ESC-50 отдельно от `cache_hard`: в `cache_hard` они уже лежат нарезанными, и склейка через `stitch` даёт достаточную дорожку для FA/час. Полноценный корпус из исходников — этап 4, когда появится полевой фон.
- Не считает псевдоисточники кластеризацией (спецификация §5.5) — это требует манифеста, то есть этапа 1.
