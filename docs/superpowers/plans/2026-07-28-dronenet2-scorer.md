# DroneNet2 Scorer + первый реальный бенч — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Дать `DroneNet2` (этап 3а-в) фасад `Scorer` (протокол этапа 0),
подключить его к уже существующему, уже отревьюженному `run_bench()`
(`airadar/bench/report.py`) без единой правки внутри него, и прогнать
харнес на реальном (пусть недообученном) чекпоинте — получить настоящий
SNR50 на обеих полевых записях, сравнимый с базовой цифрой
`dronenet_local.pt` (SNR50=NaN, `bench_out/dronenet_local.json`).

**Architecture:** `run_bench()` уже не знает, что внутри модели — весь
контракт держится на `Scorer.score(audio) -> logits`. Поля SNR50/`auc_fh`
(§2-3 отчёта) читают `corpus.field_records()` — реальные WAV, без
привязки к старому `cache_dads`/`cache_hard`. Рабочая точка/перенос
порога/f0-страты (§1,4,5) по-прежнему числятся по СТАРОМУ кэшу — это
известное, задокументированное ограничение отчёта (см. `план не делает`
внизу), не то, что чинит этот план.

**Tech Stack:** Продолжает `airadar/bench/{scorer,report}.py` (этап 0),
`airadar/train/checkpoint.py`, `airadar/features/frontend.py`,
`airadar/models/dronenet2.py` (этап 3а-в).

## Global Constraints

- `Scorer` — протокол `airadar/bench/scorer.py`: `hop_s`, `context_s`,
  `score(audio: float32[N]) -> float32[n_scores(N, context_s, hop_s)]`,
  ЛОГИТЫ, не вероятности.
- Каждый модуль имеет `--selfcheck` (§8, конвенция проекта).
- `cache_dads`, `cache_hard`, `field/drone_video{1,2}.wav`,
  `bench_out/dronenet_local.json` уже на диске — проверено перед
  написанием этого плана, план на них не покушается.

---

### Task 1: `DroneNet2Scorer` в `airadar/bench/scorer.py`

**Files:**
- Modify: `airadar/bench/scorer.py`

**Interfaces:**
- Consumes: `airadar.train.checkpoint.load_checkpoint`,
  `airadar.config.{FeatureCfg, ModelCfg}`, `airadar.features.frontend.
  Frontend`, `airadar.models.dronenet2.DroneNet2`.
- Produces: `airadar.bench.scorer.DroneNet2Scorer` — класс с `hop_s`
  (аргумент конструктора, по умолчанию `1.0`), `context_s` (из чекпоинта,
  `TrainCfg.target_samples / FeatureCfg.sr` — не выбор этого класса, а
  контекст, под который спроектирована архитектура), `score(audio, bs=16)
  -> np.float32[n]`.

`hop_s=1.0` (по умолчанию, переопределяемо) — инженерное решение этого
плана: спецификация фиксирует шаг РАНТАЙМА (§7, отдельная стадия), не
шаг офлайн-бенча для новой модели. 1.0с даёт разумное временное
разрешение FA/час, не пересчитывая CQT почти на каждый отсчёт (контекст
12с у этой модели, не 0.5с как у `LegacyScorer`).

- [ ] **Step 1: Дописать `DroneNet2Scorer` в конец `airadar/bench/scorer.py`**

```python
class DroneNet2Scorer:
    """DroneNet2 (этап 3) за фасадом Scorer.

    context_s берётся из чекпоинта (TrainCfg.target_samples / FeatureCfg.sr,
    обычно 12.0с — 8с истории + 4с окна модели), не задаётся руками: это
    контекст, для которого архитектура спроектирована (этап 2/3а), а не
    произвольный выбор бенча.
    """

    def __init__(self, ckpt_path, device="cpu", hop_s=1.0):
        import torch
        from airadar.train.checkpoint import load_checkpoint
        from airadar.config import FeatureCfg, ModelCfg
        from airadar.features.frontend import Frontend
        from airadar.models.dronenet2 import DroneNet2

        self._torch = torch
        ck = load_checkpoint(ckpt_path, device=device)
        feature_cfg = FeatureCfg(**ck["feature_cfg"])
        model_cfg = ModelCfg(**ck["model_cfg"])

        self.hop_s = hop_s
        self.context_s = ck["train_cfg"]["target_samples"] / feature_cfg.sr

        self.frontend = Frontend(feature_cfg).to(device)
        self.model = DroneNet2(model_cfg).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device
        self._sr = feature_cfg.sr

    def score(self, audio, bs=16):
        torch = self._torch
        ctx = int(round(self.context_s * self._sr))
        hop = int(round(self.hop_s * self._sr))
        n = n_scores(len(audio), self.context_s, self.hop_s, sr=self._sr)
        if n == 0:
            return np.zeros(0, np.float32)
        win = np.stack([audio[i * hop:i * hop + ctx] for i in range(n)])
        out = np.empty(n, np.float32)
        with torch.no_grad():
            for i in range(0, n, bs):
                x = torch.from_numpy(win[i:i + bs]).to(self.device).float()
                feat = self.frontend(x)
                feat = self.frontend.last_model_frames(feat)
                out[i:i + bs] = self.model(feat)["clip_logit"].cpu().numpy()
        return out
```

- [ ] **Step 2: Дополнить `selfcheck` в `airadar/bench/scorer.py`**

Добавить перед `print("scorer selfcheck ok")`:

```python
    # DroneNet2Scorer: синтетический чекпоинт (случайные веса, как в
    # airadar/train/checkpoint.py:selfcheck), проверка контракта Scorer
    import tempfile
    import torch
    from airadar.train.checkpoint import save_checkpoint
    from airadar.config import FeatureCfg, ModelCfg, AugCfg, TrainCfg
    from airadar.models.dronenet2 import DroneNet2

    with tempfile.TemporaryDirectory() as d:
        manifest_path = os.path.join(d, "fake_manifest.bin")
        with open(manifest_path, "wb") as f:
            f.write(b"fake")
        ckpt_path = os.path.join(d, "dn2.pt")
        model = DroneNet2(ModelCfg())
        save_checkpoint(ckpt_path, model, opt=None, sched=None, epoch=0,
                        feature_cfg=FeatureCfg(), model_cfg=ModelCfg(),
                        aug_cfg=AugCfg(), train_cfg=TrainCfg(),
                        manifest_path=manifest_path)

        s2 = DroneNet2Scorer(ckpt_path, device="cpu")
        assert s2.context_s == TrainCfg().target_samples / FeatureCfg().sr
        assert s2.hop_s == 1.0
        # 15с > context_s (12с) -> хотя бы одна оценка реально считается
        assert check_scorer(s2, n_samples=SR * 15) >= 1

        s2_custom_hop = DroneNet2Scorer(ckpt_path, device="cpu", hop_s=2.0)
        assert s2_custom_hop.hop_s == 2.0
        assert check_scorer(s2_custom_hop, n_samples=SR * 15) >= 1
```

Добавить `import os` в начало файла, если его там ещё нет (нужен для
`os.path.join`).

- [ ] **Step 3: Прогнать селфчек**

Run: `python -m airadar.bench.scorer --selfcheck`
Expected: `scorer selfcheck ok`

- [ ] **Step 4: Commit**

```bash
git add airadar/bench/scorer.py
git commit -m "bench: DroneNet2Scorer — фасад Scorer над DroneNet2, context_s из чекпоинта"
```

---

### Task 2: `--arch` в `cli/bench.py`

**Files:**
- Modify: `cli/bench.py`

**Interfaces:**
- Consumes: `airadar.bench.scorer.{LegacyScorer, DroneNet2Scorer}`.

- [ ] **Step 1: Добавить выбор скорера по архитектуре**

Заменить содержимое `cli/bench.py`:

```python
"""Прогон харнеса по одному чекпоинту.

    CUDA_VISIBLE_DEVICES= python cli/bench.py \
        --model models/dronenet_local.pt --name dronenet_local
    python cli/bench.py \
        --model models/dronenet2_last.pt --name dronenet2_smoke --arch dronenet2
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from airadar.bench.scorer import LegacyScorer, DroneNet2Scorer
from airadar.bench.report import run_bench, write_report

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arch", choices=["legacy", "dronenet2"], default="legacy")
ap.add_argument("--hop-s", type=float, default=1.0,
                help="только для --arch dronenet2 — шаг между окнами бенча")
a = ap.parse_args()

if a.arch == "legacy":
    scorer = LegacyScorer(a.model, a.device)
else:
    scorer = DroneNet2Scorer(a.model, a.device, hop_s=a.hop_s)

rep = run_bench(scorer, a.name, a.seed, model_path=a.model)
jp, mp = write_report(rep)

# Отчёт написан на диск (файлы уже в UTF-8) до этой точки — печать в консоль
# не должна ронять прогон. На Windows sys.stdout по умолчанию открыт в
# кодовой странице консоли (cp1251/cp866), которая не знает символов вроде
# '→' и роняет print с UnicodeEncodeError уже ПОСЛЕ того, как отчёт записан.
# Явно переключаем поток на UTF-8, а всё, что всё-таки не отобразится
# (легаси-консоль без UTF-8), заменяем, а не падаем.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(open(mp, encoding="utf-8").read())
print(f"записано: {jp}  {mp}")
```

- [ ] **Step 2: Проверить, что старый режим (`--arch legacy`, по умолчанию) не сломался**

Run: `python cli/bench.py --model models/dronenet_local.pt --name dronenet_local_recheck`
Expected: отчёт печатается, файлы `bench_out/dronenet_local_recheck.{json,md}`
создаются, числа совпадают с уже записанными в `bench_out/dronenet_local.json`
(тот же чекпоинт, тот же код §1-5, только имя другое) — SNR50 не определён
(NaN) на обеих записях, как и раньше.

Run:
```bash
python -c "
import json
old = json.load(open('bench_out/dronenet_local.json'))
new = json.load(open('bench_out/dronenet_local_recheck.json'))
assert old['snr50']['defined'] == new['snr50']['defined']
assert old['operating_point']['threshold_calibrated'] == new['operating_point']['threshold_calibrated']
print('старый режим cli/bench.py не сломан, отчёты согласованы')
"
```
Expected: `старый режим cli/bench.py не сломан, отчёты согласованы`

- [ ] **Step 3: Удалить контрольный отчёт (не нужен в git)**

```bash
rm -f bench_out/dronenet_local_recheck.json bench_out/dronenet_local_recheck.md
```

- [ ] **Step 4: Commit**

```bash
git add cli/bench.py
git commit -m "cli: bench.py --arch {legacy,dronenet2} — выбор Scorer по архитектуре"
```

---

### Task 3: Первый реальный прогон харнеса на `DroneNet2`

Не код — проверка фактом. Чекпоинт `models/dronenet2_last.pt` обучен на
2000 из 191196 строк (~1%), 3 эпохи (этап 3в) — это НЕ качественный
результат, а первый случай, когда харнес вообще может посчитать SNR50 на
новой архитектуре. Число почти наверняка будет плохим или неопределённым
(NaN) — это ожидаемо и не повод останавливаться, само по себе включение
харнеса в контур — уже проверяемый факт.

- [ ] **Step 1: Прогнать харнес на смоук-чекпоинте**

```bash
python cli/bench.py --model models/dronenet2_last.pt --name dronenet2_smoke --arch dronenet2 2>&1 | tee logs/bench_dronenet2_smoke.log
```

Ожидаемое время: контекст 12с, шаг 1.0с — на порядок больше вычислений на
окно, чем у `LegacyScorer` (контекст 0.5с). Если прогон не укладывается в
разумное время (больше ~30 минут), прервать и увеличить `--hop-s` (2.0
или 4.0) для этого прогона — это не меняет код, только временное
разрешение бенча, отметить в резюме задачи, каким `--hop-s` считалось.

- [ ] **Step 2: Записать реальные числа как есть, без интерпретации задним числом**

В сообщение коммита (Step 3) занести: `snr50.defined` (true/false) по
каждой записи, если true — `snr50_db` и его CI; `operating_point.
threshold_calibrated`; если харнес упал или число оказалось
неопределённым — записать причину из отчёта (`op["reason"]`), не
скрывать. Плохой результат на недообученной модели — ожидаемый,
корректный исход этой задачи, а не провал плана.

- [ ] **Step 3: Commit**

Заголовок коммита — фиксированная часть ниже; тело коммита (после
пустой строки) — реальные числа из Step 2, не шаблон и не placeholder:

```bash
git add logs/bench_dronenet2_smoke.log bench_out/dronenet2_smoke.json bench_out/dronenet2_smoke.md
git commit -m "$(cat <<'EOF'
bench: первый прогон харнеса на DroneNet2 (смоук-чекпоинт, 1% данных, 3 эпохи)

<здесь: snr50.defined и snr50_db/CI по каждой записи, или причина
неопределённости из operating_point.reason — реальные значения из
bench_out/dronenet2_smoke.json этого прогона, не шаблон>
EOF
)"
```

---

## Что эта задача НЕ делает (сознательно, не забыто)

- **Не переписывает §1/§4/§5 отчёта на манифест.** Рабочая точка (FA/час),
  перенос порога и recall по f0-полосам по-прежнему читают СТАРЫЙ
  `cache_dads`/`cache_hard` (сырые int16-окна старого кэша), а не
  `data/manifest.parquet`/`clips.bin`. Для `DroneNet2Scorer` это означает,
  что окна короче контекста (12с) зацикливаются до него
  (`report.py:score_windows`, уже существующий, задокументированный как
  временное ограничение искусственный приём) — числа §1/4/5 для новой
  архитектуры менее осмысленны, чем §2/3 (SNR50, auc_fh на реальных
  полевых записях), которые ЭТОТ план и делает главным результатом.
  Полная пересборка отчёта на манифест — отдельная задача, если
  окажется нужна после того, как появится по-настоящему обученный
  чекпоинт.
- **Не отбирает чекпоинт по SNR50/SWA во время обучения (§6.3).** Этот
  план подключает харнес СНАРУЖИ обучения (ручной прогон после факта),
  не встраивает его в `train/loop.py` как критерий отбора на каждую
  эпоху — прогон харнеса на 4.9ч трудного корпуса стоит минут, гонять
  его каждую эпоху при полном прогоне (десятки эпох) нерентабельно.
- **Не запускает полный прогон 2 seed и не делает формальное сравнение
  по правилу §6.2.** Чекпоинт этого прогона — смоук на 1% данных, сравнение
  с ним ничего не докажет про архитектуру. Формальное сравнение — после
  полного обучения на реальном объёме данных.