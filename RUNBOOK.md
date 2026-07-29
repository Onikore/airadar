# Как запустить

Актуально на пайплайн второго поколения (`cli/` + `airadar/`, DroneNet2).
Контекст — [README.md](README.md), приоритеты работ — [NEXT_STEPS.md](NEXT_STEPS.md).
Легаси-пайплайн первого поколения (`train.py`, `eval.py`, `detect.py`,
`web.py` в корне) — раздел «Легаси-пайплайн» внизу.

Обучение сейчас идёт **локально на GPU**, не в Colab — данные (`data/manifest.parquet`,
`data/clips.bin`) и модель помещаются на диск и в память одной рабочей машины.

---

## Проверить, что сейчас происходит

Windows (PowerShell/Git Bash):

```bash
tail -f logs/<run_name>.log            # если обучение запущено с `tee`/перенаправлением в файл
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
wmic process where "name='python.exe'" get ProcessId,CommandLine   # что именно запущено
taskkill //PID <pid> //F               # остановить конкретный процесс
```

`cli/train.py` не пишет прогресс построчно внутри эпохи — только одну строку
по завершении эпохи (`train_loss`, `val_loss`, время, `<- saved`). Тишина в
логе несколько минут — это нормально, не зависание.

---

## Установка на новой машине

```bash
git clone https://github.com/Onikore/airadar.git
cd airadar

pip install numpy scipy soundfile pyarrow huggingface_hub flask nnAudio==0.3.4
pip install torch --index-url https://download.pytorch.org/whl/cu121   # под свою версию CUDA
# без GPU — CPU-сборка:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

python cli/selfcheck.py     # вся логика пакета без данных и без GPU, 33+ проверки — обязательно прогнать первым
```

Нет `requirements.txt` — список выше собран прямым обходом импортов, версии
не запиновены нигде в коде кроме `nnAudio==0.3.4` (число кадров признака
измерено под конкретную версию, см. `airadar/features/cqt.py`).

**Токен HuggingFace обязателен** — `hub.py:token()` явно падает без него,
анонимный доступ не поддержан кодом (даже если все четыре датасета публичные):

```bash
export HF_TOKEN=hf_...   # создать на huggingface.co/settings/tokens, права read достаточно
```

Если какой-то из четырёх датасетов (DADS, DroneAudioSet, UrbanSound8K,
ESC-50 — см. README, раздел «Данные») окажется gated — принять условия на
странице датасета на HF тем же аккаунтом, которому выписан токен.

---

## Пайплайн с нуля

Нужен `data/manifest.parquet` + `data/clips.bin` (гигабайты, не в git) —
собираются из четырёх источников HuggingFace скачиванием на диск.

```bash
python cli/build_manifest.py --limit 1      # один шард на источник, быстрая проверка
python cli/build_manifest.py                # полная сборка — часы, IO-связано
                                             # чекпоинт по шардам — безопасно Ctrl+C и перезапустить

python cli/manifest_audit.py                # аудит целостности манифеста/clips.bin
python cli/label_manifest_f0.py             # f0-разметка для вспомогательных голов (не обязательна для обучения)
```

---

## Обучение

```bash
python cli/train.py --epochs 15 --bs 32 --run-name my_run \
    --save-every-epoch --seed 0 --num-workers 4
```

- `--save-every-epoch` — обязателен, если планируется bench-sweep (см. ниже):
  без него сохраняется только автовыбранный по `val_loss` "best", а
  `val_loss` систематически расходится с реальным полевым качеством
  ([findings](docs/2026-07-29-f0-extension-findings.md)).
- `--num-workers 4` — без этого случайный доступ к `clips.bin` (гигабайты)
  упирается в промах кэша страниц ОС, эпоха может идти в разы дольше
  ([findings](docs/2026-07-26-local-training-findings.md)). На Windows
  `num_workers>0` требует, чтобы `collate_fn` был picklable (уже так —
  `functools.partial` над функцией модуля, не лямбда).
- Резюмирования с чекпоинта нет (`--resume` не реализован). Расписание
  `OneCycleLR` посчитано под фиксированное число эпох — прервать и
  доучить нельзя, только новый прогон.
- Проверить перед долгим прогоном, что GPU не занят другим процессом
  (`nvidia-smi`) — в частности, забытым `cli/webdemo.py` с загруженной
  моделью.

---

## Bench — сравнение чекпоинтов на реальных данных

```bash
python cli/bench.py --model models/my_run_ep008.pt --name my_run_ep008 --arch dronenet2
# -> bench_out/my_run_ep008.json, bench_out/my_run_ep008.md
```

`--arch dronenet2` обязателен для новых чекпоинтов (`--arch legacy` — для
`dronenet.pt`/`dronenet_local.pt` первого поколения).

**Bench-sweep по всем эпохам** (нужен для честного выбора чекпоинта, см.
выше про `val_loss`):

```bash
for i in $(seq -w 1 15); do
  python cli/bench.py --model models/my_run_ep0$i.pt --name my_run_ep0$i --arch dronenet2
done
```

Отчёт (`.md`) содержит `auc_fh` по каждой полевой записи и честно сообщает,
если рабочую точку (порог FA/час) посчитать не удалось — «недостаточно фона»,
это ожидаемо на текущем корпусе (см. NEXT_STEPS, шаг 3), не баг.

**AUC — не абсолютная уверенность.** Для прямой проверки, перейдёт ли модель
порог 0.5 (тот, которым `cli/webdemo.html` красит статус), нужен отдельный
скрипт поверх `DroneNet2Scorer.score()` — в репозитории такого нет, пример
см. [docs/2026-07-29-f0-extension-findings.md](docs/2026-07-29-f0-extension-findings.md).

---

## Живой тест — веб-демка

```bash
python cli/webdemo.py --model models/dronenet2_seed0_f0ext_true_best.pt
# -> http://127.0.0.1:5000
```

Два режима: микрофон в браузере (Web Audio API, есть переключатель "raw mic" —
отключает echo cancellation/noise suppression/auto-gain, которые браузер
включает по умолчанию и которые искажают акустический признак) и загрузка
файла (обходит микрофон/динамики целиком, точнее для проверки конкретной
записи).

**Не забывать останавливать процесс после теста** — держит модель на GPU,
конкурирует за память с параллельным обучением (см. `nvidia-smi` выше).

---

## Диагностика (CPU, можно параллельно с обучением)

```bash
python cli/diag.py dads-contiguity --shard <N>   # смежность индексов DADS
python cli/selfcheck.py                          # вся логика пакета
```

Разведочные скрипты старого пайплайна (`evalx/`) всё ещё работают на новых
данных частично — `f0_survey.py`, `feat_visibility.py` завязаны на
конкретные пути `cache_dads`/`cache_hard`, которых больше нет; перед
использованием проверить актуальность путей.

---

## Легаси-пайплайн (первое поколение, `DroneNet`)

Не активная разработка, но код рабочий — `models/dronenet_local.pt` остаётся
базовой цифрой сравнения (`bench_out/dronenet_local.json`).

```bash
python3 -u train.py 2>&1 | tee logs/train.log
python3 eval.py
python3 detect.py --file field/drone_video1.wav
python3 web.py --no-audio
```

Секция «Обучение в Colab» и `notebooks/` из истории проекта — в `archive/`,
не актуальны: обучение теперь локальное на GPU, Colab не используется.

Каждый скрипт легаси-пайплайна — `--selfcheck`:

```bash
for f in train.py eval.py detect.py web.py spectrum.py \
         diag_leak.py diag_hard.py diag_compare.py hub.py prep_hf.py; do
    python3 "$f" --selfcheck
done
```

---

## Окружение

Обучение — локальный GPU (RTX-класс, 12ГБ). CPU-режим для проверки кода без
карты: `torch.cuda.is_available()` в `airadar/train/loop.py` сам падает на
`cpu`, `cli/selfcheck.py` проходит без данных и без GPU.

`data/clips.bin` — гигабайты, случайный доступ; узкое место — не вычисления
на GPU, а промах кэша страниц ОС на большом файле (решается
`num_workers`, см. выше).
