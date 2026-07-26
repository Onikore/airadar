# Как запустить

Актуально на состояние проекта — см. [README.md](README.md) (полная история
находок) и [NEXT_STEPS.md](NEXT_STEPS.md) (план работ по приоритету).

---

## Проверить, что сейчас происходит

```bash
tail -f logs/train_v2_clean.log      # если идёт обучение DroneNetV2
ps aux | grep python3                # какие процессы реально живы
```

Ctrl+C выходит из `tail -f`, само обучение не трогает.

Остановить обучение:

```bash
pkill -f "train2\|import train"
```

---

## Сравнить чекпоинты (DroneNet против DroneNetV2)

Честная метрика — `AUC_fh` (полевые окна против удержанных трудных
негативов), не recall: recall на записи `drone_video2.wav` часто упирается в
0.0% и перестаёт различать модели, см. README.

```bash
python3 -c "
import numpy as np, torch, train, train2
from sklearn.metrics import roc_auc_score

Xh, cat, _, hva = train.load_hard()
Xh = Xh[hva][np.isin(cat[hva], list(train.HARD_CATS))]
Xf = train.load_field()
lm = train.LogMel().to(train.DEV)

for path, cls, tag in [('models/dronenet.pt', train.DroneNet, 'DroneNet'),
                        ('models/dronenet_v2.pt', train2.DroneNetV2, 'DroneNetV2')]:
    m = cls().to(train.DEV)
    m.load_state_dict(torch.load(path, map_location=train.DEV, weights_only=False)['model'])
    m.eval()
    ph = train._probs(m, lm, Xh)
    print(f'\n{tag}:')
    for nm, w in Xf.items():
        pf = train._probs(m, lm, w)
        auc = roc_auc_score(np.r_[np.ones(len(pf)), np.zeros(len(ph))], np.r_[pf, ph])
        print(f'  {nm}: AUC_fh={auc:.4f}  медиана p={np.median(pf):.4f}')
"
```

Для доверительных интервалов вместо точечной оценки — `evalx/field_ci.py`
(список моделей задан константой `MODELS` в начале файла, дополнить при
необходимости):

```bash
CUDA_VISIBLE_DEVICES= python3 evalx/field_ci.py
```

---

## Пайплайн с нуля

Нужны `DADS/` (скачивается с HuggingFace, см. README) и
`DroneAudioDataset/` (старый Kaggle-набор, источник трудных негативов).

```bash
pip install --break-system-packages soundfile datasets huggingface_hub

hf download geronimobasso/drone-audio-detection-samples \
    --repo-type dataset --local-dir DADS

python3 -u prep_dads.py 2>&1 | tee logs/prep_dads.log     # ~15 мин
python3 -u prep_hard.py 2>&1 | tee logs/prep_hard.log
python3 -u train.py     2>&1 | tee logs/train.log         # эталон, DroneNet, models/dronenet.pt
python3 -u train2.py 12                                    # DroneNetV2, models/dronenet_v2.pt
python3 eval.py
```

`train2.py` принимает число эпох первым аргументом (по умолчанию 12) и сам
подбирает батч под архитектуру — DroneNetV2 требовательнее к памяти
(inverted bottleneck на полном разрешении), при OOM на другой GPU передать
свой `bs` через `train.main(..., bs=...)` напрямую.

---

## Обучение в Colab

Локально обучать нечем — данных на диске нет, GPU нет. Прогон идёт в Colab,
кэш и результаты живут в приватном репозитории HF `Onikore/airadar-hub`.

Один раз добавить в Colab секрет `HF_TOKEN` с правом записи: значок ключа
слева, включить доступ для ноутбука.

```
notebooks/01_prep.ipynb    сборка кэша из четырёх источников HF, ~30-50 мин, один раз
notebooks/02_train.ipynb   обучение, кэш тянется за минуты
```

Ноутбуки клонируют https://github.com/Onikore/airadar.git, поэтому правки
надо запушить до запуска.

**Обрыв сессии.** Просто запустить ноутбук заново.

- `01_prep` продолжит с недосчитанного шарда (`manifest.json` на HF).
- `02_train` при `resume=True` подхватит `last.pt` и продолжит с прерванной
  эпохи, включая состояние оптимизатора, расписания и генераторов случайных
  чисел — метрики продолжатся побитово, а не с новой траектории.

**Число эпох в начатом прогоне менять нельзя.** `OneCycleLR` задаёт кривую
скорости обучения на фиксированном горизонте шагов, и возобновление с другим
числом эпох либо рассыпает расписание, либо ведёт LR не по той кривой. Чтобы
учить дольше — новое имя `run`.

Читать лог прогона снаружи, без Colab:

```bash
HF_TOKEN=<токен> python -c "
import hub, json
p = hub.pull('runs/dronenet_hf/metrics.jsonl', 'logs/remote.jsonl')
for l in open(p, encoding='utf-8'):
    r = json.loads(l)
    print(r['ep'], r.get('auc_hard'), r.get('rec_field'))
"
```

Полевые записи извлекаются из видео отдельным шагом (нужен только при
добавлении новых записей):

```bash
pip install imageio-ffmpeg
python prep_field.py          # *.MOV / *.mp4 в корне -> field/drone_video*.wav
```

---

## Живой детектор

```bash
python3 detect.py                       # микрофон по умолчанию, консоль
python3 detect.py --device plughw:1,6   # конкретный вход, см. `arecord -l`
python3 detect.py --file field/drone_video.wav   # прогон готовой записи

python3 web.py                          # http://127.0.0.1:8000, живое обнаружение
python3 web.py --no-audio               # только прогресс обучения, без микрофона
```

Калибровочные ручки (`--threshold`, `--k`, `--m`) — отправная точка, на
реальной точке подбираются по месту.

---

## Диагностика (не требует GPU, можно параллельно с обучением)

```bash
python3 diag_leak.py       # дубликаты, ближайший сосед, тривиальный baseline
python3 diag_hard.py       # пересечение cache_hard с обучающей выборкой
python3 diag_compare.py    # CNN против baseline при одинаковом recall
python3 spectrum.py        # спектрограммы field/*.wav, поиск гребёнки
```

Каждый скрипт проекта — `--selfcheck`:

```bash
for f in train.py train2.py eval.py detect.py web.py spectrum.py \
         prep_dads.py prep_hard.py prep_field.py \
         hub.py hf_sources.py prep_hf.py; do
    python3 "$f" --selfcheck
done
```

Все проходят без данных и без GPU: нужны `numpy`, `scipy`, `soundfile`,
`sklearn`, `pyarrow`, `huggingface_hub` и CPU-сборка torch (см. README,
раздел «Машина без GPU»).
