# %% [markdown]
# # Обучение детектора
#
# Кэш тянется с HF за несколько минут — препроцессинг заново не нужен.
#
# **Перед запуском:** значок ключа слева → секрет `HF_TOKEN` → доступ для этого
# ноутбука. Кэш должен быть собран — см. `01_prep.ipynb`.
#
# **Сессия оборвалась** — просто запустите ноутбук заново. `resume=True`
# подхватит `last.pt` с HF и продолжит с прерванной эпохи, включая состояние
# оптимизатора, расписания и генераторов случайных чисел.

# %%
!pip install -q pyarrow soundfile huggingface_hub

# %%
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

if not os.path.isdir("/content/airadar"):
    !git clone -q https://github.com/Onikore/airadar.git /content/airadar
%cd /content/airadar
!git pull -q && git log --oneline -1
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# %% [markdown]
# ## Кэш и полевые записи с HF

# %%
import hub

# Проверяем права до скачивания кэша: обучение всё равно упрётся в выгрузку
# после первой эпохи, а это уже потраченное GPU-время.
print("токен принадлежит:", hub.check_access())

for d in ("cache_dads", "cache_hard"):
    if os.path.exists(f"{d}/meta.npz"):
        print(f"{d}: уже на диске")
    else:
        hub.pull(f"cache/{d}", ".")
        print(f"{d}: скачан")

if os.path.isdir("field") and os.listdir("field"):
    print("field:", sorted(os.listdir("field")))
else:
    try:
        hub.pull("field", ".")
        print("field:", sorted(os.listdir("field")))
    except Exception as e:
        print(f"полевых записей на HF нет ({e}) — recall_поле считаться не будет")

!du -sh cache_dads cache_hard field 2>/dev/null

# %% [markdown]
# ## Состав кэша и сплита
#
# Глазами, до запуска обучения. Проверка «ни одна группа не пересекает границу
# сплита» — это и есть защита от надутой метрики: куски одной записи в train и
# val дали бы завышенный AUC, и всё остальное измерение потеряло бы смысл.

# %%
import numpy as np

for cache in ("cache_dads", "cache_hard"):
    m = np.load(f"{cache}/meta.npz", allow_pickle=True)
    sp, y, g = m["split"], m["y"], m["group"]
    print(f"\n{cache}: окон {int(m['n'])}  групп {len(np.unique(g))}")
    for p, nm in enumerate(("train", "val", "test")):
        s = sp == p
        print(f"  {nm:5s} {int(s.sum()):>7}  дрон {int(y[s].sum()):>7}  ({100*s.mean():.1f}%)")
    for gid in np.unique(g):
        assert len(np.unique(sp[g == gid])) == 1, f"группа {gid} разорвана"

print("\nни одна группа не пересекает границу сплита")

# %% [markdown]
# ## Прогон
#
# `run` задаёт каталог `runs/<run>/` на HF, куда после **каждой** эпохи уходят
# `train.log`, `metrics.jsonl` и `last.pt`.
#
# Число эпох менять на ходу нельзя: `OneCycleLR` задаёт кривую скорости
# обучения на фиксированном горизонте, и возобновление с другим числом эпох
# либо рассыпает расписание, либо ведёт LR не по той кривой. Чтобы учить
# дольше — новое имя `run`.

# %%
import train

RUN = "dronenet_hf"
EPOCHS = 12

# out_name задаёт файл лучшего чекпоинта, run — каталог прогона на HF.
# Держим их согласованными, иначе eval ниже будет искать не тот файл.
train.main(epochs=EPOCHS, bs=256, out_name=f"{RUN}.pt", run=RUN, resume=True)

# %% [markdown]
# ## Метрики прогона
#
# Те же строки лежат на HF в `runs/<run>/metrics.jsonl` — их можно читать
# снаружи, не открывая Colab.
#
# Какая метрика чего стоит:
#
# * `auc` на val DADS упирается в 0.99 за пару эпох и модели не различает.
# * `auc_hard` — дрон против механического шума и погоды, по ней отбирается
#   чекпоинт.
# * `recall_поле` — единственная ненасыщенная: модель либо слышит гармоники
#   тяжёлого дрона на реальной записи, либо нет.

# %%
import json

rows = [json.loads(l) for l in open(f"logs/{RUN}.jsonl", encoding="utf-8")]
print(f"{'эп':>3} {'loss':>7} {'auc':>7} {'auc_hard':>9} {'FAR@r90':>8}  recall_поле")
for r in rows:
    field = "  ".join(f"{k.replace('drone_video','v').replace('.wav','')}={v*100:.1f}%"
                      for k, v in r.get("rec_field", {}).items())
    print(f"{r['ep']:>3} {r['loss']:>7.4f} {r.get('auc', float('nan')):>7.4f} "
          f"{r.get('auc_hard', float('nan')):>9.4f} "
          f"{r.get('far_hard@r90', float('nan'))*100:>7.1f}%  {field}")

# %% [markdown]
# ## Оценка в операционных терминах
#
# Recall при заданном FAR и таблица ложных срабатываний по категориям шума.
# Считается на **val**; удержанная часть (`split == 2`) не трогается — она для
# финальной проверки, и вызывать `load_hard_test()` до неё не нужно.

# %%
import shutil
shutil.copyfile(f"models/{RUN}.pt", "models/dronenet.pt")   # eval.py читает это имя
!python eval.py

# %%
hub.push("eval.json", f"runs/{RUN}/eval.json")
print("eval.json выгружен на HF")
