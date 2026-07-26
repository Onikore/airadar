# %% [markdown]
# # Сборка кэша окон из четырёх источников HF
#
# Запускается **один раз**, примерно 30–50 минут. Дальше каждая сессия
# обучения тянет готовый кэш за минуты.
#
# **Перед запуском:** значок ключа слева → добавить секрет `HF_TOKEN` с правом
# записи → включить доступ для этого ноутбука.
#
# Обрыв сессии не страшен: прогресс пишется в `manifest.json` на HF, каждый
# посчитанный шард сразу уходит туда же. Повторный запуск продолжит с места
# обрыва, а не начнёт заново.

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

# %% [markdown]
# ## Место на диске
#
# Нужно ~6,7 ГБ под кэш плюс ~18 ГБ под шарды источников в кэше
# `huggingface_hub`. На бесплатном Colab доступно порядка 110 ГБ — с запасом.

# %%
!df -h /content | tail -1
!free -g | head -2

# %% [markdown]
# ## Проверка токена
#
# Делается до всего остального. Первая выгрузка случается только в конце
# первого шарда, и узнавать про нехватку прав через сорок минут работы обидно.
#
# Токену нужен scope **`repo.write`** на владельца `Onikore`. Fine-grained
# токен, выписанный на другой namespace или без прав, отдаёт 403 — тогда
# сообщение ниже скажет, что именно не так.

# %%
import hub

print("репозиторий:", hub.REPO)
print("токен принадлежит:", hub.check_access())
print("права на запись есть")

# %% [markdown]
# ## Схемы источников
#
# Печатаются **до** обработки. Разбор parquet написан под конкретные схемы,
# разведанные 2026-07-26; если датасет на HF перезалили, число шардов
# изменится, и лучше остановиться здесь, чем молотить 18 ГБ впустую.

# %%
import hf_sources as S
import prep_hf

for src in prep_hf.ORDER:
    sh = S.shards(src)
    mark = "ok" if len(sh) == S.N_SHARDS[src] else f"РАСХОЖДЕНИЕ, ожидалось {S.N_SHARDS[src]}"
    print(f"{S.NAMES[src]:16s} шардов {len(sh):3d}  {mark}")
    assert len(sh) == S.N_SHARDS[src], f"{S.NAMES[src]}: схема могла измениться"

# %% [markdown]
# ## Сборка
#
# Источники идут от мелких к крупным: ошибка разбора вылезет за минуту на
# ESC-50, а не через час на DADS.
#
# Порядок работы: ESC-50 (2 шарда) → UrbanSound8K (16) → DroneAudioSet (28) →
# DADS (39). Итого 85 шардов.
#
# **Коммиты на HF.** Лимит — 128 в час на репозиторий, и это главная мина
# этого шага. По умолчанию промежуточные части **не выгружаются**: весь прогон
# стоит 2 коммита, только на готовый кэш. Перезапуск внутри сессии всё равно
# продолжится с места обрыва — по локальному манифесту в `parts/`.
#
# `upload_parts=True` добавляет выгрузку частей раз в 15 шардов (ещё ~17
# коммитов). Нужно только если сессия уже рвалась и пересчитывать полчаса
# не хочется.
#
# Если лимит выбит — отказ выгрузки больше не роняет прогон: части и кэш
# остаются на диске, а в конце печатается команда для повторной выгрузки.

# %%
prep_hf.main(upload=True, upload_parts=False)

# %% [markdown]
# ## Полевые записи
#
# Единственная ненасыщенная метрика проекта. Уже лежат на HF (`field/`),
# извлечённые `prep_field.py` из исходных видео: f0 60,5 и 78,0 Гц,
# гребёнка выражена в 100% окон.
#
# Ячейка ниже нужна только если вы добавили **новые** записи: положите видео
# в `/content/airadar/`, и она извлечёт из них звук и выгрузит на HF.

# %%
import glob
import prep_field

if glob.glob("*.MOV") or glob.glob("*.mp4"):
    prep_field.main()
    hub.push("field", "field")
    print("выгружено:", sorted(os.listdir("field")))
else:
    hub.pull("field", ".")
    print("новых видео нет, полевые записи взяты с HF:", sorted(os.listdir("field")))

# %% [markdown]
# ## Что получилось
#
# Сверка с ожиданием. Расхождение вдвое означает ошибку в `CAP` или в разборе
# схемы — это повод остановиться, а не подкрутить константу.
#
# | | ожидается |
# |---|---|
# | `cache_dads` | 355–375 тыс. окон, дрон ~186 тыс. |
# | `cache_hard` | 44–53 тыс. окон |

# %%
import numpy as np

for cache in ("cache_dads", "cache_hard"):
    m = np.load(f"{cache}/meta.npz", allow_pickle=True)
    sp, y, g = m["split"], m["y"], m["group"]
    print(f"\n{cache}: окон {int(m['n'])}  групп {len(np.unique(g))}  "
          f"({int(m['n'])*8000*2/1e9:.2f} ГБ)")
    for p, nm in enumerate(("train", "val", "test")):
        s = sp == p
        print(f"  {nm:5s} {int(s.sum()):>7}  дрон {int(y[s].sum()):>7}  "
              f"({100*s.mean():.1f}%)")
    # ни одна группа не должна пересекать границу сплита — это и есть защита
    # от надутой метрики, всё остальное строится на этом
    for gid in np.unique(g):
        assert len(np.unique(sp[g == gid])) == 1, f"группа {gid} разорвана"
    if "cat" in m.files:
        u, c = np.unique(m["cat"], return_counts=True)
        print("  категорий:", len(u), "| крупнейшие:",
              ", ".join(f"{k}={v}" for k, v in sorted(zip(u.tolist(), c.tolist()),
                                                      key=lambda t: -t[1])[:6]))

print("\nни одна группа не пересекает границу сплита")

# %% [markdown]
# Кэш на HF. Дальше — `02_train.ipynb`, он тянет его за минуты.
