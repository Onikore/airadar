"""Сборка кэша окон из четырёх источников HF с замороженным сплитом.

Отличие от prep_dads.py: сплит считается здесь и кладётся в meta.npz, а не
пересчитывается при каждом обучении. При четырёх источниках и кэше, живущем
на HF, сплит обязан быть одинаковым между сессиями — иначе два прогона
несравнимы, а именно этим проект и болел (см. NEXT_STEPS.md, шаг 0).

Третья часть (test) не трогается до финальной проверки. Сейчас в проекте
удержанной части нет вообще: auc_hard одновременно отбирает чекпоинт и
служит отчётным числом, то есть числа для отчёта в проекте нет.

Устойчивость к обрыву сессии Colab: каждый шард пишется отдельным парным
файлом в parts/ и выгружается на HF сразу. Манифест на HF перечисляет готовые
шарды. Повторный запуск считает только недостающее, а сборка кэша целиком
происходит в конце из частей.
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
PARTS = os.path.join(ROOT, "parts")
CACHES = ("cache_dads", "cache_hard")

# Версия формата частей. Поднимать при любом изменении, меняющем содержимое:
# правка CAP, правка ключей групп, правка нарезки окон. Части старой версии
# отбраковываются и пересчитываются.
#
# Заведена после того, как исправление CAP и ключей групп сделало уже
# посчитанные части несовместимыми с новыми: смешанные, они дали бы кэш с
# перекошенным балансом классов и разорванными группами, причём молча.
PREP_VERSION = 3

# Окон с одной записи.
#
# У DADS кэп больше у ФОНА, а не у дрона, и это не опечатка. Файлов дрона
# 163 591, но они по 0.6 с — с каждого выходит одно окно. Файлов фона всего
# 16 729, зато они по 7.3 с. Взяв с фона 16 окон, получаем 177 тысяч против
# 163 тысяч, то есть примерно равный вклад классов. Наоборот (16 дрону, 1
# фону) набор становится на 92% дроном — проверено, так и было.
#
# Для DroneAudioSet кэп при нынешних данных не срабатывает ни на чём (самая
# длинная запись 152 с даёт 304 окна) — это страховка на случай перезаливки.
CAP = {
    (SRC_DADS, 0): 16, (SRC_DADS, 1): 1,
    (SRC_DAS, 1): 700, (SRC_DAS, 0): 700,
    (SRC_URBAN, 0): 4, (SRC_ESC, 0): 5,
}

FRAC = (0.75, 0.15, 0.10)

# Стационарный широкополосный шум механизмов и погода — то, что стоит рядом с
# дроном в спектре. Список тот же, что в prep_hard.py.
#
# "drone_rig_silence" сюда НЕ входит и в hf_sources.py такая метка больше не
# производится: f0_survey показал, что записи *-silence.wav статистически
# неотличимы от настоящих полётов того же рига (медиана f0 206 против 207 Гц,
# salience 20.8 дБ у обоих) — это моторы/ESC на холостых без пропеллеров, а не
# тишина. Заведённые как негатив, они дали 99.86% ложных срабатываний в eval:
# не ошибка модели, а неверная метка.
HARD = {
    "chainsaw", "helicopter", "airplane", "engine", "train", "vacuum_cleaner",
    "washing_machine", "hand_saw", "wind", "rain", "thunderstorm", "crackling_fire",
    "air_conditioner", "drilling", "engine_idling", "jackhammer",
}


def windows(x, cap):
    if len(x) < WIN:
        return [np.tile(x, int(np.ceil(WIN / len(x))))[:WIN]]
    n = min(len(x) // WIN, cap)
    return [x[i * WIN:(i + 1) * WIN] for i in range(n)]


def assign_split(groups, strata, frac=FRAC, seed=0):
    """0 = train, 1 = val, 2 = test. Делим группы, не окна.

    Раскладываем группы внутри каждой страты отдельно, а не по всему набору
    сразу: иначе маленькая страта целиком уедет в одну часть и val перестанет
    её измерять. ESC-50 — это 2000 клипов против 180 тысяч у DADS, случайное
    деление всего набора легко оставило бы его без val. Для cache_dads страта
    это (источник, класс), для cache_hard — категория шума, потому что таблица
    ложных срабатываний строится по категориям.

    Групп в страте может быть мало (тишина стенда даёт всего две — по одной на
    дрон). Тогда test остаётся без неё; что именно недопредставлено, печатает
    coverage_report, чтобы это было видно, а не молча пропало.

    Группы сильно разного размера: записи DroneAudioSet длятся от 30 до 152 с,
    то есть дают от 60 до 304 окон. Раскладывать их случайно нельзя — на
    страте из 12 групп val мог бы получить втрое больше положенной доли.
    Поэтому группы идут от крупной к мелкой, и каждая попадает в ту часть, у
    которой сейчас наибольший недобор по окнам. Seed влияет только на порядок
    групп одинакового размера.
    """
    groups, strata = np.asarray(groups), np.asarray(strata)
    g_ids, inv = np.unique(groups, return_inverse=True)

    # Окна каждой группы подряд — чтобы считать размер и страту группы за один
    # проход, а не искать по всему массиву на каждую из тысяч групп.
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(g_ids) + 1))
    sizes = np.diff(bounds)

    # Страта — свойство ГРУППЫ, а не окна, и это принципиально. Одна исходная
    # запись UrbanSound8K (fsID) содержит нарезки, у которых категории могут
    # различаться. Со стратой по окну такая группа попадала в две страты,
    # раскладывалась дважды и оказывалась разорвана между train и val — ровно
    # то, от чего группировка защищает. Берём преобладающую категорию группы.
    g_str = np.empty(len(g_ids), dtype=object)
    for j in range(len(g_ids)):
        seg = strata[order[bounds[j]:bounds[j + 1]]]
        vals, cnt = np.unique(seg, return_counts=True)
        g_str[j] = vals[cnt.argmax()]
    g_str = np.asarray(g_str.tolist())

    g_part = np.full(len(g_ids), -1, np.int8)
    rng = np.random.default_rng(seed)

    for st in np.unique(g_str):
        idx = np.flatnonzero(g_str == st)
        n = len(idx)
        if n == 1:
            n_va = n_te = 0
        elif n == 2:
            n_va, n_te = 1, 0            # хотя бы в val, иначе страту нечем мерить
        else:
            n_va = min(max(1, int(round(n * frac[1]))), n - 2)
            n_te = min(max(1, int(round(n * frac[2]))), n - 1 - n_va)

        # перемешиваем до сортировки: так группы равного размера получают
        # воспроизводимый, но не зависящий от исходного порядка приоритет
        idx = idx[rng.permutation(n)]
        cnt = sizes[idx]
        total = cnt.sum()
        want = [total * frac[0], total * frac[1], total * frac[2]]
        have = [0.0, 0.0, 0.0]
        slots = [n - n_va - n_te, n_va, n_te]
        for j in np.argsort(-cnt, kind="stable"):
            p = max((q for q in (0, 1, 2) if slots[q] > 0),
                    key=lambda q: want[q] - have[q])
            g_part[idx[j]] = p
            have[p] += cnt[j]
            slots[p] -= 1

    assert (g_part >= 0).all(), "остались нераспределённые группы"
    # Раздача по окнам через inv: группа физически не может оказаться в двух
    # частях, потому что решение принимается один раз на группу.
    return g_part[inv].astype(np.int8)


def coverage_report(split, strata, label=""):
    """Печатает страты, которых нет в val или test. Молчаливая потеря страты —
    это метрика, которая делает вид, что что-то измеряет."""
    strata = np.asarray(strata)
    missing = {1: [], 2: []}
    for st in np.unique(strata):
        for part in (1, 2):
            if not ((strata == st) & (split == part)).any():
                missing[part].append(str(st))
    for part, nm in ((1, "val"), (2, "test")):
        if missing[part]:
            print(f"  {label}нет в {nm} ({len(missing[part])}): "
                  f"{', '.join(sorted(missing[part])[:12])}"
                  f"{' …' if len(missing[part]) > 12 else ''}")
    return missing


# ------------------------------------------------------------------- части

def _key(src, i):
    return f"{src}_{i:04d}"


# Мелкие источники первыми: ошибка схемы вылезет за минуту, а не за час.
ORDER = (SRC_ESC, SRC_URBAN, SRC_DAS, SRC_DADS)


def listing():
    """Все шарды всех источников как (src, индекс, путь). Ходит в сеть."""
    return [(src, i, path) for src in ORDER
            for i, path in enumerate(S.shards(src))]


def _todo(manifest, all_shards):
    """Шарды, которых ещё нет в манифесте. Чистая функция, сеть не трогает."""
    done = set((manifest or {}).get("done", []))
    return [(src, i, p) for src, i, p in all_shards if _key(src, i) not in done]


def _dest(rec):
    """Категория есть — значит негатив для таблицы ложных срабатываний."""
    return "cache_hard" if rec.cat is not None else "cache_dads"


def _write_part(key, buf):
    """Один шард — свои .bin и .npz на каждый кэш. Части неизменяемы, поэтому
    выгружаются на HF один раз и переживают обрыв сессии."""
    written = []
    for cache, d in buf.items():
        if not d["y"]:
            continue
        pd = os.path.join(PARTS, cache)
        os.makedirs(pd, exist_ok=True)
        bin_p = os.path.join(pd, key + ".bin")
        with open(bin_p, "wb") as f:
            f.write(np.concatenate(d["w"]).tobytes())
        np.savez(os.path.join(pd, key + ".npz"),
                 y=np.array(d["y"], np.int8),
                 group=np.array(d["group"], np.int64),
                 src=np.array(d["src"], np.int8),
                 cat=np.array(d["cat"]),
                 ver=np.int64(PREP_VERSION))
        written.append((cache, key))
    return written


def _empty_buf():
    return {c: {"w": [], "y": [], "group": [], "src": [], "cat": []} for c in CACHES}


# Шардов между выгрузками на HF. HF ограничивает коммиты в репозиторий —
# 128 в час, — а выгрузка каждого шарда по отдельности давала бы 4-5 коммитов
# на шард, то есть около 300 на все 85. Лимит выбивало на 25-м шарде.
# Каталог целиком уходит одним коммитом (upload_folder), поэтому чекпоинт
# раз в 15 шардов стоит 3 коммита, а весь прогон — меньше двадцати.
#
# Цена укрупнения: при обрыве сессии теряются шарды, посчитанные после
# последнего чекпоинта. Манифест пишется в том же чекпоинте, поэтому они
# просто пересчитаются — рассинхронизации не будет.
CHECKPOINT_EVERY = 15

# Шардов, скачиваемых наперёд, пока считается текущий.
#
# Замер на пяти шардах DADS по 75 МБ, холодный кэш: последовательно 9.7 МБ/с,
# с глубиной 3 — 17.2 МБ/с, с 6 — 25.3 МБ/с, с 10 — те же 25 МБ/с. То есть
# шесть потоков насыщают канал, больше не нужно. Цена — до PREFETCH шардов на
# диске одновременно (у DADS отдельные файлы до 1.3 ГБ, пик около 5 ГБ).
PREFETCH = 6


LOCAL_MANIFEST = os.path.join(PARTS, "manifest.json")


def _read_manifest():
    """Манифест из HF и с диска, объединённые.

    Локальная копия нужна для двух случаев. Первый — upload=False: HF не
    трогаем совсем, но перезапуск внутри той же сессии должен видеть уже
    посчитанное. Второй — отказ выгрузки: части на диске есть, и считать их
    заново незачем.

    Объединяем, а не выбираем: на диске могут быть шарды, не успевшие уехать
    на HF, а на HF — шарды из прошлой сессии, чьи части уже скачаны.
    """
    done = set()
    sources = []
    try:
        sources.append(("HF", hub.read_json("manifest.json")))
    except Exception as e:
        print(f"манифест с HF не прочитан ({type(e).__name__}), беру локальный")
    if os.path.exists(LOCAL_MANIFEST):
        with open(LOCAL_MANIFEST, encoding="utf-8") as f:
            sources.append(("локально", json.load(f)))

    for where, m in sources:
        if not m:
            continue
        ver = int(m.get("ver", 1))
        if ver != PREP_VERSION:
            print(f"манифест ({where}) от формата v{ver}, сейчас v{PREP_VERSION} — "
                  f"{len(m.get('done', []))} шардов будут пересчитаны")
            continue
        done |= set(m.get("done", []))
    return {"done": sorted(done), "ver": PREP_VERSION}


def _checkpoint(manifest, upload_parts):
    """Сохраняет прогресс: локально всегда, на HF — если разрешено.

    Порядок важен: сначала части, потом манифест. Манифест — обещание, что
    перечисленные шарды лежат на HF; записанный раньше частей, он при обрыве
    между двумя выгрузками заставил бы пропустить непосчитанное.

    Отказ HF не роняет прогон. Части лежат на диске, манифест тоже, а 429 по
    лимиту коммитов лечится ожиданием — терять из-за него полчаса счёта нельзя.
    """
    os.makedirs(PARTS, exist_ok=True)
    manifest["ver"] = PREP_VERSION
    with open(LOCAL_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if not upload_parts:
        return True
    try:
        for cache in CACHES:
            pd = os.path.join(PARTS, cache)
            if os.path.isdir(pd) and os.listdir(pd):
                hub.push(pd, f"parts/{cache}")
        hub.write_json(manifest, "manifest.json")
        return True
    except Exception as e:
        print(f"      выгрузка на HF не удалась ({type(e).__name__}: "
              f"{str(e).splitlines()[0][:120]})")
        print(f"      части сохранены локально, счёт продолжается")
        return False


def collect(upload_parts=True, limit=None):
    """Считает недостающие шарды в parts/. Возвращает манифест."""
    manifest = _read_manifest()
    todo = _todo(manifest, listing())
    if limit:
        todo = todo[:limit]
    print(f"шардов к обработке: {len(todo)}  (готово ранее: {len(manifest['done'])})")
    print(f"выгрузка на HF: {'раз в %d шардов' % CHECKPOINT_EVERY if upload_parts else 'ОТКЛЮЧЕНА, только локально'}"
          f"  (лимит HF — 128 коммитов в час)")

    S.enable_fast_transfer()

    # Загрузка следующих шардов идёт параллельно счёту текущего. Замер: счёт
    # шарда DADS занимает ~4 с, загрузка 75 МБ — 5-8 с, то есть последовательно
    # больше половины времени уходит на ожидание сети. Параллельные загрузки
    # к тому же складываются по пропускной способности, если канал не насыщен
    # одним потоком. Глубина 3 хватает и не раздувает диск.
    from concurrent.futures import ThreadPoolExecutor

    pending = 0
    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        futures = {}

        def submit(k):
            if 0 <= k < len(todo):
                s, _, rel = todo[k]
                futures[k] = pool.submit(S.local_shard, s, rel)

        for k in range(PREFETCH):
            submit(k)

        for n, (src, i, rel) in enumerate(todo):
            local = futures.pop(n).result()
            submit(n + PREFETCH)

            key = _key(src, i)
            buf = _empty_buf()
            for rec in S.read_shard(src, local):
                d = buf[_dest(rec)]
                for w in windows(rec.audio, CAP[(rec.src, rec.label)]):
                    d["w"].append((w * 32767).astype(np.int16))
                    d["y"].append(rec.label)
                    d["group"].append(rec.group)
                    d["src"].append(rec.src)
                    d["cat"].append(rec.cat or "")
            _write_part(key, buf)
            manifest["done"].append(key)
            pending += 1
            print(f"  [{n+1}/{len(todo)}] {S.NAMES[src]} шард {i}  "
                  f"dads={len(buf['cache_dads']['y'])} "
                  f"hard={len(buf['cache_hard']['y'])}", flush=True)
            if pending >= CHECKPOINT_EVERY:
                _checkpoint(manifest, upload_parts)
                pending = 0
                print(f"      чекпоинт: {len(manifest['done'])} шардов готово",
                      flush=True)

    if pending:
        _checkpoint(manifest, upload_parts)
    return manifest


def _part_version(path):
    try:
        m = np.load(path, allow_pickle=True)
        return int(m["ver"]) if "ver" in m.files else 1
    except Exception:
        return None                  # битый файл — считаем отсутствующим


def _local_parts(cache):
    """Ключи частей нужной версии. Части чужой версии не возвращаются, значит
    будут пересчитаны, а не подмешаны к новым."""
    pd = os.path.join(PARTS, cache)
    if not os.path.isdir(pd):
        return set()
    out, stale = set(), 0
    for f in sorted(os.listdir(pd)):
        if not f.endswith(".npz"):
            continue
        if _part_version(os.path.join(pd, f)) == PREP_VERSION:
            out.add(f[:-4])
            continue
        # Часть чужой версии не пригодится никогда, а .bin занимает до 140 МБ.
        # Удаляем сразу, иначе на диске Colab копятся гигабайты мусора, который
        # ещё и путает при разборе.
        stale += 1
        for ext in (".npz", ".bin"):
            p = os.path.join(pd, f[:-4] + ext)
            if os.path.exists(p):
                os.remove(p)
    if stale:
        print(f"  {cache}: удалено частей старого формата: {stale} "
              f"(будут пересчитаны)")
    return out


def assemble(upload=True, manifest=None):
    """Склеивает части в кэш, считает сплит, кладёт meta.npz.

    Что есть на диске, определяет каталог parts/, а не манифест: иначе сборка
    зависела бы от сети там, где все части уже лежат рядом. Манифест нужен для
    другого — после обрыва сессии Colab он говорит, какие части были посчитаны
    в прошлый раз и лежат на HF, чтобы дотянуть их, а не считать заново.

    Порядок склейки — сортировка ключей, чтобы кэш побайтово воспроизводился
    при повторной сборке и индексы в meta соответствовали windows.bin.
    """
    if manifest is None:
        manifest = hub.read_json("manifest.json") or {"done": []}
    remote_files = None
    for cache in CACHES:
        pd = os.path.join(PARTS, cache)
        keys = _local_parts(cache)
        need = sorted(set(manifest.get("done", [])) - keys)
        if need and remote_files is None:
            remote_files = hub.list_files()      # один запрос на всю сборку
        for k in need:
            # Шард мог не дать частей в этот кэш (шарды DADS не дают cache_hard),
            # поэтому отсутствие на HF — норма, а не ошибка.
            if f"parts/{cache}/{k}.npz" in remote_files:
                os.makedirs(pd, exist_ok=True)
                for ext in (".bin", ".npz"):
                    hub.pull(f"parts/{cache}/{k}{ext}", os.path.join(pd, k + ext))
        if need:
            # Пересчитываем список через _local_parts, а не добавляем скачанное
            # вслепую: на HF остаются части прошлого формата, и подмешать их в
            # сборку значило бы получить кэш с перекошенным балансом и
            # разорванными группами. _local_parts заодно их и удалит.
            keys = _local_parts(cache)
        keys = sorted(keys)
        if not keys:
            print(f"{cache}: частей нет, пропускаю")
            continue

        out = os.path.join(ROOT, cache)
        os.makedirs(out, exist_ok=True)
        y, g, src, cat = [], [], [], []
        with open(os.path.join(out, "windows.bin"), "wb") as f:
            for k in keys:
                with open(os.path.join(pd, k + ".bin"), "rb") as pf:
                    f.write(pf.read())
                m = np.load(os.path.join(pd, k + ".npz"), allow_pickle=True)
                y.append(m["y"]); g.append(m["group"])
                src.append(m["src"]); cat.append(m["cat"])
        y = np.concatenate(y); g = np.concatenate(g)
        src = np.concatenate(src); cat = np.concatenate(cat)

        n_bytes = os.path.getsize(os.path.join(out, "windows.bin"))
        assert n_bytes == len(y) * WIN * 2, \
            f"{cache}: {n_bytes} байт против {len(y)} меток — части рассинхронизованы"

        strata = cat if cache == "cache_hard" else np.char.add(
            src.astype(str), np.char.add("_", y.astype(str)))
        split = assign_split(g, strata)

        # Несущая гарантия всего измерения: куски одной записи не расходятся по
        # разным частям. assign_split делает её невозможной по построению, но
        # проверяем здесь, а не только в ноутбуке — ноутбук легко оказывается
        # старой версии, а испорченный сплит виден только по надутой метрике.
        gi, inv = np.unique(g, return_inverse=True)
        lo = np.full(len(gi), 127, np.int8)
        hi = np.full(len(gi), -1, np.int8)
        np.minimum.at(lo, inv, split)
        np.maximum.at(hi, inv, split)
        torn = np.flatnonzero(lo != hi)
        assert len(torn) == 0, (
            f"{cache}: {len(torn)} групп разорвано между частями, например "
            f"{gi[torn[:5]].tolist()}. Ключ группы не различает записи, "
            f"которые должен различать.")

        meta = dict(y=y, group=g, src=src, split=split,
                    synth=np.zeros(len(y), bool),
                    n=np.int64(len(y)), win=np.int64(WIN))
        if cache == "cache_hard":
            meta["cat"] = cat
            meta["hard"] = np.array(sorted(HARD))
        np.savez(os.path.join(out, "meta.npz"), **meta)

        print(f"\n{cache}: окон {len(y)}  ({len(y)*WIN*2/1e9:.2f} ГБ)  "
              f"групп {len(np.unique(g))}  частей {len(keys)}")
        for part, nm in enumerate(("train", "val", "test")):
            sel = split == part
            print(f"  {nm:5s} {int(sel.sum()):>7}  дрон {int(y[sel].sum()):>7}")
        coverage_report(split, strata, label=f"{cache}: ")
        if upload:
            try:
                hub.push(out, f"cache/{cache}")
                print(f"  выгружен на HF: cache/{cache}")
            except Exception as e:
                print(f"  ВЫГРУЗКА НЕ УДАЛАСЬ ({type(e).__name__}: "
                      f"{str(e).splitlines()[0][:120]})")
                print(f"  Кэш собран и лежит в {cache}/ — счёт не потерян.")
                print(f"  Повторить позже одной строкой:")
                print(f"      python -c \"import hub; hub.push('{cache}', "
                      f"'cache/{cache}')\"")


def main(upload=True, upload_parts=False, limit=None):
    """upload — выгружать ли готовый кэш (2 коммита, почти всегда нужно).
    upload_parts — выгружать ли промежуточные части (17 коммитов).

    Части по умолчанию НЕ выгружаются. Они покупают одно: возможность
    продолжить после смерти сессии Colab, не пересчитывая заново. Стоит это 17
    коммитов из 128 в час, а сам прогон занимает около получаса, то есть
    сессия обычно доживает. Внутри одной сессии перезапуск и так продолжается
    с места обрыва — по локальному манифесту в parts/.

    Ставить upload_parts=True осмысленно, если сессия уже рвалась.
    """
    hub.ensure_repo()
    manifest = collect(upload_parts=upload_parts, limit=limit)
    assemble(upload=upload, manifest=manifest)


def selfcheck():
    # Часть проверок подменяет PARTS и LOCAL_MANIFEST на временный каталог,
    # поэтому объявление обязано стоять до первого присваивания.
    global PARTS, LOCAL_MANIFEST
    import tempfile

    # --- нарезка окон: поведение как в prep_dads.py
    assert len(windows(np.zeros(WIN), 4)) == 1
    assert len(windows(np.zeros(WIN * 10), 4)) == 4
    assert len(windows(np.zeros(WIN * 10), 1)) == 1
    (w,) = windows(np.arange(100, dtype=np.float32), 4)
    assert len(w) == WIN and w[100] == 0        # зациклено, а не дополнено нулями

    # --- сплит: группа целиком в одной части
    rng = np.random.default_rng(0)
    g = rng.integers(0, 300, 5000)
    strata = (g % 4).astype(str)
    sp = assign_split(g, strata)
    assert sp.dtype == np.int8 and len(sp) == len(g)
    for gid in np.unique(g):
        assert len(np.unique(sp[g == gid])) == 1, f"группа {gid} разорвана между сплитами"

    # --- пропорции близки к заданным
    for part, want in enumerate(FRAC):
        got = (sp == part).mean()
        assert abs(got - want) < 0.10, f"часть {part}: {got:.2f} против {want}"

    # --- каждая страта представлена во всех трёх частях, когда групп хватает
    for part in (0, 1, 2):
        assert set(np.unique(strata[sp == part])) == set(np.unique(strata)), \
            f"часть {part} потеряла страту"
    assert not any(coverage_report(sp, strata).values())

    # --- детерминированность
    assert (assign_split(g, strata) == sp).all()

    # --- пропорции держатся, когда группы РАЗНОГО размера.
    # Именно это ломалось при случайной раскладке: 12 групп по 60-304 окна
    # давали val втрое больше положенного.
    gg = np.repeat(np.arange(12), np.array([304, 280, 250, 210, 190, 170,
                                            150, 130, 110, 90, 75, 60]))
    sp2 = assign_split(gg, np.zeros(len(gg), int))
    for part, want in enumerate(FRAC):
        got = (sp2 == part).mean()
        assert abs(got - want) < 0.06, f"часть {part}: {got:.3f} против {want}"
    for gid in np.unique(gg):
        assert len(np.unique(sp2[gg == gid])) == 1

    # --- самая крупная группа не обязана оказаться в train, но train не пуст
    assert (sp2 == 0).sum() > (sp2 == 1).sum() > 0

    # --- вырожденные страты: одна группа только в train, две — train и val
    one = assign_split(np.zeros(10, int), np.zeros(10, int))
    assert set(one.tolist()) == {0}
    two = assign_split(np.repeat([1, 2], 5), np.zeros(10, int))
    assert set(two.tolist()) == {0, 1}, "две группы должны дать train и val"
    three = assign_split(np.repeat([1, 2, 3], 5), np.zeros(15, int))
    assert set(three.tolist()) == {0, 1, 2}, "три группы должны покрыть все части"

    # --- страта из двух групп попадает в отчёт как отсутствующая в test
    miss = coverage_report(two, np.zeros(10, int))
    assert miss[2] and not miss[1]

    # --- ГРУППА С НЕСКОЛЬКИМИ КАТЕГОРИЯМИ не должна разрываться.
    # Так устроен UrbanSound8K: одна запись Freesound (fsID) даёт нарезки, у
    # которых категории различаются. Со стратой по окну такая группа
    # раскладывалась дважды и уезжала в train и val одновременно.
    gm = np.repeat([10, 11, 12, 13, 14, 15], 6)
    cm = np.array(
        ["car_horn"] * 4 + ["street_music"] * 2 +      # группа 10: две категории
        ["street_music"] * 6 +
        ["car_horn"] * 6 +
        ["street_music"] * 3 + ["car_horn"] * 3 +      # группа 13: ровно пополам
        ["car_horn"] * 6 +
        ["street_music"] * 6)
    spm = assign_split(gm, cm)
    for gid in np.unique(gm):
        assert len(np.unique(spm[gm == gid])) == 1, \
            f"группа {gid} разорвана: страта должна считаться по группе, не по окну"
    assert len(np.unique(spm)) >= 2, "при шести группах должны быть разные части"

    # и то же самое на «настоящем» ключе UrbanSound8K, который поймал ошибку
    import hf_sources as _S
    g_urban = np.repeat([_S.urban_group(77751), _S.urban_group(77752)], 8)
    c_urban = np.array(["car_horn"] * 5 + ["street_music"] * 3 +
                       ["street_music"] * 8)
    sp_u = assign_split(g_urban, c_urban)
    for gid in np.unique(g_urban):
        assert len(np.unique(sp_u[g_urban == gid])) == 1, f"группа {gid} разорвана"

    # --- ключи частей стабильны и сортируемы (от них зависит порядок склейки)
    assert _key(SRC_ESC, 0) == "3_0000" and _key(SRC_DADS, 12) == "0_0012"
    assert sorted([_key(0, 10), _key(0, 9), _key(0, 100)]) == ["0_0009", "0_0010", "0_0100"]

    # --- версия формата: части и манифест чужой версии не принимаются.
    # Иначе исправление CAP или ключей групп молча смешало бы несовместимые
    # части, дав перекошенный баланс классов и разорванные группы.
    with tempfile.TemporaryDirectory() as d:
        real_parts, real_lm = PARTS, LOCAL_MANIFEST
        real_rj = hub.read_json
        PARTS = d
        LOCAL_MANIFEST = os.path.join(d, "manifest.json")
        hub.read_json = lambda r: None
        try:
            pd = os.path.join(d, "cache_dads")
            os.makedirs(pd)
            # своя версия — принимается
            np.savez(os.path.join(pd, "0_0000.npz"), y=np.zeros(1, np.int8),
                     group=np.zeros(1, np.int64), src=np.zeros(1, np.int8),
                     cat=np.array([""]), ver=np.int64(PREP_VERSION))
            open(os.path.join(pd, "0_0000.bin"), "wb").close()
            # чужая — отбраковывается и удаляется вместе с .bin
            np.savez(os.path.join(pd, "0_0001.npz"), y=np.zeros(1, np.int8),
                     group=np.zeros(1, np.int64), src=np.zeros(1, np.int8),
                     cat=np.array([""]), ver=np.int64(PREP_VERSION - 1))
            open(os.path.join(pd, "0_0001.bin"), "wb").close()
            # без поля ver — тоже чужая (формат v1)
            np.savez(os.path.join(pd, "0_0002.npz"), y=np.zeros(1, np.int8),
                     group=np.zeros(1, np.int64), src=np.zeros(1, np.int8),
                     cat=np.array([""]))
            open(os.path.join(pd, "0_0002.bin"), "wb").close()

            got = _local_parts("cache_dads")
            assert got == {"0_0000"}, got
            assert not os.path.exists(os.path.join(pd, "0_0001.bin")), \
                "устаревший .bin должен удаляться, он весит до 140 МБ"
            assert not os.path.exists(os.path.join(pd, "0_0002.npz"))

            # манифест старой версии не даёт пропустить шарды
            with open(LOCAL_MANIFEST, "w", encoding="utf-8") as f:
                json.dump({"done": ["0_0000", "0_0001"], "ver": PREP_VERSION - 1}, f)
            assert _read_manifest()["done"] == [], "манифест чужой версии принят"
            with open(LOCAL_MANIFEST, "w", encoding="utf-8") as f:
                json.dump({"done": ["0_0000"], "ver": PREP_VERSION}, f)
            assert _read_manifest()["done"] == ["0_0000"]
        finally:
            PARTS, LOCAL_MANIFEST = real_parts, real_lm
            hub.read_json = real_rj

    # --- манифест: докачка считает только недостающее
    shards_ = [(SRC_ESC, 0, "a.parquet"), (SRC_ESC, 1, "b.parquet"),
               (SRC_DADS, 0, "c.parquet")]
    assert _todo(None, shards_) == shards_, "пустой манифест — считаем всё"
    assert _todo({"done": []}, shards_) == shards_
    rest = _todo({"done": [_key(SRC_ESC, 0)]}, shards_)
    assert rest == shards_[1:], rest
    assert _todo({"done": [_key(s, i) for s, i, _ in shards_]}, shards_) == [], \
        "всё готово — считать нечего"

    # --- upload_parts=False не должен делать НИ ОДНОГО обращения к HF.
    # Ровно на этом прогон падал с 429: манифест писался всегда.
    real_parts, real_lm = PARTS, LOCAL_MANIFEST
    real_push, real_wj = hub.push, hub.write_json
    calls = []
    hub.push = lambda l, r: calls.append(("push", r))
    hub.write_json = lambda o, r: calls.append(("write_json", r))
    try:
        with tempfile.TemporaryDirectory() as d:
            PARTS = d
            LOCAL_MANIFEST = os.path.join(d, "manifest.json")
            assert _checkpoint({"done": ["3_0000"]}, upload_parts=False) is True
            assert calls == [], f"upload=False, а обращения к HF были: {calls}"
            assert os.path.exists(LOCAL_MANIFEST), "локальный манифест не записан"
            with open(LOCAL_MANIFEST, encoding="utf-8") as f:
                assert json.load(f)["done"] == ["3_0000"]

            # отказ HF не роняет прогон, а возвращает False
            def boom(*a, **k):
                raise RuntimeError("429 Too Many Requests")
            hub.write_json = boom
            assert _checkpoint({"done": ["3_0000"]}, upload_parts=True) is False

            # объединение манифестов: локальный виден даже когда HF недоступен
            hub.read_json = boom
            got = _read_manifest()
            assert got["done"] == ["3_0000"], got
    finally:
        PARTS, LOCAL_MANIFEST = real_parts, real_lm
        hub.push, hub.write_json = real_push, real_wj
        import importlib
        importlib.reload(hub)

    # --- бюджет коммитов HF: 128 в час. Чекпоинт стоит 3 коммита (два каталога
    # частей плюс манифест), финальная сборка — ещё 2. Считаем на всех 85 шардах.
    n_shards = sum(N for N in S.N_SHARDS.values())
    assert n_shards == 85, n_shards
    commits = (n_shards + CHECKPOINT_EVERY - 1) // CHECKPOINT_EVERY * 3 + 2
    assert commits <= 100, (
        f"{commits} коммитов при CHECKPOINT_EVERY={CHECKPOINT_EVERY} — "
        f"лимит HF 128 в час, нужен запас на повторные запуски")

    # --- HARD не содержит категорий, которых нет ни в одном источнике
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
    }
    assert HARD <= known, f"опечатка в HARD: {HARD - known}"

    # --- маршрутизация по кэшам. DroneAudioSet больше не даёт cat вообще —
    # hf_sources.py пропускает *-silence.wav целиком, а не размечает их
    # негативом (см. HARD выше про 99.86% ложных срабатываний на неверной метке).
    R = S.Rec
    assert _dest(R(None, 0, 1, None, SRC_DADS)) == "cache_dads"
    assert _dest(R(None, 0, 1, None, SRC_DAS)) == "cache_dads"
    assert _dest(R(None, 0, 0, "wind", SRC_ESC)) == "cache_hard"

    # --- баланс классов DADS на РЕАЛЬНЫХ числах датасета.
    # Проверяем замысел, а не константу: кэп существует ради примерно равного
    # вклада классов. Перевёрнутый (16 дрону, 1 фону) даёт 91% дрона — именно
    # это и случилось на первом полном прогоне.
    N_DRONE, SEC_DRONE = 163_591, 0.60      # средняя длина клипа дрона
    N_BG, SEC_BG = 16_729, 7.28             # и фона, из README
    def _n_win(sec, cap):
        return min(max(int(sec / 0.5), 1), cap)
    w_drone = N_DRONE * _n_win(SEC_DRONE, CAP[(SRC_DADS, 1)])
    w_bg = N_BG * _n_win(SEC_BG, CAP[(SRC_DADS, 0)])
    share = w_drone / (w_drone + w_bg)
    assert 0.35 < share < 0.65, (
        f"дрон занял бы {share*100:.0f}% окон DADS. Кэп фона должен быть больше "
        f"кэпа дрона: файлов фона в 10 раз меньше, но они в 12 раз длиннее")

    # --- у каждой пары (источник, класс) есть кэп
    for src in (SRC_DADS, SRC_DAS, SRC_URBAN, SRC_ESC):
        for lab in (0, 1):
            if src in (SRC_URBAN, SRC_ESC) and lab == 1:
                continue                        # эти источники дают только негативы
            assert (src, lab) in CAP, f"нет кэпа для ({src}, {lab})"

    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
