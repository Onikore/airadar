"""Полная сборка манифеста и хранилища клипов из четырёх источников HF.

    python cli/build_manifest.py --limit 1     # по одному шарду источника, проверка
    python cli/build_manifest.py               # полный прогон — часы, IO-связанный

Чекпоинт по шардам, как в prep_hf.py: строки манифеста каждого шарда сразу
пишутся в свой parquet-файл части (data/parts/<ключ>.parquet), а в JSON-
чекпоинте хранятся только счётчики — не сами строки. Иначе на полном прогоне
(десятки тысяч строк, 85 шардов) JSON чекпоинта пришлось бы целиком
перезаписывать после каждого шарда, и это стало бы тяжелее с каждым шардом.
Финальная сборка — конкатенация всех частей одним проходом в конце, а не
накопление в памяти по ходу.

Чекпоинт — это ещё и утверждение о состоянии диска, а не только «докуда
дошли»: он хранит версию схемы, число строк КАЖДОГО шарда (включая нулевое) и
суммарное число отсчётов в clips.bin. Перед открытием писателя _reconcile
сверяет это утверждение с фактом. Без сверки потерянный чекпоинт при живых
частях означал mode="wb" — clips.bin обрезался, части выживали, и их строки
адресовали чужое аудио; манифест при этом проходил аудит как исправный.
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pyarrow as pa
import pyarrow.parquet as pq
import hf_sources as S
from airadar.data.clips import ClipWriter
from airadar.data.build import ingest_shard, _key
from airadar.data.manifest import rows_to_table, MANIFEST_VERSION
from airadar.data.split import apply_split

OUT_DIR = os.path.join(ROOT, "data")
PARTS_DIR = os.path.join(OUT_DIR, "parts")
CLIPS_BIN = os.path.join(OUT_DIR, "clips.bin")
MANIFEST_PARQUET = os.path.join(OUT_DIR, "manifest.parquet")
CHECKPOINT_JSON = os.path.join(OUT_DIR, "build_checkpoint.json")
ORDER = (S.SRC_ESC, S.SRC_URBAN, S.SRC_DAS, S.SRC_DADS)   # мелкие источники первыми


def _empty_checkpoint():
    return {"ver": MANIFEST_VERSION, "shards": {}, "next_clip_id": 0,
            "n_samples_total": 0}


def _parts_on_disk():
    if not os.path.isdir(PARTS_DIR):
        return set()
    return {f[:-len(".parquet")] for f in os.listdir(PARTS_DIR)
            if f.endswith(".parquet")}


def _park_stale(old_ver):
    """Убирает сборку чужой версии схемы в сторону — переносом, не удалением.

    Части чужой версии не пригодятся никогда (колонки не совпадают), а
    оставленный рядом clips.bin ушёл бы в mode="ab" под новые смещения, то
    есть прямиком в C1. Убрать их с путей кода обязательно — но не обязательно
    стереть.

    Триггер здесь — событие КОДА (переключились на ревизию с другой
    MANIFEST_VERSION), а не дефект данных. Сюда же попадают откат на старую
    ревизию поверх завершённой сборки и любой чекпоинт до появления поля "ver".
    Стереть в этот момент весь собранный корпус — ошибка невосстановимая;
    перенос убирает файлы из _reconcile, _assemble и ClipWriter ровно так же
    полно, но остаётся исправимым. Тот же принцип, что и с пустым шардом: не
    выбрасывать молча, а зафиксировать и оставить решение человеку.
    """
    base = os.path.join(OUT_DIR, f"stale_v{old_ver}")
    dst, n = base, 1
    while os.path.exists(dst):          # прежнюю парковку не затираем
        n += 1
        dst = f"{base}_{n}"
    os.makedirs(dst)

    moved = []
    for k in sorted(_parts_on_disk()):
        name = k + ".parquet"
        os.replace(os.path.join(PARTS_DIR, name), os.path.join(dst, name))
        moved.append(f"parts/{name}")
    for p in (CLIPS_BIN, MANIFEST_PARQUET, CHECKPOINT_JSON):
        if os.path.exists(p):
            os.replace(p, os.path.join(dst, os.path.basename(p)))
            moved.append(os.path.basename(p))
    print(f"  чекпоинт версии {old_ver}, код версии {MANIFEST_VERSION}: "
          f"перенесено {len(moved)} файлов в {dst}")
    print(f"  ({', '.join(moved)}) — сборка начнётся заново, старое не удалено")


def _load_checkpoint():
    """Читает чекпоинт; при чужой версии схемы убирает сборку в сторону.

    Формат: {"ver", "shards": {ключ: строк}, "next_clip_id", "n_samples_total"}.

    shards — словарь, а не список готовых ключей: шард, не давший ни одной
    строки, обязан остаться отличимым от успешного. Раньше он попадал в done
    без части, без записи и без следа и на каждом следующем прогоне
    пропускался молча — навсегда.

    n_samples_total — обещание о размере clips.bin, по которому _reconcile
    отличает исправное состояние от потерянного чекпоинта.
    """
    if not os.path.exists(CHECKPOINT_JSON):
        return _empty_checkpoint()
    try:
        with open(CHECKPOINT_JSON, encoding="utf-8") as f:
            ck = json.load(f)
    except json.JSONDecodeError as e:
        # Атомарная запись (_save_checkpoint) обрывков не оставляет, но файл
        # мог остаться от прежней, неатомарной версии. Естественное «удалить и
        # перезапустить» ведёт прямо в порчу clips.bin — предупреждаем.
        sys.exit(f"чекпоинт {CHECKPOINT_JSON} не разбирается: {e}\n"
                 f"Просто удалить его нельзя: части и clips.bin останутся, и "
                 f"сборка начнёт писать поверх них.\nЛибо восстановить чекпоинт "
                 f"из копии, либо удалить всё сразу — {PARTS_DIR}, {CLIPS_BIN} и "
                 f"сам чекпоинт — и собрать заново.")
    ver = int(ck.get("ver", 0))
    if ver != MANIFEST_VERSION:
        _park_stale(ver)
        return _empty_checkpoint()
    return ck


def _save_checkpoint(ck):
    """Атомарная запись: временный файл плюс os.replace.

    Убитый посреди записи процесс обязан оставить либо старый чекпоинт целиком,
    либо новый целиком. Обрывок JSON лечится «удалить чекпоинт и перезапустить»
    — а это прямой путь в порчу clips.bin, от которой защищает _reconcile.
    """
    tmp = CHECKPOINT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ck, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CHECKPOINT_JSON)


def _reconcile(ck):
    """Сверяет чекпоинт с тем, что реально лежит на диске, до открытия писателя.

    Режим записи в clips.bin раньше выбирался по пустоте done — то есть по
    чекпоинту, а не по файлу. Потерянный или сброшенный чекпоинт при живых
    частях означал mode="wb": clips.bin обрезался в ноль, старые части
    оставались и попадали в итоговый манифест, и их строки начинали адресовать
    ЧУЖОЕ аудио. Молча: audit() на таком манифесте отвечал ok: true.

    Восстановимый случай ровно один: clips.bin ДЛИННЕЕ обещанного при точно
    совпавшем составе частей. Так выглядит прерванный шард (Ctrl-C, кончившееся
    место, OOM, падение декодера): байты доходят до файла по ходу записи, а
    n_samples_total растёт только по завершении целого шарда. Хвост за
    n_samples_total по построению не адресован ни одной готовой частью — та,
    что писалась, в чекпоинт так и не попала, — поэтому обрезка до обещанного
    размера ничего учтённого не теряет. Обрывать на этом многодневный прогон
    было бы дороже самой находки.

    Остальное автоматически чинить нечем: если файл КОРОЧЕ обещанного, данных
    не хватает, а не лишку, и состав обрезанного восстановить неоткуда.
    """
    shards = dict(ck.get("shards") or {})
    parts = _parts_on_disk()
    expected = {k for k, n in shards.items() if n > 0}
    have = os.path.getsize(CLIPS_BIN) if os.path.exists(CLIPS_BIN) else 0
    want = 4 * int(ck.get("n_samples_total", 0))      # float32 = 4 байта
    tail = have - want

    bad = []
    missing, extra = sorted(expected - parts), sorted(parts - expected)
    if missing:
        bad.append(f"чекпоинт числит готовыми части, которых нет на диске: {missing}")
    if extra:
        bad.append(f"на диске есть части, о которых чекпоинт не знает: {extra}")
    if tail < 0:
        bad.append(f"clips.bin короче обещанного: {have} байт против {want} "
                   f"({ck.get('n_samples_total', 0)} отсчётов x 4 байта) — "
                   f"записанного не хватает")
    if bad:
        if tail > 0:
            # Хвост сам по себе безобиден, но при разошедшемся составе частей
            # обрезать вслепую нельзя: неизвестно, чей он.
            bad.append(f"и вдобавок clips.bin длиннее обещанного на {tail} байт")
        sys.exit("\n".join([
            "чекпоинт и данные на диске разошлись, продолжать нельзя:",
            *("  - " + b for b in bad),
            "",
            "Так выглядит потерянный, сброшенный или недописанный чекпоинт при",
            "живых частях. Дописывать в clips.bin вслепую нельзя: часть строк",
            "начнёт адресовать чужое аудио, а манифест будет выглядеть исправным.",
            "Автоматической починки нет — что лежало в обрезанном файле, знать",
            "неоткуда. Либо удалить и собрать заново:",
            f"    {PARTS_DIR}",
            f"    {CLIPS_BIN}",
            f"    {CHECKPOINT_JSON}",
            "либо вернуть недостающее из резервной копии и запустить снова.",
        ]))
    if tail > 0:
        with open(CLIPS_BIN, "r+b") as f:
            f.truncate(want)
        print(f"clips.bin был длиннее чекпоинта на {tail} байт — обрезан до {want}. "
              f"Так выглядит шард, прерванный на середине: его байты не адресованы "
              f"ни одной готовой частью, и он будет прочитан заново.")
    return shards


def _ingest_all(limit=None, allow_empty=False):
    """Скачивает и пишет недостающие шарды. Возвращает next_clip_id."""
    os.makedirs(PARTS_DIR, exist_ok=True)
    ck = _load_checkpoint()
    shards = _reconcile(ck)
    next_id = int(ck.get("next_clip_id", 0))
    total = int(ck.get("n_samples_total", 0))

    empty_before = sorted(k for k, n in shards.items() if n == 0)
    if empty_before:
        print(f"ВНИМАНИЕ: {len(empty_before)} шардов прошлых прогонов не дали ни "
              f"одной строки: {empty_before}")

    # Режим — по фактическому состоянию файла, а не по пустоте чекпоинта;
    # после _reconcile эти два состояния заведомо согласованы.
    mode = "ab" if os.path.exists(CLIPS_BIN) else "wb"
    with ClipWriter(CLIPS_BIN, mode=mode) as writer:
        for src in ORDER:
            rels = S.shards(src)
            if limit is not None:
                rels = rels[:limit]
            for i, rel in enumerate(rels):
                key = _key(src, i)
                if key in shards:
                    continue
                print(f"[{S.NAMES[src]} {i+1}/{len(rels)}] {rel}")
                path = S.local_shard(src, rel)
                new_rows, next_id = ingest_shard(S.read_shard(src, path), writer,
                                                 next_id, key)
                n_new = sum(r["n_samples"] for r in new_rows)
                total += n_new
                print(f"    шард {key}: строк {len(new_rows)}, отсчётов {n_new}, "
                      f"всего в clips.bin {total}")
                if not new_rows:
                    # Шард без строк — не то же самое, что успешно обработанный.
                    # Смена кодека, битая закачка, перезалив датасета на HF
                    # выглядят ровно так, и молча пометить его готовым значит
                    # потерять его навсегда: спецификация (§5) требует, чтобы
                    # ничего не выбрасывалось кодом молча.
                    print(f"ВНИМАНИЕ: шард {key} ({rel}) не дал ни одной строки")
                    if not allow_empty:
                        sys.exit(
                            f"шард {key} ({rel}) не дал ни одной строки — прогон "
                            f"остановлен.\nЭто может быть законно (в шарде нет "
                            f"подходящих записей), а может быть битой закачкой или "
                            f"сменой формата источника.\nЕсли пустота ожидаема, "
                            f"запустите с --allow-empty-shards: тогда факт будет "
                            f"записан в чекпоинт, а не пропущен молча.")
                else:
                    pq.write_table(rows_to_table(new_rows),
                                   os.path.join(PARTS_DIR, f"{key}.parquet"))
                shards[key] = len(new_rows)
                # Порядок важен: сначала данные на диск, потом обещание о них.
                writer.flush()
                _save_checkpoint({"ver": MANIFEST_VERSION, "shards": shards,
                                  "next_clip_id": next_id,
                                  "n_samples_total": total})
    return next_id


def _assemble():
    """Склеивает все части в один манифест и назначает сплит один раз."""
    parts = sorted(os.path.join(PARTS_DIR, f) for f in os.listdir(PARTS_DIR)
                   if f.endswith(".parquet"))
    if not parts:
        sys.exit("нет ни одной части — сборка не выполнялась")
    tables = []
    for p in parts:
        t = pq.read_table(p)
        if t.num_rows == 0:
            sys.exit(f"часть {os.path.basename(p)} пуста — пустые части не пишутся, "
                     f"файл появился не от этой сборки")
        # Вторая, независимая от чекпоинта проверка версии: части могли быть
        # записаны разными запусками, и правка схемы посреди многодневного
        # прогона не должна давать склейку несовместимых записей.
        alien = sorted({v for v in t.column("prep_version").to_pylist()
                        if v != MANIFEST_VERSION})
        if alien:
            sys.exit(f"часть {os.path.basename(p)} собрана под версиями схемы "
                     f"{alien}, код — под {MANIFEST_VERSION}: склеивать нельзя")
        tables.append(t)
    # concat_tables — в pyarrow, а не в pyarrow.parquet (в отличие от read_table/write_table)
    table = pa.concat_tables(tables)
    table = apply_split(table)
    pq.write_table(table, MANIFEST_PARQUET)
    print(f"манифест: {table.num_rows} строк из {len(parts)} частей -> {MANIFEST_PARQUET}")
    print(f"клипы: -> {CLIPS_BIN}")


def main(limit=None, allow_empty=False):
    _ingest_all(limit=limit, allow_empty=allow_empty)
    _assemble()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="шардов на источник (для проверки, не для полного прогона)")
    ap.add_argument("--allow-empty-shards", action="store_true",
                    help="не останавливаться на шарде без строк, а записать "
                         "этот факт в чекпоинт и идти дальше")
    a = ap.parse_args()
    main(limit=a.limit, allow_empty=a.allow_empty_shards)
