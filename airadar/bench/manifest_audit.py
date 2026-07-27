"""Санити-проверки манифеста как запросы, а не как ручной разбор постфактум.

Три прошлых бага (CAP перевёрнут, метка потеряна в ключе группы, страта по
окну вместо группы) обнаруживались только на полном прогоне обучения, часы
спустя. check_group_not_split здесь — прямая защита от второго из них: если
он не пуст, манифест собран неверно, и это видно за секунды, не за час.
"""

import os
import sys

# _strata переиспользуется, а не копируется: покрытие обязано проверяться по
# той же составной страте (src, label, category), по которой apply_split
# раскладывал группы. Проверка по «сырой» category молча пропускала весь
# позитивный класс — у позитивов category=None, и они не попадали в all_cats.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from airadar.data.split import _strata  # noqa: E402


def check_group_not_split(table):
    gid = table.column("group_id").to_pylist()
    sp = table.column("split").to_pylist()
    by_group = {}
    for g, s in zip(gid, sp):
        by_group.setdefault(g, set()).add(s)
    return sorted(g for g, vals in by_group.items() if len(vals) > 1)


def check_category_coverage(table, splits=(1, 2)):
    """Страты, отсутствующие целиком в запрошенных частях сплита.

    Ключ — составная страта (src, label, category) из split._strata, а не
    «сырая» category: раскладывал группы apply_split именно по ней, и
    проверять покрытие по чему-то другому значит проверять не то. По категории
    позитивный класс вообще не проверялся — у позитивов category=None, и
    «все позитивы DroneAudioSet уехали в train» проходило незамеченным.
    """
    sp = table.column("split").to_pylist()
    strat = [str(x) for x in _strata(table)]
    all_strata = set(strat)
    out = {}
    for part in splits:
        present = {c for c, s in zip(strat, sp) if s == part}
        missing = sorted(all_strata - present)
        if missing:
            out[part] = missing
    return out


def check_clip_id_unique(table):
    """clip_id, встретившиеся больше одного раза (в норме пусто).

    Повтор означает, что две строки претендуют на одну и ту же запись — так
    выглядит сборка, дописавшая клипы поверх уже учтённых (сброшенный
    чекпоинт при живых частях), и по самому манифесту это иначе не видно.
    """
    seen, dup = set(), set()
    for cid in table.column("clip_id").to_pylist():
        (dup if cid in seen else seen).add(cid)
    return sorted(dup)


def check_offset_tiling(table, clips_bin_bytes=None):
    """Клипы обязаны замостить clips.bin встык: ни дыр, ни наложений.

    Каждый байт clips.bin принадлежит ровно одному клипу — это инвариант
    записи (ClipWriter пишет подряд, без разделителей). Наложение значит, что
    две строки адресуют одно и то же аудио; дыра — что часть файла ничья;
    выход за размер файла — что строка указывает на аудио, которого уже нет.
    Именно так выглядит манифест, собранный поверх усечённого clips.bin, и без
    этой проверки он ничем не отличался от исправного.

    Сортировка по offset обязательна: _assemble склеивает части в порядке имён
    файлов, а не в порядке записи, поэтому таблица по offset не упорядочена.
    """
    off = table.column("offset").to_pylist()
    n = table.column("n_samples").to_pylist()
    problems, pos = [], 0
    for o, ln in sorted(zip(off, n)):
        if o > pos:
            problems.append(f"дыра в clips.bin: отсчёты [{pos}:{o}) ничьи")
        elif o < pos:
            problems.append(
                f"наложение: [{o}:{min(pos, o + ln)}) адресуется больше чем одним клипом")
        pos = max(pos, o + ln)
    if clips_bin_bytes is not None:
        if clips_bin_bytes % 4:
            problems.append(
                f"clips.bin повреждён: {clips_bin_bytes} байт не кратно 4 (float32)")
        have = clips_bin_bytes // 4
        if pos > have:
            problems.append(
                f"манифест адресует {pos} отсчётов, в clips.bin только {have} — "
                f"часть строк указывает на аудио, которого в файле нет")
        elif pos < have:
            problems.append(
                f"в clips.bin {have} отсчётов, манифест адресует {pos} — "
                f"{have - pos} отсчётов не принадлежат ни одной строке")
    return problems


def check_split_assigned(table):
    """Число строк без сплита. Манифест с пустым split не собран до конца:
    apply_split либо не отработал, либо отработал не по всем частям."""
    return sum(1 for s in table.column("split").to_pylist() if s is None)


def check_label_balance(table):
    sp = table.column("split").to_pylist()
    lab = table.column("label").to_pylist()
    out = {}
    for s, l in zip(sp, lab):
        out.setdefault(s, {}).setdefault(l, 0)
        out[s][l] += 1
    return out


def audit(table, clips_bin_bytes=None):
    """Сводка проверок. clips_bin_bytes — размер clips.bin в байтах; без него
    замощение проверяется только внутри манифеста, без сверки с файлом.

    "ok" реагирует только на структурные дефекты — разрыв группы между
    сплитами, повтор clip_id, разъехавшееся замощение clips.bin, пустой split.
    Дисбаланс меток и непокрытые страты в него не входят: на малых стратах они
    законны, и если бы роняли ok, аудит перестали бы читать.
    """
    broken = check_group_not_split(table)
    dup = check_clip_id_unique(table)
    tiling = check_offset_tiling(table, clips_bin_bytes)
    no_split = check_split_assigned(table)
    return {
        "n_rows": table.num_rows,
        "n_groups": len(set(table.column("group_id").to_pylist())),
        "groups_split_across_parts": broken,
        "duplicate_clip_ids": dup,
        "clips_bin_problems": tiling,
        "rows_without_split": no_split,
        "missing_strata": check_category_coverage(table),
        "label_balance": check_label_balance(table),
        "ok": not broken and not dup and not tiling and no_split == 0,
    }


def _fixture(pa, group_id, split, label, category, src=None,
             clip_id=None, offset=None, n_samples=None):
    """Минимальная таблица манифеста: только колонки, которые читают проверки.

    Клипы по умолчанию замощают файл встык по 10 отсчётов — именно так их
    кладёт ClipWriter, и отклонение от этого должно быть в фикстуре явным.
    """
    n = len(group_id)
    clip_id = list(range(n)) if clip_id is None else clip_id
    n_samples = [10] * n if n_samples is None else n_samples
    if offset is None:
        offset, pos = [], 0
        for ln in n_samples:
            offset.append(pos)
            pos += ln
    return pa.table({
        "clip_id": clip_id, "src": [0] * n if src is None else src,
        "offset": offset, "n_samples": n_samples,
        "group_id": group_id,
        "split": pa.array(split, type=pa.int8()), "label": label,
        "category": pa.array(category, type=pa.string()),
    })


def selfcheck():
    import pyarrow as pa

    good = _fixture(pa, group_id=[1, 1, 2, 2, 3], split=[0, 0, 1, 1, 2],
                    label=[1, 1, 0, 0, 1],
                    category=["wind", "wind", None, None, "rain"])
    assert check_group_not_split(good) == []

    bad = _fixture(pa, group_id=[1, 1, 2], split=[0, 1, 1],   # группа 1 разорвана
                   label=[1, 1, 0], category=[None, None, None])
    broken = check_group_not_split(bad)
    assert broken == [1], broken

    # Покрытие считается по составной страте (src, label, category), а не по
    # «сырой» категории: раскладывал группы apply_split именно по ней.
    cov = check_category_coverage(good, splits=(1, 2))
    assert cov[1] == ["0_1_rain", "0_1_wind"], cov   # в part=1 (val) только фон
    assert cov[2] == ["0_0_None", "0_1_wind"], cov   # в part=2 (test) только rain

    bal = check_label_balance(good)
    assert bal[0] == {1: 2}, bal
    assert bal[1] == {0: 2}, bal

    rep_good = audit(good, clips_bin_bytes=5 * 10 * 4)
    assert rep_good["ok"] is True, rep_good
    assert rep_good["n_rows"] == 5 and rep_good["n_groups"] == 3

    rep_bad = audit(bad)
    assert rep_bad["ok"] is False

    # Одна группа на всю таблицу — вырожденный случай для check_group_not_split:
    # множество split для единственной группы всегда синглтон, разрыва нет.
    single_group = _fixture(pa, group_id=[99, 99, 99], split=[0, 0, 0],
                            label=[1, 0, 1], category=["wind", None, "wind"])
    assert check_group_not_split(single_group) == []
    assert audit(single_group)["ok"] is True

    # Страта присутствует в каждой запрошенной части сплита — покрытие полное,
    # отчёт о пропусках должен быть пуст целиком, а не содержать части с
    # пустыми списками. Метка входит в страту, поэтому она здесь одинакова:
    # "fan у дрона" и "fan у фона" — разные страты, и это не придирка, а ровно
    # то, по чему считался сплит.
    full_coverage = _fixture(pa, group_id=[20, 21], split=[1, 2],
                             label=[0, 0], category=["fan", "fan"])
    assert check_category_coverage(full_coverage, splits=(1, 2)) == {}

    # Позитивы (category=None) обязаны проверяться на покрытие наравне с
    # категориями шума. По «сырой» категории они выпадали из проверки целиком,
    # и «все позитивы уехали в train» не отражалось в отчёте никак.
    pos_only_train = _fixture(pa, group_id=[30, 31, 32],
                              split=[0, 1, 2], label=[1, 0, 0],
                              category=[None, "rain", "rain"])
    cov_pos = check_category_coverage(pos_only_train, splits=(1, 2))
    assert cov_pos[1] == ["0_1_None"] and cov_pos[2] == ["0_1_None"], cov_pos

    # Самое важное свойство audit(): "ok" реагирует ТОЛЬКО на структурные
    # дефекты, а не на дисбаланс меток или пропуски страт — те законны на
    # малых стратах. Фикстура ниже нарочно даёт явный дисбаланс меток (2
    # против 1 внутри сплита 1) и пропуски страт, но группы не разорваны,
    # clip_id уникальны, замощение целое — ok обязан остаться True.
    benign = _fixture(pa, group_id=[10, 10, 11, 12], split=[1, 1, 1, 2],
                      label=[1, 1, 0, 0],
                      category=["wind", "wind", "wind", "rain"])
    assert check_group_not_split(benign) == []
    rep_benign = audit(benign)
    assert rep_benign["label_balance"][1] == {1: 2, 0: 1}, rep_benign   # дисбаланс есть
    assert rep_benign["missing_strata"], rep_benign                     # пропуски есть
    assert rep_benign["ok"] is True, rep_benign   # но структурного бага нет — ok не падает

    # --- C1: манифест, собранный поверх усечённого clips.bin. Части от
    # прошлого прогона выжили, чекпоинт потерян, писатель начал файл заново —
    # строки повторяют clip_id и адресуют одно и то же аудио дважды. До этих
    # проверок audit() на таком манифесте говорил ok: true.
    corrupt = _fixture(pa, group_id=[1, 2, 3, 4], split=[0, 0, 1, 2],
                       label=[1, 1, 0, 0], category=[None, None, "wind", "wind"],
                       clip_id=[0, 1, 0, 1],
                       offset=[0, 10, 0, 10], n_samples=[10, 10, 10, 10])
    assert check_clip_id_unique(corrupt) == [0, 1], check_clip_id_unique(corrupt)
    probs = check_offset_tiling(corrupt, clips_bin_bytes=20 * 4)
    assert any("наложение" in p for p in probs), probs
    rep_corrupt = audit(corrupt, clips_bin_bytes=20 * 4)
    assert rep_corrupt["ok"] is False, rep_corrupt

    # дыра в замощении: между клипами остались ничьи отсчёты
    gap = _fixture(pa, group_id=[1, 2], split=[0, 1], label=[1, 0],
                   category=[None, "wind"], offset=[0, 50], n_samples=[10, 10])
    assert any("дыра" in p for p in check_offset_tiling(gap)), check_offset_tiling(gap)

    # манифест адресует больше, чем есть в файле — ровно усечённый clips.bin
    over = _fixture(pa, group_id=[1, 2], split=[0, 1], label=[1, 0],
                    category=[None, "wind"])
    probs_over = check_offset_tiling(over, clips_bin_bytes=4 * 4)
    assert any("которого в файле нет" in p for p in probs_over), probs_over
    assert audit(over, clips_bin_bytes=4 * 4)["ok"] is False
    # и наоборот: в файле есть неадресованный хвост
    probs_tail = check_offset_tiling(over, clips_bin_bytes=100 * 4)
    assert any("не принадлежат ни одной строке" in p for p in probs_tail), probs_tail
    # размер, не кратный 4, — сам по себе дефект
    assert any("не кратно 4" in p for p in check_offset_tiling(over, clips_bin_bytes=81))

    # клип нулевой длины замощение не рвёт: он не занимает ни одного отсчёта
    empty_clip = _fixture(pa, group_id=[1, 2, 3], split=[0, 1, 2],
                          label=[1, 0, 0], category=[None, "wind", "wind"],
                          offset=[0, 10, 10], n_samples=[10, 0, 5])
    assert check_offset_tiling(empty_clip, clips_bin_bytes=15 * 4) == []

    # --- манифест без назначенного сплита не может считаться собранным
    no_split = _fixture(pa, group_id=[1, 2], split=[None, None], label=[1, 0],
                        category=[None, "wind"])
    assert check_split_assigned(no_split) == 2
    assert audit(no_split, clips_bin_bytes=20 * 4)["ok"] is False

    print("manifest_audit selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
