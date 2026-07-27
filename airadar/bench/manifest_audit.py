"""Санити-проверки манифеста как запросы, а не как ручной разбор постфактум.

Три прошлых бага (CAP перевёрнут, метка потеряна в ключе группы, страта по
окну вместо группы) обнаруживались только на полном прогоне обучения, часы
спустя. check_group_not_split здесь — прямая защита от второго из них: если
он не пуст, манифест собран неверно, и это видно за секунды, не за час.
"""

import sys


def check_group_not_split(table):
    gid = table.column("group_id").to_pylist()
    sp = table.column("split").to_pylist()
    by_group = {}
    for g, s in zip(gid, sp):
        by_group.setdefault(g, set()).add(s)
    return sorted(g for g, vals in by_group.items() if len(vals) > 1)


def check_category_coverage(table, splits=(1, 2)):
    sp = table.column("split").to_pylist()
    cat = table.column("category").to_pylist()
    all_cats = {c for c in cat if c is not None}
    out = {}
    for part in splits:
        present = {c for c, s in zip(cat, sp) if s == part and c is not None}
        missing = sorted(all_cats - present)
        if missing:
            out[part] = missing
    return out


def check_label_balance(table):
    sp = table.column("split").to_pylist()
    lab = table.column("label").to_pylist()
    out = {}
    for s, l in zip(sp, lab):
        out.setdefault(s, {}).setdefault(l, 0)
        out[s][l] += 1
    return out


def audit(table):
    broken = check_group_not_split(table)
    return {
        "n_rows": table.num_rows,
        "n_groups": len(set(table.column("group_id").to_pylist())),
        "groups_split_across_parts": broken,
        "missing_categories": check_category_coverage(table),
        "label_balance": check_label_balance(table),
        "ok": len(broken) == 0,
    }


def selfcheck():
    import pyarrow as pa

    good = pa.table({
        "group_id": [1, 1, 2, 2, 3], "split": [0, 0, 1, 1, 2],
        "label": [1, 1, 0, 0, 1],
        "category": pa.array(["wind", "wind", None, None, "rain"], type=pa.string()),
    })
    assert check_group_not_split(good) == []

    bad = pa.table({
        "group_id": [1, 1, 2], "split": [0, 1, 1],       # группа 1 разорвана
        "label": [1, 1, 0],
        "category": pa.array([None, None, None], type=pa.string()),
    })
    broken = check_group_not_split(bad)
    assert broken == [1], broken

    cov = check_category_coverage(good, splits=(1, 2))
    assert 1 in cov and "rain" in cov[1], cov          # rain нет в part=1 (val)
    assert 2 in cov and "wind" in cov[2], cov           # wind нет в part=2 (test)

    bal = check_label_balance(good)
    assert bal[0] == {1: 2}, bal
    assert bal[1] == {0: 2}, bal

    rep_good = audit(good)
    assert rep_good["ok"] is True
    assert rep_good["n_rows"] == 5 and rep_good["n_groups"] == 3

    rep_bad = audit(bad)
    assert rep_bad["ok"] is False

    # Одна группа на всю таблицу — вырожденный случай для check_group_not_split:
    # множество split для единственной группы всегда синглтон, разрыва нет.
    single_group = pa.table({
        "group_id": [99, 99, 99], "split": [0, 0, 0],
        "label": [1, 0, 1],
        "category": pa.array(["wind", None, "wind"], type=pa.string()),
    })
    assert check_group_not_split(single_group) == []
    assert audit(single_group)["ok"] is True

    # Категория присутствует в каждой запрошенной части сплита — покрытие
    # полное, отчёт о пропусках должен быть пуст целиком, а не содержать
    # части с пустыми списками.
    full_coverage = pa.table({
        "group_id": [20, 21], "split": [1, 2],
        "label": [1, 0],
        "category": pa.array(["fan", "fan"], type=pa.string()),
    })
    assert check_category_coverage(full_coverage, splits=(1, 2)) == {}

    # Самое важное свойство audit(): "ok" реагирует ТОЛЬКО на разрыв группы
    # между сплитами, а не на дисбаланс меток или пропуски категорий — те
    # законны на малых стратах и не являются структурным багом манифеста.
    # Фикстура ниже нарочно даёт явный дисбаланс меток (2 против 1 внутри
    # сплита 1) и пропуски категорий (wind/rain не пересекаются между
    # сплитами), но группы не разорваны — ok обязан остаться True.
    benign = pa.table({
        "group_id": [10, 10, 11, 12], "split": [1, 1, 1, 2],
        "label": [1, 1, 0, 0],
        "category": pa.array(["wind", "wind", "wind", "rain"], type=pa.string()),
    })
    assert check_group_not_split(benign) == []
    rep_benign = audit(benign)
    assert rep_benign["label_balance"][1] == {1: 2, 0: 1}, rep_benign   # дисбаланс есть
    assert rep_benign["missing_categories"] == {1: ["rain"], 2: ["wind"]}, rep_benign  # пропуски есть
    assert rep_benign["ok"] is True, rep_benign   # но структурного бага нет — ok не падает

    print("manifest_audit selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
