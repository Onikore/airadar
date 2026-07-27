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

    print("manifest_audit selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
