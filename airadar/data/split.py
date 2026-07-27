"""Назначение сплита один раз, на уровне манифеста.

train.py ранее сам пересчитывал разбиение при каждом обучении — это и был
источник несравнимости прогонов. Здесь сплит — колонка манифеста, вычисляемая
один раз при сборке; дальше это WHERE split='train', а не логика.

Сама раскладка групп по сплитам не переписана: prep_hf.assign_split уже
решает эту задачу (группы разного размера, страта — свойство группы) и
проверена на реальных данных. Здесь только сборка страты и запись в колонку.
"""

import os
import sys
import numpy as np
import pyarrow as pa

TRAIN, VAL, TEST = 0, 1, 2

# prep_hf.py лежит в корне репозитория, не в пакете airadar — обычный
# `import prep_hf` не найдёт его при запуске `python -m airadar.data.split`
# из корня, потому что sys.path в этом режиме содержит только корень пакета
# верхнего уровня (airadar/), а не сам репозиторий. Поэтому корень добавляется
# явно, до импорта.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from prep_hf import assign_split as _assign_split  # noqa: E402


def _strata(table):
    """Составная страта — свойство группы: источник, метка, категория шума.

    Категория есть только у части строк (жёсткие негативы cache_hard), у
    остальных None; f-строка склеивает всё в один ключ без коллизий между
    разными src/label/category.
    """
    src = table.column("src").to_pylist()
    label = table.column("label").to_pylist()
    cat = table.column("category").to_pylist()
    return np.array([f"{s}_{l}_{c}" for s, l, c in zip(src, label, cat)], dtype=object)


def assign_split(group_id: np.ndarray, strata: np.ndarray,
                  frac: tuple = (0.75, 0.15, 0.10), seed: int = 0) -> np.ndarray:
    """Тонкая обёртка над prep_hf.assign_split — не переписывает алгоритм,
    только даёт этому модулю собственную точку входа с тем же контрактом."""
    return _assign_split(group_id, strata, frac=frac, seed=seed)


def apply_split(table: "pa.Table", frac: tuple = (0.75, 0.15, 0.10),
                 seed: int = 0) -> "pa.Table":
    """Строит страту из (src, label, category) манифеста и заполняет колонку
    split результатом assign_split. Один вызов на сборку, а не на обучение."""
    group_id = np.asarray(table.column("group_id").to_pylist())
    split = assign_split(group_id, _strata(table), frac=frac, seed=seed)
    return table.set_column(
        table.schema.get_field_index("split"), "split",
        pa.array(split, type=pa.int8()))


def selfcheck():
    t = pa.table({
        "group_id": [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        "src": [0] * 15, "label": [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "category": pa.array([None] * 15, type=pa.string()),
        "split": pa.array([None] * 15, type=pa.int8()),
    })
    out = apply_split(t)
    sp = out.column("split").to_pylist()
    assert all(s in (TRAIN, VAL, TEST) for s in sp), "все строки должны получить сплит"

    # группа физически не может оказаться в двух сплитах
    gid = out.column("group_id").to_pylist()
    for g in set(gid):
        vals = {sp[i] for i in range(len(gid)) if gid[i] == g}
        assert len(vals) == 1, f"группа {g} разорвана между сплитами: {vals}"

    # детерминированность по seed
    out2 = apply_split(t)
    assert out.column("split").to_pylist() == out2.column("split").to_pylist()

    print("split selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
