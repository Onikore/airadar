"""Схема манифеста: один клип — одна строка.

Манифест — это единственный источник знаний о данных. Четыре бага одной
прошлой сессии (перевёрнутый CAP, потерянная метка в ключе группы, страта
по окну вместо группы, drone_rig_silence) были невидимы, потому что жили в
императивном коде и проявлялись только на полном прогоне. Здесь то же самое
— колонка таблицы, и неверное значение видно запросом `SELECT`, а не
всплывает через час обучения на насыщенной метрике.
"""

import sys

# v2 — добавлена колонка "shard" (происхождение строки). Правка схемы обязана
# поднимать версию, а не молча смешивать несовместимые записи: части от v1 не
# имеют этой колонки, склеить их с v2 нельзя (см. cli/build_manifest.py).
MANIFEST_VERSION = 2

# "shard" — ключ шарда-источника строки в формате build._key ("3_0000").
# Спецификация (§5.2) требует его с самого начала, и он ничего не стоит: ключ
# уже вычислен на приёме. Без него потерянный шард нельзя опознать постфактум
# («каких строк не хватает?»), а после перезалива датасета на HF — сопоставить
# строку с её исходным файлом.
SCHEMA = {
    "clip_id": "int64", "src": "int8", "shard": "string",
    "offset": "int64", "n_samples": "int64",
    "label": "int8", "label_conf": "float32", "group_id": "int64",
    "domain": "string", "category": "string", "synth": "bool",
    "f0_med": "float32", "salience": "float32", "lf_energy": "float32",
    "split": "int8", "prep_version": "int64",
}


def selfcheck():
    row = make_row(clip_id=1, src=0, shard="0_0007", offset=0, n_samples=9600,
                   label=1, label_conf=1.0, group_id=42, domain="rig1",
                   category=None)
    assert set(row) == set(SCHEMA), set(SCHEMA) - set(row)
    assert row["clip_id"] == 1 and row["n_samples"] == 9600
    assert row["shard"] == "0_0007"             # происхождение строки, §5.2
    assert row["synth"] is False               # значение по умолчанию
    assert row["split"] is None                 # заполняется позже (Task 4)
    assert row["prep_version"] == MANIFEST_VERSION

    validate_row(row)                            # не должно бросать

    bad = dict(row)
    bad["clip_id"] = "не число"
    try:
        validate_row(bad)
    except ValueError as e:
        assert "clip_id" in str(e)
    else:
        raise AssertionError("validate_row должен ловить неверный тип")

    missing = dict(row)
    del missing["group_id"]
    try:
        validate_row(missing)
    except ValueError as e:
        assert "group_id" in str(e)
    else:
        raise AssertionError("validate_row должен ловить пропущенное поле")

    # шард — обязательное поле происхождения: пустым он не проходит валидацию,
    # иначе потерянный шард нельзя было бы опознать по манифесту (см. C2)
    no_shard = dict(row)
    no_shard["shard"] = None
    try:
        validate_row(no_shard)
    except ValueError as e:
        assert "shard" in str(e)
    else:
        raise AssertionError("validate_row должен требовать shard")

    # категория, в отличие от шарда, законно пуста у позитивов
    validate_row(dict(row, category=None))

    t = rows_to_table([row, dict(row, clip_id=2)])
    assert t.num_rows == 2
    assert t.column("clip_id").to_pylist() == [1, 2]
    assert t.column("shard").to_pylist() == ["0_0007", "0_0007"]
    assert str(t.schema.field("clip_id").type) == "int64"
    assert str(t.schema.field("category").type) == "string"
    assert str(t.schema.field("shard").type) == "string"

    print("manifest selfcheck ok")


def make_row(clip_id, src, shard, offset, n_samples, label, label_conf,
            group_id, domain, category, synth=False):
    # shard без значения по умолчанию намеренно: забытый аргумент обязан быть
    # ошибкой вызова, а не тихим None в колонке происхождения.
    return {
        "clip_id": int(clip_id), "src": int(src), "shard": str(shard),
        "offset": int(offset),
        "n_samples": int(n_samples), "label": int(label),
        "label_conf": float(label_conf), "group_id": int(group_id),
        "domain": str(domain), "category": category, "synth": bool(synth),
        "f0_med": None, "salience": None, "lf_energy": None,
        "split": None, "prep_version": MANIFEST_VERSION,
    }


_PY_TYPES = {
    "int64": int, "int8": int, "float32": (float, type(None)),
    "bool": bool, "string": (str, type(None)),
}

# "split" — целочисленное поле схемы, но make_row оставляет его None: значение
# проставляет отдельный шаг (Task 4, стратифицированное разбиение). Поэтому
# оно, в отличие от остальных int-полей, не считается "пропущенным" при None.
_FILLED_LATER = {"split"}

# Строковые поля, для которых None — не «нет значения», а потеря происхождения:
# category законно пуста у позитивов, а домен и шард известны всегда.
_REQUIRED_STR = {"domain", "shard"}


def validate_row(row):
    for name, kind in SCHEMA.items():
        if name not in row:
            raise ValueError(f"в строке манифеста нет поля {name!r}")
        val = row[name]
        want = _PY_TYPES[kind]
        if val is None and kind in ("int64", "int8") and name not in _FILLED_LATER:
            raise ValueError(f"{name!r}: обязательное целочисленное поле пусто")
        if val is None and name in _REQUIRED_STR:
            raise ValueError(f"{name!r}: обязательное строковое поле пусто")
        if val is not None and not isinstance(val, want):
            raise ValueError(f"{name!r}: ожидался {want}, получено {type(val)}")


def rows_to_table(rows):
    import pyarrow as pa
    cols = {name: [r[name] for r in rows] for name in SCHEMA}
    types = {"int64": pa.int64(), "int8": pa.int8(), "float32": pa.float32(),
             "bool": pa.bool_(), "string": pa.string()}
    fields = [pa.field(name, types[kind]) for name, kind in SCHEMA.items()]
    arrays = [pa.array(cols[name], type=types[kind])
             for name, kind in SCHEMA.items()]
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
