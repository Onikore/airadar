"""Прогон всех selfcheck пакета одной командой.

Заведено потому, что модулей стало много, и проверка "всё ли ещё цело"
не должна требовать помнить список.

Отдельно проверяется САМА полнота прогона. Раньше модуль без `selfcheck`
молча пропускался, и удалённый или переименованный `selfcheck` не отличался
от модуля, у которого его и не должно быть: скрипт печатал зелёное и выходил
с нулём. Теперь список освобождённых задан явно (конвенция плана: своей
логики нет у `report.py` и у всего в `cli/`), а любой другой найденный модуль
обязан иметь `selfcheck` и обязан его отработать.
"""

import os
import sys
import pkgutil
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import airadar

# Освобождены модули без собственной логики: пустой selfcheck был бы тестом,
# который ничего не утверждает. report.py только склеивает чужие результаты;
# cli/ этим обходом вообще не проходится (walk_packages идёт по airadar).
EXEMPT = {"airadar.bench.report"}

# Нижняя граница на случай, если обход внезапно перестанет находить пакет
# (сломанный __init__, съехавший sys.path): нулевой прогон не должен
# выглядеть успешным. На 2026-07-28 модулей с логикой тридцать три — девять
# в bench/, пять в data/, два в diag/, четыре в features/, четыре в
# models/, пять в augment/, четыре в train/, один airadar/config.py.
#
# Граница держится вплотную к факту намеренно. С запасом в пять любой из этих
# пакетов мог целиком выпасть из обхода, а прогон остался бы зелёным — то есть
# сторож молчал бы ровно в том случае, ради которого заведён. Цена — правка
# этой строки при добавлении модуля; она дешевле незамеченной пропажи пакета.
MIN_CHECKS = 33

fail = 0
found, ran = [], []
for mod in pkgutil.walk_packages(airadar.__path__, "airadar."):
    if mod.ispkg:                      # пакеты — это __init__.py, логики в них нет
        continue
    if mod.name in EXEMPT:
        continue
    found.append(mod.name)
    m = importlib.import_module(mod.name)
    fn = getattr(m, "selfcheck", None)
    if fn is None:
        print(f"ПРОВАЛ {mod.name}: нет selfcheck, а модуль не в списке освобождённых")
        fail += 1
        continue
    try:
        fn()
    except Exception as e:
        print(f"ПРОВАЛ {mod.name}: {e}")
        fail += 1
    else:
        ran.append(mod.name)

if len(ran) != len(found) - fail:
    print(f"ПРОВАЛ: отработало {len(ran)} из {len(found)} найденных модулей")
    fail += 1
if len(found) < MIN_CHECKS:
    print(f"ПРОВАЛ: найдено {len(found)} модулей с логикой, ожидалось не меньше "
          f"{MIN_CHECKS} — обход пакета что-то не видит")
    fail += 1

print(f"прогнано {len(ran)}/{len(found)} модулей, освобождено {len(EXEMPT)}")
sys.exit(1 if fail else 0)
