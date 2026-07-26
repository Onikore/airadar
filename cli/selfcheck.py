"""Прогон всех selfcheck пакета одной командой.

Заведено потому, что модулей стало много, и проверка "всё ли ещё цело"
не должна требовать помнить список.
"""

import os
import sys
import pkgutil
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import airadar

fail = 0
for mod in pkgutil.walk_packages(airadar.__path__, "airadar."):
    m = importlib.import_module(mod.name)
    fn = getattr(m, "selfcheck", None)
    if fn is None:
        continue
    try:
        fn()
    except Exception as e:
        print(f"ПРОВАЛ {mod.name}: {e}")
        fail += 1
sys.exit(1 if fail else 0)
