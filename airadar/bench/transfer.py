"""Насколько соврёт порог, откалиброванный в лаборатории, приехав в лес.

Отношение факт/номинал прямо отвечает на вопрос, который иначе выясняется
только на месте. Метрика не насыщается: это мера сдвига распределений, а не
качества модели. Если дрейф больше зазора между дроном и негативом, никакой
фиксированный порог не работает в принципе — и тогда адаптивный порог не
улучшение, а необходимость.
"""

import sys
import numpy as np


def selfcheck():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 100000).astype(np.float32)

    # порог по FAR 1% на стандартной нормали ~ 2.326
    thr = threshold_at_far(a, 0.01)
    assert abs(thr - 2.326) < 0.05, thr

    # тот же корпус — перенос идеален
    r = transfer_error(a, a.copy(), 0.01)
    assert abs(r["ratio"] - 1.0) < 0.05, r

    # корпус B сдвинут на +1 сигму: фактический FAR обязан вырасти в разы
    b = a + 1.0
    r2 = transfer_error(a, b, 0.01)
    assert r2["far_actual"] > 0.05, r2
    assert r2["ratio"] > 5.0, r2

    # дрейф квантиля в единицах разброса A
    assert abs(drift(a, b, q=0.99) - 1.0) < 0.05, drift(a, b, q=0.99)
    assert abs(drift(a, a.copy(), q=0.99)) < 0.02

    print("transfer selfcheck ok")


def threshold_at_far(logits_neg, far):
    """Порог, дающий заданную долю ложных на корпусе негативов."""
    return float(np.quantile(np.asarray(logits_neg, np.float64), 1.0 - far))


def transfer_error(logits_a, logits_b, far):
    """Порог откалиброван на A, применён к B. Отношение факт/номинал."""
    thr = threshold_at_far(logits_a, far)
    actual = float((np.asarray(logits_b) >= thr).mean())
    return {"threshold": thr, "far_nominal": float(far),
            "far_actual": actual, "ratio": actual / far if far > 0 else float("inf")}


def drift(logits_a, logits_b, q=0.99):
    """Сдвиг квантиля B относительно A, выраженный в стандартных отклонениях A.

    Единица выбрана так, чтобы величину можно было сравнивать с зазором
    между дроном и негативом, измеренным на том же корпусе.
    """
    a = np.asarray(logits_a, np.float64)
    s = float(a.std()) + 1e-12
    return float((np.quantile(np.asarray(logits_b, np.float64), q)
                  - np.quantile(a, q)) / s)


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
