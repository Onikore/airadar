"""D0: являются ли соседние индексы DADS кусками одной записи?

От ответа зависит доступный контекст. Клипы DADS длиной 0.6 с; если соседние
номера смежны, 27 часов позитивов восстанавливаются в непрерывные дорожки и
накопление по 4 с даёт заявленные 9 дБ. Если нет — контекст для 86% позитивов
ограничен, выигрыш падает примерно до 5 дБ, и обучение идёт через MIL с
позитивом, положенным в случайное место фона.

Статистика: скачок на стыке в единицах типичного межотсчётного перепада.
У смежного аудио стык ничем не отличается от любой другой точки, отношение
около 1. У несвязанных клипов скачок порядка полного размаха сигнала, а он
для низкочастотно-доминированного звука много больше межотсчётного перепада.
"""

import re
import sys
import numpy as np

_NUM = re.compile(r"(\d+)")


def seam_jump(a, b):
    """Скачок на стыке a|b в единицах типичного межотсчётного перепада a."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = np.median(np.abs(np.diff(a))) + 1e-9
    return float(abs(b[0] - a[-1]) / scale)


def verdict(adj, ctl):
    """Решение по разделению двух распределений скачков.

    Порог намеренно грубый: нас интересует разница на порядок, а не на
    проценты. Промежуточный результат — тоже результат, и он должен
    называться промежуточным, а не округляться в удобную сторону.
    """
    adj, ctl = np.asarray(adj, np.float64), np.asarray(ctl, np.float64)
    ma, mc = float(np.median(adj)), float(np.median(ctl))
    ratio = mc / (ma + 1e-9)
    if ratio > 5.0 and ma < 5.0:
        return True, (f"соседние клипы смежны: медиана скачка {ma:.2f} против "
                      f"{mc:.2f} у контроля (в {ratio:.1f} раз)")
    if ratio < 1.5:
        return False, (f"соседние клипы НЕ смежны: медиана скачка {ma:.2f}, "
                       f"контроль {mc:.2f} — распределения совпадают")
    return False, (f"промежуточный результат: медиана {ma:.2f}, контроль "
                   f"{mc:.2f} (в {ratio:.1f} раз). Смежна лишь часть пар — "
                   f"нужен разбор по группам, не обобщать")


def scan_shard(local_path, n_pairs=300, seed=0):
    """Собирает скачки для соседних пар и для контрольных случайных пар."""
    import pyarrow.parquet as pq
    import io
    import soundfile as sf

    pf = pq.ParquetFile(local_path)
    clips = {}
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=["audio", "label"]).to_pylist():
            if int(r["label"]) != 1:
                continue
            m = _NUM.search(r["audio"]["path"] or "")
            if not m:
                continue
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if sr != 16000 or x.ndim != 1:
                continue
            clips[int(m.group(1))] = x
            if len(clips) > n_pairs * 4:
                break
        if len(clips) > n_pairs * 4:
            break

    keys = sorted(clips)
    adj = [seam_jump(clips[k], clips[k + 1])
           for k in keys[:-1] if k + 1 in clips][:n_pairs]
    rng = np.random.default_rng(seed)
    ctl = []
    for _ in range(len(adj)):
        i, j = rng.integers(0, len(keys), 2)
        if abs(keys[i] - keys[j]) > 10:
            ctl.append(seam_jump(clips[keys[i]], clips[keys[j]]))
    return np.array(adj), np.array(ctl)


def selfcheck():
    # смежные куски одного синуса: стык неотличим от внутренней точки
    t = np.arange(4000) / 16000.0
    x = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    a, b = x[:2000], x[2000:]
    assert seam_jump(a, b) < 3.0, seam_jump(a, b)

    # несвязанные куски: другая фаза даёт скачок много больше межотсчётного.
    # Инверсия (-x) для контроля не годится: обе последовательности проходят
    # через ноль в одной точке, и скачок выходит обманчиво малым.
    c = np.sin(2 * np.pi * 200 * t + 1.5).astype(np.float32)[:2000]
    assert seam_jump(a, c) > 10.0, seam_jump(a, c)

    # решение принимается по разделению распределений, а не по одной паре
    adj = np.full(200, 1.2)
    ctl = np.full(200, 40.0)
    ok, msg = verdict(adj, ctl)
    assert ok and "смежны" in msg, msg
    ok2, msg2 = verdict(ctl, ctl)
    assert not ok2, msg2

    # третья ветка: отношение 2.0 между порогами 1.5 и 5.0 — ни один из двух
    # уверенных вердиктов не должен сработать, иначе перепутанные условия
    # if (смежны/не смежны) молча выдадут ложную уверенность вместо
    # "разбираться отдельно".
    adj3 = np.full(200, 3.0)
    ctl3 = np.full(200, 6.0)
    ok3, msg3 = verdict(adj3, ctl3)
    assert not ok3, msg3
    assert "промежуточ" in msg3, msg3

    print("dads_contiguity selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
