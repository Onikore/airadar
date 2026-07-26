"""Загрузка корпусов и сборка непрерывного фона.

Кэши состоят из нарезанных окон, а метрика FA/час требует непрерывности:
событие определено на дорожке, а не на мешке окон. Клипы склеиваются обратно
с кроссфейдом 50 мс, а окна, пересекающие стык, исключаются из подсчёта —
скачок уровня на стыке читается детектором как событие и завышает FA/час.
"""

import os
import sys
import glob
import wave
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SR = 16000


def selfcheck():
    # склейка: длина = сумма минус перекрытия, стыки на своих местах
    a = np.ones(1600, np.float32)
    b = np.full(1600, 2.0, np.float32)
    xf = 800                                   # 0.05 с при 16 кГц
    out, seams = stitch([a, b, a], xfade_s=0.05)
    assert len(out) == 3 * 1600 - 2 * xf, len(out)
    assert list(seams) == [1600 - xf, 2 * 1600 - 2 * xf], seams
    # кроссфейд монотонно переводит 1.0 в 2.0, разрывов нет
    seg = out[seams[0]:seams[0] + xf]
    assert seg[0] < seg[-1] and np.all(np.diff(seg) >= -1e-6)

    # маска стыков: окно 0.5 с шагом 0.25 с, стык на 1.0 с (отсчёт 16000)
    m = seam_mask(6, np.array([16000]), context_s=0.5, hop_s=0.25)
    # окна начинаются в 0.00 0.25 0.50 0.75 1.00 1.25 и длятся 0.5 с.
    # Стык в 1.0 с пересекает только окно, начинающееся в 0.75 (оно идёт
    # до 1.25). Окно с 0.50 кончается ровно на стыке и его НЕ пересекает —
    # границы полуоткрыты, иначе маска выбрасывала бы вдвое больше нужного.
    assert list(m) == [True, True, True, False, True, True], list(m)

    # короткий кроссфейд не должен превышать длину клипа
    try:
        stitch([np.ones(100, np.float32)] * 2, xfade_s=0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("stitch не поймал клип короче кроссфейда")

    # восстановление исходных клипов: соседние окна одной группы смежны
    # по построению (prep_hf.windows режет подряд и без перекрытия), поэтому
    # склеиваются встык, без кроссфейда, и стык внутри группы не возникает
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    g = np.array([7, 7, 7, 9, 9, 7])
    tr = regroup(X, g)
    assert len(tr) == 2, len(tr)
    assert list(tr[0]) == list(range(0, 12)) + list(range(20, 24)), tr[0]
    assert list(tr[1]) == list(range(12, 20)), tr[1]

    print("corpus selfcheck ok")


def stitch(clips, xfade_s=0.05):
    """Склейка клипов в непрерывную дорожку с кроссфейдом.

    Возвращает (дорожка, позиции стыков). Позиция стыка — начало зоны
    кроссфейда: именно там уровень меняется, и именно эту зону надо
    вырезать из подсчёта событий.
    """
    xf = int(round(xfade_s * SR))
    for c in clips:
        if len(c) < 2 * xf:
            raise ValueError(f"клип {len(c)} отсчётов короче двух кроссфейдов {2*xf}")
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    out = clips[0].astype(np.float32).copy()
    seams = []
    for c in clips[1:]:
        c = c.astype(np.float32)
        seams.append(len(out) - xf)
        out[-xf:] = out[-xf:] * (1.0 - ramp) + c[:xf] * ramp
        out = np.concatenate([out, c[xf:]])
    return out, np.array(seams, np.int64)


def seam_mask(n, seams, context_s, hop_s, sr=SR):
    """True для окон, НЕ пересекающих ни один стык."""
    ctx, hop = int(round(context_s * sr)), int(round(hop_s * sr))
    starts = np.arange(n) * hop
    ok = np.ones(n, bool)
    for s in np.atleast_1d(seams):
        ok &= ~((starts < s) & (starts + ctx > s))
    return ok


def read_wav_mono16k(path):
    with wave.open(path) as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            raise ValueError(f"{path}: нужен моно {SR} Гц, "
                             f"а тут {w.getnchannels()} кан. {w.getframerate()} Гц")
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: нужно 16 бит (2 байта/отсчёт), "
                             f"а тут {w.getsampwidth() * 8} бит")
        raw = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return raw.astype(np.float32) / 32768.0


def field_records(pattern="field/drone_video*.wav"):
    """Полевые записи целиком, НЕ нарезанные.

    Не усредняются между собой: у них разная основная частота (78.0 и
    60.5 Гц), и общее число спрятало бы, что одна пропускается целиком.
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        out[os.path.basename(p)] = read_wav_mono16k(p)
    if not out:
        raise FileNotFoundError(f"нет записей по шаблону {pattern}")
    return out


def load_cache(name):
    """Кэш окон: (X memmap [N,8000] int16, y, split, cat, group).

    Ключи meta.npz сверены с диском 2026-07-26:
    y, group, src, split, synth, n, win (+ cat, hard в cache_hard).
    """
    d = os.path.join(ROOT, name)
    meta = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    n, win = int(meta["n"]), int(meta["win"])
    X = np.memmap(os.path.join(d, "windows.bin"), np.int16, "r", shape=(n, win))
    cat = meta["cat"] if "cat" in meta.files else np.full(n, "", "<U16")
    return X, meta["y"], meta["split"], cat, meta["group"]


def hard_holdout(cat_filter=None):
    """Удержанные трудные негативы. split != 0 — то, чего не было в обучении."""
    X, y, split, cat, group = load_cache("cache_hard")
    sel = np.flatnonzero(split != 0)
    if cat_filter is not None:
        sel = sel[np.isin(cat[sel], list(cat_filter))]
    sel = np.sort(sel)
    return (np.ascontiguousarray(X[sel]).astype(np.float32) / 32768.0,
            cat[sel], group[sel])


def regroup(X, group):
    """Восстановить исходные клипы из окон кэша.

    prep_hf.windows режет клип на подряд идущие непересекающиеся куски,
    поэтому окна одной группы склеиваются ВСТЫК и дают ровно исходное аудио.
    Это принципиально: если склеивать 0.5-секундные окна с кроссфейдом, стык
    возникает каждые 0.45 с, а при контексте 0.5 с его задевает каждое окно —
    считать FA/час становится не на чем.
    """
    order = np.argsort(group, kind="stable")
    out, i = [], 0
    while i < len(order):
        j = i
        while j < len(order) and group[order[j]] == group[order[i]]:
            j += 1
        out.append(np.concatenate([X[k] for k in order[i:j]]))
        i = j
    return out


def hard_categories():
    """16 трудных категорий (механический гул и погода) из meta["hard"].

    Список зафиксирован при сборке кэша, а не продублирован здесь: копия
    разъехалась бы с данными при первой же пересборке, и метрика молча
    начала бы считаться по другому пулу.
    """
    meta = np.load(os.path.join(ROOT, "cache_hard", "meta.npz"),
                   allow_pickle=True)
    return meta["hard"]


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
