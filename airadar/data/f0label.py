"""f0/salience/lf_energy по манифесту: перенос evalx/f0_survey.py (HPS в
лог-домене) на строки манифеста через ClipReader, вместо старого кэша окон
cache_dads. Алгоритм не меняется — меняется только источник аудио.

Одно значение на клип, не на окно: центральное окно длиной WIN=8000
отсчётов (0.5 с, тот же WIN, что в evalx/f0_survey.py). Клипы короче
WIN — f0_med/salience/lf_energy остаются None. У реальных источников такое
не встречалось (D0, этап 0: минимальная длина позитива DADS 0.6с=9600>WIN),
но проверка есть, чтобы не падать, а не потому что ожидается сработать.
"""
import sys
import numpy as np

SR = 16000
WIN = 8000
NFFT = 32768
F0_LO, F0_HI, F0_STEP = 40.0, 400.0, 0.5
NHARM = 8


def _cand_index():
    f0 = np.arange(F0_LO, F0_HI + 1e-9, F0_STEP)
    k = np.arange(1, NHARM + 1)
    idx = np.rint(f0[:, None] * k[None, :] * NFFT / SR).astype(np.int64)
    idx = np.clip(idx, 0, NFFT // 2)
    return f0, idx


F0_CAND, CAND_IDX = _cand_index()
_WINDOW = np.hanning(WIN).astype(np.float32)
_FREQS = np.fft.rfftfreq(NFFT, 1 / SR)
_LO_MASK = _FREQS < 300.0
_BAND_MASK = (_FREQS >= 40.0) & (_FREQS <= 4000.0)


def f0_salience_lfenergy(w):
    """w: [WIN] float32, пик нормирован (см. airadar/data/clips.py) ->
    (f0_hz, salience_db, lf_energy). lf_energy — доля энергии ниже 300 Гц
    в полосе 40-4000 Гц, тот же признак, что evalx/f0_survey.py:blo."""
    x = np.asarray(w, dtype=np.float32)
    x = x - x.mean()
    P = np.abs(np.fft.rfft(x * _WINDOW, n=NFFT)) ** 2 + 1e-12
    L = 10.0 * np.log10(P)
    S = L[CAND_IDX].mean(axis=1)                 # [n_cand] гармоническая сумма
    j = int(S.argmax())
    f0 = float(F0_CAND[j])
    salience = float(S.max() - np.median(S))
    lf_energy = float(P[_LO_MASK].sum() / (P[_BAND_MASK].sum() + 1e-12))
    return f0, salience, lf_energy


def label_row(reader, offset, n_samples):
    """Возвращает (f0, salience, lf_energy) для строки манифеста, или None,
    если клип короче WIN — центральное окно взять неоткуда."""
    if n_samples < WIN:
        return None
    start = offset + (n_samples - WIN) // 2
    w = reader.read(start, WIN)
    return f0_salience_lfenergy(w)


def selfcheck():
    import tempfile
    import os
    from airadar.data.clips import ClipWriter, ClipReader

    t = np.arange(WIN, dtype=np.float32) / SR
    tone = 0.5 * np.sin(2 * np.pi * 120.0 * t).astype(np.float32)   # f0=120Гц
    for k in range(2, NHARM + 1):
        tone += (0.5 / k) * np.sin(2 * np.pi * 120.0 * k * t).astype(np.float32)

    f0, sal, lf = f0_salience_lfenergy(tone)
    assert abs(f0 - 120.0) < 1.0, f0            # оценщик находит гармонику
    assert sal > 6.0, sal                       # выраженная гребёнка

    silence = np.zeros(WIN, dtype=np.float32)
    _, sal_s, _ = f0_salience_lfenergy(silence)
    assert np.isfinite(sal_s)                   # не падает на тишине

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clips.bin")
        with ClipWriter(path) as w:
            off_long, n_long = w.write(np.tile(tone, 3))               # длиннее WIN
            off_short, n_short = w.write(np.zeros(100, dtype=np.float32))  # короче WIN
        with ClipReader(path) as r:
            got = label_row(r, off_long, n_long)
            assert got is not None
            assert abs(got[0] - 120.0) < 1.0, got

            assert label_row(r, off_short, n_short) is None   # короткий клип пропущен

    print("f0label selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
