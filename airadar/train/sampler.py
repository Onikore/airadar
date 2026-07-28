"""Сборка одного обучающего примера из строки манифеста (этап 1) —
единственное место, где решается, что делать с коротким позитивом DADS
(86% данных, D0: клипы не смежны физически, см. спецификация §5.4)
против длинного непрерывного позитива DAS.

Три режима по длине клипа:
  - длинный (>= target_samples, 12с): случайное окно нужной длины прямо
    из клипа — контекста достаточно для полного 8с+4с окна модели.
  - средний (>= model_samples, 4с, но короче 12с): клип целиком — короче
    идеала, но Frontend/last_model_frames справляется (даёт меньше
    "истории" для ch1, не меньше кадров модели).
  - короткий (< model_samples): клип кладётся на случайный офсет внутри
    канвы длиной model_samples, остальное — фон при случайном SNR.
    Деградационная ветка D0 (этап 0): "позитив 0.6с кладётся в случайное
    место 4-секундного фона" — применена буквально.

Для негативов (label=0) режимы длины те же, но без вложения на офсет
(негатив негативен целиком, вкладывать некуда) — только выбор окна
нужной длины из его собственного аудио, с зацикливанием, если аудио
короче model_samples.
"""
import sys
import numpy as np

from airadar.augment.pitch import sample_r, pitch_shift
from airadar.augment.hum import add_hum
from airadar.augment.mixing import snr_scale, mix_at_snr, random_gain, place_at_offset, normalize_rms
from airadar.augment.acoustic import cyclic_shift


def draw_background(neg_pool, reader, length, rng):
    """neg_pool: [K,2] int64 (offset, n_samples) строк label=0 -> [length]
    float32, случайный негатив нужной длины (зацикленный, если короче)."""
    i = int(rng.integers(0, len(neg_pool)))
    offset, n = int(neg_pool[i, 0]), int(neg_pool[i, 1])
    audio = reader.read(offset, n)
    if n >= length:
        start = int(rng.integers(0, n - length + 1))
        return audio[start:start + length].astype(np.float32)
    reps = -(-length // n)   # ceil division — зацикливание, шов не сглаживается
    return np.tile(audio, reps)[:length].astype(np.float32)


def _own_window(audio, length, rng):
    """audio: [n] float32 -> [length] float32: случайное окно, если
    audio длиннее length, иначе зацикленное audio."""
    n = len(audio)
    if n >= length:
        start = int(rng.integers(0, n - length + 1))
        return audio[start:start + length].astype(np.float32)
    reps = -(-length // n)
    return np.tile(audio, reps)[:length].astype(np.float32)


def assemble_example(row, reader, neg_pool, rng, aug_cfg=None, train_cfg=None):
    from airadar.config import AugCfg, TrainCfg
    aug_cfg = aug_cfg or AugCfg()
    train_cfg = train_cfg or TrainCfg()

    label = int(row["label"])
    meta = {"snr_db": None, "pitch_r": None, "mode": None, "hum_added": False}

    if label == 1:
        pos = reader.read(row["offset"], row["n_samples"]).astype(np.float32)
        if rng.random() < aug_cfg.pitch_prob:
            r = sample_r(rng, aug_cfg)
            pos = pitch_shift(pos, r)
            meta["pitch_r"] = r
        n = len(pos)

        if n >= train_cfg.target_samples:
            meta["mode"] = "long"
            start = int(rng.integers(0, n - train_cfg.target_samples + 1))
            signal = pos[start:start + train_cfg.target_samples]
            bg = draw_background(neg_pool, reader, len(signal), rng)
            snr_db = float(rng.uniform(aug_cfg.snr_db_lo, aug_cfg.snr_db_hi))
            wav = mix_at_snr(signal, bg, snr_db)
            meta["snr_db"] = snr_db
        elif n >= train_cfg.model_samples:
            meta["mode"] = "medium"
            signal = pos
            bg = draw_background(neg_pool, reader, len(signal), rng)
            snr_db = float(rng.uniform(aug_cfg.snr_db_lo, aug_cfg.snr_db_hi))
            wav = mix_at_snr(signal, bg, snr_db)
            meta["snr_db"] = snr_db
        else:
            meta["mode"] = "short"
            canvas_len = train_cfg.model_samples
            bg_full = draw_background(neg_pool, reader, canvas_len, rng)
            canvas, offset = place_at_offset(pos, canvas_len, rng)
            meta["offset"] = offset
            snr_db = float(rng.uniform(aug_cfg.snr_db_lo, aug_cfg.snr_db_hi))
            # масштаб фона считается ПО ЛОКАЛЬНОМУ участку (позитив против
            # фона ровно под ним), применяется ко всей канве фона —
            # см. докстринг airadar/augment/mixing.py:snr_scale
            local_bg = bg_full[offset:offset + len(pos)]
            scale = snr_scale(pos, local_bg, snr_db)
            wav = canvas + bg_full * scale
            meta["snr_db"] = snr_db
    else:
        if rng.random() < aug_cfg.hum_only_prob:
            meta["mode"] = "hum_only"
            wav = np.zeros(train_cfg.model_samples, dtype=np.float32)
        else:
            meta["mode"] = "negative"
            neg = reader.read(row["offset"], row["n_samples"]).astype(np.float32)
            # тот же трёхуровневый выбор длины, что у позитивов выше: клип
            # средней длины используется целиком, а не обрезается до
            # model_samples — иначе доступный контекст терялся бы без причины
            if row["n_samples"] >= train_cfg.target_samples:
                length = train_cfg.target_samples
            elif row["n_samples"] >= train_cfg.model_samples:
                length = row["n_samples"]
            else:
                length = train_cfg.model_samples
            wav = _own_window(neg, length, rng)

    # универсальная пост-обработка сырого аудио, для обеих меток
    wav = cyclic_shift(wav, rng)
    if meta["mode"] == "hum_only" or rng.random() < aug_cfg.hum_prob:
        wav = add_hum(wav, rng, amp_max=aug_cfg.hum_amp_max)
        meta["hum_added"] = True
    # RMS-нормализация ПОСЛЕ смешивания и гула, ДО random_gain: mix_at_snr
    # складывает сигнал и фон без последующей нормализации (позитив =
    # сигнал+фон, негатив = только фон) — сумма систематически громче
    # одного слагаемого, и без этого шага позитивы медианно вдвое громче
    # негативов (найдено систематической отладкой на реальном чекпоинте:
    # клип_логит менял знак на РЕАЛЬНОЙ полевой записи при простом
    # усилении громкости в 10-20 раз без изменения содержимого — модель
    # училась громкости, не гребёнке). Пик не трогаем (щелчок внутри окна
    # не должен диктовать масштаб — ровно за это пиковую убрали раньше),
    # RMS от единичного импульса почти не двигается.
    wav = normalize_rms(wav, target_rms=aug_cfg.target_rms)
    wav = random_gain(wav, rng, lo=aug_cfg.gain_db_lo, hi=aug_cfg.gain_db_hi)

    assert len(wav) >= train_cfg.model_samples, (len(wav), train_cfg.model_samples)
    return wav, label, meta


def selfcheck():
    import tempfile
    import os
    from airadar.data.clips import ClipWriter, ClipReader
    from airadar.config import AugCfg, TrainCfg

    sr = 16000
    tc = TrainCfg()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clips.bin")
        with ClipWriter(path) as w:
            # длинный позитив (12с+) — режим "long"
            long_pos = np.random.default_rng(1).standard_normal(tc.target_samples + 1000).astype(np.float32) * 0.1
            off_long, n_long = w.write(long_pos)
            # средний позитив (между model_samples и target_samples)
            med_pos = np.random.default_rng(2).standard_normal(tc.model_samples + 5000).astype(np.float32) * 0.1
            off_med, n_med = w.write(med_pos)
            # короткий позитив (0.6с, как DADS) — режим "short"
            short_pos = np.random.default_rng(3).standard_normal(round(0.6 * sr)).astype(np.float32) * 0.1
            off_short, n_short = w.write(short_pos)
            # негативы для neg_pool — разной длины
            neg_offsets = []
            for i in range(20):
                neg = np.random.default_rng(100 + i).standard_normal(tc.model_samples).astype(np.float32) * 0.05
                o, n = w.write(neg)
                neg_offsets.append((o, n))

        neg_pool = np.array(neg_offsets, dtype=np.int64)

        with ClipReader(path) as reader:
            rng = np.random.default_rng(42)
            # pitch_prob=0 здесь: длина после сдвига f0 умножается на 1/r,
            # r в [0.35, 1.5] может перекинуть клип через границу
            # длинный/средний/короткий — эти два случая проверяют именно
            # выбор режима ПО ДЛИНЕ, сдвиг проверен отдельно в pitch.py
            no_pitch = AugCfg(pitch_prob=0.0)

            row_long = {"offset": off_long, "n_samples": n_long, "label": 1}
            wav, label, meta = assemble_example(row_long, reader, neg_pool, rng,
                                                aug_cfg=no_pitch)
            assert label == 1 and meta["mode"] == "long"
            assert len(wav) >= tc.model_samples
            assert np.isfinite(wav).all()

            row_med = {"offset": off_med, "n_samples": n_med, "label": 1}
            wav, label, meta = assemble_example(row_med, reader, neg_pool, rng,
                                                aug_cfg=no_pitch)
            assert meta["mode"] == "medium"
            assert len(wav) >= tc.model_samples

            row_short = {"offset": off_short, "n_samples": n_short, "label": 1}
            wav, label, meta = assemble_example(row_short, reader, neg_pool, rng)
            assert meta["mode"] == "short"
            assert len(wav) == tc.model_samples   # короткий позитив -> ровно окно модели
            assert 0 <= meta["offset"] <= tc.model_samples - n_short

            row_neg = {"offset": neg_offsets[0][0], "n_samples": neg_offsets[0][1], "label": 0}
            wav, label, meta = assemble_example(row_neg, reader, neg_pool, rng)
            assert label == 0
            assert len(wav) >= tc.model_samples

            # hum_only реально срабатывает при hum_only_prob=1.0
            cfg_always_hum = AugCfg(hum_only_prob=1.0)
            wav, label, meta = assemble_example(row_neg, reader, neg_pool, rng,
                                                aug_cfg=cfg_always_hum)
            assert meta["mode"] == "hum_only" and meta["hum_added"] is True

            # детерминированность: тот же seed -> тот же результат
            rng_a = np.random.default_rng(7)
            rng_b = np.random.default_rng(7)
            wav_a, _, _ = assemble_example(row_short, reader, neg_pool, rng_a)
            wav_b, _, _ = assemble_example(row_short, reader, neg_pool, rng_b)
            assert np.array_equal(wav_a, wav_b)

    print("sampler selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
