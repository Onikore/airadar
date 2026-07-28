"""Dataset поверх манифеста: одна строка манифеста -> один обучающий
пример через airadar.train.sampler.assemble_example.

Метка f0/salience для вспомогательной головы считается ЗАНОВО на лету, не
берётся из колонок манифеста: f0_med/salience манифеста относятся к
исходному, НЕаугментированному клипу (этап 3а, f0label.py), а после
f0-сдвига (r != 1) и вложения в фон (короткий позитив) настоящая частота
в окне, которое видит модель, уже другая. Оценщик тот же
(airadar.data.f0label.f0_salience_lfenergy) — только считается по
фактическому центру 4-секундного окна МОДЕЛИ (последние model_samples
собранного примера), а не по центру исходного клипа.
"""
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

from airadar.data.clips import ClipReader
from airadar.data.f0label import WIN, f0_salience_lfenergy
from airadar.train.sampler import assemble_example

AUX_MIN_SALIENCE = 6.0   # как evalx/f0_survey.load_f0_estimates — слабую гребёнку не учим


class ManifestDataset(Dataset):
    def __init__(self, offsets, n_samples, labels, clip_ids, clips_path,
                neg_pool, aug_cfg=None, train_cfg=None, deterministic=False):
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.n_samples = np.asarray(n_samples, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.clip_ids = np.asarray(clip_ids, dtype=np.int64)
        self.clips_path = clips_path
        self.neg_pool = neg_pool
        self.aug_cfg = aug_cfg
        self.train_cfg = train_cfg
        self.deterministic = deterministic
        self._reader = None

    def __len__(self):
        return len(self.labels)

    def _reader_(self):
        if self._reader is None:
            self._reader = ClipReader(self.clips_path)
        return self._reader

    def close(self):
        # На Windows незакрытый memmap держит clips.bin залоченным (см.
        # airadar/data/clips.py) — важно освобождать явно, а не полагаться
        # на сборщик мусора, иначе следующий открывающий процесс/тест упадёт.
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __getitem__(self, i):
        from airadar.config import TrainCfg
        train_cfg = self.train_cfg or TrainCfg()
        row = {"offset": int(self.offsets[i]), "n_samples": int(self.n_samples[i]),
               "label": int(self.labels[i])}
        rng = (np.random.default_rng(int(self.clip_ids[i])) if self.deterministic
               else np.random.default_rng())
        wav, label, meta = assemble_example(row, self._reader_(), self.neg_pool, rng,
                                            aug_cfg=self.aug_cfg, train_cfg=train_cfg)

        tail = wav[-train_cfg.model_samples:]
        c = len(tail) // 2
        center = tail[c - WIN // 2: c - WIN // 2 + WIN]
        f0, sal, _ = f0_salience_lfenergy(center)

        return {"wav": wav, "label": np.float32(label), "f0": np.float32(f0),
               "salience": np.float32(sal), "has_aux": bool(sal >= AUX_MIN_SALIENCE)}


def collate_batch(items, target_samples):
    B = len(items)
    wavs = np.zeros((B, target_samples), dtype=np.float32)
    for i, it in enumerate(items):
        w = it["wav"]
        n = min(len(w), target_samples)
        wavs[i, target_samples - n:] = w[-n:]   # левый паддинг, хвост -- реальное окно
    labels = np.array([it["label"] for it in items], dtype=np.float32)
    f0 = np.array([it["f0"] for it in items], dtype=np.float32)
    sal = np.array([it["salience"] for it in items], dtype=np.float32)
    has_aux = np.array([it["has_aux"] for it in items], dtype=bool)
    return (torch.from_numpy(wavs), torch.from_numpy(labels),
            torch.from_numpy(f0), torch.from_numpy(sal), torch.from_numpy(has_aux))


def selfcheck():
    import tempfile
    import os
    from airadar.data.clips import ClipWriter
    from airadar.config import TrainCfg, AugCfg

    sr = 16000
    tc = TrainCfg()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clips.bin")
        with ClipWriter(path) as w:
            # клип ДЛИННЕЕ model_samples (4с): иначе assemble_example
            # берёт режим "short" и кладёт тон на случайный офсет в
            # канву фона при случайном SNR (вплоть до -15дБ) — метка f0
            # тогда законно может не совпасть с чистым тоном
            dur = tc.model_samples + 20000
            t = np.arange(dur, dtype=np.float32) / sr
            tone = (0.3 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)
            for k in range(2, 6):
                tone += (0.3 / k) * np.sin(2 * np.pi * 150.0 * k * t).astype(np.float32)
            off_pos, n_pos = w.write(tone)

            neg_offsets = []
            for i in range(10):
                neg = np.random.default_rng(i).standard_normal(tc.model_samples).astype(np.float32) * 0.02
                o, n = w.write(neg)
                neg_offsets.append((o, n))

        neg_pool = np.array(neg_offsets, dtype=np.int64)
        # без сдвига f0 и с высоким принудительным SNR: f0-метка обязана
        # совпасть с чистым тоном детерминированно, а не "обычно совпадает"
        clean_cfg = AugCfg(pitch_prob=0.0, snr_db_lo=20.0, snr_db_hi=20.0, hum_prob=0.0)

        ds = ManifestDataset(
            offsets=[off_pos] + [o for o, n in neg_offsets],
            n_samples=[n_pos] + [n for o, n in neg_offsets],
            labels=[1] + [0] * 10,
            clip_ids=list(range(11)),
            clips_path=path, neg_pool=neg_pool, aug_cfg=clean_cfg, deterministic=True)

        assert len(ds) == 11

        item = ds[0]
        assert item["label"] == 1.0
        assert np.isfinite(item["wav"]).all()
        # 150 Гц с явными гармониками, SNR=20дБ -> salience выше порога
        assert item["has_aux"] is True, item["salience"]
        assert abs(item["f0"] - 150.0) < 5.0, item["f0"]

        # детерминизм: тот же индекс -> тот же результат
        item_again = ds[0]
        assert np.array_equal(item["wav"], item_again["wav"])
        assert item["f0"] == item_again["f0"]

        items = [ds[i] for i in range(4)]
        wav, label, f0, sal, has_aux = collate_batch(items, tc.target_samples)
        assert wav.shape == (4, tc.target_samples)
        assert label.shape == (4,) and label.dtype == torch.float32
        assert f0.shape == (4,) and sal.shape == (4,)
        assert has_aux.shape == (4,) and has_aux.dtype == torch.bool
        assert label[0].item() == 1.0

        # левый паддинг: хвост батч-строки 0 обязан совпасть с последними
        # отсчётами исходного wav этого примера
        n0 = min(len(items[0]["wav"]), tc.target_samples)
        assert torch.allclose(wav[0, -n0:], torch.from_numpy(items[0]["wav"][-n0:]))
        if n0 < tc.target_samples:
            assert torch.all(wav[0, :tc.target_samples - n0] == 0.0)

        ds.close()   # освобождает memmap ДО выхода из TemporaryDirectory (Windows)

    print("dataset selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
