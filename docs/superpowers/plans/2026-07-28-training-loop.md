# Обучающий цикл — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Довести цепочку манифест → `assemble_example` (этап 3б) →
`Frontend` (этап 2) → `DroneNet2` (этап 3а) до РЕАЛЬНОГО обучающего цикла
с чекпоинтом, и получить первый настоящий обученный чекпоинт на реальных
данных (191196 строк, `data/manifest.parquet`) — не просто код, а
подтверждённый прогоном факт, что loss падает и веса сохраняются/
загружаются корректно.

**Architecture:** `ManifestDataset` (PyTorch `Dataset`) оборачивает
`assemble_example` построчно; `collate_batch` левой заливкой нулями
приводит переменную длину примера к фиксированной `target_samples`
(12с) — фиксированная форма батча проще ragged-обработки и уже
отбенчмаркана этапом 2 (94-кадровый случай). Вспомогательная метка
f0/salience считается НЕ из манифеста (там она относится к другому,
исходному окну клипа — после f0-сдвига и подмешивания фона не совпадает
с тем, что видит модель), а заново, на лету, с ЦЕНТРА фактического
4-секундного окна модели, тем же оценщиком, что заполнял манифест
(`airadar/data/f0label.py`, этап 3а) — метка и предсказание всегда
про один и тот же звук.

**Tech Stack:** PyTorch (`Dataset`/`DataLoader`, `AdamW`, `OneCycleLR`,
`BCEWithLogitsLoss`), продолжает `airadar/{config,train/sampler,
features/frontend,models/dronenet2,data/f0label}.py`.

## Global Constraints

- Чекпоинт = веса + `FeatureCfg`/`ModelCfg`/`AugCfg`/`TrainCfg` + git sha +
  хэш манифеста (§8, инвариант, начат этапом 3а).
- Каждый модуль имеет `--selfcheck` (§8, конвенция проекта).
- Целевой размер модуля — до ~200 строк (§8).
- **Не входит в этот план** (сознательно, см. конец плана): отбор
  чекпоинта по SNR50/SWA (§6.3), интеграция с `bench/` (Scorer для
  `DroneNet2`), формальное сравнение с `dronenet_local.pt` по правилу
  §6.2, полный прогон 2 seed. Этот план отбирает чекпоинт по val BCE —
  временный, самый простой критерий, чтобы вообще получить обученные веса
  и проверить, что цикл работает, прежде чем городить SWA поверх
  неработающего цикла.

---

### Task 1: `airadar/train/checkpoint.py`

**Files:**
- Create: `airadar/train/checkpoint.py`

**Interfaces:**
- Produces: `airadar.train.checkpoint.git_sha() -> str`,
  `airadar.train.checkpoint.manifest_hash(path: str) -> str`,
  `airadar.train.checkpoint.save_checkpoint(path, model, opt, sched,
  epoch, feature_cfg, model_cfg, aug_cfg, train_cfg, extra=None) -> None`,
  `airadar.train.checkpoint.load_checkpoint(path, device="cpu") -> dict`
  (возвращает сырой словарь чекпоинта — реконструкцию `DroneNet2`/
  `Frontend` из него делает вызывающий код, Task 3/4, а не этот модуль:
  `checkpoint.py` не должен знать про архитектуру моделей).

- [ ] **Step 1: Написать `airadar/train/checkpoint.py`**

```python
"""Чекпоинт = веса + вся конфигурация, из которой они получены (§8,
инвариант). Раньше (train.py, архивный) чекпоинт хранил только
n_fft/hop/n_mels — часть конфигурации фронтенда терялась, и detect.py
восстанавливал остальное из глобалов модуля, рискуя рассинхроном.
Здесь сериализуется вся связка FeatureCfg/ModelCfg/AugCfg/TrainCfg
целиком, плюс git sha и хэш манифеста — воспроизводимость прогона
проверяема постфактум, а не на честном слове.
"""
import sys
import os
import hashlib
import subprocess
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def manifest_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path, model, opt, sched, epoch, feature_cfg, model_cfg,
                    aug_cfg, train_cfg, manifest_path, extra=None):
    import torch
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "sched": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "feature_cfg": asdict(feature_cfg),
        "model_cfg": asdict(model_cfg),
        "aug_cfg": asdict(aug_cfg),
        "train_cfg": asdict(train_cfg),
        "git_sha": git_sha(),
        "manifest_hash": manifest_hash(manifest_path),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    import torch
    return torch.load(path, map_location=device, weights_only=False)


def selfcheck():
    import tempfile
    import torch
    import torch.nn as nn
    from airadar.config import FeatureCfg, ModelCfg, AugCfg, TrainCfg

    sha = git_sha()
    assert isinstance(sha, str) and len(sha) > 0

    with tempfile.TemporaryDirectory() as d:
        manifest_path = os.path.join(d, "fake_manifest.bin")
        with open(manifest_path, "wb") as f:
            f.write(b"hello manifest")
        h1 = manifest_hash(manifest_path)
        h2 = manifest_hash(manifest_path)
        assert h1 == h2 and len(h1) == 64, h1   # sha256 hex -> 64 символа, детерминирован

        with open(manifest_path, "ab") as f:
            f.write(b"!")
        h3 = manifest_hash(manifest_path)
        assert h3 != h1, "изменение файла обязано менять хэш"

        net = nn.Linear(4, 1)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=10)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sched.step()

        ckpt_path = os.path.join(d, "ck.pt")
        save_checkpoint(ckpt_path, net, opt, sched, epoch=3,
                        feature_cfg=FeatureCfg(), model_cfg=ModelCfg(),
                        aug_cfg=AugCfg(), train_cfg=TrainCfg(),
                        manifest_path=manifest_path, extra={"val_loss": 0.42})

        ck = load_checkpoint(ckpt_path)
        assert ck["epoch"] == 3 and ck["val_loss"] == 0.42
        assert ck["feature_cfg"]["sr"] == 16000
        assert ck["model_cfg"]["branch_hidden"] == 128
        assert ck["aug_cfg"]["snr_db_lo"] == -15.0
        assert ck["train_cfg"]["model_samples"] == 64000
        assert ck["git_sha"] == sha
        assert ck["manifest_hash"] == h3

        net2 = nn.Linear(4, 1)
        net2.load_state_dict(ck["model"])
        for a, b in zip(net.state_dict().values(), net2.state_dict().values()):
            assert torch.equal(a, b)

    print("checkpoint selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать селфчек**

Run: `python -m airadar.train.checkpoint --selfcheck`
Expected: `checkpoint selfcheck ok`

- [ ] **Step 3: Commit**

```bash
git add airadar/train/checkpoint.py
git commit -m "train: checkpoint.py — веса + FeatureCfg/ModelCfg/AugCfg/TrainCfg + git sha + хэш манифеста"
```

---

### Task 2: `airadar/train/dataset.py` — `ManifestDataset`, `collate_batch`

**Files:**
- Create: `airadar/train/dataset.py`

**Interfaces:**
- Consumes: `airadar.train.sampler.assemble_example`,
  `airadar.data.clips.ClipReader`, `airadar.data.f0label.{WIN,
  f0_salience_lfenergy}` (этап 3а — тот же оценщик, что заполнял
  `f0_med`/`salience` манифеста).
- Produces: `airadar.train.dataset.ManifestDataset` (`torch.utils.data.
  Dataset`; `__getitem__(i) -> dict(wav, label, f0, salience, has_aux)`),
  `airadar.train.dataset.collate_batch(items, target_samples) -> (wav:
  Tensor[B,target_samples], label: Tensor[B], f0: Tensor[B], salience:
  Tensor[B], has_aux: Tensor[B] bool)`, `airadar.train.dataset.
  AUX_MIN_SALIENCE = 6.0`.

`ManifestDataset(deterministic=True)` использует `clip_id` строки как
seed ГСЧ — тот же самый пример (то же окно, тот же SNR, тот же f0-сдвиг)
получается КАЖДУЮ эпоху. Нужно для `val`: val loss обязан быть сравним
между эпохами, иначе спад loss нельзя отличить от того, что val-примеры
каждый раз собираются по-новому со случайными SNR/сдвигом. `train` —
`deterministic=False`, свежая случайность каждый вызов, это и есть
аугментация.

- [ ] **Step 1: Написать `airadar/train/dataset.py`**

```python
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
    from airadar.config import TrainCfg

    sr = 16000
    tc = TrainCfg()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clips.bin")
        with ClipWriter(path) as w:
            t = np.arange(round(2.0 * sr), dtype=np.float32) / sr
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

        ds = ManifestDataset(
            offsets=[off_pos] + [o for o, n in neg_offsets],
            n_samples=[n_pos] + [n for o, n in neg_offsets],
            labels=[1] + [0] * 10,
            clip_ids=list(range(11)),
            clips_path=path, neg_pool=neg_pool, deterministic=True)

        assert len(ds) == 11

        item = ds[0]
        assert item["label"] == 1.0
        assert np.isfinite(item["wav"]).all()
        # 150 Гц с явными гармониками -> salience выше порога, has_aux True
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

    print("dataset selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 2: Прогнать селфчек**

Run: `python -m airadar.train.dataset --selfcheck`
Expected: `dataset selfcheck ok`

- [ ] **Step 3: Commit**

```bash
git add airadar/train/dataset.py
git commit -m "train: dataset.py — ManifestDataset (аукс-метка на лету с окна модели), collate_batch"
```

---

### Task 3: `airadar/train/loop.py` — обучающий цикл

**Files:**
- Create: `airadar/train/loop.py`

**Interfaces:**
- Consumes: `airadar.train.dataset.{ManifestDataset, collate_batch}`,
  `airadar.train.checkpoint.save_checkpoint`, `airadar.features.frontend.
  Frontend`, `airadar.models.dronenet2.DroneNet2`,
  `airadar.augment.acoustic.{apply_air_absorption, spec_augment}`,
  `airadar.config.{FeatureCfg, ModelCfg, AugCfg, TrainCfg}`.
- Produces: `airadar.train.loop.build_neg_pool(manifest_path, split) ->
  np.ndarray[K,2] int64`, `airadar.train.loop.pos_weight_for(manifest_path,
  split) -> float`, `airadar.train.loop.train_epoch(model, frontend,
  loader, opt, sched, bce, device, aug_cfg) -> float` (средний loss),
  `airadar.train.loop.eval_epoch(model, frontend, loader, bce, device) ->
  float`, `airadar.train.loop.main(manifest_path, clips_path, epochs, bs,
  lr, out_dir, limit=None, device=None) -> None`.

Аугментация в признаковом домене (затухание верхов, SpecAugment) — здесь,
не в `sampler.py`/`dataset.py`: обе действуют на `[2,183,T]` ПОСЛЕ
`Frontend`, и обе — только для `train`, не для `val` (val должен мерить
качество на чистом, не специально испорченном признаке). `apply_air_absorption`
уже батчевая; `spec_augment` (этап 3б) написана на ОДИН пример — здесь
применяется в цикле по батчу (простой Python-цикл по индексам, не
векторизовано: маскирование — это обнуление пары срезов, стоимость
ничтожна рядом со сверстками и CQT).

- [ ] **Step 1: Добавить `frequencies` property в `Frontend`**

`_feature_augment` (Step 2) нужны частоты бинов CQT-сетки для затухания
верхов. `LogCQT.frequencies` уже есть (этап 2), но `Frontend` его не
экспонирует — вызывающему коду пришлось бы читать приватный
`frontend._logcqt.frequencies`, а не публичный интерфейс. Добавить в
`airadar/features/frontend.py`, класс `Frontend`, рядом с `forward`:

```python
    @property
    def frequencies(self):
        return self._logcqt.frequencies
```

- [ ] **Step 2: Прогнать селфчек фронтенда (не должен был сломаться)**

Run: `python -m airadar.features.frontend --selfcheck`
Expected: `frontend selfcheck ok`

- [ ] **Step 3: Написать `airadar/train/loop.py`**

```python
"""Обучающий цикл DroneNet2. Optimizer/schedule (AdamW + OneCycleLR) и
взвешивание классов через pos_weight — перенесены из train.py (архивный,
проверенный паттерн), не изобретены заново. Что новое — сборка примера
через Frontend + DroneNet2 вместо LogMel + DroneNet, и вспомогательные
потери f0/salience.

Отбор чекпоинта здесь — по val BCE loss, простейший критерий. Отбор по
SNR50 худшей f0-полосы (§6.3) — отдельная задача после того, как этот
цикл подтверждён рабочим на реальном прогоне.
"""
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq

from airadar.config import FeatureCfg, ModelCfg, AugCfg, TrainCfg
from airadar.features.frontend import Frontend
from airadar.models.dronenet2 import DroneNet2
from airadar.augment.acoustic import apply_air_absorption, spec_augment
from airadar.train.dataset import ManifestDataset, collate_batch
from airadar.train.checkpoint import save_checkpoint

AUX_WEIGHT = 0.1


def _read_manifest_columns(manifest_path):
    t = pq.read_table(manifest_path)
    return {
        "split": np.array(t.column("split").to_pylist()),
        "label": np.array(t.column("label").to_pylist()),
        "offset": np.array(t.column("offset").to_pylist()),
        "n_samples": np.array(t.column("n_samples").to_pylist()),
        "clip_id": np.array(t.column("clip_id").to_pylist()),
    }


def build_neg_pool(manifest_path, split):
    cols = _read_manifest_columns(manifest_path)
    sel = (cols["split"] == split) & (cols["label"] == 0)
    return np.stack([cols["offset"][sel], cols["n_samples"][sel]], axis=1).astype(np.int64)


def pos_weight_for(manifest_path, split):
    cols = _read_manifest_columns(manifest_path)
    sel = cols["split"] == split
    n_pos = int((cols["label"][sel] == 1).sum())
    n_neg = int((cols["label"][sel] == 0).sum())
    return n_neg / max(n_pos, 1)


def _feature_augment(feat, freqs, aug_cfg, device):
    """feat: [B,2,F,T] -> с затуханием верхов на ch0 и SpecAugment
    (одинаковая маска на оба канала, по каждому примеру отдельно)."""
    B = feat.shape[0]
    k = torch.empty(B, device=device).uniform_(0.0, aug_cfg.air_k_max)
    feat = feat.clone()
    feat[:, 0] = apply_air_absorption(feat[:, 0], freqs, k)
    rng = np.random.default_rng()
    for i in range(B):
        feat[i] = spec_augment(feat[i], rng, n_masks=aug_cfg.spec_mask_n,
                               max_frac=aug_cfg.spec_mask_frac)
    return feat


def _step(model, frontend, wav, label, f0_t, sal_t, has_aux, bce, device,
         aug_cfg=None, train=True):
    wav = wav.to(device)
    label = label.to(device)
    feat = frontend(wav)
    feat = frontend.last_model_frames(feat)
    if train:
        feat = _feature_augment(feat, frontend.frequencies, aug_cfg, device)
    out = model(feat)

    loss_main = bce(out["clip_logit"], label)

    has_aux = has_aux.to(device)
    if has_aux.any():
        f0_t_d, sal_t_d = f0_t.to(device), sal_t.to(device)
        f0_hat_clip = (out["attn"] * out["f0_hat"]).sum(-1)
        sal_hat_clip = (out["attn"] * out["salience_hat"]).sum(-1)
        f0_loss = ((torch.log2(f0_hat_clip[has_aux]) - torch.log2(f0_t_d[has_aux])) ** 2).mean()
        sal_loss = ((sal_hat_clip[has_aux] - sal_t_d[has_aux]) ** 2).mean()
        aux_loss = f0_loss + sal_loss
    else:
        aux_loss = torch.zeros((), device=device)

    loss = loss_main + AUX_WEIGHT * aux_loss
    return loss, loss_main.detach()


def train_epoch(model, frontend, loader, opt, sched, bce, device, aug_cfg):
    model.train()
    tot, n = 0.0, 0
    for wav, label, f0_t, sal_t, has_aux in loader:
        loss, _ = _step(model, frontend, wav, label, f0_t, sal_t, has_aux,
                        bce, device, aug_cfg, train=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def eval_epoch(model, frontend, loader, bce, device):
    model.eval()
    tot, n = 0.0, 0
    for wav, label, f0_t, sal_t, has_aux in loader:
        _, loss_main = _step(model, frontend, wav, label, f0_t, sal_t, has_aux,
                             bce, device, aug_cfg=None, train=False)
        tot += loss_main.item()
        n += 1
    return tot / max(n, 1)


def main(manifest_path, clips_path, epochs=3, bs=32, lr=3e-4, out_dir="models",
        limit=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cfg, model_cfg, aug_cfg, train_cfg = FeatureCfg(), ModelCfg(), AugCfg(), TrainCfg()

    cols = _read_manifest_columns(manifest_path)
    tr_sel = cols["split"] == 0
    va_sel = cols["split"] == 1
    if limit is not None:
        tr_idx = np.flatnonzero(tr_sel)[:limit]
        va_idx = np.flatnonzero(va_sel)[:max(limit // 4, 8)]
        tr_sel = np.zeros_like(tr_sel); tr_sel[tr_idx] = True
        va_sel_new = np.zeros_like(va_sel); va_sel_new[va_idx] = True
        va_sel = va_sel_new

    neg_pool_tr = build_neg_pool(manifest_path, split=0)
    neg_pool_va = build_neg_pool(manifest_path, split=1)
    pw = pos_weight_for(manifest_path, split=0)
    print(f"train: {int(tr_sel.sum())} строк, val: {int(va_sel.sum())} строк, pos_weight={pw:.3f}")

    ds_tr = ManifestDataset(cols["offset"][tr_sel], cols["n_samples"][tr_sel],
                            cols["label"][tr_sel], cols["clip_id"][tr_sel],
                            clips_path, neg_pool_tr, aug_cfg, train_cfg, deterministic=False)
    ds_va = ManifestDataset(cols["offset"][va_sel], cols["n_samples"][va_sel],
                            cols["label"][va_sel], cols["clip_id"][va_sel],
                            clips_path, neg_pool_va, aug_cfg, train_cfg, deterministic=True)

    collate = lambda items: collate_batch(items, train_cfg.target_samples)
    ld_tr = torch.utils.data.DataLoader(ds_tr, batch_size=bs, shuffle=True, collate_fn=collate)
    ld_va = torch.utils.data.DataLoader(ds_va, batch_size=bs, shuffle=False, collate_fn=collate)

    frontend = Frontend(feature_cfg).to(device)
    model = DroneNet2(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(epochs * (len(ds_tr) // bs), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    os.makedirs(out_dir, exist_ok=True)
    best_val = float("inf")
    for ep in range(epochs):
        t0 = time.time()
        tr_loss = train_epoch(model, frontend, ld_tr, opt, sched, bce, device, aug_cfg)
        va_loss = eval_epoch(model, frontend, ld_va, bce, device)
        dt = time.time() - t0
        tag = ""
        if va_loss < best_val:
            best_val = va_loss
            save_checkpoint(os.path.join(out_dir, "dronenet2_best.pt"), model, opt, sched,
                            ep + 1, feature_cfg, model_cfg, aug_cfg, train_cfg,
                            manifest_path, extra={"val_loss": va_loss})
            tag = "  <- saved"
        save_checkpoint(os.path.join(out_dir, "dronenet2_last.pt"), model, opt, sched,
                        ep + 1, feature_cfg, model_cfg, aug_cfg, train_cfg,
                        manifest_path, extra={"val_loss": va_loss})
        print(f"эпоха {ep+1}/{epochs}  train_loss {tr_loss:.4f}  val_loss {va_loss:.4f}  "
              f"{dt:.1f}с{tag}", flush=True)


def selfcheck():
    """Проверяет arifметику pos_weight/neg_pool на синтетическом манифесте
    и один шаг обучения на синтетических данных — не полный прогон
    (см. cli/train.py --limit для реального смоук-теста)."""
    import tempfile
    import pyarrow as pa
    from airadar.data.clips import ClipWriter

    with tempfile.TemporaryDirectory() as d:
        clips_path = os.path.join(d, "clips.bin")
        rows = []
        with ClipWriter(clips_path) as w:
            for i in range(6):
                label = 1 if i < 4 else 0   # 4 позитива, 2 негатива -> pos_weight = 2/4=0.5
                audio = np.random.default_rng(i).standard_normal(70000).astype(np.float32) * 0.05
                off, n = w.write(audio)
                rows.append({"offset": off, "n_samples": n, "label": label,
                            "split": 0 if i < 5 else 1, "clip_id": i})

        manifest_path = os.path.join(d, "manifest.parquet")
        table = pa.table({
            "offset": [r["offset"] for r in rows],
            "n_samples": [r["n_samples"] for r in rows],
            "label": [r["label"] for r in rows],
            "split": [r["split"] for r in rows],
            "clip_id": [r["clip_id"] for r in rows],
        })
        import pyarrow.parquet as pq
        pq.write_table(table, manifest_path)

        pw = pos_weight_for(manifest_path, split=0)
        assert abs(pw - 2 / 3) < 1e-6, pw   # train: label1=3,label0=2 (индексы 0-4) -> 2/3

        neg_pool = build_neg_pool(manifest_path, split=0)
        assert neg_pool.shape[1] == 2 and len(neg_pool) == 1   # один негатив в train (индекс 4)

        # один реальный шаг обучения на синтетике, GPU если есть иначе CPU
        main(manifest_path, clips_path, epochs=1, bs=2, out_dir=d, limit=4)
        assert os.path.exists(os.path.join(d, "dronenet2_last.pt"))
        assert os.path.exists(os.path.join(d, "dronenet2_best.pt"))

    print("loop selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
```

- [ ] **Step 4: Прогнать селфчек**

Run: `python -m airadar.train.loop --selfcheck`
Expected: `loop selfcheck ok` (запускает один реальный шаг обучения на
синтетике — медленнее обычного селфчека, но не минуты: 4 примера, 1
эпоха).

- [ ] **Step 5: Commit**

```bash
git add airadar/features/frontend.py airadar/train/loop.py
git commit -m "train: loop.py — обучающий цикл (AdamW+OneCycleLR, pos_weight, аукс-потери, отбор по val loss)"
```

---

### Task 4: `cli/train.py`

**Files:**
- Create: `cli/train.py`

**Interfaces:**
- Consumes: `airadar.train.loop.main`.

- [ ] **Step 1: Написать `cli/train.py`**

```python
"""CLI: обучение DroneNet2 на реальном манифесте.

    python cli/train.py --limit 200 --epochs 2     # смоук-тест, минуты
    python cli/train.py --epochs 30                # полный прогон
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from airadar.train.loop import main

MANIFEST_PATH = os.path.join(ROOT, "data", "manifest.parquet")
CLIPS_PATH = os.path.join(ROOT, "data", "clips.bin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "models"))
    a = ap.parse_args()
    main(MANIFEST_PATH, CLIPS_PATH, epochs=a.epochs, bs=a.bs, lr=a.lr,
        out_dir=a.out_dir, limit=a.limit)
```

- [ ] **Step 2: Смоук-тест на маленьком куске реального манифеста**

Run: `python cli/train.py --limit 64 --epochs 1 --bs 8`
Expected: одна строка `эпоха 1/1  train_loss ...  val_loss ...  N.Nс`,
без ошибок, `models/dronenet2_last.pt` и `models/dronenet2_best.pt`
создаются.

- [ ] **Step 3: Прогнать общий свип селфчеков**

Run: `python cli/selfcheck.py`
Expected: `прогнано N/N модулей, освобождено 1` — этот план добавил 3
новых модуля с `selfcheck` (`checkpoint`, `dataset`, `loop`) к прежним 30
(этап 3б) -> 33. Поднять `MIN_CHECKS` в `cli/selfcheck.py` с `30` до `33`.

- [ ] **Step 4: Commit**

```bash
git add cli/train.py cli/selfcheck.py
git commit -m "cli: train.py — тонкая обёртка над train/loop.py"
```

---

### Task 5: Реальный короткий прогон обучения (не код — проверка фактом)

Эта задача не пишет новый код — она подтверждает, что весь цикл реально
работает на реальных данных, а не только на синтетике селфчеков.

- [ ] **Step 1: Прогон на ограниченном куске реального манифеста, 3 эпохи**

Run: `python cli/train.py --limit 2000 --epochs 3 --bs 32`

Записать в `logs/train_smoke.log`:
```bash
python cli/train.py --limit 2000 --epochs 3 --bs 32 2>&1 | tee logs/train_smoke.log
```

Expected: 3 строки `эпоха N/3 ...`, `train_loss` не `nan`/`inf` ни на
одной эпохе. Не требуется, чтобы `val_loss` монотонно падал (2000 строк,
3 эпохи — не про качество, про то, что цикл живой), но он обязан быть
конечным числом каждую эпоху.

- [ ] **Step 2: Проверить, что чекпоинт реально загружается и считает**

```bash
python -c "
import torch
from airadar.train.checkpoint import load_checkpoint
from airadar.config import FeatureCfg, ModelCfg
from airadar.features.frontend import Frontend
from airadar.models.dronenet2 import DroneNet2

ck = load_checkpoint('models/dronenet2_last.pt')
fc = FeatureCfg(**ck['feature_cfg'])
mc = ModelCfg(**ck['model_cfg'])
fe = Frontend(fc)
model = DroneNet2(mc)
model.load_state_dict(ck['model'])
model.eval()

wav = torch.zeros(2, fc.hop_length * 94)   # произвольный вход подходящей длины
feat = fe.last_model_frames(fe(wav))
with torch.no_grad():
    out = model(feat)
assert out['clip_logit'].shape == (2,)
assert torch.isfinite(out['clip_logit']).all()
print('checkpoint round-trip ok, epoch', ck['epoch'], 'val_loss', ck['val_loss'])
print('git_sha', ck['git_sha'][:8], 'manifest_hash', ck['manifest_hash'][:16])
"
```
Expected: `checkpoint round-trip ok, epoch 3 val_loss <число>`, печатает
реальные `git_sha`/`manifest_hash` — подтверждает, что чекпоинт несёт
происхождение, не только веса.

- [ ] **Step 3: Записать реальные числа в резюме задачи**

В коммит-сообщении (Step 4) записать фактическое время эпохи и train/val
loss по эпохам — измеренные числа, не оценка. Если `train_loss` не
уменьшается за 3 эпохи — это не блокер этой задачи (3 эпохи на 2000
строк ничего не доказывают о качестве модели), но должно быть отмечено
честно, а не замолчано.

- [ ] **Step 4: Commit**

```bash
git add logs/train_smoke.log
git commit -m "train: первый реальный прогон (смоук, 2000 строк, 3 эпохи) — цикл подтверждён рабочим"
```

---

## Что эта задача НЕ делает (сознательно, не забыто)

- **Отбор чекпоинта по SNR50 худшей f0-полосы + SWA (§6.3).** Этот план
  отбирает по val BCE loss — простейший работающий критерий, не критерий
  спецификации. Нужен `Scorer` для `DroneNet2` (протокол `airadar/bench/
  scorer.py`, этап 0) и пересборка `airadar/bench/strata.py` на f0-полосы
  из МАНИФЕСТА (`f0_med`, этап 3а), а не из старого `evalx/f0_dads_*.npz`
  — отдельная задача.
- **Формальное сравнение с `dronenet_local.pt` по правилу §6.2** (знак
  согласован на обеих полевых записях, CI не пересекает ноль, разница
  больше seed-разброса). Требует Scorer выше и полного прогона.
- **Полный прогон, 2 seed (§9, этап 3).** Эта задача — смоук на 2000
  строк, 3 эпохи, для проверки цикла. Полный прогон на всех ~145k
  train-строках, десятки эпох, 2 независимых seed — следующий шаг после
  того, как есть Scorer для сравнения результатов.
- **Балансировка состава батча.** `pos_weight` в BCE взвешивает ТОЛЬКО
  функцию потерь; сам батч по-прежнему собирается uniform-сэмплингом по
  индексам, то есть в среднем ~85% позитивный по факту (реальные train-
  счётчики манифеста: label=1 122857, label=0 21016). Для `BatchNorm`
  внутри `DroneNet2` это означает, что статистика батчей в среднем
  смещена к "дрон слышен". Не исправлено в этом плане — если реальный
  прогон (следующий план) покажет, что это мешает обучению, стратифицированный
  сэмплер — точечная правка `ManifestDataset`, не архитектурная.