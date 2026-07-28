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
        limit=None, device=None, run_name="dronenet2", save_every_epoch=False):
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
    # ceil, не floor: DataLoader(drop_last=False, по умолчанию) отдаёт
    # укороченный последний батч, а не отбрасывает его -- при floor-делении
    # реальных шагов за эпоху оказывается на один больше бюджета, и на
    # многоэпоховом прогоне OneCycleLR переполняется на последней эпохе
    # (поймано реальным прогоном на 2000 строк, не селфчеком: там размер
    # датасета случайно делился на bs без остатка и баг был не виден)
    steps_per_epoch = -(-len(ds_tr) // bs)
    steps = max(epochs * steps_per_epoch, 1)
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
            save_checkpoint(os.path.join(out_dir, f"{run_name}_best.pt"), model, opt, sched,
                            ep + 1, feature_cfg, model_cfg, aug_cfg, train_cfg,
                            manifest_path, extra={"val_loss": va_loss})
            tag = "  <- saved"
        save_checkpoint(os.path.join(out_dir, f"{run_name}_last.pt"), model, opt, sched,
                        ep + 1, feature_cfg, model_cfg, aug_cfg, train_cfg,
                        manifest_path, extra={"val_loss": va_loss})
        if save_every_epoch:
            # По эпохе на файл (этап 3г ещё не отбирает чекпоинт по SNR50 —
            # временный критерий val BCE, §6.3, отдельная задача); история
            # по эпохам даёт возможность пересчитать лучший постфактум через
            # bench-харнес, не перезапуская обучение.
            save_checkpoint(os.path.join(out_dir, f"{run_name}_ep{ep+1:03d}.pt"),
                            model, opt, sched, ep + 1, feature_cfg, model_cfg,
                            aug_cfg, train_cfg, manifest_path, extra={"val_loss": va_loss})
        print(f"эпоха {ep+1}/{epochs}  train_loss {tr_loss:.4f}  val_loss {va_loss:.4f}  "
              f"{dt:.1f}с{tag}", flush=True)

    ds_tr.close()
    ds_va.close()


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
        # train = индексы 0-4 (split=0): label [1,1,1,1,0] -> n_pos=4, n_neg=1 -> pw=1/4
        assert abs(pw - 0.25) < 1e-6, pw

        neg_pool = build_neg_pool(manifest_path, split=0)
        assert neg_pool.shape[1] == 2 and len(neg_pool) == 1   # один негатив в train (индекс 4)

        # один реальный шаг обучения на синтетике, GPU если есть иначе CPU
        main(manifest_path, clips_path, epochs=1, bs=2, out_dir=d, limit=4)
        assert os.path.exists(os.path.join(d, "dronenet2_last.pt"))
        assert os.path.exists(os.path.join(d, "dronenet2_best.pt"))

    print("loop selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
