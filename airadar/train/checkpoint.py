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
