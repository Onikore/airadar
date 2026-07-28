"""Единственный мост между харнесом и моделями.

Харнес не должен знать, что внутри модели. Он знает одно: дай непрерывное
аудио — получи ряд логитов с известным шагом. Тогда нынешняя 0.5-секундная
модель и будущая 4-секундная меряются одним и тем же кодом, а сравнение
между ними осмысленно.

Возвращаются логиты, а не вероятности: sigmoid в float32 упирается в 1.0 и
стирает порядок в верхнем хвосте распределения — ровно там, где стоит
рабочая точка.
"""

import os
import sys
from typing import Protocol

import numpy as np

SR = 16000


class Scorer(Protocol):
    """Контракт, который обязан выполнять любой детектор.

    hop_s     — шаг между соседними оценками
    context_s — сколько секунд аудио нужно на одну оценку
    score     — float32 [T] -> float32 [n_scores(T)], ЛОГИТЫ
    """

    hop_s: float
    context_s: float

    def score(self, audio: np.ndarray) -> np.ndarray: ...


def n_scores(n_samples, context_s, hop_s, sr=SR):
    """Сколько оценок даст ряд из n_samples отсчётов."""
    ctx, hop = int(round(context_s * sr)), int(round(hop_s * sr))
    if n_samples < ctx:
        return 0
    return 1 + (n_samples - ctx) // hop


def frame_times(n, context_s, hop_s):
    """Время центра каждого окна. Нужно, чтобы сшивать оценку с разметкой."""
    return context_s / 2.0 + np.arange(n) * hop_s


def check_scorer(s, n_samples=SR * 4):
    """Проверяет, что скорер держит собственный контракт по длине ряда.

    Заведено потому, что рассинхронизация шага и длины ряда молча сдвигает
    всю разметку по времени, и обнаруживается это уже в метрике, где выглядит
    как «модель хуже», а не как баг.
    """
    audio = np.zeros(n_samples, np.float32)
    out = s.score(audio)
    want = n_scores(n_samples, s.context_s, s.hop_s)
    assert out.ndim == 1, f"score вернул {out.ndim} измерений, нужен 1"
    assert len(out) == want, f"score вернул {len(out)} оценок, ожидалось {want}"
    assert out.dtype == np.float32, f"score вернул {out.dtype}, нужен float32"
    return len(out)


class LegacyScorer:
    """Нынешняя DroneNet за фасадом Scorer. Базовая линия для сравнений.

    Повторяет предобработку detect.py дословно: окно 0.5 с, шаг 0.25 с,
    пиковая нормализация. Отклоняться нельзя — иначе базовая цифра будет
    измерять не ту модель, что стоит в проекте.
    """

    hop_s, context_s = 0.25, 0.5

    def __init__(self, ckpt_path, device="cpu"):
        import torch
        from train import LogMel, DroneNet
        self._torch = torch
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = DroneNet().to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.logmel = LogMel().to(device)
        self.device = device

    def score(self, audio, bs=512):
        torch = self._torch
        ctx = int(self.context_s * SR)
        hop = int(self.hop_s * SR)
        n = n_scores(len(audio), self.context_s, self.hop_s)
        if n == 0:
            return np.zeros(0, np.float32)
        win = np.stack([audio[i * hop:i * hop + ctx] for i in range(n)])
        out = np.empty(n, np.float32)
        with torch.no_grad():
            for i in range(0, n, bs):
                x = torch.from_numpy(win[i:i + bs]).to(self.device).float()
                x = x / (x.abs().amax(1, keepdim=True) + 1e-8)
                out[i:i + bs] = self.model(self.logmel(x).unsqueeze(1)).cpu().numpy()
        return out


class DroneNet2Scorer:
    """DroneNet2 (этап 3) за фасадом Scorer.

    context_s берётся из чекпоинта (TrainCfg.target_samples / FeatureCfg.sr,
    обычно 12.0с — 8с истории + 4с окна модели), не задаётся руками: это
    контекст, для которого архитектура спроектирована (этап 2/3а), а не
    произвольный выбор бенча.
    """

    def __init__(self, ckpt_path, device="cpu", hop_s=1.0):
        import torch
        from airadar.train.checkpoint import load_checkpoint
        from airadar.config import FeatureCfg, ModelCfg
        from airadar.features.frontend import Frontend
        from airadar.models.dronenet2 import DroneNet2

        self._torch = torch
        ck = load_checkpoint(ckpt_path, device=device)
        feature_cfg = FeatureCfg(**ck["feature_cfg"])
        model_cfg = ModelCfg(**ck["model_cfg"])

        self.hop_s = hop_s
        self.context_s = ck["train_cfg"]["target_samples"] / feature_cfg.sr

        self.frontend = Frontend(feature_cfg).to(device)
        self.model = DroneNet2(model_cfg).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device
        self._sr = feature_cfg.sr

    def score(self, audio, bs=16):
        torch = self._torch
        ctx = int(round(self.context_s * self._sr))
        hop = int(round(self.hop_s * self._sr))
        n = n_scores(len(audio), self.context_s, self.hop_s, sr=self._sr)
        if n == 0:
            return np.zeros(0, np.float32)
        win = np.stack([audio[i * hop:i * hop + ctx] for i in range(n)])
        out = np.empty(n, np.float32)
        with torch.no_grad():
            for i in range(0, n, bs):
                x = torch.from_numpy(win[i:i + bs]).to(self.device).float()
                feat = self.frontend(x)
                feat = self.frontend.last_model_frames(feat)
                out[i:i + bs] = self.model(feat)["clip_logit"].cpu().numpy()
        return out


def selfcheck():
    # длина ряда оценок: первое окно занимает context, дальше шаг hop
    assert n_scores(8000, 0.5, 0.25) == 1          # ровно одно окно
    assert n_scores(12000, 0.5, 0.25) == 2         # 0.75 с -> окна в 0, 0.25
    assert n_scores(4000, 0.5, 0.25) == 0          # короче контекста
    assert n_scores(64000, 4.0, 0.128) == 1        # ровно 4 с

    # центры окон: первое окно [0, 0.5) -> центр 0.25
    t = frame_times(3, 0.5, 0.25)
    assert np.allclose(t, [0.25, 0.5, 0.75]), t

    # адаптер должен считать на синтетике без чекпоинта: подменяем модель
    class Fake:
        hop_s, context_s = 0.25, 0.5

        def score(self, audio):
            n = n_scores(len(audio), self.context_s, self.hop_s)
            return np.zeros(n, np.float32)

    s = Fake()
    assert check_scorer(s, n_samples=12000) == 2

    # рассинхронизация шага и контекста должна ловиться, а не молча врать
    class Broken(Fake):
        def score(self, audio):
            return np.zeros(99, np.float32)

    try:
        check_scorer(Broken(), n_samples=12000)
    except AssertionError:
        pass
    else:
        raise AssertionError("check_scorer не поймал неверную длину ряда")

    # DroneNet2Scorer: синтетический чекпоинт (случайные веса, как в
    # airadar/train/checkpoint.py:selfcheck), проверка контракта Scorer
    import tempfile
    from airadar.train.checkpoint import save_checkpoint
    from airadar.config import FeatureCfg, ModelCfg, AugCfg, TrainCfg
    from airadar.models.dronenet2 import DroneNet2

    with tempfile.TemporaryDirectory() as d:
        manifest_path = os.path.join(d, "fake_manifest.bin")
        with open(manifest_path, "wb") as f:
            f.write(b"fake")
        ckpt_path = os.path.join(d, "dn2.pt")
        model = DroneNet2(ModelCfg())
        save_checkpoint(ckpt_path, model, opt=None, sched=None, epoch=0,
                        feature_cfg=FeatureCfg(), model_cfg=ModelCfg(),
                        aug_cfg=AugCfg(), train_cfg=TrainCfg(),
                        manifest_path=manifest_path)

        s2 = DroneNet2Scorer(ckpt_path, device="cpu")
        assert s2.context_s == TrainCfg().target_samples / FeatureCfg().sr
        assert s2.hop_s == 1.0
        # 15с > context_s (12с) -> хотя бы одна оценка реально считается
        assert check_scorer(s2, n_samples=SR * 15) >= 1

        s2_custom_hop = DroneNet2Scorer(ckpt_path, device="cpu", hop_s=2.0)
        assert s2_custom_hop.hop_s == 2.0
        assert check_scorer(s2_custom_hop, n_samples=SR * 15) >= 1

    print("scorer selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
