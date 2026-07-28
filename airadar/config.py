"""Конфигурация признака и модели как данные, а не глобалы модуля.

Инвариант §8 спецификации: фронтенд конструируется из FeatureCfg, лежащего
в чекпоинте, а не из констант airadar.features.cqt/frontend напрямую.
Раньше (detect.py, архивный) рассинхрон train/inference признака (win/sr)
был возможен и только предупреждался, не блокировался. FeatureCfg —
единственный источник параметров фронтенда; сериализуется вместе с весами
в train/checkpoint.py.

Значения по умолчанию здесь ОБЯЗАНЫ совпадать с модульными константами
airadar/features/cqt.py и airadar/features/frontend.py, background.py —
иначе LogCQT()/Frontend() без аргументов (уже смерженный код этапа 2)
изменят поведение задним числом.
"""
import sys
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class FeatureCfg:
    sr: int = 16000
    hop_s: float = 0.128
    fmin: float = 40.0
    fmax: float = 8000.0
    bins_per_octave: int = 24
    n_bins: int = 183
    model_frames: int = 32
    bg_window_frames: int = 63
    bg_quantile: float = 0.20

    @property
    def hop_length(self):
        return round(self.hop_s * self.sr)


@dataclass(frozen=True)
class ModelCfg:
    """Только то, что реально параметризует DroneNet2 (airadar/models/
    dronenet2.py). f0-диапазон (40-400 Гц) и число гармоник (8) — не здесь:
    они зашиты в airadar/features/harmonic.py как производные сетки CQT
    (bins_per_octave), а не независимые настройки — гармонический индекс
    round(24*log2 k) осмыслен только при фиксированном bins_per_octave
    FeatureCfg, дублировать его в ModelCfg значило бы завести два источника
    истины для одного и того же числа."""
    branch_hidden: int = 128
    mil_hidden: int = 16


@dataclass(frozen=True)
class AugCfg:
    """Диапазоны аугментации (§4). pitch_prob/hum_prob/hum_only_prob —
    инженерное решение этого плана (спецификация задаёт диапазоны и
    эффекты, не частоту применения)."""
    pitch_r_lo: float = 0.35
    pitch_r_hi: float = 1.5
    pitch_prob: float = 0.7          # доля позитивов со сдвинутым f0
    snr_db_lo: float = -15.0
    snr_db_hi: float = 20.0
    gain_db_lo: float = -6.0
    gain_db_hi: float = 6.0
    hum_amp_max: float = 0.8
    hum_f0_lo: float = 49.8
    hum_f0_hi: float = 50.2
    hum_prob: float = 0.3            # доля примеров с подмешанным гулом
    hum_only_prob: float = 0.05      # доля НЕГАТИВОВ, заменяемых на чистый гул
    air_k_max: float = 2.5
    spec_mask_n: int = 2
    spec_mask_frac: float = 1.0 / 6.0


@dataclass(frozen=True)
class TrainCfg:
    """Длины окна сборки примера, отсчёты при 16 кГц. target_samples —
    8с истории (BG_WINDOW_FRAMES) + 4с окна модели (MODEL_FRAMES), см.
    airadar/features/frontend.py, этап 2. model_samples — минимум, ниже
    которого Frontend.last_model_frames не наберёт MODEL_FRAMES кадров."""
    model_samples: int = 64000    # 4.0с
    target_samples: int = 192000  # 12.0с


def selfcheck():
    cfg = FeatureCfg()
    assert cfg.hop_length == 2048, cfg.hop_length   # совпадает с cqt.HOP_LENGTH
    d = asdict(cfg)
    assert d["sr"] == 16000 and d["n_bins"] == 183

    cfg2 = FeatureCfg(sr=8000)
    assert cfg2.hop_length == round(0.128 * 8000)

    # frozen — конфиг разделяется между обучением и инференсом, случайная
    # мутация одного не должна быть возможна в принципе
    try:
        cfg.sr = 44100
    except Exception:
        pass
    else:
        raise AssertionError("FeatureCfg обязан быть frozen")

    mc = ModelCfg()
    assert mc.branch_hidden == 128 and mc.mil_hidden == 16

    ac = AugCfg()
    assert ac.pitch_r_lo == 0.35 and ac.pitch_r_hi == 1.5
    assert ac.snr_db_lo == -15.0 and ac.snr_db_hi == 20.0
    assert ac.hum_amp_max == 0.8

    tc = TrainCfg()
    assert tc.model_samples == 64000 and tc.target_samples == 192000

    print("config selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
