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

    print("config selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else sys.exit("нечего запускать")
