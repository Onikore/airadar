"""
Сравнение чекпоинтов на полевых записях при одинаковой рабочей точке.

Чекпоинты обучены с разным n_fft, поэтому каждый грузится со своими
константами препроцессинга — иначе спектрограмма не совпадёт с той, на которой
модель училась, и сравнение будет бессмысленным.

Рабочая точка задаётся по трудным негативам (бензопила, авиация, ветер), а не
по val DADS: там негативы — лай и звонки, и порог получается неоправданно мягким.

    python3 compare_models.py
    python3 compare_models.py --models models/dronenet.pt:2048 models/dronenet_fft512.pt:512
"""

import os
import sys
import glob
import wave
import argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
FARS = [0.05, 0.01, 0.001]


def load_module(n_fft):
    """train.py с подменённым n_fft — чтобы поднять модель её же препроцессингом."""
    src = open(os.path.join(ROOT, "train.py")).read()
    old = "N_FFT, HOP, N_MELS = 2048, 128, 64"
    if old not in src:
        sys.exit("не нашёл строку с N_FFT в train.py — поправь load_module")
    src = src.replace(old, f"N_FFT, HOP, N_MELS = {n_fft}, 128, 64")
    mod = type(sys)(f"train{n_fft}")
    mod.__file__ = os.path.join(ROOT, "train.py")
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def field_windows(win=8000):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "field", "drone_video*.wav"))):
        w = wave.open(p)
        raw = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        out[os.path.basename(p)] = np.stack(
            [raw[i:i + win] for i in range(0, len(raw) - win, win // 2)])
    return out


def evaluate(path, n_fft, fields):
    t = load_module(n_fft)
    X, cat, _, hva = t.load_hard()
    Xh = X[hva][np.isin(cat[hva], list(t.HARD_CATS))]

    model = t.DroneNet().to(t.DEV)
    ck = torch.load(path, map_location=t.DEV, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    lm = t.LogMel().to(t.DEV)

    ph = t._probs(model, lm, Xh)
    res = {}
    for name, W in fields.items():
        pf = t._probs(model, lm, W)
        res[name] = {far: float((pf > np.quantile(ph, 1 - far)).mean()) for far in FARS}
        res[name]["median_p"] = float(np.median(pf))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["models/dronenet_fft512.pt:512", "models/dronenet.pt:2048"],
                    help="путь:n_fft")
    args = ap.parse_args()

    fields = field_windows()
    if not fields:
        sys.exit("нет записей field/drone_video*.wav")
    print("полевые записи: " + ", ".join(f"{k} ({len(v)} окон)" for k, v in fields.items()))

    table = {}
    for spec in args.models:
        path, _, n_fft = spec.rpartition(":")
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            print(f"пропуск, нет файла: {path}")
            continue
        print(f"считаю {path} (n_fft={n_fft})...", flush=True)
        table[f"{os.path.basename(path)} n_fft={n_fft}"] = evaluate(full, int(n_fft), fields)

    for name in fields:
        print(f"\n=== {name} — recall при рабочей точке по трудным негативам ===")
        head = f"{'модель':38s}" + "".join(f"FAR {f*100:g}%".rjust(10) for f in FARS) + "   медиана p"
        print(head)
        print("-" * len(head))
        for mname, r in table.items():
            row = "".join(f"{r[name][f]*100:9.1f}%" for f in FARS)
            print(f"{mname:38s}{row}   {r[name]['median_p']:.3f}")


if __name__ == "__main__":
    main()
