"""Локальный веб-демо: слушает микрофон, каждые несколько секунд шлёт
кусок звука на /predict, показывает вердикт DroneNet2 в реальном времени.

Не часть пайплайна (нет --selfcheck, не проходит cli/selfcheck.py по
конвенции "cli/ — тонкие обёртки") — интерактивный инструмент для ручной
проверки чекпоинта ушами, а не автоматизированный тест.

    python cli/webdemo.py --model models/dronenet2_seed0_rmsfix_true_best.pt
    затем открыть http://127.0.0.1:5000 в браузере, разрешить микрофон.
"""
import os
import sys
import io
import wave
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from math import gcd
from scipy.signal import resample_poly
from flask import Flask, request, jsonify, Response

from airadar.train.checkpoint import load_checkpoint
from airadar.config import FeatureCfg, ModelCfg, TrainCfg
from airadar.features.frontend import Frontend
from airadar.models.dronenet2 import DroneNet2
from airadar.augment.mixing import normalize_rms

app = Flask(__name__)
STATE = {}


def _resample_to_16k(audio, sr):
    if sr == STATE["sr"]:
        return audio
    g = gcd(sr, STATE["sr"])
    return resample_poly(audio, STATE["sr"] // g, sr // g).astype(np.float32)


def _decode_wav(raw_bytes):
    with wave.open(io.BytesIO(raw_bytes)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sr


@app.route("/")
def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webdemo.html")
    with open(html_path, encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    audio, sr = _decode_wav(request.get_data())
    audio = _resample_to_16k(audio, sr)

    min_len = STATE["model_samples"]
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    audio = normalize_rms(audio, target_rms=STATE["target_rms"])

    x = torch.from_numpy(audio).unsqueeze(0).to(STATE["device"])
    with torch.no_grad():
        feat = STATE["frontend"](x)
        feat = STATE["frontend"].last_model_frames(feat)
        out = STATE["model"](feat)

    logit = float(out["clip_logit"].item())
    prob = float(torch.sigmoid(torch.tensor(logit)))
    f0 = float(out["f0_hat"].mean().item())
    salience = float(out["salience_hat"].mean().item())
    return jsonify({"logit": logit, "prob": prob, "f0_hat": f0, "salience": salience})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "dronenet2_seed0_rmsfix_true_best.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_checkpoint(a.model, device=device)
    feature_cfg = FeatureCfg(**ck["feature_cfg"])
    model_cfg = ModelCfg(**ck["model_cfg"])
    train_cfg = TrainCfg(**ck["train_cfg"])

    frontend = Frontend(feature_cfg).to(device)
    model = DroneNet2(model_cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # model_samples (4с) -- измеренный минимум для 32 кадров CQT (этап 2),
    # не выводится формулой из frames*hop_length (внутренний паддинг
    # nnAudio нелинеен) -- поэтому берётся из TrainCfg чекпоинта, а не
    # пересчитывается здесь
    STATE.update(
        frontend=frontend, model=model, device=device,
        sr=feature_cfg.sr, model_samples=train_cfg.model_samples,
        target_rms=ck["aug_cfg"].get("target_rms", 0.05),
    )
    print(f"модель: {a.model}")
    print(f"эпоха чекпоинта: {ck.get('epoch')}  val_loss: {ck.get('val_loss')}")
    print(f"устройство: {device}")
    print(f"минимальное окно: {STATE['model_samples']} отсчётов ({STATE['model_samples']/feature_cfg.sr:.1f} с)")
    print(f"открой http://127.0.0.1:{a.port} в браузере")
    app.run(host="127.0.0.1", port=a.port, debug=False)


if __name__ == "__main__":
    main()
