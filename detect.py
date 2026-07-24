"""
Детектор дрона в реальном времени с микрофона.

Ключевое отличие от eval: решение принимается не по одному окну, а по
последовательности. Одиночное окно 0.5 с на дальней дистанции почти
неотличимо от шума — именно поэтому попытка вытянуть дальность аугментацией
до -12 дБ дала детектор ветра. Дрон приближается секундами и даёт устойчивую
серию срабатываний, порыв ветра и хлопок двери — нет. Правило "k из m окон"
убирает их почти даром и одновременно поднимает дальность.

Звук берётся через arecord (ALSA), новых зависимостей не нужно.

    python3 detect.py                      # с микрофона по умолчанию
    python3 detect.py --device plughw:1,6  # конкретный вход
    python3 detect.py --file запись.wav    # прогнать готовый файл
"""

import os
import sys
import json
import argparse
import subprocess
import collections
import numpy as np
import torch

from train import LogMel, DroneNet, DEV, ROOT, SR

WIN = 8000              # 0.5 с — как при обучении
HOP = 4000              # 0.25 с, окна с перекрытием вдвое


def load_model(path):
    ckpt = torch.load(path, map_location=DEV, weights_only=False)
    model = DroneNet().to(DEV)
    model.load_state_dict(ckpt["model"])
    model.eval()
    cfg = ckpt.get("cfg", {})
    if cfg.get("win", WIN) != WIN or cfg.get("sr", SR) != SR:
        print(f"внимание: чекпоинт обучен на win={cfg.get('win')} sr={cfg.get('sr')}, "
              f"а здесь win={WIN} sr={SR}", file=sys.stderr)
    return model, LogMel().to(DEV)


def default_threshold():
    """Порог из eval.json (рабочая точка FAR 1%), иначе половина."""
    p = os.path.join(ROOT, "eval.json")
    try:
        with open(p) as f:
            return float(json.load(f)["all"]["operating_points"]["0.01"]["threshold"])
    except Exception:
        return 0.5


def prob(model, logmel, win):
    x = torch.from_numpy(win).to(DEV).float().unsqueeze(0) / 32768.0
    x = x / (x.abs().amax(1, keepdim=True) + 1e-8)
    with torch.no_grad():
        return float(torch.sigmoid(model(logmel(x).unsqueeze(1)))[0])


def audio_source(args):
    """Отдаёт куски по HOP отсчётов int16. Микрофон или файл."""
    if args.file:
        import wave
        w = wave.open(args.file)
        if w.getframerate() != SR or w.getnchannels() != 1:
            sys.exit(f"нужен моно {SR} Гц, а в файле {w.getnchannels()} кан. {w.getframerate()} Гц")
        while True:
            raw = w.readframes(HOP)
            if len(raw) < HOP * 2:
                return
            yield np.frombuffer(raw, np.int16)
    else:
        cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE",
               "-r", str(SR), "-c", "1", "-D", args.device]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            while True:
                raw = proc.stdout.read(HOP * 2)
                if len(raw) < HOP * 2:
                    err = proc.stderr.read().decode(errors="replace").strip()
                    sys.exit(f"arecord оборвался: {err or 'нет данных'}")
                yield np.frombuffer(raw, np.int16)
        finally:
            proc.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "dronenet.pt"))
    ap.add_argument("--device", default="default", help="устройство ALSA, см. arecord -l")
    ap.add_argument("--file", help="прогнать wav вместо микрофона (моно 16 кГц)")
    # Калибровочные ручки. Значения по умолчанию — отправная точка, на реальной
    # точке их подбирают по месту: лес шумит иначе, чем ESC-50.
    ap.add_argument("--threshold", type=float, default=None, help="порог по окну")
    ap.add_argument("--k", type=int, default=4, help="сколько окон из m должны сработать")
    ap.add_argument("--m", type=int, default=6, help="длина окна накопления")
    args = ap.parse_args()

    if args.k > args.m:
        sys.exit("--k не может быть больше --m")

    thr = args.threshold if args.threshold is not None else default_threshold()
    model, logmel = load_model(args.model)
    print(f"порог {thr:.4f}   правило {args.k} из {args.m} окон "
          f"({args.m * HOP / SR:.1f} с)   устройство {args.file or args.device}")
    print("Ctrl+C для выхода\n")

    buf = np.zeros(WIN, np.int16)
    recent = collections.deque(maxlen=args.m)
    filled = 0
    was_alarm = False

    try:
        for chunk in audio_source(args):
            buf = np.concatenate([buf[len(chunk):], chunk])
            filled = min(filled + len(chunk), WIN)
            if filled < WIN:
                continue

            p = prob(model, logmel, buf)
            recent.append(p > thr)
            hits = sum(recent)
            alarm = hits >= args.k

            rms = float(np.sqrt((buf.astype(np.float32) / 32768.0) ** 2).mean())
            state = "ДРОН    " if alarm else ("внимание" if hits else "тихо    ")
            bar = "#" * int(p * 30)
            print(f"\r{state} p={p:.3f} {hits}/{args.m}  ур.{rms:.3f} |{bar:<30}|",
                  end="", flush=True)
            if alarm and not was_alarm:
                print("\n>>> ОБНАРУЖЕН ДРОН", flush=True)
            was_alarm = alarm
    except KeyboardInterrupt:
        print("\nостановлено")


def selfcheck():
    """Правило k из m: серия срабатывает, одиночные выбросы — нет."""
    def run(seq, k, m):
        d, out = collections.deque(maxlen=m), []
        for v in seq:
            d.append(v)
            out.append(sum(d) >= k)
        return out

    # одиночные ложные срабатывания вразнобой не должны поднимать тревогу
    assert not any(run([1, 0, 0, 1, 0, 0, 1, 0, 0], k=4, m=6))
    # устойчивая серия — должна
    assert any(run([0, 1, 1, 1, 1, 1], k=4, m=6))
    # ровно k подряд достаточно, k-1 нет
    assert run([1, 1, 1, 1], k=4, m=6)[-1]
    assert not run([1, 1, 1, 0], k=4, m=6)[-1]
    # окно накопления забывает старое
    assert not run([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], k=4, m=6)[-1]
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
