"""
Сборка негативов с категориями из старого Kaggle-набора.

В DADS имена обрезаны до no-drone-1.wav, категория потеряна, поэтому нельзя
спросить "на чём именно детектор ложно срабатывает". В старом наборе имена
целы: UrbanSound8K ({fsID}-{classID}-{occ}-{slice}.wav) и ESC-50, порезанный
на 5 сегментов ({fold}-{clip}-{take}-{target}{seg}.wav).

Нужно это затем, что diag_leak показал: 5 тупых признаков дают AUC 0.9844.
Негативы DADS (лай, плач, звонки) отличаются от дрона громкостью и спектральным
центроидом, а не гармониками. В лесу рядом с ЛЭП такой детектор будет орать на
бензопилу, генератор и ветер. Здесь собираются именно те негативы, на которых
он должен молчать.
"""

import os
import re
import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "DroneAudioDataset", "no_drone")
OUT = os.path.join(ROOT, "cache_hard")

SR, WIN, CAP = 16000, 8000, 4

ESC50 = {
    0: "dog", 1: "rooster", 2: "pig", 3: "cow", 4: "frog", 5: "cat", 6: "hen",
    7: "insects", 8: "sheep", 9: "crow", 10: "rain", 11: "sea_waves",
    12: "crackling_fire", 13: "crickets", 14: "chirping_birds", 15: "water_drops",
    16: "wind", 17: "pouring_water", 18: "toilet_flush", 19: "thunderstorm",
    20: "crying_baby", 21: "sneezing", 22: "clapping", 23: "breathing",
    24: "coughing", 25: "footsteps", 26: "laughing", 27: "brushing_teeth",
    28: "snoring", 29: "drinking", 30: "door_knock", 31: "mouse_click",
    32: "keyboard", 33: "door_creaks", 34: "can_opening", 35: "washing_machine",
    36: "vacuum_cleaner", 37: "clock_alarm", 38: "clock_tick", 39: "glass_breaking",
    40: "helicopter", 41: "chainsaw", 42: "siren", 43: "car_horn", 44: "engine",
    45: "train", 46: "church_bells", 47: "airplane", 48: "fireworks", 49: "hand_saw",
}

URBAN = {
    0: "air_conditioner", 1: "car_horn", 2: "children_playing", 3: "dog_bark",
    4: "drilling", 5: "engine_idling", 6: "gun_shot", 7: "jackhammer",
    8: "siren", 9: "street_music",
}

# Стационарный широкополосный шум механизмов + погода. Именно это стоит рядом
# с дроном в спектре и именно это будет вокруг микрофона в лесу у ЛЭП.
HARD = {
    "chainsaw", "helicopter", "airplane", "engine", "train", "vacuum_cleaner",
    "washing_machine", "hand_saw", "wind", "rain", "thunderstorm", "crackling_fire",
    "air_conditioner", "drilling", "engine_idling", "jackhammer",
}

RE_ESC = re.compile(r"^\d+-\d+-[A-Z]-(\d+)\.wav$")
RE_URBAN = re.compile(r"^\d+-(\d+)-\d+-\d+\.wav$")


def category(name):
    m = RE_URBAN.match(name)
    if m:
        return URBAN.get(int(m.group(1)))
    m = RE_ESC.match(name)
    if m:
        # Клипы ESC-50 порезаны на 5 сегментов, номер сегмента дописан в конец
        # поля target: "362" = target 36, сегмент 2. У target 0 поле вырождается
        # в одну цифру ("3" = target 0, сегмент 3) — отсюда пустой префикс.
        f = m.group(1)
        return ESC50.get(int(f[:-1]) if f[:-1] else 0)
    return None


def load(path):
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None
    if x.size == 0:
        return None
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    peak = np.abs(x).max()
    return None if (not np.isfinite(peak) or peak < 1e-6) else x / peak


def windows(x):
    if len(x) < WIN:
        return [np.tile(x, int(np.ceil(WIN / len(x))))[:WIN]]
    return [x[i * WIN:(i + 1) * WIN] for i in range(min(len(x) // WIN, CAP))]


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sorted(os.listdir(SRC))
    cats, skipped, bad = [], 0, 0

    with open(os.path.join(OUT, "windows.bin"), "wb") as out:
        for i, name in enumerate(names):
            c = category(name)
            if c is None:
                skipped += 1
                continue
            x = load(os.path.join(SRC, name))
            if x is None:
                bad += 1
                continue
            for w in windows(x):
                out.write((w * 32767).astype(np.int16).tobytes())
                cats.append(c)
            if i % 4000 == 0:
                print(f"  {i}/{len(names)}  окон: {len(cats)}", flush=True)

    cats = np.array(cats)
    np.savez(os.path.join(OUT, "meta.npz"), cat=cats,
             n=np.int64(len(cats)), win=np.int64(WIN),
             hard=np.array(sorted(HARD)))

    print(f"\nбез категории: {skipped}   битых: {bad}")
    print(f"окон: {len(cats)}  ({len(cats)*WIN*2/1e9:.2f} GB)")
    hard_n = np.isin(cats, list(HARD)).sum()
    print(f"трудных: {hard_n}  прочих: {len(cats)-hard_n}")
    u, c = np.unique(cats[np.isin(cats, list(HARD))], return_counts=True)
    for k, v in sorted(zip(u, c), key=lambda t: -t[1]):
        print(f"    {k:20s} {v}")


def selfcheck():
    assert category("100032-3-0-0.wav") == "dog_bark"          # UrbanSound8K classID 3
    assert category("100263-2-0-117.wav") == "children_playing"
    assert category("2-122820-B-362.wav") == "vacuum_cleaner"  # target 36, сегмент 2
    assert category("5-9032-A-410.wav") == "chainsaw"          # target 41, сегмент 0
    assert category("5-9032-A-404.wav") == "helicopter"        # target 40, сегмент 4
    assert category("1-7973-A-7.wav") == "dog"                 # target 0, сегмент 7 -> префикс пуст
    assert category("BACKGROUND_001.wav") is None
    assert len(windows(np.zeros(WIN * 99))) == CAP
    assert len(windows(np.arange(50, dtype=np.float32))) == 1
    assert HARD <= (set(ESC50.values()) | set(URBAN.values()))  # опечаток в HARD нет
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
