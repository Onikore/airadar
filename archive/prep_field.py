"""Извлечение звука из полевых видео в field/drone_video*.wav.

Полевые записи — единственная ненасыщенная метрика проекта: на публичных
данных AUC упирается в 0.99 за пару эпох, а здесь модель либо слышит гармоники
тяжёлого дрона, либо нет (см. README, раздел про 98.9% против 0.0%).

Приходят они с телефона и с камеры, то есть в контейнере с видео и в разных
частотах дискретизации. train.load_field ждёт строго 16 кГц моно WAV, иначе
молча пропускает файл — отсюда этот шаг.

ffmpeg берётся из пакета imageio-ffmpeg (статическая сборка, без установки в
систему и без прав администратора).
"""

import os
import re
import sys
import glob
import subprocess
import wave

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FIELD = os.path.join(ROOT, "field")
SR = 16000

# Всё, во что телефон или камера может завернуть звук.
SRC_EXT = (".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4a", ".aac", ".mp3", ".flac")


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("нет ffmpeg: pip install imageio-ffmpeg")


def probe(path):
    """(секунды, частота, каналов) или None, если звуковой дорожки нет."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dur = sr = ch = None
    for line in p.stderr.splitlines():
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", line)
        if m:
            h, mi, s = m.groups()
            dur = int(h) * 3600 + int(mi) * 60 + float(s)
        m = re.search(r"Audio: .*?, (\d+) Hz, (mono|stereo|\d+ channels)", line)
        if m:
            sr = int(m.group(1))
            ch = {"mono": 1, "stereo": 2}.get(m.group(2))
            if ch is None:
                ch = int(m.group(2).split()[0])
    return None if sr is None else (dur, sr, ch)


def extract(src, dst):
    """Звук в 16 кГц моно PCM. Нормализацию НЕ делаем: load_field и _probs
    нормируют окна сами, а нормировка целой записи сместила бы уровень
    относительно того, что увидит детектор в реальном времени."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-i", src, "-vn", "-ac", "1", "-ar", str(SR),
           "-c:a", "pcm_s16le", dst]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg упал на {src}:\n{p.stderr}")
    return dst


def check(path):
    """Читает результат так же, как train.load_field, и возвращает статистику."""
    w = wave.open(path)
    assert w.getframerate() == SR, f"{path}: {w.getframerate()} Гц вместо {SR}"
    assert w.getnchannels() == 1, f"{path}: {w.getnchannels()} каналов вместо 1"
    assert w.getsampwidth() == 2, f"{path}: {w.getsampwidth()*8} бит вместо 16"
    raw = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    win = 8000
    n_win = max(0, (len(raw) - win) // (win // 2))
    peak = int(np.abs(raw).max()) if len(raw) else 0
    return dict(sec=len(raw) / SR, windows=n_win, peak=peak,
                rms=float(np.sqrt(np.mean((raw / 32768.0) ** 2))) if len(raw) else 0.0)


def sources():
    """Видео и аудио в корне проекта, кроме уже готовых field/*.wav."""
    out = []
    for f in sorted(os.listdir(ROOT)):
        if os.path.splitext(f)[1].lower() in SRC_EXT:
            out.append(os.path.join(ROOT, f))
    return out


def main():
    src = sources()
    if not src:
        print(f"в корне проекта нет файлов с расширениями {', '.join(SRC_EXT)}")
        return
    print(f"найдено источников: {len(src)}\n")

    n = 0
    for path in src:
        info = probe(path)
        name = os.path.basename(path)
        if info is None:
            print(f"  {name}: звуковой дорожки нет, пропускаю")
            continue
        dur, sr, ch = info
        n += 1
        dst = os.path.join(FIELD, f"drone_video{n}.wav")
        print(f"  {name}")
        print(f"    источник: {dur:.1f} с, {sr} Гц, каналов {ch}")
        extract(path, dst)
        st = check(dst)
        print(f"    -> field/drone_video{n}.wav: {st['sec']:.1f} с, "
              f"окон {st['windows']}, пик {st['peak']}/32767, RMS {st['rms']:.4f}")
        if st["peak"] >= 32767:
            print(f"       ВНИМАНИЕ: запись клиппирует, гармоники могут быть искажены")
        if st["rms"] < 0.005:
            print(f"       ВНИМАНИЕ: очень тихая запись, проверьте усиление")

    print(f"\nготово: {n} записей в field/")
    if n:
        print("проверить, что train.load_field их видит:")
        print("  python -c \"import train; "
              "print({k: len(v) for k, v in train.load_field().items()})\"")


def selfcheck():
    # разбор вывода ffmpeg — та часть, которая ломается при смене версии
    import tempfile

    global ffmpeg_exe
    orig = ffmpeg_exe

    class FakeRun:
        def __init__(self, stderr):
            self.stderr, self.returncode = stderr, 0

    def fake_probe(text):
        real = subprocess.run
        subprocess.run = lambda *a, **k: FakeRun(text)
        try:
            return probe("не важно")
        finally:
            subprocess.run = real

    ffmpeg_exe = lambda: "ffmpeg"
    try:
        assert fake_probe(
            "  Duration: 00:00:39.00, start: 0.000000, bitrate: 441 kb/s\n"
            "  Stream #0:1: Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 31 kb/s"
        ) == (39.0, 44100, 1)
        assert fake_probe(
            "  Duration: 00:01:48.20, start: 0.0, bitrate: 3253 kb/s\n"
            "  Stream #0:1: Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 195 kb/s"
        ) == (108.2, 48000, 2)
        assert fake_probe(
            "  Duration: 00:00:10.00, start: 0.0, bitrate: 100 kb/s\n"
            "  Stream #0:1: Audio: pcm_s16le, 16000 Hz, 4 channels, s16, 1024 kb/s"
        ) == (10.0, 16000, 4)
        # видео без звука
        assert fake_probe(
            "  Duration: 00:00:05.00, start: 0.0, bitrate: 400 kb/s\n"
            "  Stream #0:0: Video: h264 (High), yuv420p, 256x480, 30 fps"
        ) is None
    finally:
        ffmpeg_exe = orig

    # check() ловит неверный формат, а не молча пропускает, как load_field
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "good.wav")
        w = wave.open(good, "wb")
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.arange(SR * 2, dtype=np.int16) % 1000).tobytes())
        w.close()
        st = check(good)
        assert abs(st["sec"] - 2.0) < 1e-6, st
        assert st["windows"] == (SR * 2 - 8000) // 4000 == 6, st

        bad = os.path.join(d, "bad.wav")
        w = wave.open(bad, "wb")
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes(np.zeros(1000, np.int16).tobytes())
        w.close()
        try:
            check(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("стерео 44.1 кГц должно отбраковываться")

    assert ".mov" in SRC_EXT and ".mp4" in SRC_EXT
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
