"""
Обучение бинарного детектора дрона (drone / no_drone) по звуку.

Ключевая идея: сам датасет — это записи с близкой дистанции в тепличных
условиях. Модель, обученная на них напрямую, развалится на 300 метрах в лесу.
Поэтому дальность закладывается через аугментацию: каждый обучающий пример
искусственно "отодвигается" — падает SNR, срезаются высокие частоты
(поглощение воздухом + листва), подмешивается гул ЛЭП 50 Гц.

Пеленгация здесь не участвует — азимут считается алгоритмически (GCC-PHAT)
по 4 каналам, эта ветка полностью отдельная.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

try:                              # прогресс по батчам — необязательная зависимость,
    from tqdm import tqdm         # без неё поведение не меняется, просто без бара
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))

SR = 16000
# n_fft=2048 даёт шаг 7.81 Гц. При 512 шаг был 31.25 Гц, и гармоники тяжёлого
# дрона (основная ~70 Гц) разделялись всего 2.2 бинами — гребёнка сливалась в
# сплошное пятно, признака в спектрограмме физически не было. На реальной записи
# грузового дрона это давало recall 24% против 98.9% на DADS, где почти все
# записи — малые квадрокоптеры с основной ~200 Гц.
# Вход сети не меняется: при hop=128 кадров по-прежнему 63, mel-полос 64.
N_FFT, HOP, N_MELS = 2048, 128, 64
FMIN, FMAX = 20, 8000
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- features

def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=FMIN, fmax=FMAX):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], max(bins[i + 1], bins[i] + 1), max(bins[i + 2], bins[i + 1] + 2)
        r = min(r, fb.shape[1])
        c = min(c, r - 1)
        fb[i, l:c] = np.linspace(0, 1, c - l, endpoint=False)
        fb[i, c:r] = np.linspace(1, 0, r - c, endpoint=False)
    return torch.from_numpy(fb)


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fb", mel_filterbank())
        self.register_buffer("window", torch.hann_window(N_FFT))
        # частоты бинов — нужны для частотно-зависимого затухания (дальность)
        self.register_buffer("freqs", torch.linspace(0, SR / 2, N_FFT // 2 + 1))

    def forward(self, wav, air_absorb=None):
        spec = torch.stft(wav, N_FFT, HOP, window=self.window,
                          return_complex=True, center=True, pad_mode="reflect")
        power = spec.real ** 2 + spec.imag ** 2               # [B, F, T]
        if air_absorb is not None:
            # exp(-k * f * d): чем дальше цель, тем сильнее срезаны верхи
            att = torch.exp(-air_absorb[:, None] * self.freqs[None, :] / 1000.0)
            power = power * (att ** 2)[:, :, None]
        mel = self.fb @ power
        return torch.log(mel + 1e-8)


# ------------------------------------------------------------ augmentation

class Augment(nn.Module):
    """Аугментация вида 'отодвинь цель дальше'. Работает на батче на GPU."""

    def __init__(self, noise_pool):
        super().__init__()
        self.register_buffer("noise_pool", noise_pool)   # int16 окна no_drone

    def forward(self, wav, y):
        B, L = wav.shape
        dev = wav.device

        # 1. случайный циклический сдвиг — модель не должна цепляться за позицию
        shift = torch.randint(0, L, (1,)).item()
        wav = torch.roll(wav, shift, dims=1)

        # 2. подмешивание реального фона при случайном SNR (главный фактор дальности)
        idx = torch.randint(0, self.noise_pool.shape[0], (B,), device=dev)
        noise = self.noise_pool[idx].float() / 32768.0
        noise = torch.roll(noise, torch.randint(0, L, (1,)).item(), dims=1)
        # Нижняя граница SNR намеренно не ниже -6 дБ. При -12 дБ в окне 0.5 с
        # дрона физически не слышно: пример состоит почти целиком из шума, а
        # метка говорит "дрон". Сеть выучивала из этого, что широкополосный шум
        # и есть признак дрона, и путалась на ветре в 20 раз чаще тривиального
        # baseline. Дальность берётся накоплением по нескольким окнам подряд,
        # а не выкручиванием SNR в область, где информации не осталось.
        snr_db = torch.empty(B, device=dev).uniform_(-6.0, 15.0)
        sig_p = wav.pow(2).mean(1, keepdim=True) + 1e-10
        noi_p = noise.pow(2).mean(1, keepdim=True) + 1e-10
        scale = torch.sqrt(sig_p / (noi_p * 10 ** (snr_db[:, None] / 10.0)))
        wav = wav + noise * scale

        # 3. гул ЛЭП: 50 Гц + гармоники, случайная фаза и уровень.
        #
        # Амплитуда снижена с 0.8 до 0.25. Гармоники сети (50/100/150/200 Гц)
        # попадают ровно туда, где живёт основная частота тяжёлого дрона:
        # у наших полевых записей это 67-80 и 98-103 Гц. При амплитуде 0.8
        # модель двенадцать эпох училась, что энергия в этой полосе — помеха,
        # и пропускала грузовой дрон целиком (recall 0.0% на drone_video2).
        #
        # Совсем убирать гул нельзя — рядом с ЛЭП он будет. Различить их можно:
        # сетевая наводка это чистый стационарный тон ровно на 100.00 Гц, а дрон
        # даёт кластер шириной несколько герц из-за плавающих оборотов. При
        # n_fft=2048 (шаг 7.81 Гц) разница уже видна, если её не заглушать.
        t = torch.arange(L, device=dev, dtype=torch.float32) / SR
        hum = torch.zeros(B, L, device=dev)
        for k, w in ((1, 1.0), (2, 0.5), (3, 0.35), (4, 0.2)):
            ph = torch.rand(B, 1, device=dev) * 2 * np.pi
            hum += w * torch.sin(2 * np.pi * 50 * k * t[None, :] + ph)
        hum_amp = torch.empty(B, 1, device=dev).uniform_(0.0, 0.25)
        wav = wav + hum * hum_amp * wav.abs().amax(1, keepdim=True)

        # 4. нормализация — громкость не должна быть признаком класса
        wav = wav / (wav.abs().amax(1, keepdim=True) + 1e-8)

        # 5. коэффициент поглощения верхов (применяется уже в спектре)
        air = torch.empty(B, device=dev).uniform_(0.0, 2.5)
        return wav, air


def spec_augment(x):
    """Маскирование полос частот и времени. x: [B, 1, M, T]"""
    B, _, M, T = x.shape
    for _ in range(2):
        f = torch.randint(0, M // 6, (1,)).item()
        f0 = torch.randint(0, max(M - f, 1), (1,)).item()
        x[:, :, f0:f0 + f, :] = 0
        t = torch.randint(0, T // 6, (1,)).item()
        t0 = torch.randint(0, max(T - t, 1), (1,)).item()
        x[:, :, :, t0:t0 + t] = 0
    return x


# ------------------------------------------------------------------- model

def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class DroneNet(nn.Module):
    """Компактная CNN. Держим маленькой — потом крутить на слабом железе в поле."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        x = self.body(x)
        x = x.mean(dim=(2, 3))          # global average pool
        return self.head(x).squeeze(1)


# ---------------------------------------------------------------- training

def load_field(pattern="field/drone_video*.wav"):
    """Полевые записи с известной разметкой: грузовой дрон слышен всю дорогу.

    Единственная ненасыщенная метрика в проекте. На публичных данных всё
    упирается в 0.99 за пару эпох, здесь же recall честный — модель либо
    слышит гармоники тяжёлого дрона, либо нет.

    Возвращает {имя: окна}, чтобы записи не усреднялись: у них разная основная
    частота (67-80 Гц против 98-103 Гц) и общий recall скрыл бы, что одна из
    них пропускается целиком.
    """
    import glob
    import wave
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        w = wave.open(p)
        if w.getframerate() != SR or w.getnchannels() != 1:
            print(f"{os.path.basename(p)}: нужен моно {SR} Гц, пропускаю")
            continue
        raw = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        win = 8000
        if len(raw) <= win:
            continue
        out[os.path.basename(p)] = np.stack(
            [raw[i:i + win] for i in range(0, len(raw) - win, win // 2)])
    return out or None


HARD_CATS = {
    "chainsaw", "helicopter", "airplane", "engine", "train", "vacuum_cleaner",
    "washing_machine", "hand_saw", "wind", "rain", "thunderstorm", "crackling_fire",
    "air_conditioner", "drilling", "engine_idling", "jackhammer",
}


def load_hard(train_frac=0.5):
    """Трудные негативы (бензопила, ветер, двигатель) с категориями.

    Возвращает (X, cat, train_idx, val_idx) — форма прежняя, её ждут eval.py,
    compare_models.py и evalx/field_ci.py.

    Изменилось происхождение индексов. Раньше делили пополам по порядку окон
    внутри категории; срез примерно совпадал с границей между записями, но
    именно что примерно — окна одной записи могли попасть по разные стороны, и
    тогда таблица ложных срабатываний мерила ложные на обучающих данных. Теперь
    берём замороженный сплит по группам из meta.npz (см. prep_hf.py).

    Аргумент train_frac сохранён для совместимости и игнорируется, когда в meta
    есть split.
    """
    d = os.path.join(ROOT, "cache_hard")
    if not os.path.exists(os.path.join(d, "meta.npz")):
        return None, None, None, None
    m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    X = np.memmap(os.path.join(d, "windows.bin"), dtype=np.int16,
                  mode="r", shape=(int(m["n"]), int(m["win"])))
    cat = m["cat"]
    if "split" in m.files:
        s = m["split"]
        return X, cat, np.flatnonzero(s == 0), np.flatnonzero(s == 1)
    tr, va = [], []
    for c in np.unique(cat):
        idx = np.flatnonzero(cat == c)
        cut = int(len(idx) * train_frac)
        tr.append(idx[:cut]); va.append(idx[cut:])
    return X, cat, np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def load_cache(name="cache_dads"):
    """Кэш не грузится в RAM целиком — на DADS это ~5 ГБ. Работаем через
    memmap, страницы подтягивает ОС. Индексы батча держим отсортированными,
    иначе случайный доступ по файлу упрётся в диск."""
    d = os.path.join(ROOT, name)
    m = np.load(os.path.join(d, "meta.npz"))
    if os.path.exists(os.path.join(d, "windows.bin")):
        X = np.memmap(os.path.join(d, "windows.bin"), dtype=np.int16,
                      mode="r", shape=(int(m["n"]), int(m["win"])))
    else:
        X = np.load(os.path.join(d, "windows.npy"), mmap_mode="r")
    synth = m["synth"] if "synth" in m.files else np.zeros(len(X), bool)
    return X, m["y"].astype(np.float32), m["group"], synth


def load_split(name="cache_dads"):
    """Сплит берётся из meta.npz, куда его заморозил prep_hf.py.

    0 = train, 1 = val, 2 = test. Пересчитывать его при каждом обучении, как
    было раньше, нельзя: при четырёх источниках любое изменение порядка данных
    молча переставит границу и сделает два прогона несравнимыми. Ровно на это
    жалуется NEXT_STEPS.md, шаг 0.

    Старые кэши от prep_dads.py поля split не имеют — для них остаётся прежнее
    деление с тем же random_state, чтобы ранее полученные числа воспроизводились.
    """
    d = name if os.path.isdir(name) else os.path.join(ROOT, name)
    m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    if "split" in m.files:
        return m["split"].astype(np.int8)
    n = int(m["n"])
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
                 .split(np.arange(n), m["y"], m["group"]))
    s = np.zeros(n, np.int8)
    s[va] = 1
    return s


def load_hard_test():
    """Удержанная часть трудных негативов — (X, cat).

    Не вызывать до финальной проверки: всё, что попадает в отбор чекпоинта,
    перестаёт быть честной метрикой. Ради этого она и отделена от load_hard.
    """
    d = os.path.join(ROOT, "cache_hard")
    if not os.path.exists(os.path.join(d, "meta.npz")):
        return None, None
    m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    X = np.memmap(os.path.join(d, "windows.bin"), dtype=np.int16,
                  mode="r", shape=(int(m["n"]), int(m["win"])))
    te = np.flatnonzero(load_split("cache_hard") == 2)
    return X[te], m["cat"][te]


def _probs(model, logmel, X, bs=512):
    """Вероятности без аугментации. Нормализация та же, что при обучении."""
    was_training = model.training
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            wav = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(DEV).float() / 32768.0
            wav = wav / (wav.abs().amax(1, keepdim=True) + 1e-8)
            probs.append(torch.sigmoid(model(logmel(wav).unsqueeze(1))).cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(probs)


def evaluate(model, logmel, X, y, synth, bs=512):
    p = _probs(model, logmel, X, bs)
    out = {"auc": roc_auc_score(y, p), "acc": ((p > 0.5) == (y > 0.5)).mean()}
    # отдельно — только реальные записи дрона против всего фона.
    # синтетика (Griffin-Lim) легко узнаётся по артефактам и завышает метрику.
    real = (y == 0) | (~synth)
    if real.sum() > 0 and len(np.unique(y[real])) == 2:
        out["auc_real"] = roc_auc_score(y[real], p[real])
    return out


def _metrics_row(ep, loss, m):
    """Одна строка jsonl на эпоху. Текстовый лог читает человек, jsonl —
    инструменты: прогон идёт в Colab, и разбирать его приходится по файлу,
    вытянутому с HF, а не по экрану, которого уже нет."""
    row = {"ep": int(ep), "loss": float(loss)}
    for k in ("auc", "auc_real", "auc_hard", "far_hard@r90"):
        if k in m:
            row[k] = float(m[k])
    if "rec_field" in m:
        row["rec_field"] = {str(k): float(v) for k, v in m["rec_field"].items()}
    return row


def main(epochs=30, bs=256, lr=3e-4, use_synth=False, model_cls=None,
         out_name="dronenet.pt", run=None, resume=False):
    X, y, groups, synth = load_cache()
    sp = load_split()
    tr, va = np.flatnonzero(sp == 0), np.flatnonzero(sp == 1)
    if (sp == 2).any():
        print(f"удержано в test и не используется: {(sp == 2).sum()} окон")

    # Синтетика (Griffin-Lim) даёт AUC=1.0000 — модель ловит артефакты генерации,
    # а не дрон, и на реальных записях сыпется. Выкидываем из обучения.
    # Валидация не трогается, иначе метрики станут несравнимыми.
    if not use_synth:
        n0 = len(tr)
        tr = tr[~synth[tr]]
        print(f"выброшено синтетических окон из train: {n0 - len(tr)}")

    print(f"train окон: {len(tr)}   val окон: {len(va)}")
    print(f"val: drone={int(y[va].sum())} no_drone={int((y[va]==0).sum())}")

    va = np.sort(va)
    Xva, yva, sva = X[va], y[va], synth[va]      # X — memmap, читается лениво

    # Трудные негативы: механический гул и погода. Негативы DADS — это лай,
    # плач и дверные звонки, на них сеть не учится молчать при ветре.
    Xf = load_field()
    if Xf is not None:
        for nm, w in Xf.items():
            print(f"полевая запись {nm}: {len(w)} окон (дрон слышен всю запись)")
    Xh, cat_h, htr, hva = load_hard()
    if Xh is not None:
        Hbuf = torch.from_numpy(np.ascontiguousarray(Xh[htr])).to(DEV)
        print(f"трудных негативов в train: {len(htr)}  (в eval отложено: {len(hva)})")
    else:
        Hbuf = None
        print("cache_hard нет — обучение без трудных негативов")

    # пул фона для аугментации берём только из train — иначе утечка в val.
    # Половина пула из трудных, чтобы "дрон на фоне ветра" встречался в обучении.
    noise_idx = np.sort(tr[y[tr] == 0])
    noise_idx = noise_idx[np.linspace(0, len(noise_idx) - 1, min(6000, len(noise_idx))).astype(int)]
    noise_pool = torch.from_numpy(np.ascontiguousarray(X[noise_idx])).to(DEV)
    if Hbuf is not None:
        sel = torch.linspace(0, len(Hbuf) - 1, min(6000, len(Hbuf))).long()
        noise_pool = torch.cat([noise_pool, Hbuf[sel]])

    logmel = LogMel().to(DEV)
    aug = Augment(noise_pool).to(DEV)
    model = (model_cls or DroneNet)().to(DEV)
    print(f"архитектура: {type(model).__name__}  параметров: {sum(p.numel() for p in model.parameters())/1e3:.0f}k")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    pos_w = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)], device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    print(f"pos_weight: {pos_w.item():.2f}")

    tr = np.sort(tr)
    ytr = y[tr]
    best = 0.0

    # Прогон живёт на HF: бесплатный Colab рвёт сессию и теряет диск, а лог
    # обучения нужно читать снаружи. Имя прогона задаёт каталог runs/<run>/.
    run = run or os.path.splitext(out_name)[0]
    remote = f"runs/{run}"
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    log_p = os.path.join(ROOT, "logs", f"{run}.log")
    jsonl_p = os.path.join(ROOT, "logs", f"{run}.jsonl")
    last_p = os.path.join(ROOT, "models", f"{run}_last.pt")
    start_ep = 0

    if resume:
        import hub
        if hub.exists(f"{remote}/last.pt"):
            ck = torch.load(hub.pull(f"{remote}/last.pt", last_p),
                            map_location=DEV, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            best = ck.get("best", 0.0)
            start_ep = int(ck["ep"])
            torch.set_rng_state(ck["rng"].cpu().to(torch.uint8))
            np.random.set_state(tuple(ck["np_rng"]))

            # Горизонт эпох берём из чекпоинта, а не из аргумента. OneCycleLR
            # держит total_steps внутри state_dict, то есть расписание скорости
            # обучения определено на фиксированном числе шагов. Продолжить
            # прогон на 4 эпохи как прогон на 12 нельзя: либо расписание
            # рассыпается (шаг 37 из 36), либо LR идёт не по той кривой, и
            # результат уже не воспроизводится. Чтобы учить дольше — новый run.
            ck_epochs = int(ck.get("epochs", epochs))
            if ck_epochs != epochs:
                print(f"ВНИМАНИЕ: прогон {run} начат на {ck_epochs} эпох, "
                      f"запрошено {epochs}. Продолжаю до {ck_epochs} — расписание "
                      f"OneCycleLR задано на этом горизонте. Для другого числа "
                      f"эпох запустите новый run.")
                epochs = ck_epochs

            # Лог и метрики тоже с HF: иначе локальный файл начнётся заново
            # пустым и следующая выгрузка затрёт на HF всю прошлую историю.
            for rp, lp in ((f"{remote}/train.log", log_p),
                           (f"{remote}/metrics.jsonl", jsonl_p)):
                if hub.exists(rp):
                    hub.pull(rp, lp)
            print(f"продолжаем с эпохи {start_ep + 1} из {epochs}, "
                  f"лучшая метрика {best:.4f}")
        else:
            print(f"на HF нет {remote}/last.pt — начинаем с нуля")
    else:
        # Не resume — прогон с чистого листа, старый лог не дописываем.
        for p in (log_p, jsonl_p):
            if os.path.exists(p):
                os.remove(p)

    for ep in range(start_ep, epochs):
        perm = np.random.permutation(len(tr))
        tot, n = 0.0, 0
        starts = range(0, len(perm) - bs + 1, bs)
        # tqdm необязателен: если пакета нет, поведение не меняется, просто
        # без бара. При выводе в файл (tee) \r считается .NET-ридером за конец
        # строки, поэтому Get-Content -Wait в PowerShell покажет прокручивающийся
        # список обновлений, а не одну перезаписываемую строку — это нормально.
        bar = tqdm(starts, desc=f"эпоха {ep+1}/{epochs}", unit="батч",
                  leave=False) if _HAS_TQDM else starts
        for i in bar:
            b = np.sort(perm[i:i + bs])          # сортировка = локальность в memmap
            wav = torch.from_numpy(np.ascontiguousarray(X[tr[b]])).to(DEV).float() / 32768.0
            lab = torch.from_numpy(ytr[b]).to(DEV)
            if Hbuf is not None:                 # добавка трудных негативов в батч
                j = torch.randint(0, len(Hbuf), (bs // 4,), device=DEV)
                wav = torch.cat([wav, Hbuf[j].float() / 32768.0])
                lab = torch.cat([lab, torch.zeros(bs // 4, device=DEV)])
            wav, air = aug(wav, lab)
            spec = spec_augment(logmel(wav, air_absorb=air).unsqueeze(1))
            loss = lossf(model(spec), lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); n += 1
            if _HAS_TQDM:
                bar.set_postfix(loss=f"{tot/n:.4f}")
        m = evaluate(model, logmel, Xva, yva, sva)
        # Отбираем чекпоинт по AUC "дрон против механического шума и погоды",
        # а не по AUC на val DADS: там негативы — лай и звонки, метрика упирается
        # в потолок 0.99 и перестаёт различать модели.
        if Xh is not None:
            # Только механический шум и погода. Если сюда попадут лай и звонки,
            # метрика разбавится лёгкими негативами и перестанет ловить провал
            # на ветре — ровно то, ради чего она заведена.
            ph = _probs(model, logmel, Xh[hva[np.isin(cat_h[hva], list(HARD_CATS))]])
            pd = _probs(model, logmel, Xva[yva == 1])
            m["auc_hard"] = roc_auc_score(
                np.r_[np.ones(len(pd)), np.zeros(len(ph))], np.r_[pd, ph])
            m["far_hard@r90"] = float((ph > np.quantile(pd, 0.10)).mean())
            if Xf is not None:
                # Порог по рабочей точке FAR 1% на трудных негативах.
                # Отбор чекпоинта по этим числам НЕ ведём: две записи, один тип
                # дрона — подгонимся под конкретные пролёты.
                t99 = np.quantile(ph, 0.99)
                m["rec_field"] = {nm: float((_probs(model, logmel, w) > t99).mean())
                                  for nm, w in Xf.items()}
        key = m.get("auc_hard", m["auc"])
        tag = ""
        if key > best:
            best = key
            torch.save({"model": model.state_dict(),
                        "cfg": dict(sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS,
                                    fmin=FMIN, fmax=FMAX, win=X.shape[1],
                                    arch=type(model).__name__)},
                       os.path.join(ROOT, "models", out_name))
            tag = "  <- saved"
        line = (f"ep{ep+1:02d} loss {tot/n:.4f}  auc {m['auc']:.4f}  "
                f"auc_hard {m.get('auc_hard', float('nan')):.4f}  "
                f"FAR_hard@r90 {m.get('far_hard@r90', float('nan'))*100:.1f}%  "
                f"recall_поле " + " ".join(
                    f"{v*100:.1f}%" for v in m.get("rec_field", {}).values()) + tag)
        print(line, flush=True)

        # Выгрузка после КАЖДОЙ эпохи, а не в конце: потерять восемь эпох
        # из-за обрыва на девятой недопустимо. Сбой выгрузки не должен
        # останавливать обучение — сеть отвалилась, а GPU-время идёт.
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        with open(jsonl_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(_metrics_row(ep + 1, tot / n, m),
                               ensure_ascii=False) + "\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "ep": ep + 1, "best": best,
                    "epochs": epochs, "rng": torch.get_rng_state(),
                    "np_rng": np.random.get_state()}, last_p)
        try:
            import hub
            hub.push(log_p, f"{remote}/train.log")
            hub.push(jsonl_p, f"{remote}/metrics.jsonl")
            hub.push(last_p, f"{remote}/last.pt")
            if tag:
                hub.push(os.path.join(ROOT, "models", out_name), f"{remote}/best.pt")
        except Exception as e:
            print(f"  выгрузка на HF не удалась ({type(e).__name__}: {e}); "
                  f"обучение продолжается", flush=True)


def selfcheck():
    """Проверяет форму фич и то, что аугментация не ломает сигнал."""
    lm = LogMel().to(DEV)
    wav = torch.randn(4, 8000, device=DEV)
    s = lm(wav)
    assert s.shape == (4, N_MELS, 8000 // HOP + 1), s.shape
    assert torch.isfinite(s).all()
    # затухание верхов реально режет высокие частоты
    air = torch.tensor([0.0, 2.5, 2.5, 2.5], device=DEV)
    s2 = lm(wav, air_absorb=air)
    assert s2[1, -1].mean() < s[1, -1].mean() - 1.0
    aug = Augment(torch.randint(-3000, 3000, (64, 8000), dtype=torch.int16).to(DEV)).to(DEV)
    w2, a = aug(wav, torch.ones(4, device=DEV))
    assert torch.isfinite(w2).all() and w2.abs().max() <= 1.0 + 1e-5
    m = DroneNet().to(DEV)
    assert m(s.unsqueeze(1)).shape == (4,)

    # --- замороженный сплит читается из meta, а не пересчитывается.
    # Настоящего кэша на машине без данных нет, поэтому кладём синтетический.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        n = 400
        want = (np.arange(n) % 3).astype(np.int8)
        np.savez(os.path.join(d, "meta.npz"),
                 y=(np.arange(n) % 2).astype(np.int8),
                 group=np.arange(n) // 4, src=np.zeros(n, np.int8),
                 split=want, synth=np.zeros(n, bool),
                 n=np.int64(n), win=np.int64(8000))
        got = load_split(d)
        assert got.dtype == np.int8 and (got == want).all(), \
            "сплит должен читаться из meta без изменений"

        # откат на GroupShuffleSplit, если поля split нет (кэши от prep_dads.py)
        mm = dict(np.load(os.path.join(d, "meta.npz"), allow_pickle=True))
        del mm["split"]
        np.savez(os.path.join(d, "meta.npz"), **mm)
        old = load_split(d)
        assert set(np.unique(old).tolist()) <= {0, 1}, "откат даёт только train и val"
        assert 0.10 < (old == 1).mean() < 0.20, (old == 1).mean()

    # --- строка метрик: ровно одна строка jsonl на эпоху, кириллица не ломается
    row = _metrics_row(3, 0.42, {"auc": 0.99, "auc_hard": 0.87,
                                 "far_hard@r90": 0.031,
                                 "rec_field": {"drone_video1.wav": 0.241}})
    dumped = json.dumps(row, ensure_ascii=False)
    assert "\n" not in dumped, "строка jsonl не должна содержать переводов строки"
    back = json.loads(dumped)
    assert back["ep"] == 3 and back["loss"] == 0.42
    assert back["auc_hard"] == 0.87 and back["rec_field"]["drone_video1.wav"] == 0.241
    # отсутствующие метрики не подставляются нулями и не роняют сериализацию
    bare = _metrics_row(1, 1.0, {"auc": 0.5})
    assert set(bare) == {"ep", "loss", "auc"}, bare

    # --- круг сохранения и восстановления состояния прогона.
    # Именно этим живёт resume: после обрыва сессии Colab всё, кроме этого
    # словаря, теряется вместе с диском.
    with tempfile.TemporaryDirectory() as d:
        net = DroneNet().to(DEV)
        o = torch.optim.AdamW(net.parameters(), lr=1e-3)
        sc = torch.optim.lr_scheduler.OneCycleLR(o, 1e-3, total_steps=10)
        import warnings
        with warnings.catch_warnings():          # шаг без backward — это тест, не обучение
            warnings.simplefilter("ignore")
            sc.step()
        p = os.path.join(d, "last.pt")
        torch.save({"model": net.state_dict(), "opt": o.state_dict(),
                    "sched": sc.state_dict(), "ep": 4, "best": 0.87,
                    "rng": torch.get_rng_state(),
                    "np_rng": np.random.get_state()}, p)
        ck = torch.load(p, map_location=DEV, weights_only=False)
        net2 = DroneNet().to(DEV)
        o2 = torch.optim.AdamW(net2.parameters(), lr=1e-3)
        sc2 = torch.optim.lr_scheduler.OneCycleLR(o2, 1e-3, total_steps=10)
        net2.load_state_dict(ck["model"])
        o2.load_state_dict(ck["opt"])
        sc2.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng"].cpu().to(torch.uint8))
        np.random.set_state(tuple(ck["np_rng"]))
        assert int(ck["ep"]) == 4 and ck["best"] == 0.87
        assert sc2.last_epoch == sc.last_epoch, "планировщик должен продолжиться, а не начаться"
        for a, b in zip(net.state_dict().values(), net2.state_dict().values()):
            assert torch.equal(a, b), "веса не совпали после восстановления"

    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
