"""
Веб-интерфейс: живое обнаружение с микрофона + прогресс обучения.

Только стандартная библиотека — http.server и поток на захват звука.
Никакого Flask: одна страница, один JSON-эндпоинт, опрос раз в 250 мс.

    python3 web.py                      # http://127.0.0.1:8000
    python3 web.py --no-audio           # только прогресс обучения
    python3 web.py --device plughw:1,6  # конкретный вход
"""

import os
import re
import sys
import json
import time
import argparse
import threading
import collections
import http.server
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
HIST = 240                       # точек графика ~ 60 с при шаге 0.25 с

state = {
    "prob": 0.0, "hits": 0, "alarm": False, "level": 0.0,
    "history": collections.deque(maxlen=HIST), "audio": "выключено",
    "k": 4, "m": 6, "threshold": 0.5,
}
lock = threading.Lock()

EP_RE = re.compile(
    r"^ep(\d+)\s+loss\s+([\d.]+)\s+auc\s+([\d.]+)\s+auc_hard\s+([\d.nan]+)\s+"
    r"FAR_hard@r90\s+([\d.nan]+)%", re.M)


def parse_training(path):
    """Разбор лога обучения. Файл может не существовать или писаться прямо сейчас."""
    try:
        with open(path) as f:
            txt = f.read()
    except OSError:
        return {"epochs": [], "total": None}
    eps = [{"ep": int(m[0]), "loss": float(m[1]), "auc": float(m[2]),
            "auc_hard": float(m[3]) if m[3] != "nan" else None,
            "far_hard": float(m[4]) if m[4] != "nan" else None}
           for m in EP_RE.findall(txt)]
    return {"epochs": eps, "total": None}


def audio_loop(args):
    """Захват и инференс в отдельном потоке. Падения не должны ронять сервер."""
    try:
        import torch
        from train import LogMel, DroneNet, DEV
        import detect
    except Exception as e:
        with lock:
            state["audio"] = f"импорт не удался: {e}"
        return

    try:
        model, logmel = detect.load_model(args.model)
        thr = args.threshold if args.threshold is not None else detect.default_threshold()
    except Exception as e:
        with lock:
            state["audio"] = f"модель не загружена: {e}"
        return

    with lock:
        state.update(threshold=thr, k=args.k, m=args.m, audio="ждёт звук")

    buf = np.zeros(detect.WIN, np.int16)
    recent = collections.deque(maxlen=args.m)
    filled = 0
    try:
        for chunk in detect.audio_source(args):
            buf = np.concatenate([buf[len(chunk):], chunk])
            filled = min(filled + len(chunk), detect.WIN)
            if filled < detect.WIN:
                continue
            p = detect.prob(model, logmel, buf)
            recent.append(p > thr)
            with lock:
                state["prob"] = p
                state["hits"] = int(sum(recent))
                state["alarm"] = sum(recent) >= args.k
                state["level"] = float(np.abs(buf / 32768.0).mean())
                state["history"].append(round(p, 4))
                state["audio"] = "работает"
    except SystemExit as e:
        with lock:
            state["audio"] = str(e)
    except Exception as e:
        with lock:
            state["audio"] = f"ошибка: {e}"


class Handler(http.server.BaseHTTPRequestHandler):
    log_file = "logs/train_hard.log"

    def _send(self, body, ctype):
        body = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/status"):
            with lock:
                s = {k: (list(v) if isinstance(v, collections.deque) else v)
                     for k, v in state.items()}
            s["training"] = parse_training(os.path.join(ROOT, self.log_file))
            self._send(json.dumps(s), "application/json")
        elif self.path == "/":
            self._send(PAGE, "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass                                    # без лога каждого опроса


PAGE = """<!doctype html><meta charset=utf-8><title>AirRadar</title>
<style>
:root{--bg:#111418;--fg:#e6e6e6;--dim:#8a8f98;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--line:#2a2f36}
@media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#1a1a1a;--dim:#666;--line:#e0e0e0}}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
     font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
h1{font-size:15px;font-weight:600;margin:0 0 20px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.card{border:1px solid var(--line);border-radius:8px;padding:18px;margin-bottom:16px}
#state{font-size:34px;font-weight:700;letter-spacing:.04em;margin-bottom:4px}
.quiet{color:var(--dim)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.meta{color:var(--dim);font-size:12px}
canvas{width:100%;height:120px;display:block;margin-top:14px}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:10px}
td,th{text-align:right;padding:3px 8px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500}td:first-child,th:first-child{text-align:left}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:10px}
.bar>div{height:100%;background:var(--ok);width:0;transition:width .3s}
.wrap{overflow-x:auto}
</style>
<h1>AirRadar — акустическое обнаружение БПЛА</h1>

<div class=card>
  <div id=state class=quiet>—</div>
  <div class=meta id=meta></div>
  <canvas id=c></canvas>
</div>

<div class=card>
  <div class=meta>обучение <span id=epmeta></span></div>
  <div class=bar><div id=prog></div></div>
  <div class=wrap><table id=tbl></table></div>
</div>

<script>
const c=document.getElementById('c'),g=c.getContext('2d');
let thr=0.5;
function draw(h){
  const w=c.width=c.clientWidth*devicePixelRatio, ht=c.height=120*devicePixelRatio;
  g.clearRect(0,0,w,ht);
  const cs=getComputedStyle(document.documentElement);
  g.strokeStyle=cs.getPropertyValue('--bad').trim();g.setLineDash([4,4]);g.lineWidth=1;
  g.beginPath();g.moveTo(0,ht*(1-thr));g.lineTo(w,ht*(1-thr));g.stroke();g.setLineDash([]);
  if(!h.length)return;
  g.strokeStyle=cs.getPropertyValue('--ok').trim();g.lineWidth=1.5*devicePixelRatio;
  g.beginPath();
  h.forEach((v,i)=>{const x=i/(240-1)*w,y=ht*(1-v);i?g.lineTo(x,y):g.moveTo(x,y)});
  g.stroke();
}
async function tick(){
  try{
    const s=await(await fetch('/status')).json();
    thr=s.threshold;
    const el=document.getElementById('state');
    el.textContent=s.alarm?'ДРОН':(s.hits?'внимание':'тихо');
    el.className=s.alarm?'bad':(s.hits?'warn':'quiet');
    document.getElementById('meta').textContent=
      `p=${s.prob.toFixed(3)}  ${s.hits}/${s.m} окон  порог ${s.threshold.toFixed(3)}  `+
      `уровень ${s.level.toFixed(3)}  звук: ${s.audio}`;
    draw(s.history);
    const t=s.training.epochs;
    document.getElementById('epmeta').textContent=t.length?`эпоха ${t[t.length-1].ep}`:'лога нет';
    document.getElementById('prog').style.width=t.length?Math.min(100,t[t.length-1].ep/15*100)+'%':'0';
    document.getElementById('tbl').innerHTML=
      '<tr><th>эпоха<th>loss<th>auc<th>auc_hard<th>FAR_hard@r90</tr>'+
      t.slice(-10).map(e=>`<tr><td>${e.ep}<td>${e.loss.toFixed(4)}<td>${e.auc.toFixed(4)}`+
        `<td>${e.auc_hard!=null?e.auc_hard.toFixed(4):'—'}`+
        `<td>${e.far_hard!=null?e.far_hard.toFixed(1)+'%':'—'}</tr>`).join('');
  }catch(e){}
}
setInterval(tick,250);tick();
</script>
"""


def selfcheck():
    log = ("ep01 loss 0.3150  auc 0.9885  auc_hard 0.9891  FAR_hard@r90 0.5%  <- saved\n"
           "ep02 loss 0.1543  auc 0.9932  auc_hard 0.9936  FAR_hard@r90 0.0%\n"
           "ep03 loss 0.1253  auc 0.9956  auc_hard nan  FAR_hard@r90 nan%\n")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(log); p = f.name
    e = parse_training(p)["epochs"]
    assert len(e) == 3, e
    assert e[0]["ep"] == 1 and abs(e[0]["far_hard"] - 0.5) < 1e-9
    assert e[2]["auc_hard"] is None and e[2]["far_hard"] is None   # nan не должен ронять
    assert parse_training("/нет/такого/файла")["epochs"] == []     # и отсутствие файла тоже
    os.unlink(p)
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log", default="logs/train_hard.log")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "dronenet.pt"))
    ap.add_argument("--device", default="default")
    ap.add_argument("--file", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    Handler.log_file = args.log
    if not args.no_audio:
        threading.Thread(target=audio_loop, args=(args,), daemon=True).start()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"http://127.0.0.1:{args.port}   Ctrl+C для выхода")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
