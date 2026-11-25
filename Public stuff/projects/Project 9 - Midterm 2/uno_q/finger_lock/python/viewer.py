"""A live view of the lock, in a web browser on any computer on the same Wi-Fi.

main.py starts it, then open  http://<board IP>:8000  in a browser. The page has three panels:

  camera          the webcam with the reading, the lock state and the LEDs drawn on top
  inside the net  the image going through every layer: the 64 x 64 input, the 8, 16 and 32 feature maps,
                  the six probabilities, and a heatmap of which parts of the picture raised the winning score
  lock policy     the state machine with the current state lit, and a timeline of the last 15 seconds of
                  readings against the confidence threshold

Each image is only drawn and turned into a JPEG while someone is watching it, so the page costs nothing otherwise.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import camera

PORT = 8000

BGR = {"yellow": (0, 215, 255), "green": (60, 200, 60), "red": (50, 50, 230), "off": (70, 70, 70),
       "white": (255, 255, 255), "gray": (200, 200, 200)}
LATCH_TEXT = {None: "latch: ?", 0: "latch: locked", 1: "latch: open", 2: "latch: FAULT (switch did not click)",
              3: "latch: moving"}

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Finger lock live</title>
<style>
  :root { --bg: #14161a; --panel: #1d2026; --line: #343844; --text: #e6e6e6; --dim: #9aa0aa;
          --yellow: #ffd23f; --green: #3cc864; --red: #e64646; --blue: #5aa0ff; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.4 system-ui, sans-serif; }
  header { padding: 14px 16px 4px; display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: baseline; }
  h1 { font-size: 20px; margin: 0; }
  #stats { color: var(--dim); font-size: 13px; }
  main { display: grid; grid-template-columns: minmax(0, 640px) minmax(0, 660px); gap: 14px; padding: 10px 16px 20px;
         justify-content: center; }
  section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; min-width: 0; }
  section.wide { grid-column: 1 / -1; }
  h2 { font-size: 15px; margin: 0 0 8px; color: var(--dim); font-weight: 600; }
  img, svg, canvas { display: block; width: 100%; height: auto; border-radius: 4px; }
  #press { margin: 0 0 8px; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--line);
           background: #262a32; color: var(--text); font: inherit; cursor: pointer; }
  .note { color: var(--dim); font-size: 13px; margin: 6px 0 0; }
  #digits span { display: inline-block; width: 14px; height: 14px; border-radius: 50%; margin-right: 6px;
                 background: #464a54; vertical-align: -2px; }
  #digits span.on { background: var(--yellow); }
  #msg { margin: 6px 0 8px; }
  svg text { font: 13px system-ui, sans-serif; fill: var(--text); }
  svg .lbl { font-size: 11px; fill: var(--dim); stroke: var(--panel); stroke-width: 4px; paint-order: stroke; }
  .node rect { fill: #262a32; stroke: #4a5060; stroke-width: 1.5; }
  .node.on rect { stroke-width: 3; }
  @media (max-width: 1340px) { main { grid-template-columns: minmax(0, 1fr); } }
</style></head>
<body>
<header><h1>Finger lock, live</h1><span id="stats"></span></header>
<main>
  <section><h2>Camera</h2><img id="cam" src="/stream/cam" alt="camera with the lock state drawn on top"></section>
  <section>
    <h2>Lock policy</h2>
    <div id="msg"></div>
    <button id="press" hidden>Press the lock button</button>
    <div id="digits"><span></span><span></span><span></span></div><div id="fails" class="note"></div>
    <svg id="fsm" viewBox="0 0 660 330" role="img" aria-label="lock state machine"></svg>
    <canvas id="tl" width="1300" height="300"></canvas>
    <p class="note">Last 15 s: the line is how sure the network is, colored by the count it reads. The dashed line is
      the threshold a reading has to beat. The shading is the lock state.</p>
  </section>
  <section class="wide"><h2>Inside the network</h2><img id="net" src="/stream/net" alt="feature maps of every layer">
    <p class="note">Brighter means a stronger response. Each map is one filter's answer to "is my pattern here?".
      The heatmap adds up, over the last layer, how much each spot raised the winning score.</p></section>
</main>
<script>
const COUNT_COLORS = ["#9aa0aa", "#5aa0ff", "#3cc8c8", "#3cc864", "#ffb43c", "#e664c8"];
const STATE_COLORS = {LOCKED: "#9aa0aa", CLEAR: "#ffd23f", READ: "#ffd23f", OPEN: "#3cc864", WRONG: "#e64646",
                      LOCKOUT: "#e64646"};
const NODES = {LOCKED: [90, 170], CLEAR: [290, 55], READ: [560, 55], OPEN: [570, 205], WRONG: [380, 290],
               LOCKOUT: [90, 295]};
// [from, to, label, bend]: bend pushes the curve sideways so arrows don't overlap
const EDGES = [["LOCKED", "CLEAR", "button", -25], ["CLEAR", "READ", "empty view 0.4 s", 0],
  ["READ", "CLEAR", "digit held 1 s", 45], ["READ", "OPEN", "3rd digit, right code", -40],
  ["READ", "WRONG", "wrong code", 0], ["WRONG", "LOCKED", "after 1.5 s", 0],
  ["READ", "LOCKOUT", "3rd wrong code in a row", 70], ["LOCKOUT", "LOCKED", "after 30 s", -30],
  ["OPEN", "LOCKED", "button", -35], ["CLEAR", "LOCKED", "timeout or button", -25]];
const svg = document.getElementById("fsm"), NS = "http://www.w3.org/2000/svg";
function el(name, attrs, parent) {
  const e = document.createElementNS(NS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  (parent || svg).appendChild(e); return e;
}
el("defs", {}).innerHTML = '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" ' +
  'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#7a8090"/></marker>';
const W = 118, H = 40;
function edgePoint(c, toward) {   // where the line from c toward a point leaves the box around c
  const dx = toward[0] - c[0], dy = toward[1] - c[1];
  const s = Math.min(Math.abs((W / 2 + 4) / (dx || 1e-9)), Math.abs((H / 2 + 4) / (dy || 1e-9)));
  return [c[0] + dx * s, c[1] + dy * s];
}
for (const [a, b, label, bend] of EDGES) {
  const p = NODES[a], q = NODES[b];
  const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2, len = Math.hypot(q[0] - p[0], q[1] - p[1]);
  const nx = -(q[1] - p[1]) / len, ny = (q[0] - p[0]) / len;
  const cx = mx + nx * bend, cy = my + ny * bend;
  const s = edgePoint(p, [cx, cy]), e = edgePoint(q, [cx, cy]);
  el("path", {d: `M${s[0]},${s[1]} Q${cx},${cy} ${e[0]},${e[1]}`, fill: "none", stroke: "#7a8090",
              "stroke-width": 1.5, "marker-end": "url(#arr)"});
  const t = el("text", {x: (s[0] + 2 * cx + e[0]) / 4, y: (s[1] + 2 * cy + e[1]) / 4 - 5, "text-anchor": "middle",
                        class: "lbl"});
  t.textContent = label;
}
const nodeEls = {};
for (const name in NODES) {
  const [x, y] = NODES[name];
  const g = el("g", {class: "node"});
  el("rect", {x: x - W / 2, y: y - H / 2, width: W, height: H, rx: 8}, g);
  const t = el("text", {x: x, y: y + 5, "text-anchor": "middle"}, g); t.textContent = name;
  const bar = el("rect", {x: x - W / 2 + 8, y: y + H / 2 - 7, width: 0, height: 3, rx: 1.5,
                          fill: "#ffd23f"}, g);
  nodeEls[name] = {g, bar};
}

const tl = document.getElementById("tl"), ctx = tl.getContext("2d"), SPAN = 15;
function drawTimeline(s) {
  const w = tl.width, h = tl.height, padL = 60, padB = 34, padT = 44, pw = w - padL - 12, ph = h - padB - padT;
  ctx.clearRect(0, 0, w, h);
  const X = t => padL + pw * (1 - (s.now - t) / SPAN), Y = c => padT + ph * (1 - c);
  const hist = s.history;
  for (let i = 1; i < hist.length; i++) {      // state shading
    ctx.fillStyle = STATE_COLORS[hist[i][3]] + "22";
    ctx.fillRect(X(hist[i - 1][0]), padT, X(hist[i][0]) - X(hist[i - 1][0]) + 1, ph);
  }
  ctx.strokeStyle = "#343844"; ctx.lineWidth = 1; ctx.font = "20px system-ui"; ctx.fillStyle = "#9aa0aa";
  for (const c of [0, 0.5, 1]) {
    ctx.beginPath(); ctx.moveTo(padL, Y(c)); ctx.lineTo(padL + pw, Y(c)); ctx.stroke();
    ctx.fillText(Math.round(c * 100) + "%", 4, Y(c) + 7);
  }
  for (let k = 0; k <= SPAN; k += 5) ctx.fillText(k ? "-" + k + " s" : "now", X(s.now - k) - 24, h - 6);
  ctx.setLineDash([10, 8]); ctx.strokeStyle = "#e6e6e6"; ctx.beginPath();
  ctx.moveTo(padL, Y(s.min_conf)); ctx.lineTo(padL + pw, Y(s.min_conf)); ctx.stroke(); ctx.setLineDash([]);
  ctx.lineWidth = 4;
  for (let i = 1; i < hist.length; i++) {
    ctx.strokeStyle = COUNT_COLORS[hist[i][1]];
    ctx.beginPath(); ctx.moveTo(X(hist[i - 1][0]), Y(hist[i - 1][2])); ctx.lineTo(X(hist[i][0]), Y(hist[i][2]));
    ctx.stroke();
  }
  let x = padL;                                 // legend
  ctx.font = "20px system-ui"; ctx.fillStyle = "#9aa0aa"; ctx.fillText("count read:", x, 25); x += 120;
  for (let c = 0; c < 6; c++) {
    ctx.fillStyle = COUNT_COLORS[c]; ctx.fillRect(x, 10, 16, 16);
    ctx.fillStyle = "#e6e6e6"; ctx.fillText(String(c), x + 22, 25); x += 54;
  }
}

async function poll() {
  try {
    const s = await (await fetch("/status", {cache: "no-store"})).json();
    for (const name in nodeEls) {
      const on = name === s.state, n = nodeEls[name];
      n.g.classList.toggle("on", on);
      n.g.querySelector("rect").style.stroke = on ? STATE_COLORS[name] : "";
      n.bar.setAttribute("width", on && s.progress ? (W - 16) * s.progress : 0);
    }
    document.getElementById("msg").textContent = s.message + (s.time_left != null ? `  (${Math.ceil(s.time_left)} s)` : "");
    document.querySelectorAll("#digits span").forEach((d, i) => d.classList.toggle("on", i < s.digits));
    document.getElementById("fails").textContent = s.fails ? `wrong codes ${s.fails}/${s.max_fails}` : "";
    document.getElementById("stats").textContent =
      `${s.fps.toFixed(0)} frames/s · network ${s.infer_ms.toFixed(1)} ms per frame · ${s.where}`;
    document.getElementById("press").hidden = !s.can_press;
    drawTimeline(s);
  } catch (e) {
    document.getElementById("stats").textContent = "not connected, retrying";
  }
  setTimeout(poll, 250);
}
poll();
document.getElementById("press").onclick = () => fetch("/press");
// reconnect the picture streams if they drop (for example when the app restarts)
for (const id of ["cam", "net"]) {
  const img = document.getElementById(id);
  img.onerror = () => setTimeout(() => { img.src = `/stream/${id}?` + Date.now(); }, 2000);
}
</script>
</body></html>"""


class Viewer:
    """A small web server: the page, two MJPEG picture streams (cam, net), and /status as JSON."""

    def __init__(self, port=PORT):
        self.cond = threading.Condition()
        self.jpg, self.seq, self.clients = {}, {}, {}
        self.status, self.status_asked = {}, 0.0
        self.on_press = None      # set only on the computer: a button on the page stands in for the real one
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):     # keep the app log quiet
                pass

            def do_GET(self):
                if self.path == "/":
                    self.reply(PAGE.encode(), "text/html; charset=utf-8")
                elif self.path.startswith("/status"):
                    viewer.status_asked = time.monotonic()
                    self.reply(json.dumps(viewer.status).encode(), "application/json")
                elif self.path == "/press" and viewer.on_press:
                    viewer.on_press()
                    self.reply(b"ok", "text/plain")
                elif self.path.startswith("/stream/"):
                    viewer.stream(self, self.path[len("/stream/"):].split("?")[0])
                elif self.path.startswith("/stream"):      # the old single-stream address
                    viewer.stream(self, "cam")
                else:
                    self.send_error(404)

            def reply(self, body, kind):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"live view: http://<board IP>:{port}", flush=True)

    def watching(self, name="cam"):
        return self.clients.get(name, 0) > 0

    @property
    def wants_status(self):
        return time.monotonic() - self.status_asked < 3

    def publish(self, image, name="cam"):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.cond:
                self.jpg[name], self.seq[name] = buf.tobytes(), self.seq.get(name, 0) + 1
                self.cond.notify_all()

    def stream(self, handler, name):
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        with self.cond:
            self.clients[name] = self.clients.get(name, 0) + 1
        last = -1
        try:
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: self.seq.get(name, 0) != last, timeout=5)
                    jpg, last = self.jpg.get(name), self.seq.get(name, 0)
                if jpg is None:
                    continue
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                    + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self.cond:
                self.clients[name] -= 1


# Inside the network ---------------------------------------------------------------------------------------------

NET_W, NET_H = 1280, 560
INK, DIM, BG = (235, 235, 235), (160, 154, 150), (38, 32, 29)


def label(img, s, org, scale=0.5, color=INK, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def tile(maps, cols, cell, gap, vmax):
    """Feature maps side by side in a grid, with one brightness scale for the whole layer, so a filter that
    finds nothing stays dark."""
    n = len(maps)
    rows = (n + cols - 1) // cols
    out = np.full((rows * cell + (rows - 1) * gap, cols * cell + (cols - 1) * gap, 3), BG, np.uint8)
    for k, m in enumerate(maps):
        r, c = divmod(k, cols)
        u8 = np.clip(m / vmax * 255, 0, 255).astype(np.uint8)
        im = cv2.applyColorMap(cv2.resize(u8, (cell, cell), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)
        out[r * (cell + gap):r * (cell + gap) + cell, c * (cell + gap):c * (cell + gap) + cell] = im
    return out


def paste(dst, src, x, y):
    dst[y:y + src.shape[0], x:x + src.shape[1]] = src


def arrow(img, x0, x1, y):
    cv2.arrowedLine(img, (x0, y), (x1, y), DIM, 2, cv2.LINE_AA, tipLength=0.25)


def filters_strip(w, cell=26, gap=6):
    """The 8 learned 5 x 5 conv1 filters: red is a positive weight, blue a negative one."""
    k = w[:, 0]
    out = np.full((cell, len(k) * (cell + gap) - gap, 3), BG, np.uint8)
    vmax = np.abs(k).max() + 1e-9
    for i, f in enumerate(k):
        v = f / vmax          # -1 to 1; white at 0, red toward +1, blue toward -1 (channels are B, G, R)
        img = np.stack([1 - np.clip(v, 0, 1), 1 - np.abs(v), 1 - np.clip(-v, 0, 1)], -1)
        img = (255 * img).astype(np.uint8)
        out[:, i * (cell + gap):i * (cell + gap) + cell] = cv2.resize(img, (cell, cell),
                                                                      interpolation=cv2.INTER_NEAREST)
    return out


def contribution(maps, fc_w, cls):
    """How much each of the 8 x 8 spots in the last layer added to the score for `cls`.

    The linear layer comes straight after conv3, so the score is a plain sum: for every filter k and spot (i, j),
    weight[cls, k, i, j] times activation[k, i, j]. Adding up over k leaves one number per spot."""
    a = maps[2]
    return (fc_w[cls].reshape(a.shape) * a).sum(0)


def draw_network(inp, maps, probs, weights, min_conf):
    img = np.full((NET_H, NET_W, 3), BG, np.uint8)
    count = int(np.argmax(probs))

    # top row: input -> conv1 -> conv2 -> conv3
    gray = cv2.normalize(inp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    paste(img, cv2.cvtColor(cv2.resize(gray, (192, 192), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR),
          20, 50)
    label(img, "input", (20, 28), 0.65)
    label(img, "1 x 64 x 64", (20, 44), 0.45, DIM)

    layers = [("conv1", "8 x 32 x 32", maps[0], 4, 64, 250),
              ("conv2", "16 x 16 x 16", maps[1], 4, 48, 560),
              ("conv3", "32 x 8 x 8", maps[2], 8, 40, 830)]
    prev_end = 212
    for name, size, m, cols, cell, x in layers:
        t = tile(m, cols, cell, 4, m.max() + 1e-6)
        paste(img, t, x, 50)
        label(img, name, (x, 28), 0.65)
        label(img, f"{size}  (filters, ReLU, 2x2 max pool)", (x, 44), 0.45, DIM)
        arrow(img, prev_end + 4, x - 8, 146)
        prev_end = x + t.shape[1]

    strip = filters_strip(weights["conv1_w"])
    paste(img, strip, 250, 200)
    label(img, "the 8 learned 5 x 5 filters (red +, blue -)", (250, 248), 0.45, DIM)

    # bottom row: heatmap, linear layer, probabilities
    y0 = 300
    c = np.maximum(contribution(maps, weights["fc_w"], count), 0)
    c = cv2.resize((c / (c.max() + 1e-9)).astype(np.float32), (192, 192), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((np.clip(c, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(cv2.resize(gray, (192, 192)), cv2.COLOR_GRAY2BGR)
    alpha = np.clip(c, 0, 1)[..., None] * 0.65
    paste(img, (base * (1 - alpha) + heat * alpha).astype(np.uint8), 20, y0 + 30)
    label(img, f"where it looked for '{count}'", (20, y0 + 18), 0.55)

    label(img, "flatten: 32 x 8 x 8 = 2048 numbers", (250, y0 + 60), 0.55)
    label(img, "linear layer: 6 weighted sums of all 2048,", (250, y0 + 95), 0.55)
    label(img, "one score per count", (250, y0 + 120), 0.55)
    label(img, "softmax: scores -> probabilities", (250, y0 + 155), 0.55)
    label(img, "that add up to 100%", (250, y0 + 180), 0.55)
    arrow(img, 610, 680, y0 + 120)

    bx, bw, bh = 740, 460, 30
    label(img, "probability for each count", (bx, y0 + 18), 0.55)
    tx = bx + 30 + int(bw * min_conf)
    for k, p in enumerate(probs):
        y = y0 + 34 + k * (bh + 8)
        label(img, str(k), (bx, y + 22), 0.65)
        cv2.rectangle(img, (bx + 30, y), (bx + 30 + bw, y + bh), (60, 54, 50), -1)
        color = (60, 200, 60) if k == count and p >= min_conf else (0, 200, 255) if k == count else (150, 140, 130)
        cv2.rectangle(img, (bx + 30, y), (bx + 30 + int(bw * float(p)), y + bh), color, -1)
        label(img, f"{p:.0%}", (bx + 40 + bw, y + 22), 0.55)
    cv2.line(img, (tx, y0 + 28), (tx, y0 + 34 + 6 * (bh + 8)), INK, 1, cv2.LINE_AA)
    label(img, f"{min_conf:.0%} needed", (tx - 40, y0 + 52 + 6 * (bh + 8)), 0.45, DIM)
    return img


# Camera panel ---------------------------------------------------------------------------------------------------

def text(img, s, org, scale=0.6, color=BGR["white"]):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def message_frame(message, w=640, h=480):
    """A dark frame with one line of text, for when there is no camera picture."""
    img = np.full((h, w, 3), 30, np.uint8)
    text(img, message, (20, h // 2), 0.7)
    return img


def draw(frame, inp, policy, count, conf, now, latch_state, fps, inputs=None):
    """The camera frame with the lock's state on top."""
    view = frame.copy()
    h, w = view.shape[:2]
    leds = policy.leds(now)
    listening = policy.state in ("CLEAR", "READ")

    # the square the network looks at
    x, y, side = camera.crop_box(frame)
    color = BGR["yellow"] if listening else BGR[leds["status"]] if leds["status"] != "off" else BGR["gray"]
    cv2.rectangle(view, (x, y), (x + side, y + side), color, 2)

    # what the network gets: 64 x 64, stretched to 0-255 so it can be seen
    small = cv2.normalize(inp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    small = cv2.cvtColor(cv2.resize(small, (128, 128), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    view[8:136, w - 136:w - 8] = small
    cv2.rectangle(view, (w - 137, 7), (w - 7, 137), BGR["white"], 1)
    text(view, "network input", (w - 136, 154), 0.45)

    # top: reading, state, latch
    sure = conf >= policy.min_conf
    text(view, f"reading {count}  ({conf:.0%})", (10, 28), 0.8, BGR["green"] if sure else BGR["gray"])
    text(view, f"{policy.state}: {policy.message}", (10, 58))
    text(view, LATCH_TEXT.get(latch_state, f"latch: {latch_state}"), (10, 84), 0.55,
         BGR["red"] if latch_state == 2 else BGR["white"])
    left = policy.time_left(now)
    extra = f"wrong codes {policy.fails}/{policy.max_fails}" if policy.fails else ""
    if left is not None and left != float("inf"):
        extra = (extra + "   " if extra else "") + f"{left:.0f} s left"
    if extra:
        text(view, extra, (10, 108), 0.55)

    # hold progress
    p = policy.progress(now)
    if listening and p > 0:
        cv2.rectangle(view, (x, y + side - 10), (x + int(side * p), y + side), BGR["yellow"], -1)

    # bottom: the LEDs as they are on the box
    for i, lit in enumerate(leds["digits"]):
        cv2.circle(view, (25 + 35 * i, h - 25), 12, BGR["yellow"] if lit else BGR["off"], -1)
    cv2.circle(view, (25 + 35 * 3 + 10, h - 25), 12, BGR["green"] if leds["status"] == "green" else BGR["off"], -1)
    cv2.circle(view, (25 + 35 * 4 + 10, h - 25), 12, BGR["red"] if leds["status"] == "red" else BGR["off"], -1)
    text(view, f"{fps:.0f} fps", (w - 80, h - 18), 0.5)

    # raw inputs from the sketch: shows at once whether the button and switches are wired right
    if inputs:
        def state(v, on):
            return "?" if v is None else on if v else "-"
        line = (f"button: {state(inputs['button'], 'DOWN')}   locked sw: {state(inputs['locked'], 'CLICK')}"
                f"   open sw: {state(inputs['open'], 'CLICK')}")
        text(view, line, (10, h - 50), 0.5, BGR["yellow"] if inputs["button"] else BGR["white"])
    return view
