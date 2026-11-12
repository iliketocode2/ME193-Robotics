"""A live view of the board's camera, in a web browser on any computer on the same Wi-Fi.

main.py starts it, then open  http://<board IP>:8000  in a browser. The page shows the camera with the lock's
state drawn on top, like lock.py on the computer, plus the 64 x 64 image the network actually gets.
Frames are only turned into JPEGs while someone has the page open, so it costs nothing otherwise.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import camera

PORT = 8000

BGR = {"yellow": (0, 215, 255), "green": (60, 200, 60), "red": (50, 50, 230), "off": (70, 70, 70),
       "white": (255, 255, 255), "gray": (200, 200, 200)}
LATCH_TEXT = {None: "latch: ?", 0: "latch: locked", 1: "latch: open", 2: "latch: FAULT (switch did not click)",
              3: "latch: moving"}

PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Finger lock camera</title>
<style>
  body { margin: 0; background: #111; color: #ddd; font: 15px system-ui, sans-serif; text-align: center; }
  img { display: block; margin: 12px auto; max-width: 100%; height: auto; border-radius: 6px; }
  p { margin: 8px 16px; }
</style></head>
<body>
  <img id="v" src="/stream" alt="camera">
  <p id="s">If the picture freezes, the app has stopped. Reload once it is running again.</p>
<script>
  // reconnect if the stream drops (for example when the app restarts)
  const v = document.getElementById("v");
  v.onerror = () => setTimeout(() => { v.src = "/stream?" + Date.now(); }, 2000);
</script>
</body></html>"""


class Viewer:
    def __init__(self, port=PORT):
        self.cond = threading.Condition()
        self.jpg, self.seq, self.clients = None, 0, 0
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):     # keep the app log quiet
                pass

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif self.path.startswith("/stream"):
                    viewer.stream(self)
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"live view: http://<board IP>:{port}", flush=True)

    @property
    def watching(self):
        return self.clients > 0

    def publish(self, image):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.cond:
                self.jpg, self.seq = buf.tobytes(), self.seq + 1
                self.cond.notify_all()

    def stream(self, handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        with self.cond:
            self.clients += 1
        last = -1
        try:
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: self.seq != last, timeout=5)
                    jpg, last = self.jpg, self.seq
                if jpg is None:
                    continue
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                    + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self.cond:
                self.clients -= 1


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
