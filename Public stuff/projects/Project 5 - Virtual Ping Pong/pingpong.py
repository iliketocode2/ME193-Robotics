"""
Rogers Cup -- virtual ping-pong with a LEGO Double Motor paddle.

  pose (MediaPipe)      -> is your paddle in the right place?   (vision.py)
  IMU (Double Motor)    -> are you swinging?                     (paddle_imu.py)
  AprilTags             -> pick your opponent = ball speed       (vision.py, opponents.py)
  MQTT                  -> record # of continuous hits (float)   -> ME193/Rogers/WilliamGoldman

Python runs the whole game (game_logic.py) and streams it over a websocket
to a Three.js page in your browser (web/), which only draws it.

Run:
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/pingpong.py"
    ... --sim        no Double Motor: SPACE in the browser swings
    ... --no-mqtt    don't publish
    ... --no-ai      no AI colour commentator (Ray, the scripted announcer, only)
    ... --perf-log   write per-second performance numbers to <repo>/logs/
Browser: click the on-screen buttons (Start match / Home / Rematch / sound
toggles), or keys ENTER start / home, R rematch, M mute announcer, C mute
crowd, D debug overlay, [ / ] swing sensitivity, ESC quit.
"""

import argparse
import asyncio
import functools
import http.server
import json
import os
import sys
import threading
import time
import webbrowser

import legoeducation as le  # noqa: F401 -- for le.LEGO_COLOR_* in PADDLE_CARD_COLOR
from websockets.asyncio.server import broadcast, serve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "useful libraries"))

from camlib import pick_camera  # noqa: E402
from mqttlib import MQTTClient  # noqa: E402

import game_logic as gl  # noqa: E402
from ai_commentator import Analyst, set_process_mode  # noqa: E402
from commentary import Announcer, Booth  # noqa: E402
from opponents import public_info  # noqa: E402
from paddle_imu import KeyboardPaddle, MotorPaddle  # noqa: E402
from vision import BODY_X_GAIN, MASK_SIZE, SHOULDER_WORLD, SHOULDER_Y, Vision  # noqa: E402

# --- Hardware placeholders -------------------------------------------------------
# None = connect to the first advertising Double Motor. Fine alone, ambiguous
# with other Double Motors nearby -- put your Connection Card's values here.
PADDLE_CARD_SERIAL = 1126
PADDLE_CARD_COLOR = le.LEGO_COLOR_GREEN

# --- MQTT ------------------------------------------------------------------------
MQTT_TOPIC = "ME193/Rogers/WilliamGoldman"
HEARTBEAT_S = 1.0

# --- Servers / timing ------------------------------------------------------------
HTTP_PORT = 8193
WS_PORT = HTTP_PORT + 1          # web/game.js assumes page port + 1
TICK_HZ = 60
VIDEO_HZ = 60                    # cap only -- a frame is sent as soon as vision finishes it
START_HOLD_S = 25                # wait this long for Sonia's warm-up before starting a match
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "logs")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class App:
    def __init__(self, vision, paddle, mqtt, sim, analyst, perf_log=None):
        self.vision = vision
        self.paddle = paddle
        self.mqtt = mqtt
        self.sim = sim
        self.game = gl.Game()
        self.announcer = Announcer()      # Ray: scripted play-by-play
        self.booth = Booth()              # Sonia: decides when the AI analyst speaks
        self.analyst = analyst            # Sonia's AI model, in its own process
        self.clients = set()
        self.stop = asyncio.Event()
        self._last_frame_id = -1
        self._last_cut_id = -1
        self._last_record_sent = None
        self.commentary_on = True         # browser's 🎙 toggle: muted = no AI work at all
        self.start_at = None              # match start held while Sonia warms up (deadline)
        self.held_once = False            # ...at most once per session
        self.browser_perf = {}            # render fps / quality reported by the page
        self.tick_hz = 0                  # measured, for the debug overlay and perf log
        self.tick_gap_ms = 0
        self.perf_log = perf_log

    # ------------------------------------------------------------ browser I/O
    async def handler(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps({
            "type": "hello",
            "opponents": public_info(),
            "sim": self.sim,
            "topic": MQTT_TOPIC,
            "table": {"half_w": gl.TABLE_HALF_W, "half_l": gl.TABLE_HALF_L, "net_h": gl.NET_H,
                      "hit_z": gl.PLAYER_HIT_Z, "opp_z": gl.OPP_HIT_Z, "hit_rx": gl.HIT_RADIUS_X},
            "body": {"shoulder_world": SHOULDER_WORLD, "shoulder_y": SHOULDER_Y,
                     "body_x_gain": BODY_X_GAIN},
            "mask_size": list(MASK_SIZE),
            "game_over_lock": gl.GAME_OVER_LOCK_S,
        }))
        try:
            async for msg in ws:
                self._on_command(json.loads(msg))
        except Exception as e:
            print(f"browser disconnected: {e}")
        finally:
            self.clients.discard(ws)

    def _on_command(self, msg):
        cmd = msg.get("cmd")
        if cmd == "confirm":
            self._confirm()
        elif cmd == "commentary":
            self.commentary_on = bool(msg.get("on", True))
        elif cmd == "perf":
            self.browser_perf = {"fps": msg.get("fps"), "quality": msg.get("quality")}
        elif cmd == "home":
            self.game.go_home()
        elif cmd == "rematch":
            self.game.rematch()
        elif cmd == "swing" and self.sim:
            self.paddle.key_swing()
        elif cmd == "sens_up":
            self.paddle.nudge_threshold(1 / 1.15)
        elif cmd == "sens_down":
            self.paddle.nudge_threshold(1.15)
        elif cmd == "quit":
            print("Quit from browser.")
            self.stop.set()

    def _confirm(self):
        """Start (or leave) a match. If Sonia is still warming up, hold the
        start up to START_HOLD_S so her one-time GPU warm-up never overlaps
        play; pressing Start again skips the wait."""
        if (self.game.state == gl.SELECT and self.game.highlight and self.analyst.status == "loading"
                and self.commentary_on and not self.held_once):
            if self.start_at is None:
                self.start_at = time.monotonic() + START_HOLD_S
                return
            self.start_at = None                    # second press: start now
        if self.game.state == gl.SELECT and self.game.highlight:
            self.held_once = True                   # waited (or chose not to) once; never again
        self.game.confirm()

    # -------------------------------------------------------------- game tick
    async def run(self):
        async with serve(self.handler, "127.0.0.1", WS_PORT):
            print(f"Game running at http://localhost:{HTTP_PORT}/  (Ctrl-C or ESC to quit)")
            self._publish_record(force=True)
            await self._tick_loop()

    async def _tick_loop(self):
        period = 1 / TICK_HZ
        prev = time.monotonic()
        next_video = next_beat = next_tick = prev
        ticks, worst_gap, stats_t = 0, 0.0, prev
        while not self.stop.is_set():
            now = time.monotonic()
            dt = min(now - prev, 0.1)
            worst_gap = max(worst_gap, now - prev)
            prev = now
            ticks += 1
            if now - stats_t >= 1.0:
                self.tick_hz, self.tick_gap_ms = ticks, round(worst_gap * 1000)
                ticks, worst_gap, stats_t = 0, 0.0, now

            if self.start_at is not None and (self.analyst.status != "loading" or now >= self.start_at
                                              or not self.commentary_on):
                self.start_at = None
                self.held_once = True
                self.game.confirm()

            vs = self.vision.read()
            self.vision.want_tags = self.game.state == gl.SELECT
            if vs.tag_opponent:
                self.game.set_highlight(vs.tag_opponent)   # sticky: put the tag down, then press Enter

            paddle = gl.Paddle(vs.paddle_x, vs.paddle_y, vs.paddle_vx, vs.visible)
            swings = self.paddle.take_swings()
            events = self.game.update(now, dt, paddle, swings)
            events += self.announcer.update(now, events, self.game, handoff=self.analyst.ready)
            events += self._booth(now, events)
            for ev in events:
                self._on_event(ev)
            # Sonia may only use the GPU while the ball is dead (and someone is listening)
            self.analyst.set_gate(self.commentary_on and self.game.state != gl.RALLY)

            if self.clients:
                snap = self.game.snapshot()
                snap.update(type="state", events=events, swung=bool(swings),
                            paddle={"x": vs.paddle_x, "y": vs.paddle_y, "visible": vs.visible},
                            body={"u": vs.shoulder_u, "v": vs.shoulder_v, "w": vs.shoulder_w},
                            debug={"gyro": round(self.paddle.gyro_mag), "threshold": round(self.paddle.threshold),
                                   "fps": round(vs.fps, 1), "paddle_connected": self.paddle.connected,
                                   "frame_age_ms": round((time.time() - vs.frame_t) * 1000) if vs.frame_t else None,
                                   "tick_hz": self.tick_hz, "tick_gap_ms": self.tick_gap_ms},
                            mqtt_ok=self.mqtt is not None, booth=self.analyst.info(),
                            hold=round(self.start_at - now, 1) if self.start_at else None)
                broadcast(self.clients, json.dumps(snap))
                if now >= next_video:
                    next_video = now + 1 / VIDEO_HZ
                    self._send_video()

            if now >= next_beat:
                next_beat = now + HEARTBEAT_S
                self._publish_record(force=True)
                self._log_perf(vs)

            # fixed-rate schedule: aim at the next 60 Hz slot rather than sleeping a
            # fixed amount after the work (which drifts slower and slower)
            next_tick += period
            if next_tick < time.monotonic() - period:
                next_tick = time.monotonic()          # fell behind: resync, don't burst
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    def _log_perf(self, vs):
        if not self.perf_log:
            return
        b, a = self.browser_perf, self.analyst.info()
        age = round((time.time() - vs.frame_t) * 1000) if vs.frame_t else ""
        try:
            self.perf_log.write(f"{time.time():.1f},{self.game.state},{self.tick_hz},{self.tick_gap_ms},{vs.fps:.1f},{age},"
                                f"{b.get('fps', '')},{b.get('quality', '')},{a['status']},{int(a['gate'])},"
                                f"{int(a['generating'])},{a['cancelled']}\n")
            self.perf_log.flush()
        except OSError as e:                     # a diagnostic log must never end the game
            print(f"perf log stopped: {e}")
            self.perf_log = None

    def _send_video(self):
        video_id, video_jpeg, cut_id, cut_jpeg, mask = self.vision.latest_images()
        if self.game.state == gl.SELECT:
            # select screen: every camera frame (you + your AprilTag outlines)
            if video_jpeg and video_id != self._last_frame_id:
                self._last_frame_id = video_id
                broadcast(self.clients, b"F" + video_jpeg)
        elif cut_jpeg and mask and cut_id != self._last_cut_id:
            # in game: the frame pose ran on + its person mask. The browser's
            # GPU uses the mask as transparency to cut you out (much cheaper
            # than encoding a transparent image here).
            self._last_cut_id = cut_id
            broadcast(self.clients, b"C" + mask + cut_jpeg)

    def _on_event(self, ev):
        kind = ev["type"]
        if kind == "hit":
            self.paddle.feedback("hit")
        elif kind == "point":
            self.paddle.feedback("point" if ev["winner"] == gl.PLAYER else "miss")
        elif kind == "game_over":
            self.paddle.feedback("win" if ev["winner"] == gl.PLAYER else "lose")
        elif kind == "state" and ev["state"] == gl.SELECT:
            self.paddle.feedback("idle")
        elif kind == "record":
            self._publish_record()
        elif kind == "say":
            print(f'  {self.game.opp["name"]}: "{ev["text"]}"')
        elif kind == "call":
            who = "Ray" if ev.get("speaker") == "ray" else f'Sonia, AI {ev.get("latency", 0):.1f}s'
            print(f'  [{who}] {ev["text"]}')

    def _booth(self, now, events):
        """Ask the AI analyst for lines when the Booth wants one; turn finished
        lines into Sonia call events (late or stale ones are dropped)."""
        for req in self.booth.update(now, events, self.game, self.analyst.ready and self.commentary_on):
            self.analyst.request(req)
        calls = []
        for res in self.analyst.poll():
            call = self.booth.accept(res, now, self.game)
            if call:
                calls.append(call)
        return calls

    def _publish_record(self, force=False):
        """Record # of continuous hits, as a bare float string (e.g. '12.0')."""
        if self.mqtt is None:
            return
        rec = self.game.record
        if force or rec != self._last_record_sent:
            try:
                self.mqtt.publish(MQTT_TOPIC, f"{rec:.1f}")
                self._last_record_sent = rec
            except Exception as e:
                print(f"MQTT publish failed: {e}")


def start_http():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", HTTP_PORT),
                                            functools.partial(Quiet, directory=WEB_DIR))
    threading.Thread(target=httpd.serve_forever, name="http", daemon=True).start()
    return httpd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="no Double Motor; SPACE in the browser swings")
    ap.add_argument("--no-mqtt", action="store_true", help="don't publish the record")
    ap.add_argument("--no-ai", action="store_true", help="no AI colour commentator")
    ap.add_argument("--perf-log", action="store_true", help="write per-second performance numbers to logs/")
    args = ap.parse_args()

    # Windows' default timer ticks every 15.6 ms, which turns a 60 Hz loop into
    # ~32 Hz (measured). Ask for 1 ms timers while the game runs.
    winmm = None
    if sys.platform == "win32":
        import ctypes
        winmm = ctypes.WinDLL("winmm")
        winmm.timeBeginPeriod(1)
    paddle = analyst = perf_log = cap = vision = mqtt = httpd = app = None
    try:
        # The camera and game threads live in this process: ask Windows to
        # keep it on the performance cores.
        set_process_mode(background=False)
        if args.perf_log:
            os.makedirs(LOG_DIR, exist_ok=True)
            path = os.path.join(LOG_DIR, time.strftime("perf_%Y%m%d_%H%M%S.csv"))
            perf_log = open(path, "w")
            perf_log.write("t,state,tick_hz,tick_gap_ms,vision_fps,frame_age_ms,render_fps,quality,"
                           "ai_status,ai_gate,ai_generating,ai_cancelled\n")
            print(f"Performance log -> {os.path.abspath(path)}")

        # Start the AI analyst first: its model loads in the background while
        # the paddle connects and you pick a camera.
        analyst = Analyst(enabled=not args.no_ai)
        analyst.start()

        paddle = KeyboardPaddle() if args.sim else MotorPaddle(PADDLE_CARD_SERIAL, PADDLE_CARD_COLOR)
        print("Connecting paddle (Double Motor)..." if not args.sim else "Simulation mode.")
        paddle.connect()

        cap, start_ms = pick_camera(width=640, height=480)
        vision = Vision(cap, start_ms)
        vision.start()

        if not args.no_mqtt:
            try:
                mqtt = MQTTClient()
                mqtt.connect()
                print(f"MQTT connected -> publishing record to {MQTT_TOPIC}")
            except Exception as e:
                print(f"MQTT unavailable ({e}); playing offline.")
                mqtt = None

        httpd = start_http()
        app = App(vision, paddle, mqtt, args.sim, analyst, perf_log)
        webbrowser.open(f"http://localhost:{HTTP_PORT}/")
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nCtrl-C -- shutting down.")
    finally:
        if paddle is not None:
            paddle.close()                      # stops both motors, then disconnects
        if analyst is not None:
            analyst.close()                     # stop the AI process (bounded wait, never raises)
        if mqtt is not None:
            try:
                if app is not None:
                    mqtt.publish(MQTT_TOPIC, f"{app.game.record:.1f}")
                    print(f"Final record: {app.game.record:.1f}")
                mqtt.disconnect()
            except Exception as e:
                print(f"MQTT shutdown failed: {e}")
        if vision is not None:
            vision.stop()
        if cap is not None:
            cap.release()
        if httpd is not None:
            httpd.shutdown()
        if perf_log is not None:
            try:
                perf_log.close()
            except OSError:
                pass
        if winmm is not None:
            winmm.timeEndPeriod(1)


if __name__ == "__main__":
    main()
