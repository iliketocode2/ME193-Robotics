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
Browser keys: ENTER confirm opponent / play again, D debug overlay,
[ / ] swing sensitivity, ESC quit.
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
from opponents import public_info  # noqa: E402
from paddle_imu import KeyboardPaddle, MotorPaddle  # noqa: E402
from vision import BODY_X_GAIN, SHOULDER_WORLD, SHOULDER_Y, Vision  # noqa: E402

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
VIDEO_HZ = 20
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class App:
    def __init__(self, vision, paddle, mqtt, sim):
        self.vision = vision
        self.paddle = paddle
        self.mqtt = mqtt
        self.sim = sim
        self.game = gl.Game()
        self.clients = set()
        self.stop = asyncio.Event()
        self._last_frame_id = -1
        self._last_record_sent = None

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
            self.game.confirm()
        elif cmd == "swing" and self.sim:
            self.paddle.key_swing()
        elif cmd == "sens_up":
            self.paddle.nudge_threshold(1 / 1.15)
        elif cmd == "sens_down":
            self.paddle.nudge_threshold(1.15)
        elif cmd == "quit":
            print("Quit from browser.")
            self.stop.set()

    # -------------------------------------------------------------- game tick
    async def run(self):
        async with serve(self.handler, "127.0.0.1", WS_PORT):
            print(f"Game running at http://localhost:{HTTP_PORT}/  (Ctrl-C or ESC to quit)")
            self._publish_record(force=True)
            await self._tick_loop()

    async def _tick_loop(self):
        period = 1 / TICK_HZ
        prev = time.monotonic()
        next_video = next_beat = prev
        while not self.stop.is_set():
            now = time.monotonic()
            dt = min(now - prev, 0.1)
            prev = now

            vs = self.vision.read()
            self.vision.want_tags = self.game.state == gl.SELECT
            if vs.tag_opponent:
                self.game.set_highlight(vs.tag_opponent)   # sticky: put the tag down, then press Enter

            paddle = gl.Paddle(vs.paddle_x, vs.paddle_y, vs.paddle_vx, vs.visible)
            swings = self.paddle.take_swings()
            events = self.game.update(now, dt, paddle, swings)
            for ev in events:
                self._on_event(ev)

            if self.clients:
                snap = self.game.snapshot()
                snap.update(type="state", events=events, swung=bool(swings),
                            paddle={"x": vs.paddle_x, "y": vs.paddle_y, "visible": vs.visible},
                            body={"u": vs.shoulder_u, "v": vs.shoulder_v, "w": vs.shoulder_w},
                            debug={"gyro": round(self.paddle.gyro_mag), "threshold": round(self.paddle.threshold),
                                   "fps": round(vs.fps, 1), "paddle_connected": self.paddle.connected},
                            mqtt_ok=self.mqtt is not None)
                broadcast(self.clients, json.dumps(snap))
                if now >= next_video:
                    next_video = now + 1 / VIDEO_HZ
                    self._send_video()

            if now >= next_beat:
                next_beat = now + HEARTBEAT_S
                self._publish_record(force=True)

            await asyncio.sleep(max(0.0, period - (time.monotonic() - now)))

    def _send_video(self):
        frame_id, jpeg, cutout = self.vision.latest_images()
        if frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id
        # select screen: full camera view (you + your AprilTag); in game: just you
        if self.game.state == gl.SELECT and jpeg:
            broadcast(self.clients, b"F" + jpeg)
        elif cutout:
            broadcast(self.clients, b"C" + cutout)

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
    args = ap.parse_args()

    paddle = KeyboardPaddle() if args.sim else MotorPaddle(PADDLE_CARD_SERIAL, PADDLE_CARD_COLOR)
    cap = vision = mqtt = httpd = app = None
    try:
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
        app = App(vision, paddle, mqtt, args.sim)
        webbrowser.open(f"http://localhost:{HTTP_PORT}/")
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nCtrl-C -- shutting down.")
    finally:
        paddle.close()                          # stops both motors, then disconnects
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


if __name__ == "__main__":
    main()
