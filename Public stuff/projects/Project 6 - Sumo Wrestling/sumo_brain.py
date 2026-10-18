"""
sumo_brain.py -- the laptop half of the sumo robot. Run this one.

  phone camera --HTTP--> camera thread --> vision thread (vision.py)
                                                  |
  MQTT (T/sensor, T/status, ME193/Rogers) --------+--> behavior tree tick (15 Hz)
                                                  |        (behavior_tree.py)
  web UI (browser) <--websocket-- state + video --+--> MQTT T/cmd --> robot relay
                   --websocket--> E-stop, start/pause, manual keys, click target,
                                  strategy slider, HSV calibration

Run (from the repo root, robot relay already running on the UNO Q):
    my_env\\Scripts\\python "Public stuff/projects/Project 6 - Sumo Wrestling/sumo_brain.py"
then the dashboard opens at http://localhost:8293/ . Ctrl-C to quit; the
robot stops by itself within CMD_TIMEOUT_S once commands stop arriving.

Threads: camera, vision, paho (MQTT callbacks) and http each run on their
own thread and only write into Brain under self.lock; everything that
decides or sends happens in the asyncio tick loop.
"""

import asyncio
import collections
import functools
import http.server
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import webbrowser

import cv2
import paho.mqtt.client as mqtt
from websockets.asyncio.server import broadcast, serve

import behavior_tree as bt
import songs_bridge
from config import (BROKER_HOST, BROKER_PORT, CAMERA_STALE_S, FRAME_ROTATION, HERE,
                    HTTP_PORT, PHONE_STREAM_URL, PROCESS_WIDTH, ROBOT_STALE_S, SENSOR_STALE_S,
                    TEAM, TICK_HZ, TOPIC_CMD, TOPIC_REFEREE, TOPIC_ROOT, TOPIC_SENSOR,
                    TOPIC_STATUS, TOPIC_UNOQ, UNOQ_STALE_S, VIDEO_JPEG_QUALITY, VIDEO_MAX_FPS,
                    WS_PORT)
from vision import Vision

WEB_DIR = os.path.join(HERE, "web")
TOPIC_CHAT = f"{TOPIC_ROOT}/chat"
TOPIC_RESULT = f"{TOPIC_ROOT}/result"
HIGH_RATE = {TOPIC_SENSOR, TOPIC_CMD, TOPIC_STATUS, TOPIC_UNOQ}
TAG_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")       # "[ME193/Rogers] start" -> "start" (Project 3)
MANUAL_FRESH_S = 0.35
TARGET_TTL_S = 8.0
BLUE_DEBOUNCE = 2                                   # consecutive sensor samples to flip on/off blue
BLUE_RECENT_S = 3.0
OPP_WAS_INSIDE_S = 3.0                              # opponent must have been in the ring this recently to "win"


class Brain:
    def __init__(self):
        self.lock = threading.Lock()
        self.vision = Vision()
        self.tree = bt.build_tree()
        self.bb = bt.make_blackboard()
        # camera / vision
        self.frame, self.frame_t = None, 0.0
        self.result, self.result_t = None, 0.0
        self.jpeg, self.jpeg_seq, self.sent_seq = None, 0, 0
        self.cam_fps, self.cam_status = 0.0, "starting"
        # robot links
        self.mqtt_ok = False
        self.status, self.status_t = {}, 0.0
        self.unoq, self.unoq_t = {}, 0.0          # fan RPM report from the UNO Q
        self.sensor, self.sensor_t = {}, 0.0
        self.blue_run = self.clear_run = 0
        self.rear_on_blue, self.last_blue_t = False, 0.0
        # teleop + match
        self.match = "idle"
        self.estop = False
        self.dry_run = False
        self.strategy = 0.5
        self.manual, self.manual_t = None, 0.0
        self.target, self.target_t = None, 0.0
        self.our_out_s = self.opp_out_s = 0.0
        self.timed_result_t = 0.0                  # vision result the out-timers last counted
        self.opp_inside_t = -1e9                   # last time the opponent was seen inside the ring
        self.last_out = (0, 0)
        self.seq = 0
        self.chat = collections.deque(maxlen=200)
        self.chat_last = {}
        self.chat_id = self.chat_sent_id = 0
        self.clients = set()
        self.stop = threading.Event()
        # MQTT
        self.mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mq.on_connect = self._on_connect
        self.mq.on_disconnect = self._on_disconnect
        self.mq.on_message = self._on_message
        self.mq.suppress_exceptions = True       # a bad payload must never kill paho's thread
        self.mq.reconnect_delay_set(min_delay=1, max_delay=10)

    # ================================================================ MQTT
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(f"{TOPIC_ROOT}/#", qos=0)     # (re)subscribe after every reconnect
        client.subscribe(TOPIC_REFEREE, qos=0)
        with self.lock:
            self.mqtt_ok = True
        self._chat("system", f"MQTT connected to {BROKER_HOST}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        with self.lock:
            self.mqtt_ok = False
        self._chat("system", "MQTT disconnected - retrying")

    def _on_message(self, client, userdata, msg):
        text = msg.payload.decode("utf-8", errors="replace").strip()
        now = time.monotonic()
        if msg.topic == TOPIC_SENSOR:
            try:
                data = json.loads(text)
            except ValueError:
                return
            if not isinstance(data, dict):
                return
            with self.lock:
                self.sensor, self.sensor_t = data, now
                if data.get("name") == "blue":
                    self.blue_run, self.clear_run = self.blue_run + 1, 0
                    self.last_blue_t = now
                else:
                    self.blue_run, self.clear_run = 0, self.clear_run + 1
                if self.blue_run >= BLUE_DEBOUNCE:
                    self.rear_on_blue = True
                elif self.clear_run >= BLUE_DEBOUNCE:
                    self.rear_on_blue = False
        elif msg.topic == TOPIC_STATUS:
            try:
                data = json.loads(text)
            except ValueError:
                return
            if not isinstance(data, dict):
                return
            with self.lock:
                self.status, self.status_t = data, now
        elif msg.topic == TOPIC_UNOQ:
            try:
                data = json.loads(text)
            except ValueError:
                return
            if not isinstance(data, dict):
                return
            with self.lock:
                self.unoq, self.unoq_t = data, now
        elif msg.topic == TOPIC_REFEREE:
            if TAG_PREFIX.sub("", text).lower() == "start":
                self.command({"cmd": "start", "source": "referee"})
        self._chat(msg.topic, text)

    def _chat(self, topic, text):
        now = time.monotonic()
        hi = topic in HIGH_RATE
        with self.lock:
            if hi and now - self.chat_last.get(topic, 0) < 1.0:
                return                                   # 20 Hz sensor spam -> 1 line/s in the feed
            self.chat_last[topic] = now
            self.chat_id += 1
            self.chat.append({"type": "chat", "id": self.chat_id, "topic": topic, "text": text[:400],
                              "ts": time.time(), "hi": hi})

    def publish(self, topic, text):
        self.mq.publish(topic, text, qos=0, retain=False)

    # ================================================================ camera + vision threads
    def _open_camera(self):
        if PHONE_STREAM_URL is None:
            sys.path.insert(0, os.path.join(HERE, "..", "..", "useful libraries"))
            from camlib import pick_camera
            cap, _ = pick_camera()
            return cap
        cap = cv2.VideoCapture(PHONE_STREAM_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def camera_loop(self):
        while not self.stop.is_set():
            self.cam_status = "connecting"
            cap = self._open_camera()
            if not cap.isOpened():
                self.cam_status = f"can't open {PHONE_STREAM_URL}"
                time.sleep(2)
                continue
            self.cam_status = "live"
            while not self.stop.is_set():
                ok, frame = cap.read()             # always read, so we hold only the newest frame
                if not ok:
                    self.cam_status = "stream dropped - reconnecting"
                    break
                if FRAME_ROTATION is not None:
                    frame = cv2.rotate(frame, FRAME_ROTATION)
                with self.lock:
                    self.frame, self.frame_t = frame, time.monotonic()
            cap.release()
            time.sleep(1)

    def vision_loop(self):
        last_t, last_jpeg = 0.0, 0.0
        while not self.stop.is_set():
            with self.lock:
                frame, t = self.frame, self.frame_t
            if frame is None or t == last_t:
                time.sleep(0.005)
                continue
            last_t = t
            h, w = frame.shape[:2]
            if w > PROCESS_WIDTH:
                frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * PROCESS_WIDTH / w)), interpolation=cv2.INTER_AREA)
            try:
                result = self.vision.process(frame)
            except Exception as e:                 # never let one bad frame kill the thread
                print(f"[vision] {type(e).__name__}: {e}")
                continue
            result["size"] = [frame.shape[1], frame.shape[0]]
            now = time.monotonic()
            with self.lock:
                dt = now - self.result_t if self.result_t else 0
                self.cam_fps = 0.8 * self.cam_fps + 0.2 * (1 / dt) if dt > 0 else self.cam_fps
                self.result, self.result_t = result, now
            if now - last_jpeg >= 1.0 / VIDEO_MAX_FPS:
                last_jpeg = now
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, VIDEO_JPEG_QUALITY])
                if ok:
                    with self.lock:
                        self.jpeg, self.jpeg_seq = buf.tobytes(), self.jpeg_seq + 1

    # ================================================================ commands from the UI / referee
    def command(self, msg):
        cmd = msg.get("cmd")
        play_reset = False
        with self.lock:
            if cmd == "estop":
                self.estop = bool(msg.get("on", not self.estop))
            elif cmd == "start":
                if self.match != "live":
                    self.match = "live"
                    self.our_out_s = self.opp_out_s = 0.0
                    self.opp_inside_t = -1e9
                    self.vision.reset_opponent_track()
                    self.vision.lock_ring(True)
                    play_reset = True
            elif cmd == "pause":
                if self.match == "live":
                    self.match = "idle"
                    self.vision.lock_ring(False)
            elif cmd == "reset":
                self.match, self.target = "idle", None
                self.vision.lock_ring(False)
                play_reset = True
            elif cmd == "strategy":
                self.strategy = max(0.0, min(1.0, float(msg.get("value", 0.5))))
            elif cmd == "manual":
                if msg.get("release"):
                    self.manual = None
                else:
                    self.manual = (int(msg.get("l", 0)), int(msg.get("r", 0)))
                    self.manual_t = time.monotonic()
            elif cmd == "target":
                ring = self.result["ring"] if self.result else None
                if ring is not None:
                    self.target, self.target_t = ring.to_ring(float(msg["x"]), float(msg["y"])), time.monotonic()
            elif cmd == "clear_target":
                self.target = None
            elif cmd == "dry_run":
                self.dry_run = bool(msg.get("on"))
            elif cmd == "hsv":
                self.vision.set_hsv(msg["low"], msg["high"])
            elif cmd == "refit_ring":
                self.vision.reset_ring()
            elif cmd == "say":
                text = str(msg.get("text", ""))[:200]
                if text:
                    self.publish(TOPIC_CHAT, text)
        if play_reset:
            songs_bridge.stop()
        if cmd == "start":
            self._chat("system", f"MATCH START ({msg.get('source', 'ui')})")
        if cmd == "test_song" and msg.get("song") in ("win", "lose"):
            songs_bridge.play(msg["song"], self.publish)

    # ================================================================ the decision tick
    def _fill_blackboard(self, now, dt):
        bb = self.bb
        res = self.result if now - self.result_t < CAMERA_STALE_S else None
        ring = res["ring"] if res else None
        bb.now, bb.match, bb.estop, bb.dry_run, bb.strategy = now, self.match, self.estop, self.dry_run, self.strategy
        bb.robot_ok = now - self.status_t < ROBOT_STALE_S and bool(self.status.get("motor"))
        bb.manual = self.manual if self.manual and now - self.manual_t < MANUAL_FRESH_S else None
        bb.camera_ok = ring is not None
        bb.sensor_ok = now - self.sensor_t < SENSOR_STALE_S and self.sensor.get("color") is not None
        bb.rear_on_blue = self.rear_on_blue
        bb.blue_recent = now - self.last_blue_t < BLUE_RECENT_S
        bb.sensor_name = self.sensor.get("name", "?")
        if self.target and now - self.target_t > TARGET_TTL_S:
            self.target = None
        bb.target = self.target
        bb.us = bb.opp = None
        if ring is not None and res["us"]:
            u = res["us"]
            pos = ring.to_ring(u["x"], u["y"])
            bb.us = {"pos": pos, "r": math.hypot(*pos), "heading": ring.dir_to_ring(u["hx"], u["hy"])}
        if ring is not None and res["opp"]:
            pos = ring.to_ring(res["opp"]["x"], res["opp"]["y"])
            bb.opp = {"pos": pos, "r": math.hypot(*pos)}
        # "Out for X seconds" timers advance only on NEW camera frames, by real
        # frame time -- a reused frame must not count twice. The opponent only
        # counts as pushed out if it was actually inside the ring recently, so
        # a robot waiting beside the ring (or a hand) can't hand us a fake win.
        if res and self.result_t != self.timed_result_t:
            frame_dt = min(self.result_t - self.timed_result_t, CAMERA_STALE_S) if self.timed_result_t else 0.0
            self.timed_result_t = self.result_t
            self.our_out_s = self.our_out_s + frame_dt if bb.us and bb.us["r"] > bt.OUT_R else 0.0
            if bb.opp and bb.opp["r"] < 1.0:
                self.opp_inside_t = now
            was_in = now - self.opp_inside_t < OPP_WAS_INSIDE_S
            self.opp_out_s = (self.opp_out_s + frame_dt
                              if bb.opp and bb.opp["r"] > bt.OPP_OUT_R and was_in else 0.0)
        elif not res:
            self.our_out_s = self.opp_out_s = 0.0
        bb.our_out_s, bb.opp_out_s = self.our_out_s, self.opp_out_s
        return res, ring

    def tick(self, now, dt):
        with self.lock:
            res, ring = self._fill_blackboard(now, dt)
            statuses = bt.tick(self.tree, self.bb)
            bb = self.bb
            if bb.target is None:
                self.target = None                    # tree consumed the target ("reached")
            event = bb.event
            if event:
                self.match = "lost" if event == "lose" else "won"
            out = (0, 0) if self.dry_run else bb.out
            self.seq += 1
            cmd = {"seq": self.seq, "l": out[0], "r": out[1], "mode": bb.action[:40]}
            snap = self._snapshot(res, ring, statuses, out, now)
        self.publish(TOPIC_CMD, json.dumps(cmd))
        if event:
            songs_bridge.play("lose" if event == "lose" else "win", self.publish)
            self.publish(TOPIC_RESULT, json.dumps({"event": event, "team": TEAM}))
            self._chat("system", "WE'RE OUT - sad song" if event == "lose" else "OPPONENT OUT - we win!")
        return snap

    def _snapshot(self, res, ring, statuses, out, now):
        bb = self.bb
        sp = bt.strategy_params(self.strategy)

        def px(p):
            return list(ring.to_pixel(*p)) if (ring is not None and p is not None) else None

        world = {"ring": ring.as_dict() if ring is not None else None,
                 "ring_locked": self.vision.ring_locked,
                 "edge": sp["edge"], "out_r": bt.OUT_R,
                 "us": None, "opp": None, "robots": [],
                 "goal_px": px(bb.goal), "target_px": px(self.target)}
        if res:
            world["robots"] = [{"box": r["box"], "conf": round(r["conf"], 2), "ours": bool(r.get("ours"))}
                               for r in res["robots"]]
            if res["us"]:
                world["us"] = dict(res["us"], r=round(bb.us["r"], 3) if bb.us else None)
            if res["opp"]:
                world["opp"] = {"box": res["opp"]["box"], "x": res["opp"]["x"], "y": res["opp"]["y"],
                                "r": round(bb.opp["r"], 3) if bb.opp else None}
        sensor_age = now - self.sensor_t if self.sensor_t else None
        return {
            "type": "state", "match": self.match, "action": bb.action, "bt": statuses,
            "out": list(bb.out), "sent": list(out),
            "fan_rpm": self.unoq.get("fan_rpm") if now - self.unoq_t < UNOQ_STALE_S else None,
            "estop": self.estop, "dry_run": self.dry_run,
            "strategy": self.strategy, "strategy_params": sp, "manual": bb.manual is not None,
            "links": {"mqtt": self.mqtt_ok, "robot": bb.robot_ok, "motor": bool(self.status.get("motor")),
                      "color": bool(self.status.get("color")), "relay_host": self.status.get("host"),
                      "camera": bb.camera_ok, "cam_status": self.cam_status,
                      "fps": round(self.cam_fps, 1), "vision_ms": round(res["ms"], 1) if res else None,
                      "mode": self.vision.mode},
            "sensor": dict(self.sensor, on_blue=self.rear_on_blue, ok=bb.sensor_ok,
                           age=round(sensor_age, 2) if sensor_age is not None else None),
            "world": world, "frame": res["size"] if res else None,
            "hsv": dict(zip(("low", "high"), self.vision.get_hsv())),
        }

    # ================================================================ websocket server
    async def handler(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps({"type": "hello", "team": TEAM, "topic_root": TOPIC_ROOT,
                                  "referee": TOPIC_REFEREE, "tree": bt.branch_names(self.tree),
                                  "chat": list(self.chat)}))
        try:
            async for raw in ws:
                try:
                    self.command(json.loads(raw))
                except (ValueError, KeyError, TypeError) as e:
                    print(f"[ui] bad command {raw!r}: {e}")
        except Exception as e:
            print(f"[ui] browser disconnected: {e}")
        finally:
            self.clients.discard(ws)

    async def tick_loop(self):
        period = 1.0 / TICK_HZ
        last = time.monotonic()
        while True:
            now = time.monotonic()
            try:
                snap = self.tick(now, now - last)
            except Exception:
                # keep serving the UI; the robot gets an explicit stop and its
                # watchdog would stop it anyway if we stayed broken
                traceback.print_exc()
                self.publish(TOPIC_CMD, json.dumps({"seq": -1, "l": 0, "r": 0, "mode": "brain error"}))
                await asyncio.sleep(0.5)
                continue
            last = now
            with self.lock:
                chat = [c for c in self.chat if c["id"] > self.chat_sent_id]
                self.chat_sent_id = self.chat_id
                jpeg = self.jpeg if self.jpeg_seq != self.sent_seq else None
                self.sent_seq = self.jpeg_seq
            if self.clients:
                broadcast(self.clients, json.dumps(snap))
                for c in chat:
                    broadcast(self.clients, json.dumps(c))
                if jpeg:
                    broadcast(self.clients, b"F" + jpeg)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - now)))

    async def run(self):
        async with serve(self.handler, "127.0.0.1", WS_PORT):
            await self.tick_loop()

    def shutdown(self):
        self.stop.set()
        for _ in range(3):                         # belt and braces; relay watchdog stops it anyway
            self.publish(TOPIC_CMD, json.dumps({"seq": -1, "l": 0, "r": 0, "mode": "shutdown"}))
            time.sleep(0.05)
        songs_bridge.stop()
        self.mq.loop_stop()
        self.mq.disconnect()


def start_http():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), functools.partial(Quiet, directory=WEB_DIR))
    threading.Thread(target=httpd.serve_forever, name="http", daemon=True).start()
    return httpd


def main():
    brain = Brain()
    print(f"[brain] vision mode: {brain.vision.mode}")
    brain.mq.connect_async(BROKER_HOST, BROKER_PORT)
    brain.mq.loop_start()
    threading.Thread(target=brain.camera_loop, name="camera", daemon=True).start()
    threading.Thread(target=brain.vision_loop, name="vision", daemon=True).start()
    httpd = start_http()
    url = f"http://localhost:{HTTP_PORT}/"
    print(f"[brain] dashboard at {url}  (Ctrl-C to quit)")
    webbrowser.open(url)
    try:
        asyncio.run(brain.run())
    except KeyboardInterrupt:
        pass
    finally:
        brain.shutdown()
        httpd.shutdown()
        print("[brain] stopped")


if __name__ == "__main__":
    main()
