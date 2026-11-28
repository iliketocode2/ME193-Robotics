"""jim: an UNO Q that takes plain-English commands over MQTT.

  ws://test.mosquitto.org:8080
  ME193/q/will/cmd      commands in (plain English, or /status /stop /help /planner)
  ME193/q/will/reply    answers when a command doesn't name its own topic
  ME193/q/will/status   /status answers
  ME193/q/will/log      one JSON record per command (plan, results, timings)
  ME193/q/will/online   retained "online" / "offline" (MQTT last will)

A command goes into a queue. The worker thread turns it into a plan
(planner.py: LLM with a JSON grammar, or regex rules if the LLM server is
down or PLANNER_MODE = "rules"), checks it (planner.validate), and runs it
(skills.py). Slash commands are answered straight from the MQTT thread so
/status and /stop work even while the model is busy.
"""

import json
import queue
import threading
import time
import uuid

import numpy as np
import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge, Frame

import matrix
import planner
from imu import Imu
from skills import Cancelled, Executor, describe

# --- configuration -------------------------------------------------------------
NAME = "will"
BOARD = "jim"
BROKER = "test.mosquitto.org"
PORT = 8080                                   # MQTT over websockets
TOPIC_CMD = f"ME193/q/{NAME}/cmd"
TOPIC_REPLY = f"ME193/q/{NAME}/reply"
TOPIC_STATUS = f"ME193/q/{NAME}/status"
TOPIC_LOG = f"ME193/q/{NAME}/log"
TOPIC_ONLINE = f"ME193/q/{NAME}/online"
PLANNER_MODE = "llm"          # "llm" (falls back to rules if the server is down) or "rules"
MAX_COMMAND_CHARS = 500
MAX_QUEUE = 5
FPS = 15                      # LED matrix refresh rate while scrolling

START = time.monotonic()

# --- LED matrix ------------------------------------------------------------------


class Display:
    """Holds what the matrix should show; loop() pushes frames to the sketch."""

    def __init__(self):
        self._lock = threading.Lock()
        self._scene = None            # TextScene, an 8x13 array, or None (blank)
        self._t0 = time.monotonic()
        self._last_sent = None
        self.description = "blank"

    def show(self, scene):
        with self._lock:
            self._scene = scene
            self._t0 = time.monotonic()
            self.description = (f"text {scene.text!r}" if isinstance(scene, matrix.TextScene)
                                else "icon")

    def clear(self):
        with self._lock:
            self._scene = None
            self.description = "blank"

    def frame(self):
        with self._lock:
            scene, t = self._scene, time.monotonic() - self._t0
        if scene is None:
            return matrix.blank()
        if isinstance(scene, matrix.TextScene):
            return scene.frame(t)
        return scene

    def push(self):
        f = self.frame()
        if self._last_sent is not None and np.array_equal(f, self._last_sent):
            return
        Bridge.call("draw", Frame(f).to_board_bytes())
        self._last_sent = f.copy()


display = Display()
imu = Imu()
Bridge.provide("imu_batch", imu.on_batch)
Bridge.provide("imu_status", imu.on_status)

# --- MQTT ------------------------------------------------------------------------

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                     client_id=f"{BOARD}-{uuid.uuid4().hex[:8]}",
                     transport="websockets")
client.will_set(TOPIC_ONLINE, "offline", qos=1, retain=True)
client.reconnect_delay_set(min_delay=1, max_delay=30)
mqtt_connected = threading.Event()


def publish(topic, text, qos=1):
    info = client.publish(topic, text, qos=qos)
    print(f"[mqtt] -> {topic}: {text}", flush=True)
    return info


# --- planner + worker ------------------------------------------------------------

llm = planner.LLMPlanner()
rules = planner.RulePlanner()
mode = PLANNER_MODE
commands = queue.Queue(maxsize=MAX_QUEUE)   # items: (epoch when received, text)
# /stop bumps the epoch and sets the current cancel event, then swaps in a
# fresh one. Anything tagged with an older epoch is dropped or abandoned.
stop_lock = threading.Lock()
epoch = 0
cancel_event = threading.Event()
state = {"doing": "idle", "since": time.monotonic(), "last": "nothing yet", "done": 0}
state_lock = threading.Lock()
FORBIDDEN_TOPICS = {TOPIC_CMD, TOPIC_STATUS, TOPIC_LOG, TOPIC_ONLINE}

executor = Executor(display, imu, publish, lambda: status_text(), TOPIC_REPLY)


def set_doing(text):
    with state_lock:
        state["doing"] = text
        state["since"] = time.monotonic()


def ask_llm(command, cancel):
    """llm.plan() in a helper thread so /stop doesn't have to wait for the
    model to finish (llama-server finishes that request on its own)."""
    box = {}

    def call():
        try:
            box["result"] = llm.plan(command)
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=call, daemon=True)
    t.start()
    while t.is_alive():
        if cancel.wait(0.1):
            raise Cancelled()
    if "error" in box:
        raise box["error"]
    return box["result"]


def make_plan(command, cancel):
    """Returns (steps, planner_name, info)."""
    if mode == "llm":
        if llm_warm.is_set():
            set_doing(f"thinking about {command!r} with {llm.model}")
            info = {}
            try:
                plan, info = ask_llm(command, cancel)
                return planner.validate(plan, FORBIDDEN_TOPICS), llm.name, info
            except planner.PlanError as e:
                # A plan jim can't carry out (unknown icon, 5-minute wait...): say why.
                return [{"skill": "refuse", "reason": str(e)}], llm.name, info
            except OSError as e:
                print(f"[llm] request failed: {e}; using rules", flush=True)
        else:
            print("[llm] not ready (offline or warming up); using rules", flush=True)
    plan, info = rules.plan(command)
    try:
        steps = planner.validate(plan, FORBIDDEN_TOPICS)
    except planner.PlanError as e:
        steps = [{"skill": "refuse", "reason": str(e)}]
    if mode == "llm" and steps[0]["skill"] == "refuse" and llm.healthy():
        steps = [{"skill": "refuse", "reason": "my language model is still warming up; "
                  "try again in a minute (simple 'type X on the matrix' commands work now)."}]
    return steps, rules.name, info


llm_warm = threading.Event()


def llm_watch():
    """Whenever llama-server (re)appears, send it one throwaway command so
    the long, fixed system prompt is processed and cached before a real
    command arrives (that first pass takes ~0.5-2 min on the A53 cores)."""
    was_up = False
    while True:
        up = llm.healthy()   # never raises
        if up and not was_up:
            print(f"[llm] {llm.model} is up; warming the prompt cache", flush=True)
            t0 = time.monotonic()
            try:
                # The first pass over the whole prompt can take minutes on
                # these cores, far longer than a normal command's timeout.
                llm.plan("clear the screen", timeout=600)
                llm_warm.set()
                print(f"[llm] warm after {time.monotonic() - t0:.0f}s", flush=True)
            except Exception as e:
                print(f"[llm] warm-up failed: {e}", flush=True)
                up = False
        if not up:
            llm_warm.clear()
            llm.model = None          # the model may be different when it comes back
        was_up = up
        time.sleep(10)


def worker():
    while True:
        my_epoch, command = commands.get()
        with stop_lock:
            if my_epoch != epoch:
                continue          # arrived before a /stop: dropped
            cancel = cancel_event  # /stop will set exactly this event
        t0 = time.monotonic()
        record = {"command": command}
        try:
            steps, used, info = make_plan(command, cancel)
            record.update(planner=used, plan=steps,
                          plan_ms=round(info.get("ms", 0)),
                          **{k: v for k, v in info.items() if k not in ("ms", "raw")})
            print(f"[plan] {used}: {[describe(s) for s in steps]}", flush=True)
            set_doing(f"running {command!r}")
            record["results"] = executor.run(steps, cancel)
            record["ok"] = True
        except Cancelled:
            record["results"] = ["stopped by /stop"]
            record["ok"] = False
        except Exception as e:  # never let one bad command kill the worker
            record["results"] = [f"error: {e}"]
            record["ok"] = False
            publish(TOPIC_REPLY, f"Something went wrong running {command!r}: {e}")
        record["total_ms"] = round((time.monotonic() - t0) * 1000)
        publish(TOPIC_LOG, json.dumps(record))
        with state_lock:
            state["last"] = f"{command!r} -> {'; '.join(record['results'])} ({record['total_ms']} ms)"
            state["done"] += 1
        set_doing("idle")


# --- slash commands --------------------------------------------------------------


def uptime():
    s = int(time.monotonic() - START)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def status_text():
    with state_lock:
        doing, since, last, done = state["doing"], state["since"], state["last"], state["done"]
    if doing != "idle":
        doing = f"{doing} ({time.monotonic() - since:.1f}s so far"
        if executor.current != "idle":
            doing += f", {executor.current}"
        doing += ")"
    if not llm.healthy():
        llm_state = "offline, using rules"
    elif not llm_warm.is_set():
        llm_state = f"{llm.model} warming up"
    else:
        llm_state = f"{llm.model} ready"
    return (f"{BOARD} is running (up {uptime()}). Now: {doing}. "
            f"Queue: {commands.qsize()}. Commands done: {done}. Last: {last}. "
            f"Planner: {mode} (LLM {llm_state}). Display: {display.description}. "
            f"{imu.describe()}.")


HELP = (f"Send plain English to {TOPIC_CMD}, e.g. 'Type \"Hi\" on the LED matrix and then send back "
        f"on MQTT on topic {NAME} message \"I am done\"' or 'what is your yaw? reply on topic {NAME}'. "
        f"Skills: {', '.join(planner.SKILLS)}. Live values: "
        f"{', '.join('{' + p + '}' for p in planner.PLACEHOLDERS)}. "
        f"Slash commands: /status, /stop, /help, /planner llm|rules. "
        f"Replies go to {TOPIC_REPLY} unless you name a topic; /status goes to {TOPIC_STATUS}.")


def stop_everything():
    global epoch, cancel_event
    with stop_lock:
        epoch += 1
        cancel_event.set()
        cancel_event = threading.Event()
    dropped = 0
    while True:
        try:
            commands.get_nowait()
            dropped += 1
        except queue.Empty:
            break
    display.clear()
    return dropped


def handle_slash(text):
    global mode
    word, _, arg = text.partition(" ")
    word = word.lower()
    if word == "/status":
        publish(TOPIC_STATUS, status_text())
    elif word == "/stop":
        dropped = stop_everything()
        publish(TOPIC_STATUS, f"Stopped. Cleared the display and dropped {dropped} queued command(s).")
    elif word == "/help":
        publish(TOPIC_STATUS, HELP)
    elif word == "/planner" and arg.strip().lower() in ("llm", "rules"):
        mode = arg.strip().lower()
        publish(TOPIC_STATUS, f"Planner is now {mode}.")
    else:
        publish(TOPIC_STATUS, f"Unknown command {word}. Try /help.")


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[mqtt] connected ({reason_code}); listening on {TOPIC_CMD}", flush=True)
    client.subscribe(TOPIC_CMD, qos=0)
    client.publish(TOPIC_ONLINE, "online", qos=1, retain=True)
    mqtt_connected.set()


def on_disconnect(client, userdata, flags, reason_code, properties):
    mqtt_connected.clear()
    print(f"[mqtt] disconnected ({reason_code})", flush=True)


def on_message(client, userdata, msg):
    # An exception escaping a paho callback ends paho's network thread for
    # good, and /status and /stop with it. Nothing gets out of here.
    try:
        handle_message(msg)
    except Exception as e:
        print(f"[mqtt] error handling {msg.topic}: {e!r}", flush=True)
        try:
            publish(TOPIC_STATUS, f"Error handling that message: {e}")
        except Exception:
            pass


def handle_message(msg):
    if msg.retain:
        return  # a stale command someone left on the broker, not a new one
    text = msg.payload.decode("utf-8", errors="replace").strip()
    print(f"[mqtt] <- {msg.topic}: {text}", flush=True)
    if not text:
        return
    if text.startswith("/"):
        handle_slash(text)
        return
    if len(text) > MAX_COMMAND_CHARS:
        publish(TOPIC_REPLY, f"That command is too long ({len(text)} characters, max {MAX_COMMAND_CHARS}).")
        return
    try:
        commands.put_nowait((epoch, text))
    except queue.Full:
        publish(TOPIC_REPLY, f"I'm busy with {commands.qsize()} commands already; try again soon.")


client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.connect_async(BROKER, PORT, keepalive=30)
client.loop_start()
threading.Thread(target=worker, daemon=True, name="worker").start()
threading.Thread(target=llm_watch, daemon=True, name="llm_watch").start()

# Boot sign: a check mark until the first command arrives.
display.show(matrix.icon_frame("check"))
print(f"[{BOARD}] started; planner mode {mode}; LLM at {llm.url} "
      f"({'up' if llm.healthy() else 'not reachable yet'})", flush=True)


def loop():
    display.push()
    time.sleep(1.0 / FPS)


App.run(user_loop=loop)
