"""
world_cup.py -- phone-tone-controlled LEGO Education "car" for the ME193
World Cup match, coordinated with an opponent robot over MQTT on topic
ME193/Rogers.

Play fixed pitches from a phone tone-generator app into the laptop mic
(see tone_policy.py for the exact frequencies):
    - DRIVE computer (connected to the robot): forward / left / right,
      plus a held "goal" tone. No tone = stop.
    - SHIELD co-pilot computer (no robot connection): "up" / "down" for
      the Single Motor shield, relayed to the drive computer over MQTT.
      The shield tones are all higher than the drive tones, and each
      computer only listens inside its own range, so the two phones never
      interfere.

See this project's README.md for the frequency table, the MQTT message
schema, and venv setup.

Run (from the my_env_audio venv -- NOT my_env, see README):
    my_env_audio/Scripts/python "Public stuff/projects/Project 3 - World Cup/world_cup.py"
"""

import json
import os
import sys
import threading
import time
import tkinter as tk
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation

import legoeducation as le

# lelib.py / mqttlib.py live in the shared "useful libraries" folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "useful libraries"))

from lelib import colorSensor, doubleMotor, singleMotor  # noqa: E402
from mqttlib import MQTTClient  # noqa: E402

from pyaudio_mic import pick_mic
from songs import DEATH_SONG, SUCCESS_SONG, play_both
from tone_policy import (BAND_HALF_WIDTH_HZ, BLOCK_SIZE, DRIVE_TONES, HOLD_S, SAMPLE_RATE,
                         SHIELD_TONES, TONAL_RATIO_GATE, ToneDetector)

# --- Match-day settings -- chosen in the setup dialog (run_setup_dialog()) ---
ROLE = None                # "ball" or "goalie"
TEAM_NAME = None           # must be typed identically on the co-pilot's computer
OPPONENT_TEAM_NAME = None  # blank = don't react to anyone's fail/goal messages
CONTROL_CHANNEL = None     # "drive" (connects to the robot) or "shield" (co-pilot, no hardware)
CONTROL_TOPIC = None       # f"{MQTT_TOPIC}/control/{TEAM_NAME}"
DEFAULT_TEAM_NAME = "Cucurella"

# --- Hardware -- fill in with your LEGO Connection Card values -----------------
CAR_CARD_SERIAL = 1126     # doubleMotor (drive)
CAR_CARD_COLOR = le.LEGO_COLOR_GREEN
SHIELD_CARD_SERIAL = 1126  # singleMotor (shield arm)
SHIELD_CARD_COLOR = le.LEGO_COLOR_GREEN
SENSOR_CARD_SERIAL = 1126  # colorSensor (front-facing)
SENSOR_CARD_COLOR = le.LEGO_COLOR_GREEN

# --- Drive tuning ---------------------------------------------------------------
# Flip whichever wheel drives the wrong way once you watch the car move.
INVERT_LEFT_MOTOR = True
INVERT_RIGHT_MOTOR = True
FORWARD_SPEED = 60  # % -- both wheels while the "forward" tone plays
TURN_SPEED = 40     # % -- wheels spin opposite ways (turn in place) for "left"/"right"
GOAL_HOLD_S = 0.5   # hold the "goal" tone this long to call a goal (ball only)

# --- Shield tuning ----------------------------------------------------------------
# The arm's position when the script connects counts as "down" (0). Start
# every match with the arm DOWN, out of the sensor's way. "up" rotates it
# SHIELD_UP_DEGREES from there -- make this negative if it swings the wrong way.
SHIELD_UP_DEGREES = 90
SHIELD_SPEED = 100
SHIELD_RESEND_S = 1.0  # co-pilot re-sends its shield state this often, in case an MQTT message is lost

# --- Proximity ("caught") -----------------------------------------------------------
# reflection() reads roughly 0-100 on real hardware. Calibrate on-site
# against the opponent robot and room lighting.
PROXIMITY_REFLECTION_THRESHOLD = 70

MQTT_TOPIC = "ME193/Rogers"
CONTROL_LOOP_PERIOD_S = 0.05
# If the mic stops delivering audio (device unplugged, laptop asleep, a
# crash in the callback), treat the last reading as "no tone" after this
# long, so the car can't keep driving on a frozen "forward".
STALE_READING_S = HOLD_S + 0.2

# Wheel speeds (left, right) for each drive command, before INVERT_* flags.
# None = no tone = stop. "goal" also holds still while it's being held.
DRIVE_COMMANDS = {
    None: (0, 0),
    "forward": (FORWARD_SPEED, FORWARD_SPEED),
    "left": (-TURN_SPEED, TURN_SPEED),
    "right": (TURN_SPEED, -TURN_SPEED),
    "goal": (0, 0),
}

# --- State shared between the audio callback, MQTT thread, control loop and
# the dashboard. Only touched under state_lock. -------------------------------------
state_lock = threading.Lock()
shared = {
    "reading": None,             # latest tone_policy.ToneReading
    "reading_time": 0.0,         # time.monotonic() when it arrived
    "game_active": False,        # True once "start" arrives (or 's' is pressed)
    "game_over": False,
    "game_over_reason": None,
    "remote_result_pending": None,  # (song, message) queued by on_mqtt_message
    "shield_target": "down",     # drive: latest from co-pilot. co-pilot: what it's sending.
    "last_drive": (0, 0),        # last (left, right) actually sent to the car
    "last_proximity": None,
    "mqtt_log": deque(maxlen=10),
}

detector = None  # ToneDetector, created in main() once we know which computer this is
car = None
shield = None
sensor = None
mqtt = None


def _log_mqtt(direction, payload):
    with state_lock:
        shared["mqtt_log"].append({"t": time.time(), "dir": direction, "payload": payload})


def _publish(payload_dict):
    text = json.dumps(payload_dict)
    mqtt.publish(MQTT_TOPIC, text)
    _log_mqtt("sent", text)


def drive_wheels(command):
    left, right = DRIVE_COMMANDS[command]
    if INVERT_LEFT_MOTOR:
        left = -left
    if INVERT_RIGHT_MOTOR:
        right = -right
    return left, right


def announce_result(song, message):
    """Stop the car and play the song. Control-loop thread only (BLE calls)."""
    with state_lock:
        shared["game_over"] = True
        shared["game_active"] = False
        shared["game_over_reason"] = message
        shared["last_drive"] = (0, 0)
    print(f"[GAME] {message}")
    try:
        car.movement_stop()  # not lelib's car.stop(), which only stops the left motor
    except Exception:
        pass
    play_both(car, song)


# --- MQTT ------------------------------------------------------------------------------

def on_mqtt_message(topic, payload):
    """Game topic. On the drive computer this can end the match; on the
    co-pilot it only updates the dashboard."""
    text = payload.strip()
    _log_mqtt("recv", text)

    if text.lower() == "start":
        with state_lock:
            if not shared["game_over"]:
                shared["game_active"] = True
        print("[MQTT] Match started!")
        return

    try:
        data = json.loads(text)
    except ValueError:
        return
    if not isinstance(data, dict):
        return
    event, team = data.get("event"), data.get("team")
    if event not in ("fail", "goal"):
        return

    if CONTROL_CHANNEL == "shield":
        with state_lock:
            shared["game_over"] = True
            shared["game_active"] = False
            shared["game_over_reason"] = f"{team}: {event}"
        return

    if team == TEAM_NAME:
        return  # our own publish echoed back by the broker
    if not OPPONENT_TEAM_NAME or team != OPPONENT_TEAM_NAME:
        return

    # Queue it -- the control loop does the BLE stop + song, not paho's thread.
    with state_lock:
        if event == "fail":
            shared["remote_result_pending"] = (SUCCESS_SONG, f"{OPPONENT_TEAM_NAME} got caught -- you win!")
        else:
            shared["remote_result_pending"] = (DEATH_SONG, f"{OPPONENT_TEAM_NAME} scored -- you lose!")


def on_control_message(topic, payload):
    """Drive computer: {"shield": "up"|"down"} from the co-pilot."""
    try:
        data = json.loads(payload)
    except ValueError:
        return
    target = data.get("shield") if isinstance(data, dict) else None
    if target in SHIELD_TONES:
        with state_lock:
            shared["shield_target"] = target


# --- Audio -------------------------------------------------------------------------------

def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio's real-time thread: DSP only, no BLE or MQTT calls here --
    legoeducation calls block on a BLE round-trip, which would drop audio."""
    reading = detector.update(np.frombuffer(in_data, dtype=np.float32))
    with state_lock:
        shared["reading"] = reading
        shared["reading_time"] = time.monotonic()
    return (None, pyaudio.paContinue)


# --- Control loops ---------------------------------------------------------------------

def run_drive_loop(stop_event):
    """Drive computer: the only thread that talks to the robot. Motor
    commands are only sent when they CHANGE, not re-sent every tick."""
    sent_drive = None
    sent_shield = None
    goal_started = None  # when the current "goal" tone began, or None
    goal_fired = False

    while not stop_event.is_set():
        with state_lock:
            reading = shared["reading"]
            if time.monotonic() - shared["reading_time"] > STALE_READING_S:
                reading = None
            live = shared["game_active"] and not shared["game_over"]
            remote_result = shared["remote_result_pending"]
            shared["remote_result_pending"] = None
            shield_target = shared["shield_target"]

        try:
            if remote_result is not None:
                announce_result(*remote_result)
                sent_drive = (0, 0)
                continue

            command = reading.tone if (live and reading is not None) else None
            wheels = drive_wheels(command)
            if wheels != sent_drive:
                if wheels == (0, 0):
                    car.movement_stop()
                else:
                    car.movement_move_tank(*wheels, blocking=False)
                sent_drive = wheels
                with state_lock:
                    shared["last_drive"] = wheels

            # Shield and sensor get their own try: a failure there shouldn't
            # stop the car and make a held drive tone stutter.
            reflection = None
            try:
                if live and shield_target != sent_shield:
                    position = SHIELD_UP_DEGREES if shield_target == "up" else 0
                    shield.motor_run_to_relative_position(position, speed=SHIELD_SPEED, blocking=False)
                    sent_shield = shield_target
                reflection = sensor.reflection()
                with state_lock:
                    shared["last_proximity"] = reflection
            except Exception as e:
                print(f"[control loop] Shield/sensor error: {e}")
                sent_shield = shield_target  # don't retry every tick

            if not live or ROLE != "ball":
                continue

            if command == "goal":
                goal_started = goal_started or time.monotonic()
                if time.monotonic() - goal_started >= GOAL_HOLD_S and not goal_fired:
                    goal_fired = True
                    _publish({"event": "goal", "team": TEAM_NAME})
                    announce_result(SUCCESS_SONG, "GOAL! You scored.")
                    sent_drive = (0, 0)
            else:
                goal_started, goal_fired = None, False
                if reflection is not None and reflection >= PROXIMITY_REFLECTION_THRESHOLD:
                    _publish({"event": "fail", "team": TEAM_NAME})
                    announce_result(DEATH_SONG, "Caught! You failed.")
                    sent_drive = (0, 0)
        except Exception as e:
            print(f"[control loop] Error, stopping car: {e}")
            try:
                car.movement_stop()
            except Exception:
                pass
            sent_drive = (0, 0)
        finally:
            stop_event.wait(CONTROL_LOOP_PERIOD_S)


def run_copilot_loop(stop_event):
    """Shield co-pilot: the last "up"/"down" tone heard sticks until the
    other one is played. Publishes on change, and every SHIELD_RESEND_S."""
    target = "down"
    last_sent_target, last_sent_time = None, 0.0

    while not stop_event.is_set():
        with state_lock:
            reading = shared["reading"]
            if time.monotonic() - shared["reading_time"] > STALE_READING_S:
                reading = None
        if reading is not None and reading.tone in SHIELD_TONES:
            target = reading.tone

        now = time.monotonic()
        if target != last_sent_target or now - last_sent_time >= SHIELD_RESEND_S:
            mqtt.publish(CONTROL_TOPIC, json.dumps({"shield": target}))
            if target != last_sent_target:
                _log_mqtt("sent", f"shield -> {target}  ({CONTROL_TOPIC})")
            last_sent_target, last_sent_time = target, now
            with state_lock:
                shared["shield_target"] = target

        stop_event.wait(CONTROL_LOOP_PERIOD_S)


# --- Dashboard ---------------------------------------------------------------------------

def _on_dashboard_key(event):
    """Local-test hotkeys (drive computer only): 's' = simulate start, 'r' = reset."""
    if event.key == "s":
        with state_lock:
            if not shared["game_over"]:
                shared["game_active"] = True
        _log_mqtt("local", "start  (simulated with 's')")
    elif event.key == "r":
        with state_lock:
            shared["game_active"] = False
            shared["game_over"] = False
            shared["game_over_reason"] = None
        _log_mqtt("local", "reset  (simulated with 'r')")


def run_dashboard(mode):
    """Status bar, live spectrum with the command frequencies marked, what
    the robot is doing, and the MQTT feed. Blocks until the window closes."""
    own_tones = DRIVE_TONES if mode == "drive" else SHIELD_TONES
    other_tones = SHIELD_TONES if mode == "drive" else DRIVE_TONES

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title(f"World Cup [{mode.upper()}] -- team {TEAM_NAME}")
    gs = fig.add_gridspec(3, 2, height_ratios=(0.35, 1.4, 1.0), width_ratios=(1.6, 1.0),
                          hspace=0.45, wspace=0.25, left=0.06, right=0.97, top=0.95, bottom=0.05)
    ax_status = fig.add_subplot(gs[0, :])
    ax_spec = fig.add_subplot(gs[1, :])
    ax_info = fig.add_subplot(gs[2, 0])
    ax_mqtt = fig.add_subplot(gs[2, 1])

    ax_status.set_xticks([])
    ax_status.set_yticks([])
    status_text = ax_status.text(0.01, 0.5, "", fontsize=14, fontweight="bold", va="center",
                                 transform=ax_status.transAxes)
    hint = "[s] simulate start   [r] reset" if mode == "drive" else f"relaying shield to {CONTROL_TOPIC}"
    ax_status.text(0.99, 0.5, hint, fontsize=9, va="center", ha="right", color="#bbbbbb",
                   transform=ax_status.transAxes)

    freqs = detector.freqs
    spec_line, = ax_spec.plot(freqs, np.zeros_like(freqs), color="#39d6ff", linewidth=1)
    ax_spec.set_xlim(500, 6000)
    ax_spec.set_xlabel("Hz")
    ax_spec.set_title("Spectrum -- bright bands are this computer's commands, grey is the other computer's",
                      fontsize=10)
    for name, f in own_tones.items():
        ax_spec.axvspan(f - BAND_HALF_WIDTH_HZ, f + BAND_HALF_WIDTH_HZ, color="#4dff88", alpha=0.25)
        ax_spec.text(f, 1.01, f"{name}\n{f}", ha="center", va="bottom", fontsize=8,
                     transform=ax_spec.get_xaxis_transform())
    for f in other_tones.values():
        ax_spec.axvspan(f - BAND_HALF_WIDTH_HZ, f + BAND_HALF_WIDTH_HZ, color="#888888", alpha=0.15)

    ax_info.axis("off")
    info_text = ax_info.text(0.0, 1.0, "", fontsize=11, va="top", family="monospace",
                             transform=ax_info.transAxes)
    ax_mqtt.axis("off")
    ax_mqtt.set_title(f"MQTT on '{MQTT_TOPIC}' (newest first)", fontsize=10, loc="left")
    mqtt_text = ax_mqtt.text(0.0, 1.0, "", fontsize=8, va="top", family="monospace",
                             transform=ax_mqtt.transAxes)

    if mode == "drive":
        fig.canvas.mpl_connect("key_press_event", _on_dashboard_key)

    def update(_frame):
        with state_lock:
            reading = shared["reading"]
            live = shared["game_active"]
            over = shared["game_over"]
            reason = shared["game_over_reason"]
            last_drive = shared["last_drive"]
            shield_target = shared["shield_target"]
            proximity = shared["last_proximity"]
            log = list(shared["mqtt_log"])

        if over:
            color, label = "#5a2d7a", f"GAME OVER -- {reason}"
        elif live:
            color, label = "#1f6f43", "LIVE"
        else:
            color, label = "#555555", "WAITING FOR 'start'"
        ax_status.set_facecolor(color)
        status_text.set_text(f"{ROLE.upper()}  |  team {TEAM_NAME}  |  opponent "
                             f"{OPPONENT_TEAM_NAME or 'none'}  |  {label}")

        lines = []
        if reading is not None:
            spec_line.set_ydata(reading.spectrum)
            ax_spec.set_ylim(0, max(float(reading.spectrum.max()) * 1.1, 1.0))
            lines.append(f"Peak:     {reading.frequency:6.0f} Hz   ratio {reading.tonal_ratio:5.1f}"
                         f" (need {TONAL_RATIO_GATE:.0f})")
            lines.append(f"Command:  {(reading.tone or 'none').upper()}")
        if mode == "drive":
            lines.append(f"Wheels:   L={last_drive[0]:+d}%  R={last_drive[1]:+d}%")
            lines.append(f"Shield:   {shield_target} (from co-pilot)")
            prox = "--" if proximity is None else f"{proximity}"
            lines.append(f"Sensor:   {prox}  (caught at >= {PROXIMITY_REFLECTION_THRESHOLD})")
        else:
            lines.append(f"Sending:  shield {shield_target}")
        info_text.set_text("\n".join(lines))

        arrows = {"sent": "->", "recv": "<-", "local": "**"}
        mqtt_text.set_text("\n".join(
            f"{time.strftime('%H:%M:%S', time.localtime(e['t']))} {arrows[e['dir']]} {e['payload']}"
            for e in reversed(log)) or "(no MQTT traffic yet)")

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    return ani


# --- Setup + main --------------------------------------------------------------------------

def run_setup_dialog():
    """Small tkinter window: role, which computer this is, team names.
    Closing the window exits the program."""
    result = {}
    root = tk.Tk()
    root.title("World Cup -- setup")
    root.resizable(False, False)

    role_var = tk.StringVar(value="ball")
    channel_var = tk.StringVar(value="drive")
    team_var = tk.StringVar(value=DEFAULT_TEAM_NAME)
    opponent_var = tk.StringVar(value="")
    pad = {"padx": 16, "pady": (10, 2)}
    bold = ("Segoe UI", 10, "bold")

    tk.Label(root, text="Game role", font=bold).grid(row=0, column=0, sticky="w", **pad)
    tk.Radiobutton(root, text="Ball", variable=role_var, value="ball").grid(row=1, column=0, sticky="w", padx=32)
    tk.Radiobutton(root, text="Goalie", variable=role_var, value="goalie").grid(row=2, column=0, sticky="w", padx=32)

    tk.Label(root, text="This computer controls", font=bold).grid(row=3, column=0, sticky="w", **pad)
    tk.Radiobutton(root, text="Drive (Double Motor) -- connects to the robot",
                   variable=channel_var, value="drive").grid(row=4, column=0, sticky="w", padx=32)
    tk.Radiobutton(root, text="Shield (Single Motor) co-pilot -- no robot connection",
                   variable=channel_var, value="shield").grid(row=5, column=0, sticky="w", padx=32)

    tk.Label(root, text="Team name (must match your co-pilot's exactly)", font=bold).grid(
        row=6, column=0, sticky="w", **pad)
    team_entry = tk.Entry(root, textvariable=team_var, width=30)
    team_entry.grid(row=7, column=0, sticky="w", padx=32)

    tk.Label(root, text="Opponent's team name (optional -- blank = no auto win/lose)", font=bold).grid(
        row=8, column=0, sticky="w", **pad)
    tk.Entry(root, textvariable=opponent_var, width=30).grid(row=9, column=0, sticky="w", padx=32)

    error_label = tk.Label(root, text="", fg="red")
    error_label.grid(row=10, column=0, sticky="w", padx=32)

    def on_start():
        team, opponent = team_var.get().strip(), opponent_var.get().strip()
        if not team:
            error_label.config(text="Team name can't be empty.")
            return
        if opponent == team:
            error_label.config(text="Opponent's team name can't be the same as your own.")
            return
        result.update(role=role_var.get(), channel=channel_var.get(), team=team, opponent=opponent)
        root.destroy()

    tk.Button(root, text="Start", command=on_start, width=14).grid(row=11, column=0, pady=14)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    team_entry.focus_set()
    root.mainloop()

    if not result:
        print("Setup cancelled -- exiting.")
        sys.exit(0)
    return result["role"], result["channel"], result["team"], result["opponent"]


def _open_stream(pa, device_index):
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                     input_device_index=device_index, frames_per_buffer=BLOCK_SIZE,
                     stream_callback=audio_callback)
    stream.start_stream()
    return stream


def _run_as_drive():
    global car, shield, sensor, mqtt

    pa = pyaudio.PyAudio()
    device_index = pick_mic(pa)

    mqtt = MQTTClient()
    mqtt.connect()
    mqtt.subscribe(MQTT_TOPIC, on_mqtt_message)
    mqtt.subscribe(CONTROL_TOPIC, on_control_message)
    print(f"[MQTT] Subscribed to {MQTT_TOPIC} and {CONTROL_TOPIC}. Waiting for 'start'...")

    car, shield, sensor = doubleMotor(), singleMotor(), colorSensor()
    stream = None
    loop_thread = None
    stop_event = threading.Event()
    try:
        print("Connecting to Double Motor (drive)...")
        car.connect(card_serial=CAR_CARD_SERIAL, card_color=CAR_CARD_COLOR)
        print("Connecting to Single Motor (shield)...")
        shield.connect(card_serial=SHIELD_CARD_SERIAL, card_color=SHIELD_CARD_COLOR)
        print("Connecting to Color Sensor...")
        sensor.connect(card_serial=SENSOR_CARD_SERIAL, card_color=SENSOR_CARD_COLOR)
        if not (car.connected and shield.connected and sensor.connected):
            raise ConnectionError("One or more devices did not connect.")

        shield.motor_reset_relative_position()  # wherever the arm is now = "down"

        loop_thread = threading.Thread(target=run_drive_loop, args=(stop_event,), daemon=True)
        loop_thread.start()
        stream = _open_stream(pa, device_index)
        run_dashboard("drive")
    finally:
        stop_event.set()
        if loop_thread is not None:
            loop_thread.join(timeout=1.0)
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()
        for cleanup in (car.movement_stop, shield.stop,
                        car.disconnect, shield.disconnect, sensor.disconnect, mqtt.disconnect):
            try:
                cleanup()
            except Exception as e:
                print(f"[shutdown] {e}")


def _run_as_copilot():
    global mqtt

    pa = pyaudio.PyAudio()
    device_index = pick_mic(pa)

    mqtt = MQTTClient()
    mqtt.connect()
    mqtt.subscribe(MQTT_TOPIC, on_mqtt_message)
    print(f"[MQTT] Watching {MQTT_TOPIC}; sending shield commands to {CONTROL_TOPIC}.")

    stream = None
    loop_thread = None
    stop_event = threading.Event()
    try:
        loop_thread = threading.Thread(target=run_copilot_loop, args=(stop_event,), daemon=True)
        loop_thread.start()
        stream = _open_stream(pa, device_index)
        run_dashboard("shield")
    finally:
        stop_event.set()
        if loop_thread is not None:
            loop_thread.join(timeout=1.0)
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()
        mqtt.disconnect()


def main():
    global ROLE, CONTROL_CHANNEL, TEAM_NAME, OPPONENT_TEAM_NAME, CONTROL_TOPIC, detector

    ROLE, CONTROL_CHANNEL, TEAM_NAME, OPPONENT_TEAM_NAME = run_setup_dialog()
    CONTROL_TOPIC = f"{MQTT_TOPIC}/control/{TEAM_NAME}"
    tones = DRIVE_TONES if CONTROL_CHANNEL == "drive" else SHIELD_TONES
    detector = ToneDetector(tones)

    print(f"Role: {ROLE}   Computer: {CONTROL_CHANNEL}   Team: {TEAM_NAME}   "
          f"Opponent: {OPPONENT_TEAM_NAME or '(none)'}")
    print("Play these tones from your phone:")
    for name, freq in tones.items():
        print(f"  {freq:>5} Hz  ->  {name}")

    if CONTROL_CHANNEL == "drive":
        _run_as_drive()
    else:
        _run_as_copilot()


if __name__ == "__main__":
    main()
