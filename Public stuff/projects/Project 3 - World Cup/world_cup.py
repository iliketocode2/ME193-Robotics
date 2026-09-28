"""
world_cup.py -- phone-tone-controlled LEGO Education "car" for the ME193
World Cup match, coordinated with an opponent robot over MQTT on topic
ME193/Rogers.

Play fixed pitches from a phone tone-generator app into the laptop mic
(see tone_policy.py for the exact frequencies):
    - DRIVE computer (connected to the robot): forward / left / right,
      plus a held "goal" tone. No tone = stop.
    - SHIELD co-pilot computer (no robot connection): "up" / "down" spin
      the Single Motor shield continuously one way or the other while the
      tone is held (no tone = shield stops), relayed to the drive computer
      over MQTT.
      The shield tones are all higher than the drive tones, and each
      computer only listens inside its own range, so the two phones never
      interfere.

See this project's README.md for the frequency table, the MQTT message
schema, and venv setup.

Run (from the my_env_audio venv -- NOT my_env, see README):
    my_env_audio/Scripts/python "Public stuff/projects/Project 3 - World Cup/world_cup.py"
"""

import json
import math
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
from songs import DEATH_SONG, SUCCESS_SONG, play_lose, play_win
from tone_policy import (BLOCK_SIZE, DRIVE_TONES, SAMPLE_RATE, SHIELD_TONES, TONAL_RATIO_GATE,
                         ToneDetector, band_half_width)

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
GOAL_HOLD_S = 0.5   # hold the "goal" tone this long to call a goal (ball only)

# --- Turn tuning -------------------------------------------------------------------
# "left"/"right" each make ONE in-place turn of TURN_DEGREES (the Double
# Motor's own IMU-controlled movement_turn_for_degrees), then the car stops.
# To turn again, the tone has to go away for TURN_REARM_S and come back (or
# switch to the other direction) -- holding it doesn't keep turning.
TURN_DEGREES = 90
TURN_SPEED = 40               # %
SWAP_TURN_DIRECTIONS = False  # flip if "left" turns the car right
TURN_REARM_S = 0.3            # tone must be gone this long to count as a NEW command (a mic glitch isn't)
# The turn counts as finished once the yaw has covered TURN_DONE_FRACTION of
# the turn and then held still for TURN_SETTLE_S. The Double Motor reports
# yaw only every ~100ms, so the settle window spans a few reports -- one late
# report mustn't look like "stopped". Finishing never sends a stop (the
# firmware ends its own turn); only TURN_TIMEOUT_S, the backstop for e.g. no
# yaw readings, stops the car.
TURN_DONE_FRACTION = 0.8
TURN_SETTLE_S = 0.3
TURN_TIMEOUT_S = 3.0

# --- Shield tuning ----------------------------------------------------------------
# The shield motor spins CONTINUOUSLY while "up" or "down" is held and stops
# when the tone stops -- hold longer for a bigger movement. Flip
# INVERT_SHIELD if "up" spins the wrong way.
INVERT_SHIELD = False
SHIELD_SPEED = 100      # % -- the motor's own speed setting (always positive; direction picks the way)
SHIELD_RESEND_S = 0.25  # co-pilot re-sends its shield state this often, in case an MQTT message is lost
# Dead-man: the drive computer stops the shield if it hears nothing from the
# co-pilot for this long, so a lost "stop" message or a crashed co-pilot
# can't leave it spinning.
SHIELD_TIMEOUT_S = 1.0

# --- Proximity ("caught") -----------------------------------------------------------
# reflection() reads roughly 0-100 on real hardware. Calibrate on-site
# against the opponent robot and room lighting.
PROXIMITY_REFLECTION_THRESHOLD = 70

MQTT_TOPIC = "ME193/Rogers"
CONTROL_LOOP_PERIOD_S = 0.05
# If the mic stops delivering audio (device unplugged, laptop asleep, a
# crash in the callback), treat the last reading as "no tone" after this
# long, so the car can't keep driving on a frozen "forward". Readings
# normally arrive every ~46ms; this allows for a few late ones.
STALE_READING_S = 0.25

# Wheel speeds (left, right) for each drive command, before INVERT_* flags.
# None = no tone = stop. "goal" also holds still while it's being held.
# "left"/"right" aren't here -- they're one-shot turns (start_turn()).
DRIVE_COMMANDS = {
    None: (0, 0),
    "forward": (FORWARD_SPEED, FORWARD_SPEED),
    "goal": (0, 0),
}
TURN_COMMANDS = ("left", "right")

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
    "shield_target": "stop",     # "up"/"down"/"stop". drive: latest from co-pilot. co-pilot: what it's sending.
    "shield_msg_time": 0.0,      # drive: time.monotonic() of the last message from the co-pilot
    "last_drive": (0, 0),        # last (left, right) actually sent to the car
    "turn_status": "ready",      # dashboard text for the one-shot turns
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
    left, right = DRIVE_COMMANDS[None if command in TURN_COMMANDS else command]
    if INVERT_LEFT_MOTOR:
        left = -left
    if INVERT_RIGHT_MOTOR:
        right = -right
    return left, right


def _yaw_delta(yaw, start_yaw):
    """|yaw - start_yaw| in degrees, wrapped to 0-180 (NaN if either is unknown)."""
    return abs((yaw - start_yaw + 180) % 360 - 180)


def start_turn(command):
    """Send one IMU-controlled turn of TURN_DEGREES. Non-blocking: the
    control loop keeps running (sensor, goal, shield) and watches the yaw
    to see when it's done. Returns the turn's tracking state."""
    left = (command == "left") != SWAP_TURN_DIRECTIONS
    direction = le.MOVEMENT_TURN_DIRECTION_LEFT if left else le.MOVEMENT_TURN_DIRECTION_RIGHT
    yaw = car.yaw()
    car.movement_turn_for_degrees(TURN_DEGREES, direction=direction, speed=TURN_SPEED, blocking=False)
    now = time.monotonic()
    return {"command": command, "start_time": now, "start_yaw": yaw,
            "last_yaw": yaw, "steady_since": now}


def turn_finished(turn, now):
    """True once the turn is done: yaw has covered most of it and stopped
    changing, or TURN_TIMEOUT_S passed (then the car is stopped here)."""
    yaw = car.yaw()
    steady = not (math.isnan(yaw) or math.isnan(turn["last_yaw"])) and abs(yaw - turn["last_yaw"]) < 1.0
    if not steady:
        turn["last_yaw"], turn["steady_since"] = yaw, now
    turned = _yaw_delta(yaw, turn["start_yaw"])
    shown = "?" if math.isnan(turned) else f"{turned:.0f}"
    with state_lock:
        shared["turn_status"] = f"turning {turn['command']}  {shown} / {TURN_DEGREES} deg"
    if turned >= TURN_DONE_FRACTION * TURN_DEGREES and now - turn["steady_since"] >= TURN_SETTLE_S:
        return True
    if now - turn["start_time"] >= TURN_TIMEOUT_S:
        print(f"[turn] {turn['command']} didn't confirm within {TURN_TIMEOUT_S}s "
              f"(yaw moved {shown} deg) -- stopping")
        car.movement_stop()
        return True
    return False


def announce_result(song, message):
    """Stop the car and play the song. Control-loop thread only (BLE calls)."""
    with state_lock:
        shared["game_over"] = True
        shared["game_active"] = False
        shared["game_over_reason"] = message
        shared["last_drive"] = (0, 0)
    print(f"[GAME] {message}")
    for stop in (car.movement_stop, shield.stop):  # not lelib's car.stop(), which only stops the left motor
        try:
            stop()
        except Exception:
            pass
    # mp3 on the computer, the matching beeped song on the hub (see songs.py)
    if song is SUCCESS_SONG:
        play_win(car)
    else:
        play_lose(car)


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
    """Drive computer: {"shield": "up"|"down"|"stop"} from the co-pilot."""
    try:
        data = json.loads(payload)
    except ValueError:
        return
    target = data.get("shield") if isinstance(data, dict) else None
    if target in SHIELD_TONES or target == "stop":
        with state_lock:
            shared["shield_target"] = target
            shared["shield_msg_time"] = time.monotonic()


def run_shield(command):
    """Spin the shield continuously for "up"/"down", stop it for "stop"."""
    if command == "stop":
        shield.stop()
        return
    clockwise = (command == "up") != INVERT_SHIELD
    direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if clockwise else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
    shield.motor_run(direction=direction, speed=SHIELD_SPEED, blocking=False)


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
    turn = None            # start_turn()'s state while a one-shot turn is running
    last_turn = None       # the turn command that must be released before it can fire again
    released_since = None  # when last_turn's tone was last seen gone

    while not stop_event.is_set():
        with state_lock:
            reading = shared["reading"]
            if time.monotonic() - shared["reading_time"] > STALE_READING_S:
                reading = None
            live = shared["game_active"] and not shared["game_over"]
            remote_result = shared["remote_result_pending"]
            shared["remote_result_pending"] = None
            shield_target = shared["shield_target"]
            if time.monotonic() - shared["shield_msg_time"] > SHIELD_TIMEOUT_S:
                shield_target = "stop"  # co-pilot went quiet -- don't leave the shield spinning

        try:
            if remote_result is not None:
                announce_result(*remote_result)
                sent_drive, turn = (0, 0), None
                continue

            command = reading.tone if (live and reading is not None) else None
            now = time.monotonic()

            # Re-arm: the tone behind the last turn has to be gone for
            # TURN_REARM_S before that same turn can fire again.
            if command == last_turn:
                released_since = None
            elif released_since is None:
                released_since = now
            if last_turn is not None and released_since is not None and now - released_since >= TURN_REARM_S:
                last_turn = None

            if turn is not None:
                if not live:
                    car.movement_stop()  # match ended/reset mid-turn
                    turn, sent_drive = None, (0, 0)
                elif turn_finished(turn, now):
                    turn = None
                    # The firmware stops the car itself at the end of the turn, so
                    # count it as stopped -- sending a stop now could cut a turn
                    # short if "finished" was judged a little early.
                    sent_drive = (0, 0)

            if turn is None and command in TURN_COMMANDS and command != last_turn:
                turn = start_turn(command)
                last_turn, released_since = command, None
                sent_drive = None
                with state_lock:
                    shared["last_drive"] = (0, 0)
            elif turn is None:
                wheels = drive_wheels(command)
                if wheels != sent_drive:
                    if wheels == (0, 0):
                        car.movement_stop()
                    else:
                        car.movement_move_tank(*wheels, blocking=False)
                    sent_drive = wheels
                    with state_lock:
                        shared["last_drive"] = wheels
                with state_lock:
                    shared["turn_status"] = (f"{last_turn} done -- release the tone to turn again"
                                             if last_turn else "ready")

            # Shield and sensor get their own try: a failure there shouldn't
            # stop the car and make a held drive tone stutter.
            reflection = None
            try:
                # Once shutdown starts, only ever stop -- never start the shield
                # after main() has already sent its cleanup stop.
                shield_command = shield_target if live and not stop_event.is_set() else "stop"
                if shield_command != sent_shield:
                    sent_shield = shield_command  # set first: a failing command isn't retried every tick
                    try:
                        run_shield(shield_command)
                    except Exception:
                        if shield_command == "stop":
                            sent_shield = None  # a failed STOP must be retried -- the shield may still be spinning
                        raise
                reflection = sensor.reflection()
                with state_lock:
                    shared["last_proximity"] = reflection
            except Exception as e:
                print(f"[control loop] Shield/sensor error: {e}")

            if not live or ROLE != "ball":
                continue

            if command == "goal":
                goal_started = goal_started or time.monotonic()
                if time.monotonic() - goal_started >= GOAL_HOLD_S and not goal_fired:
                    goal_fired = True
                    _publish({"event": "goal", "team": TEAM_NAME})
                    announce_result(SUCCESS_SONG, "GOAL! You scored.")
                    sent_drive, turn = (0, 0), None
            else:
                goal_started, goal_fired = None, False
                if reflection is not None and reflection >= PROXIMITY_REFLECTION_THRESHOLD:
                    _publish({"event": "fail", "team": TEAM_NAME})
                    announce_result(DEATH_SONG, "Caught! You failed.")
                    sent_drive, turn = (0, 0), None
        except Exception as e:
            print(f"[control loop] Error, stopping car and shield: {e}")
            for stop in (car.movement_stop, shield.stop):
                try:
                    stop()
                except Exception:
                    pass
            sent_drive, sent_shield, turn = (0, 0), "stop", None
        finally:
            stop_event.wait(CONTROL_LOOP_PERIOD_S)


def run_copilot_loop(stop_event):
    """Shield co-pilot: "up"/"down" while that tone is held, "stop" the
    moment it isn't. Publishes on change, and every SHIELD_RESEND_S."""
    target = "stop"
    last_sent_target, last_sent_time = None, 0.0

    while not stop_event.is_set():
        with state_lock:
            reading = shared["reading"]
            if time.monotonic() - shared["reading_time"] > STALE_READING_S:
                reading = None
        target = reading.tone if reading is not None and reading.tone in SHIELD_TONES else "stop"

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
        hw = band_half_width(name)
        ax_spec.axvspan(f - hw, f + hw, color="#4dff88", alpha=0.25)
        ax_spec.text(f, 1.01, f"{name}\n{f}", ha="center", va="bottom", fontsize=8,
                     transform=ax_spec.get_xaxis_transform())
    for name, f in other_tones.items():
        hw = band_half_width(name)
        ax_spec.axvspan(f - hw, f + hw, color="#888888", alpha=0.15)

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
            turn_status = shared["turn_status"]
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
            lines.append(f"Turn:     {turn_status}")
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
