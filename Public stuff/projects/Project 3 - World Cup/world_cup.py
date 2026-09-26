"""
world_cup.py -- whistle-controlled LEGO Education "car" for the ME193
World Cup match: drive by whistling into the laptop mic (pitch -> a
continuous speed/turn gradient, a 3-pulse rhythm -> goal command), and
optionally a second whistler on a second computer whose pitch continuously
drives the Single Motor shield the same gradient way, coordinated with an
opponent robot over MQTT on topic ME193/Rogers.

See this project's README.md for: the full policy description, the "no
whistle detected" behavior, the noise-masking approach, the MQTT message
schema (must be agreed with the opponent team before a match), and the
setup instructions for the two venvs this project spans.

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

# lelib.py / mqttlib.py live in the shared "useful libraries" folder, two
# levels up from this script -- add it to sys.path so the imports below
# resolve no matter where this script is run from (same pattern as
# mqtt_chat.py / the Project 2 scripts).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "useful libraries"))

from lelib import colorSensor, doubleMotor, singleMotor  # noqa: E402
from mqttlib import MQTTClient  # noqa: E402

from pyaudio_mic import pick_mic
from songs import DEATH_SONG, SUCCESS_SONG, play_both
from whistle_policy import DEFAULT_CONFIG, WhistlePolicy

# --- Match-day settings -- chosen via the pre-flight setup dialog in main(),
# not hardcoded here. See run_setup_dialog(). --------------------------------
ROLE = None             # "ball" or "goalie"
TEAM_NAME = None        # must be typed identically on a co-pilot's computer, if using one
OPPONENT_TEAM_NAME = None  # the name their team enters as OWN TEAM_NAME on their computer;
                            # blank = don't auto-react to anyone's fail/goal messages
CONTROL_CHANNEL = None  # "drive" (connects to the robot) or "shield" (remote co-pilot, no hardware)
CONTROL_TOPIC = None    # f"{MQTT_TOPIC}/control/{TEAM_NAME}" -- set once TEAM_NAME is known
DEFAULT_TEAM_NAME = "Cucurella"  # just a prefill convenience for the dialog

# --- Hardware placeholders -- fill in with your LEGO Connection Card values --
CAR_CARD_SERIAL = 1126  # doubleMotor (drive) -- None = first Double Motor found, ambiguous with >1 nearby
CAR_CARD_COLOR = le.LEGO_COLOR_GREEN
SHIELD_CARD_SERIAL = 1126  # singleMotor (cardboard shield arm)
SHIELD_CARD_COLOR = le.LEGO_COLOR_GREEN
SENSOR_CARD_SERIAL = 1126  # colorSensor (front-facing light sensor)
SENSOR_CARD_COLOR = le.LEGO_COLOR_GREEN

# --- Drive tuning -------------------------------------------------------------
# Motor mounting is mirrored side-to-side on some chassis -- flip whichever
# one drives the wrong way once you watch the car actually move, same as
# arm_race_control.py's INVERT_LEFT_MOTOR/INVERT_RIGHT_MOTOR.
INVERT_LEFT_MOTOR = True
INVERT_RIGHT_MOTOR = True

# --- Shield tuning -------------------------------------------------------------
# The shield is continuously position-controlled (see set_shield_position()):
# absolute position 0 = fully retracted, SHIELD_SWING_DEGREES = fully deployed.
# motor_run_to_absolute_position() treats 0 as whatever the hub's own internal
# zero reference is -- verify on the physical robot that "0" actually lines up
# with the shield retracted, and adjust SHIELD_SWING_DEGREES/sign if not.
#
# Deliberately tuned to feel snappier than driving: a swinging cardboard flap
# isn't a safety concern the way a fast-turning drive wheel is, so there's no
# reason to make it gentle. SHIELD_SPEED is the physical swing speed (higher
# than the double motor's max_speed); SHIELD_MAX_TURN_SPEED/
# SHIELD_TURN_DEADZONE_FRAC give the shield co-pilot's *own* WhistlePolicy
# instance (see _run_as_copilot()) a much more sensitive gradient than the
# drive's -- full -100..100 range, no dead zone -- so it reacts to a smaller
# whistle-pitch change instead of needing the same wide "straight" zone/gentle
# cap that's tuned for comfortable steering.
SHIELD_SWING_DEGREES = 90   # how far the shield arm swings between retracted/deployed
SHIELD_SPEED = 100          # % speed for shield moves -- fast, unlike the gentler drive speeds
SHIELD_MAX_TURN_SPEED = 100     # shield co-pilot's own gradient cap: full range, not the drive's gentler one
SHIELD_TURN_DEADZONE_FRAC = 0.0  # shield co-pilot's own deadzone: none -- react readily, don't require a wide "center" zone

# --- Proximity ("caught") tuning ----------------------------------------------
# LELIB.md and lelib.py's raw_reading() both document/assume reflection() as
# 0-255 (raw_reading() even divides by 255 to "normalize" it), but on real
# hardware it was observed to only ever read roughly 0-100 -- this looks like
# a real, unverified-against-hardware inaccuracy in that shared documentation/
# normalization, not a problem with this sensor's calibration (see the
# ColorSensorNotification wire format in legoeducation/rpc_message.py: it's an
# unsigned byte, so 0-255 is representable, but the firmware apparently never
# reports above ~100 -- a % of reflected light, not a raw 0-255 value). Using
# the old 0-255-based threshold (200) here would have meant handle_caught()
# could never fire at all. Higher = more light reflected = something close/
# bright in front of the sensor. Still has to be calibrated on-site against
# the actual opponent robot and room lighting on match day -- 70 is a
# starting guess within the real range, not a verified value.
PROXIMITY_REFLECTION_THRESHOLD = 70

# --- MQTT ----------------------------------------------------------------------
MQTT_TOPIC = "ME193/Rogers"

# --- Audio ---------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "whistle_config.json")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        whistle_config = json.load(f)
    print(f"[audio] Loaded calibrated whistle bands from {CONFIG_PATH}")
else:
    whistle_config = DEFAULT_CONFIG
    print(f"[audio] WARNING: {CONFIG_PATH} not found -- using DEFAULT_CONFIG placeholder bands. "
          f"Run calibrate.py first for a real match.")

SAMPLE_RATE = whistle_config["sample_rate"]
BLOCK_SIZE = whistle_config["block_size"]

# Same calibrated frequency bands, but a much more sensitive turn-band gradient
# (see the "Shield tuning" comment above) -- used only by the shield co-pilot's
# own WhistlePolicy instance, swapped in by _run_as_copilot() before its threads
# start. The drive operator keeps using `whistle_config` as-is via `policy` below.
shield_whistle_config = {
    **whistle_config,
    "max_turn_speed": SHIELD_MAX_TURN_SPEED,
    "turn_deadzone_frac": SHIELD_TURN_DEADZONE_FRAC,
}

policy = WhistlePolicy(whistle_config)

# --- Shared state between the audio callback thread, the MQTT thread, the
# control thread, and the main/plotting thread. Only ever mutated/read under
# `state_lock`. See audio_callback()'s and run_control_loop()'s docstrings for
# why BLE calls live on their own dedicated thread, not the audio callback. --
state_lock = threading.Lock()
MQTT_LOG_MAXLEN = 10  # how many recent MQTT sent/received entries the dashboard shows
shared = {
    "command": None,     # most recent whistle_policy.Command
    "block": np.zeros(BLOCK_SIZE, dtype=np.float32),
    "game_active": False,       # True once "start" arrives on MQTT_TOPIC (or the local-test hotkey)
    "game_over": False,         # True once this robot has failed or scored
    "game_over_reason": None,   # human-readable reason string, for the dashboard
    "goal_pending": False,            # set by the audio callback, consumed by the control loop
    "remote_result_pending": None,    # set by on_mqtt_message, consumed by the control loop: (song, message) or None
    # drive mode: latest -100..100 gradient from on_control_message. Defaults to -100
    # (fully retracted) rather than 0 (which would be the swing's MIDPOINT, not
    # retracted) -- see set_shield_position()'s mapping -- so the shield sits safely
    # out of the sensor's way if no co-pilot is connected.
    "remote_shield_gradient": -100.0,
    "mqtt_log": deque(maxlen=MQTT_LOG_MAXLEN),  # dicts: t, dir ("sent"/"recv"/"local"), topic, payload
    "last_drive": (0.0, 0.0),   # last (left, right) wheel speed actually sent to movement_move_tank
    "last_shield_position": 0.0,  # last commanded shield position, 0..SHIELD_SWING_DEGREES
    "last_proximity": None,     # last sensor.reflection() reading (tracked regardless of ROLE, for display)
}

CONTROL_LOOP_PERIOD_S = 0.05  # ~20Hz -- plenty responsive for a human whistling, well clear of BLE round-trip time

# Set up in main(); referenced by the callbacks/control loop by module-level
# lookup, same pattern as arm_race_control.py's module-level hardware handles.
car = None
shield = None
sensor = None
mqtt = None


def _clamp(value, lo=-100, hi=100):
    return max(lo, min(hi, value))


def _mix_tank_drive(forward_speed, turn_bias):
    """Arcade-drive mixing (same scheme as arm_race_control.py): forward moves
    both wheels together, turn spins them apart."""
    left = forward_speed + turn_bias
    right = forward_speed - turn_bias
    if INVERT_LEFT_MOTOR:
        left = -left
    if INVERT_RIGHT_MOTOR:
        right = -right
    return _clamp(left), _clamp(right)


def _log_mqtt(direction, payload):
    """Record one MQTT event (direction: 'sent', 'recv', or 'local') for the
    dashboard's live traffic feed. `local` is for the run_live_plot() test
    hotkeys, so a simulated 'start' is never confused with a real one."""
    with state_lock:
        shared["mqtt_log"].append({"t": time.time(), "dir": direction, "payload": payload})


def _publish(payload_dict):
    text = json.dumps(payload_dict)
    mqtt.publish(MQTT_TOPIC, text)
    _log_mqtt("sent", text)


def set_shield_position(gradient):
    """gradient: -100..100 (a whistle_policy.Command.turn_bias value, reused
    here as a continuous shield-position gradient rather than a turn command
    -- see Command's docstring). Maps linearly to an absolute shield angle:
    -100 -> fully retracted (0deg), +100 -> fully deployed (SHIELD_SWING_DEGREES).
    Sent every control-loop tick (like movement_move_tank), not just on change,
    so the shield continuously tracks the whistled pitch in real time."""
    frac = max(0.0, min(1.0, (gradient + 100.0) / 200.0))
    target_degrees = frac * SHIELD_SWING_DEGREES
    shield.motor_run_to_absolute_position(target_degrees, speed=SHIELD_SPEED, blocking=False)
    with state_lock:
        shared["last_shield_position"] = target_degrees


def announce_result(song, message):
    """Only ever called from run_control_loop()'s thread -- both car.movement_stop()
    and play_both()'s hub-side beep() calls do real BLE round-trips, which must
    never run on the PyAudio callback thread or the MQTT network thread."""
    with state_lock:
        shared["game_over"] = True
        shared["game_active"] = False
    print(f"[GAME] {message}")
    try:
        # car.stop() (lelib.py) calls motor_stop() with no motor= kwarg, which only
        # stops motor index 0 (the LEFT motor only) -- verified in
        # legoeducation/device.py's DEFAULT_MOTOR=0. movement_stop() is the correct
        # whole-robot stop for a car driven via movement_move_tank().
        car.movement_stop()
    except Exception:
        pass
    play_both(car, song)  # plays on the LEGO hub's speaker AND the computer's speaker at once


def handle_caught():
    """This robot's own front sensor detected the opponent close up -- only
    meaningful for the ball (see the README's proximity/fail rule). Tags the
    publish with this robot's own TEAM_NAME (not ROLE) so the opponent can
    identify it by name -- see on_mqtt_message()."""
    _publish({"event": "fail", "team": TEAM_NAME})
    with state_lock:
        shared["game_over_reason"] = "Caught! You failed."
    announce_result(DEATH_SONG, "Caught! You failed.")


def handle_scored():
    """This robot's own whistle policy fired the goal gesture -- only
    meaningful for the ball. Tags the publish with this robot's own
    TEAM_NAME (not ROLE) -- see handle_caught()'s comment."""
    _publish({"event": "goal", "team": TEAM_NAME})
    with state_lock:
        shared["game_over_reason"] = "GOAL! You scored."
    announce_result(SUCCESS_SONG, "GOAL! You scored.")


def on_mqtt_message(topic, payload):
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
        return  # not "start" and not JSON -- ignore (e.g. unrelated traffic on a shared public broker)

    event, team = data.get("event"), data.get("team")
    if team == TEAM_NAME:
        return  # our own publish, echoed back by the broker -- already handled locally

    if not OPPONENT_TEAM_NAME or team != OPPONENT_TEAM_NAME:
        return  # not the team we're configured to react to -- e.g. a third team's traffic

    # Queue the result rather than calling announce_result() directly: this callback
    # runs on paho's own network thread, and announce_result() does a real BLE stop
    # plus a multi-second sequence of blocking beep() calls -- blocking paho's thread
    # for that long risks missing keepalives/other messages. The control loop thread
    # picks this up within one CONTROL_LOOP_PERIOD_S tick. Symmetric and role-agnostic:
    # whichever of us actually has a local trigger (today, only the ball's proximity
    # sensor/goal whistle), the other one reacts oppositely just by being the
    # configured opponent -- no hardcoded "ball"/"goalie" assumption needed here.
    if event == "fail":
        print(f"[MQTT] {OPPONENT_TEAM_NAME} reported it failed.")
        with state_lock:
            shared["remote_result_pending"] = (SUCCESS_SONG, f"{OPPONENT_TEAM_NAME} got caught -- you win!")
    elif event == "goal":
        print(f"[MQTT] {OPPONENT_TEAM_NAME} reported a goal.")
        with state_lock:
            shared["remote_result_pending"] = (DEATH_SONG, f"{OPPONENT_TEAM_NAME} scored -- you lose!")


def on_mqtt_message_readonly(topic, payload):
    """Shield co-pilot only: mirrors game status from MQTT_TOPIC on its own
    dashboard for display purposes -- never touches hardware (there is none
    on this instance) or plays a song (only the drive operator's hub/computer
    does that). No self-echo filtering needed: this instance never publishes
    to MQTT_TOPIC itself, so every message here is genuinely someone else's."""
    text = payload.strip()
    _log_mqtt("recv", text)

    if text.lower() == "start":
        with state_lock:
            if not shared["game_over"]:
                shared["game_active"] = True
        return

    try:
        data = json.loads(text)
    except ValueError:
        return

    event, team = data.get("event"), data.get("team")
    if event in ("fail", "goal"):
        with state_lock:
            shared["game_over"] = True
            shared["game_active"] = False
            shared["game_over_reason"] = f"{team}: {event}"


def on_control_message(topic, payload):
    """Drive operator only: a continuous shield-position gradient relayed from
    a remote co-pilot's whistle, on CONTROL_TOPIC. Only the co-pilot ever
    publishes to this topic and only the drive operator ever subscribes to
    it, so unlike on_mqtt_message()'s bidirectional MQTT_TOPIC protocol,
    there's no self-echo case to filter here. Not logged to mqtt_log on every
    message (the co-pilot publishes ~20x/second) -- see run_control_loop(),
    which logs the value actually applied at its own tick rate instead."""
    try:
        data = json.loads(payload)
    except ValueError:
        return
    gradient = data.get("shield_gradient")
    if isinstance(gradient, (int, float)):
        with state_lock:
            shared["remote_shield_gradient"] = float(gradient)


def audio_callback(in_data, frame_count, time_info, status):
    """Runs on PyAudio's real-time callback thread. Must stay fast: only DSP
    (policy.update() is pure numpy/FFT, no I/O) plus a lock-guarded dict write.

    No BLE or MQTT calls happen here, even "non-blocking" ones -- verified in
    legoeducation/_platform.py's _run_sync_cpython: every synchronous
    legoeducation call blocks its calling thread on a real BLE round-trip via
    `wrapper_future.result()` regardless of that call's own `blocking=` kwarg
    (which only controls whether it *additionally* waits for a completion
    response). If a BLE round-trip ever approached this callback's ~46ms block
    period, PortAudio would report input overflow / drop audio. Motor/shield/
    song/proximity work all happens on the dedicated run_control_loop() thread
    instead -- the same fix Project 2's README describes for the same reason
    (blocking BLE calls stalling a camera loop that shared its thread)."""
    block = np.frombuffer(in_data, dtype=np.float32)
    cmd = policy.update(block, now=time.monotonic())

    with state_lock:
        shared["command"] = cmd
        shared["block"] = block
        if shared["game_active"] and not shared["game_over"] and ROLE == "ball" and cmd.goal_detected:
            shared["goal_pending"] = True

    return (None, pyaudio.paContinue)


def run_control_loop(stop_event):
    """Dedicated thread: the only thread that ever issues BLE motor/shield/song
    calls or MQTT publishes for game events. Polls the state the audio callback
    and on_mqtt_message wrote, at CONTROL_LOOP_PERIOD_S -- comfortably above any
    single BLE round-trip, and fully decoupled from the audio callback's much
    tighter real-time budget."""
    while not stop_event.is_set():
        with state_lock:
            cmd = shared["command"]
            game_active, game_over = shared["game_active"], shared["game_over"]
            do_goal = shared["goal_pending"]
            shared["goal_pending"] = False
            remote_result = shared["remote_result_pending"]
            shared["remote_result_pending"] = None
            remote_shield_gradient = shared["remote_shield_gradient"]

        try:
            if remote_result is not None:
                song, message = remote_result
                with state_lock:
                    shared["game_over_reason"] = message
                announce_result(song, message)
            elif game_active and not game_over and cmd is not None:
                left, right = _mix_tank_drive(cmd.forward_speed, cmd.turn_bias)
                car.movement_move_tank(left, right, blocking=False)
                with state_lock:
                    shared["last_drive"] = (left, right)

                # Read once and reuse for both the dashboard and the fail check --
                # tracked regardless of ROLE so the dashboard is useful to the
                # goalie too, even though only the ball acts on it.
                reflection = sensor.reflection()
                with state_lock:
                    shared["last_proximity"] = reflection

                # Shield is generic hardware, not ball-specific game logic (see
                # audio_callback()'s matching comment) -- driven exclusively by a
                # remote co-pilot's continuous gradient (there's no local whistle
                # bandwidth left for it once stop/turn/forward already use the
                # whole pitch range). Sent every tick, like movement_move_tank
                # above, so the shield continuously tracks the co-pilot's pitch.
                set_shield_position(remote_shield_gradient)

                if ROLE == "ball":
                    if do_goal:
                        handle_scored()
                    elif reflection >= PROXIMITY_REFLECTION_THRESHOLD:
                        handle_caught()
        except Exception as e:
            # Hard stop path, independent of whatever the policy/hardware just did --
            # a runaway command driving the car indefinitely is the real failure mode
            # to guard against here, not a clean exit.
            print(f"[control loop] Error during control step, stopping: {e}")
            try:
                car.movement_stop()
            except Exception:
                pass

        stop_event.wait(CONTROL_LOOP_PERIOD_S)


def copilot_audio_callback(in_data, frame_count, time_info, status):
    """Shield co-pilot's real-time audio thread. Same DSP-only discipline as
    audio_callback() (see its docstring) -- no MQTT publish happens here,
    just the lock-guarded state write; run_copilot_loop() does the actual
    publish on its own thread."""
    block = np.frombuffer(in_data, dtype=np.float32)
    cmd = policy.update(block, now=time.monotonic())

    with state_lock:
        shared["command"] = cmd
        shared["block"] = block

    return (None, pyaudio.paContinue)


def run_copilot_loop(stop_event):
    """Shield co-pilot's dedicated thread -- the only place that publishes to
    CONTROL_TOPIC, keeping even a fast MQTT publish off the audio callback
    thread (same reasoning as run_control_loop(), just for a much lighter
    workload since there's no BLE hardware involved on this instance).
    Publishes every tick, like run_control_loop()'s movement_move_tank, so
    the drive operator's shield continuously tracks this whistle's pitch."""
    while not stop_event.is_set():
        with state_lock:
            cmd = shared["command"]

        if cmd is not None:
            text = json.dumps({"shield_gradient": cmd.turn_bias})
            mqtt.publish(CONTROL_TOPIC, text)
            with state_lock:
                shared["last_shield_position"] = cmd.turn_bias  # for this dashboard's own display

        stop_event.wait(CONTROL_LOOP_PERIOD_S)


def _on_dashboard_key(event):
    """Local-testing hotkeys, for driving/exercising the game logic without a
    second machine publishing real MQTT messages. Every use is logged into the
    same mqtt_log feed as real traffic, tagged 'local', so it's never confused
    with an actual message from the opponent/instructor."""
    if event.key == "s":
        with state_lock:
            already_over = shared["game_over"]
            if not already_over:
                shared["game_active"] = True
        if not already_over:
            _log_mqtt("local", "start  (simulated with 's' -- no real MQTT message was sent)")
            print("[LOCAL TEST] Simulated 'start'.")
    elif event.key == "r":
        with state_lock:
            shared["game_active"] = False
            shared["game_over"] = False
            shared["game_over_reason"] = None
        _log_mqtt("local", "reset  (simulated with 'r' -- local test only)")
        print("[LOCAL TEST] Reset game state.")


def run_live_plot(mode):
    """A dark, dashboard-style live view: game status, waveform, spectrum
    (with calibrated bands), the whistle policy's decision, and a live feed
    of every MQTT message sent/received.

    `mode` is `"drive"` (full dashboard: drive gauge, proximity gauge, local-
    test hotkeys 's'/'r' -- see _on_dashboard_key()) or `"shield"` (co-pilot:
    no hardware to show a drive/proximity reading for, and the hotkeys aren't
    wired up since this instance never truly starts/ends the match)."""
    plt.style.use("dark_background")
    detector_freqs = policy.detector.freqs

    fig = plt.figure(figsize=(14, 9))
    mode_label = "DRIVE" if mode == "drive" else "SHIELD CO-PILOT"
    fig.canvas.manager.set_window_title(f"World Cup dashboard [{mode_label}] -- role={ROLE} team={TEAM_NAME}")
    gs = fig.add_gridspec(4, 3, height_ratios=(0.45, 1.3, 1.0, 1.4),
                           hspace=0.6, wspace=0.35, left=0.06, right=0.97, top=0.95, bottom=0.04)

    ax_status = fig.add_subplot(gs[0, :])
    ax_wave = fig.add_subplot(gs[1, 0])
    ax_spec = fig.add_subplot(gs[1, 1:])
    ax_left = fig.add_subplot(gs[2, 0])   # drive gauge (drive mode) or co-pilot info (shield mode)
    ax_mid = fig.add_subplot(gs[2, 1])    # proximity gauge (drive mode) or relay target (shield mode)
    ax_decision = fig.add_subplot(gs[2, 2])
    ax_mqtt = fig.add_subplot(gs[3, :])

    # --- status bar: the single most important thing on screen -----------
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    status_text = ax_status.text(0.02, 0.5, "", fontsize=15, fontweight="bold",
                                  va="center", ha="left", color="white")
    hint = ("[s] simulate start   [r] reset   (local testing only)" if mode == "drive"
            else f"co-pilot -- relaying shield commands to '{CONTROL_TOPIC}'")
    ax_status.text(0.98, 0.5, hint, fontsize=9, va="center", ha="right", color="#bbbbbb")

    # --- waveform -----------------------------------------------------------
    wave_line, = ax_wave.plot(np.zeros(BLOCK_SIZE), color="#39d6ff", linewidth=0.8)
    ax_wave.set_ylim(-1.0, 1.0)
    ax_wave.set_title("Microphone waveform", fontsize=10)
    ax_wave.set_xticks([])

    # --- spectrum, with the calibrated bands shaded -------------------------
    spec_line, = ax_spec.plot(detector_freqs, np.zeros_like(detector_freqs), color="#39d6ff")
    ax_spec.set_xlim(0, max(4500, whistle_config["forward_band"][1] + 500))
    ax_spec.set_title("Spectrum (calibrated bands shaded)", fontsize=10)
    ax_spec.set_xlabel("Hz")

    band_colors = {"stop_band": "#4da3ff", "turn_band": "#ffb84d", "forward_band": "#4dff88"}
    for key, color in band_colors.items():
        lo, hi = whistle_config[key]
        ax_spec.axvspan(lo, hi, color=color, alpha=0.18, label=key.replace("_band", ""))
    ax_spec.legend(loc="upper right", fontsize=8, framealpha=0.3)

    drive_bars = None
    prox_bar = None
    if mode == "drive":
        # --- drive gauge: exactly what's being sent to movement_move_tank ---
        ax_left.set_xlim(-100, 100)
        ax_left.set_xticks([-100, -50, 0, 50, 100])
        ax_left.set_title("Drive command sent to car (%)", fontsize=10)
        ax_left.axvline(0, color="#888888", linewidth=0.8)
        drive_bars = ax_left.barh(["right", "left"], [0, 0], color="#4dff88", height=0.5)

        # --- proximity gauge: the "caught" sensor, vs. its threshold --------
        # 0-100, not 0-255 -- see PROXIMITY_REFLECTION_THRESHOLD's comment
        # above for why (LELIB.md's documented 0-255 range doesn't match
        # real hardware).
        ax_mid.set_xlim(0, 100)
        ax_mid.set_ylim(0, 1)
        ax_mid.set_yticks([])
        ax_mid.set_title("Front sensor reflection (0-100)", fontsize=10)
        ax_mid.axvline(PROXIMITY_REFLECTION_THRESHOLD, color="#ff4d4d", linewidth=1.4, linestyle="--")
        prox_bar = ax_mid.barh([0.5], [0], height=0.6, color="#4dff88")[0]
    else:
        # Shield co-pilot: no local hardware to show a drive/proximity reading
        # for -- replace those two panels with what this instance actually does.
        ax_left.axis("off")
        ax_left.text(0.5, 0.5, "SHIELD CO-PILOT\n(no local hardware)", ha="center", va="center",
                     fontsize=12, fontweight="bold", color="#ffb84d")

        ax_mid.axis("off")
        ax_mid.set_title("Relaying shield gradient to", fontsize=10, loc="left")
        ax_mid.text(0.0, 0.6, CONTROL_TOPIC, fontsize=9, family="monospace", color="white")

    # --- whistle-policy decision panel ---------------------------------------
    ax_decision.axis("off")
    ax_decision.set_title("Whistle decision", fontsize=10, loc="left")
    decision_text = ax_decision.text(0.0, 0.85, "", fontsize=10, va="top", ha="left",
                                      family="monospace", color="white")

    # --- MQTT traffic feed: exactly what's sent and what's received --------
    ax_mqtt.axis("off")
    ax_mqtt.set_title(f"MQTT traffic on '{MQTT_TOPIC}' (newest first)", fontsize=10, loc="left")
    mqtt_text = ax_mqtt.text(0.0, 0.95, "", fontsize=9, va="top", ha="left",
                              family="monospace", color="white")

    if mode == "drive":
        fig.canvas.mpl_connect("key_press_event", _on_dashboard_key)

    def update(_frame):
        with state_lock:
            cmd = shared["command"]
            block = shared["block"]
            game_active = shared["game_active"]
            game_over = shared["game_over"]
            game_over_reason = shared["game_over_reason"]
            last_drive = shared["last_drive"]
            last_shield_position = shared["last_shield_position"]
            last_proximity = shared["last_proximity"]
            mqtt_log = list(shared["mqtt_log"])

        # status bar
        if game_over:
            color, label = "#5a2d7a", f"GAME OVER -- {game_over_reason or 'unknown reason'}"
        elif game_active:
            color, label = "#1f6f43", "LIVE"
        else:
            color, label = "#555555", "WAITING FOR 'start'"
        ax_status.set_facecolor(color)
        opponent_str = OPPONENT_TEAM_NAME or "none set"
        status_text.set_text(
            f"ROLE: {ROLE.upper()}   TEAM: {TEAM_NAME}   OPPONENT: {opponent_str}   ●  {label}")

        # waveform + spectrum
        wave_line.set_ydata(block)
        if cmd is not None:
            spec_line.set_ydata(cmd.spectrum)
            top = float(np.max(cmd.spectrum)) if cmd.spectrum.size else 1.0
            ax_spec.set_ylim(0, max(top * 1.2, 1.0))

        if mode == "drive":
            # drive gauge (index 0 = "right", index 1 = "left", per the barh() call above)
            right, left = last_drive[1], last_drive[0]
            drive_bars[0].set_width(right)
            drive_bars[0].set_color("#4dff88" if right >= 0 else "#ff6b6b")
            drive_bars[1].set_width(left)
            drive_bars[1].set_color("#4dff88" if left >= 0 else "#ff6b6b")

            # proximity gauge
            if last_proximity is not None:
                prox_bar.set_width(last_proximity)
                prox_bar.set_color("#ff4d4d" if last_proximity >= PROXIMITY_REFLECTION_THRESHOLD else "#4dff88")

        # decision panel
        if cmd is not None:
            freq_str = f"{cmd.frequency:.0f} Hz" if cmd.frequency else "--"
            if mode == "drive":
                extra = (f"Shield pos:  {last_shield_position:.0f} deg (0-{SHIELD_SWING_DEGREES}, "
                         f"from co-pilot)\n"
                         f"Drive sent:  L={last_drive[0]:+.0f}%  R={last_drive[1]:+.0f}%")
            else:
                extra = (f"Shield gradient sent: {last_shield_position:+.0f}\n"
                         f"  (-100=retracted .. +100=deployed) to:\n  {CONTROL_TOPIC}")
            decision_text.set_text(
                f"Frequency:   {freq_str}\n"
                f"Tonal ratio: {cmd.tonal_ratio:.1f}  (gate {whistle_config['tonal_ratio_gate']})\n"
                f"State:       {cmd.state}\n"
                f"{extra}")
        else:
            decision_text.set_text("(warming up...)")

        # MQTT traffic feed
        if mqtt_log:
            arrows = {"sent": "-> SENT ", "recv": "<- RECV ", "local": "** LOCAL"}
            lines = []
            for entry in reversed(mqtt_log):
                ts = time.strftime("%H:%M:%S", time.localtime(entry["t"]))
                lines.append(f"{ts}  {arrows[entry['dir']]}  {entry['payload']}")
            mqtt_text.set_text("\n".join(lines))
        else:
            mqtt_text.set_text("(no MQTT traffic yet -- waiting for 'start' on "
                                f"'{MQTT_TOPIC}', or press 's' to simulate it locally)")

        # blit=False below means this return value isn't actually used for
        # rendering, but keep it valid regardless of mode (drive_bars/prox_bar
        # are None in "shield" mode -- see their setup above).
        extra_artists = (prox_bar, *drive_bars) if mode == "drive" else ()
        return (wave_line, spec_line, status_text, decision_text, mqtt_text, *extra_artists)

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    return ani  # keep a reference alive for the duration of plt.show()


def run_setup_dialog():
    """Collect role / control-channel / team name / opponent team name before
    anything else starts (MQTT, BLE, audio) -- a small modal tkinter window,
    same pattern already used in this repo for a startup prompt
    (mqtt_chat.py's username dialog). Returns (role, channel, team_name,
    opponent_team_name). Closing the window (X) exits the whole program
    instead of proceeding with nothing selected."""
    result = {}

    root = tk.Tk()
    root.title("World Cup -- setup")
    root.resizable(False, False)

    role_var = tk.StringVar(value="ball")
    channel_var = tk.StringVar(value="drive")
    team_var = tk.StringVar(value=DEFAULT_TEAM_NAME)
    opponent_var = tk.StringVar(value="")

    pad = {"padx": 16, "pady": (10, 2)}

    tk.Label(root, text="Game role", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", **pad)
    tk.Radiobutton(root, text="Ball", variable=role_var, value="ball").grid(row=1, column=0, sticky="w", padx=32)
    tk.Radiobutton(root, text="Goalie", variable=role_var, value="goalie").grid(row=2, column=0, sticky="w", padx=32)

    tk.Label(root, text="This computer controls", font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w", **pad)
    tk.Radiobutton(root, text="Drive (forward/turn) -- connects to the robot",
                   variable=channel_var, value="drive").grid(row=4, column=0, sticky="w", padx=32)
    tk.Radiobutton(root, text="Shield only (remote co-pilot) -- no robot connection",
                   variable=channel_var, value="shield").grid(row=5, column=0, sticky="w", padx=32)

    tk.Label(root, text="Team name (must match your co-pilot's exactly)",
             font=("Segoe UI", 10, "bold")).grid(row=6, column=0, sticky="w", **pad)
    team_entry = tk.Entry(root, textvariable=team_var, width=30)
    team_entry.grid(row=7, column=0, sticky="w", padx=32)

    tk.Label(root, text="Opponent's team name (optional -- the name THEY enter above on"
                        "\ntheir own computer; leave blank to skip auto win/lose reactions)",
             font=("Segoe UI", 10, "bold"), justify="left").grid(row=8, column=0, sticky="w", **pad)
    opponent_entry = tk.Entry(root, textvariable=opponent_var, width=30)
    opponent_entry.grid(row=9, column=0, sticky="w", padx=32)

    error_label = tk.Label(root, text="", fg="red")
    error_label.grid(row=10, column=0, sticky="w", padx=32)

    def on_start():
        team = team_var.get().strip()
        if not team:
            error_label.config(text="Team name can't be empty.")
            return
        opponent = opponent_var.get().strip()
        if opponent and opponent == team:
            error_label.config(text="Opponent's team name can't be the same as your own.")
            return
        result["role"] = role_var.get()
        result["channel"] = channel_var.get()
        result["team_name"] = team
        result["opponent_team_name"] = opponent  # "" means "don't auto-react"
        root.destroy()

    tk.Button(root, text="Start", command=on_start, width=14).grid(row=11, column=0, pady=14)

    def on_close():
        root.destroy()  # result stays empty -- checked below

    root.protocol("WM_DELETE_WINDOW", on_close)
    team_entry.focus_set()
    root.mainloop()

    if "role" not in result:
        print("Setup cancelled -- exiting.")
        sys.exit(0)

    return result["role"], result["channel"], result["team_name"], result["opponent_team_name"]


def _run_as_drive_operator():
    """CONTROL_CHANNEL == 'drive': today's full flow -- connects to all three
    BLE devices, drives via audio_callback/run_control_loop -- plus one
    addition: also subscribes to CONTROL_TOPIC so a remote shield co-pilot's
    whistle can continuously drive this car's shield position too."""
    global car, shield, sensor, mqtt

    print("Make sure the opponent team has agreed on the same MQTT event schema "
          "(see this project's README) before the match.")

    pa = pyaudio.PyAudio()
    device_index = pick_mic(pa)

    mqtt = MQTTClient()
    mqtt.connect()
    mqtt.subscribe(MQTT_TOPIC, on_mqtt_message)
    mqtt.subscribe(CONTROL_TOPIC, on_control_message)
    print(f"[MQTT] Subscribed to {MQTT_TOPIC} and {CONTROL_TOPIC}. Waiting for 'start'...")

    car = doubleMotor()
    shield = singleMotor()
    sensor = colorSensor()

    stream = None
    control_thread = None
    stop_event = threading.Event()
    try:
        print("Connecting to Double Motor (drive)...")
        car.connect(card_serial=CAR_CARD_SERIAL, card_color=CAR_CARD_COLOR)
        print("Connecting to Single Motor (shield)...")
        shield.connect(card_serial=SHIELD_CARD_SERIAL, card_color=SHIELD_CARD_COLOR)
        print("Connecting to Color Sensor (front light sensor)...")
        sensor.connect(card_serial=SENSOR_CARD_SERIAL, card_color=SENSOR_CARD_COLOR)

        if not (car.connected and shield.connected and sensor.connected):
            raise ConnectionError("One or more devices did not connect.")

        control_thread = threading.Thread(target=run_control_loop, args=(stop_event,), daemon=True)
        control_thread.start()

        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                          input_device_index=device_index, frames_per_buffer=BLOCK_SIZE,
                          stream_callback=audio_callback)
        stream.start_stream()

        run_live_plot(mode="drive")  # blocks until the plot window is closed
    finally:
        stop_event.set()
        if control_thread is not None:
            control_thread.join(timeout=2 * CONTROL_LOOP_PERIOD_S)
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()
        # Hard stop before disconnecting -- independent of whatever game state we
        # were in when the window closed / an exception hit. movement_stop(), not
        # lelib's car.stop() (which only stops the left motor -- see
        # run_control_loop()'s comment).
        try:
            car.movement_stop()
        except Exception:
            pass
        try:
            shield.stop()
        except Exception:
            pass
        car.disconnect()
        shield.disconnect()
        sensor.disconnect()
        mqtt.disconnect()


def _run_as_copilot():
    """CONTROL_CHANNEL == 'shield': no BLE connection at all on this
    computer. Opens the mic, runs a WhistlePolicy tuned for shield
    sensitivity (shield_whistle_config -- full gradient range, no dead
    zone, unlike the drive's gentler tuning) purely to read this whistler's
    own turn_bias gradient, and continuously relays it to CONTROL_TOPIC for
    a drive-operator instance (on a different computer) to apply to its
    shield. Also mirrors game status from MQTT_TOPIC, read-only, for
    display."""
    global mqtt, policy

    # Swap the module-level `policy` (shared by copilot_audio_callback) to the
    # shield-sensitive config before the audio stream starts -- must happen
    # before any thread that reads `policy` is running.
    policy = WhistlePolicy(shield_whistle_config)

    print(f"Running as SHIELD CO-PILOT for team '{TEAM_NAME}' -- no robot connection "
          f"on this computer.")

    pa = pyaudio.PyAudio()
    device_index = pick_mic(pa)

    mqtt = MQTTClient()
    mqtt.connect()
    mqtt.subscribe(MQTT_TOPIC, on_mqtt_message_readonly)
    print(f"[MQTT] Subscribed to {MQTT_TOPIC} (read-only). "
          f"Relaying shield commands to {CONTROL_TOPIC}.")

    stream = None
    copilot_thread = None
    stop_event = threading.Event()
    try:
        copilot_thread = threading.Thread(target=run_copilot_loop, args=(stop_event,), daemon=True)
        copilot_thread.start()

        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                          input_device_index=device_index, frames_per_buffer=BLOCK_SIZE,
                          stream_callback=copilot_audio_callback)
        stream.start_stream()

        run_live_plot(mode="shield")  # blocks until the plot window is closed
    finally:
        stop_event.set()
        if copilot_thread is not None:
            copilot_thread.join(timeout=2 * CONTROL_LOOP_PERIOD_S)
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()
        mqtt.disconnect()


def main():
    global ROLE, TEAM_NAME, OPPONENT_TEAM_NAME, CONTROL_CHANNEL, CONTROL_TOPIC

    ROLE, CONTROL_CHANNEL, TEAM_NAME, OPPONENT_TEAM_NAME = run_setup_dialog()
    CONTROL_TOPIC = f"{MQTT_TOPIC}/control/{TEAM_NAME}"

    print(f"Role: {ROLE}   Channel: {CONTROL_CHANNEL}   Team: {TEAM_NAME}   "
          f"Opponent: {OPPONENT_TEAM_NAME or '(none set -- no auto win/lose reactions)'}")
    print(f"MQTT game topic: {MQTT_TOPIC}   Control topic: {CONTROL_TOPIC}")

    if CONTROL_CHANNEL == "drive":
        _run_as_drive_operator()
    else:
        _run_as_copilot()


if __name__ == "__main__":
    main()
