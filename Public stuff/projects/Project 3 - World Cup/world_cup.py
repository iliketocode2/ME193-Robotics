"""
world_cup.py -- whistle-controlled LEGO Education "car" for the ME193
World Cup match: drive by whistling into the laptop mic (pitch -> speed/
turn, rhythm -> shield toggle / goal command), coordinated with an
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
from songs import DEATH_SONG, SUCCESS_SONG, play_song
from whistle_policy import DEFAULT_CONFIG, WhistlePolicy

# --- Match-day placeholders -- fill these in before running -----------------
ROLE = "ball"  # "ball" or "goalie" -- set to whatever role you're assigned on match day
TEAM_NAME = "Cucurella"  # used only for logging, not part of the MQTT protocol

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
INVERT_LEFT_MOTOR = False
INVERT_RIGHT_MOTOR = False

# --- Shield tuning -------------------------------------------------------------
SHIELD_SWING_DEGREES = 90  # how far the shield arm swings between retracted/deployed

# --- Proximity ("caught") tuning ----------------------------------------------
# reflection() is 0-255; higher = more light reflected = something close/bright
# in front of the sensor. This has to be calibrated on-site against the actual
# opponent robot and room lighting on match day -- it is NOT a safe default to
# trust as-is.
PROXIMITY_REFLECTION_THRESHOLD = 200

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

policy = WhistlePolicy(whistle_config)

# --- Shared state between the audio callback thread, the MQTT thread, the
# control thread, and the main/plotting thread. Only ever mutated/read under
# `state_lock`. See audio_callback()'s and run_control_loop()'s docstrings for
# why BLE calls live on their own dedicated thread, not the audio callback. --
state_lock = threading.Lock()
shared = {
    "command": None,     # most recent whistle_policy.Command
    "block": np.zeros(BLOCK_SIZE, dtype=np.float32),
    "game_active": False,       # True once "start" arrives on MQTT_TOPIC
    "game_over": False,         # True once this robot has failed or scored
    "shield_toggle_pending": False,   # set by the audio callback, consumed by the control loop
    "goal_pending": False,            # set by the audio callback, consumed by the control loop
    "remote_result_pending": None,    # set by on_mqtt_message, consumed by the control loop: (song, message) or None
}

CONTROL_LOOP_PERIOD_S = 0.05  # ~20Hz -- plenty responsive for a human whistling, well clear of BLE round-trip time

# Only the control loop thread ever touches this -- no lock needed (single writer/reader).
shield_deployed = False

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


def toggle_shield():
    global shield_deployed
    direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if not shield_deployed
                 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
    shield.motor_run_for_degrees(SHIELD_SWING_DEGREES, direction=direction, blocking=False)
    shield_deployed = not shield_deployed
    print(f"[shield] {'Deploying' if shield_deployed else 'Retracting'}")


def announce_result(song, message):
    """Only ever called from run_control_loop()'s thread -- both car.movement_stop()
    and play_song()'s sequence of blocking beep() calls do real BLE round-trips,
    which must never run on the PyAudio callback thread or the MQTT network thread."""
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
    play_song(car, song)


def handle_caught():
    """This robot's own front sensor detected the opponent close up -- only
    meaningful for the ball (see the README's proximity/fail rule)."""
    mqtt.publish(MQTT_TOPIC, json.dumps({"event": "fail", "team": ROLE}))
    announce_result(DEATH_SONG, "Caught! You failed.")


def handle_scored():
    """This robot's own whistle policy fired the goal gesture -- only
    meaningful for the ball."""
    mqtt.publish(MQTT_TOPIC, json.dumps({"event": "goal", "team": ROLE}))
    announce_result(SUCCESS_SONG, "GOAL! You scored.")


def on_mqtt_message(topic, payload):
    text = payload.strip()

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
    if team == ROLE:
        return  # our own publish, echoed back by the broker -- already handled locally

    # Queue the result rather than calling announce_result() directly: this callback
    # runs on paho's own network thread, and announce_result() does a real BLE stop
    # plus a multi-second sequence of blocking beep() calls -- blocking paho's thread
    # for that long risks missing keepalives/other messages. The control loop thread
    # picks this up within one CONTROL_LOOP_PERIOD_S tick.
    if event == "fail" and team == "ball" and ROLE == "goalie":
        print("[MQTT] Ball reported it failed.")
        with state_lock:
            shared["remote_result_pending"] = (SUCCESS_SONG, "The ball got caught -- goalie wins!")
    elif event == "goal" and team == "ball" and ROLE == "goalie":
        print("[MQTT] Ball reported a goal.")
        with state_lock:
            shared["remote_result_pending"] = (DEATH_SONG, "The ball scored -- goalie loses!")


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
        if ROLE == "ball" and shared["game_active"] and not shared["game_over"]:
            if cmd.shield_toggle:
                shared["shield_toggle_pending"] = True
            if cmd.goal_detected:
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
            do_shield = shared["shield_toggle_pending"]
            shared["shield_toggle_pending"] = False
            do_goal = shared["goal_pending"]
            shared["goal_pending"] = False
            remote_result = shared["remote_result_pending"]
            shared["remote_result_pending"] = None

        try:
            if remote_result is not None:
                song, message = remote_result
                announce_result(song, message)
            elif game_active and not game_over and cmd is not None:
                left, right = _mix_tank_drive(cmd.forward_speed, cmd.turn_bias)
                car.movement_move_tank(left, right, blocking=False)

                if ROLE == "ball":
                    if do_shield:
                        toggle_shield()
                    if do_goal:
                        handle_scored()
                    elif sensor.reflection() >= PROXIMITY_REFLECTION_THRESHOLD:
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


def _band_edges():
    return (whistle_config["stop_band"], whistle_config["turn_band"], whistle_config["forward_band"])


def run_live_plot():
    detector_freqs = policy.detector.freqs
    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(9, 7))
    fig.canvas.manager.set_window_title(f"World Cup whistle control -- role={ROLE}")

    wave_line, = ax_wave.plot(np.zeros(BLOCK_SIZE))
    ax_wave.set_ylim(-1.0, 1.0)
    ax_wave.set_title("Microphone waveform")
    ax_wave.set_xlabel("sample")
    ax_wave.set_ylabel("amplitude")

    spec_line, = ax_spec.plot(detector_freqs, np.zeros_like(detector_freqs))
    ax_spec.set_xlim(0, max(4500, whistle_config["forward_band"][1] + 500))
    ax_spec.set_title("Spectrum (calibrated bands shown as shaded regions)")
    ax_spec.set_xlabel("Hz")
    ax_spec.set_ylabel("magnitude")

    band_colors = {"stop": "tab:blue", "turn": "tab:orange", "forward": "tab:green"}
    for label, key in (("stop", "stop_band"), ("turn", "turn_band"), ("forward", "forward_band")):
        lo, hi = whistle_config[key]
        ax_spec.axvspan(lo, hi, color=band_colors[label], alpha=0.15, label=f"{label} band")
    ax_spec.legend(loc="upper right", fontsize=8)

    hud_text = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    def update(_frame):
        with state_lock:
            cmd = shared["command"]
            block = shared["block"]
            game_active, game_over = shared["game_active"], shared["game_over"]

        wave_line.set_ydata(block)

        if cmd is not None:
            spec_line.set_ydata(cmd.spectrum)
            top = float(np.max(cmd.spectrum)) if cmd.spectrum.size else 1.0
            ax_spec.set_ylim(0, max(top * 1.2, 1.0))
            freq_str = f"{cmd.frequency:.0f} Hz" if cmd.frequency else "--"
            decision = cmd.state
            tonal_str = f"{cmd.tonal_ratio:.1f}"
        else:
            freq_str, decision, tonal_str = "--", "(warming up)", "--"

        status = "GAME OVER" if game_over else ("LIVE" if game_active else "WAITING FOR 'start'")
        hud_text.set_text(
            f"[{status}]  role={ROLE}  freq={freq_str}  tonal ratio={tonal_str}  ->  {decision}")

        return wave_line, spec_line, hud_text

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    return ani  # keep a reference alive for the duration of plt.show()


def main():
    global car, shield, sensor, mqtt

    print(f"Role: {ROLE}   Team: {TEAM_NAME}   MQTT topic: {MQTT_TOPIC}")
    print("Make sure the opponent team has agreed on the same MQTT event schema "
          "(see this project's README) before the match.")

    pa = pyaudio.PyAudio()
    device_index = pick_mic(pa)

    mqtt = MQTTClient()
    mqtt.connect()
    mqtt.subscribe(MQTT_TOPIC, on_mqtt_message)
    print(f"[MQTT] Subscribed to {MQTT_TOPIC}. Waiting for 'start'...")

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

        run_live_plot()  # blocks until the plot window is closed
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


if __name__ == "__main__":
    main()
