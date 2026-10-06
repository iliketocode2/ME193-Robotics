"""
The LEGO Double Motor as a handheld paddle.

  * IMU  -> swing detection. legoeducation updates dm.imu_device (gyro +
    accelerometer) every time the hub sends a notification; we just read
    the attribute. A swing = gyro magnitude rising past a threshold.
  * Motors + light -> haptic feedback (buzz on a hit, rumble on a miss).

All BLE traffic (reads AND commands) happens on this one thread, so the
camera loop and the 60 Hz game tick never wait on Bluetooth (same idea as
Project 2's _MotorDriver).

KeyboardPaddle is a drop-in stand-in (Space in the browser = swing) for
building/testing the game without the hardware: `pingpong.py --sim`.
"""

import math
import queue
import threading
import time

import legoeducation as le
from lelib import doubleMotor

from game_logic import Swing

# --- Swing detection -----------------------------------------------------------
# Gyro values are raw int16 from the hub -- the units aren't documented, so
# this threshold is tuned on the real device: press D in the game to see the
# live gyro magnitude, and [ / ] to lower / raise the threshold.
SWING_GYRO_THRESHOLD = 600
SWING_RELEASE_FRAC = 0.6       # swing ends when magnitude drops below threshold * this
SWING_REFRACTORY_S = 0.30      # one swing per this long, max
NOTIFICATION_MS = 30           # IMU update period requested from the hub (15-1000 ms;
                               # lelib's connect() leaves the 100 ms default -- too slow
                               # to catch a fast swing)
POLL_S = 0.005

# --- Haptics -------------------------------------------------------------------
# Hold the motor with NOTHING attached to the shafts -- they really spin.
HAPTIC_SPEED = 35              # percent; low on purpose
HIT_BUZZ_MS = 80
MISS_RUMBLE_MS = 400
HAPTIC_MUTE_PAD_S = 0.15       # ignore the IMU while (and just after) the motors buzz,
                               # so the paddle's own rumble isn't read as a swing


class MotorPaddle:
    def __init__(self, card_serial, card_color):
        self.dm = doubleMotor()
        self.card_serial = card_serial
        self.card_color = card_color
        self.threshold = SWING_GYRO_THRESHOLD
        self.gyro_mag = 0.0            # latest value, for the debug overlay
        self.connected = False
        self._swings = []
        self._lock = threading.Lock()
        self._commands = queue.Queue()
        self._mute_until = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="imu", daemon=True)

    # ---------------------------------------------------------- lifecycle
    def connect(self):
        self.dm.connect(card_serial=self.card_serial, card_color=self.card_color)
        if not self.dm.connected:
            raise ConnectionError("Double Motor did not connect")
        self.connected = True
        self.dm.device_notification_request(NOTIFICATION_MS)
        self.dm.light_color(le.LEGO_COLOR_BLUE)
        self._thread.start()

    def close(self):
        """Stop the IMU thread, stop both motors, disconnect -- each step on
        its own so one failure doesn't skip the rest."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        # dm.connected, not self.connected: the library can be connected even
        # if connect() raised partway through. BaseException so a second
        # Ctrl-C here can't skip the motor stop / disconnect.
        if not self.dm.connected:
            return
        try:
            self.dm.motor_stop(motor=le.MOTOR_BOTH)
        except BaseException as e:
            print(f"motor_stop failed: {e!r}")
        try:
            self.dm.disconnect()
        except BaseException as e:
            print(f"disconnect failed: {e!r}")

    # ---------------------------------------------------------- game API
    def take_swings(self):
        """Swings detected since the last call (list of game_logic.Swing)."""
        with self._lock:
            out, self._swings = self._swings, []
        return out

    def feedback(self, kind):
        """Queue haptics: 'hit', 'miss', 'point', 'win', 'lose', 'idle'."""
        self._commands.put(kind)

    def nudge_threshold(self, factor):
        self.threshold = max(50, self.threshold * factor)

    # ---------------------------------------------------------- IMU thread
    def _run(self):
        last_sample = None
        last_gesture = self.dm.imu_gesture
        last_gesture_value = None
        in_swing = False
        last_swing_t = 0.0
        while not self._stop.is_set():
            self._send_commands()

            sample = self.dm.imu_device
            now = time.monotonic()
            if sample is not last_sample:          # a new notification arrived
                last_sample = sample
                g = (sample.gyroscopeX, sample.gyroscopeY, sample.gyroscopeZ)
                if not any(math.isnan(v) for v in g):
                    mag = math.sqrt(sum(v * v for v in g))
                    self.gyro_mag = mag
                    if now < self._mute_until:
                        in_swing = False
                    elif not in_swing and mag >= self.threshold and now - last_swing_t >= SWING_REFRACTORY_S:
                        in_swing, last_swing_t = True, now
                        self._add_swing(now, mag / self.threshold)
                    elif in_swing and mag < self.threshold * SWING_RELEASE_FRAC:
                        in_swing = False

            # Backup signal: the hub's own SHAKE gesture. Only a CHANGE to
            # SHAKE counts, in case the hub keeps re-reporting the last gesture.
            gesture = self.dm.imu_gesture
            if gesture is not last_gesture:
                last_gesture = gesture
                value = gesture.gesture
                if (value == le.MOTION_GESTURE_SHAKE and last_gesture_value != le.MOTION_GESTURE_SHAKE
                        and now >= self._mute_until and now - last_swing_t >= SWING_REFRACTORY_S):
                    last_swing_t = now
                    self._add_swing(now, 1.0)
                last_gesture_value = value

            time.sleep(POLL_S)

    def _add_swing(self, t, strength):
        with self._lock:
            self._swings.append(Swing(t, strength))

    def _send_commands(self):
        while True:
            try:
                kind = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._haptic(kind)
            except Exception as e:      # a dropped haptic must never kill swing detection
                print(f"haptic '{kind}' failed: {e}")

    def _haptic(self, kind):
        dm = self.dm
        if kind in ("hit", "miss"):
            ms = HIT_BUZZ_MS if kind == "hit" else MISS_RUMBLE_MS
            self._mute_until = time.monotonic() + ms / 1000 + HAPTIC_MUTE_PAD_S
        if kind == "hit":
            dm.light_color(le.LEGO_COLOR_GREEN, blocking=False)
            dm.motor_run_for_time(HIT_BUZZ_MS, motor=le.MOTOR_BOTH, speed=HAPTIC_SPEED, blocking=False)
        elif kind == "miss":
            dm.light_color(le.LEGO_COLOR_RED, blocking=False)
            dm.motor_run_for_time(MISS_RUMBLE_MS, motor=le.MOTOR_BOTH, speed=HAPTIC_SPEED, blocking=False)
        elif kind == "point":
            dm.light_color(le.LEGO_COLOR_YELLOW, blocking=False)
        elif kind == "win":
            dm.light_color(le.LEGO_COLOR_GREEN, blocking=False)
            dm.beep(frequency=880, count=3, blocking=False)
        elif kind == "lose":
            dm.light_color(le.LEGO_COLOR_RED, blocking=False)
            dm.beep(frequency=220, count=1, blocking=False)
        elif kind == "idle":
            dm.light_color(le.LEGO_COLOR_BLUE, blocking=False)


class KeyboardPaddle:
    """Same interface as MotorPaddle; swings come from the browser's Space key."""

    def __init__(self):
        self.threshold = SWING_GYRO_THRESHOLD
        self.gyro_mag = 0.0
        self.connected = False
        self._swings = []
        self._lock = threading.Lock()

    def connect(self):
        print("[sim] No Double Motor -- press SPACE in the browser to swing.")

    def close(self):
        pass

    def key_swing(self):
        with self._lock:
            self._swings.append(Swing(time.monotonic(), 1.2))

    def take_swings(self):
        with self._lock:
            out, self._swings = self._swings, []
        return out

    def feedback(self, kind):
        pass

    def nudge_threshold(self, factor):
        pass
