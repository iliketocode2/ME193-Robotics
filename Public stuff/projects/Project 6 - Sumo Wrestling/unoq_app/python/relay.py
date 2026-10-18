"""
relay.py -- MQTT <-> BLE relay for the sumo robot's LEGO hardware.

  T/cmd   (MQTT)  -> doubleMotor.movement_move_tank(l, r)   (BLE)
  T/song  (MQTT)  -> stop + beep SUCCESS_SONG / DEATH_SONG   (BLE)
  colorSensor.sensor notifications (BLE) -> T/sensor (MQTT), ~20 Hz
  link health                            -> T/status (MQTT), 1 Hz

The SAME file runs in three places, so there is only one copy of the logic:
  1. inside the UNO Q App Lab app (python/main.py imports RobotRelay),
  2. directly on the UNO Q's Linux host over SSH, if the App Lab container
     can't reach Bluetooth:   python3 relay.py
  3. on the laptop as the fallback (../../ble_relay_laptop.py).

Threading rules (see the mqtt-game-protocol skill): paho's callback thread
only stores the latest command under a lock. Every BLE call happens on the
one control-loop thread in run(), so BLE calls never race each other.

Safety: if no T/cmd arrives for CMD_TIMEOUT_S the motors stop, so a dead
laptop, dead Wi-Fi or crashed brain can never leave the robot driving.
Every stop uses movement_stop() -- lelib's doubleMotor.stop() only stops the
LEFT motor (it calls motor_stop() with the default motor=0).

Never block the loop for long while the motors could be moving:
  * drive commands use blocking=False (a lost BLE reply can't hang the loop),
  * the FIRST connect uses lelib's 5x retry (robot is stopped then), but a
    mid-run RECONNECT is one short scan, only while no fresh command is
    arriving for the color sensor, and after an explicit movement_stop().
    It bypasses lelib's connect() on purpose: that retry can block ~50 s.
"""

import json
import math
import socket
import threading
import time

import legoeducation as le
import paho.mqtt.client as mqtt

from lelib import colorSensor, doubleMotor
from sumo_topics import (BROKER_HOST, BROKER_PORT, CMD_TIMEOUT_S, COLOR_CARD_COLOR,
                         COLOR_CARD_SERIAL, DEATH_SONG, MOTOR_CARD_COLOR, MOTOR_CARD_SERIAL,
                         SUCCESS_SONG, TOPIC_CMD, TOPIC_SENSOR, TOPIC_SONG, TOPIC_STATUS)

LOOP_HZ = 20
SENSOR_NOTIFY_MS = 30          # ask the color sensor for a notification every 30 ms (allowed 15-1000)
RECONNECT_SCAN_S = 1.5         # one short BLE scan per reconnect attempt
RECONNECT_EVERY_S = 3.0
FLIP_DRIVE = False             # True if "forward" on the laptop drives the robot backward
SWAP_SIDES = False             # True if left/right turn the wrong way
BEEP_REST_S = 0.05             # same rest between notes as Project 3's songs.py

COLOR_NAMES = {
    le.LEGO_COLOR_NOCOLOR: "none", le.LEGO_COLOR_RED: "red", le.LEGO_COLOR_YELLOW: "yellow",
    le.LEGO_COLOR_BLUE: "blue", le.LEGO_COLOR_TEAL: "teal", le.LEGO_COLOR_GREEN: "green",
    le.LEGO_COLOR_PURPLE: "purple", le.LEGO_COLOR_WHITE: "white", le.LEGO_COLOR_MAGENTA: "magenta",
    le.LEGO_COLOR_ORANGE: "orange", le.LEGO_COLOR_AZURE: "azure",
}


def _num(v):
    """Sensor fields start as NaN before the first notification; JSON can't carry NaN."""
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


class RobotRelay:
    def __init__(self, host_label=None):
        self.host = host_label or socket.gethostname()
        self.lock = threading.Lock()
        self.cmd = {"l": 0, "r": 0, "t": 0.0}
        self.pending_song = None
        self.stop_event = threading.Event()
        self.motor = doubleMotor()
        self.color = colorSensor()
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_message
        self.mqtt.suppress_exceptions = True   # a callback error must never kill paho's network thread
        self.mqtt.reconnect_delay_set(min_delay=1, max_delay=10)

    # ----------------------------------------------------------- MQTT thread
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        # subscribe here so subscriptions come back after an automatic reconnect
        client.subscribe(TOPIC_CMD, qos=0)
        client.subscribe(TOPIC_SONG, qos=0)
        print(f"[relay] MQTT connected ({reason_code}); listening on {TOPIC_CMD}")

    def _on_message(self, client, userdata, msg):
        # Public broker: anyone can publish junk on our topics. Ignore anything
        # that isn't a well-formed command instead of letting it raise.
        try:
            data = json.loads(msg.payload.decode("utf-8", errors="replace"))
            if not isinstance(data, dict):
                return
            if msg.topic == TOPIC_CMD:
                cmd = {"l": _clamp(data.get("l", 0), -100, 100),
                       "r": _clamp(data.get("r", 0), -100, 100),
                       "t": time.monotonic()}
                with self.lock:
                    self.cmd = cmd
            elif msg.topic == TOPIC_SONG and data.get("song") in ("win", "lose"):
                with self.lock:
                    self.pending_song = data["song"]
        except (ValueError, TypeError, OverflowError):
            return

    # ------------------------------------------------------------ BLE helpers
    @staticmethod
    def _reconnect_quick(dev, raw_cls, serial, color, label):
        """One short scan + raw connect(device), bypassing lelib's slow retry."""
        try:
            found = dev.search(timeout=RECONNECT_SCAN_S, card_serial=serial, card_color=color)
            if found:
                raw_cls.connect(dev, found[0])
            if dev.connected:
                print(f"[relay] {label} reconnected")
                return True
        except Exception as e:
            print(f"[relay] {label} reconnect failed: {e}")
        return False

    def _try_connect(self):
        """First connect at startup (robot not moving yet): lelib's full retry."""
        if not self.motor.connected:
            try:
                self.motor.connect(card_serial=MOTOR_CARD_SERIAL, card_color=MOTOR_CARD_COLOR)
                self.motor.movement_set_acceleration(100, 100)
                self.motor.movement_stop()
                print("[relay] double motor connected")
            except Exception as e:
                print(f"[relay] double motor not connected: {e}")
        if not self.color.connected:
            try:
                self.color.connect(COLOR_CARD_SERIAL, card_color=COLOR_CARD_COLOR)
                self.color.device_notification_request(SENSOR_NOTIFY_MS)
                print("[relay] color sensor connected")
            except Exception as e:
                print(f"[relay] color sensor not connected: {e}")

    def _drive(self, left, right):
        if FLIP_DRIVE:
            left, right = -left, -right
        if SWAP_SIDES:
            left, right = right, left
        if left == 0 and right == 0:
            self.motor.movement_stop(blocking=False)
        else:
            self.motor.movement_move_tank(left, right, blocking=False)

    def _play_song(self, which):
        notes = SUCCESS_SONG if which == "win" else DEATH_SONG
        self.motor.movement_stop(blocking=False)
        for f in notes:
            try:
                self.motor.beep(pattern=le.SOUND_PATTERN_BEEP_SINGLE, frequency=f, blocking=True)
            except Exception as e:
                print(f"[relay] skipped note {f} Hz: {e}")
            time.sleep(BEEP_REST_S)

    def _sensor_msg(self, s):
        color = _num(s.color)
        return {
            "color": color,
            "name": COLOR_NAMES.get(color, "unknown") if color is not None else "unknown",
            "reflection": _num(s.reflection), "hue": _num(s.hue),
            "sat": _num(s.saturation), "val": _num(s.value),
            "rgb": [_num(s.rawRed), _num(s.rawGreen), _num(s.rawBlue)],
            "ts": time.time(),
        }

    # ------------------------------------------------------------ main loop
    def run(self):
        self.mqtt.connect_async(BROKER_HOST, BROKER_PORT)
        self.mqtt.loop_start()
        started = time.monotonic()
        applied = None                       # last (l, r) actually sent over BLE
        last_status = last_reconnect = 0.0
        last_reading = None                  # the .sensor object last published
        period = 1.0 / LOOP_HZ
        try:
            self._try_connect()
            while not self.stop_event.is_set():
                tick = time.monotonic()

                with self.lock:
                    cmd = dict(self.cmd)
                    song, self.pending_song = self.pending_song, None
                fresh = tick - cmd["t"] < CMD_TIMEOUT_S

                # Reconnect: the motor any time (if it's gone it isn't driving);
                # the color sensor only while idle, so losing it mid-match never
                # stalls driving or the watchdog.
                if tick - last_reconnect > RECONNECT_EVERY_S:
                    if not self.motor.connected:
                        last_reconnect = tick
                        applied = None
                        self._reconnect_quick(self.motor, le.DoubleMotor, MOTOR_CARD_SERIAL,
                                              MOTOR_CARD_COLOR, "double motor")
                    elif not self.color.connected and not fresh:
                        last_reconnect = tick
                        try:
                            self.motor.movement_stop(blocking=False)
                            applied = (0, 0)
                        except Exception:
                            applied = None
                        if self._reconnect_quick(self.color, le.ColorSensor, COLOR_CARD_SERIAL,
                                                 COLOR_CARD_COLOR, "color sensor"):
                            self.color.device_notification_request(SENSOR_NOTIFY_MS)
                    tick = time.monotonic()
                    fresh = tick - cmd["t"] < CMD_TIMEOUT_S
                target = (cmd["l"], cmd["r"]) if fresh else (0, 0)

                if self.motor.connected:
                    try:
                        if song:
                            self._play_song(song)
                            applied = (0, 0)
                        elif target != applied:
                            self._drive(*target)
                            applied = target
                    except Exception as e:
                        print(f"[relay] motor command failed: {e}")
                        applied = None

                # publish only NEW notifications (each one replaces .sensor), so a
                # stalled sensor goes stale on the laptop instead of looking fresh
                reading = self.color.sensor
                if self.color.connected and reading is not last_reading:
                    last_reading = reading
                    self.mqtt.publish(TOPIC_SENSOR, json.dumps(self._sensor_msg(reading)))

                if tick - last_status > 1.0:
                    last_status = tick
                    self.mqtt.publish(TOPIC_STATUS, json.dumps({
                        "motor": bool(self.motor.connected), "color": bool(self.color.connected),
                        "uptime": round(tick - started), "host": self.host,
                        "cmd_fresh": fresh}))

                time.sleep(max(0.0, period - (time.monotonic() - tick)))
        finally:
            # motors first, each step guarded, so nothing can skip the stop
            for dev, stop in ((self.motor, self.motor.movement_stop), (self.color, None)):
                try:
                    if dev.connected:
                        if stop:
                            stop()
                        dev.disconnect()
                except Exception as e:
                    print(f"[relay] cleanup: {e}")
            self.mqtt.loop_stop()
            self.mqtt.disconnect()
            print("[relay] stopped")


if __name__ == "__main__":
    # Standalone mode (UNO Q host over SSH, or laptop)
    relay = RobotRelay()
    try:
        relay.run()
    except KeyboardInterrupt:
        relay.stop_event.set()
