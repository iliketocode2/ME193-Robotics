"""
UNO Q app (Linux side) for the sumo robot.

Always: reads the downforce fan's speed from the sketch (tach wire, see
sketch.ino) and publishes it as T/unoq {"fan_rpm", "relay_ble", "ts"} once a
second. The fan itself is wired straight to 5V, so it runs whenever the board
is powered; there's nothing to switch.

RELAY_BLE = True  (the plan): this app also talks to the LEGO double motor and
    color sensor over Bluetooth itself (relay.py).
RELAY_BLE = False (fallback, if the App Lab container can't reach
    Bluetooth -- run ble_probe_unoq.py to find out): relay.py runs somewhere
    else (on the UNO Q host via SSH, or on the laptop) and this app only
    reports the fan.
"""

import json
import signal
import threading
import time

import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge

from sumo_topics import BROKER_HOST, BROKER_PORT, TOPIC_UNOQ

# False: the App Lab container has no Bluetooth/D-Bus access (probe result,
# 2026-10-18), so the relay runs on the board's host via run_on_host.sh.
RELAY_BLE = False

current_relay = None
shutting_down = threading.Event()


def run_ble_relay():
    global current_relay
    from relay import RobotRelay  # imports legoeducation/bleak only when actually used
    while not shutting_down.is_set():
        try:
            current_relay = RobotRelay(host_label="unoq")
            current_relay.run()
        except Exception as e:
            print(f"[main] relay crashed, restarting in 3 s: {e}")
            time.sleep(3)


def run_fan_monitor():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    client.connect_async(BROKER_HOST, BROKER_PORT)
    client.loop_start()
    while not shutting_down.is_set():
        try:
            rpm = int(Bridge.call("fan_rpm"))
        except Exception as e:
            print(f"[main] fan_rpm read failed: {e}")
            rpm = None
        client.publish(TOPIC_UNOQ, json.dumps({"fan_rpm": rpm, "relay_ble": RELAY_BLE, "ts": time.time()}))
        time.sleep(1.0)
    client.loop_stop()
    client.disconnect()


def on_sigterm(signum, frame):
    # Stopping the app sends SIGTERM; daemon threads would just vanish without
    # their finally blocks, so stop the relay (movement_stop + disconnect) first.
    shutting_down.set()
    if current_relay is not None:
        current_relay.stop_event.set()
    if relay_thread is not None and relay_thread.is_alive():
        relay_thread.join(timeout=3)
    raise SystemExit(0)


threading.Thread(target=run_fan_monitor, daemon=True).start()
relay_thread = None
if RELAY_BLE:
    relay_thread = threading.Thread(target=run_ble_relay, daemon=True)
    relay_thread.start()
signal.signal(signal.SIGTERM, on_sigterm)
App.run()
