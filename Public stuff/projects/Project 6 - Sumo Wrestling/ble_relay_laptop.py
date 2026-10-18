"""
ble_relay_laptop.py -- fallback: run the robot's MQTT <-> BLE relay on the laptop.

Use this only if the UNO Q can't reach the LEGO devices over Bluetooth
(see unoq_app/python/ble_probe_unoq.py). It runs the exact same relay code
as the UNO Q (unoq_app/python/relay.py), with the same MQTT topics, so
sumo_brain.py doesn't know or care where the relay is running. Set
RELAY_BLE = False in unoq_app/python/main.py so the UNO Q only reports the fan RPM.

Run (in its own terminal, alongside sumo_brain.py):
    my_env\\Scripts\\python "Public stuff/projects/Project 6 - Sumo Wrestling/ble_relay_laptop.py"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "unoq_app", "python"))

from relay import RobotRelay  # noqa: E402

if __name__ == "__main__":
    relay = RobotRelay(host_label="laptop")
    try:
        relay.run()
    except KeyboardInterrupt:
        relay.stop_event.set()
