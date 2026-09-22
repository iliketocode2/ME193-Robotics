"""
Quick smoke test for mqttlib.py -- subscribes and publishes on the same
topic against the public test.mosquitto.org broker, confirming the round
trip actually works (the broker relays every publish back to any
subscriber on that topic, including yourself). See MQTTLIB.md.

Run:
    python "Public stuff/projects/mqtt_chat/mqtt_test.py"
"""

import os
import sys
import time

# mqttlib.py lives in the shared "useful libraries" folder, two levels
# up from this script (Public stuff/projects/mqtt_chat/) -- add it to
# sys.path so the import below resolves no matter where this script is
# run from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "useful libraries"))

from mqttlib import MQTTClient

TOPIC = "ME193"


def on_message(topic, payload):
    print(f"Got it back: [{topic}] {payload}")


with MQTTClient() as client:
    client.subscribe(TOPIC, on_message)
    time.sleep(1)  # give the subscription time to reach the broker

    client.publish(TOPIC, "WOAH I LOVE PIZZA")
    print(f"Published 'WOAH I LOVE PIZZA' to '{TOPIC}' on test.mosquitto.org")

    time.sleep(1)  # give the message time to come back before disconnecting
