#!/usr/bin/env python3
"""Send jim a command over MQTT and print everything it says back.

    python tools/mqtt_cmd.py 'Type "Hi" on the LED matrix and then send back on MQTT on topic fred message "I am done"'
    python tools/mqtt_cmd.py /status
    python tools/mqtt_cmd.py --listen            just watch ME193/q/will/# (and topic fred)

Uses the same broker and transport as the assignment
(ws://test.mosquitto.org:8080). Listens on ME193/q/will/# plus any bare
topics given with --also (default: fred, will), so replies sent to a topic
the command names are shown too. Exits after --wait seconds (default 90) or
once a log record for the command arrives and a few more seconds pass.
"""

import argparse
import json
import sys
import time
import uuid

import paho.mqtt.client as mqtt

NAME = "will"
BROKER, PORT = "test.mosquitto.org", 8080


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?")
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--wait", type=float, default=90)
    ap.add_argument("--also", nargs="*", default=["fred", "will"])
    a = ap.parse_args()
    if not a.command and not a.listen:
        ap.error("give a command or --listen")

    t0 = time.monotonic()
    finished_at = [None]
    sent = [False]

    def on_connect(c, userdata, flags, rc, props):
        c.subscribe(f"ME193/q/{NAME}/#")
        for t in a.also:
            c.subscribe(t)
        if a.command and not sent[0]:          # only once, even if we reconnect
            sent[0] = True
            c.publish(f"ME193/q/{NAME}/cmd", a.command, qos=1)
            print(f"[{time.monotonic() - t0:6.1f}s] sent: {a.command}", flush=True)

    def on_message(c, userdata, msg):
        if msg.topic.endswith("/cmd"):
            return
        text = msg.payload.decode(errors="replace")
        if msg.topic.endswith("/log"):
            try:
                rec = json.loads(text)
                text = json.dumps(rec, indent=2)
                if a.command and rec.get("command") == a.command:
                    finished_at[0] = time.monotonic()
            except ValueError:
                pass
        elif a.command and a.command.startswith("/") and msg.topic.endswith("/status"):
            finished_at[0] = time.monotonic()
        tag = " (retained)" if msg.retain else ""
        print(f"[{time.monotonic() - t0:6.1f}s] {msg.topic}{tag}: {text}", flush=True)

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"laptop-{uuid.uuid4().hex[:8]}",
                    transport="websockets")
    c.on_connect, c.on_message = on_connect, on_message
    c.connect(BROKER, PORT, keepalive=30)
    c.loop_start()
    try:
        while True:
            time.sleep(0.2)
            if not a.listen and time.monotonic() - t0 > a.wait:
                print("(timed out)")
                break
            if finished_at[0] and time.monotonic() - finished_at[0] > 2:
                break
    except KeyboardInterrupt:
        pass
    c.loop_stop()


if __name__ == "__main__":
    sys.exit(main())
