"""
Live ASCII speedometer -- run this ON THE RASPBERRY PI, not on the
laptop driving the robot.

    python3 pi_dashboard.py

Listens for UDP telemetry sent by piviz.send_telemetry() (called from
arm_race_control.py on the laptop) and prints a live-updating dashboard:
finger count, speed as a bar, and which way you're pointing.

This has no dependency on anything else in this repo -- it only needs
Python's standard library, so it runs on the Pi as-is with no venv/pip
install step.
"""
import json
import socket

PORT = 5566

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
print(f"Listening for race-car telemetry on UDP port {PORT}... (Ctrl+C to stop)")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        msg = json.loads(data)

        speed = msg["speed"]
        forward, turn = msg["forward"], msg["turn"]

        bar_len = int(abs(speed) / 5)  # 0-20 chars for 0-100%
        bar = ("#" * bar_len).ljust(20)

        if abs(turn) > 0.3 and abs(turn) > abs(forward):
            arrow = "<" if turn < 0 else ">"
        elif abs(forward) > 0.1:
            arrow = "^" if forward > 0 else "v"
        else:
            arrow = "o"

        print(f"\rFingers:{msg['fingers']}  Speed:{speed:+6.1f}%  [{bar}]  {arrow}   ",
              end="", flush=True)
except KeyboardInterrupt:
    print("\nStopped.")
