"""
Optional fun add-on: fire-and-forget UDP telemetry to a Raspberry Pi
running pi_dashboard.py, for a live speedometer/direction display
separate from the laptop's own preview window.

Usage (from arm_race_control.py or similar):
    from piviz import send_telemetry
    send_telemetry(fingers=3, speed=42, forward=0.8, turn=-0.1)

Setup:
    1. Set PI_HOST below to your Pi's IP address or hostname. The
       default "raspberrypi.local" only works if mDNS resolution works
       on this machine (needs Bonjour on Windows) AND the Pi is
       reachable over some real network link -- most Raspberry Pi
       models' USB-C port is power-only, not a data connection, so
       plugging the Pi into this laptop via USB-C alone does NOT put
       it on the network. Get the Pi's IP by running `hostname -I` in
       a terminal on the Pi itself (it needs Wi-Fi or Ethernet).
    2. Copy pi_dashboard.py onto the Pi and run it there:
           python3 pi_dashboard.py
"""
import json
import socket

PI_HOST = "raspberrypi.local"  # <-- replace with your Pi's IP if this doesn't resolve
PI_PORT = 5566

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sock.setblocking(False)

# Resolve once at import time, not per-send -- sendto() with an
# unresolvable hostname would otherwise redo a DNS/mDNS lookup on every
# single call (every video frame), and a lookup that fails slowly could
# visibly stall the robot control loop. Resolving once and disabling
# telemetry immediately on failure keeps every later send_telemetry()
# call a cheap no-op instead.
try:
    _pi_addr = (socket.gethostbyname(PI_HOST), PI_PORT)
    _enabled = True
except OSError:
    print(f"piviz: could not resolve {PI_HOST!r} -- Pi telemetry disabled for this run. "
          f"Set PI_HOST in piviz.py to your Pi's IP address (find it with `hostname -I` "
          f"on the Pi) and restart.")
    _pi_addr = None
    _enabled = False


def send_telemetry(fingers, speed, forward, turn):
    """Best-effort, non-blocking UDP send. Does nothing if the Pi's
    hostname didn't resolve at import time, or if the send itself fails
    (Pi offline/asleep/unreachable) -- never raises, so a missing Pi can
    never crash or stall the robot control loop."""
    if not _enabled:
        return
    payload = json.dumps({
        "fingers": fingers,
        "speed": round(speed, 1),
        "forward": round(forward, 2),
        "turn": round(turn, 2),
    }).encode()
    try:
        _sock.sendto(payload, _pi_addr)
    except OSError:
        pass
