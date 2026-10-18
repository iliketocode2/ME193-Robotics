"""
test_sumo_offline.py -- check vision + fusion + behavior tree with NO hardware.

Draws synthetic overhead frames (floor, blue tape ellipse, our AprilTag at a
chosen spot/heading, a dark opponent block), runs them through the real
Vision and Brain.tick(), and checks the decisions -- including the sign of
turns, which is the easiest thing to get backwards. Nothing is sent to a
robot (MQTT is never connected), songs are stubbed out.

Run:
    my_env\\Scripts\\python "Public stuff/projects/Project 6 - Sumo Wrestling/test_sumo_offline.py"
"""

import math
import time

import cv2
import numpy as np

import config
import songs_bridge
import sumo_brain
from vision import Vision

W, H = 960, 540
RING = ((480, 270), (700, 460), 0)          # center, full axes, angle (a camera-tilted ellipse)
TAG_PX = 46
played = []
songs_bridge.play = lambda which, publish: played.append(which)
songs_bridge.stop = lambda: None


def ring_px(u, v):
    (cx, cy), (w, h), _ = RING
    return cx + u * w / 2, cy + v * h / 2


def make_frame(us=None, heading_deg=0.0, opp=None):
    """us/opp in ring coords (u, v); heading 0 = facing up the image, + = clockwise."""
    img = np.full((H, W, 3), (150, 160, 165), np.uint8)
    cv2.ellipse(img, RING[0], (RING[1][0] // 2, RING[1][1] // 2), RING[2], 0, 360, (200, 90, 20), 14)
    if opp is not None:
        x, y = ring_px(*opp)
        cv2.rectangle(img, (int(x - 30), int(y - 30)), (int(x + 30), int(y + 30)), (40, 40, 40), -1)
    if us is not None:
        d = cv2.aruco.getPredefinedDictionary(config.APRILTAG_DICTIONARY)
        tag = cv2.aruco.generateImageMarker(d, config.OUR_TAG_ID, TAG_PX)
        pad = TAG_PX // 3
        card = cv2.copyMakeBorder(tag, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)
        n = card.shape[0]
        x, y = ring_px(*us)
        M = cv2.getRotationMatrix2D((n / 2, n / 2), -heading_deg, 1.0)   # cv2: + = counter-clockwise
        M[:, 2] += (x - n / 2, y - n / 2)
        mask = cv2.warpAffine(np.full((n, n), 255, np.uint8), M, (W, H))
        warped = cv2.warpAffine(card, M, (W, H))
        img[mask > 0] = warped[mask > 0]
    return img


def run(brain, frame, sensor=None, ticks=8, dt=1 / 15):
    res = brain.vision.process(frame)
    res["size"] = [W, H]
    snap = None
    for _ in range(ticks):
        now = time.monotonic()
        with brain.lock:
            brain.result, brain.result_t = res, now
            brain.status, brain.status_t = {"motor": True, "color": True}, now
            if sensor is not None:
                brain.sensor, brain.sensor_t = {"color": 3 if sensor == "blue" else 7, "name": sensor}, now
                brain.rear_on_blue = sensor == "blue"
                if sensor == "blue":
                    brain.last_blue_t = now
        snap = brain.tick(now, dt)
        time.sleep(dt)                       # real time: out-of-ring timers count real frame time
    return res, snap


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        check.failed += 1


check.failed = 0


def main():
    print("vision")
    v = Vision()
    res = v.process(make_frame(us=(0, 0), heading_deg=0, opp=(0.5, 0)))
    ring = res["ring"]
    check("ring found", ring is not None)
    check("ring center", ring and abs(ring.cx - 480) < 15 and abs(ring.cy - 270) < 15,
          f"{ring and (round(ring.cx), round(ring.cy))}")
    check("our tag found", res["us"] is not None)
    if res["us"]:
        hx, hy = res["us"]["hx"], res["us"]["hy"]
        check("heading points up", hy < -0.95, f"({hx:.2f}, {hy:.2f})")
    check("opponent found (tag-only blob mode)", res["opp"] is not None)
    res90 = v.process(make_frame(us=(0, 0), heading_deg=90))
    if res90["us"]:
        check("heading 90 deg points right", res90["us"]["hx"] > 0.95, f"{res90['us']['hx']:.2f}")

    print("behavior tree")
    brain = sumo_brain.Brain()
    _, s = run(brain, make_frame(us=(0, 0), opp=(0.5, 0)))
    check("idle waits for start", s["action"] == "Waiting for start" and s["sent"] == [0, 0], s["action"])

    brain.command({"cmd": "start"})
    brain.vision.lock_ring(False)            # synthetic frames: let it keep fitting
    _, s = run(brain, make_frame(us=(0, 0), heading_deg=0, opp=(0.5, 0)), sensor="white")
    check("opponent to the right -> turn right (L>R)", "ATTACK" in s["action"] and s["sent"][0] > s["sent"][1],
          f"{s['action']} {s['sent']}")
    _, s = run(brain, make_frame(us=(0, 0), heading_deg=0, opp=(-0.5, 0)), sensor="white")
    check("opponent to the left -> turn left (R>L)", s["sent"][1] > s["sent"][0], f"{s['action']} {s['sent']}")
    _, s = run(brain, make_frame(us=(0, 0.1), heading_deg=0, opp=(0, -0.45)), sensor="white")
    check("opponent dead ahead -> charge", "ATTACK" in s["action"] and min(s["sent"]) > 40, f"{s['action']} {s['sent']}")

    _, s = run(brain, make_frame(us=(0.88, 0), heading_deg=90, opp=(-0.5, 0.3)), sensor="white")
    check("near edge facing out -> escape toward center (backs up)", "ESCAPE" in s["action"] and sum(s["sent"]) < 0,
          f"{s['action']} {s['sent']}")
    _, s = run(brain, make_frame(us=(0.3, 0), heading_deg=0, opp=(-0.5, 0.3)), sensor="blue")
    check("blue under rear sensor -> both wheels forward", "ESCAPE" in s["action"] and min(s["sent"]) > 0,
          f"{s['action']} {s['sent']}")

    brain.command({"cmd": "manual", "l": 50, "r": -50})
    _, s = run(brain, make_frame(us=(0, 0), opp=(0.5, 0)), sensor="white", ticks=1)
    check("manual override wins", s["action"].startswith("MANUAL") and s["sent"] == [50, -50], s["action"])
    brain.command({"cmd": "manual", "release": True})

    brain.command({"cmd": "estop", "on": True})
    _, s = run(brain, make_frame(us=(0, 0), opp=(0.5, 0)), sensor="white", ticks=1)
    check("E-stop wins over everything", s["action"] == "E-STOP" and s["sent"] == [0, 0], s["action"])
    brain.command({"cmd": "estop", "on": False})

    run(brain, make_frame(us=(0.95, 0), heading_deg=90, opp=(-0.3, 0)), sensor="blue", ticks=2)
    _, s = run(brain, make_frame(us=(1.2, 0), heading_deg=90, opp=(-0.3, 0)), sensor="white", ticks=10)
    check("pushed out -> LOST + sad song", s["match"] == "lost" and played[-1:] == ["lose"],
          f"{s['match']} {played}")
    _, s = run(brain, make_frame(us=(1.2, 0), opp=(-0.3, 0)), sensor="white", ticks=2)
    check("stays stopped after losing", s["sent"] == [0, 0] and s["match"] == "lost", s["action"])

    brain.command({"cmd": "reset"})
    brain.command({"cmd": "start"})
    brain.vision.lock_ring(False)
    for u in (0.6, 0.8, 0.95, 1.1):          # opponent gets pushed out, frame by frame
        run(brain, make_frame(us=(0, 0), opp=(u, 0)), sensor="white", ticks=2)
    _, s = run(brain, make_frame(us=(0, 0), opp=(1.25, 0)), sensor="white", ticks=15)
    check("opponent out -> WON + win song", s["match"] == "won" and played[-1:] == ["win"], f"{s['match']} {played}")

    brain.command({"cmd": "reset"})
    brain.command({"cmd": "start"})
    brain.vision.lock_ring(False)
    _, s = run(brain, make_frame(us=(0, 0), opp=(1.3, 0.3)), sensor="white", ticks=20)
    check("robot standing OUTSIDE the ring is not a fake win", s["match"] == "live", s["match"])

    print(f"\n{'ALL PASSED' if not check.failed else f'{check.failed} FAILED'}")
    return check.failed


if __name__ == "__main__":
    raise SystemExit(main())
