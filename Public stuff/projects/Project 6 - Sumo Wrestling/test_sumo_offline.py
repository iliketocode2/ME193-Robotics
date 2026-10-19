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
import vision
from vision import Vision

# synthetic frames have grey squares, not LEGO cars: test the change-detection
# path even if a trained model exists in models/
vision._load_detector = lambda: (None, "change detection (test)")

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


def run(brain, frame, sensor=None, ticks=8, dt=1 / 15, before_tick=None):
    snap = res = None
    for _ in range(ticks):
        res = brain.vision.process(frame)        # every tick, like the live vision thread (tracks confirm)
        res["size"] = [W, H]
        now = time.monotonic()
        with brain.lock:
            brain.result, brain.result_t = res, now
            brain.status, brain.status_t = {"motor": True, "color": True}, now
            if sensor is not None:
                brain.sensor, brain.sensor_t = {"color": 3 if sensor == "blue" else 7, "name": sensor}, now
                brain.rear_on_blue = sensor == "blue"
                if sensor == "blue":
                    brain.last_blue_t = now
        if before_tick:
            before_tick()                    # e.g. a UI command the browser re-sends every tick
        snap = brain.tick(now, dt)
        time.sleep(dt)                       # real time: out-of-ring timers count real frame time
    return res, snap


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        check.failed += 1


check.failed = 0


def frames(v, frame, n):
    res = None
    for _ in range(n):
        res = v.process(frame)
    return res


def floor_only():
    return np.full((H, W, 3), (150, 160, 165), np.uint8)


def main():
    print("vision: ring")
    v = Vision()
    res = v.process(make_frame(us=(0, 0)))
    check("a new ring is not believed after one frame", res["ring"] is None and res["ring_pending"])
    res = frames(v, make_frame(us=(0, 0), heading_deg=0, opp=(0.5, 0)), 4)
    ring = res["ring"]
    check("ring found after a few frames", ring is not None, f"tape coverage {res['ring_quality']}")
    check("ring center", ring and abs(ring.cx - 480) < 15 and abs(ring.cy - 270) < 15,
          f"{ring and (round(ring.cx), round(ring.cy))}")

    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.3:
        res = v.process(floor_only())
    check("blue tape gone -> no field (after ~1 s)", res["ring"] is None)

    junk = floor_only()                                  # blue things that are NOT a ring
    cv2.rectangle(junk, (100, 100), (260, 220), (200, 90, 20), -1)
    cv2.line(junk, (500, 80), (900, 140), (200, 90, 20), 12)
    cv2.circle(junk, (700, 400), 40, (200, 90, 20), -1)
    res = frames(Vision(), junk, 8)
    check("blue rectangle + line + disc -> no field", res["ring"] is None)

    gappy = floor_only()                                 # imperfect ring: ~55% of the tape, in pieces
    for a0 in (0, 80, 170, 260):
        cv2.ellipse(gappy, RING[0], (350, 230), 0, a0, a0 + 50, (200, 90, 20), 14)
    res = frames(Vision(), gappy, 5)
    r = res["ring"]
    check("ring with big gaps still found", r is not None and abs(r.cx - 480) < 20 and abs(r.cy - 270) < 20,
          f"{r and (round(r.cx), round(r.cy), round(r.w), round(r.h))}")

    cluttered = make_frame(us=(0, 0))
    cv2.rectangle(cluttered, (5, 5), (120, 90), (200, 90, 20), -1)   # a blue box in the corner
    res = frames(Vision(), cluttered, 5)
    r = res["ring"]
    check("ring + blue clutter -> ring still accurate",
          r is not None and abs(r.cx - 480) < 15 and abs(max(r.w, r.h) - 700) < 40 and abs(min(r.w, r.h) - 460) < 40,
          f"{r and (round(r.cx), round(r.cy), round(r.w), round(r.h))}")

    print("vision: us + opponent")
    v = Vision()
    res = frames(v, make_frame(us=(0, 0), heading_deg=0, opp=(0.5, 0)), 5)
    check("our tag found", res["us"] is not None)
    if res["us"]:
        hx, hy = res["us"]["hx"], res["us"]["hy"]
        check("heading points up", hy < -0.95, f"({hx:.2f}, {hy:.2f})")
    check("opponent confirmed after a few frames", res["opp"] is not None)
    res90 = v.process(make_frame(us=(0, 0), heading_deg=90))
    if res90["us"]:
        check("heading 90 deg points right", res90["us"]["hx"] > 0.95, f"{res90['us']['hx']:.2f}")

    junk = make_frame(us=(-0.3, 0))
    cv2.line(junk, (500, 200), (700, 330), (40, 40, 40), 4)            # a cable / thin shadow
    cv2.circle(junk, (420, 380), 6, (40, 40, 40), -1)                   # a speck
    res = frames(Vision(), junk, 6)
    check("thin line + speck are not an opponent", res["opp"] is None, f"{len(res['robots'])} candidates")

    v = Vision()
    v.process(make_frame(us=(-0.3, 0)))
    flash = make_frame(us=(-0.3, 0), opp=(0.4, 0))
    res = v.process(flash)
    check("one-frame blob is not yet an opponent", res["opp"] is None)

    stain = make_frame(us=(-0.3, 0))
    cv2.rectangle(stain, (560, 330), (640, 400), (95, 105, 110), -1)    # a dark patch ON the floor
    res = frames(Vision(), stain, 6)
    check("without snapshot: floor stain looks like an opponent", res["opp"] is not None)
    v = Vision()
    frames(v, stain, 4)
    v.request_reference()                                               # snapshot of the (stained) empty arena
    v.process(stain)
    v.reset_opponent_track()
    res = frames(v, stain, 6)
    check("with snapshot: floor stain ignored", res["opp"] is None)
    real = stain.copy()
    x, y = ring_px(-0.1, -0.5)
    cv2.rectangle(real, (int(x - 30), int(y - 30)), (int(x + 30), int(y + 30)), (40, 40, 40), -1)
    res = frames(v, real, 5)
    check("with snapshot: real opponent found", res["opp"] is not None and abs(res["opp"]["x"] - x) < 15,
          f"{res['opp'] and round(res['opp']['x'])} vs {round(x)}")

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

    # the browser re-sends held keys every 100 ms, so send right before the tick
    # (one slow vision frame could otherwise make a single early command stale)
    _, s = run(brain, make_frame(us=(0, 0), opp=(0.5, 0)), sensor="white", ticks=1,
               before_tick=lambda: brain.command({"cmd": "manual", "l": 50, "r": -50}))
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
