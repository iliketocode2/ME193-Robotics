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


def paste_tag(img, pos, tag_id, heading_deg=0.0):
    """Draw a tag36h11 card (with white border) centered at ring coords `pos`."""
    d = cv2.aruco.getPredefinedDictionary(config.APRILTAG_DICTIONARY)
    tag = cv2.aruco.generateImageMarker(d, tag_id, TAG_PX)
    pad = TAG_PX // 3
    card = cv2.cvtColor(cv2.copyMakeBorder(tag, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255),
                        cv2.COLOR_GRAY2BGR)
    n = card.shape[0]
    x, y = ring_px(*pos)
    M = cv2.getRotationMatrix2D((n / 2, n / 2), -heading_deg, 1.0)   # cv2: + = counter-clockwise
    M[:, 2] += (x - n / 2, y - n / 2)
    mask = cv2.warpAffine(np.full((n, n), 255, np.uint8), M, (W, H))
    warped = cv2.warpAffine(card, M, (W, H))
    img[mask > 0] = warped[mask > 0]


def make_frame(us=None, heading_deg=0.0, opp=None, opp_tag=None, tags=()):
    """us/opp in ring coords (u, v); heading 0 = facing up the image, + = clockwise.
    opp_tag: put this tag id on the opponent's block; tags: extra (pos, id) tag cards."""
    img = np.full((H, W, 3), (150, 160, 165), np.uint8)
    cv2.ellipse(img, RING[0], (RING[1][0] // 2, RING[1][1] // 2), RING[2], 0, 360, (200, 90, 20), 14)
    if opp is not None:
        x, y = ring_px(*opp)
        cv2.rectangle(img, (int(x - 45), int(y - 45)), (int(x + 45), int(y + 45)), (40, 40, 40), -1)
        if opp_tag is not None:
            paste_tag(img, opp, opp_tag)
    for pos, tag_id in tags:
        paste_tag(img, pos, tag_id)
    if us is not None:
        paste_tag(img, us, config.OUR_TAG_ID, heading_deg)
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


def blind(brain, seconds, dt=1 / 15):
    """Camera gives NOTHING for `seconds` (phone swung away): robot link and
    sensor stay fresh, but no new vision results arrive. Returns snaps over time."""
    snaps, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        now = time.monotonic()
        with brain.lock:
            brain.status, brain.status_t = {"motor": True, "color": True}, now
            brain.sensor, brain.sensor_t = {"color": 7, "name": "white"}, now
        snaps.append((now - t0, brain.tick(now, dt)))
        time.sleep(dt)
    return snaps


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
    print("vision: field (any shape)")
    v = Vision()
    res = v.process(make_frame(us=(0, 0)))
    check("a new field is not believed after one frame", res["field"] is None and res["field_pending"])
    res = frames(v, make_frame(us=(0, 0), heading_deg=0, opp=(0.5, 0)), 4)
    fd = res["field"]
    check("closed tape loop found after a few frames", fd is not None and res["field_kind"] == "closed")
    check("field center", fd and abs(fd.cx - 480) < 15 and abs(fd.cy - 270) < 15,
          f"{fd and (round(fd.cx), round(fd.cy))}")
    if fd:
        rs = [fd.radius_at(*ring_px(u, v_)) for u, v_ in ((1, 0), (0, 1), (-1, 0), (0, -1))]
        check("tape is at r ~= 1 in every direction (wide ellipse)", all(0.95 < r < 1.08 for r in rs),
              str([round(r, 2) for r in rs]))

    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.3:
        res = v.process(floor_only())
    check("blue tape gone -> no field (after ~1 s)", res["field"] is None)

    junk = floor_only()                                  # blue things that are NOT a field border
    cv2.rectangle(junk, (100, 100), (260, 220), (200, 90, 20), -1)
    cv2.line(junk, (500, 80), (900, 140), (200, 90, 20), 12)
    cv2.circle(junk, (700, 400), 40, (200, 90, 20), -1)
    res = frames(Vision(), junk, 8)
    check("filled blue box + line + disc -> no field", res["field"] is None)

    rect = floor_only()                                  # a long RECTANGULAR field
    cv2.rectangle(rect, (130, 150), (830, 390), (200, 90, 20), 14)
    res = frames(Vision(), rect, 4)
    fd = res["field"]
    ok = fd is not None
    check("rectangular tape -> field found", ok, res["field_kind"] or "")
    if ok:
        on_short, on_long, corner = fd.radius_at(137, 270), fd.radius_at(480, 157), fd.radius_at(140, 160)
        check("rectangle: r ~= 1 on the near long side AND the far short side", 0.95 < on_short < 1.05 and 0.95 < on_long < 1.05,
              f"short side {on_short:.2f}, long side {on_long:.2f}")
        check("rectangle: corner also r ~= 1 (not a circle!)", 0.93 < corner < 1.07, f"{corner:.2f}")
        mid_y = fd.radius_at(480, 270 - 0.8 * 113)       # 80% of the way to the long side
        check("rectangle: 80% toward the near side reads r ~= 0.8", 0.75 < mid_y < 0.85, f"{mid_y:.2f}")

    blob = floor_only()                                  # hand-taped, lopsided, dented loop
    pts = []
    for a in np.linspace(0, 2 * np.pi, 60, endpoint=False):
        r = 200 + 35 * np.sin(3 * a) + (60 if np.cos(a) > 0.6 else 0)
        pts.append((480 + 1.3 * r * np.cos(a), 270 + r * np.sin(a)))
    pts = np.array(pts, np.int32)
    cv2.polylines(blob, [pts], True, (200, 90, 20), 14)
    res = frames(Vision(), blob, 4)
    fd = res["field"]
    check("lopsided hand-taped loop -> field found", fd is not None)
    if fd:
        rs = [fd.radius_at(*p) for p in pts[::6]]
        check("lopsided loop: r ~= 1 all along its tape", all(0.95 < r < 1.12 for r in rs),
              f"min {min(rs):.2f} max {max(rs):.2f}")

    gappy = floor_only()                                 # broken loop: ~55% of the tape, in pieces
    for a0 in (0, 80, 170, 260):
        cv2.ellipse(gappy, RING[0], (350, 230), 0, a0, a0 + 50, (200, 90, 20), 14)
    res = frames(Vision(), gappy, 5)
    fd = res["field"]
    check("tape with big gaps -> partial field still found",
          fd is not None and res["field_kind"] == "partial" and abs(fd.cx - 480) < 20 and abs(fd.cy - 270) < 20,
          f"{fd and (round(fd.cx), round(fd.cy))} {res['field_kind']}")

    cluttered = make_frame(us=(0, 0))
    cv2.rectangle(cluttered, (5, 5), (120, 90), (200, 90, 20), -1)   # a blue box in the corner
    res = frames(Vision(), cluttered, 5)
    fd = res["field"]
    check("field + blue clutter -> field still accurate",
          fd is not None and abs(fd.cx - 480) < 15 and 0.95 < fd.radius_at(*ring_px(1, 0)) < 1.08,
          f"{fd and (round(fd.cx), round(fd.cy))}")

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

    print("vision: opponents' AprilTags")
    v = Vision()
    frames(v, make_frame(us=(-0.4, 0)), 4)                              # ring confirmed, no opponent yet
    res = v.process(make_frame(us=(-0.4, 0), opp=(0.4, 0.1), opp_tag=3))
    check("tagged opponent confirmed on the FIRST frame", res["opp"] is not None and res["opp"].get("tag_id") == 3,
          f"{res['opp'] and res['opp'].get('tag_id')}")
    ox, oy = ring_px(0.4, 0.1)
    check("opponent position = their tag center", res["opp"] and abs(res["opp"]["x"] - ox) < 6 and abs(res["opp"]["y"] - oy) < 6)
    check("our own tag (#6) is never the opponent", res["us"] is not None and res["opp"]["tag_id"] != config.OUR_TAG_ID)

    v = Vision()
    res = frames(v, make_frame(us=(-0.4, 0), tags=[((1.15, 0.2), 9)]), 6)  # tagged robot waiting OUTSIDE the ring
    check("tagged robot outside the ring is ignored", res["opp"] is None and len(res["their_tags"]) == 1)

    v = Vision()
    res = frames(v, make_frame(us=(-0.4, 0), opp=(0.3, -0.3), tags=[((0.2, 0.45), 4)]), 6)
    check("a tag beats an untagged blob", res["opp"] is not None and res["opp"].get("tag_id") == 4,
          f"{res['opp'] and res['opp'].get('tag_id')}")

    print("behavior tree")
    brain = sumo_brain.Brain()
    _, s = run(brain, make_frame(us=(0, 0), opp=(0.5, 0)))
    check("idle waits for start", s["action"] == "Waiting for start" and s["sent"] == [0, 0], s["action"])

    brain.command({"cmd": "start"})
    brain.vision.lock_field(False)            # synthetic frames: let it keep fitting
    _, s = run(brain, make_frame(us=(0, 0), heading_deg=0, opp=(0.5, 0)), sensor="white")
    check("opponent to the right -> turn right (L>R)", "ATTACK" in s["action"] and s["sent"][0] > s["sent"][1],
          f"{s['action']} {s['sent']}")
    _, s = run(brain, make_frame(us=(0, 0), heading_deg=0, opp=(-0.5, 0)), sensor="white")
    check("opponent to the left -> turn left (R>L)", s["sent"][1] > s["sent"][0], f"{s['action']} {s['sent']}")
    _, s = run(brain, make_frame(us=(0, 0.1), heading_deg=0, opp=(0, -0.45)), sensor="white")
    check("opponent dead ahead -> charge", "ATTACK" in s["action"] and min(s["sent"]) > 40, f"{s['action']} {s['sent']}")
    charge = s["sent"]

    # --- vision dropouts: keep executing the last move for a moment ---
    _, s = run(brain, make_frame(opp=(0, -0.45)), sensor="white", ticks=4)          # our tag blurred out
    check("our tag lost -> HOLD the charge", s["action"].startswith("HOLD") and s["sent"] == charge,
          f"{s['action']} {s['sent']}")
    _, s = run(brain, make_frame(opp=(0, -0.45)), sensor="white", ticks=18)         # ...for too long
    check("tag lost > 1 s -> stop holding", not s["action"].startswith("HOLD") and s["sent"] == [0, 0], s["action"])

    run(brain, make_frame(us=(0, 0.1), opp=(0, -0.45)), sensor="white")             # charging again
    snaps = blind(brain, 2.4)                                                       # phone swung away entirely
    held = [snap for t, snap in snaps if 0.8 < t < 1.4]
    after = [snap for t, snap in snaps if t > 2.0]
    check("camera lost -> keeps driving the last command", held and all(sn["sent"] == charge for sn in held),
          f"{held and held[-1]['action']}")
    check("camera lost too long -> GUARD (stop)", after and all(sn["sent"] == [0, 0] for sn in after),
          f"{after and after[-1]['action']}")

    run(brain, make_frame(us=(0, 0.1), opp=(0, -0.45)), sensor="white")             # charging again
    brain.command({"cmd": "estop", "on": True}); run(brain, make_frame(us=(0, 0.1), opp=(0, -0.45)), ticks=1)
    brain.command({"cmd": "estop", "on": False})
    _, s = run(brain, make_frame(opp=(0, -0.45)), sensor="white", ticks=3)
    check("after an E-stop a dropout does NOT resume the old move", not s["action"].startswith("HOLD"), s["action"])

    _, s = run(brain, make_frame(us=(0, 0.2), heading_deg=0, opp=(0.5, -0.2)), sensor="white")
    check("opponent off to the side -> ARC (both wheels forward, turning)", "closing in" in s["action"]
          and min(s["sent"]) >= 0 and s["sent"][0] > s["sent"][1], f"{s['action']} {s['sent']}")

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
    brain.vision.lock_field(False)
    for u in (0.6, 0.8, 0.95, 1.1):          # opponent gets pushed out, frame by frame
        run(brain, make_frame(us=(0, 0), opp=(u, 0)), sensor="white", ticks=2)
    _, s = run(brain, make_frame(us=(0, 0), opp=(1.25, 0)), sensor="white", ticks=15)
    check("opponent out -> WON + win song", s["match"] == "won" and played[-1:] == ["win"], f"{s['match']} {played}")

    brain.command({"cmd": "reset"})
    brain.command({"cmd": "start"})
    brain.vision.lock_field(False)
    _, s = run(brain, make_frame(us=(0, 0), opp=(1.3, 0.3)), sensor="white", ticks=20)
    check("robot standing OUTSIDE the ring is not a fake win", s["match"] == "live", s["match"])

    # long thin RECTANGULAR field: 100 px from the center is near the tape one way,
    # nowhere near it the other way -- a circle model would treat both the same
    def rect_frame(us_xy, heading_deg):
        img = floor_only()
        cv2.rectangle(img, (130, 150), (830, 390), (200, 90, 20), 14)    # inside: 693 x 233 px
        paste_tag(img, ((us_xy[0] - 480) / 350, (us_xy[1] - 270) / 230), config.OUR_TAG_ID, heading_deg)
        return img
    brain.command({"cmd": "strategy", "value": 0.2})      # edge limit r = 0.74 (Balanced is 0.80)
    _, s = run(brain, rect_frame((480, 370), 0), sensor="white", ticks=8)        # 100 px toward a LONG side
    check("rectangle: 100 px toward the near long side -> ESCAPE toward center", "ESCAPE" in s["action"]
          and min(s["sent"]) > 0, f"r={s['world']['us'] and s['world']['us']['r']} {s['action']} {s['sent']}")
    _, s = run(brain, rect_frame((580, 270), 0), sensor="white", ticks=8)        # 100 px toward a SHORT side
    check("rectangle: 100 px toward the far short side -> no escape", "ESCAPE" not in s["action"],
          f"r={s['world']['us'] and s['world']['us']['r']} {s['action']}")

    print("drive calibration (simulated robots)")
    import drive_calibration as dcal

    class SimRobot:
        """Kinematics only: pose in image pixels, angle = physical front, + = clockwise on screen."""
        def __init__(self, tag_backward=False, motors_reversed=False, sides_swapped=False):
            self.x, self.y, self.a = 480.0, 270.0, -math.pi / 2          # facing up the image
            self.tag_backward, self.rev, self.swap = tag_backward, motors_reversed, sides_swapped

        def drive(self, cmd, dt):
            l, r = cmd
            if self.swap:
                l, r = r, l
            if self.rev:
                l, r = -l, -r
            v, w = (l + r) / 2 * 2.0, (l - r) / 2 * 0.03          # px/s per %, rad/s per %
            self.a += w * dt
            self.x += v * dt * math.cos(self.a)
            self.y += v * dt * math.sin(self.a)

        def tag(self):
            a = self.a + (math.pi if self.tag_backward else 0.0)
            return {"x": self.x, "y": self.y, "hx": math.cos(a), "hy": math.sin(a)}

    for name, kw in (("normal", {}), ("tag backwards", {"tag_backward": True}),
                     ("motors reversed", {"motors_reversed": True}), ("sides swapped", {"sides_swapped": True})):
        bot, cal, t, dt = SimRobot(**kw), dcal.Calibrator(0.0), 0.0, 1 / 15
        start = (bot.x, bot.y)
        while cal.result is None and cal.error is None and t < 6:
            cmd, _ = cal.step(t, bot.tag(), 275.0)
            bot.drive(cmd, dt)
            t += dt
        res = cal.result
        if not res:
            check(f"calibration '{name}' finishes", False, cal.error)
            continue
        # after calibration: brain "forward" must move along the tag arrow, brain "right" must turn clockwise
        tag0 = bot.tag(); x0, y0 = bot.x, bot.y
        bot.drive(dcal.to_motor((50, 50), res), 0.5)
        along = (bot.x - x0) * tag0["hx"] + (bot.y - y0) * tag0["hy"]
        h0 = bot.tag(); bot.drive(dcal.to_motor((40, -40), res), 0.3); h1 = bot.tag()
        cw = h0["hx"] * h1["hy"] - h0["hy"] * h1["hx"]
        check(f"calibration '{name}': forward along tag arrow + right turn is clockwise", along > 0 and cw > 0,
              f"{res['forward']:+d}/{res['turn']:+d}, ends {math.hypot(bot.x - start[0], bot.y - start[1]):.0f}px from start")

    print(f"\n{'ALL PASSED' if not check.failed else f'{check.failed} FAILED'}")
    return check.failed


if __name__ == "__main__":
    raise SystemExit(main())
