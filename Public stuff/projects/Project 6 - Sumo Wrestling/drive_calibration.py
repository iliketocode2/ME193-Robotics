"""
drive_calibration.py -- make "forward" and "turn right" mean the same thing to
the brain and to the real robot.

The brain steers by our AprilTag: "forward" = the way the tag's arrow points,
"right turn" = clockwise seen from above. Two build details can silently
break that, and they need DIFFERENT fixes:
  * the tag is mounted backwards -> robot drives along -arrow, turns are fine
  * the motors are reversed      -> robot drives along -arrow AND turns flip
(plus left/right motor outputs swapped -> only turns flip).

So instead of guessing, the "Calibrate drive" button runs a 3-second test with
the robot in the middle of the field: drive forward briefly, turn briefly,
undo both, while the camera watches the tag. It measures
    forward = +1 if a (+, +) command moved the robot along its tag arrow, else -1
    turn    = +1 if a (+, -) command turned it clockwise, else -1
and saves them to drive_calibration.json. Every command the brain sends
(behavior tree AND the WASD keys) then goes through to_motor():
    v = (l + r) / 2 * forward,   w = (l - r) / 2 * turn,   motor = (v + w, v - w)
which fixes all four cases (normal, tag backwards, motors reversed, sides swapped).
"""

import json
import math
import os
import time

from config import CAMERA_MIRRORED, HERE, REVERSE_DRIVE, REVERSE_TURN

CAL_PATH = os.path.join(HERE, "drive_calibration.json")
CAL_SPEED = 45
# (phase, raw motor command, seconds)
PHASES = [
    ("forward", (CAL_SPEED, CAL_SPEED), 0.6),
    ("settle", (0, 0), 0.5),
    ("turn", (CAL_SPEED, -CAL_SPEED), 0.5),
    ("settle2", (0, 0), 0.5),
    ("undo turn", (-CAL_SPEED, CAL_SPEED), 0.5),
    ("undo forward", (-CAL_SPEED, -CAL_SPEED), 0.6),
]
MIN_MOVE_FRAC = 0.04      # forward step must move the tag at least this x field size
MIN_TURN_DEG = 8.0


def load():
    """Saved calibration if there is one, else config.py's REVERSE_DRIVE / REVERSE_TURN."""
    try:
        with open(CAL_PATH) as f:
            d = json.load(f)
        return {"forward": 1 if d.get("forward", 1) >= 0 else -1, "turn": 1 if d.get("turn", 1) >= 0 else -1,
                "when": d.get("when")}
    except (OSError, ValueError):
        return {"forward": -1 if REVERSE_DRIVE else 1, "turn": -1 if REVERSE_TURN else 1, "when": None}


def save(cal):
    with open(CAL_PATH, "w") as f:
        json.dump(cal, f, indent=2)


def to_motor(out, cal):
    """Brain command (l, r) in the tag's frame -> what to send to the motors."""
    l, r = out
    v, w = (l + r) / 2 * cal["forward"], (l - r) / 2 * cal["turn"]
    clamp = lambda x: int(max(-100, min(100, round(x))))  # noqa: E731
    return clamp(v + w), clamp(v - w)


def describe(cal):
    f = "normal" if cal["forward"] > 0 else "REVERSED"
    t = "normal" if cal["turn"] > 0 else "FLIPPED"
    src = f"calibrated {cal['when']}" if cal.get("when") else "set in config.py"
    return f"forward {f}, turning {t} ({src})"


class Calibrator:
    """Runs PHASES; step() is called every brain tick with our latest tag pose."""

    def __init__(self, now):
        self.t0, self.phase_i = now, 0
        self.p0 = self.p1 = self.h0 = self.h1 = self.h2 = None
        self.result = None          # dict when finished OK
        self.error = None           # str when it failed

    def _phase(self, now):
        t = now - self.t0
        for i, (name, cmd, dur) in enumerate(PHASES):
            if t < dur:
                return i, name, cmd
            t -= dur
        return len(PHASES), "done", (0, 0)

    def step(self, now, us, scale):
        """us = latest fresh tag pose {"x","y","hx","hy"} or None. -> (raw_cmd, label);
        sets self.result / self.error when finished."""
        i, name, cmd = self._phase(now)
        entering = i != self.phase_i
        prev = PHASES[self.phase_i][0] if self.phase_i < len(PHASES) else "done"
        self.phase_i = i
        if self.p0 is None:                         # first tick: remember where we start
            if us is None:
                self.error = "can't see our tag -- put the robot in view, inside the field"
                return (0, 0), "CALIBRATION FAILED"
            self.p0, self.h0 = (us["x"], us["y"]), (us["hx"], us["hy"])
        if entering and prev == "settle":           # measure the forward step
            if us is None:
                self.error = "lost our tag after driving forward"
                return (0, 0), "CALIBRATION FAILED"
            dx, dy = us["x"] - self.p0[0], us["y"] - self.p0[1]
            if math.hypot(dx, dy) < MIN_MOVE_FRAC * scale:
                self.error = "the robot barely moved -- is it connected, on the floor, not against the tape?"
                return (0, 0), "CALIBRATION FAILED"
            self.forward = 1 if dx * self.h0[0] + dy * self.h0[1] > 0 else -1
            self.h1 = (us["hx"], us["hy"])
        if entering and prev == "settle2":          # measure the turn
            if us is None:
                self.error = "lost our tag after turning"
                return (0, 0), "CALIBRATION FAILED"
            h2 = (us["hx"], us["hy"])
            # image y points down: positive cross = clockwise on screen = right turn seen from above
            dth = math.degrees(math.atan2(self.h1[0] * h2[1] - self.h1[1] * h2[0],
                                          self.h1[0] * h2[0] + self.h1[1] * h2[1]))
            if CAMERA_MIRRORED:
                dth = -dth
            if abs(dth) < MIN_TURN_DEG:
                self.error = f"the robot barely turned ({dth:+.0f} deg) -- check both wheels touch the floor"
                return (0, 0), "CALIBRATION FAILED"
            self.turn = 1 if dth > 0 else -1
        if name == "done":
            self.result = {"forward": self.forward, "turn": self.turn,
                           "when": time.strftime("%Y-%m-%d %H:%M")}
            return (0, 0), "CALIBRATION DONE"
        return cmd, f"CALIBRATING DRIVE - {name}"
