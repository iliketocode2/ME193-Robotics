"""
behavior_tree.py -- the sumo robot's decision policy (a behavior tree).

A behavior tree is ticked ~15 times a second. The root is a priority
SELECTOR: it tries its children top to bottom and the FIRST one that
doesn't fail wins that tick. Each child here is a SEQUENCE of
[condition -> action]: the condition decides whether this behavior applies,
the action writes the motor command. So higher branches pre-empt lower ones
-- e.g. "tape under our rear sensor" always beats "attack".

    Sumo (Selector)
     |- E-stop                 estop pressed               -> stop
     |- Manual drive           WASD held in the UI         -> pass keys through
     |- Robot link             no heartbeat from robot     -> stop
     |- Wait for start         match not live              -> stop
     |- Pushed out -> LOSE     camera + color sensor agree -> stop, sad song
     |- Opponent out -> WIN    opponent outside the ring   -> stop, win song
     |- Escape edge            rear sensor sees blue tape  -> drive forward
     |                         or camera says near edge    -> head to center
     |- Go to clicked target   user clicked the video      -> drive there
     |- Attack                 opponent visible            -> align, charge
     |- Guard (camera lost)    no camera                   -> hold (sensor still guards)
     '- Patrol center          otherwise                   -> back to center, scan

Sensor fusion: the CAMERA (phone, autonomous) gives positions in ring
coordinates (field.py: tape = radius 1). The LEGO COLOR SENSOR (autonomous,
on the robot's rear, facing down) sees the tape itself. Escape fires on
either; "we lost" needs the camera AND (the sensor having crossed the tape
or the camera being very sure), so one noisy sensor can't end our match.
Teleoperation enters as blackboard inputs: estop, manual, target, strategy.

The blackboard `bb` is a SimpleNamespace filled in by sumo_brain.py; see
make_blackboard() for every field.
"""

import math
from types import SimpleNamespace

from field import bearing_deg, wrap_deg

SUCCESS, FAILURE, RUNNING = "success", "failure", "running"

# ---- tunables ----------------------------------------------------------------
OUT_R = 1.15             # our center this far out (ring radii) = we're out of the ring
OUT_HOLD_S = 0.4         # ...for this long
OUT_SURE_R = 1.35        # camera alone is sure beyond this
OPP_OUT_R = 1.2
OPP_OUT_HOLD_S = 0.6
PUSH_DIST = 0.45         # opponent this close and dead ahead = we're pushing it
PUSH_BEARING = 30
TARGET_REACHED = 0.10
TURN_GAIN = 1.2          # motor % per degree of bearing error
MAX_TURN = 70


def strategy_params(s):
    """s = 0 (defensive) .. 1 (aggressive)."""
    return {"charge": 60 + 40 * s,            # % speed when charging
            "edge": 0.65 + 0.20 * s,          # ring radius where we turn back
            "align": 25 - 10 * s}             # degrees of bearing error tolerated before charging


def make_blackboard():
    return SimpleNamespace(
        now=0.0, match="idle", estop=False, robot_ok=False, dry_run=False,
        manual=None,                 # (l, r) while keys are held, else None
        camera_ok=False,
        us=None,                     # {"pos": (u, v), "r": float, "heading": (hx, hy)} in ring coords
        opp=None,                    # {"pos": (u, v), "r": float}
        sensor_ok=False, rear_on_blue=False, blue_recent=False, sensor_name="?",
        our_out_s=0.0, opp_out_s=0.0,
        target=None,                 # (u, v) clicked in ring coords
        strategy=0.5,
        # outputs
        out=(0, 0), action="", goal=None, event=None, trace=[],
    )


# ---- tiny BT framework ----------------------------------------------------------
class Node:
    def __init__(self, name):
        self.name = name

    def tick(self, bb):
        status = self._tick(bb)
        bb.trace.append((self.name, status))
        return status


class Selector(Node):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def _tick(self, bb):
        for child in self.children:
            status = child.tick(bb)
            if status != FAILURE:
                return status
        return FAILURE


class Sequence(Node):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def _tick(self, bb):
        for child in self.children:
            status = child.tick(bb)
            if status != SUCCESS:
                return status
        return SUCCESS


class Condition(Node):
    def __init__(self, name, fn):
        super().__init__(name)
        self.fn = fn

    def _tick(self, bb):
        return SUCCESS if self.fn(bb) else FAILURE


class Action(Node):
    def __init__(self, name, fn):
        super().__init__(name)
        self.fn = fn

    def _tick(self, bb):
        return self.fn(bb) or RUNNING


# ---- motion helpers -------------------------------------------------------------
def _clamp(v, lim=100):
    return int(max(-lim, min(lim, round(v))))


def drive_toward(bearing, speed, allow_reverse=False, min_forward=0.0, forward_only=False):
    """Tank command that turns toward `bearing` (deg, + = right) while
    driving; slows to turn in place for big errors. With allow_reverse, a
    target behind us is reached by backing up instead of a U-turn. With
    forward_only, both wheels keep turning forward (the turn is capped) so
    the robot's rear can't swing backward."""
    if allow_reverse and abs(bearing) > 110:
        rb = wrap_deg(bearing - 180)
        turn = max(-MAX_TURN, min(MAX_TURN, TURN_GAIN * rb))
        s = speed * max(0.0, 1 - abs(rb) / 60)
        return _clamp(-s + turn), _clamp(-s - turn)
    turn = max(-MAX_TURN, min(MAX_TURN, TURN_GAIN * bearing))
    s = max(min_forward, speed * max(0.0, 1 - abs(bearing) / 60))
    if forward_only:
        turn = max(-0.6 * s, min(0.6 * s, turn))
    return _clamp(s + turn), _clamp(s - turn)


def _set(bb, out, action, goal=None):
    bb.out, bb.action, bb.goal = out, action, goal
    return RUNNING


def _bearing_to(bb, point):
    return bearing_deg(bb.us["heading"], bb.us["pos"], point)


# ---- conditions -------------------------------------------------------------------
def pushing(bb):
    if not (bb.us and bb.opp):
        return False
    du = bb.opp["pos"][0] - bb.us["pos"][0], bb.opp["pos"][1] - bb.us["pos"][1]
    return (math.hypot(*du) < PUSH_DIST and abs(_bearing_to(bb, bb.opp["pos"])) < PUSH_BEARING
            and bb.opp["r"] > bb.us["r"])


def near_edge(bb):
    if bb.sensor_ok and bb.rear_on_blue:
        return True
    if not (bb.camera_ok and bb.us):
        return False
    limit = 1.0 if pushing(bb) else strategy_params(bb.strategy)["edge"]
    return bb.us["r"] > limit


def we_are_out(bb):
    if bb.our_out_s < OUT_HOLD_S or not bb.us:
        return False
    sensor_agrees = bb.sensor_ok and bb.blue_recent and not bb.rear_on_blue
    return sensor_agrees or not bb.sensor_ok or bb.us["r"] > OUT_SURE_R


# ---- actions ------------------------------------------------------------------------
def act_stop(label):
    return lambda bb: _set(bb, (0, 0), label)


def act_manual(bb):
    l, r = bb.manual
    return _set(bb, (_clamp(l), _clamp(r)), f"MANUAL  L{l:+d} R{r:+d}")


def act_wait(bb):
    labels = {"idle": "Waiting for start", "won": "WON - celebrating", "lost": "OUT - match over"}
    return _set(bb, (0, 0), labels.get(bb.match, "Paused"))


def act_lose(bb):
    bb.event = "lose"
    return _set(bb, (0, 0), "PUSHED OUT - stopping, sad song")


def act_win(bb):
    bb.event = "win"
    return _set(bb, (0, 0), "OPPONENT OUT - victory song")


def act_escape(bb):
    if bb.sensor_ok and bb.rear_on_blue:
        if bb.camera_ok and bb.us:
            out = drive_toward(_bearing_to(bb, (0, 0)), 85, min_forward=60, forward_only=True)
        else:
            out = (80, 80)
        return _set(bb, out, "ESCAPE - tape under rear sensor, driving forward", (0, 0))
    b = _bearing_to(bb, (0, 0))
    return _set(bb, drive_toward(b, 75, allow_reverse=True),
                f"ESCAPE - near edge (r={bb.us['r']:.2f}), heading to center", (0, 0))


def act_target(bb):
    d = math.hypot(bb.target[0] - bb.us["pos"][0], bb.target[1] - bb.us["pos"][1])
    if d < TARGET_REACHED:
        goal, bb.target = bb.target, None
        return _set(bb, (0, 0), "TARGET reached", goal)
    b = _bearing_to(bb, bb.target)
    return _set(bb, drive_toward(b, 70, allow_reverse=True), f"GO TO TARGET ({d:.2f} away)", bb.target)


def act_attack(bb):
    p = strategy_params(bb.strategy)
    b = _bearing_to(bb, bb.opp["pos"])
    if abs(b) > p["align"]:
        turn = max(-60, min(60, TURN_GAIN * b))
        return _set(bb, (_clamp(turn), _clamp(-turn)), f"ATTACK - aligning ({b:+.0f} deg)", bb.opp["pos"])
    out = drive_toward(b, p["charge"], min_forward=p["charge"] * 0.8)
    verb = "PUSHING" if pushing(bb) else "charging"
    return _set(bb, out, f"ATTACK - {verb} at {p['charge']:.0f}%", bb.opp["pos"])


def act_patrol(bb):
    if bb.us is None:
        return _set(bb, (0, 0), "Can't see our tag - holding")
    if bb.us["r"] > 0.25:
        return _set(bb, drive_toward(_bearing_to(bb, (0, 0)), 50, allow_reverse=True),
                    "Return to center", (0, 0))
    return _set(bb, (25, -25), "Center - scanning for opponent")


def build_tree():
    def branch(name, cond, act):
        return Sequence(name, [Condition(name + "?", cond), Action(name + "!", act)])

    return Selector("Sumo", [
        branch("E-stop", lambda bb: bb.estop, act_stop("E-STOP")),
        branch("Manual drive", lambda bb: bb.manual is not None, act_manual),
        branch("Robot link", lambda bb: not bb.robot_ok and not bb.dry_run, act_stop("Waiting for robot link")),
        branch("Wait for start", lambda bb: bb.match != "live", act_wait),
        branch("Pushed out -> LOSE", we_are_out, act_lose),
        branch("Opponent out -> WIN",
               lambda bb: bb.opp is not None and bb.opp_out_s >= OPP_OUT_HOLD_S, act_win),
        branch("Escape edge", near_edge, act_escape),
        branch("Go to target", lambda bb: bb.target is not None and bb.camera_ok and bb.us, act_target),
        branch("Attack", lambda bb: bb.camera_ok and bb.us and bb.opp, act_attack),
        branch("Guard (camera lost)", lambda bb: not bb.camera_ok,
               act_stop("GUARD - camera lost, holding (rear sensor still active)")),
        Action("Patrol center", act_patrol),
    ])


def branch_names(tree):
    return [c.name for c in tree.children]


def tick(tree, bb):
    """One tick. Returns per-branch status for the UI: 'active' for the branch
    that ran, 'failed' for ones checked and skipped, 'idle' below it."""
    bb.trace, bb.event, bb.goal = [], None, None
    tree.tick(bb)
    ran = {name: status for name, status in bb.trace}
    statuses, found = [], False
    for c in tree.children:
        if found:
            statuses.append("idle")
        elif ran.get(c.name, FAILURE) != FAILURE:
            statuses.append("active")
            found = True
        else:
            statuses.append("failed")
    return statuses
