"""Settings shared by the board app, the training code and the laptop tools."""

from pathlib import Path

# ---------- sensor (must match sketch.ino) ----------
SAMPLE_HZ = 100
ACC_COUNTS_PER_G = 2048.0           # sketch sets +-16 g
GYRO_COUNTS_PER_DPS = 16.4          # sketch sets +-2000 deg/s

# ---------- actions ----------
# A maneuver is something the walker does at one place on the route. The policy
# predicts, at every step, "straight" or "<maneuver>_soon" / "<maneuver>_now".
MANEUVERS = ["left", "right", "stairs", "arrive"]

# Combined maneuvers: two maneuvers closer than COMBINE_M can't each get their own
# "now" zone (it spans NOW_BEFORE_M + NOW_AFTER_M), so the later one used to swallow
# the earlier. They are cued together instead: "Turn left, then take the stairs".
# Whether a pair is combined is decided per route from the median gap over the
# training walks (features.route_plan), so every walk is labeled the same way.
COMBINE_M = 6.0
COMBINED = [f"{a}+{b}" for a in ("left", "right", "stairs") for b in MANEUVERS]
CUE_MANEUVERS = MANEUVERS + COMBINED
CLASSES = ["straight"] + [f"{m}_{w}" for m in CUE_MANEUVERS for w in ("soon", "now")]

SPOKEN = {
    "left_soon": "Left turn coming up",
    "left_now": "Turn left",
    "right_soon": "Right turn coming up",
    "right_now": "Turn right",
    "stairs_soon": "Stairs coming up",
    "stairs_now": "Take the stairs",
    "arrive_soon": "Almost there",
    "arrive_now": "You have arrived",
}
_SOON_NAME = {"left": "left turn", "right": "right turn", "stairs": "stairs", "arrive": "destination"}
_NOW_NAME = {"left": "turn left", "right": "turn right", "stairs": "take the stairs", "arrive": "you're there"}
for _c in COMBINED:
    _a, _b = _c.split("+")
    _then = f"{_b} again" if _a == _b and _a in ("left", "right") else _NOW_NAME[_b]
    SPOKEN[f"{_c}_soon"] = (f"Two {_a} turns coming up" if _a == _b and _a in ("left", "right")
                            else f"{_SOON_NAME[_a].capitalize()}, then {_SOON_NAME[_b]} coming up")
    SPOKEN[f"{_c}_now"] = f"{_NOW_NAME[_a].capitalize()}, then {_then}"

# ---------- labels ----------
NOW_BEFORE_M = 3.0      # steps from this far before a maneuver...
NOW_AFTER_M = 2.0       # ...to this far after are labeled "<m>_now": early when position is sure, and
                        # right as the turn starts when it is not (the gyro sees the turn begin)
LOOKAHEAD_M = 8.0       # the stretch before that is labeled "<m>_soon"
SNAP_WINDOW_M = 6.0     # review page: a tap snaps to a detected turn at most this far away

# ---------- features ----------
TURN_DEG = 60.0         # heading change that counts as a turn
TURN_SPAN_M = 4.0       # ...when it happens within this much walking
FEATURES = ["meters", "meters_since_turn", "turns", "heading", "last_turn_sign", "stairs"]
# k-NN distance weights, applied after standardizing each feature. The turn count says
# which leg of the route you're on, so it dominates; heading helps when a turn is missed.
FEATURE_WEIGHTS = {"meters": 1.0, "meters_since_turn": 1.5, "turns": 4.0,
                   "heading": 1.0, "last_turn_sign": 1.0, "stairs": 0.5}

# ---------- policy ----------
K = 3
CONFIRM_STEPS = 2       # a cue has to win this many steps in a row before it is spoken

# ---------- walks ----------
DEFAULT_ROUTE = "nolop"
DEFAULT_METERS_PER_STEP = 1.4   # one shoe = one count per stride (two footsteps)
WALK_BUDGET = 10                # course limit: demos + DAgger walks used for training
WALK_KINDS = ["demo", "dagger", "test", "calibrate", "sensor"]
SENSOR_ACTIVITIES = ["walk", "stairs", "turn", "still"]   # "sensor" sessions: training data for sensing
TRAINING_KINDS = ["demo", "dagger"]

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
WALKS_DIR = DATA_DIR / "walks"
WALKERS_PATH = DATA_DIR / "walkers.json"
POLICY_DIR = DATA_DIR / "policy"
