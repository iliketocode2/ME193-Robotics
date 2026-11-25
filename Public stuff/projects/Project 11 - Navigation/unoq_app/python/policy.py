"""The behavior-cloned navigation policy: k-nearest-neighbors over step states.

Training = storing every labeled step from the included walks (after
standardizing the features). Predicting = finding the K stored steps most like
the current one and letting them vote, closer ones counting more. With ~10
walks this beats a neural net: there is too little data to fit one, and k-NN
can't extrapolate to something it never saw.

`CueFilter` turns the per-step predictions into the cues the walker hears, and
`leave_one_walk_out` measures how well that whole chain works on walks the
policy didn't train on.
"""

import json

import numpy as np

import config as cfg
import features as feat

# a feature's spread is never treated as smaller than this when standardizing,
# so a feature that barely varies in training can't dominate the distance
SCALE_FLOOR = {"meters": 10.0, "meters_since_turn": 5.0, "turns": 0.5,
               "heading": 20.0, "last_turn_sign": 0.5, "stairs": 0.5}


class Policy:
    def __init__(self):
        self.x = None
        self.y = None
        self.mean = None
        self.scale = None
        self.weights = np.array([cfg.FEATURE_WEIGHTS[f] for f in cfg.FEATURES])
        self.info = {}

    @property
    def trained(self):
        return self.x is not None and len(self.x) > 0

    def fit(self, x, y, info=None):
        x = np.asarray(x, dtype=float)
        self.mean = x.mean(axis=0)
        floor = np.array([SCALE_FLOOR[f] for f in cfg.FEATURES])
        self.scale = np.maximum(x.std(axis=0), floor)
        self.x = self._norm(x)
        self.y = np.asarray(y, dtype=int)
        self.info = info or {}
        return self

    def _norm(self, x):
        return (x - self.mean) / self.scale * self.weights

    def predict(self, x):
        """One feature vector -> (class name, share of the vote it got)."""
        if not self.trained:
            return "straight", 0.0
        d = np.linalg.norm(self.x - self._norm(np.asarray(x, dtype=float)), axis=1)
        k = min(cfg.K, len(d))
        nearest = np.argpartition(d, k - 1)[:k]
        votes = np.zeros(len(cfg.CLASSES))
        for i in nearest:
            votes[self.y[i]] += 1.0 / (d[i] + 1e-3)
        best = int(votes.argmax())
        return cfg.CLASSES[best], float(votes[best] / votes.sum())

    # ---------- disk ----------

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "policy.npz", x=self.x, y=self.y, mean=self.mean, scale=self.scale,
                 weights=self.weights)
        (folder / "policy.json").write_text(json.dumps(
            {"features": cfg.FEATURES, "classes": cfg.CLASSES, "k": cfg.K, **self.info}, indent=2))

    @classmethod
    def load(cls, folder):
        p = cls()
        npz, js = folder / "policy.npz", folder / "policy.json"
        if not npz.exists() or not js.exists():
            return p
        info = json.loads(js.read_text())
        if info.get("features") != cfg.FEATURES or info.get("classes") != cfg.CLASSES:
            return p        # trained with different settings; retrain
        with np.load(npz) as f:
            p.x, p.y, p.mean, p.scale, p.weights = f["x"], f["y"], f["mean"], f["scale"], f["weights"]
        p.info = info
        return p


class CueFilter:
    """Per-step predictions -> spoken cues.

    A new cue has to win CONFIRM_STEPS steps in a row before it is spoken, except
    "<m>_now" right after "<m>_soon", which is spoken at once (by then you're at the
    turn). Going from "<m>_now" back to "<m>_soon" is ignored, so each maneuver is
    announced at most once as soon and once as now.
    """

    def __init__(self):
        self.current = "straight"
        self.candidate = None
        self.count = 0

    def feed(self, cls):
        """Returns the newly spoken cue, or None."""
        if cls == self.current:
            self.candidate, self.count = None, 0
            return None
        if self.current.endswith("_now") and cls == self.current.replace("_now", "_soon"):
            return None
        if cls == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = cls, 1
        need = cfg.CONFIRM_STEPS
        if cls.endswith("_now") and self.current == cls.replace("_now", "_soon"):
            need = 1
        if self.count < need:
            return None
        self.current, self.candidate, self.count = cls, None, 0
        return None if cls == "straight" else cls


# ---------- training ----------

def cue_maneuvers(walk, plan):
    """The walk's maneuvers as they are cued: close pairs combined (config.COMBINE_M)."""
    return feat.combine_maneuvers(walk["steps"], walk["maneuvers"], walk["meters_per_step"], plan)


def walk_dataset(walk, plan=None):
    """walk: dict with steps, maneuvers, meters_per_step -> (x, y)."""
    x, _ = feat.walk_features(walk["steps"], walk["meters_per_step"])
    y = feat.walk_labels(walk["steps"], cue_maneuvers(walk, plan), walk["meters_per_step"])
    return x, y


def train(walks):
    """walks: list of dicts (id, steps, maneuvers, meters_per_step). Returns (Policy, report)."""
    xs, ys = [], []
    plan = feat.route_plan(walks)
    for w in walks:
        x, y = walk_dataset(w, plan)
        xs.append(x)
        ys.append(y)
    if not xs or sum(len(x) for x in xs) == 0:
        raise ValueError("No reviewed walks with steps to train on.")
    counts = np.bincount(np.concatenate(ys), minlength=len(cfg.CLASSES))
    policy = Policy().fit(np.vstack(xs), np.concatenate(ys), info={
        "walks": [w["id"] for w in walks],
        "steps": int(sum(len(x) for x in xs)),
        "class_counts": {c: int(n) for c, n in zip(cfg.CLASSES, counts) if n},
        "route_plan": plan,
    })
    report = leave_one_walk_out(walks) if len(walks) >= 2 else None
    policy.info["report"] = report
    return policy, report


def run_walk(policy, walk):
    """Replay a walk through the live chain. Returns [(step index, meters, cue)] for spoken cues."""
    x, _ = feat.walk_features(walk["steps"], walk["meters_per_step"])
    cues = CueFilter()
    out = []
    for i, row in enumerate(x):
        cls, _ = policy.predict(row)
        spoken = cues.feed(cls)
        if spoken:
            out.append((i, float(row[0]), spoken))
    return out


def leave_one_walk_out(walks):
    """Train on all walks but one, replay the held-out walk, compare cues with its maneuvers.

    For each true maneuver: the first matching "<m>_now" cue within LOOKAHEAD_M of it
    counts as a hit, and its error is (cue meters - maneuver meters), negative = early.
    "_now" cues that match no maneuver are false alarms.
    """
    per_walk = []
    errors = []
    hits = misses = false_alarms = soon_hits = 0
    for i, held in enumerate(walks):
        rest = walks[:i] + walks[i + 1:]
        plan = feat.route_plan(rest)
        xs, ys = zip(*(walk_dataset(w, plan) for w in rest))
        policy = Policy().fit(np.vstack(xs), np.concatenate(ys))
        cues = run_walk(policy, held)
        steps = held["steps"]
        times = np.array([float(s["t"]) for s in steps])
        meters = np.array([int(s["n"]) for s in steps]) * held["meters_per_step"]
        used = set()
        walk_rows = []
        for m in sorted(cue_maneuvers(held, plan), key=lambda m: m["t"]):
            if m["action"] not in cfg.CUE_MANEUVERS or len(times) == 0:
                continue
            at = float(meters[int(np.argmin(np.abs(times - m["t"])))])
            match = None
            for j, (_, cm, cue) in enumerate(cues):
                if j in used or cue != f"{m['action']}_now":
                    continue
                if abs(cm - at) <= cfg.LOOKAHEAD_M:
                    match = j
                    break
            had_soon = any(cue == f"{m['action']}_soon" and at - cfg.LOOKAHEAD_M - cfg.NOW_BEFORE_M - 4 <= cm <= at
                           for _, cm, cue in cues)
            soon_hits += had_soon
            if match is None:
                misses += 1
                walk_rows.append({"action": m["action"], "at_m": round(at, 1), "hit": False})
            else:
                used.add(match)
                hits += 1
                err = cues[match][1] - at
                errors.append(err)
                walk_rows.append({"action": m["action"], "at_m": round(at, 1), "hit": True,
                                  "error_m": round(err, 1), "heads_up": bool(had_soon)})
        fa = sum(1 for j, (_, _, cue) in enumerate(cues) if cue.endswith("_now") and j not in used)
        false_alarms += fa
        per_walk.append({"walk": held["id"], "maneuvers": walk_rows, "false_alarms": fa})
    total = hits + misses
    err = np.array(errors)
    return {
        "walks": len(walks),
        "maneuvers": total,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "hit_rate": round(hits / total, 3) if total else None,
        "heads_up_rate": round(soon_hits / total, 3) if total else None,
        "mean_error_m": round(float(err.mean()), 2) if len(err) else None,
        "mean_abs_error_m": round(float(np.abs(err).mean()), 2) if len(err) else None,
        "std_error_m": round(float(err.std(ddof=1)), 2) if len(err) > 1 else None,
        "per_walk": per_walk,
    }
