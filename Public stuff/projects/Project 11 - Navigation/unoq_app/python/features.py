"""Steps -> the state the policy sees, and cleaned maneuvers -> per-step labels.

The same `StateTracker` runs during training (over a recorded walk) and live
(one step at a time), so the policy always sees features computed exactly the
way it was trained on.
"""

import numpy as np

import config as cfg


class StateTracker:
    """Feed steps one at a time; `update` returns that step's feature vector."""

    def __init__(self, meters_per_step):
        self.mps = float(meters_per_step)
        self.history = []           # (meters, heading) per step, for turn detection
        self.turns = 0
        self.last_turn_m = 0.0
        self.last_turn_heading = 0.0
        self.last_turn_sign = 0
        self.turn_marks = []        # (step index, sign) where a turn was detected

    def update(self, step):
        """step: object or dict with n, heading_deg, stairs."""
        n, heading, stairs = _fields(step)
        meters = n * self.mps
        self.history.append((meters, heading))

        # a turn = heading moved TURN_DEG away from the heading after the last turn,
        # within TURN_SPAN_M of walking
        for m_back, h_back in reversed(self.history[:-1]):
            if meters - m_back > cfg.TURN_SPAN_M:
                break
            change = heading - h_back
            if abs(change) >= cfg.TURN_DEG and abs(heading - self.last_turn_heading) >= cfg.TURN_DEG:
                self.turns += 1
                self.last_turn_m = meters
                self.last_turn_heading = heading
                self.last_turn_sign = 1 if change > 0 else -1
                self.turn_marks.append((len(self.history) - 1, self.last_turn_sign))
                break

        return np.array([
            meters,
            meters - self.last_turn_m,
            self.turns,
            heading,
            self.last_turn_sign,
            max(stairs, 0),
        ], dtype=float)


def _fields(step):
    if isinstance(step, dict):
        return int(step["n"]), float(step["heading_deg"]), int(step.get("stairs", -1))
    return int(step.n), float(step.heading_deg), int(step.stairs)


def walk_features(steps, meters_per_step):
    """All feature vectors for a recorded walk, plus where turns were detected."""
    tracker = StateTracker(meters_per_step)
    rows = [tracker.update(s) for s in steps]
    x = np.array(rows) if rows else np.zeros((0, len(cfg.FEATURES)))
    return x, tracker.turn_marks


def walk_labels(steps, maneuvers, meters_per_step):
    """Per-step class index from the cleaned maneuver list [{t, action}, ...].

    Each maneuver is placed at the step nearest its time. Steps from NOW_BEFORE_M
    before it to NOW_AFTER_M after get "<action>_now", the LOOKAHEAD_M before that
    "<action>_soon", the rest
    "straight". When zones overlap, the later maneuver's "soon" never overwrites an
    earlier maneuver's "now".
    """
    n = len(steps)
    labels = np.zeros(n, dtype=int)
    if n == 0:
        return labels
    times = np.array([_t(s) for s in steps])
    meters = np.array([_fields(s)[0] for s in steps]) * meters_per_step
    now_mask = np.zeros(n, dtype=bool)
    for m in sorted(maneuvers, key=lambda m: m["t"]):
        action = m["action"]
        if action not in cfg.CUE_MANEUVERS:
            continue
        i = int(np.argmin(np.abs(times - m["t"])))
        at = meters[i]
        start = at - cfg.NOW_BEFORE_M
        soon = (meters >= start - cfg.LOOKAHEAD_M) & (meters < start) & ~now_mask
        now = (meters >= start) & (meters <= at + cfg.NOW_AFTER_M)
        labels[soon] = cfg.CLASSES.index(f"{action}_soon")
        labels[now] = cfg.CLASSES.index(f"{action}_now")
        now_mask |= now
    return labels


# ---------- combined maneuvers (see config.COMBINE_M) ----------

def _sequence(maneuvers):
    return sorted((m for m in maneuvers if m["action"] in cfg.MANEUVERS), key=lambda m: m["t"])


def _positions(steps, seq, meters_per_step):
    """Meters walked at each maneuver (step nearest its time)."""
    if not steps:
        return np.zeros(len(seq))
    times = np.array([_t(s) for s in steps])
    meters = np.array([_fields(s)[0] for s in steps]) * meters_per_step
    return np.array([meters[int(np.argmin(np.abs(times - m["t"])))] for m in seq])


def route_plan(walks):
    """{"actions": [...], "combine": [bool per neighbouring pair]} for the route, or None.

    Only when every walk has the same maneuver sequence: then each neighbouring pair is
    combined if its median gap over the walks is under COMBINE_M, so all walks (and the
    held-out walk in the report) are labeled the same way.
    """
    seqs = [_sequence(w["maneuvers"]) for w in walks]
    if not seqs or not seqs[0]:
        return None
    actions = [m["action"] for m in seqs[0]]
    if any([m["action"] for m in s] != actions for s in seqs):
        return None
    gaps = np.array([np.diff(_positions(w["steps"], s, w["meters_per_step"])) for w, s in zip(walks, seqs)])
    median = np.median(gaps, axis=0) if gaps.size else np.zeros(0)
    return {"actions": actions, "combine": [bool(g < cfg.COMBINE_M) for g in median],
            "median_gap_m": [round(float(g), 1) for g in median]}


def combine_maneuvers(steps, maneuvers, meters_per_step, plan=None):
    """The maneuver list with close pairs merged into one "<a>+<b>" maneuver at the first one's place.

    Uses the route plan when this walk's sequence matches it, otherwise this walk's own gaps.
    Pairs are taken greedily from the start, so three close maneuvers become a pair plus one.
    """
    seq = _sequence(maneuvers)
    if plan and [m["action"] for m in seq] == plan["actions"]:
        combine = plan["combine"]
    else:
        combine = [bool(g < cfg.COMBINE_M) for g in np.diff(_positions(steps, seq, meters_per_step))]
    out, i = [], 0
    while i < len(seq):
        a = seq[i]
        if i + 1 < len(seq) and combine[i] and f"{a['action']}+{seq[i + 1]['action']}" in cfg.COMBINED:
            out.append({"t": a["t"], "action": f"{a['action']}+{seq[i + 1]['action']}"})
            i += 2
        else:
            out.append({"t": a["t"], "action": a["action"]})
            i += 1
    return out


def _t(step):
    return float(step["t"]) if isinstance(step, dict) else float(step.t)


def snap_suggestions(steps, maneuvers, meters_per_step):
    """For the review page: where each turn tap would move if snapped to a detected turn.

    Returns a list (same order as maneuvers) of suggested times, or None when no
    detected turn of the right direction is within SNAP_WINDOW_M. Left/right snap
    to the start of the heading change; other maneuvers are left alone.
    """
    if not steps:
        return [None] * len(maneuvers)
    times = np.array([_t(s) for s in steps])
    heading = np.array([_fields(s)[1] for s in steps])
    meters = np.array([_fields(s)[0] for s in steps]) * meters_per_step
    turn_starts = detected_turn_starts(meters, heading)
    out = []
    for m in maneuvers:
        if m["action"] not in ("left", "right") or not turn_starts:
            out.append(None)
            continue
        want = 1 if m["action"] == "left" else -1
        i = int(np.argmin(np.abs(times - m["t"])))
        best = None
        for j, sign in turn_starts:
            if sign != want:
                continue
            d = abs(meters[j] - meters[i])
            if d <= cfg.SNAP_WINDOW_M and (best is None or d < best[0]):
                best = (d, j)
        out.append(None if best is None else float(times[best[1]]))
    return out


def detected_turn_starts(meters, heading):
    """(step index where the turn begins, sign) for every turn in a walk.

    A turn is a stretch where heading changes by at least TURN_DEG within TURN_SPAN_M.
    Its start is the first step of that stretch where the heading began moving.
    """
    out = []
    i, n = 0, len(meters)
    while i < n:
        j = i + 1
        found = None
        while j < n and meters[j] - meters[i] <= cfg.TURN_SPAN_M:
            if abs(heading[j] - heading[i]) >= cfg.TURN_DEG:
                found = j
                break
            j += 1
        if found is None:
            i += 1
            continue
        sign = 1 if heading[found] > heading[i] else -1
        # start = last step before the heading moved more than 10% of the way
        start = i
        target = abs(heading[found] - heading[i]) * 0.1
        while start + 1 < found and abs(heading[start + 1] - heading[i]) < target:
            start += 1
        out.append((start, sign))
        # skip past the rest of this turn
        k = found
        while k + 1 < n and abs(heading[k + 1] - heading[k]) > cfg.TURN_DEG / 6:
            k += 1
        i = k + 1
    return out
