"""Walks on disk. One folder per walk under data/walks/:

    imu.csv       t,ax,ay,az,gx,gy,gz    every raw sample (g, deg/s), so walks can be
                                         re-run through improved sensing later
    steps.csv     t,n,heading_deg,stairs one row per detected step
    events.jsonl  one JSON object per line:
                    {"id", "t", "type": "label", "action"}       walker tapped a maneuver
                    {"id", "t", "type": "undo"}                  removes the previous label
                    {"id", "t", "type": "wrong_cue"}             DAgger: the cue just given is wrong
                    {"t", "type": "cue", "action"}               the policy spoke a cue
    meta.json     walker, kind, route, meters_per_step, sensing, start, end,
                  reviewed, include, maneuvers [{t, action}] (set on the review page)

Files are appended and flushed as the walk goes, so a crash or power loss keeps
everything up to that moment.
"""

import json
import re
import time

import numpy as np

import config as cfg


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9-]+", "-", name.strip()).strip("-")[:24] or "walker"


class WalkRecorder:
    def __init__(self, walker, kind, route, meters_per_step, sensing_version, now=None):
        now = time.time() if now is None else now
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        self.id = f"{stamp}_{safe_name(walker)}_{kind}"
        self.folder = cfg.WALKS_DIR / self.id
        n = 1
        while self.folder.exists():
            n += 1
            self.id = f"{stamp}-{n}_{safe_name(walker)}_{kind}"
            self.folder = cfg.WALKS_DIR / self.id
        self.folder.mkdir(parents=True)
        self.meta = {"walker": walker, "kind": kind, "route": route,
                     "meters_per_step": float(meters_per_step), "sensing": sensing_version,
                     "start": now, "end": None, "reviewed": False, "include": True, "maneuvers": []}
        self._write_meta()
        self.imu = open(self.folder / "imu.csv", "w")
        self.imu.write("t,ax,ay,az,gx,gy,gz\n")
        self.steps = open(self.folder / "steps.csv", "w")
        self.steps.write("t,n,heading_deg,stairs\n")
        self.events = open(self.folder / "events.jsonl", "w")
        self.event_ids = set()
        self.step_count = 0

    def _write_meta(self):
        (self.folder / "meta.json").write_text(json.dumps(self.meta, indent=2))

    def add_imu(self, t, rows):
        count = len(rows)
        for i, r in enumerate(rows):
            ts = t - (count - 1 - i) / cfg.SAMPLE_HZ
            self.imu.write(f"{ts:.3f}," + ",".join(f"{v:.4f}" for v in r) + "\n")
        self.imu.flush()

    def add_step(self, step):
        self.steps.write(f"{step.t:.3f},{step.n},{step.heading_deg:.2f},{step.stairs}\n")
        self.steps.flush()
        self.step_count += 1

    def add_event(self, event):
        """Returns False for a repeated event id (the phone resends taps after reconnecting)."""
        eid = event.get("id")
        if eid is not None:
            if eid in self.event_ids:
                return False
            self.event_ids.add(eid)
        self.events.write(json.dumps(event) + "\n")
        self.events.flush()
        return True

    def close(self, now=None):
        self.meta["end"] = time.time() if now is None else now
        self.meta["maneuvers"] = raw_maneuvers(read_events(self.folder))
        self._write_meta()
        for f in (self.imu, self.steps, self.events):
            f.close()
        return self.id


# ---------- reading ----------

def read_steps(folder):
    path = folder / "steps.csv"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        next(f, None)
        for line in f:
            p = line.strip().split(",")
            if len(p) == 4:
                out.append({"t": float(p[0]), "n": int(p[1]), "heading_deg": float(p[2]), "stairs": int(p[3])})
    return out


def read_events(folder):
    path = folder / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass        # a half-written last line after power loss
    return out


def read_meta(folder):
    return json.loads((folder / "meta.json").read_text())


def raw_maneuvers(events):
    """The walker's label taps with undos applied, sorted by time."""
    labels = []
    for e in sorted(events, key=lambda e: e["t"]):
        if e["type"] == "label":
            labels.append({"t": e["t"], "action": e["action"]})
        elif e["type"] == "undo" and labels:
            labels.pop()
    return labels


def list_walks():
    if not cfg.WALKS_DIR.exists():
        return []
    out = []
    for folder in sorted(cfg.WALKS_DIR.iterdir()):
        if not (folder / "meta.json").exists():
            continue
        meta = read_meta(folder)
        steps = read_steps(folder)
        out.append({"id": folder.name, **{k: meta[k] for k in
                    ("walker", "kind", "route", "reviewed", "include", "sensing", "start", "end")},
                    "steps": len(steps), "maneuvers": len(meta["maneuvers"]),
                    "meters": round(len(steps) * meta["meters_per_step"], 1)})
    return out


def load_walk(walk_id):
    folder = _folder(walk_id)
    meta = read_meta(folder)
    return {"id": walk_id, **meta, "steps": read_steps(folder), "events": read_events(folder)}


def _folder(walk_id):
    folder = cfg.WALKS_DIR / walk_id
    if "/" in walk_id or "\\" in walk_id or ".." in walk_id or not (folder / "meta.json").exists():
        raise KeyError(f"no walk {walk_id}")
    return folder


def save_review(walk_id, maneuvers, include, meters_per_step=None):
    folder = _folder(walk_id)
    meta = read_meta(folder)
    meta["maneuvers"] = sorted(({"t": float(m["t"]), "action": m["action"]} for m in maneuvers
                                if m.get("action") in cfg.MANEUVERS), key=lambda m: m["t"])
    meta["include"] = bool(include)
    meta["reviewed"] = True
    if meters_per_step:
        meta["meters_per_step"] = float(meters_per_step)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def training_walks(route=None):
    out = []
    for s in list_walks():
        if s["kind"] in cfg.TRAINING_KINDS and s["reviewed"] and s["include"] and (route is None or s["route"] == route):
            out.append(load_walk(s["id"]))
    return out


def budget(route=None):
    walks = [s for s in list_walks() if s["kind"] in cfg.TRAINING_KINDS and (route is None or s["route"] == route)]
    return {"used": sum(1 for s in walks if s["include"]), "recorded": len(walks), "limit": cfg.WALK_BUDGET}


def reprocess(walk_id, sensing):
    """Re-run sensing over a walk's saved raw IMU and rewrite steps.csv."""
    folder = _folder(walk_id)
    path = folder / "imu.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2) if path.stat().st_size > 40 else np.zeros((0, 7))
    sensing.reset()
    steps = []
    for start in range(0, len(data), 10):
        chunk = data[start:start + 10]
        steps.extend(sensing.feed(float(chunk[-1, 0]), chunk[:, 1:].tolist()))
    with open(folder / "steps.csv", "w") as f:
        f.write("t,n,heading_deg,stairs\n")
        for s in steps:
            f.write(f"{s.t:.3f},{s.n},{s.heading_deg:.2f},{s.stairs}\n")
    meta = read_meta(folder)
    meta["sensing"] = sensing.VERSION
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return len(steps)


# ---------- walkers ----------

def load_walkers():
    if cfg.WALKERS_PATH.exists():
        return json.loads(cfg.WALKERS_PATH.read_text())
    return {}


def save_walker(name, meters_per_step):
    walkers = load_walkers()
    walkers[name] = {"meters_per_step": round(float(meters_per_step), 4), "calibrated": time.time()}
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.WALKERS_PATH.write_text(json.dumps(walkers, indent=2))
    return walkers
