"""Load IMU recordings for training the step network and the stairs classifier.

Sources (all become a `Recording`: times, raw IMU N x 6, activity, where it came from):
  training_data/p8/*.csv + .json      Project 8 sessions (activity in the json)
  <walks folder>/<walk>/imu.csv       Project 11 walks copied off the board
  an Export JSON from the Review page made with "Export with raw IMU"
  foot_sim                            simulated walks with known truth

Labels come from the foot tracker (foot_ins.py): one step = the start of each
stance phase that follows a real swing, so every recording is labeled the same
way (one count per stride of the instrumented foot). `check_plots` draws them
so you can see that they sit where they should before training on them.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "unoq_app" / "python"), str(ROOT / "tools")]

import foot_ins as fi           # noqa: E402

P8_DIR = ROOT / "training_data" / "p8"


@dataclass
class Recording:
    name: str
    t: np.ndarray
    imu: np.ndarray
    activity: str = "walk"          # walk | turn | stairs | still | mixed
    source: str = ""
    truth_strides: list = field(default_factory=list)   # simulated walks only
    _tracked: tuple = None

    def tracked(self):
        """(strides, channels, stance, heading) from the foot tracker, computed once."""
        if self._tracked is None:
            self._tracked = fi.run(self.t, self.imu)
        return self._tracked


def _uniform(t, imu):
    """Recordings should already be 100 Hz; resample if a few batches went missing."""
    dt = np.diff(t)
    if len(dt) and np.all(np.abs(dt - 0.01) < 0.002):
        return t, imu
    grid = np.arange(t[0], t[-1] + 1e-9, 0.01)
    return grid, np.column_stack([np.interp(grid, t, imu[:, c]) for c in range(6)])


def load_p8(folder=P8_DIR):
    out = []
    for csv in sorted(Path(folder).glob("*.csv")):
        meta = json.loads(csv.with_suffix(".json").read_text())
        d = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)
        if len(d) < 200:
            continue
        t, imu = _uniform(d[:, 0], d[:, 1:7])
        out.append(Recording(csv.stem, t, imu, meta.get("activity", "walk"), "p8"))
    return out


def load_walk_folders(folder):
    """Project 11 walk folders (data/walks/ copied off the board)."""
    out = []
    for w in sorted(Path(folder).iterdir()):
        csv, meta_p = w / "imu.csv", w / "meta.json"
        if not csv.exists() or not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        d = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)
        if len(d) < 200:
            continue
        t, imu = _uniform(d[:, 0], d[:, 1:7])
        out.append(Recording(w.name, t, imu, meta.get("activity") or "mixed", "p11"))
    return out


def load_export(path):
    """An Export JSON that includes raw IMU (walk["imu"] = [[t, ax, ..., gz], ...])."""
    data = json.loads(Path(path).read_text())
    out = []
    for w in data["walks"]:
        if not w.get("imu") or len(w["imu"]) < 200:
            continue
        d = np.array(w["imu"], dtype=float)
        t, imu = _uniform(d[:, 0], d[:, 1:7])
        out.append(Recording(w["id"], t, imu, w.get("activity") or "mixed", "p11"))
    return out


def load_sim(count=12, seed=100):
    """Simulated walks: varied strides, mounts, turns and stairs."""
    import foot_sim as fs
    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        route = [("walk", rng.uniform(10, 25)), ("left",), ("walk", rng.uniform(5, 15)),
                 ("stairs", int(rng.integers(8, 16)), 1), ("walk", rng.uniform(4, 10)), ("still", 2.0),
                 ("right",), ("walk", rng.uniform(5, 15)), ("stairs", int(rng.integers(8, 16)), -1),
                 ("walk", rng.uniform(5, 15))]
        sim = fs.simulate(route, stride_m=float(rng.uniform(1.1, 1.7)), seed=seed + k,
                          stride_s=float(rng.uniform(0.95, 1.35)))
        out.append(Recording(f"sim{k:02d}", sim["t"], sim["imu"], "mixed", "sim",
                             truth_strides=sim["truth"]["strides"]))
    return out


def load(sources):
    """sources: list of 'p8', 'sim', a walks folder, or an export .json."""
    recs = []
    for s in sources:
        if s == "p8":
            recs += load_p8()
        elif s == "sim":
            recs += load_sim()
        elif str(s).endswith(".json"):
            recs += load_export(s)
        else:
            recs += load_walk_folders(s)
    return recs


def step_times(rec):
    return np.array([s.t for s in rec.tracked()[0]])


def check_plots(recs, out_dir, seconds=20):
    """One PNG per recording: |gyro|, |accel|, stance, and the step labels, for eyeballing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        strides, _, stance, _ = rec.tracked()
        t0 = rec.t[0] + max(0.0, (rec.t[-1] - rec.t[0] - seconds) / 2)
        m = (rec.t >= t0) & (rec.t <= t0 + seconds)
        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.plot(rec.t[m], np.linalg.norm(rec.imu[m, 3:], axis=1) / 100, lw=1, label="|gyro| / 100 deg/s")
        ax.plot(rec.t[m], np.linalg.norm(rec.imu[m, :3], axis=1), lw=1, label="|accel| g")
        ax.fill_between(rec.t[m], -0.6, -0.6 + 0.4 * stance[m], step="mid", alpha=0.4, label="stance")
        for s in strides:
            if t0 <= s.t <= t0 + seconds:
                ax.axvline(s.t, color="g", ls="--", lw=1)
        ax.set_title(f"{rec.name} ({rec.source}, {rec.activity}): {len(strides)} steps, green = step labels")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{rec.name}.png", dpi=80)
        plt.close(fig)
