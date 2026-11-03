"""Sessions on disk -> resampled, normalized, windowed training data.

A session is two files in data/sessions/:
    <name>.csv   t,ax,ay,az,gx,gy,gz   (t in s on the board's sample clock)
    <name>.json  {"activity", "steps": [step times in s], "true_steps", "trial", ...}
"""

import json
import random
from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class Session:
    name: str
    meta: dict
    t: np.ndarray
    x: np.ndarray
    events: np.ndarray = field(default=None)

    @property
    def activity(self):
        return self.meta.get("activity", "walk")

    @property
    def is_trial(self):
        return bool(self.meta.get("trial"))

    @property
    def labeled(self):
        return self.meta.get("labeled", False)


def save_session(name, t, x, meta, folder=config.SESSIONS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    np.savetxt(folder / f"{name}.csv", np.column_stack([t, x]), delimiter=",", fmt="%.4f",
               header="t," + ",".join(config.CHANNELS), comments="")
    save_meta(name, meta, folder)


def save_meta(name, meta, folder=config.SESSIONS_DIR):
    (folder / f"{name}.json").write_text(json.dumps(meta, indent=1))


def resample(t, x, hz=config.SAMPLE_HZ):
    """Linear interpolation onto an even grid, filling any dropped batches."""
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / hz)
    return grid, np.column_stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])])


def load_session(name, folder=config.SESSIONS_DIR):
    raw = np.loadtxt(folder / f"{name}.csv", delimiter=",", skiprows=1, ndmin=2)
    meta = json.loads((folder / f"{name}.json").read_text())
    t, x = resample(raw[:, 0], raw[:, 1:7])
    events = np.zeros(len(t), dtype=np.float32)
    for s in meta.get("steps", []):
        i = int(round((s - t[0]) * config.SAMPLE_HZ))
        if 0 <= i < len(t):
            events[i] = 1.0
    return Session(name, meta, t, x, events)


def load_sessions(folder=config.SESSIONS_DIR):
    names = sorted(p.stem for p in folder.glob("*.json") if (folder / f"{p.stem}.csv").exists())
    return [load_session(n, folder) for n in names]


def split_sessions(sessions, seed=0):
    """Training sessions (labeled, not trials) split by session, per activity, ~70/15/15."""
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    by_activity = {}
    for s in sessions:
        if s.labeled and not s.is_trial:
            by_activity.setdefault(s.activity, []).append(s.name)
    for names in by_activity.values():
        rng.shuffle(names)
        n_hold = round(0.15 * len(names)) if len(names) >= 7 else (1 if len(names) >= 3 else 0)
        split["test"] += names[:n_hold]
        split["val"] += names[n_hold:2 * n_hold]
        split["train"] += names[2 * n_hold:]
    if not split["val"] and len(split["train"]) >= 2:
        split["val"].append(split["train"].pop(rng.randrange(len(split["train"]))))
    return split


def norm_stats(sessions):
    x = np.concatenate([s.x for s in sessions])
    return x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)


def normalize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def window_targets(events, centers, tol=config.EVENT_TOLERANCE):
    """1 where a step lies within `tol` samples of the judged sample."""
    near = np.convolve(events, np.ones(2 * tol + 1), mode="same") > 0
    return near[centers].astype(np.float32)


def make_windows(x, centers, window=config.WINDOW, center=config.CENTER):
    """(len(centers), channels, window) slices with each index at position `center`, edge-padded."""
    padded = np.pad(x, ((center, window - center), (0, 0)), mode="edge")
    idx = centers[:, None] + np.arange(window)[None, :]
    return np.ascontiguousarray(padded[idx].transpose(0, 2, 1))


def build_arrays(sessions, mean, std, channels, hop=config.TRAIN_HOP, center=config.CENTER):
    xs, ys = [], []
    for s in sessions:
        centers = np.arange(0, len(s.t), hop)
        xs.append(make_windows(normalize(s.x, mean, std)[:, channels], centers, center=center))
        ys.append(window_targets(s.events, centers))
    return np.concatenate(xs), np.concatenate(ys)
