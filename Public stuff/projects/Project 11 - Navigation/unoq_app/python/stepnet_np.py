"""The step network's forward pass in plain numpy, for the UNO Q (no PyTorch there).

Same architecture as Project 8's StepNet, retrained by tools/train_stepnet.py
on four mount-independent channels from the foot tracker (vertical / horizontal
acceleration and rotation rate) instead of raw sensor axes, so it keeps working
when the board is mounted differently. One window = 1 s (100 samples); the
question it answers is "does a stride end at sample 70 of this window?".

  conv(4->16, k7) + ReLU, max-pool 2       batch norm is folded into each conv's
  conv(16->32, k5) + ReLU, max-pool 2      weights and bias at export, so the
  conv(32->32, k5) + ReLU                  numpy version has nothing to normalize
  linear(800->32) + ReLU, linear(32->1), sigmoid

train_stepnet.py checks this file and PyTorch give the same probabilities.
"""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODEL_PATH = Path(__file__).resolve().parent / "models" / "stepnet.npz"


def conv1d(x, w, b):
    """x: (C_in, L), w: (C_out, C_in, k); zero padding keeps L."""
    k = w.shape[-1]
    p = k // 2
    c_in, length = x.shape
    xp = np.pad(x, ((0, 0), (p, p)))
    windows = sliding_window_view(xp, k, axis=1)           # (C_in, L, k)
    cols = windows.transpose(1, 0, 2).reshape(length, c_in * k)   # one row per output position
    return w.reshape(w.shape[0], -1) @ cols.T + b[:, None]


def maxpool2(x):
    n = x.shape[1] // 2
    return x[:, : 2 * n].reshape(x.shape[0], n, 2).max(axis=2)


class StepNet:
    def __init__(self, path=MODEL_PATH):
        with np.load(path) as f:
            self.p = {k: f[k].astype(np.float32) for k in f.files}
        self.mean = self.p["mean"]
        self.std = self.p["std"]
        self.window = int(self.p["window"])
        self.center = int(self.p["center"])
        self.threshold = float(self.p["threshold"])
        self.min_gap = float(self.p["min_gap"])

    def prob(self, x):
        """x: (window, 4) raw tracker channels, oldest first -> P(stride ends at sample `center`)."""
        p = self.p
        h = ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).T
        h = maxpool2(np.maximum(conv1d(h, p["c1_w"], p["c1_b"]), 0))
        h = maxpool2(np.maximum(conv1d(h, p["c2_w"], p["c2_b"]), 0))
        h = np.maximum(conv1d(h, p["c3_w"], p["c3_b"]), 0)
        h = np.maximum(p["f1_w"] @ h.reshape(-1) + p["f1_b"], 0)
        z = float(p["f2_w"] @ h + p["f2_b"])
        return 1.0 / (1.0 + np.exp(-z))


class StepDetector:
    """Online debouncer from Project 8: each run of probability above the threshold
    becomes one step at its peak; steps closer than min_gap seconds merge."""

    def __init__(self, threshold, min_gap):
        self.threshold = threshold
        self.min_gap = min_gap
        self.peak = None
        self.last_step = -np.inf

    def update(self, t, p):
        """Feed one (time, probability); returns the step time once confirmed, else None."""
        if p >= self.threshold:
            if self.peak is None or p > self.peak[1]:
                self.peak = (t, p)
            return None
        if self.peak is None:
            return None
        step_t, self.peak = self.peak[0], None
        if step_t - self.last_step < self.min_gap:
            return None
        self.last_step = step_t
        return step_t


def load():
    """The trained network, or None if it hasn't been trained/copied to the board yet."""
    try:
        return StepNet()
    except (OSError, KeyError):
        return None
