"""Window probabilities -> discrete steps (debounced), plus event matching for scoring."""

import numpy as np
import torch

import config
import dataset


class StepDetector:
    """Online debouncer: each excursion of the probability above `threshold`
    becomes one step at its peak, and steps closer than `min_gap` s merge."""

    def __init__(self, threshold, min_gap=config.MIN_STEP_GAP_S):
        self.threshold = threshold
        self.min_gap = min_gap
        self.peak = None
        self.last_step = -np.inf

    def update(self, t, p):
        """Feed one (time, probability); returns the step time when one is confirmed, else None."""
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


def detect_steps(times, probs, threshold, min_gap=config.MIN_STEP_GAP_S):
    det = StepDetector(threshold, min_gap)
    steps = [s for t, p in zip(times, probs) if (s := det.update(t, p)) is not None]
    if (s := det.update(np.inf, -1.0)) is not None:
        steps.append(s)
    return np.array(steps)


@torch.no_grad()
def session_probs(model, ckpt, session, hop=config.LIVE_HOP):
    """Probability of a step at every `hop`-th sample of a session."""
    x = dataset.normalize(session.x, ckpt["mean"], ckpt["std"])[:, ckpt["channels"]]
    centers = np.arange(0, len(session.t), hop)
    windows = torch.from_numpy(dataset.make_windows(x, centers, ckpt["window"], ckpt["center"]))
    probs = torch.cat([torch.sigmoid(model(b)) for b in windows.split(1024)]).numpy()
    return session.t[centers], probs


def count_session(model, ckpt, session):
    times, probs = session_probs(model, ckpt, session)
    return detect_steps(times, probs, ckpt["threshold"])


def match_events(pred, true, tol=config.MATCH_TOLERANCE_S):
    """Greedy one-to-one matching within `tol` s. Returns (hits, false positives, misses)."""
    true = list(np.sort(true))
    hits = 0
    for p in np.sort(pred):
        if true:
            j = int(np.argmin([abs(p - t) for t in true]))
            if abs(true[j] - p) <= tol:
                true.pop(j)
                hits += 1
    return hits, len(pred) - hits, len(true)
