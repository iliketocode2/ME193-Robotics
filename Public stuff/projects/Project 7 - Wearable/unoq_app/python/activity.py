"""
Feature extraction and the two activity classifiers. No neural nets, and no
board-only imports, so this file can be tested on a laptop
(see ../../test_activity_offline.py).

Samples are rows of [ax, ay, az, gx, gy, gz] in g and deg/s at SAMPLE_HZ. The
stream is cut into WINDOW-sample windows every HOP samples, and each window is
classified on its own.

Model 1 -- PeakModel: one number per window, the peak acceleration along a
    single axis (largest |a - mean| in the window, so gravity drops out). Each
    activity is a 1-D Gaussian over that number; the most likely one wins.
    The axis is whichever separates the activities best in the training data
    (Fisher ratio), unless one is forced.

Model 2 -- KnnModel: all six IMU channels plus |a| and |g|, summarised as
    mean / std / min / max, plus the step cadence (dominant frequency of |a|).
    k-nearest-neighbours on standardised features; the majority of the k
    closest training windows wins.
"""

import numpy as np

LABELS = ["still", "walk", "run", "bike"]
SAMPLE_HZ = 100
WINDOW = 200            # 2 s
HOP = 50                # a new prediction every 0.5 s
AXES = ["ax", "ay", "az"]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "|a|", "|g|"]
FEATURE_NAMES = ([f"{s}({c})" for s in ("mean", "std", "min", "max") for c in CHANNELS]
                 + ["cadence_hz", "cadence_strength"])


def windows(samples, window=WINDOW, hop=HOP):
    samples = np.asarray(samples, dtype=float)
    return [samples[i:i + window] for i in range(0, len(samples) - window + 1, hop)]


def peak_features(win):
    """Peak |acceleration - window mean| on each of ax, ay, az, in g."""
    acc = np.asarray(win, dtype=float)[:, :3]
    return np.max(np.abs(acc - acc.mean(axis=0)), axis=0)


def cadence(signal):
    """(dominant frequency in 0.5-5 Hz, fraction of the signal's power at it)."""
    x = signal - signal.mean()
    n = 1024                                    # zero-pad: finer frequency steps than 0.5 Hz
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_HZ)
    band = (freqs >= 0.5) & (freqs <= 5.0)
    total = power[1:].sum()
    if total <= 1e-12:
        return 0.0, 0.0
    i = np.argmax(power[band])
    return float(freqs[band][i]), float(power[band][i] / total)


def full_features(win):
    win = np.asarray(win, dtype=float)
    acc_mag = np.linalg.norm(win[:, :3], axis=1)
    gyro_mag = np.linalg.norm(win[:, 3:6], axis=1)
    chans = np.column_stack([win[:, :6], acc_mag, gyro_mag])
    return np.concatenate([chans.mean(axis=0), chans.std(axis=0),
                           chans.min(axis=0), chans.max(axis=0), cadence(acc_mag)])


class PeakModel:
    MIN_STD = 0.05      # g; keeps a very consistent class (standing still) from claiming everything near it

    def __init__(self, axis=None, stats=None):
        self.axis = axis            # index into AXES
        self.stats = stats or {}    # label -> (mean, std)

    def fit(self, peaks, y, axis=None):
        peaks, y = np.asarray(peaks, dtype=float), np.asarray(y)
        self.axis = self._best_axis(peaks, y) if axis is None else axis
        x = peaks[:, self.axis]
        self.stats = {lab: (float(x[y == lab].mean()), float(max(x[y == lab].std(), self.MIN_STD)))
                      for lab in LABELS if np.any(y == lab)}
        return self

    @staticmethod
    def _best_axis(peaks, y):
        scores = []
        for a in range(peaks.shape[1]):
            groups = [peaks[y == lab, a] for lab in np.unique(y)]
            means = np.array([g.mean() for g in groups])
            within = np.mean([g.var() for g in groups]) + 1e-9
            scores.append(means.var() / within)
        return int(np.argmax(scores))

    def predict(self, peaks):
        x = float(peaks[self.axis])
        labels = list(self.stats)
        loglik = np.array([-0.5 * ((x - m) / s) ** 2 - np.log(s) for m, s in self.stats.values()])
        post = np.exp(loglik - loglik.max())
        post /= post.sum()
        i = int(np.argmax(post))
        return labels[i], float(post[i])

    def to_dict(self):
        return {"axis": self.axis, "stats": self.stats}

    @classmethod
    def from_dict(cls, d):
        return cls(d["axis"], {k: tuple(v) for k, v in d["stats"].items()})


class KnnModel:
    def __init__(self, k=5, mean=None, scale=None, X=None, y=None):
        self.k = k
        self.mean, self.scale = mean, scale
        self.X, self.y = X, y

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        self.scale = np.maximum(X.std(axis=0), 1e-6)
        self.X = (X - self.mean) / self.scale
        self.y = np.asarray(y)
        return self

    def predict(self, x):
        d = np.linalg.norm(self.X - (np.asarray(x) - self.mean) / self.scale, axis=1)
        nearest = self.y[np.argsort(d)[:self.k]]
        labels, votes = np.unique(nearest, return_counts=True)
        best = labels[votes == votes.max()]
        label = next(lab for lab in nearest if lab in best)   # tie -> whichever is closest
        return str(label), float(votes.max() / len(nearest))

    def to_dict(self):
        return {"k": self.k, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "X": np.round(self.X, 4).tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["k"], np.array(d["mean"]), np.array(d["scale"]), np.array(d["X"]), np.array(d["y"]))


MAP_NAMES = [f"peak {a}" for a in AXES] + FEATURE_NAMES


def map_vector(win):
    """Everything the feature-map page can plot for one window: Model 1's peaks, then Model 2's features."""
    return np.concatenate([peak_features(win), full_features(win)])


def pick_map_y(F, y, x_index, confusion=None):
    """The feature (not the x one) that best separates the pair of activities
    Model 1 mixes up most -- or all the activities, if there's no confusion
    matrix yet. Returns (index into MAP_NAMES, the pair or None)."""
    F, y = np.asarray(F, dtype=float), np.asarray(y)
    groups = list(np.unique(y))
    pair = None
    if confusion:
        m = np.array(confusion["matrix"], dtype=float)
        np.fill_diagonal(m, 0)
        m = m + m.T
        if m.max() > 0:
            i, j = np.unravel_index(np.argmax(m), m.shape)
            pair = [confusion["labels"][i], confusion["labels"][j]]
            groups = pair
    best, best_score = None, -1.0
    for f in range(len(AXES), F.shape[1]):          # only Model 2's features
        if f == x_index:
            continue
        vals = [F[y == lab, f] for lab in groups if np.any(y == lab)]
        if len(vals) < 2:
            continue
        score = np.var([v.mean() for v in vals]) / (np.mean([v.var() for v in vals]) + 1e-9)
        if score > best_score:
            best, best_score = f, score
    return best, pair


def _confusion(model_predict, feats, y):
    present = [lab for lab in LABELS if lab in set(y)]
    m = np.zeros((len(present), len(present)), dtype=int)
    for f, lab in zip(feats, y):
        pred = model_predict(f)[0]
        if pred in present:
            m[present.index(lab), present.index(pred)] += 1
    return {"labels": present, "matrix": m.tolist(), "accuracy": float(np.trace(m) / max(m.sum(), 1))}


def train(recordings, peak_axis=None, k=5):
    """recordings: list of (label, samples). Returns (peak_model, knn_model, report).

    The report's accuracies come from the last 25% of every recording, held
    out while fitting (with a gap so no test window overlaps a training one).
    The returned models are then refit on everything.
    """
    gap = WINDOW // HOP - 1
    tr_p, tr_f, tr_y, te_p, te_f, te_y = [], [], [], [], [], []
    for label, samples in recordings:
        wins = windows(samples)
        split = int(len(wins) * 0.75)
        for i, w in enumerate(wins):
            if split <= i < split + gap:
                continue
            p, f = peak_features(w), full_features(w)
            if i < split:
                tr_p.append(p), tr_f.append(f), tr_y.append(label)
            else:
                te_p.append(p), te_f.append(f), te_y.append(label)
    if len(set(tr_y)) < 2:
        raise ValueError("need recordings of at least two activities")

    report = {"train_windows": len(tr_y), "test_windows": len(te_y)}
    if te_y:
        pm = PeakModel().fit(tr_p, tr_y, peak_axis)
        km = KnnModel(k).fit(tr_f, tr_y)
        report["model1"] = _confusion(pm.predict, te_p, te_y)
        report["model2"] = _confusion(km.predict, te_f, te_y)

    all_y = tr_y + te_y
    peak_model = PeakModel().fit(tr_p + te_p, all_y, peak_axis)
    knn_model = KnnModel(k).fit(tr_f + te_f, all_y)
    report["peak_axis"] = AXES[peak_model.axis]
    report["windows_per_label"] = {lab: all_y.count(lab) for lab in LABELS}
    return peak_model, knn_model, report
