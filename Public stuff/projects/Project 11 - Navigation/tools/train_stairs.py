"""Train the stairs classifier (unoq_app/python/models/stairs.json).

Training strides come from:
  simulated walks      every stride has a known stairs / flat label
  sensor sessions      recorded on the board in "Sensor data" mode: every stride of a
                       session labeled "stairs" counts as stairs, of "walk"/"turn" as flat
                       (pass the copied walks folder or an Export JSON with raw IMU)

Project 8's sessions are only scored, never trained on: that sensor kept moving
while the foot was planted, so its stride heights are unreliable.

    python tools/train_stairs.py                       simulated walks only
    python tools/train_stairs.py --data sim export.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "unoq_app" / "python")]

import sensor_data as sd        # noqa: E402
import stairs                   # noqa: E402

FLAT = {"walk", "turn", "still"}


def labeled_strides(rec):
    """[(Stride, label)] for a recording, or [] if it has no usable labels."""
    strides = rec.tracked()[0]
    if rec.truth_strides:
        # match each tracked stride to the true stride that ended closest in time
        tt = np.array([s["t_strike"] for s in rec.truth_strides])
        out = []
        for s in strides:
            j = int(np.argmin(np.abs(tt - s.t)))
            if abs(tt[j] - s.t) < 0.4:
                out.append((s, rec.truth_strides[j]["stairs"]))
        return out
    if rec.activity == "stairs":
        return [(s, 1) for s in strides[1:-1]]      # first/last stride may be on the landing
    if rec.activity in FLAT:
        return [(s, 0) for s in strides]
    return []


def fit_logistic(x, y, l2=1e-2, steps=3000, lr=0.5):
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    w, b = np.zeros(z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        w -= lr * (z.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return mean, std, w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["sim"])
    args = ap.parse_args()

    recs = sd.load(args.data)
    pairs = [p for r in recs for p in labeled_strides(r)]
    if not pairs or len({y for _, y in pairs}) < 2:
        sys.exit("Need both stairs and flat strides to train.")
    x = np.array([stairs.features(s) for s, _ in pairs])
    y = np.array([lab for _, lab in pairs], dtype=float)

    # hold out every 4th recording to report accuracy honestly
    names = sorted({r.name for r in recs})
    held = set(names[::4])
    tr = [i for i, r in enumerate(r for r in recs for _ in labeled_strides(r)) if r.name not in held]
    te = [i for i in range(len(y)) if i not in set(tr)]
    mean, std, w, b = fit_logistic(x[tr], y[tr])
    pred = (1 / (1 + np.exp(-(((x[te] - mean) / std) @ w + b)))) >= 0.5
    acc = float(np.mean(pred == y[te])) if te else float("nan")

    mean, std, w, b = fit_logistic(x, y)     # final model on everything
    stairs.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    stairs.MODEL_PATH.write_text(json.dumps({"features": stairs.FEATURES, "mean": mean.tolist(),
                                             "std": std.tolist(), "w": w.tolist(), "b": b,
                                             "trained_on": args.data, "strides": len(y),
                                             "stairs_strides": int(y.sum()), "heldout_accuracy": acc},
                                            indent=2))
    print(f"{len(y)} strides ({int(y.sum())} stairs) from {len(recs)} recordings; "
          f"held-out accuracy {acc:.1%} on {len(te)} strides")
    print("weights (standardized):", dict(zip(stairs.FEATURES, np.round(w, 2))))

    clf = stairs.StairsClassifier()
    for r in sd.load(["p8"]):
        st = r.tracked()[0]
        frac = np.mean([clf.classify(s) for s in st]) if st else 0
        print(f"  P8 {r.name:28s} ({r.activity:6s}): {frac:.0%} of {len(st)} strides called stairs")
    print(f"saved {stairs.MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
