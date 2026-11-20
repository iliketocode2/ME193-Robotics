"""Is this stride on stairs? A small classifier on what the foot tracker measured.

Each completed stride (foot_ins.Stride) gives a height change, a horizontal
length, a duration and a swing time. On stairs the foot rises or falls about
two steps (~0.3 m) per stride and travels less far forward; on flat ground the
height change is near zero. A logistic regression trained by
tools/train_stairs.py weighs those four numbers; until it's trained, a plain
threshold on the height change is used.
"""

import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "stairs.json"
FEATURES = ["abs_dz", "length", "duration", "swing"]
FALLBACK_DZ_M = 0.17        # without a trained model: more than one stair of height change = stairs


def features(stride):
    return np.array([abs(stride.dz), stride.length, stride.duration, stride.swing], dtype=float)


class StairsClassifier:
    def __init__(self, path=MODEL_PATH):
        self.model = None
        try:
            m = json.loads(Path(path).read_text())
            if m.get("features") == FEATURES:
                self.model = {k: np.array(m[k], dtype=float) for k in ("mean", "std", "w")}
                self.model["b"] = float(m["b"])
        except (OSError, ValueError, KeyError):
            pass

    def prob(self, stride):
        if self.model is None:
            return 1.0 if abs(stride.dz) > FALLBACK_DZ_M else 0.0
        z = (features(stride) - self.model["mean"]) / self.model["std"]
        return float(1.0 / (1.0 + np.exp(-(z @ self.model["w"] + self.model["b"]))))

    def classify(self, stride):
        """-> 1 on stairs, 0 not. (Direction is the sign of stride.dz.)"""
        return int(self.prob(stride) >= 0.5)
