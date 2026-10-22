"""
Laptop-only check of unoq_app/python/activity.py (no board needed): makes
fake shoe-IMU recordings for each activity, trains both models, and prints
their held-out accuracy and a few live-style predictions. Real data will be
messier -- this only proves the pipeline runs end to end.

    my_env\\Scripts\\python "Public stuff/projects/Project 7 - Wearable/test_activity_offline.py"
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "unoq_app" / "python"))
import activity as act

rng = np.random.default_rng(0)


def fake(label, seconds):
    """Rough shoe-top IMU: gravity on z when upright, on x when lying on your back."""
    t = np.arange(int(seconds * act.SAMPLE_HZ)) / act.SAMPLE_HZ
    n = len(t)
    acc = np.zeros((n, 3))
    gyro = np.zeros((n, 3))
    if label == "bike":
        acc[:, 0] = -1.0                                  # foot pointing up while air-biking
        f = 1.2
        acc[:, 1] += 0.8 * np.sin(2 * np.pi * f * t)
        acc[:, 2] += 0.8 * np.cos(2 * np.pi * f * t)
        gyro[:, 1] = 120 * np.sin(2 * np.pi * f * t)
    else:
        acc[:, 2] = 1.0
    if label in ("walk", "run"):
        f, impact, swing = (0.9, 2.0, 150) if label == "walk" else (1.4, 6.0, 450)
        phase = (t * f) % 1.0
        heel = np.exp(-((phase - 0.05) / 0.02) ** 2) * impact
        acc[:, 2] += heel
        acc[:, 0] += 0.6 * impact / 2 * np.sin(2 * np.pi * f * t)
        gyro[:, 1] = swing * np.sin(2 * np.pi * f * t)
    acc += rng.normal(0, 0.03, acc.shape)
    gyro += rng.normal(0, 2.0, gyro.shape)
    return np.column_stack([acc, gyro])


recordings = [(lab, fake(lab, 40)) for lab in act.LABELS for _ in range(2)]
peak, knn, report = act.train(recordings)

print(f"peak axis: {report['peak_axis']}   windows per label: {report['windows_per_label']}")
for key, name in (("model1", "Model 1 (peak)"), ("model2", "Model 2 (k-NN)")):
    r = report[key]
    print(f"{name}: {r['accuracy'] * 100:.0f}% held-out   confusion {r['matrix']}")
print("peak stats:", {k: tuple(round(x, 2) for x in v) for k, v in peak.stats.items()})

# round-trip through the JSON-able dicts main.py saves, then classify fresh data
peak = act.PeakModel.from_dict(peak.to_dict())
knn = act.KnnModel.from_dict(knn.to_dict())
for lab in act.LABELS:
    win = act.windows(fake(lab, 3))[0]
    print(f"fresh {lab:5s} -> model1 {peak.predict(act.peak_features(win))}  "
          f"model2 {knn.predict(act.full_features(win))}")
