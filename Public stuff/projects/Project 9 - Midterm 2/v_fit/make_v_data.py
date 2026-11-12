"""Make the V data set, the same way Anna's Code makes its "V" example.

Anna's Code (Public stuff/AnnaCode/network/training.py, EXAMPLES["v"]) draws 120 points with
x uniform in [20, 100] and y = 2|x - 60| + Gaussian noise (sigma = 2). This script does the same thing
with a fixed seed, so the Excel sheet, the Python fit and Anna's Code all use the same points.

The output is a tab-separated file in Anna's Code format (Time, input column, output column),
so it can be loaded straight back into Anna's Code with its Load button.

    python make_v_data.py            -> writes v_data.tsv next to this file
"""
import random
from pathlib import Path

N_POINTS = 120
X_MIN, X_MAX = 20.0, 100.0
NOISE = 2.0
SEED = 193


def v(x: float) -> float:
    return 2 * abs(x - 60)


def make_points(seed: int = SEED):
    rng = random.Random(seed)
    xs = [rng.uniform(X_MIN, X_MAX) for _ in range(N_POINTS)]
    return [(x, v(x) + rng.gauss(0, NOISE)) for x in xs]


def main():
    out = Path(__file__).with_name("v_data.tsv")
    lines = ["Time (s)\tx\ty"]
    for i, (x, y) in enumerate(make_points()):
        lines.append(f"{i * 0.1:.1f}\t{x:.2f}\t{y:.2f}")   # Anna's Code wants a time column first
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {N_POINTS} points to {out}")


if __name__ == "__main__":
    main()
