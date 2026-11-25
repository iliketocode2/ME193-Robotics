"""Synthetic walks in the shape the navigator's tests and dev server use.

A thin wrapper over foot_sim.py (the physically consistent foot simulator):
it splits the IMU into 10-sample batches with board timestamps, like the sketch
sends them, and adds labeled taps that come 0.3-1.5 s late, like a person
tapping a phone mid-walk.

A route is a list of legs:  ("walk", meters) | ("left",) | ("right",) |
("stairs", rises, +1|-1) | ("stairs",) (= 12 up) | ("arrive",)

    python tools/fake_walk.py            prints what sensing makes of the default route
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foot_sim as fs       # noqa: E402

DEFAULT_ROUTE = [("walk", 60), ("left",), ("walk", 45), ("right",), ("walk", 30), ("stairs", 12, 1),
                 ("walk", 25), ("left",), ("walk", 20), ("arrive",)]


def simulate(route=None, meters_per_step=1.4, seed=0, start_t=1000.0, stride_s=None, length_jitter=0.05):
    """Returns {"batches": [(t_last, rows)], "maneuvers": [{t, action}], "taps": [{t, action}], "truth"}.

    meters_per_step is the stride length (one count per stride of the instrumented foot).
    length_jitter scales each leg by up to +-5% so no two walks are identical.
    """
    route = DEFAULT_ROUTE if route is None else [
        ("stairs", 12, 1) if tuple(leg) == ("stairs",) else leg for leg in route]
    rng = np.random.default_rng(seed + 7919)
    sim = fs.simulate(route, stride_m=meters_per_step, seed=seed, stride_s=stride_s, length_jitter=length_jitter)
    t = sim["t"] + start_t
    imu = sim["imu"]
    batches = []
    for k in range(0, len(t), 10):
        batches.append((float(t[min(k + 9, len(t) - 1)]), imu[k:k + 10].tolist()))
    maneuvers = [{"t": round(m["t"] + start_t, 3), "action": m["action"]} for m in sim["truth"]["maneuvers"]]
    taps = [{"t": round(m["t"] + rng.uniform(0.3, 1.5), 3), "action": m["action"]} for m in maneuvers]
    return {"batches": batches, "maneuvers": maneuvers, "taps": taps, "meters_per_step": meters_per_step,
            "truth": sim["truth"]}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unoq_app" / "python"))
    from sensing import Sensing

    walk = simulate()
    s = Sensing()
    steps = [st for t, rows in walk["batches"] for st in s.feed(t, rows)]
    true = walk["truth"]
    print(f"sensing {s.VERSION}: {len(walk['batches']) / 10:.0f} s of IMU, {len(steps)} steps detected "
          f"({len(true['strides'])} true strides)")
    print(f"final heading {steps[-1].heading_deg:.0f} deg (true {true['final_heading_deg']:.0f})")
    print(f"steps flagged stairs: {sum(st.stairs == 1 for st in steps)} "
          f"(true {sum(x['stairs'] for x in true['strides'])})")
    for m in walk["maneuvers"]:
        print(f"  {m['action']:7s} at t={m['t'] - 1000:.1f} s")
