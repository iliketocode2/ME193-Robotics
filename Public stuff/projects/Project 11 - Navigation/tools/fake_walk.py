"""Synthetic shoe-IMU walks, for testing the navigator without the board.

A route is a list of legs:  ("walk", meters) | ("left",) | ("right",) | ("stairs",) | ("arrive",)

The fake shoe is mounted at a random fixed angle. Each stride has a heel-strike
jolt, a swing where the foot pitches forward and back, and a still stance phase.
Turns spread the heading change over about two strides. The gyro has a constant
zero offset and noise. The labeled taps come 0.3-1.5 s late, like a person
tapping a phone mid-walk.

    python tools/fake_walk.py            prints a summary of the default route
"""

import numpy as np

HZ = 100
DEFAULT_ROUTE = [("walk", 60), ("left",), ("walk", 45), ("right",), ("walk", 30), ("stairs",),
                 ("walk", 25), ("left",), ("walk", 20), ("arrive",)]


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def simulate(route=None, meters_per_step=1.4, seed=0, start_t=1000.0, stride_s=None, length_jitter=0.05):
    """Returns {"batches": [(t_last, rows)], "maneuvers": [{t, action}], "taps": [{t, action}]}.

    length_jitter scales each leg by up to +-5% so no two walks are identical.
    """
    route = DEFAULT_ROUTE if route is None else route
    rng = np.random.default_rng(seed)
    rot = _rotation(rng)                 # world -> sensor
    bias = rng.normal(0, 1.5, 3)         # deg/s
    stride_s = stride_s or (0.95 + 0.1 * meters_per_step / 1.4) * rng.uniform(0.95, 1.05)
    n_stride = int(stride_s * HZ)

    # build a per-stride plan: (yaw deg this stride, maneuver to mark at this stride or None)
    plan = [(0.0, None)] * 3             # stand-still lead-in counted as strides with no motion
    for leg in route:
        if leg[0] == "walk":
            meters = leg[1] * rng.uniform(1 - length_jitter, 1 + length_jitter)
            plan += [(0.0, "step")] * max(1, round(meters / meters_per_step))
        elif leg[0] in ("left", "right"):
            angle = rng.uniform(80, 100) * (1 if leg[0] == "left" else -1)
            plan += [(angle / 2, ("step", leg[0])), (angle / 2, "step")]
        else:
            plan += [(0.0, ("step", leg[0]))]
    plan += [(0.0, None)] * 2

    rows, maneuvers = [], []
    t = start_t
    for yaw, what in plan:
        moving = what is not None
        mark = what[1] if isinstance(what, tuple) else None
        if mark:
            maneuvers.append({"t": round(t, 3), "action": mark})
        for i in range(n_stride):
            ph = i / n_stride
            acc = np.array([0.0, 0.0, 1.0])
            gyro = np.zeros(3)
            if moving:
                if ph < 0.06:                                   # heel strike
                    acc += [0.3, 0.0, 2.2 * np.sin(np.pi * ph / 0.06)]
                elif 0.45 < ph < 0.95:                          # swing
                    s = (ph - 0.45) / 0.5
                    gyro[1] = 250 * np.sin(2 * np.pi * s)
                    acc += [0.4 * np.sin(2 * np.pi * s), 0.0, 0.25 * np.sin(np.pi * s)]
                if yaw and 0.4 < ph < 1.0:
                    gyro[2] += yaw / (0.6 * stride_s) * np.sin(np.pi * (ph - 0.4) / 0.6) * np.pi / 2
            acc_s = rot @ acc + rng.normal(0, 0.02, 3)
            gyro_s = rot @ gyro + bias + rng.normal(0, 0.8, 3)
            rows.append(np.concatenate([acc_s, gyro_s]))
            t += 1.0 / HZ
    rows = np.array(rows)
    batches = []
    t0 = start_t
    for k in range(0, len(rows), 10):
        chunk = rows[k:k + 10]
        batches.append((t0 + (k + len(chunk) - 1) / HZ, chunk.tolist()))
    taps = [{"t": round(m["t"] + rng.uniform(0.3, 1.5), 3), "action": m["action"]} for m in maneuvers]
    return {"batches": batches, "maneuvers": maneuvers, "taps": taps, "meters_per_step": meters_per_step}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unoq_app" / "python"))
    from sensing import Sensing

    walk = simulate()
    s = Sensing()
    steps = [st for t, rows in walk["batches"] for st in s.feed(t, rows)]
    print(f"{len(walk['batches']) / 10:.0f} s of IMU, {len(steps)} steps detected")
    print(f"final heading {steps[-1].heading_deg:.0f} deg (true route: left + right + left, about +90)")
    for m in walk["maneuvers"]:
        print(f"  {m['action']:7s} at t={m['t'] - 1000:.1f} s")
