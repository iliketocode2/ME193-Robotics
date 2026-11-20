"""Physically consistent foot-mounted IMU simulator.

Unlike a hand-drawn signal, this moves a rigid foot through space and derives
what the IMU on it would read, so orientation tracking, stride lengths and
stair heights can be tested against a known truth:

  stance  (first ~40% of each stride)   foot flat and still, rolling slightly heel to toe
  swing   (rest of the stride)          foot travels one stride forward on a smooth
                                        path, lifts, pitches toe-up then toe-down,
                                        turns if the route turns, rises on stairs
  strike                                a short ringing jolt as the heel lands

Gyro = the sensor's angular velocity in its own frame. Accelerometer = specific
force (acceleration minus gravity, in g) in its own frame. The board sits at a
random fixed angle on the shoe. Both sensors get a constant offset and noise.

A route is a list of legs:
    ("walk", meters)          flat walking
    ("left",) / ("right",)    a ~90 degree turn spread over two strides
    ("stairs", rises, +1|-1)  a flight of `rises` steps up (+1) or down (-1)
    ("still", seconds)        standing still
    ("arrive",)               marks the destination (no motion)

    simulate(route, stride_m=1.4, seed=0) -> dict with t, imu (N x 6), and the truth
"""

import numpy as np

HZ = 100
DT = 1.0 / HZ
G = 9.81
RISE_M = 0.17           # one stair
STAIR_STRIDE_M = 0.56   # one stride of the instrumented foot covers two stairs


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def log_so3(r):
    """Rotation matrix -> rotation vector."""
    cos = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    return angle / (2 * np.sin(angle)) * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


# swing pitch keyframes (fraction of swing, degrees): continues the stance's toe roll,
# toe-off, swing forward, heel-first landing, flat again
PITCH_KEYS = ([0.0, 0.2, 0.6, 0.85, 1.0], [-12.0, -35.0, 0.0, 15.0, 0.0])


def swing_pitch(s):
    """Cosine-smoothed interpolation through PITCH_KEYS (zero slope at each key)."""
    xs, ys = PITCH_KEYS
    s = min(max(s, 0.0), 1.0)
    for k in range(len(xs) - 1):
        if s <= xs[k + 1]:
            f = (s - xs[k]) / (xs[k + 1] - xs[k])
            return ys[k] + (ys[k + 1] - ys[k]) * (1 - np.cos(np.pi * f)) / 2
    return ys[-1]


def simulate(route, stride_m=1.4, seed=0, stride_s=None, mount=None, length_jitter=0.0,
             gyro_bias_dps=1.5, acc_bias_g=0.01, gyro_noise_dps=0.6, acc_noise_g=0.012):
    """Returns {"t", "imu" (N x 6: ax ay az in g, gx gy gz in deg/s), "truth": {...}}.

    truth.strides: one dict per stride: t_strike (heel lands), t_lift, heading_deg (after
    the stride), dz (m), length (m), stairs (0/1).  truth.maneuvers: [{t, action}].
    """
    rng = np.random.default_rng(seed)
    mount = random_rotation(rng) if mount is None else mount      # sensor -> foot
    gyro_bias = rng.normal(0, gyro_bias_dps, 3)
    acc_bias = rng.normal(0, acc_bias_g, 3)
    stride_s = stride_s or 0.95 + 0.12 * (stride_m - 1.2)

    # ---- plan: one entry per stride, plus still periods ----
    plan = [("still", 1.5)]
    maneuvers = []
    for leg in route:
        kind = leg[0]
        if kind == "walk":
            meters = leg[1] * (1 + rng.uniform(-length_jitter, length_jitter))
            n = max(1, int(round(meters / stride_m)))
            plan += [("stride", meters / n, 0.0, 0.0)] * n
        elif kind in ("left", "right"):
            angle = np.radians(rng.uniform(82, 98)) * (1 if kind == "left" else -1)
            plan += [("mark", kind), ("stride", stride_m * 0.7, angle / 2, 0.0),
                     ("stride", stride_m * 0.7, angle / 2, 0.0)]
        elif kind == "stairs":
            rises, direction = leg[1], leg[2]
            plan += [("mark", "stairs")]
            plan += [("stride", STAIR_STRIDE_M, 0.0, direction * 2 * RISE_M)] * max(1, rises // 2)
        elif kind == "still":
            plan += [("still", leg[1])]
        elif kind == "arrive":
            plan += [("mark", "arrive")]
    plan += [("still", 3.0)]

    # ---- foot trajectory, sampled at HZ ----
    pos, rots, times = [], [], []
    p = np.zeros(3)
    heading = 0.0
    t = 0.0
    strides = []
    pending_mark = None

    def emit(pp, rr):
        nonlocal t
        pos.append(pp.copy())
        rots.append(rr)
        times.append(t)
        t += DT

    for item in plan:
        if item[0] == "mark":
            pending_mark = item[1]
            continue
        if item[0] == "still":
            if pending_mark == "arrive":            # arriving = the moment you stop
                maneuvers.append({"t": round(t, 3), "action": "arrive"})
                pending_mark = None
            r = rot_z(heading)
            for _ in range(int(item[1] * HZ)):
                emit(p, r)
            continue
        _, length, dyaw, dz = item
        period = stride_s * rng.uniform(0.96, 1.04) * (1.25 if dz else 1.0)
        n = int(period * HZ)
        n_stance = int(0.42 * n)
        start_p, start_h = p.copy(), heading
        direction = np.array([np.cos(start_h + dyaw / 2), np.sin(start_h + dyaw / 2), 0.0])
        t_lift = None
        if pending_mark:
            maneuvers.append({"t": round(t + n_stance * DT, 3), "action": pending_mark})
            pending_mark = None
        for i in range(n):
            if i < n_stance:
                # stance: flat, still, slight heel-to-toe roll at the end
                s = i / n_stance
                pitch = -np.radians(12) * minjerk((s - 0.7) / 0.3) if s > 0.7 else 0.0
                emit(start_p, rot_z(start_h) @ rot_y(pitch))
                continue
            if t_lift is None:
                t_lift = t
            s = (i - n_stance + 1) / (n - n_stance)       # reaches 1 on the last sample
            m = minjerk(s)
            clearance = 0.08 * np.sin(np.pi * s) ** 2
            here = start_p + direction * length * m + np.array([0, 0, dz * minjerk(s * 1.15) + clearance])
            pitch = np.radians(swing_pitch(s))
            emit(here, rot_z(start_h + dyaw * m) @ rot_y(pitch))
        p = start_p + direction * length + np.array([0, 0, dz])
        heading = start_h + dyaw
        strides.append({"t_strike": round(t, 3), "t_lift": round(t_lift, 3),
                        "heading_deg": float(np.degrees(heading)), "dz": dz, "length": length,
                        "stairs": int(dz != 0)})

    pos = np.array(pos)
    times = np.array(times)

    # ---- IMU readings ----
    vel = np.gradient(pos, DT, axis=0)
    acc = np.gradient(vel, DT, axis=0)
    imu = np.zeros((len(times), 6))
    strike_idx = [int(round(s["t_strike"] * HZ)) for s in strides]
    ring = np.zeros(len(times))
    for k in strike_idx:                     # heel-strike ringing, mostly vertical
        seg = np.arange(0, 12)
        idx = k + seg
        idx = idx[idx < len(times)]
        ring[idx] += 2.0 * np.exp(-seg[: len(idx)] / 3.0) * np.cos(seg[: len(idx)] * 1.3)
    for k in range(len(times)):
        r_ws = rots[k] @ mount                   # sensor -> world
        f_world = acc[k] / G + np.array([0, 0, 1.0 + ring[k]])
        imu[k, :3] = r_ws.T @ f_world
        if k + 1 < len(times):
            r_next = rots[k + 1] @ mount
            imu[k, 3:] = np.degrees(log_so3(r_ws.T @ r_next) / DT)
        else:
            imu[k, 3:] = imu[k - 1, 3:]
    imu[:, :3] += acc_bias + rng.normal(0, acc_noise_g, (len(times), 3))
    imu[:, 3:] += gyro_bias + rng.normal(0, gyro_noise_dps, (len(times), 3))

    return {"t": times, "imu": imu, "stride_m": stride_m,
            "truth": {"strides": strides, "maneuvers": maneuvers, "pos": pos,
                      "final_heading_deg": float(np.degrees(heading)), "mount": mount}}


if __name__ == "__main__":
    route = [("walk", 20), ("left",), ("walk", 10), ("stairs", 12, 1), ("walk", 6), ("right",), ("walk", 8), ("arrive",)]
    sim = simulate(route, seed=3)
    tr = sim["truth"]
    print(f"{len(sim['t']) / HZ:.0f} s, {len(tr['strides'])} strides, final heading {tr['final_heading_deg']:.0f} deg, "
          f"end position {np.round(tr['pos'][-1], 2)}")
    print("|accel| at rest:", np.round(np.linalg.norm(sim["imu"][:50, :3], axis=1).mean(), 3), "g")
