"""Foot-mounted inertial tracking with zero-velocity updates.

A shoe is the one place on the body that stops completely on every stride: for
a few tenths of a second the foot is flat on the ground. This tracker uses
those moments (stance phases) to keep dead reckoning honest:

  every sample   rotate the orientation estimate by the gyro reading (3D, so the
                 foot's big pitch swings don't leak into heading); between stances,
                 integrate acceleration into velocity and position
  during stance  the foot is still, so: velocity is zero (that resets the drift
                 accumulated during the swing), the accelerometer reads pure gravity
                 (that levels the orientation), and the gyro should read zero (whatever
                 it reads is its offset, which is re-measured)
  stance begins  one stride is complete: report where the foot landed relative to
                 the last stance (length and height change) and the new heading

Without a magnetometer the absolute compass heading is unknowable, so heading
is measured from where the walk started (+ = turned left). The board can be
mounted at any angle: nothing here depends on which axis points where.

Also produces, for every sample, four mount-independent channels for the step
network: vertical and horizontal acceleration, vertical and horizontal rotation
rate (`channels`).
"""

from dataclasses import dataclass

import numpy as np

HZ = 100
DT = 1.0 / HZ
G = 9.81
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Stride:
    t: float            # when the foot came to rest (start of stance)
    n: int              # 1, 2, 3...
    length: float       # horizontal distance from the previous stance, m
    dz: float           # height change, m (+ = up)
    duration: float     # time since the previous stance began, s
    swing: float        # time the foot was moving, s
    heading_deg: float  # heading at this stance, unwrapped, + = left


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_from_rotvec(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, v[0] / 2, v[1] / 2, v[2] / 2])
    axis = v / angle
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_between(a, b):
    """Shortest rotation taking unit vector a onto unit vector b."""
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        perp = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0, 1.0, 0])
        return np.concatenate([[0.0], perp / np.linalg.norm(perp)])
    q = np.array([1.0 + d, *c])
    return q / np.linalg.norm(q)


class FootTracker:
    # stance = foot still: rotation and |acceleration - 1 g| both small for STANCE_MIN samples
    STANCE_GYRO_DPS = 40.0
    STANCE_ACC_G = 0.12
    STANCE_MIN = 6          # samples in a row to start a stance (0.06 s)
    SWING_MIN = 4           # moving samples in a row to end one
    MIN_STRIDE_S = 0.4      # a "stride" shorter than this is a stumble or shuffle; merged into the next
    SWING_PEAK_DPS = 150.0  # a real swing rotates the foot at least this fast; smaller wobbles (like the
                            # jolt when the OTHER foot lands) end a stance briefly but are not strides
    LEVEL_GAIN = 0.08       # how hard stance samples pull the orientation level (per sample)
    BIAS_GAIN = 0.02        # how fast the gyro offset estimate follows stance readings
    BIAS_STILL_DPS = 12.0   # ...only from samples this still (the foot rolls heel to toe in stance)

    def __init__(self):
        self.reset()

    def reset(self):
        self.q = None                   # orientation, sensor -> world (world z = up)
        self.bias = np.zeros(3)         # gyro offset, deg/s
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.ref = None                 # a sensor axis that was horizontal at the start (heading reference)
        self.heading = 0.0              # latched at each stance, unwrapped, degrees
        self._last_angle = None
        self.stance = False
        self.still_run = 0
        self.move_run = 0
        self.t = None
        self.stance_start_t = None
        self.swing_start_t = None
        self.stance_p = np.zeros(3)     # where the foot stood at the last stance
        self.n = 0
        self.swing_peak = 0.0           # fastest rotation since the last stance, deg/s
        self.channels = np.zeros(4)

    # ---------- per sample ----------

    def update(self, t, row):
        """row: ax, ay, az (g), gx, gy, gz (deg/s). Returns a Stride when one completes, else None."""
        a = np.asarray(row[:3], dtype=float)
        w_raw = np.asarray(row[3:], dtype=float)
        if self.q is None:
            self._init(a)
        w = np.radians(w_raw - self.bias)

        self.q = quat_mul(self.q, quat_from_rotvec(w * DT))
        self.q /= np.linalg.norm(self.q)
        r = quat_to_matrix(self.q)
        a_w = r @ a
        w_w = r @ w
        self.channels = np.array([a_w[2] - 1.0, np.hypot(a_w[0], a_w[1]),
                                  np.degrees(w_w[2]), np.degrees(np.hypot(w_w[0], w_w[1]))])

        still = (np.linalg.norm(w_raw - self.bias) < self.STANCE_GYRO_DPS
                 and abs(np.linalg.norm(a) - 1.0) < self.STANCE_ACC_G)
        self.still_run = self.still_run + 1 if still else 0
        self.move_run = 0 if still else self.move_run + 1

        stride = None
        if not self.stance and self.still_run >= self.STANCE_MIN:
            stride = self._begin_stance(t - (self.STANCE_MIN - 1) * DT)
        elif self.stance and self.move_run >= self.SWING_MIN:
            self.stance = False
            self.swing_start_t = t - (self.SWING_MIN - 1) * DT
            self.swing_peak = 0.0
        if not self.stance:
            self.swing_peak = max(self.swing_peak, float(np.linalg.norm(w_raw - self.bias)))

        if self.stance:
            self.v[:] = 0.0
            # level: rotate the world estimate so measured gravity points straight up
            g_dir = a_w / max(np.linalg.norm(a_w), 1e-9)
            err = np.cross(g_dir, UP)
            self.q = quat_mul(quat_from_rotvec(err * self.LEVEL_GAIN), self.q)
            self.q /= np.linalg.norm(self.q)
            if np.linalg.norm(w_raw - self.bias) < self.BIAS_STILL_DPS:
                self.bias += self.BIAS_GAIN * (w_raw - self.bias)
            self._latch_heading()
        else:
            lin = (a_w - UP) * G
            self.v += lin * DT
            self.p += self.v * DT
        self.t = t
        return stride

    def _init(self, a):
        g = a / max(np.linalg.norm(a), 1e-9)
        self.q = quat_between(g, UP)
        # heading reference: the sensor axis most perpendicular to gravity, made horizontal
        axis = np.eye(3)[int(np.argmin(np.abs(g)))]
        self.ref = axis - np.dot(axis, g) * g
        self.ref /= np.linalg.norm(self.ref)
        self._last_angle = None
        self._latch_heading()

    def _latch_heading(self):
        f = quat_to_matrix(self.q) @ self.ref
        if np.hypot(f[0], f[1]) < 0.3:      # reference axis nearly vertical right now: keep the last value
            return
        angle = np.degrees(np.arctan2(f[1], f[0]))
        if self._last_angle is not None:
            self.heading += (angle - self._last_angle + 180.0) % 360.0 - 180.0
        else:
            self.heading = 0.0
            self._origin = angle
        self._last_angle = angle

    def _begin_stance(self, t0):
        self.stance = True
        self._latch_heading()               # the stride's heading includes the swing just finished
        prev = self.stance_start_t
        swing_s = (t0 - self.swing_start_t) if self.swing_start_t is not None else 0.0
        # the velocity left over at landing is integration drift; assume it grew
        # linearly through the swing and take it back out of the position
        self.p -= self.v * swing_s / 2
        self.v[:] = 0.0
        if prev is None or swing_s <= 0:
            self.stance_start_t, self.stance_p = t0, self.p.copy()
            return None
        if t0 - prev < self.MIN_STRIDE_S or self.swing_peak < self.SWING_PEAK_DPS:
            return None                     # a wobble, not a stride: keep accumulating toward the next one
        d = self.p - self.stance_p
        self.stance_p = self.p.copy()
        self.stance_start_t = t0
        self.n += 1
        return Stride(t=t0, n=self.n, length=float(np.hypot(d[0], d[1])), dz=float(d[2]),
                      duration=float(t0 - prev), swing=float(swing_s), heading_deg=float(self.heading))


def run(t, imu):
    """Whole recording at once -> (strides, channels N x 4, stance flags N, heading N)."""
    ft = FootTracker()
    strides, chans, stance, heading = [], np.zeros((len(t), 4)), np.zeros(len(t), bool), np.zeros(len(t))
    for i in range(len(t)):
        s = ft.update(float(t[i]), imu[i])
        if s is not None:
            strides.append(s)
        chans[i] = ft.channels
        stance[i] = ft.stance
        heading[i] = ft.heading
    return strides, chans, stance, heading
