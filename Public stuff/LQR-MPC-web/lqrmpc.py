"""
Line-follower simulator: LQR vs MPC, no robot required.

Pure numpy. This is the same model as the course's sim.py, with the
hardware-calibration file removed and every weight passed in as an argument, so
the web pages can move the sliders and re-run it.

The model is the standard lateral-tracking (Dubins) model for a
differential-drive robot following a line:

    e_dot     = v * sin(theta)  ~=  v * theta      lateral offset [mm]
    theta_dot = omega - kappa*v                    heading error  [rad]

with omega [rad/s] the commanded yaw rate and kappa [1/mm] the curvature of
the line the robot is trying to follow. e > 0 means the robot is left of the line.
"""

from functools import lru_cache

import numpy as np

# Generic reference build (the numbers quoted in the notes)
V_REF = 150.0     # mm/s   nominal forward speed
TRACK_W = 96.0    # mm     wheel contact patch spacing
WHEEL_MAX = 400.0 # mm/s   wheel speed at 100%
E_MAX = 6.0       # mm     the sensor's linear region, half-width
DT = 0.05         # s      control period

Q_E, Q_TH, R_W = 1.0, 0.0, 25.0   # LQR weights; only the ratios matter

CANDIDATES = (40., 60., 80., 100., 120., 150., 180., 210.)
MPC_CANDIDATES = (40., 60., 80., 100., 120., 150., 180., 210., 240., 280.)


# --------------------------------------------------------------------------
# LQR
# --------------------------------------------------------------------------
def lqr_gains_analytic(v, q_e=Q_E, q_th=Q_TH, r=R_W):
    """Closed-form solution of the continuous Riccati equation.

    Returns (k_e, k_theta) such that omega = -k_e*e - k_theta*theta.
    """
    k_e = np.sqrt(q_e / r)
    k_th = np.sqrt(q_th / r + 2.0 * v * k_e)
    return k_e, k_th


def dlqr(A, B, Q, R, iters=100_000, tol=1e-12):
    """Discrete infinite-horizon LQR by Riccati iteration.

    This *is* the value-iteration backup of an MPC with horizon N, run until
    it stops changing.
    """
    P = np.array(Q, dtype=float)
    for _ in range(iters):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P_next = A.T @ P @ (A - B @ K) + Q
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)
    return K, P


def discrete_model(v, dt=DT):
    """Zero-order-hold discretization of the lateral-tracking model."""
    A = np.array([[1.0, v * dt], [0.0, 1.0]])
    B = np.array([[v * dt * dt / 2.0], [dt]])
    return A, B


@lru_cache(maxsize=4096)
def lqr_gains_discrete(v, dt=DT, q_e=Q_E, q_th=Q_TH, r=R_W):
    """Cached: the MPC asks for the same eight gain pairs every control step."""
    A, B = discrete_model(v, dt)
    K, _ = dlqr(A, B, np.diag([q_e, q_th]) * dt, np.array([[r * dt]]))
    return float(K[0, 0]), float(K[0, 1])


# --------------------------------------------------------------------------
# Plant, sensor, track
# --------------------------------------------------------------------------
def step(e, theta, v, omega, kappa, dt=DT):
    """One control period of the plant (forward Euler)."""
    return e + dt * v * np.sin(theta), theta + dt * (omega - kappa * v)


def coerce(e, e_max=E_MAX):
    """What one light sensor can tell you: e, saturated at +-e_max."""
    return float(np.clip(e, -e_max, e_max))


def wheel_speeds(v, omega, track_w=TRACK_W):
    """Differential-drive inverse kinematics, mm/s."""
    return v - omega * track_w / 2.0, v + omega * track_w / 2.0


def to_percent(v_wheel, wheel_max=WHEEL_MAX):
    return float(np.clip(100.0 * v_wheel / wheel_max, -100.0, 100.0))


class Track:
    """Piecewise-constant-curvature track indexed by arclength s [mm].

    segments = [(length_mm, radius_mm or None), ...]; None is straight,
    positive radius turns left, negative turns right.
    """

    def __init__(self, segments):
        self.segments = segments
        self.length = sum(L for L, _ in segments)

    def kappa(self, s):
        if s < 0:
            return 0.0
        acc = 0.0
        for L, radius in self.segments:
            if s < acc + L:
                return 0.0 if radius is None else 1.0 / radius
            acc += L
        return 0.0

    def geometry(self, ds=2.0):
        """Centerline points (x, y, heading) sampled every ds mm."""
        n = int(self.length / ds) + 1
        s = np.arange(n) * ds
        kap = np.array([self.kappa(x) for x in s])
        psi = np.concatenate([[0.0], np.cumsum(kap[:-1] * ds)])
        x = np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1]) * ds)])
        y = np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1]) * ds)])
        return s, x, y, psi

    def robot_xy(self, log):
        """World path of the robot: centerline point + e along the left normal."""
        s_line, x, y, psi = self.geometry()
        s, e = log[:, 1], log[:, 2]
        xl, yl, pl = (np.interp(s, s_line, a) for a in (x, y, psi))
        return xl - e * np.sin(pl), yl + e * np.cos(pl)


def arc(angle_deg, radius):
    """A turn segment: left if angle_deg > 0, right if < 0, sized by angle."""
    return (abs(np.radians(angle_deg)) * radius, radius if angle_deg > 0 else -radius)


TRACKS = {
    "90° turn, R = 80 mm (the notes' example)":
        Track([(400.0, None), arc(90, 80.0), (400.0, None)]),
    "Gentle bend, R = 300 mm":
        Track([(400.0, None), arc(90, 300.0), (400.0, None)]),
    "Hairpin, R = 40 mm":
        Track([(300.0, None), arc(180, 40.0), (300.0, None)]),
    "S-curve (left, then right), R = 100 mm":
        Track([(300.0, None), arc(90, 100.0), arc(-90, 100.0), (300.0, None)]),
    "Chicane (three quick turns), R = 70 mm":
        Track([(250.0, None), arc(60, 70.0), arc(-120, 70.0), arc(60, 70.0), (250.0, None)]),
}
DEMO_TRACK = TRACKS["90° turn, R = 80 mm (the notes' example)"]


# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------
def run_fixed_gain(track, v_cmd, k_e, k_th, dt=DT, e_max=E_MAX, t_max=14.0):
    """Fixed speed, fixed gains: omega = -k_e*coerce(e) - k_th*theta. P/PD and LQR both use this."""
    s = e = theta = 0.0
    log = []
    for i in range(int(t_max / dt)):
        if s > track.length:
            break
        omega = -k_e * coerce(e, e_max) - k_th * theta
        log.append((i * dt, s, e, theta, v_cmd, omega))
        e, theta = step(e, theta, v_cmd, omega, track.kappa(s), dt)
        s += v_cmd * dt
    return np.array(log)


def run_lqr(track, v_cmd=V_REF, dt=DT, e_max=E_MAX, t_max=14.0,
            q_e=Q_E, q_th=Q_TH, r=R_W):
    """Fixed-speed, fixed-gain LQR. Gains computed once, offline."""
    k_e, k_th = lqr_gains_discrete(float(v_cmd), dt, q_e, q_th, r)
    return run_fixed_gain(track, v_cmd, k_e, k_th, dt, e_max, t_max), (k_e, k_th)


def mpc_choose_speed(e, theta, s, track, candidates=CANDIDATES, horizon=15,
                     dt=DT, e_max=E_MAX, v_ref=V_REF, q_e=Q_E, q_th=Q_TH,
                     r=R_W, q_v=0.01, pen=1e4):
    """One MPC solve: pick the forward speed for the next control period.

    For each candidate speed, roll the previewed curvature forward under the
    LQR steering law re-tuned for that speed, then score the predicted
    trajectory. Returns (best v, table of (v, peak |e|, cost)).
    """
    best_cost, best_v, table = np.inf, candidates[0], []
    for v in candidates:
        k_e, k_th = lqr_gains_discrete(float(v), dt, q_e, q_th, r)
        ep, thp, sp = e, theta, s
        cost, peak = 0.0, abs(ep)
        for _ in range(horizon):
            omega = -k_e * coerce(ep, e_max) - k_th * thp
            cost += (q_e * ep ** 2 + q_th * thp ** 2) * dt
            cost += pen * max(0.0, abs(ep) - e_max) ** 2 * dt
            ep, thp = step(ep, thp, v, omega, track.kappa(sp), dt)
            sp += v * dt
            peak = max(peak, abs(ep))
        cost += q_v * (v - v_ref) ** 2 * horizon * dt
        table.append((v, peak, cost))
        if cost < best_cost:
            best_cost, best_v = cost, v
    return best_v, table


def run_mpc(track, dt=DT, e_max=E_MAX, t_max=14.0, candidates=CANDIDATES,
            horizon=15, v_ref=V_REF, q_e=Q_E, q_th=Q_TH, r=R_W, q_v=0.01):
    s = e = theta = 0.0
    log = []
    for i in range(int(t_max / dt)):
        if s > track.length:
            break
        v, _ = mpc_choose_speed(e, theta, s, track, candidates, horizon, dt,
                                e_max, v_ref, q_e, q_th, r, q_v)
        k_e, k_th = lqr_gains_discrete(float(v), dt, q_e, q_th, r)
        omega = -k_e * coerce(e, e_max) - k_th * theta
        log.append((i * dt, s, e, theta, v, omega))
        e, theta = step(e, theta, v, omega, track.kappa(s), dt)
        s += v * dt
    return np.array(log)


def stats(log, e_max=E_MAX):
    """(peak |e|, % of the run blind, lap time, mean speed). Blind = sensor pinned at the window edge."""
    t, s, e, theta, v, omega = log.T
    dt = t[1] - t[0] if len(t) > 1 else DT
    return (float(np.abs(e).max()), float(100 * (np.abs(e) >= e_max * 0.999).mean()),
            float(t[-1] + dt), float(v.mean()))


def turn_track(radius, lead_in=400.0, straight_out=400.0):
    """Straight, then a 90 degree turn of the given radius, then straight."""
    return Track([(lead_in, None), (radius * np.pi / 2, radius), (straight_out, None)])


def turn_times(track, v):
    """(time the turn starts, time it ends) at constant speed v."""
    lead = track.segments[0][0]
    return lead / v, (lead + track.segments[1][0]) / v


def horizon_gain(N, x, v=V_REF, dt=DT, q_e=Q_E, q_th=Q_TH, r=R_W):
    """First move of an unconstrained N-step MPC from state x = (e, theta)."""
    A, B = discrete_model(v, dt)
    Q, R = np.diag([q_e, q_th]) * dt, np.array([[r * dt]])
    P = Q.copy()
    for _ in range(N - 1):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P = A.T @ P @ (A - B @ K) + Q
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)
    return float((-K @ np.array([[x[0]], [x[1]]]))[0, 0])
