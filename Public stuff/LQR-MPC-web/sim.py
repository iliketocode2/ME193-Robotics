"""
Line-follower simulator: LQR vs MPC, no robot required.

Pure numpy (no scipy). Run it to reproduce every number quoted in
docs/01-lqr.md, docs/02-mpc.md and docs/03-lqr-vs-mpc.md:

    python3 code/sim.py

(Web edition: the same file as the repository's code/sim.py, except it does not
read code/calibration.json. Edit it in the notes and press Run.)

The model is the standard lateral-tracking (Dubins) model for a
differential-drive robot following a line:

    e_dot     = v * sin(theta)  ~=  v * theta      lateral offset [mm]
    theta_dot = omega - kappa*v                    heading error  [rad]

with omega [rad/s] the commanded yaw rate and kappa [1/mm] the curvature
of the line the robot is trying to follow.
"""

import numpy as np

# --------------------------------------------------------------------------
# Build constants. On the web there is no calibration file, so these are the
# generic reference build -- the one quoted in the docs. (The version in the
# repository reads code/calibration.json, written by code/calibrate.py, when it
# exists.) V_REF is a choice of how fast to drive, not something you measure.
# --------------------------------------------------------------------------
_cal = {}

V_REF = 150.0                          # mm/s   nominal forward speed
TRACK_W = _cal.get("track_width_mm", 96.0)    # mm   wheel contact patch spacing
WHEEL_MAX = _cal.get("wheel_max_mms", 400.0)  # mm/s wheel speed at 100%
E_MAX = _cal.get("e_max_mm", 6.0)             # mm   sensor's linear region, half-width
DT = _cal.get("dt", 0.05)                     # s    control period (= notification delay)

# LQR weights. Only the ratios matter; see docs/01-lqr.md.
Q_E = 1.0            # cost per mm^2 of lateral error
Q_TH = 0.0           # cost per rad^2 of heading error
R_W = 25.0           # cost per (rad/s)^2 of yaw rate


# --------------------------------------------------------------------------
# LQR
# --------------------------------------------------------------------------
def lqr_gains_analytic(v, q_e=Q_E, q_th=Q_TH, r=R_W):
    """Closed-form solution of the continuous Riccati equation for this model.

    Returns (k_e, k_theta) such that omega = -k_e*e - k_theta*theta.
    Derived by hand in docs/01-lqr.md.
    """
    k_e = np.sqrt(q_e / r)
    k_th = np.sqrt(q_th / r + 2.0 * v * k_e)
    return k_e, k_th


def dlqr(A, B, Q, R, iters=100_000, tol=1e-12):
    """Discrete-time infinite-horizon LQR by Riccati iteration.

    This *is* the value-iteration backup of an MPC with horizon N, run until
    it stops changing -- which is the whole point of docs/03-lqr-vs-mpc.md.
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
    """Exact zero-order-hold discretization of the lateral-tracking model."""
    A = np.array([[1.0, v * dt], [0.0, 1.0]])
    B = np.array([[v * dt * dt / 2.0], [dt]])
    return A, B


def lqr_gains_discrete(v, dt=DT, q_e=Q_E, q_th=Q_TH, r=R_W):
    A, B = discrete_model(v, dt)
    K, _ = dlqr(A, B, np.diag([q_e, q_th]) * dt, np.array([[r * dt]]))
    return K[0, 0], K[0, 1]


# --------------------------------------------------------------------------
# Plant + track
# --------------------------------------------------------------------------
def step(e, theta, v, omega, kappa, dt=DT):
    """One control period of the plant (small-angle, forward Euler)."""
    e_next = e + dt * v * np.sin(theta)
    th_next = theta + dt * (omega - kappa * v)
    return e_next, th_next


def coerce(e, e_max=E_MAX):
    """What one light sensor can actually tell you: e, saturated.

    Outside +-e_max the sensor is fully on black or fully on white and reports
    only the *sign* of the error, not its size.
    """
    return float(np.clip(e, -e_max, e_max))


def wheel_speeds(v, omega, track_w=TRACK_W):
    """Differential-drive inverse kinematics, in mm/s."""
    return v - omega * track_w / 2.0, v + omega * track_w / 2.0


def to_percent(v_wheel, wheel_max=WHEEL_MAX):
    return float(np.clip(100.0 * v_wheel / wheel_max, -100.0, 100.0))


class Track:
    """Piecewise-constant curvature track, indexed by arclength s [mm].

    segments = [(length_mm, radius_mm or None), ...]; None means straight,
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


DEMO_TRACK = Track([(400.0, None), (125.7, 80.0), (400.0, None)])  # 90 deg, R=80


# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------
def run_lqr(track, v_cmd=V_REF, dt=DT, e_max=E_MAX, t_max=12.0):
    """Fixed-speed, fixed-gain LQR. Gains computed once, offline."""
    k_e, k_th = lqr_gains_discrete(v_cmd, dt)
    s = e = theta = 0.0
    log = []
    n = int(t_max / dt)
    for i in range(n):
        if s > track.length:
            break
        omega = -k_e * coerce(e, e_max) - k_th * theta
        log.append((i * dt, s, e, theta, v_cmd, omega))
        kappa = track.kappa(s)
        e, theta = step(e, theta, v_cmd, omega, kappa, dt)
        s += v_cmd * dt
    return np.array(log), (k_e, k_th)


def mpc_choose_speed(e, theta, s, track, candidates, horizon=15, dt=DT,
                     e_max=E_MAX, v_ref=V_REF, q_e=Q_E, q_th=Q_TH,
                     q_v=0.01, pen=1e4):
    """One MPC solve: pick the forward speed for the next control period.

    Decision variable: a single constant speed v over the horizon. For each
    candidate we roll the *previewed* curvature forward under the LQR steering
    law re-tuned for that speed, then score the predicted trajectory. The
    winner's first move is applied; next period we solve again.

    Enumeration is a legitimate way to solve an MPC when the decision is
    low-dimensional -- and it is the version a student can single-step in a
    debugger. docs/02-mpc.md shows the same problem as a QP.
    """
    best = (np.inf, candidates[0], None)
    table = []
    for v in candidates:
        k_e, k_th = lqr_gains_discrete(v, dt)
        ep, thp, sp = e, theta, s
        cost = 0.0
        peak = abs(ep)
        for _ in range(horizon):
            omega = -k_e * coerce(ep, e_max) - k_th * thp
            cost += (q_e * ep ** 2 + q_th * thp ** 2) * dt
            cost += pen * max(0.0, abs(ep) - e_max) ** 2 * dt
            ep, thp = step(ep, thp, v, omega, track.kappa(sp), dt)
            sp += v * dt
            peak = max(peak, abs(ep))
        cost += q_v * (v - v_ref) ** 2 * horizon * dt
        table.append((v, peak, cost))
        if cost < best[0]:
            best = (cost, v, table)
    return best[1], table


def run_mpc(track, dt=DT, e_max=E_MAX, t_max=12.0,
            candidates=(40., 60., 80., 100., 120., 150., 180., 210.),
            horizon=15):
    s = e = theta = 0.0
    log = []
    n = int(t_max / dt)
    for i in range(n):
        if s > track.length:
            break
        v, _ = mpc_choose_speed(e, theta, s, track, candidates, horizon, dt, e_max)
        k_e, k_th = lqr_gains_discrete(v, dt)
        omega = -k_e * coerce(e, e_max) - k_th * theta
        log.append((i * dt, s, e, theta, v, omega))
        e, theta = step(e, theta, v, omega, track.kappa(s), dt)
        s += v * dt
    return np.array(log)


def summarize(name, log, e_max=E_MAX):
    t, s, e, theta, v, omega = log.T
    lost = np.abs(e) > e_max
    return (f"{name:26s} peak |e| = {np.abs(e).max():6.2f} mm   "
            f"blind {100 * lost.mean():5.1f}% of the run   "
            f"lap {t[-1] + (t[1] - t[0]):5.2f} s   "
            f"mean v {v.mean():6.1f} mm/s")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=4, suppress=True)
    bar = "=" * 74

    print("Using the generic reference-build numbers quoted in the docs.\n")

    print(bar)
    print("1. LQR GAINS  (Q = diag(%g, %g), R = %g, v = %g mm/s)"
          % (Q_E, Q_TH, R_W, V_REF))
    print(bar)
    k_e, k_th = lqr_gains_analytic(V_REF)
    wn = np.sqrt(V_REF * k_e)
    zeta = k_th / (2 * wn)
    print(f"  analytic (continuous):  k_e = {k_e:.4f} rad/s/mm   "
          f"k_theta = {k_th:.4f} 1/s")
    print(f"  natural frequency wn  = {wn:.3f} rad/s")
    print(f"  damping ratio  zeta   = {zeta:.4f}   (= 1/sqrt(2) whenever q_th = 0)")
    print(f"  settle 4/(zeta*wn)    = {4 / (zeta * wn):.2f} s "
          f"= {4 / (zeta * wn) * V_REF:.0f} mm of travel")
    print()
    for dt in (0.02, 0.05, 0.10, 0.20):
        kd_e, kd_th = lqr_gains_discrete(V_REF, dt)
        A, B = discrete_model(V_REF, dt)
        eig = np.linalg.eigvals(A - B @ np.array([[kd_e, kd_th]]))
        print(f"  discrete dt={dt:4.2f}s:  k_e = {kd_e:.4f}  k_theta = {kd_th:.4f}"
              f"   |closed-loop poles| = {np.abs(eig)[0]:.3f}")

    print()
    print(bar)
    r_black = _cal.get("reflect_black", 12.0)
    r_white = _cal.get("reflect_white", 85.0)
    span = r_white - r_black
    print(f"2. THE THREE SENSOR STATES  (black={r_black:g}, white={r_white:g}, "
          f"theta = 0)")
    print(bar)
    fracs = [("fully ON the line", 0.0), ("edge, drifted onto line", 0.35),
             ("edge, on target", 0.5), ("edge, drifted off line", 0.65),
             ("fully OFF the line", 1.0)]
    rows = [(name, r_black + f * span) for name, f in fracs]
    print(f"  {'sensor state':26s} {'refl':>5s} {'s':>6s} {'e mm':>7s} "
          f"{'w rad/s':>9s} {'vL %':>7s} {'vR %':>7s}")
    for name, refl in rows:
        s_norm = (refl - r_black) / (r_white - r_black)
        e = float(np.clip(2 * E_MAX * (s_norm - 0.5), -E_MAX, E_MAX))
        omega = -k_e * e
        v_l, v_r = wheel_speeds(V_REF, omega)
        print(f"  {name:26s} {refl:5.1f} {s_norm:6.3f} {e:+7.2f} {omega:+9.3f} "
              f"{to_percent(v_l):7.1f} {to_percent(v_r):7.1f}")

    print()
    print(bar)
    print("3. WHY LQR BREAKS ON SHARP TURNS:  e_ss = -kappa*v/k_e")
    print(bar)
    print(f"  {'turn radius':>12s} {'kappa 1/mm':>11s} {'e_ss @150':>10s} "
          f"{'verdict':>9s} {f'max v for |e|<{E_MAX:g}mm':>19s}")
    for radius in (np.inf, 300.0, 150.0, 80.0, 40.0):
        kappa = 0.0 if np.isinf(radius) else 1.0 / radius
        e_ss = -kappa * V_REF / k_e
        v_lim = np.inf if kappa == 0 else E_MAX * k_e / kappa
        ok = "ok" if abs(e_ss) <= E_MAX else "BLIND"
        rad = "straight" if np.isinf(radius) else f"{radius:.0f} mm"
        lim = "any" if np.isinf(v_lim) else f"{v_lim:.0f} mm/s"
        print(f"  {rad:>12s} {kappa:11.5f} {e_ss:+9.2f} {ok:>10s} {lim:>19s}")

    print()
    print(bar)
    print("4. ONE MPC SOLVE, entering the R=80mm turn (s=390mm, e=0, theta=0)")
    print(bar)
    _, table = mpc_choose_speed(0.0, 0.0, 390.0, DEMO_TRACK,
                                (40., 60., 80., 100., 120., 150., 180., 210.))
    print(f"  {'candidate v':>12s} {'peak |e| over horizon':>23s} "
          f"{'cost J':>10s}   choice")
    best_v = min(table, key=lambda row: row[2])[0]
    for v, peak, cost in table:
        flag = "  <-- chosen" if v == best_v else ("  (violates window)"
                                                  if peak > E_MAX else "")
        print(f"  {v:9.0f} mm/s {peak:19.2f} mm {cost:10.2f}{flag}")

    print()
    print(bar)
    print("5. HEAD TO HEAD on 400mm straight -> 90deg R=80mm turn -> 400mm straight")
    print(bar)
    log_lqr, _ = run_lqr(DEMO_TRACK)
    log_mpc = run_mpc(DEMO_TRACK)
    print("  " + summarize("LQR, fixed v = 150", log_lqr))
    print("  " + summarize("MPC, v scheduled", log_mpc))
    print(f"  MPC speed profile: min {log_mpc[:, 4].min():.0f} mm/s, "
          f"max {log_mpc[:, 4].max():.0f} mm/s")

    print()
    print(bar)
    print("6. MPC WITH NO CONSTRAINTS AND A LONG HORIZON *IS* LQR")
    print(bar)
    A, B = discrete_model(V_REF)
    Q, R = np.diag([Q_E, Q_TH]) * DT, np.array([[R_W * DT]])
    K_inf, _ = dlqr(A, B, Q, R)
    x = np.array([[4.0], [0.1]])
    print(f"  test state: e = 4.0 mm, theta = 0.10 rad")
    print(f"  {'horizon N':>10s} {'first move omega_0 [rad/s]':>28s} "
          f"{'gap to LQR':>12s}")
    u_lqr = float((-K_inf @ x)[0, 0])
    for N in (1, 2, 5, 10, 20, 50, 200):
        P = Q.copy()
        for _ in range(N - 1):
            S = R + B.T @ P @ B
            K = np.linalg.solve(S, B.T @ P @ A)
            P = A.T @ P @ (A - B @ K) + Q
        S = R + B.T @ P @ B
        K_N = np.linalg.solve(S, B.T @ P @ A)
        u_N = float((-K_N @ x)[0, 0])
        print(f"  {N:10d} {u_N:28.6f} {abs(u_N - u_lqr):12.2e}")
    print(f"  {'infinite':>10s} {u_lqr:28.6f} {0.0:12.2e}   <-- the LQR gain K")


if __name__ == "__main__":
    main()
