"""MPC line following: the pure control rule, no browser needed.

The MPC tab shows `build`, `solve` and `MPC.decide` as its code card, so keep them short."""
from bangbang import clamp
import lqr
from track import Tracker

STEP = 0.05          # seconds between the planned steering moves


def build(horizon, q_error, q_rate, effort):
    # Predict with e'' = -b u. Every future error and rate is "what happens with no steering"
    # (set by the latest measured e and the estimated rate) plus a fixed effect per planned move u_0..u_N-1, so the cost
    # is  1/2 u'Hu + f'u  (plus a constant that cannot change which moves are best):
    #   H  N x N, fixed: how the moves interact in the cost (Qe and Qd on the predicted
    #      states, R on the diagonal for the steering itself); it does not depend on the car
    #   f  N numbers, f = f_e*e + f_d*rate: which way each move should push, given where
    #      the car is now
    # With no steering limit the best moves are where the slope H u + f is zero.
    b, n = lqr.PLANT_GAIN, int(horizon)
    effort = max(effort, 1e-6)
    h = [[0.0] * n for _ in range(n)]
    f_e, f_d = [0.0] * n, [0.0] * n
    for k in range(1, n + 1):                       # the state k steps ahead
        gain_e = [-b * STEP ** 2 * (k - 1 - j + 0.5) if j < k else 0.0 for j in range(n)]
        gain_d = [-b * STEP if j < k else 0.0 for j in range(n)]
        for i in range(k):
            f_e[i] += 2 * q_error * gain_e[i]
            f_d[i] += 2 * (q_error * gain_e[i] * k * STEP + q_rate * gain_d[i])
            for j in range(k):
                h[i][j] += 2 * (q_error * gain_e[i] * gain_e[j] + q_rate * gain_d[i] * gain_d[j])
    for i in range(n):
        h[i][i] += 2 * effort
    return h, f_e, f_d


def solve(h, f, u, limit):
    # minimize 1/2 u'Hu + f'u with |u| <= limit: sweep the moves one at a time, clipping each
    n = len(u)
    for _ in range(25):
        for i in range(n):
            slope = f[i] + sum(h[i][j] * u[j] for j in range(n))
            u[i] = max(-limit, min(limit, u[i] - slope / h[i][i]))
    return u


class MPC:
    def __init__(self):
        self.tracker = Tracker()
        self.key = None
        self.plan = []

    def decide(self, reflection, now, on_line, off_line, follow_right, base,
               q_error, q_rate, effort, horizon, max_steer):
        error, rate = self.tracker.update(reflection, now, on_line, off_line)
        horizon = max(1, min(30, int(horizon)))

        # (re)build the prediction model whenever a weight, the horizon or b changes
        key = (horizon, q_error, q_rate, effort, lqr.PLANT_GAIN)
        if key != self.key:
            self.key, self.model = key, build(*key[:4])
            self.plan = [0.0] * horizon
        h, f_e, f_d = self.model

        # plan the best steering moves for the next few steps, then use only the first
        f = [fe * error + fd * rate for fe, fd in zip(f_e, f_d)]
        self.plan = solve(h, f, self.plan, max_steer / 100)
        steer = 100 * self.plan[0]
        self.plan = self.plan[1:] + self.plan[-1:]      # warm start for the next reading

        if not follow_right:
            steer = -steer
        left = base + steer
        right = base - steer
        return clamp(left), clamp(right)
