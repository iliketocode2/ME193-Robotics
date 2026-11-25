"""LQR line following: the pure control rule, no browser needed.

The LQR tab shows `gains` and `LQR.decide` as its code card, so keep them short and readable."""
from math import sqrt

from bangbang import clamp

# The model is  e'' = -PLANT_GAIN * u : full steering (u = 1) changes the error's rate of
# change by this much per second. 1000 is what we measured on the test car; the Calibrate tab
# can measure it on yours and
# overwrite this value while the page runs (see steering.py).
DEFAULT_PLANT_GAIN = 1000
PLANT_GAIN = DEFAULT_PLANT_GAIN


def gains(q_error, q_rate, effort):
    # exact solution of the Riccati equation for this two-state model
    effort = max(effort, 1e-6)
    k_error = sqrt(q_error / effort)
    k_rate = sqrt(q_rate / effort + 2 * k_error / PLANT_GAIN)
    return k_error, k_rate


class LQR:
    def __init__(self):
        self.last_error = None
        self.last_time = None
        self.rate = 0.0

    def decide(self, reflection, now, on_line, off_line, follow_right, base,
               q_error, q_rate, effort):
        # state 1: error, +1 well onto the line, -1 well off it
        goal = (on_line + off_line) / 2
        half_range = (off_line - on_line) / 2
        error = (goal - reflection) / half_range

        # state 2: the error's rate of change, estimated by smoothed differencing
        if self.last_time is not None:
            dt = max(now - self.last_time, 0.01)
            raw_rate = (error - self.last_error) / dt
            self.rate += dt / (dt + 0.05) * (raw_rate - self.rate)
        self.last_error, self.last_time = error, now

        # optimal feedback on both states, as a percentage of full steering
        k_error, k_rate = gains(q_error, q_rate, effort)
        steer = 100 * (k_error * error + k_rate * self.rate)
        if not follow_right:
            steer = -steer

        left = base + steer
        right = base - steer
        return clamp(left), clamp(right)
