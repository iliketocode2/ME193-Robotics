"""PD line following: the pure control rule, no browser needed.

The PD tab shows `PD.decide` as its code card, so keep it short and readable."""
from bangbang import clamp


class PD:
    def __init__(self):
        self.last_error = None
        self.last_time = None

    def decide(self, reflection, now, on_line, off_line, follow_right, base, kp, kd):
        # error: +1 well onto the line, -1 well off it, 0 right on the target
        goal = (on_line + off_line) / 2
        half_range = (off_line - on_line) / 2
        error = (goal - reflection) / half_range

        # derivative: how fast the error is changing (per second)
        if self.last_time is None:
            slope = 0
        else:
            slope = (error - self.last_error) / max(now - self.last_time, 0.01)
        self.last_error, self.last_time = error, now

        # steer in proportion to the error, and against its rate of change
        steer = kp * error + kd * slope
        if not follow_right:
            steer = -steer

        left = base + steer
        right = base - steer
        return clamp(left), clamp(right)
