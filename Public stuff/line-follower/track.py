"""Shared by the MPC, Q-table and neural-net tabs: turn a reflection reading into the two numbers
those controllers work with, the error and its rate of change."""


class Tracker:
    def __init__(self):
        self.last_error = None
        self.last_time = None
        self.rate = 0.0

    def update(self, reflection, now, on_line, off_line):
        # error: +1 well onto the line, -1 well off it, 0 right on the target
        goal = (on_line + off_line) / 2
        half_range = (off_line - on_line) / 2
        error = (goal - reflection) / half_range

        # rate: how fast the error is changing (per second), smoothed to tame sensor noise
        if self.last_time is not None:
            dt = max(now - self.last_time, 0.01)
            raw_rate = (error - self.last_error) / dt
            self.rate += dt / (dt + 0.05) * (raw_rate - self.rate)
        self.last_error, self.last_time = error, now
        return error, self.rate
