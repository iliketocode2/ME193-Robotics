"""Turns raw IMU samples into steps. PARTNER A REPLACES THIS FILE.

This is a simple baseline so the rest of the navigator runs end to end. Keep
the interface below and everything else (recording, review, training, live
navigation) works unchanged with your better version:

    s = Sensing()
    s.reset()                         # at the start of every walk
    steps = s.feed(t, rows)           # call for every batch from the sketch
        t     board time (time.time()) of the LAST row in the batch
        rows  list of [ax, ay, az, gx, gy, gz] in g and deg/s, 100 Hz
        returns a list of Step (usually empty, sometimes one)

    Step.t            time of the step (board seconds)
    Step.n            1, 2, 3... since reset()
    Step.heading_deg  heading since reset(), unwrapped, + = turned left (counterclockwise from above)
    Step.stairs       1 on stairs, 0 not, -1 if this sensing can't tell

Change VERSION whenever the output changes, so walks recorded with older
sensing can be found and reprocessed (review page -> "Reprocess walks").

Baseline method:
  steps    peak of the smoothed acceleration magnitude above a threshold, with a
           minimum gap; one count per footfall of the shoe wearing the board
  heading  the rotation rate around the vertical, where "vertical" is the slowly
           averaged accelerometer direction (gravity), so it works however the
           board is mounted; the gyro's zero offset is re-measured whenever the
           foot is still, then integrated
"""

from dataclasses import dataclass

import numpy as np

SAMPLE_HZ = 100
DT = 1.0 / SAMPLE_HZ


@dataclass
class Step:
    t: float
    n: int
    heading_deg: float
    stairs: int = -1


class Sensing:
    VERSION = "baseline-1"

    STEP_HIGH_G = 0.45      # smoothed |a| - 1 g must pass this to start a step
    STEP_LOW_G = 0.15       # ...and fall below this before the next one can start
    MIN_GAP_S = 0.45        # never two steps closer than this
    GRAVITY_ALPHA = 0.02    # gravity direction: ~0.5 s average
    MAG_ALPHA = 0.25        # light smoothing of |a| before peak picking
    STILL_WINDOW = 20       # samples (0.2 s) checked for stillness: shorter than a stance phase
    STILL_ACC_STD_G = 0.03
    STILL_GYRO_DPS = 6.0
    BIAS_ALPHA = 0.2

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.heading = 0.0
        self.gravity = None
        self.mag = 1.0
        self.armed = True
        self.last_step_t = -1e9
        self.bias = np.zeros(3)
        self.recent = []            # last STILL_WINDOW rows, for the bias check
        self.since_still_check = 0

    def feed(self, t, rows):
        steps = []
        count = len(rows)
        for i, row in enumerate(rows):
            ts = t - (count - 1 - i) * DT
            step = self._sample(ts, np.asarray(row, dtype=float))
            if step is not None:
                steps.append(step)
        return steps

    def _sample(self, ts, row):
        acc, gyro = row[:3], row[3:]

        # gravity direction (points up: a resting accelerometer reads +1 g upward)
        if self.gravity is None:
            self.gravity = acc.copy()
        else:
            self.gravity += self.GRAVITY_ALPHA * (acc - self.gravity)
        norm = np.linalg.norm(self.gravity)
        up = self.gravity / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])

        # gyro zero offset, re-measured while the foot is planted
        self.recent.append(row)
        if len(self.recent) > self.STILL_WINDOW:
            self.recent.pop(0)
        self.since_still_check += 1
        if self.since_still_check >= self.STILL_WINDOW // 2 and len(self.recent) == self.STILL_WINDOW:
            self.since_still_check = 0
            block = np.array(self.recent)
            mags = np.linalg.norm(block[:, :3], axis=1)
            if mags.std() < self.STILL_ACC_STD_G and np.abs(block[:, 3:]).max() < self.STILL_GYRO_DPS:
                self.bias += self.BIAS_ALPHA * (block[:, 3:].mean(axis=0) - self.bias)

        # heading: rotation about the vertical
        self.heading += float(np.dot(gyro - self.bias, up)) * DT

        # steps: hysteresis peak picking on smoothed |a|
        self.mag += self.MAG_ALPHA * (np.linalg.norm(acc) - self.mag)
        excess = self.mag - 1.0
        if self.armed and excess > self.STEP_HIGH_G and ts - self.last_step_t >= self.MIN_GAP_S:
            self.armed = False
            self.last_step_t = ts
            self.n += 1
            return Step(t=ts, n=self.n, heading_deg=self.heading, stairs=-1)
        if not self.armed and excess < self.STEP_LOW_G:
            self.armed = True
        return None
