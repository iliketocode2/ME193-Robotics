"""Raw IMU samples -> steps with heading and a stairs flag (Partner A's half).

Interface used by the rest of the navigator (nav_core.py, walks.py):

    s = Sensing()
    s.reset()                         # at the start of every walk
    steps = s.feed(t, rows)           # call for every batch from the sketch
        t     board time (time.time()) of the LAST row in the batch
        rows  list of [ax, ay, az, gx, gy, gz] in g and deg/s, 100 Hz
        returns a list of Step (usually empty, sometimes one)

    Step.t            time of the step (board seconds)
    Step.n            1, 2, 3... since reset()
    Step.heading_deg  heading since reset(), unwrapped, + = turned left (counterclockwise from above)
    Step.stairs       1 on stairs, 0 not, -1 if it can't tell

One "step" is one stride of the foot wearing the board (two footsteps).
Calibration turns that into meters, so only consistency matters.

How it works, every sample:
  1. foot_ins.FootTracker follows the foot's 3D orientation, finds the stance
     phases (foot planted and still), and at each one reports the stride just
     finished: heading, length, height change.
  2. The step network (stepnet_np.py, Project 8's CNN retrained on consistent
     labels) looks at the last second of the tracker's four mount-independent
     channels every 4 samples and says whether a stride just ended; Project 8's
     debouncer turns that into step events. If the network file is missing, or
     STEP_SOURCE = "tracker", the tracker's strides are the steps instead.
  3. The stairs classifier (stairs.py) labels each stride. If the tracker's
     strides are implausibly short (board not rigid on the foot), stairs = -1.

Change VERSION whenever the output changes, so walks recorded with older
sensing can be reprocessed (review page -> "Reprocess walks").
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

import foot_ins
import stairs as stairs_mod
import stepnet_np

STEP_SOURCE = "net"         # "net" (the trained step counter) or "tracker" (stance phases)
HOP = 4                     # network evaluated every 4 samples (25 times a second)
STAIRS_MATCH_S = 0.6        # a step takes the stairs label of a stride that ended this close to it
HOLD_S = 1.0                # a step with no stride that close yet waits up to this long for one (on the
                            # way downstairs the network can fire before the tracker confirms the stride)
MIN_PLAUSIBLE_STRIDE_M = 0.4    # median stride shorter than this = tracking unreliable, stairs unknown


@dataclass
class Step:
    t: float
    n: int
    heading_deg: float
    stairs: int = -1


class Sensing:
    def __init__(self, step_source=STEP_SOURCE):
        self.net = stepnet_np.load() if step_source == "net" else None
        self.stairs = stairs_mod.StairsClassifier()
        self.VERSION = ("stepnet" if self.net else "tracker") + "-ins-1"
        self.reset()

    def reset(self):
        self.tracker = foot_ins.FootTracker()
        window = self.net.window if self.net else 1
        self.chans = deque(maxlen=window)
        self.times = deque(maxlen=window)
        self.detector = stepnet_np.StepDetector(self.net.threshold, self.net.min_gap) if self.net else None
        self.samples = 0
        self.n = 0
        self.lengths = []
        self.last_stride = None         # (time, stairs flag)
        self.pending = []               # network step times waiting for their stride

    def feed(self, t, rows):
        out = []
        count = len(rows)
        for i, row in enumerate(rows):
            ts = t - (count - 1 - i) * foot_ins.DT
            stride = self.tracker.update(ts, row)
            if stride is not None:
                flag = self._stairs(stride)
                self.last_stride = (stride.t, flag)
                if self.net is None:
                    out.append(self._step(stride.t, flag))
                else:
                    out += self._release(ts, stride_arrived=True)
            if self.net is not None:
                self.chans.append(self.tracker.channels)
                self.times.append(ts)
                self.samples += 1
                if len(self.chans) == self.net.window and self.samples % HOP == 0:
                    p = self.net.prob(np.array(self.chans))
                    step_t = self.detector.update(self.times[self.net.center], p)
                    if step_t is not None:
                        self.pending.append(step_t)
                out += self._release(ts)
        return out

    def _release(self, now, stride_arrived=False):
        """Emit held network steps once their stride is known, or once they've waited HOLD_S."""
        out = []
        while self.pending:
            step_t = self.pending[0]
            near = self.last_stride and abs(self.last_stride[0] - step_t) <= STAIRS_MATCH_S
            if near:
                flag = self.last_stride[1]
            elif now - step_t >= HOLD_S:
                flag = -1 if self.last_stride and self.last_stride[1] == -1 else 0
            else:
                break
            self.pending.pop(0)
            out.append(self._step(step_t, flag))
        return out

    def _stairs(self, stride):
        self.lengths.append(stride.length)
        if len(self.lengths) >= 4 and np.median(self.lengths[-20:]) < MIN_PLAUSIBLE_STRIDE_M:
            return -1
        return self.stairs.classify(stride)

    def _step(self, t, flag):
        self.n += 1
        return Step(t=float(t), n=self.n, heading_deg=float(self.tracker.heading), stairs=int(flag))
