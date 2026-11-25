"""Neural-net line following: the pure network and training, no browser needed.

The Neural net tab shows `Net.forward`, `Policy.decide` and `train_steps` as its code card."""
import json
from math import tanh
from random import gauss, sample, shuffle

from bangbang import clamp
from track import Tracker

RATE_SCALE = 10.0       # the rate is divided by this so both inputs are about the same size


def teacher(error, rate, kp, kd):
    """What the PD controller would steer, in percent: the example the network learns from."""
    return kp * error + kd * rate


class Net:
    """Two inputs (error, rate), one hidden layer of tanh units, one output (steering)."""

    def __init__(self, hidden=8):
        self.w1 = [[gauss(0, 0.5), gauss(0, 0.5)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [gauss(0, 0.5) for _ in range(hidden)]
        self.b2 = 0.0

    def forward(self, error, rate):
        x = (error, max(-1.5, min(1.5, rate / RATE_SCALE)))
        hidden = [tanh(w[0] * x[0] + w[1] * x[1] + b) for w, b in zip(self.w1, self.b1)]
        return sum(w * h for w, h in zip(self.w2, hidden)) + self.b2, x, hidden

    def to_json(self):
        return json.dumps({"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2})

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        net = cls(len(data["w2"]))
        net.w1, net.b1, net.w2, net.b2 = data["w1"], data["b1"], data["w2"], data["b2"]
        return net


def train_steps(net, data, epochs, rate, batch=600):
    """Gradient descent on (error, rate, steering) examples; yields the mean loss after each epoch."""
    for _ in range(epochs):
        examples = sample(data, min(batch, len(data)))
        total = 0.0
        for error, speed, target in examples:
            out, x, hidden = net.forward(error, speed)
            miss = out - target                          # the network's mistake
            total += 0.5 * miss * miss
            for j, h in enumerate(hidden):               # push every weight against its share of it
                back = miss * net.w2[j] * (1 - h * h)
                net.w2[j] -= rate * miss * h
                net.w1[j][0] -= rate * back * x[0]
                net.w1[j][1] -= rate * back * x[1]
                net.b1[j] -= rate * back
            net.b2 -= rate * miss
        yield total / len(examples)


class Policy:
    def __init__(self, net):
        self.net = net
        self.tracker = Tracker()

    def decide(self, reflection, now, on_line, off_line, follow_right, base):
        error, rate = self.tracker.update(reflection, now, on_line, off_line)
        steer = 100 * self.net.forward(error, rate)[0]
        if not follow_right:
            steer = -steer
        left = base + steer
        right = base - steer
        return clamp(left), clamp(right)
