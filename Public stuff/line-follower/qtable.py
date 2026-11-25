"""Q-table line following: the pure learning rule, no browser needed.

The Q-table tab shows `state_of` and `QLearner.decide` as its code card, so keep them short."""
from random import choice, random, randrange

from bangbang import clamp
from track import Tracker

ERROR_EDGES = (-0.6, -0.2, 0.2, 0.6)     # splits the error into 5 bins
RATE_EDGES = (-2.0, 2.0)                 # splits its rate into 3 bins: falling, steady, rising
ACTIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)    # steering, as a fraction of the Turn setting
DECIDE_EVERY = 0.1                       # seconds between choices, so each one has time to act


def state_of(error, rate):
    error_bin = sum(error > edge for edge in ERROR_EDGES)
    rate_bin = sum(rate > edge for edge in RATE_EDGES)
    return error_bin * 3 + rate_bin


class QLearner:
    def __init__(self):
        self.q = [[0.0] * len(ACTIONS) for _ in range(5 * 3)]
        self.steps = 0
        self.restart()

    def restart(self):
        self.tracker = Tracker()
        self.state = self.action = None
        self.next_time = 0.0

    def decide(self, reflection, now, on_line, off_line, follow_right, base,
               turn, alpha, gamma, epsilon, learning):
        error, rate = self.tracker.update(reflection, now, on_line, off_line)

        if now >= self.next_time:
            state = state_of(error, rate)

            # learn from the move just made: reward is better the closer to the target
            if learning and self.state is not None:
                reward = -abs(error)
                target = reward + gamma * max(self.q[state])
                self.q[self.state][self.action] += alpha * (target - self.q[self.state][self.action])

            # choose the next move: usually the best known, sometimes a random one to explore
            if learning and random() < epsilon:
                self.action = randrange(len(ACTIONS))
            else:
                best = max(self.q[state])
                self.action = choice([a for a, value in enumerate(self.q[state]) if value == best])
            self.state = state
            self.next_time = now + DECIDE_EVERY
            self.steps += 1

        steer = ACTIONS[self.action] * turn
        if not follow_right:
            steer = -steer
        left = base + steer
        right = base - steer
        return clamp(left), clamp(right)
