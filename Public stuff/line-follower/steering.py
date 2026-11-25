"""Estimate the car's steering authority b from a wobble, no browser needed.

Drive with proportional steering only (u = Kp*e/100): the car then wobbles across the edge.
For the model e'' = -b u that wobble has angular frequency w = sqrt(b * Kp/100), so
b = w**2 / (Kp/100), with w = 2*pi / period."""
from math import pi
from statistics import median


class PeriodFinder:
    """Watches the error and times each upward crossing of zero (with a dead band, so sensor
    noise near zero doesn't count as a crossing)."""

    def __init__(self, band=0.1):
        self.band = band
        self.was_low = False
        self.rises = []              # times of the upward crossings
        self.cycle_peak = 0.0
        self.peaks = []              # the largest |error| in each finished cycle

    def update(self, now, error):
        self.cycle_peak = max(self.cycle_peak, abs(error))
        if error < -self.band:
            self.was_low = True
        elif error > self.band and self.was_low:
            self.was_low = False
            self.rises.append(now)
            self.peaks.append(self.cycle_peak)
            self.cycle_peak = 0.0


def estimate(finder, kp):
    """(period, b, cycles, unevenness, amplitude) from the last few wobbles, or None until
    there are three full cycles to go on."""
    rises = finder.rises
    if len(rises) < 4 or kp <= 0:
        return None
    periods = [later - earlier for earlier, later in zip(rises, rises[1:])][-6:]
    period = median(periods)
    unevenness = (max(periods) - min(periods)) / period
    amplitude = median(finder.peaks[-4:])
    b = (2 * pi / period) ** 2 / (kp / 100)
    return period, b, len(rises) - 1, unevenness, amplitude
