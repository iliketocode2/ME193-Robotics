"""
whistle_policy.py -- pure signal-processing + decision logic for whistle-controlled
driving.

This module has no hardware and no I/O: it takes raw audio blocks in and
returns Command objects out, so the trickiest logic in the project (pitch
detection, noise masking, gesture recognition) can be unit-tested with
synthetic signals -- see test_whistle_policy.py -- without a microphone or
any BLE hardware attached. world_cup.py and calibrate.py are the only
places this gets wired up to a real audio stream / real motors.

Usage:
    from whistle_policy import WhistlePolicy

    policy = WhistlePolicy(config)   # config from calibrate.py's whistle_config.json
    cmd = policy.update(audio_block)   # audio_block: 1-D float32 numpy array
    # cmd.forward_speed, cmd.turn_bias, cmd.goal_detected, cmd.state
"""

import dataclasses
import time

import numpy as np


@dataclasses.dataclass
class Command:
    """One control decision, computed from a single audio block."""
    forward_speed: float        # -100..100, signed forward(+)/back(-) drive speed
    turn_bias: float            # -100..100, signed turn: right(+) / left(-). Within the
                                 # turn band this is a straight linear gradient of pitch
                                 # position (low edge of the band = full left, high edge =
                                 # full right, center = straight) -- world_cup.py's shield
                                 # co-pilot mode reuses this exact value as a continuous
                                 # shield-position gradient instead of a turn command.
    goal_detected: bool         # True exactly once when the goal gesture fires
    state: str                  # human-readable label for the on-screen HUD
    frequency: float            # Hz, smoothed peak frequency this block; None = no tone
    tonal_ratio: float          # peak/mean spectral ratio this block (noise-masking diagnostic)
    rms: float                  # this block's RMS amplitude (noise-masking diagnostic)
    spectrum: np.ndarray        # magnitude spectrum this block, for the live plot
    freqs: np.ndarray           # frequency axis matching `spectrum`, for the live plot


# Defaults are placeholders -- run calibrate.py before a real match and use the
# whistle_config.json it writes instead of these. They're tuned for "a person
# whistling into a laptop mic from a couple feet away in a normal room," not
# for any specific whistler.
DEFAULT_CONFIG = {
    "sample_rate": 44100,
    "block_size": 2048,            # ~46ms/block @ 44100Hz, ~21.5Hz frequency resolution

    # A block only counts as "a whistle" if BOTH gates pass -- see whistle_policy's
    # module docstring / the project README's noise-masking answer.
    "tonal_ratio_gate": 6.0,       # peak/mean spectral magnitude (empirically: white noise
                                    # tops out ~4 at this block size, a clean tone hits ~450)
    "noise_floor_rms": 0.01,       # calibrated ambient RMS floor; block RMS must also clear this

    "stop_band": (600, 1000),      # Hz -- a low, comfortable whistle -> STOP
    "turn_band": (1000, 2200),     # Hz -- mid whistle -> TURN, a linear gradient of pitch
                                    # position within the band: the low edge is full left,
                                    # the high edge is full right, the center is straight.
    "forward_band": (2200, 4000),  # Hz -- high whistle -> FORWARD, speed scales with pitch

    "min_forward_speed": 20,       # % speed at the bottom of forward_band
    "max_speed": 55,               # % speed at the top of forward_band / max turn magnitude
    "ema_alpha": 0.3,              # frequency smoothing: higher = less smoothing, more responsive

    "silence_timeout_s": 0.4,      # no tonal block for this long -> start ramping to a stop
    "ramp_time_s": 0.3,            # time to ramp speed/turn down to zero once the timeout hits

    # Gesture recognition: short pulses in the forward band, counted in a rolling
    # window -- the "made it in the goal" command. (Shield control used to be a
    # separate 2-pulse gesture here; it's now a continuous gradient -- see
    # turn_bias's docstring above -- so there's nothing shield-specific left.)
    "pulse_max_duration_s": 0.4,   # a "pulse" must be shorter than this to count
    "goal_band": "forward",
    "goal_window_s": 2.0,
    "goal_pulse_count": 3,
}


class PitchDetector:
    """FFT-based peak-frequency + tonal-ratio estimator, one audio block at a time."""

    def __init__(self, sample_rate, block_size, ema_alpha):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.ema_alpha = ema_alpha
        self._window = np.hanning(block_size)
        self._freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)
        self._smoothed_freq = None

    @property
    def freqs(self):
        return self._freqs

    def analyze(self, block):
        """Return (smoothed_freq_or_None, tonal_ratio, rms, magnitude_spectrum) for one block.

        `smoothed_freq` is None only when the block is silent enough that the
        spectrum has no usable peak at all (all-zero input).
        """
        if len(block) != self.block_size:
            block = np.resize(np.asarray(block, dtype=np.float64), self.block_size)

        rms = float(np.sqrt(np.mean(np.square(block))))
        spectrum = np.abs(np.fft.rfft(block * self._window))

        # Bin 0 is DC (mic/amp offset, no pitch information) -- never treat it as the peak.
        usable = spectrum[1:]
        if usable.size == 0 or not np.any(usable):
            return None, 0.0, rms, spectrum

        peak_idx = int(np.argmax(usable)) + 1
        peak_mag = float(spectrum[peak_idx])
        mean_mag = float(np.mean(usable)) or 1e-9
        tonal_ratio = peak_mag / mean_mag
        raw_freq = float(self._freqs[peak_idx])

        if self._smoothed_freq is None:
            self._smoothed_freq = raw_freq
        else:
            self._smoothed_freq = (self.ema_alpha * raw_freq
                                    + (1.0 - self.ema_alpha) * self._smoothed_freq)

        return self._smoothed_freq, tonal_ratio, rms, spectrum


class WhistlePolicy:
    """Stateful policy: feed it audio blocks in order, get a Command back each time.

    See the project README's "describe the policy" section for the full
    explanation of the decision process this implements.
    """

    def __init__(self, config=None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.detector = PitchDetector(
            self.cfg["sample_rate"], self.cfg["block_size"], self.cfg["ema_alpha"])

        self._last_tone_time = None
        self._current_pulse_start = None
        self._current_pulse_band = None
        self._recent_pulses = []     # [(end_time, band), ...] completed short pulses
        self._ramp_start_time = None
        self._ramp_start_forward = 0.0
        self._ramp_start_turn = 0.0
        self._last_forward = 0.0
        self._last_turn = 0.0

    def _band_of(self, freq):
        for band, key in (("stop", "stop_band"), ("turn", "turn_band"), ("forward", "forward_band")):
            lo, hi = self.cfg[key]
            if lo <= freq <= hi:
                return band
        return None  # outside every expected whistle range -- treat like no tone at all

    def _track_pulses(self, has_whistle, band, now):
        """Update pulse-start/pulse-end bookkeeping and return goal_detected."""
        cfg = self.cfg

        if has_whistle:
            if self._current_pulse_start is None:
                self._current_pulse_start = now
                self._current_pulse_band = band
        elif self._current_pulse_start is not None:
            duration = now - self._current_pulse_start
            if duration <= cfg["pulse_max_duration_s"]:
                self._recent_pulses.append((now, self._current_pulse_band))
            self._current_pulse_start = None
            self._current_pulse_band = None

        self._recent_pulses = [(t, b) for t, b in self._recent_pulses if now - t <= cfg["goal_window_s"]]

        goal_pulses = [t for t, b in self._recent_pulses if b == cfg["goal_band"]]
        if len(goal_pulses) >= cfg["goal_pulse_count"]:
            self._recent_pulses.clear()
            return True
        return False

    def _drive_command(self, has_whistle, band, freq, now):
        """Return (forward_speed, turn_bias, state) for the continuous drive mapping."""
        cfg = self.cfg

        if has_whistle and band == "stop":
            self._ramp_start_time = None
            return 0.0, 0.0, "STOP"

        if has_whistle and band == "forward":
            self._ramp_start_time = None
            lo, hi = cfg["forward_band"]
            frac = 0.0 if hi == lo else max(0.0, min(1.0, (freq - lo) / (hi - lo)))
            speed = cfg["min_forward_speed"] + frac * (cfg["max_speed"] - cfg["min_forward_speed"])
            return speed, 0.0, f"FORWARD {speed:.0f}%"

        if has_whistle and band == "turn":
            self._ramp_start_time = None
            lo, hi = cfg["turn_band"]
            # 0.0 at the band's low edge, 1.0 at its high edge, 0.5 (straight) at the
            # center -- a direct gradient of pitch position, not a rate-of-change/slope.
            frac = 0.5 if hi == lo else max(0.0, min(1.0, (freq - lo) / (hi - lo)))
            turn = (frac - 0.5) * 2.0 * cfg["max_speed"]  # low edge -> -max_speed, high edge -> +max_speed
            direction = "RIGHT" if turn > 0 else ("LEFT" if turn < 0 else "CENTER")
            return 0.0, turn, f"TURN {direction} {abs(turn):.0f}%"

        # No usable tone this block -- see the README's "no whistle detected" answer.
        silent_for = None if self._last_tone_time is None else now - self._last_tone_time
        if silent_for is None or silent_for < cfg["silence_timeout_s"]:
            return self._last_forward, self._last_turn, "NO WHISTLE (holding)"

        if self._ramp_start_time is None:
            self._ramp_start_time = now
            self._ramp_start_forward = self._last_forward
            self._ramp_start_turn = self._last_turn

        t = (now - self._ramp_start_time) / cfg["ramp_time_s"]
        if t >= 1.0:
            return 0.0, 0.0, "NO WHISTLE (stopped)"
        return (self._ramp_start_forward * (1.0 - t),
                self._ramp_start_turn * (1.0 - t),
                f"NO WHISTLE (ramping down, {100 * (1 - t):.0f}% left)")

    def update(self, block, now=None):
        """Feed one audio block (1-D float array, ideally length == block_size). Returns a Command."""
        now = time.monotonic() if now is None else now
        cfg = self.cfg

        freq, tonal_ratio, rms, spectrum = self.detector.analyze(block)
        is_tonal = freq is not None and tonal_ratio >= cfg["tonal_ratio_gate"] and rms >= cfg["noise_floor_rms"]
        band = self._band_of(freq) if is_tonal else None
        has_whistle = band is not None

        if has_whistle:
            self._last_tone_time = now

        goal_detected = self._track_pulses(has_whistle, band, now)
        forward, turn, state = self._drive_command(has_whistle, band, freq, now)
        self._last_forward, self._last_turn = forward, turn

        if goal_detected:
            state = "GOAL COMMAND!"

        return Command(
            forward_speed=forward,
            turn_bias=turn,
            goal_detected=goal_detected,
            state=state,
            frequency=freq if has_whistle else None,
            tonal_ratio=tonal_ratio,
            rms=rms,
            spectrum=spectrum,
            freqs=self.detector.freqs,
        )
