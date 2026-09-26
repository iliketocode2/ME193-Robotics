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


# Bands are now fixed/hardcoded, not calibrated per-whistler -- these are meant
# to be played from a precise external tone source (e.g. a phone tone-generator
# app) rather than an actual human whistle, which is exactly what removes the
# need for calibrate.py's per-user band-fitting: a tone app can hit 700Hz or
# 4000Hz exactly and consistently, where a whistled pitch can't. calibrate.py
# still exists and still works (e.g. for someone who'd rather actually whistle),
# it's just no longer the primary/expected path -- whistle_config.json is a
# plain hand-edited file now if you want to override these.
DEFAULT_CONFIG = {
    "sample_rate": 44100,
    "block_size": 2048,            # ~46ms/block @ 44100Hz, ~21.5Hz frequency resolution

    # A block only counts as "a whistle" if BOTH gates pass -- see whistle_policy's
    # module docstring / the project README's noise-masking answer.
    "tonal_ratio_gate": 4.5,       # peak/mean spectral magnitude (empirically: white noise
                                    # tops out ~3.9 at this block size, a clean tone hits ~450
                                    # regardless of volume -- but a *quiet* whistle sitting on
                                    # top of real ambient noise drifts down toward that ~3.9
                                    # ceiling as it gets quieter, so this is lower than a naive
                                    # noise-vs-clean-tone comparison alone would suggest, to let
                                    # a quiet-but-real whistle register instead of only a loud,
                                    # very clean one -- at some cost of margin over noise)
    "noise_floor_rms": 0.01,       # calibrated ambient RMS floor; block RMS must also clear this

    "stop_band": (400, 650),       # Hz -- low tone (e.g. ~500Hz) -> STOP
    "turn_band": (700, 4000),      # Hz -- mid tone -> TURN, a linear gradient of pitch
                                    # position within the band: 700Hz is full left, 4000Hz is
                                    # full right, the center (~2350Hz) is straight. A wide zone
                                    # around that center (turn_deadzone_frac) counts as
                                    # "straight" too, not just one exact pitch -- and gently
                                    # creeps forward there instead of sitting still, so "going
                                    # straight" doesn't require jumping all the way to
                                    # forward_band. (The dead zone matters less with a precise
                                    # tone source than it did for an actual whistle, but doesn't
                                    # hurt to keep -- see turn_deadzone_frac.)
    "forward_band": (4050, 4650),  # Hz -- high tone -> FORWARD, speed scales with pitch
    # stop_band/forward_band deliberately leave a ~50Hz gap on either side of
    # turn_band's exact 700/4000Hz edges rather than touching them -- with
    # ~21.5Hz FFT bins at this block size, a tone dialed to exactly one of
    # those shared boundary values can otherwise snap to either neighboring
    # band unpredictably (verified: a 700Hz test tone classified as TURN, not
    # STOP, purely from bin quantization). The gap is a silent "no whistle"
    # zone instead -- safer than silently misclassifying a boundary tone as
    # the wrong command.

    "min_forward_speed": 20,       # % speed at the bottom of forward_band, and while
                                    # straight/creeping in the turn band's deadzone
    "max_speed": 55,               # % speed at the top of forward_band
    "max_turn_speed": 25,          # % speed cap for turning -- deliberately much lower than
                                    # max_speed so turns are gentle/controllable, not sharp
    "turn_deadzone_frac": 0.35,    # fraction of each half of turn_band (from center outward)
                                    # that counts as "straight" -- 0.35 means the middle 35%
                                    # of the band is a dead zone, with the turn gradient
                                    # ramping from 0 to max_turn_speed across the remaining
                                    # 65%, split evenly left/right
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
    """FFT-based peak-frequency + tonal-ratio estimator, one audio block at a
    time -- purely per-block, no state/smoothing carried between calls. That's
    deliberate: smoothing has to happen in WhistlePolicy instead (see its
    `_smoothed_freq`), because only WhistlePolicy knows which blocks actually
    pass the tonal-ratio/RMS gates. Smoothing every raw peak here regardless
    of gating (an earlier version of this class did exactly that) let a
    single noisy/gated-out block's spurious peak quietly drag the running
    average off course -- so a held, genuinely steady tone could drift out of
    its own band over time with no gap ever appearing in the "no whistle"
    sense, only fixable by whistling a new, strong-enough frequency to
    overwhelm the contaminated average. Keeping this class stateless and
    resetting the smoothed estimate on every non-tonal block (in
    WhistlePolicy.update()) avoids that entirely."""

    def __init__(self, sample_rate, block_size):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._window = np.hanning(block_size)
        self._freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)

    @property
    def freqs(self):
        return self._freqs

    def analyze(self, block):
        """Return (raw_freq_or_None, tonal_ratio, rms, magnitude_spectrum) for
        one block, with no smoothing applied. `raw_freq` is None only when the
        block is silent enough that the spectrum has no usable peak at all
        (all-zero input).
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

        return raw_freq, tonal_ratio, rms, spectrum


class WhistlePolicy:
    """Stateful policy: feed it audio blocks in order, get a Command back each time.

    See the project README's "describe the policy" section for the full
    explanation of the decision process this implements.
    """

    def __init__(self, config=None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.detector = PitchDetector(self.cfg["sample_rate"], self.cfg["block_size"])

        self._smoothed_freq = None  # only ever updated on a block that passes both gates -- see update()
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
            mid = (lo + hi) / 2.0
            half_width = (hi - lo) / 2.0
            deadzone_half = cfg["turn_deadzone_frac"] * half_width
            offset = freq - mid  # signed distance from center: negative=low/left, positive=high/right

            if half_width <= 0 or abs(offset) <= deadzone_half:
                # Wide "straight" zone around the center, not a single exact pitch --
                # creep forward gently rather than sit still, so this is a real
                # "go straight" state, not just "don't turn."
                return cfg["min_forward_speed"], 0.0, "STRAIGHT"

            active_width = half_width - deadzone_half
            magnitude = min(1.0, (abs(offset) - deadzone_half) / active_width)
            turn = magnitude * cfg["max_turn_speed"] * (1.0 if offset > 0 else -1.0)
            direction = "RIGHT" if turn > 0 else "LEFT"
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

        raw_freq, tonal_ratio, rms, spectrum = self.detector.analyze(block)
        is_tonal = raw_freq is not None and tonal_ratio >= cfg["tonal_ratio_gate"] and rms >= cfg["noise_floor_rms"]

        # Smoothing only ever incorporates a block that actually passed both
        # gates, and resets on one that doesn't -- see PitchDetector's and
        # this class's __init__ comments for why: blending in a gated-out
        # block's raw peak (noise, a transient dip, whatever caused this one
        # block to fail) would let it quietly drag the average off course for
        # every *later* tonal block too, with nothing to ever undo the drift
        # short of a strong new frequency overwhelming it.
        if is_tonal:
            if self._smoothed_freq is None:
                self._smoothed_freq = raw_freq
            else:
                self._smoothed_freq = (cfg["ema_alpha"] * raw_freq
                                        + (1.0 - cfg["ema_alpha"]) * self._smoothed_freq)
        else:
            self._smoothed_freq = None
        freq = self._smoothed_freq

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
