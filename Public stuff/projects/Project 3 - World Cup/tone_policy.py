"""
tone_policy.py -- turns microphone audio into a named command by listening
for a few fixed pitches played from a phone tone-generator app.

No hardware and no I/O here: audio blocks in, a ToneReading out. That keeps
it testable with synthetic sine waves (see test_tone_policy.py).

Each computer only listens to ITS OWN tones:
    - the drive computer (Double Motor) uses DRIVE_TONES
    - the shield co-pilot (Single Motor) uses SHIELD_TONES, all higher
The peak is only searched for inside that computer's own frequency range,
so the other team member's phone playing at the same time is ignored.

Usage:
    from tone_policy import ToneDetector, DRIVE_TONES
    detector = ToneDetector(DRIVE_TONES)
    reading = detector.update(audio_block)   # 1-D float32 numpy array
    reading.tone   # "forward" / "left" / ... or None
"""

import dataclasses
import time

import numpy as np

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048  # ~46ms of new audio per update
# Each FFT covers the last FFT_SIZE samples (this block + the one before), so
# updates still come every ~46ms but the bins are ~10.8Hz wide. A steady tone
# stays in one bin while broadband noise spreads over twice as many, so a
# tone stands ~1.4x higher above the noise than with a 2048-sample FFT.
FFT_SIZE = 2 * BLOCK_SIZE

# Play these EXACT frequencies from the phone. A tone counts if it lands
# within band_half_width(name) of one of them (BAND_HALF_WIDTH_HZ, unless
# BAND_HALF_WIDTHS_HZ gives that command its own).
#
# The shield tones sit above the drive tones. All of them were picked so
# that no 2nd/3rd/4th harmonic of a drive tone (a loud phone speaker
# distorts a little) lands within 200Hz of ANY command -- anywhere in the
# goal's 2150-2450 range included. E.g. "goal" (2300) sits between 2x1000
# and 2x1250; the shield tones sit in the clean gap between 3000 (2x1500,
# 3x1000) and 3750 (3x1250), well below the goal's 2x (4300-4900). Re-check
# that if you change these (2500 would NOT work -- it's exactly 2x "left").
DRIVE_TONES = {
    "forward": 1000,
    "left": 1250,
    "right": 1500,
    "goal": 2300,
}
SHIELD_TONES = {
    "up": 3250,
    "down": 3500,
}
BAND_HALF_WIDTH_HZ = 60  # phone tones are exact; a narrow band gives noise fewer bins to land in
# "goal" gets a wider range (2150-2450Hz) so it's easy to hit. It can't grow
# past this: 2000 (2x forward) and 2500 (2x left) sit just outside it, and
# a loud speaker plays those harmonics whenever forward/left are held.
BAND_HALF_WIDTHS_HZ = {"goal": 150}
DISPLAY_MARGIN_HZ = 250  # dashboard's "peak" readout looks this far past the outermost bands

# Noise masking: a phone tone is one sharp spike in the spectrum; talking,
# motors, and room noise are spread out. The peak must be this many times
# the spectrum's MEDIAN level to count. Median rather than mean so that the
# other phone's tone (a second spike elsewhere) doesn't raise the bar.
# Measured (4096-sample FFT): white noise never exceeds ~4.2; a tone quieter
# than the room noise (sine amplitude 0.03 under white noise of 0.05) reads
# ~17-19. Raise TONAL_RATIO_GATE if a noisy room triggers commands by itself.
#
# Two gates (hysteresis): a NEW command must clear TONAL_RATIO_GATE, but the
# command that's already active only has to stay above SUSTAIN_RATIO_GATE.
# That lets the start gate be strict against noise without a held tone
# flickering out when it dips for a moment.
TONAL_RATIO_GATE = 12.0
SUSTAIN_RATIO_GATE = 8.0
MIN_RMS = 0.0005  # below this the mic is essentially silent -- ignore

# A new tone must be heard in this many updates in a row, AND its peak must
# stay within STABLE_BINS FFT bins the whole time, before switching to it.
# A phone tone sits dead still; voices, whistles, and clanks wander in pitch
# or are over in a fraction of a second. 4 updates ~ 185ms (windows overlap,
# so ~140ms of actual tone).
CONFIRM_BLOCKS = 4
STABLE_BINS = 1
HOLD_S = 0.3  # keep the current tone active through dropouts shorter than this

# These commands drop the INSTANT their tone isn't heard in range -- no
# HOLD_S grace -- so the car (and the continuously-spinning shield) stops
# exactly when the tone stops or drifts out of its band. To keep a momentary dropout from making the car stutter, the
# SAME tone coming back within HOLD_S resumes right away, without waiting
# CONFIRM_BLOCKS again. "goal" keeps HOLD_S so a held goal isn't reset by a blip.
IMMEDIATE_STOP_TONES = {"forward", "left", "right", "up", "down"}


def band_half_width(name):
    return BAND_HALF_WIDTHS_HZ.get(name, BAND_HALF_WIDTH_HZ)


@dataclasses.dataclass
class ToneReading:
    tone: str           # the active command, debounced -- None means "no tone"
    heard: str          # what THIS block alone matched (for the dashboard); None if nothing
    frequency: float    # strongest peak near this computer's bands, Hz (for the dashboard)
    tonal_ratio: float  # that peak / median spectrum level (for the dashboard)
    spectrum: np.ndarray


class ToneDetector:
    def __init__(self, tones, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, fft_size=FFT_SIZE):
        self.tones = dict(tones)
        self.block_size = block_size
        self.freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        self._window = np.hanning(fft_size)
        self._buffer = np.zeros(fft_size)  # the last fft_size samples, newest at the end

        self._band_idx = {name: np.nonzero(np.abs(self.freqs - f) <= band_half_width(name))[0]
                          for name, f in self.tones.items()}
        lo = min(self.tones.values()) - DISPLAY_MARGIN_HZ
        hi = max(self.tones.values()) + DISPLAY_MARGIN_HZ
        self._display_idx = np.nonzero((self.freqs >= lo) & (self.freqs <= hi))[0]
        self._reference = (self.freqs >= 100) & (self.freqs <= 10000)

        self.active = None
        self._last_heard_time = None
        self._candidate = None
        self._candidate_bin = None
        self._candidate_count = 0
        self._dropped = None       # immediate-stop command that just dropped out...
        self._dropped_time = None  # ...and when, so it can resume if it's back within HOLD_S

    def _match(self, spectrum, floor):
        """(name, peak bin) of the loudest band holding a real tone, or
        (None, None). Each band is checked on its own, so a louder tone
        elsewhere (the other phone, a harmonic) can't hide this one. The
        band's maximum must be a true peak -- not its edge bin -- otherwise
        it's just the skirt of a strong tone sitting outside the band. The
        active command wins whenever it's still there, so a stray sound in
        another band can't knock it out."""
        best, best_bin, best_mag = None, None, 0.0
        for name, idx in self._band_idx.items():
            i = idx[int(np.argmax(spectrum[idx]))]
            if i == idx[0] or i == idx[-1]:
                continue
            mag = float(spectrum[i])
            gate = SUSTAIN_RATIO_GATE if name == self.active else TONAL_RATIO_GATE
            if mag < gate * floor:
                continue
            if name == self.active:
                return name, int(i)
            if mag > best_mag:
                best, best_bin, best_mag = name, int(i), mag
        return best, best_bin

    def update(self, block, now=None):
        now = time.monotonic() if now is None else now
        block = np.asarray(block, dtype=np.float64)
        if len(block) != self.block_size:
            block = np.resize(block, self.block_size)

        self._buffer = np.concatenate((self._buffer[len(block):], block))

        spectrum = np.abs(np.fft.rfft(self._buffer * self._window))
        floor = float(np.median(spectrum[self._reference])) or 1e-12
        rms = float(np.sqrt(np.mean(block ** 2)))
        heard, heard_bin = self._match(spectrum, floor) if rms >= MIN_RMS else (None, None)

        peak_idx = self._display_idx[int(np.argmax(spectrum[self._display_idx]))]
        frequency = float(self.freqs[peak_idx])
        tonal_ratio = float(spectrum[peak_idx]) / floor

        if self.active in IMMEDIATE_STOP_TONES and heard != self.active:
            self._dropped, self._dropped_time = self.active, now
            self.active = None

        if heard is not None and heard == self.active:
            self._last_heard_time = now
            self._candidate, self._candidate_count = None, 0
        elif heard is not None and heard == self._dropped and now - self._dropped_time <= HOLD_S:
            self.active = heard  # same tone back after a momentary dropout -- resume
            self._last_heard_time = now
            self._candidate, self._candidate_count = None, 0
        elif heard is not None:
            steady = (heard == self._candidate
                      and abs(heard_bin - self._candidate_bin) <= STABLE_BINS)
            if steady:
                self._candidate_count += 1
            else:
                self._candidate, self._candidate_count = heard, 1
            self._candidate_bin = heard_bin
            if self._candidate_count >= CONFIRM_BLOCKS:
                self.active = heard
                self._last_heard_time = now
                self._candidate, self._candidate_count = None, 0
        else:
            self._candidate, self._candidate_count = None, 0

        if self.active is not None and now - self._last_heard_time > HOLD_S:
            self.active = None

        return ToneReading(tone=self.active, heard=heard, frequency=frequency,
                           tonal_ratio=tonal_ratio, spectrum=spectrum)
