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
BLOCK_SIZE = 2048  # ~46ms per block, ~21.5Hz per FFT bin

# Play these EXACT frequencies from the phone. A tone counts if it lands
# within BAND_HALF_WIDTH_HZ of one of them.
#
# The shield tones sit well above the drive tones. All of them were picked
# so that no 2nd/3rd/4th harmonic of a drive tone (a loud phone speaker
# distorts a little) lands within 200Hz of ANY command -- e.g. 2x1000 =
# 2000Hz is past "goal" (1750), and 3x1500 = 4500 / 4x1000 = 4000 are both
# 250Hz clear of "up" (4250). Re-check that if you change these.
DRIVE_TONES = {
    "forward": 1000,
    "left": 1250,
    "right": 1500,
    "goal": 1750,
}
SHIELD_TONES = {
    "up": 4250,
    "down": 5600,
}
BAND_HALF_WIDTH_HZ = 100
DISPLAY_MARGIN_HZ = 250  # dashboard's "peak" readout looks this far past the outermost bands

# Noise masking: a phone tone is one sharp spike in the spectrum; talking,
# motors, and room noise are spread out. The peak must be this many times
# the spectrum's MEDIAN level to count. Median rather than mean so that the
# other phone's tone (a second spike elsewhere) doesn't raise the bar.
# Measured: white noise never exceeds ~4; a tone quieter than the room
# noise (sine amplitude 0.03 under white noise of 0.05) already clears 8.
# Raise this if a noisy room triggers commands by itself.
TONAL_RATIO_GATE = 8.0
MIN_RMS = 0.0005  # below this the mic is essentially silent -- ignore

CONFIRM_BLOCKS = 2  # a new tone must be heard in this many blocks in a row before switching to it
HOLD_S = 0.3        # keep the current tone active through dropouts shorter than this


@dataclasses.dataclass
class ToneReading:
    tone: str           # the active command, debounced -- None means "no tone"
    heard: str          # what THIS block alone matched (for the dashboard); None if nothing
    frequency: float    # strongest peak near this computer's bands, Hz (for the dashboard)
    tonal_ratio: float  # that peak / median spectrum level (for the dashboard)
    spectrum: np.ndarray


class ToneDetector:
    def __init__(self, tones, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE):
        self.tones = dict(tones)
        self.block_size = block_size
        self.freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)
        self._window = np.hanning(block_size)

        self._band_idx = {name: np.nonzero(np.abs(self.freqs - f) <= BAND_HALF_WIDTH_HZ)[0]
                          for name, f in self.tones.items()}
        lo = min(self.tones.values()) - DISPLAY_MARGIN_HZ
        hi = max(self.tones.values()) + DISPLAY_MARGIN_HZ
        self._display_idx = np.nonzero((self.freqs >= lo) & (self.freqs <= hi))[0]
        self._reference = (self.freqs >= 100) & (self.freqs <= 10000)

        self.active = None
        self._last_heard_time = None
        self._candidate = None
        self._candidate_count = 0

    def _match(self, spectrum, floor):
        """Name of the loudest band holding a real tone, or None. Each band is
        checked on its own, so a louder tone elsewhere (the other phone, a
        harmonic) can't hide this one. The band's maximum must be a true peak
        -- not its edge bin -- otherwise it's just the skirt of a strong tone
        sitting outside the band."""
        best, best_mag = None, 0.0
        for name, idx in self._band_idx.items():
            i = idx[int(np.argmax(spectrum[idx]))]
            if i == idx[0] or i == idx[-1]:
                continue
            mag = float(spectrum[i])
            if mag >= TONAL_RATIO_GATE * floor and mag > best_mag:
                best, best_mag = name, mag
        return best

    def update(self, block, now=None):
        now = time.monotonic() if now is None else now
        block = np.asarray(block, dtype=np.float64)
        if len(block) != self.block_size:
            block = np.resize(block, self.block_size)

        spectrum = np.abs(np.fft.rfft(block * self._window))
        floor = float(np.median(spectrum[self._reference])) or 1e-12
        rms = float(np.sqrt(np.mean(block ** 2)))
        heard = self._match(spectrum, floor) if rms >= MIN_RMS else None

        peak_idx = self._display_idx[int(np.argmax(spectrum[self._display_idx]))]
        frequency = float(self.freqs[peak_idx])
        tonal_ratio = float(spectrum[peak_idx]) / floor

        if heard is not None and heard == self.active:
            self._last_heard_time = now
            self._candidate, self._candidate_count = None, 0
        elif heard is not None:
            if heard == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate, self._candidate_count = heard, 1
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
