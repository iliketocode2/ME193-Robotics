"""
songs.py -- "death" and "success" melodies, played on TWO independent audio
outputs at once: the LEGO Education hub's own speaker (play_song) and the
laptop's speaker (play_computer_song) -- see play_both() below, which is
what world_cup.py actually calls. Two outputs on purpose: the hub's buzzer
is small/easy to miss, and playing both also makes it obvious from the
computer alone whether a death/success trigger actually fired, independent
of any BLE/hardware issue with the hub's speaker.

Hub side verified against my_env/Lib/site-packages/legoeducation/
basic_device.py's `beep()`: shared by every device class (SingleMotor/
DoubleMotor/Controller/ColorSensor all subclass _BasicDevice), signature
`beep(*, pattern=le.SOUND_PATTERN_BEEP_SINGLE, frequency=440, count=1,
blocking=True)`, frequency clamped 0-2700 Hz. There's no volume control
anywhere in basic_device.py -- beep()/stop_beep() is the entire sound API,
so if the hub is silent it's a hardware/volume-knob issue, not a missed
software setting. There's also no explicit note-duration parameter -- each
beep() call plays one short, fixed-length blip, so a "song" here is just a
sequence of blocking beep() calls at different frequencies with a short
rest between notes; blocking=True makes each call wait for that blip to
finish before the next one starts, which is what actually turns discrete
beeps into a recognizable melody instead of them overlapping/racing.

Computer side used to use `winsound.Beep()` (Python stdlib, Windows-only --
fine here since this whole repo targets Windows, see CLAUDE.md's setup
instructions), but `Beep()` has **no volume/amplitude parameter at all** --
its loudness is whatever Windows' internal tone generator happens to use,
not something callable code can turn up. To make the win/lose cue
genuinely louder, this instead synthesizes each note as a plain sine wave
at a controllable amplitude (`COMPUTER_VOLUME`, near full-scale) and plays
it through `winsound.PlaySound(..., winsound.SND_MEMORY)` -- still stdlib,
still Windows-only, but now the amplitude is ours to set. `PlaySound`
without `SND_ASYNC` blocks until playback finishes, same blocking-in-
sequence idea as the hub side's `beep(blocking=True)`.
"""

import io
import threading
import time
import wave
import winsound

import numpy as np

import legoeducation as le

REST_BETWEEN_NOTES_S = 0.05
COMPUTER_NOTE_DURATION_MS = 180
COMPUTER_SAMPLE_RATE = 44100
COMPUTER_VOLUME = 0.95  # 0-1, near full-scale -- as loud as a single clean tone gets before clipping

# Descending, off-key-ish -- meant to sound like a "whomp whomp" failure.
DEATH_SONG = [523, 466, 415, 349, 311, 233]

# Rising major-ish arpeggio -- meant to sound triumphant.
SUCCESS_SONG = [523, 659, 784, 1047, 1319]


def play_song(device, notes, rest_s=REST_BETWEEN_NOTES_S):
    """Play `notes` (a list of frequencies in Hz) as a sequence of blocking
    beeps on `device` (any lelib/legoeducation device -- beep() is shared by
    all four classes). Any single note failing (e.g. a momentary BLE hiccup)
    is skipped rather than aborting the rest of the song."""
    for frequency in notes:
        try:
            device.beep(pattern=le.SOUND_PATTERN_BEEP_SINGLE, frequency=frequency, blocking=True)
        except Exception as e:
            print(f"[songs] Skipping hub note {frequency}Hz after an error: {e}")
        time.sleep(rest_s)


def _synthesize_tone_wav(frequency, duration_ms, sample_rate=COMPUTER_SAMPLE_RATE, amplitude=COMPUTER_VOLUME):
    """A single-channel 16-bit PCM WAV, in memory, of one sine tone at a
    controllable amplitude -- see the module docstring for why this replaced
    winsound.Beep() (no volume control there at all)."""
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.arange(n_samples) / sample_rate
    samples = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def play_computer_song(notes, duration_ms=COMPUTER_NOTE_DURATION_MS, rest_s=REST_BETWEEN_NOTES_S,
                        amplitude=COMPUTER_VOLUME):
    """Play `notes` through the computer's own speaker, loud (see module
    docstring) -- a synthesized sine tone per note via winsound.PlaySound."""
    for frequency in notes:
        try:
            wav_bytes = _synthesize_tone_wav(frequency, duration_ms, amplitude=amplitude)
            winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
        except RuntimeError as e:
            print(f"[songs] Skipping computer note {frequency}Hz after an error: {e}")
        time.sleep(rest_s)


def play_both(device, notes):
    """Play `notes` on the computer speaker and the hub speaker at the same
    time (the computer song runs on a background thread while the hub song
    plays on the calling thread), so both are heard together instead of one
    after the other, and a working computer speaker confirms the trigger
    fired even if the hub's beep is inaudible for some other reason."""
    computer_thread = threading.Thread(target=play_computer_song, args=(notes,), daemon=True)
    computer_thread.start()
    play_song(device, notes)
    computer_thread.join(timeout=5)
