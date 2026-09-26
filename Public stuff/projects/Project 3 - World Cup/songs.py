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

Computer side: both platforms generate each note ourselves (a sine wave
with a short linear fade in/out, so notes don't click) at a controllable
amplitude (`COMPUTER_VOLUME`, near full-scale by default -- loud on
purpose, so the win/lose cue is hard to miss), rather than relying on
`winsound.Beep()`'s fixed, non-adjustable volume:
  - **Windows**: the generated tone is wrapped as an in-memory WAV and
    played via stdlib `winsound.PlaySound(..., SND_MEMORY)`. Without
    `SND_ASYNC` this blocks until playback finishes, same blocking-in-
    sequence idea as the hub side's `beep(blocking=True)`.
  - **macOS/Linux**: `winsound` doesn't exist there at all (it's Windows-
    only stdlib), so the same generated samples are written to a blocking
    PyAudio output stream instead (PyAudio is already a dependency of this
    project, for the microphone input side).
"""

import io
import sys
import threading
import time
import wave

import numpy as np

import legoeducation as le

if sys.platform == "win32":
    import winsound
else:
    import pyaudio

REST_BETWEEN_NOTES_S = 0.05
COMPUTER_NOTE_DURATION_MS = 180
COMPUTER_SAMPLE_RATE = 44100
COMPUTER_VOLUME = 0.95   # 0-1, near full-scale -- as loud as a single clean tone gets before clipping
COMPUTER_FADE_S = 0.005  # fade in/out per note so notes don't click (audible at an abrupt start/stop)
MIN_PLAYABLE_HZ = 37     # below this isn't meaningfully audible as a "note"
MAX_PLAYABLE_HZ = COMPUTER_SAMPLE_RATE // 2  # Nyquist limit at COMPUTER_SAMPLE_RATE

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


def _tone_samples(frequency, duration_ms, amplitude=COMPUTER_VOLUME,
                   sample_rate=COMPUTER_SAMPLE_RATE, fade_s=COMPUTER_FADE_S):
    """One note as float32 samples in [-amplitude, amplitude]: a sine wave
    with a short linear fade at each end. Shared by both the Windows
    (WAV/winsound) and non-Windows (PyAudio) playback paths below, so
    volume/fade tuning only has to happen in one place."""
    n = int(sample_rate * duration_ms / 1000)
    t = np.arange(n) / sample_rate
    samples = amplitude * np.sin(2 * np.pi * frequency * t)
    fade = min(int(sample_rate * fade_s), n // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        samples[:fade] *= ramp
        samples[-fade:] *= ramp[::-1]
    return samples.astype(np.float32)


def _play_computer_song_windows(notes, duration_ms, rest_s, amplitude):
    """Windows: wrap each generated tone as an in-memory 16-bit PCM WAV and
    play it via winsound.PlaySound -- see the module docstring for why this
    replaced winsound.Beep() (no volume control there at all)."""
    for frequency in notes:
        freq = int(round(frequency))
        if not (MIN_PLAYABLE_HZ <= freq <= MAX_PLAYABLE_HZ):
            print(f"[songs] Skipping computer note {freq}Hz -- outside the playable "
                  f"{MIN_PLAYABLE_HZ}-{MAX_PLAYABLE_HZ}Hz range.")
            continue
        try:
            samples = (_tone_samples(freq, duration_ms, amplitude=amplitude) * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(COMPUTER_SAMPLE_RATE)
                wf.writeframes(samples.tobytes())
            winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)
        except RuntimeError as e:
            print(f"[songs] Skipping computer note {freq}Hz after an error: {e}")
        time.sleep(rest_s)


def _play_computer_song_pyaudio(notes, duration_ms, rest_s, amplitude):
    """macOS/Linux (no winsound there at all): write each generated tone to
    a blocking PyAudio output stream, so each write waits for the note to
    finish, same sequencing as the Windows path. Uses its own PyAudio
    instance, separate from world_cup.py's microphone input stream."""
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=COMPUTER_SAMPLE_RATE, output=True)
    except Exception as e:
        print(f"[songs] Couldn't open the computer speaker, skipping computer song: {e}")
        pa.terminate()
        return
    try:
        for frequency in notes:
            freq = int(round(frequency))
            if not (MIN_PLAYABLE_HZ <= freq <= MAX_PLAYABLE_HZ):
                print(f"[songs] Skipping computer note {freq}Hz -- outside the playable "
                      f"{MIN_PLAYABLE_HZ}-{MAX_PLAYABLE_HZ}Hz range.")
                continue
            try:
                stream.write(_tone_samples(freq, duration_ms, amplitude=amplitude).tobytes())
            except Exception as e:
                print(f"[songs] Skipping computer note {freq}Hz after an error: {e}")
            time.sleep(rest_s)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def play_computer_song(notes, duration_ms=COMPUTER_NOTE_DURATION_MS, rest_s=REST_BETWEEN_NOTES_S,
                        amplitude=COMPUTER_VOLUME):
    """Play `notes` through the computer's own speaker, loud (see module
    docstring): winsound on Windows, a PyAudio output stream everywhere
    else, both playing the exact same generated tone."""
    if sys.platform == "win32":
        _play_computer_song_windows(notes, duration_ms, rest_s, amplitude)
    else:
        _play_computer_song_pyaudio(notes, duration_ms, rest_s, amplitude)


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
