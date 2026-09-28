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

On the computer, winning and losing actually play mp3s instead of the beeps
(the hub still beeps): play_win() plays WIN_MP3, play_lose() plays
LOSE_MP3 -- see below.
"""

import ctypes
import io
import os
import shutil
import subprocess
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

# Played on the COMPUTER when we win / lose, instead of the beeped
# SUCCESS_SONG / DEATH_SONG (the hub still beeps those). Uses what each OS
# already has, so no extra pip install: Windows' built-in MCI player (winmm)
# or macOS's afplay; on Linux, whichever of ffplay/mpg123 is installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
WIN_MP3 = os.path.join(_HERE, "waka_waka_final.mp3")
LOSE_MP3 = os.path.join(_HERE, "jb_sorry.mp3")

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


_MCI_ALIAS = "world_cup_win"
_mp3_process = None  # non-Windows player subprocess, so a replay can stop the old one


def _mci(command):
    """Send one Windows MCI command string; raises RuntimeError on failure."""
    buf = ctypes.create_unicode_buffer(256)
    err = ctypes.windll.winmm.mciSendStringW(command, buf, len(buf), None)
    if err:
        msg = ctypes.create_unicode_buffer(256)
        ctypes.windll.winmm.mciGetErrorStringW(err, msg, len(msg))
        raise RuntimeError(f"MCI '{command}': {msg.value or err}")
    return buf.value


def stop_computer_mp3():
    """Stop the win mp3 if it's playing (safe to call when it isn't)."""
    global _mp3_process
    if sys.platform == "win32":
        try:
            _mci(f"close {_MCI_ALIAS}")
        except RuntimeError:
            pass  # wasn't open
    elif _mp3_process is not None:
        _mp3_process.terminate()
        _mp3_process = None


def play_computer_mp3(path=WIN_MP3):
    """Start playing `path` on the computer speaker WITHOUT waiting for it to
    finish (the whole song keeps playing in the background). Returns True if
    it started, False if it couldn't (missing file, no player) -- the caller
    falls back to beeps then, so a win/loss is never silent."""
    global _mp3_process
    if not os.path.exists(path):
        print(f"[songs] {path} not found")
        return False
    stop_computer_mp3()  # a second result (after an 'r' reset) replaces the old song instead of overlapping
    try:
        if sys.platform == "win32":
            _mci(f'open "{path}" type mpegvideo alias {_MCI_ALIAS}')
            _mci(f"play {_MCI_ALIAS}")
            return True
        player = (["afplay", path] if shutil.which("afplay")
                  else ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path] if shutil.which("ffplay")
                  else ["mpg123", "-q", path] if shutil.which("mpg123")
                  else None)
        if player is None:
            print("[songs] No mp3 player found (afplay/ffplay/mpg123)")
            return False
        _mp3_process = subprocess.Popen(player, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (RuntimeError, OSError) as e:
        print(f"[songs] Couldn't play {os.path.basename(path)}: {e}")
        return False


def _play_mp3_and_beeps(device, mp3, notes):
    """`mp3` on the computer, `notes` beeped on the hub. Falls back to the
    beeped song on the computer too if the mp3 can't play."""
    if play_computer_mp3(mp3):
        play_song(device, notes)
    else:
        play_both(device, notes)


def play_win(device):
    """Win: WIN_MP3 on the computer, SUCCESS_SONG beeped on the hub."""
    _play_mp3_and_beeps(device, WIN_MP3, SUCCESS_SONG)


def play_lose(device):
    """Loss: LOSE_MP3 on the computer, DEATH_SONG beeped on the hub."""
    _play_mp3_and_beeps(device, LOSE_MP3, DEATH_SONG)


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
