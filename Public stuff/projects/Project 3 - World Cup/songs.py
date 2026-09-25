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

Computer side uses `winsound.Beep()` (Python stdlib, Windows-only -- fine
here since this whole repo targets Windows, see CLAUDE.md's setup
instructions). It takes an explicit duration in milliseconds and blocks the
calling thread for that long, same blocking-in-sequence idea as the hub side.
"""

import threading
import time
import winsound

import legoeducation as le

REST_BETWEEN_NOTES_S = 0.05
COMPUTER_NOTE_DURATION_MS = 180
WINSOUND_MIN_HZ, WINSOUND_MAX_HZ = 37, 32767  # winsound.Beep's own valid range

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


def play_computer_song(notes, duration_ms=COMPUTER_NOTE_DURATION_MS, rest_s=REST_BETWEEN_NOTES_S):
    """Play `notes` through the computer's own speaker via winsound.Beep.
    Frequencies outside winsound's valid range are skipped (clamped, not
    silently distorted) rather than raising."""
    for frequency in notes:
        freq = int(round(frequency))
        if not (WINSOUND_MIN_HZ <= freq <= WINSOUND_MAX_HZ):
            print(f"[songs] Skipping computer note {freq}Hz -- outside winsound's "
                  f"{WINSOUND_MIN_HZ}-{WINSOUND_MAX_HZ}Hz range.")
            continue
        try:
            winsound.Beep(freq, duration_ms)
        except RuntimeError as e:
            print(f"[songs] Skipping computer note {freq}Hz after an error: {e}")
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
