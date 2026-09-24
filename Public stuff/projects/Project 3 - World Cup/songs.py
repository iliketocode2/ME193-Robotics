"""
songs.py -- "death" and "success" melodies played on a LEGO Education hub's
own speaker, plus the helper that plays them.

Verified against my_env/Lib/site-packages/legoeducation/basic_device.py's
`beep()`: shared by every device class (SingleMotor/DoubleMotor/Controller/
ColorSensor all subclass _BasicDevice), signature
`beep(*, pattern=le.SOUND_PATTERN_BEEP_SINGLE, frequency=440, count=1,
blocking=True)`, frequency clamped 0-2700 Hz. There's no explicit
note-duration parameter -- each beep() call plays one short, fixed-length
blip, so a "song" here is just a sequence of blocking beep() calls at
different frequencies with a short rest between notes; blocking=True makes
each call wait for that blip to finish before the next one starts, which is
what actually turns discrete beeps into a recognizable melody instead of
them overlapping/racing.
"""

import time

import legoeducation as le

REST_BETWEEN_NOTES_S = 0.05

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
            print(f"[songs] Skipping note {frequency}Hz after an error: {e}")
        time.sleep(rest_s)
