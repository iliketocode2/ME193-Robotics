"""
Synthetic-signal self-check for tone_policy.py -- no microphone or BLE
hardware needed. Feeds ToneDetector hand-built sine waves / noise and checks
it picks the right command.

Run:
    my_env_audio/Scripts/python "Public stuff/projects/Project 3 - World Cup/test_tone_policy.py"
"""

import numpy as np

from tone_policy import (BLOCK_SIZE, CONFIRM_BLOCKS, DRIVE_TONES, HOLD_S, SAMPLE_RATE,
                         SHIELD_TONES, ToneDetector, band_half_width)

DT = BLOCK_SIZE / SAMPLE_RATE
RNG = np.random.default_rng(0)
failures = []
_phase = {}  # per-frequency running phase, so a tone is continuous across blocks like a real phone


def tone(freq, amplitude=0.3):
    start = _phase.setdefault(freq, RNG.uniform(0, 2 * np.pi))
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    _phase[freq] = (start + 2 * np.pi * freq * DT) % (2 * np.pi)
    return amplitude * np.sin(2 * np.pi * freq * t + start)


def warble(center, depth_hz, rate_hz, n_blocks, amplitude=0.3):
    """A tone whose pitch wanders +/-depth_hz around center, rate_hz times a
    second -- like a human whistle, a voice, or a squeaky motor."""
    t = np.arange(n_blocks * BLOCK_SIZE) / SAMPLE_RATE
    phase = 2 * np.pi * center * t - (depth_hz / rate_hz) * np.cos(2 * np.pi * rate_hz * t)
    return np.split(amplitude * np.sin(phase), n_blocks)


def noise(amplitude=0.05):
    return amplitude * RNG.standard_normal(BLOCK_SIZE)


def feed(detector, blocks, start=0.0):
    """Feed blocks one per DT; return the list of readings."""
    return [detector.update(b, now=start + i * DT) for i, b in enumerate(blocks)]


def check(name, condition):
    print(f"[{'OK' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(name)


# Every drive tone and every shield tone is recognized by its own detector.
for tones in (DRIVE_TONES, SHIELD_TONES):
    for name, freq in tones.items():
        readings = feed(ToneDetector(tones), [tone(freq) + noise() for _ in range(5)])
        check(f"{freq} Hz -> {name}", readings[-1].tone == name)

# Each detector ignores the other computer's tones completely.
for name, freq in SHIELD_TONES.items():
    readings = feed(ToneDetector(DRIVE_TONES), [tone(freq) for _ in range(10)])
    check(f"drive detector ignores shield tone {freq} Hz", all(r.tone is None for r in readings))
for name, freq in DRIVE_TONES.items():
    readings = feed(ToneDetector(SHIELD_TONES), [tone(freq) for _ in range(10)])
    check(f"shield detector ignores drive tone {freq} Hz", all(r.tone is None for r in readings))

# Both phones at once (other phone LOUDER): each computer still hears its own.
both = [tone(DRIVE_TONES["forward"], 0.1) + tone(SHIELD_TONES["up"], 0.5) + noise() for _ in range(5)]
check("drive hears 'forward' under a louder shield tone", feed(ToneDetector(DRIVE_TONES), both)[-1].tone == "forward")
both = [tone(DRIVE_TONES["left"], 0.5) + tone(SHIELD_TONES["down"], 0.1) + noise() for _ in range(5)]
check("shield hears 'down' under a louder drive tone", feed(ToneDetector(SHIELD_TONES), both)[-1].tone == "down")

# A badly distorted drive tone (harmonics as loud as the tone itself) must
# still read as itself on the drive computer and not move the shield.
def distorted(f):
    return tone(f) + tone(2 * f) + tone(3 * f, 0.2) + tone(4 * f, 0.2) + noise(0.005)


for name, f in DRIVE_TONES.items():
    readings = feed(ToneDetector(DRIVE_TONES), [distorted(f) for _ in range(10)])
    check(f"distorted {name} ({f} Hz) still reads as {name}",
          all(r.tone == name for r in readings[CONFIRM_BLOCKS:]))
    readings = feed(ToneDetector(SHIELD_TONES), [distorted(f) for _ in range(10)])
    check(f"harmonics of {name} ({f} Hz) don't move the shield", all(r.tone is None for r in readings))

# A loud tone just OUTSIDE a band doesn't leak into it.
for tones in (DRIVE_TONES, SHIELD_TONES):
    for name, f in tones.items():
        for offset in (-band_half_width(name) - 50, band_half_width(name) + 50):
            if any(abs(f + offset - other_f) <= band_half_width(other)
                   for other, other_f in tones.items()):
                continue  # that offset is inside a neighboring band -- a real command
            readings = feed(ToneDetector(tones), [tone(f + offset, 0.9) + noise(0.002) for _ in range(10)])
            check(f"{f + offset} Hz (just outside {name}) -> no command", all(r.tone is None for r in readings))

# The other phone's distorted tone, louder, doesn't hide the shield tone.
blocks = [tone(SHIELD_TONES["up"], 0.1) + distorted(DRIVE_TONES["right"]) for _ in range(5)]
check("shield hears 'up' under a louder distorted drive tone",
      feed(ToneDetector(SHIELD_TONES), blocks)[-1].tone == "up")

# Noise alone never produces a command.
for detector_tones in (DRIVE_TONES, SHIELD_TONES):
    readings = feed(ToneDetector(detector_tones), [noise(a) for a in (0.01, 0.1, 0.5) for _ in range(100)])
    check(f"noise alone -> no command ({list(detector_tones)})", all(r.tone is None for r in readings))

# A quiet tone buried in louder room noise still registers.
readings = feed(ToneDetector(DRIVE_TONES), [tone(DRIVE_TONES["forward"], 0.03) + noise(0.05) for _ in range(40)])
check("quiet tone under louder noise -> forward (every block once confirmed)",
      all(r.tone == "forward" for r in readings[CONFIRM_BLOCKS:]))

# A pitch between two commands does nothing.
readings = feed(ToneDetector(DRIVE_TONES), [tone(1125) for _ in range(10)])
check("1125 Hz (between forward and left) -> no command", all(r.tone is None for r in readings))

# Forward/left/right stop the moment their tone isn't heard, and the same
# tone coming back after a short dropout resumes right away (no re-confirm).
blocks = []
for _ in range(10):
    blocks += [tone(DRIVE_TONES["forward"]) for _ in range(4)] + [np.zeros(BLOCK_SIZE)] * 2
readings = feed(ToneDetector(DRIVE_TONES), blocks)
first = next(i for i, r in enumerate(readings) if r.tone == "forward")
check("~90ms dropouts in forward: car stops during each dropout",
      all(r.tone is None for r in readings[first:] if r.heard is None))
check("~90ms dropouts in forward: resumes as soon as the tone is back",
      all(r.tone == "forward" for r in readings[first:] if r.heard == "forward"))

# Silence stops forward/left/right within one update (the overlapping FFT
# window still holds the last tone block for one update after it ends).
for name in ("forward", "left", "right"):
    blocks = [tone(DRIVE_TONES[name]) for _ in range(8)] + [np.zeros(BLOCK_SIZE)] * 5
    readings = feed(ToneDetector(DRIVE_TONES), blocks)
    check(f"silence -> '{name}' stops within one update",
          readings[7].tone == name and all(r.tone is None for r in readings[9:]))

# A drive tone drifting out of its range stops the car just as fast.
for name in ("forward", "left", "right"):
    f = DRIVE_TONES[name]
    off = f + band_half_width(name) + 40  # outside this band, not in any other
    blocks = [tone(f) for _ in range(8)] + [tone(off) for _ in range(6)]
    readings = feed(ToneDetector(DRIVE_TONES), blocks)
    check(f"'{name}' drifting to {off} Hz (out of range) -> stops within one update",
          readings[7].tone == name and all(r.tone is None for r in readings[9:]))

# "goal" keeps HOLD_S: a short blip doesn't reset a held goal...
blocks = [tone(DRIVE_TONES["goal"]) for _ in range(6)] + [np.zeros(BLOCK_SIZE)] * 2     + [tone(DRIVE_TONES["goal"]) for _ in range(4)]
readings = feed(ToneDetector(DRIVE_TONES), blocks)
check("~90ms dropout in 'goal' doesn't interrupt it",
      all(r.tone == "goal" for r in readings[CONFIRM_BLOCKS:]))
# ...but real silence still clears it after HOLD_S.
blocks = [tone(DRIVE_TONES["goal"])] * 5 + [np.zeros(BLOCK_SIZE)] * 20
readings = feed(ToneDetector(DRIVE_TONES), blocks)
stop_index = next(i for i, r in enumerate(readings) if i > 5 and r.tone is None)
check(f"silence -> 'goal' clears within HOLD_S ({HOLD_S}s)", (stop_index - 4) * DT <= HOLD_S + 2 * DT)

# The wider goal range: anywhere in it counts, just outside it doesn't.
goal_lo = DRIVE_TONES["goal"] - band_half_width("goal")
goal_hi = DRIVE_TONES["goal"] + band_half_width("goal")
for f in (goal_lo + 30, 2400, goal_hi - 30):
    readings = feed(ToneDetector(DRIVE_TONES), [tone(f) + noise() for _ in range(8)])
    check(f"{f} Hz (inside the goal range {goal_lo}-{goal_hi}) -> goal", readings[-1].tone == "goal")

# A single stray block of another tone doesn't switch the command.
blocks = [tone(DRIVE_TONES["forward"])] * 5 + [tone(DRIVE_TONES["left"])] + [tone(DRIVE_TONES["forward"])] * 3
readings = feed(ToneDetector(DRIVE_TONES), blocks)
check("one stray 'left' block doesn't interrupt 'forward'",
      all(r.tone == "forward" for r in readings[CONFIRM_BLOCKS:]))

# Switching tones takes effect after CONFIRM_BLOCKS.
blocks = ([tone(DRIVE_TONES["forward"]) for _ in range(6)]
          + [tone(DRIVE_TONES["right"]) for _ in range(CONFIRM_BLOCKS + 1)])
check("forward -> right switch", feed(ToneDetector(DRIVE_TONES), blocks)[-1].tone == "right")

# Background sounds that have a sharp pitch but aren't a steady phone tone.
for tones in (DRIVE_TONES, SHIELD_TONES):
    for name, f in tones.items():
        # A whistle/voice whose pitch wanders +/-40Hz a few times a second.
        readings = feed(ToneDetector(tones), [b + noise(0.01) for b in warble(f, 40, 4, 60)])
        check(f"warbling pitch around {f} Hz ({name}) -> no command", all(r.tone is None for r in readings))
        # A short blip (a clank, a beep, one syllable) shorter than the confirm window.
        blocks = ([noise(0.01) for _ in range(5)] + [tone(f) + noise(0.01) for _ in range(CONFIRM_BLOCKS - 2)]
                  + [noise(0.01) for _ in range(10)])
        readings = feed(ToneDetector(tones), blocks)
        check(f"{CONFIRM_BLOCKS - 2}-block blip at {f} Hz ({name}) -> no command",
              all(r.tone is None for r in readings))

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
