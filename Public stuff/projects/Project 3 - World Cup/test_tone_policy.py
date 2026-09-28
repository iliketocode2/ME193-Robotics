"""
Synthetic-signal self-check for tone_policy.py -- no microphone or BLE
hardware needed. Feeds ToneDetector hand-built sine waves / noise and checks
it picks the right command.

Run:
    my_env_audio/Scripts/python "Public stuff/projects/Project 3 - World Cup/test_tone_policy.py"
"""

import numpy as np

from tone_policy import (BLOCK_SIZE, DRIVE_TONES, HOLD_S, SAMPLE_RATE, SHIELD_TONES,
                         ToneDetector)

DT = BLOCK_SIZE / SAMPLE_RATE
RNG = np.random.default_rng(0)
failures = []


def tone(freq, amplitude=0.3):
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * freq * t + RNG.uniform(0, 2 * np.pi))


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
    check(f"distorted {name} ({f} Hz) still reads as {name}", all(r.tone == name for r in readings[1:]))
    readings = feed(ToneDetector(SHIELD_TONES), [distorted(f) for _ in range(10)])
    check(f"harmonics of {name} ({f} Hz) don't move the shield", all(r.tone is None for r in readings))

# A loud tone just OUTSIDE a band doesn't leak into it.
for tones in (DRIVE_TONES, SHIELD_TONES):
    for name, f in tones.items():
        for offset in (-150, 150):
            if any(abs(f + offset - other) <= 100 for other in tones.values()):
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
check("quiet tone under louder noise -> forward (every block after the first two)",
      all(r.tone == "forward" for r in readings[2:]))

# A pitch between two commands does nothing.
readings = feed(ToneDetector(DRIVE_TONES), [tone(1125) for _ in range(10)])
check("1125 Hz (between forward and left) -> no command", all(r.tone is None for r in readings))

# Short dropouts don't interrupt a held command -- the old bug where a held
# forward tone kept cutting out.
blocks = []
for _ in range(10):
    blocks += [tone(DRIVE_TONES["forward"]) for _ in range(4)] + [np.zeros(BLOCK_SIZE)] * 2
readings = feed(ToneDetector(DRIVE_TONES), blocks)
check("held forward tone with ~90ms dropouts stays 'forward' throughout",
      all(r.tone == "forward" for r in readings[2:]))

# Real silence stops the command after HOLD_S.
blocks = [tone(DRIVE_TONES["forward"])] * 5 + [np.zeros(BLOCK_SIZE)] * 20
detector = ToneDetector(DRIVE_TONES)
readings = feed(detector, blocks)
stop_index = next(i for i, r in enumerate(readings) if i > 5 and r.tone is None)
check(f"silence -> command clears within HOLD_S ({HOLD_S}s)", (stop_index - 4) * DT <= HOLD_S + DT)

# A single stray block of another tone doesn't switch the command.
blocks = [tone(DRIVE_TONES["forward"])] * 5 + [tone(DRIVE_TONES["left"])] + [tone(DRIVE_TONES["forward"])] * 3
readings = feed(ToneDetector(DRIVE_TONES), blocks)
check("one stray 'left' block doesn't interrupt 'forward'", all(r.tone == "forward" for r in readings[2:]))

# Switching tones takes effect after CONFIRM_BLOCKS.
blocks = [tone(DRIVE_TONES["forward"])] * 5 + [tone(DRIVE_TONES["right"])] * 3
check("forward -> right switch", feed(ToneDetector(DRIVE_TONES), blocks)[-1].tone == "right")

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
