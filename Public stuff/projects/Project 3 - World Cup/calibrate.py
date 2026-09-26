"""
calibrate.py -- guided microphone/whistle calibration for world_cup.py.

Run this once per whistler/room before a match. It records a short ambient
sample (to set the noise floor) and asks you to hold your lowest and
highest comfortable whistle notes, then derives STOP/TURN/FORWARD frequency
bands from *your* actual range instead of guessed Hz values, and writes
them to whistle_config.json next to this script. world_cup.py loads that
file automatically; if it's missing, world_cup.py falls back to
whistle_policy.DEFAULT_CONFIG's placeholder bands and prints a warning.

Run (from the my_env_audio venv -- see this project's README):
    my_env_audio/Scripts/python "Public stuff/projects/Project 3 - World Cup/calibrate.py"
"""

import json
import os
import statistics

import numpy as np
import pyaudio

from pyaudio_mic import pick_mic
from whistle_policy import DEFAULT_CONFIG, PitchDetector

SAMPLE_RATE = DEFAULT_CONFIG["sample_rate"]
BLOCK_SIZE = DEFAULT_CONFIG["block_size"]
TONAL_RATIO_GATE = DEFAULT_CONFIG["tonal_ratio_gate"]

AMBIENT_DURATION_S = 2.5
WHISTLE_DURATION_S = 2.5
BAND_HALF_WIDTH_HZ = 150  # STOP/FORWARD bands are the calibrated pitch +/- this
MIN_GAP_HZ = 200          # minimum gap needed between low/high pitch to fit a turn band between them

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "whistle_config.json")


def record_phase(stream, label, duration_s):
    """Record `duration_s` seconds from `stream`, running a fresh PitchDetector
    over every block and printing a live frequency readout. Returns the list
    of (frequency, tonal_ratio, rms) for every block that passed the
    tonal-ratio gate."""
    detector = PitchDetector(SAMPLE_RATE, BLOCK_SIZE)
    n_blocks = int(duration_s * SAMPLE_RATE / BLOCK_SIZE)
    samples = []
    for _ in range(n_blocks):
        raw = stream.read(BLOCK_SIZE, exception_on_overflow=False)
        block = np.frombuffer(raw, dtype=np.float32)
        freq, tonal_ratio, rms, _ = detector.analyze(block)
        if freq is not None and tonal_ratio >= TONAL_RATIO_GATE:
            samples.append((freq, tonal_ratio, rms))
        reading = f"{freq:6.0f} Hz" if freq is not None else "   --  "
        print(f"\r  [{label}] {reading}  (tonal ratio {tonal_ratio:5.1f})", end="", flush=True)
    print()
    return samples


def prompt_ready(message):
    input(f"{message} Press Enter, then start immediately (recording starts right away): ")


def main():
    pa = pyaudio.PyAudio()
    try:
        device_index = pick_mic(pa)

        print("\n=== Step 1/3: ambient noise ===")
        prompt_ready("Stay quiet -- we're measuring the room's background noise level.")
        # Read raw RMS per block directly here rather than via record_phase(),
        # since record_phase() only returns *tonal* blocks (there shouldn't be
        # many/any during silence) and the noise floor needs every block's RMS.
        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                          input_device_index=device_index, frames_per_buffer=BLOCK_SIZE)
        try:
            rms_values = []
            n_blocks = int(AMBIENT_DURATION_S * SAMPLE_RATE / BLOCK_SIZE)
            for _ in range(n_blocks):
                raw = stream.read(BLOCK_SIZE, exception_on_overflow=False)
                block = np.frombuffer(raw, dtype=np.float32)
                rms_values.append(float(np.sqrt(np.mean(np.square(block)))))
        finally:
            stream.close()
        # The 90th percentile, not max(): a single one-off transient during this
        # 2.5s window (a click, a cough, the Enter keystroke's own sound, a chair
        # creak) would otherwise set the floor for the *entire match* at 2x that
        # spike -- max() has no defense against one outlier block, which is why
        # this kept producing wildly different floors across runs in the same
        # room (0.06 one run, 0.5+ the next). The 90th percentile only responds
        # to noise that's actually sustained across a real chunk of the window.
        ambient_p90_rms = float(np.percentile(rms_values, 90)) if rms_values else 0.001
        noise_floor_rms = max(ambient_p90_rms * 2.0, 0.005)  # 2x margin over that
        print(f"  Ambient RMS (90th percentile): {ambient_p90_rms:.4f} -> noise floor set to {noise_floor_rms:.4f}")

        print("\n=== Step 2/3: your LOWEST comfortable whistle ===")
        prompt_ready("Whistle your lowest comfortable note and hold it steady.")
        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                          input_device_index=device_index, frames_per_buffer=BLOCK_SIZE)
        try:
            low_samples = record_phase(stream, "low", WHISTLE_DURATION_S)
        finally:
            stream.close()

        print("\n=== Step 3/3: your HIGHEST comfortable whistle ===")
        prompt_ready("Whistle your highest comfortable note and hold it steady.")
        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                          input_device_index=device_index, frames_per_buffer=BLOCK_SIZE)
        try:
            high_samples = record_phase(stream, "high", WHISTLE_DURATION_S)
        finally:
            stream.close()
    finally:
        pa.terminate()

    if len(low_samples) < 5 or len(high_samples) < 5:
        print("\nCouldn't get a clear steady whistle reading in one or both phases "
              "(too few tonal blocks detected). Try again -- whistle a single steady "
              "note closer to the mic, and check the right device was selected.")
        return

    low_pitch = statistics.median(f for f, _, _ in low_samples)
    high_pitch = statistics.median(f for f, _, _ in high_samples)

    if high_pitch - low_pitch < MIN_GAP_HZ + 2 * BAND_HALF_WIDTH_HZ:
        print(f"\nYour low ({low_pitch:.0f} Hz) and high ({high_pitch:.0f} Hz) whistles are too "
              f"close together to fit a STOP band, a TURN band, and a FORWARD band without "
              f"overlap. Try widening the gap between your lowest and highest whistle and "
              f"run calibration again.")
        return

    stop_band = (low_pitch - BAND_HALF_WIDTH_HZ, low_pitch + BAND_HALF_WIDTH_HZ)
    forward_band = (high_pitch - BAND_HALF_WIDTH_HZ, high_pitch + BAND_HALF_WIDTH_HZ)
    turn_band = (stop_band[1], forward_band[0])

    config = {
        **DEFAULT_CONFIG,
        "noise_floor_rms": round(noise_floor_rms, 5),
        "stop_band": (round(stop_band[0]), round(stop_band[1])),
        "turn_band": (round(turn_band[0]), round(turn_band[1])),
        "forward_band": (round(forward_band[0]), round(forward_band[1])),
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved calibration to {CONFIG_PATH}:")
    print(f"  STOP band:    {config['stop_band'][0]}-{config['stop_band'][1]} Hz  (low whistle -> stop)")
    print(f"  TURN band:    {config['turn_band'][0]}-{config['turn_band'][1]} Hz  (mid whistle -> turn, direction from pitch slope)")
    print(f"  FORWARD band: {config['forward_band'][0]}-{config['forward_band'][1]} Hz  (high whistle -> drive forward)")
    print(f"  Noise floor:  {config['noise_floor_rms']} RMS")
    print("\nRun world_cup.py next -- it loads this file automatically.")


if __name__ == "__main__":
    main()
