"""
spectrogram_test.py -- live scrolling spectrogram + tone-detector readout,
for sanity-checking the mic, the room, and the phone tones before running
world_cup.py. No robot, no MQTT, no game logic -- just the raw signal and
what tone_policy.py makes of it.

Shows, live:
  - a scrolling spectrogram (time x frequency, color = magnitude in dB),
    with every command band from tone_policy.py shaded (drive = green,
    shield = orange) so you can see exactly where the phone's tone lands
  - for BOTH the drive and the shield detector: what this block alone
    matched, the debounced command world_cup.py would act on, and the peak
    frequency + tonal ratio against TONAL_RATIO_GATE

Use it to "clean up" the input: play each tone, talk, clap, run the motors
nearby, play both phones at once, and check that only the right command
lights up. If a real tone reads NONE, look at the ratio -- move the phone
closer or turn it up. If noise triggers a command, raise TONAL_RATIO_GATE
in tone_policy.py.

Run (from my_env_audio -- NOT my_env, see this project's README):
    my_env_audio/Scripts/python "Public stuff/projects/Project 3 - World Cup/spectrogram_test.py"
"""

import threading

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation

from pyaudio_mic import pick_mic
from tone_policy import (BAND_HALF_WIDTH_HZ, BLOCK_SIZE, DRIVE_TONES, SAMPLE_RATE,
                         SHIELD_TONES, TONAL_RATIO_GATE, ToneDetector)

MAX_DISPLAY_HZ = 6500
HISTORY_COLUMNS = 200  # ~9s of scrollback at ~46ms/block
DB_FLOOR = -80.0

detectors = {"drive": ToneDetector(DRIVE_TONES), "shield": ToneDetector(SHIELD_TONES)}
freq_mask = detectors["drive"].freqs <= MAX_DISPLAY_HZ

state_lock = threading.Lock()
latest = {
    "spectrogram": np.full((int(freq_mask.sum()), HISTORY_COLUMNS), DB_FLOOR),
    "readings": {"drive": None, "shield": None},
    "rms": 0.0,
}


def audio_callback(in_data, frame_count, time_info, status):
    block = np.frombuffer(in_data, dtype=np.float32)
    readings = {name: d.update(block) for name, d in detectors.items()}
    db = np.clip(20 * np.log10(readings["drive"].spectrum[freq_mask] + 1e-6), DB_FLOOR, None)
    rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))

    with state_lock:
        spectrogram = latest["spectrogram"]
        spectrogram[:, :-1] = spectrogram[:, 1:]
        spectrogram[:, -1] = db
        latest["readings"] = readings
        latest["rms"] = rms
    return (None, pyaudio.paContinue)


def _shade_bands(ax, tones, color):
    for name, f in tones.items():
        ax.axhspan(f - BAND_HALF_WIDTH_HZ, f + BAND_HALF_WIDTH_HZ, color=color, alpha=0.15)
        ax.text(0.05, f, f"{name} {f}", color=color, fontsize=8, va="center")


def _describe(label, reading):
    if reading is None:
        return f"{label}: --"
    return (f"{label}: {(reading.tone or 'none').upper():<8} "
            f"(this block: {reading.heard or '-'})   "
            f"peak {reading.frequency:5.0f} Hz  ratio {reading.tonal_ratio:5.1f} / {TONAL_RATIO_GATE:.0f}")


def main():
    pa = pyaudio.PyAudio()
    device_index = pick_mic(pa)

    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                     input_device_index=device_index, frames_per_buffer=BLOCK_SIZE,
                     stream_callback=audio_callback)
    stream.start_stream()

    try:
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.canvas.manager.set_window_title("Tone spectrogram test")

        seconds_of_history = HISTORY_COLUMNS * BLOCK_SIZE / SAMPLE_RATE
        image = ax.imshow(latest["spectrogram"], aspect="auto", origin="lower",
                          extent=(-seconds_of_history, 0, 0, MAX_DISPLAY_HZ),
                          cmap="magma", vmin=DB_FLOOR, vmax=20)
        ax.set_xlabel("seconds ago")
        ax.set_ylabel("Hz")
        ax.set_title("Live spectrogram (brighter = louder) -- green = drive bands, orange = shield bands")
        fig.colorbar(image, ax=ax, label="dB")
        _shade_bands(ax, DRIVE_TONES, "#4dff88")
        _shade_bands(ax, SHIELD_TONES, "#ffa94d")

        hud_drive = fig.text(0.02, 0.045, "", fontsize=11, family="monospace", fontweight="bold")
        hud_shield = fig.text(0.02, 0.015, "", fontsize=11, family="monospace", fontweight="bold")
        fig.tight_layout(rect=(0, 0.08, 1, 1))

        def update(_frame):
            with state_lock:
                image.set_data(latest["spectrogram"].copy())
                readings = dict(latest["readings"])
                rms = latest["rms"]
            for hud, name in ((hud_drive, "drive"), (hud_shield, "shield")):
                reading = readings[name]
                hud.set_text(_describe(f"{name.upper():<6}", reading)
                             + (f"   rms {rms:.4f}" if name == "drive" else ""))
                hud.set_color("lime" if reading is not None and reading.tone else "#ff6b6b")
            return image, hud_drive, hud_shield

        ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
        plt.show()
        del ani  # only needs to stay alive until plt.show() returns
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
