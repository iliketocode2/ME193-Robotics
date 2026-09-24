"""
spectrogram_test.py -- live scrolling spectrogram + noise-gate readout, for
tuning whistle_policy's thresholds and sanity-checking the mic/room before
running calibrate.py or world_cup.py. Same role as Project 2's
apriltag_stream_test.py: a standalone sanity check for the raw signal, with
no hardware/MQTT/game logic attached.

Shows, live:
  - a scrolling spectrogram (time x frequency, color = magnitude in dB),
    with the calibrated STOP/TURN/FORWARD bands shaded so you can see where
    your whistle actually lands relative to them
  - the current smoothed pitch estimate and tonal ratio
  - whether the current block passes whistle_policy's tonal-ratio + noise-
    floor gates (i.e. would register as "a whistle") -- this is the tool to
    use to "clean up" the audio input: whistle, talk, tap the table, run the
    car's motors nearby, etc. and watch which sounds light up green
    ("WHISTLE") vs. red ("no whistle / noise") to judge whether the gates
    need retuning (edit tonal_ratio_gate / noise_floor_rms in
    whistle_config.json, or rerun calibrate.py).

Run (from my_env_audio -- NOT my_env, see this project's README):
    my_env_audio/Scripts/python "Public stuff/projects/Project 3 - World Cup/spectrogram_test.py"
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation

from pyaudio_mic import pick_mic
from whistle_policy import DEFAULT_CONFIG, PitchDetector

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "whistle_config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    print(f"Loaded calibrated config from {CONFIG_PATH}")
else:
    config = DEFAULT_CONFIG
    print("No whistle_config.json found -- using DEFAULT_CONFIG placeholder thresholds "
          "(run calibrate.py for real values).")

SAMPLE_RATE = config["sample_rate"]
BLOCK_SIZE = config["block_size"]
TONAL_RATIO_GATE = config["tonal_ratio_gate"]
NOISE_FLOOR_RMS = config["noise_floor_rms"]
MAX_DISPLAY_HZ = max(4500, config["forward_band"][1] + 500)
HISTORY_COLUMNS = 200  # ~9s of scrollback at ~46ms/block (2048 samples @ 44100Hz)
DB_FLOOR = -80.0

detector = PitchDetector(SAMPLE_RATE, BLOCK_SIZE, ema_alpha=config["ema_alpha"])
freq_mask = detector.freqs <= MAX_DISPLAY_HZ
display_freqs = detector.freqs[freq_mask]

# Written by the PyAudio callback thread, read by matplotlib's animation on the
# main thread. Unlike world_cup.py's shared game-state, nothing here drives
# hardware, so a lock isn't needed for this read-only diagnostic display --
# CPython's GIL makes each individual dict-key assignment atomic, and a
# momentary torn read of the in-place spectrogram-shift is at worst a visual
# glitch for one frame, not a correctness/safety issue.
latest = {
    "spectrogram": np.full((display_freqs.size, HISTORY_COLUMNS), DB_FLOOR),
    "frequency": None,
    "tonal_ratio": 0.0,
    "rms": 0.0,
    "passes_gate": False,
}


def audio_callback(in_data, frame_count, time_info, status):
    block = np.frombuffer(in_data, dtype=np.float32)
    freq, tonal_ratio, rms, spectrum = detector.analyze(block)
    passes_gate = freq is not None and tonal_ratio >= TONAL_RATIO_GATE and rms >= NOISE_FLOOR_RMS

    db = np.clip(20 * np.log10(spectrum[freq_mask] + 1e-6), DB_FLOOR, None)
    spectrogram = latest["spectrogram"]
    spectrogram[:, :-1] = spectrogram[:, 1:]
    spectrogram[:, -1] = db

    latest["frequency"] = freq if passes_gate else None
    latest["tonal_ratio"] = tonal_ratio
    latest["rms"] = rms
    latest["passes_gate"] = passes_gate

    return (None, pyaudio.paContinue)


def main():
    pa = pyaudio.PyAudio()
    device_index = pick_mic(pa)

    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                      input_device_index=device_index, frames_per_buffer=BLOCK_SIZE,
                      stream_callback=audio_callback)
    stream.start_stream()

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.canvas.manager.set_window_title("Whistle spectrogram test")

        seconds_of_history = HISTORY_COLUMNS * BLOCK_SIZE / SAMPLE_RATE
        image = ax.imshow(latest["spectrogram"], aspect="auto", origin="lower",
                           extent=(-seconds_of_history, 0, 0, MAX_DISPLAY_HZ),
                           cmap="magma", vmin=DB_FLOOR, vmax=20)
        ax.set_xlabel("seconds ago")
        ax.set_ylabel("Hz")
        ax.set_title("Live spectrogram (brighter = louder)")
        fig.colorbar(image, ax=ax, label="dB")

        band_colors = {"stop_band": "cyan", "turn_band": "white", "forward_band": "lime"}
        for key, color in band_colors.items():
            lo, hi = config[key]
            ax.axhspan(lo, hi, color=color, alpha=0.12)
            ax.axhline(lo, color=color, linewidth=0.5, alpha=0.6)
            ax.axhline(hi, color=color, linewidth=0.5, alpha=0.6)
            ax.text(-0.3, (lo + hi) / 2, key.replace("_band", ""), color=color,
                    fontsize=8, va="center")

        hud = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0.05, 1, 1))

        def update(_frame):
            image.set_data(latest["spectrogram"])
            freq = latest["frequency"]
            freq_str = f"{freq:.0f} Hz" if freq else "--"
            passes = latest["passes_gate"]
            hud.set_text(
                f"freq={freq_str}   tonal ratio={latest['tonal_ratio']:.1f} (gate {TONAL_RATIO_GATE})"
                f"   rms={latest['rms']:.4f} (floor {NOISE_FLOOR_RMS})"
                f"   ->  {'WHISTLE' if passes else 'no whistle / noise'}")
            hud.set_color("lime" if passes else "red")
            return image, hud

        ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
        plt.show()
        del ani  # silence "unused variable" -- must stay alive only until plt.show() returns
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
