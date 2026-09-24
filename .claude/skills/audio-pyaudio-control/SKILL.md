---
name: audio-pyaudio-control
description: PyAudio-based audio streaming and pitch/whistle detection for this repo -- the dual-venv requirement (my_env's Python 3.14 has no PyAudio wheel), the callback-mode streaming pattern, and the FFT-peak + tonal-ratio noise-gating approach used for whistle-controlled robot input. Use when a task needs raw microphone audio via PyAudio specifically (not sounddevice/miclib.py), or needs to detect a whistle/tone/pitch from a live audio stream.
---

# PyAudio-based audio control — ground truth from Project 3 (World Cup)

This documents what was actually verified while building
`Public stuff/projects/Project 3 - World Cup/` — a whistle-controlled robot
using PyAudio specifically (some assignments require PyAudio by name,
unlike the sounddevice-based `miclib.py` already in
`Public stuff/useful libraries/`). Read `world_cup.py`, `calibrate.py`,
`whistle_policy.py`, and `pyaudio_mic.py` there for the full working
implementation this skill summarizes.

## The Python 3.14 blocker — check this before assuming PyAudio will just install

`my_env/` runs Python 3.14. As of this check, PyPI's `pyaudio` (0.2.14)
ships prebuilt Windows wheels for `cp38` through `cp313` **only** — no
`cp314` wheel exists yet, and building from source fails in this
environment (missing PortAudio C headers, `fatal error C1083: Cannot open
include file: 'portaudio.h'`). This is exactly the scenario CLAUDE.md's
"Adding CNN/ML dependencies" section warns about generically for
Python 3.14 — don't assume `pip install pyaudio` will succeed here, and
don't silently substitute a different library or downgrade the *shared*
`my_env` to work around it (that would break everything else that already
works on 3.14).

**Verify first, every time**, since PyAudio may eventually ship a `cp314`
wheel:

```bash
my_env/Scripts/python -m pip install pyaudio
```

If that fails, the options (surface this decision to the user, don't pick
one silently — CLAUDE.md's existing guidance on this applies directly):

1. **A second venv pinned to an older Python** (what Project 3 does):
   `winget install Python.Python.3.13` (or 3.11/3.12), then
   `py -3.13 -m venv my_env_audio`, then install `legoeducation`,
   `pyaudio`, `numpy`, `matplotlib`, `paho-mqtt` into it. This venv is
   fully self-contained for whatever audio-driven script needs it — no
   need to keep switching back to `my_env` mid-script.
2. **`sounddevice` instead of PyAudio** — already installed in `my_env`,
   wraps the same PortAudio library, works today on 3.14. Only viable if
   the assignment doesn't require PyAudio by name.
3. **Build PortAudio from source** and point PyAudio's build at it —
   fragile, slow, not attempted in Project 3.

## Streaming pattern: callback mode, not blocking reads

For a real-time control loop (as opposed to `calibrate.py`'s one-shot
guided recording, which uses simple blocking `stream.read()` calls in a
loop since it doesn't need to run concurrently with anything else):

```python
import pyaudio

pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                  input=True, input_device_index=device_index,
                  frames_per_buffer=BLOCK_SIZE, stream_callback=audio_callback)
stream.start_stream()
```

`audio_callback(in_data, frame_count, time_info, status)` runs on
**PyAudio's own background thread**, not the thread that called
`pa.open()`. In `world_cup.py` this callback thread *is* the real control
loop: it decodes the block, runs the pitch/policy logic, and issues
non-blocking BLE motor commands directly, all before returning
`(None, pyaudio.paContinue)`. A separate thread (matplotlib's
`FuncAnimation` on the main thread) only *reads* a lock-guarded shared
state dict to draw a live display — it never touches audio or BLE itself.
This split exists for the same reason Project 2's README documents moving
motor commands to their own thread: mixing a blocking call into a
render/UI loop stalls it.

```python
block = np.frombuffer(in_data, dtype=np.float32)   # in_data is raw bytes
```

## Device enumeration: PyAudio has its own API, don't reach for `miclib.pick_mic()`

`miclib.py` is built on `sounddevice`, which has a different device-info
schema than PyAudio's `get_device_info_by_index()`. A project that
specifically requires PyAudio needs its own picker
(`pyaudio_mic.pick_mic(pa)` in Project 3) rather than importing
`miclib.pick_mic()` and trying to feed its return value into a PyAudio
call — the device *indices* aren't guaranteed to line up between the two
libraries' enumeration order:

```python
pa = pyaudio.PyAudio()
info = pa.get_device_info_by_index(i)
info["maxInputChannels"]       # > 0 means it's an input device
info["name"], info["defaultSampleRate"]
pa.get_default_input_device_info()["index"]   # raises IOError if none
```

## FFT-based pitch detection + noise masking

For detecting a whistle (a narrowband tone) against room noise/talking
(broadband):

1. Window each block (Hann) before FFT-ing — reduces spectral leakage from
   the block edges not being periodic.
2. Find the magnitude spectrum's peak bin, **skipping bin 0** (DC/mic
   offset carries no pitch information and will otherwise dominate quiet
   blocks).
3. Compute **tonal ratio = peak magnitude / mean magnitude** of the rest of
   the spectrum. This is the key noise-masking signal: empirically (at a
   2048-sample block, 44.1kHz), white noise's peak/mean ratio tops out
   around ~4, while a clean whistle tone hits ~450 — a gate of 6.0 has
   wide margin on both sides. Re-verify this empirically if the block size
   or sample rate changes materially; don't assume the same threshold
   transfers.
4. Also require a minimum RMS (calibrated against a few seconds of actual
   room silence, not guessed) so a faint but tonal electronic hum doesn't
   register as an intentional whistle.
5. Smooth the detected frequency across blocks with an exponential moving
   average — a single block's FFT peak is noisy frame-to-frame even for a
   held, steady whistle.

```python
window = np.hanning(block_size)
spectrum = np.abs(np.fft.rfft(block * window))
usable = spectrum[1:]                      # drop DC
peak_idx = int(np.argmax(usable)) + 1
tonal_ratio = spectrum[peak_idx] / np.mean(usable)
freq = np.fft.rfftfreq(block_size, d=1.0/sample_rate)[peak_idx]
```

**Frequency resolution is `sample_rate / block_size`** (≈21.5Hz at
44100/2048) — a detected/smoothed frequency near a calibrated band edge
can land on either side of it just from FFT bin quantization. Don't write
tests or calibration margins tighter than a few bin-widths from a band
boundary (Project 3's `test_whistle_policy.py` hit exactly this: test
tones placed 1Hz from a band edge flapped across it after EMA smoothing;
fixed by testing well inside each band instead).

## Calibrate against the actual user/room, don't hardcode Hz guesses

Different whistlers have very different comfortable pitch ranges. Rather
than picking fixed Hz bands, `calibrate.py`'s pattern — record ambient
noise for the floor, then the user's lowest and highest comfortable
whistle, derive bands from those — produces thresholds that actually match
the person and room on the day, and is the same spirit as this repo's
existing rule about not hardcoding a guessed `card_serial`. Save the result
to a small JSON file the real script loads (with a clearly-labeled
placeholder-default fallback and a printed warning if the file is
missing) rather than silently trusting defaults for a live demo/match.

## Common mistakes to catch in review

- Assuming `pip install pyaudio` will succeed on whatever Python this repo
  happens to be running without actually checking (see the blocker above).
- Doing BLE/motor calls or matplotlib artist updates directly inside the
  PyAudio callback's *calling* thread assumption mixed up — remember the
  callback itself runs on PyAudio's thread, and anything else (a Tk/mpl
  main loop) reading state it wrote needs a lock, not direct attribute
  access.
- Testing a frequency-band boundary within a few FFT bin-widths of the
  edge and being surprised when it flaps to the other side.
- Treating `tonal_ratio`/noise-floor constants as portable across a
  different block size, sample rate, or microphone without re-verifying
  them (empirical numbers here are specific to 2048 samples @ 44100Hz).
- Reaching for `miclib.pick_mic()` in a PyAudio-based script — it returns
  a `sounddevice` device index, not a PyAudio one.
