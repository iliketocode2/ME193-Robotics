"""
pyaudio_mic.py -- microphone device selection built on PyAudio directly.

This project's assignment specifically requires grabbing the audio stream
with PyAudio (not `sounddevice`, which is what the shared miclib.py in
"Public stuff/useful libraries/" uses) -- see this project's README for why
that library split exists (Python 3.14/my_env has no PyAudio wheel; this
project runs in a separate my_env_audio, Python 3.13). This module mirrors
miclib.pick_mic()'s interface/UX for consistency, just implemented against
PyAudio's own device-enumeration API instead of sounddevice's.

Usage:
    import pyaudio
    from pyaudio_mic import pick_mic

    pa = pyaudio.PyAudio()
    device_index = pick_mic(pa)
    stream = pa.open(..., input_device_index=device_index, ...)
"""


def pick_mic(pa):
    """Print all audio input devices PyAudio can see and prompt the user to
    choose one. Returns the device index (int) to pass as
    `input_device_index=` to `pa.open(...)`, or None to use the system
    default. Never raises -- falls back to None on no devices, no input,
    or invalid input, same contract as miclib.pick_mic()."""
    device_count = pa.get_device_count()
    inputs = []
    for i in range(device_count):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            inputs.append((i, info))

    if not inputs:
        print("[pyaudio_mic] No input devices found -- using system default.")
        return None

    try:
        default_index = pa.get_default_input_device_info()["index"]
    except IOError:
        default_index = None

    print("\nAvailable microphone inputs:")
    for i, info in inputs:
        marker = " <- default" if i == default_index else ""
        rate = int(info.get("defaultSampleRate", 0))
        print(f"  [{i}] {info['name']}  ({rate} Hz){marker}")

    print()
    raw = input("Enter device number (or press Enter for default): ").strip()

    if raw == "":
        print("Using system default.\n")
        return None

    try:
        choice = int(raw)
        matched = next((info for i, info in inputs if i == choice), None)
        if matched is None:
            print(f"  '{choice}' is not an input device -- using system default.\n")
            return None
        print(f"Selected: {matched['name']}\n")
        return choice
    except ValueError:
        print("  Not a number -- using system default.\n")
        return None
