"""
Camera selection and initialisation helper for all ME193 examples.

Usage:
    from camlib import pick_camera
    cap, start_ms = pick_camera()
"""

import subprocess
import json
import sys
import time
import cv2
import numpy as np


def _camera_names():
    """Return display names for cameras from macOS system_profiler.
    Note: order may not match OpenCV's AVFoundation index order — use
    the visual preview in pick_camera() to identify cameras reliably.
    On Windows/Linux there's no equally lightweight OS call for this, so
    this returns [] and callers fall back to generic "Camera N" labels."""
    try:
        raw  = subprocess.check_output(
            ['system_profiler', 'SPCameraDataType', '-json'],
            text=True, stderr=subprocess.DEVNULL,
        )
        cams = json.loads(raw).get('SPCameraDataType', [])
        return [c.get('_name', f'Camera {i}') for i, c in enumerate(cams)]
    except Exception:
        return []


def _backend_candidates():
    """Platform-appropriate cv2.VideoCapture backend flags, in try-order.

    Opening with the wrong (or no) backend is the usual cause of black
    frames or corrupted/noisy images on a given OS: AVFoundation is the
    reliable backend on macOS, DirectShow (falling back to Media
    Foundation) on Windows.
    """
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _open_camera(index):
    """Open `index`, trying each platform backend in turn, and force MJPG
    decoding once opened.

    Without an explicit FOURCC, cv2.CAP_DSHOW on Windows frequently hands
    back a raw/YUY2 buffer that OpenCV misinterprets as BGR — the visible
    symptom is a black frame full of horizontal noise bars, not an
    exception. Forcing MJPG (widely supported, hardware-decoded on most
    webcams) fixes that.
    """
    for backend in _backend_candidates():
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            return cap
        cap.release()
    return cv2.VideoCapture(index)  # last resort, no backend hint


def pick_camera(width=640, height=480):
    """
    Scan for connected cameras, show a live preview thumbnail of each one,
    and let the user press the matching number key to choose.

    Returns:
        cap      — cv2.VideoCapture, already opened and warmed up
        start_ms — int, epoch milliseconds at the moment the camera opened
                   (use as a timestamp base for mediapipe detect_for_video)

    Raises RuntimeError if no cameras are found or the chosen camera won't open.
    """
    THUMB_W, THUMB_H = 320, 240

    print("Scanning for cameras…")
    names = _camera_names()

    # Probe each index and grab a preview frame
    available = []   # list of (index, label)
    previews  = {}   # index -> BGR thumbnail
    for i in range(6):
        cap = _open_camera(i)

        ret, frame = False, None
        # Skip the first few frames — they are often black while the sensor warms up
        for _ in range(8):
            ret, frame = cap.read()
        if ret and frame is not None:
            previews[i] = cv2.resize(frame, (THUMB_W, THUMB_H))
            label = names[i] if i < len(names) else f"Camera {i}"
            available.append((i, label))
        cap.release()
        time.sleep(0.1)  # let the driver settle between probes (Windows DirectShow is flaky about rapid open/close)

    if not available:
        raise RuntimeError(
            "No cameras found. Check System Settings → Privacy & Security → Camera "
            "(macOS) or that no other app is using the camera (Windows)."
        )

    # If only one camera, skip the chooser
    if len(available) == 1:
        chosen = available[0][0]
        print(f"One camera found: [{chosen}] {available[0][1]}\n")
    else:
        # Build a side-by-side thumbnail grid
        blank = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        tiles = []
        for idx, label in available:
            tile = previews.get(idx, blank).copy()
            # Dark banner at top for the label
            cv2.rectangle(tile, (0, 0), (THUMB_W, 34), (0, 0, 0), -1)
            cv2.putText(tile, f"[{idx}] {label}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
            # Instruction at bottom
            cv2.putText(tile, f"press  {idx}  to select",
                        (8, THUMB_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            tiles.append(tile)

        grid = np.hstack(tiles)
        cv2.imshow("Select camera — press its number key", grid)

        valid_keys = {ord(str(idx)): idx for idx, _ in available}
        chosen = available[0][0]          # default to first
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in valid_keys:
                chosen = valid_keys[key]
                break
            elif key in (27, ord('q')):   # ESC or Q = accept default
                break

        cv2.destroyAllWindows()
        print(f"Using camera {chosen}\n")

    # Open the chosen camera for real
    cap = _open_camera(chosen)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {chosen}. "
            "Check System Settings → Privacy & Security → Camera (macOS) or "
            "that no other app is using the camera (Windows)."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(0.5)                  # let the sensor stabilise
    start_ms = int(time.time() * 1000)
    return cap, start_ms
