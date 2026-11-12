"""Opening the webcam and turning a frame into the network's input.

Only OpenCV and numpy, so the UNO Q can import this file unchanged.
"""

import cv2
import numpy as np

import config


def open_camera(index=config.CAMERA_INDEX, width=config.FRAME_W, height=config.FRAME_H):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Camera {index} did not open. Run `python collect.py --list` to see which indexes work.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def list_cameras(max_index=6):
    """Indexes that open and return a frame, with the frame size."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            found.append((i, frame.shape[1], frame.shape[0]))
        cap.release()
    return found


def crop_box(frame, frac=config.CROP_FRAC):
    """(x, y, side) of the center square the network looks at."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * frac)
    return (w - side) // 2, (h - side) // 2, side


def center_square(frame, frac=config.CROP_FRAC):
    x, y, side = crop_box(frame, frac)
    return frame[y:y + side, x:x + side]


def to_saved(square_bgr):
    """The square as it is stored on disk: SAVE_SIZE x SAVE_SIZE grayscale, uint8."""
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY) if square_bgr.ndim == 3 else square_bgr
    return cv2.resize(gray, (config.SAVE_SIZE, config.SAVE_SIZE), interpolation=cv2.INTER_AREA)


def to_small(saved_gray):
    """A saved square shrunk to the network size, as float32 in [0, 1]."""
    small = cv2.resize(saved_gray, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def standardize(img):
    """Per-image: subtract the mean brightness and divide by the spread, so lighting changes matter less."""
    return (img - img.mean()) / (img.std() + 1e-6)


def frame_to_input(frame):
    """Camera frame -> standardized 64 x 64 float32 image, exactly as in training."""
    return standardize(to_small(to_saved(center_square(frame))))
