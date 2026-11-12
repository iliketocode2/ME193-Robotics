"""Collect labeled finger images from the webcam.

    python collect.py              open the camera in config.CAMERA_INDEX
    python collect.py --camera 1   use another camera
    python collect.py --list       show which camera indexes work

Keys (click the window first):
    0-5     choose the label: how many fingers you will show
    SPACE   start or stop a burst (saves 5 images a second, stops by itself after 60)
    s       save one image
    u       undo: delete the last image saved this session
    q, ESC  quit

Each run is one session. Images go to data/images/<label>/<session>_<n>.png as 160 x 160 grayscale.
Move your hand around during a burst (closer, farther, tilted, left and right hand) so the network sees variety.
Label 0 is for a fist and also for the empty view with no hand.
"""

import argparse
import time
from datetime import datetime

import cv2
import numpy as np

import camera
import config

BURST_PERIOD_S = 0.2
BURST_MAX = 60
WINDOW = "collect (0-5 label, SPACE burst, s save, u undo, q quit)"


def counts():
    return {c: len(list((config.DATA_DIR / str(c)).glob("*.png"))) for c in config.CLASSES}


def draw(frame, label, bursting, burst_n, totals, session_n):
    view = frame.copy()
    x, y, side = camera.crop_box(frame)
    color = (0, 0, 255) if bursting else (0, 200, 0)
    cv2.rectangle(view, (x, y), (x + side, y + side), color, 2)

    small = camera.to_small(camera.to_saved(camera.center_square(frame)))
    inset = cv2.resize((small * 255).astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST)
    view[8:168, 8:168] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
    cv2.putText(view, "network view", (10, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    lines = [f"label: {label} fingers",
             f"burst {burst_n}/{BURST_MAX}" if bursting else "SPACE to start a burst",
             "saved: " + "  ".join(f"{c}:{totals[c]}" for c in config.CLASSES),
             f"this session: {session_n}"]
    for i, text in enumerate(lines):
        cv2.putText(view, text, (180, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(view, text, (180, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--list", action="store_true", help="list working camera indexes and exit")
    args = ap.parse_args()

    if args.list:
        for i, w, h in camera.list_cameras():
            print(f"camera {i}: {w} x {h}")
        return

    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    for c in config.CLASSES:
        (config.DATA_DIR / str(c)).mkdir(parents=True, exist_ok=True)
    cap = camera.open_camera(args.camera)
    label, bursting, burst_n, last_save = 1, False, 0, 0.0
    saved = []
    totals = counts()

    def save(frame):
        n = len(saved)
        path = config.DATA_DIR / str(label) / f"{session}_{n:04d}.png"
        cv2.imwrite(str(path), camera.to_saved(camera.center_square(frame)))
        saved.append(path)
        totals[label] += 1

    print(f"Session {session}. Saving to {config.DATA_DIR}")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames.")
            break
        now = time.time()
        if bursting and now - last_save >= BURST_PERIOD_S:
            save(frame)
            burst_n += 1
            last_save = now
            if burst_n >= BURST_MAX:
                bursting = False
        cv2.imshow(WINDOW, draw(frame, label, bursting, burst_n, totals, len(saved)))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if ord("0") <= key <= ord("5") and not bursting:
            label = key - ord("0")
        elif key == ord(" "):
            bursting = not bursting
            burst_n = 0
        elif key == ord("s"):
            save(frame)
        elif key == ord("u") and saved:
            path = saved.pop()
            path.unlink(missing_ok=True)
            totals[int(path.parent.name)] -= 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(saved)} images this session. Totals: {counts()}")


if __name__ == "__main__":
    main()
