"""
capture_frames.py -- grab phone-camera frames to label in Roboflow.

Point the phone at the ring the way it'll be mounted for the match, put
robots in (ours AND anything opponent-like: other LEGO builds, different
colors, at the edge, touching each other, half out of the ring), and save
frames. Aim for 100-200 varied images.

Keys (in the preview window):
    space  save this frame
    a      toggle auto-save (one frame every AUTO_EVERY_S)
    q/Esc  quit

Frames go to <repo>/runs/sumo_frames/ (gitignored -- upload them to
Roboflow from there; label ONE class, "robot", on every robot including
ours; export as "YOLOv8"; unzip the export into this folder; then run
train_sumo.py).

Run:
    my_env\\Scripts\\python "Public stuff/projects/Project 6 - Sumo Wrestling/capture_frames.py"
"""

import os
import time

import cv2

from config import FRAME_ROTATION, PHONE_STREAM_URL, PROCESS_WIDTH, REPO_ROOT

OUT_DIR = os.path.join(REPO_ROOT, "runs", "sumo_frames")
AUTO_EVERY_S = 0.7


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(PHONE_STREAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Can't open {PHONE_STREAM_URL} -- check PHONE_STREAM_URL in config.py")
    auto, last, saved = False, 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("stream dropped")
                break
            if FRAME_ROTATION is not None:
                frame = cv2.rotate(frame, FRAME_ROTATION)
            h, w = frame.shape[:2]
            if w > PROCESS_WIDTH:          # save at the size the brain will actually see
                frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * PROCESS_WIDTH / w)), interpolation=cv2.INTER_AREA)
            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()
            if key == ord(" ") or (auto and now - last > AUTO_EVERY_S):
                last = now
                path = os.path.join(OUT_DIR, f"sumo_{time.strftime('%Y%m%d_%H%M%S')}_{saved:04d}.jpg")
                cv2.imwrite(path, frame)
                saved += 1
            elif key == ord("a"):
                auto = not auto
            elif key in (ord("q"), 27):
                break
            preview = frame.copy()
            cv2.putText(preview, f"saved {saved}  auto {'ON' if auto else 'off'}  [space] save [a] auto [q] quit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("capture_frames", preview)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"saved {saved} frames to {OUT_DIR}")


if __name__ == "__main__":
    main()
