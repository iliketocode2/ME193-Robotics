"""The finger-count lock on the computer: the webcam, the trained network and lock_policy.py.

    python lock.py              PyTorch model
    python lock.py --numpy      the numpy network the UNO Q will run
    python lock.py --hide       hide the network's live reading, as on the real lock

Keys: SPACE or b is the button, q or ESC quits.
The three white dots are the digit LEDs and the large dot is the status LED (yellow listening, green open,
red wrong or locked out). Every event is added to results/lock_log.csv with its time. The digits themselves are
never written down.
"""

import argparse
import csv
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import camera
import config
from live import make_runner
from lock_policy import LockPolicy, CLEAR, READ, OPEN

WINDOW = "finger-count lock (SPACE button, q quit)"
LOG_PATH = config.RESULTS_DIR / "lock_log.csv"
BGR = {"off": (60, 60, 60), "yellow": (0, 210, 255), "green": (0, 200, 0), "red": (0, 0, 230)}


def text(img, s, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw(frame, policy, count, conf, now, hide):
    view = frame.copy()
    h, w = view.shape[:2]
    leds = policy.leds(now)

    if policy.state == OPEN:  # tint the whole picture green while unlocked
        view = cv2.addWeighted(view, 0.6, np.full_like(view, (0, 160, 0)), 0.4, 0)
        text(view, "UNLOCKED", (w // 2 - 150, h // 2), 2.0, (255, 255, 255), 4)

    x, y, side = camera.crop_box(frame)
    cv2.rectangle(view, (x, y), (x + side, y + side), BGR[leds["status"]] if leds["status"] != "off" else
                  (200, 200, 200), 2)

    # The LED panel along the bottom.
    cv2.rectangle(view, (0, h - 70), (w, h), (30, 30, 30), -1)
    for i, lit in enumerate(leds["digits"]):
        cv2.circle(view, (40 + i * 50, h - 35), 16, (255, 255, 255) if lit else (70, 70, 70), -1)
    cv2.circle(view, (w - 45, h - 35), 24, BGR[leds["status"]], -1)
    text(view, policy.message, (200, h - 28), 0.6)

    # The hold bar shows how close the current reading is to being accepted.
    if policy.state in (CLEAR, READ):
        p = policy.progress(now)
        cv2.rectangle(view, (200, h - 18), (200 + int(p * (w - 290)), h - 10), BGR["yellow"], -1)

    left = policy.time_left(now)
    if left is not None:
        text(view, f"{left:4.1f} s", (w - 120, 30), 0.7)
    if policy.fails:
        text(view, f"wrong codes: {policy.fails}/{policy.max_fails}", (10, 30), 0.6, BGR["red"])
    if not hide:
        sure = conf >= policy.min_conf
        text(view, f"network: {count} ({conf:.0%})", (10, h - 85), 0.7, (0, 230, 0) if sure else (0, 200, 255))
    return view


def log_event(writer, now, event, policy):
    writer.writerow([datetime.now().isoformat(timespec="milliseconds"), f"{now:.2f}", event, policy.state,
                     len(policy.entered), policy.fails])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--numpy", action="store_true", help="run npnet.py instead of PyTorch")
    ap.add_argument("--smooth", type=int, default=5, help="average the probabilities over this many frames")
    ap.add_argument("--hide", action="store_true", help="do not show what the network reads")
    args = ap.parse_args()

    run = make_runner(args.numpy)
    cap = camera.open_camera(args.camera)
    policy = LockPolicy()
    recent = deque(maxlen=args.smooth)
    start = time.time()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new_log = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["wall_time", "t_s", "event", "state_after", "digits_entered", "fails"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time() - start
            probs, _ = run(camera.frame_to_input(frame))
            recent.append(probs)
            avg = np.mean(recent, axis=0)
            count, conf = int(np.argmax(avg)), float(avg.max())

            for event in policy.update(count, conf, now):
                log_event(writer, now, event, policy)

            cv2.imshow(WINDOW, draw(frame, policy, count, conf, now, args.hide))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("b")):
                recent.clear()   # start the attempt from fresh readings
                for event in policy.press(now):
                    log_event(writer, now, event, policy)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
