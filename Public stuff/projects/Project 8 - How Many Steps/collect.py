"""Record a session from the wearable, optionally labeling steps live.

    python collect.py --activity walk             press SPACE on every step, Q to stop
    python collect.py --activity still            no steps to label (still / shake)
    python collect.py --trial-steps 50            uncertainty trial: walk exactly 50 steps

Saves data/sessions/<name>.csv (t, ax..gz) and <name>.json (activity, step times, ...).
Fine-tune the step marks afterwards with label_tool.py.
"""

import argparse
import sys
import time

import numpy as np

import config
import dataset
from imu_stream import ImuStream
from keys import KeyReader


def record(stream, label_steps):
    t, x, steps = [], [], []
    last_t, last_wall = None, None
    started = time.time()
    with KeyReader() as keys:
        for batch in stream.batches():
            for ti, xi in batch:
                t.append(ti)
                x.append(xi)
            if batch:
                last_t, last_wall = batch[-1][0], time.time()
            key = keys.poll()
            while key is not None:
                if key in "qQ":
                    return np.array(t), np.array(x), steps
                if key == " " and label_steps and last_t is not None:
                    steps.append(round(last_t + time.time() - last_wall, 3))
                key = keys.poll()
            sys.stdout.write(f"\r  {time.time() - started:5.1f} s   {len(t):6d} samples   "
                             f"steps marked: {len(steps):4d}   IMU: {stream.status[:40]:40s}")
            sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activity", choices=config.ACTIVITIES, default="walk")
    ap.add_argument("--trial-steps", type=int, help="known step count for an uncertainty trial")
    ap.add_argument("--board", default=config.BOARD_HOST)
    ap.add_argument("--no-label", action="store_true", help="skip live labeling (label later with label_tool.py)")
    args = ap.parse_args()

    trial = args.trial_steps is not None
    activity = "walk" if trial else args.activity
    no_steps = activity in config.NON_STEP_ACTIVITIES
    label = not (no_steps or args.no_label)
    name = (f"trial{args.trial_steps}" if trial else activity) + time.strftime("_%Y%m%d-%H%M%S")

    print(f"Recording '{name}'.")
    if trial:
        print(f"Walk exactly {args.trial_steps} steps, counting them yourself.")
    if label:
        print("Press SPACE each time a foot lands (either foot). Press Q to stop.")
    else:
        print("Press Q to stop.")

    stream = ImuStream(args.board)
    try:
        t, x, steps = record(stream, label)
    except KeyboardInterrupt:
        print("\nCancelled, nothing saved.")
        return
    finally:
        stream.close()
    print()
    if len(t) < config.WINDOW:
        print("Less than a second of data arrived, nothing saved. Is the stream app running on the board?")
        return

    meta = {"activity": activity, "steps": steps, "labeled": label and bool(steps) or no_steps,
            "trial": trial, "true_steps": args.trial_steps if trial else (0 if no_steps else len(steps)),
            "lost_batches": stream.lost_batches, "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}
    dataset.save_session(name, t, x, meta)
    print(f"Saved {len(t) / config.SAMPLE_HZ:.1f} s, {len(steps)} steps marked, "
          f"{stream.lost_batches} batches lost -> data/sessions/{name}.csv")


if __name__ == "__main__":
    main()
