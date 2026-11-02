"""Stride calibration: walk a known distance, count the steps, store meters per step.

    python calibrate.py --distance 20

Mark a start and end line (e.g. 20 m of hallway). Press ENTER at the start line,
walk normally, press ENTER at the end line. Saves calibration.json, which
live.py uses for distance = steps x stride.
"""

import argparse
import json
import sys
import threading
import time

import config
from imu_stream import ImuStream
from live import LiveCounter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=float, required=True, help="distance you will walk, in meters")
    ap.add_argument("--board", default=config.BOARD_HOST)
    args = ap.parse_args()

    counter = LiveCounter()
    stream = ImuStream(args.board)
    stop = threading.Event()

    def run():
        for batch in stream.batches():
            if stop.is_set():
                return
            counter.feed(batch)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        input(f"Stand at the start line, then press ENTER and walk {args.distance:g} m... ")
        counter.reset()
        started = time.time()
        input("Press ENTER as you cross the end line. ")
        time.sleep(0.4)
        counted = len(counter.steps)
    except KeyboardInterrupt:
        print("\nCancelled.")
        stop.set()
        return
    stop.set()
    worker.join(timeout=1.0)
    stream.close()

    print(f"The network counted {counted} steps in {time.time() - started:.1f} s (IMU: {stream.status}).")
    answer = input("Press ENTER to use that, or type the number of steps you actually took: ").strip()
    steps = int(answer) if answer.isdigit() else counted
    if steps <= 0:
        sys.exit("No steps counted, calibration not saved. Is the wearable streaming?")
    stride = args.distance / steps
    config.CALIBRATION_PATH.write_text(json.dumps(
        {"stride_m": round(stride, 4), "distance_m": args.distance, "steps": steps,
         "network_counted": counted, "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    print(f"Stride length {stride:.3f} m per step saved to {config.CALIBRATION_PATH.name}.")


if __name__ == "__main__":
    main()
