"""Live step counting: wearable -> network -> debounced steps -> LÖVE game over UDP.

    python live.py                         from the wearable
    python live.py --replay walk_2026...   replay a recorded session in real time (no hardware)

Sends JSON to the game (127.0.0.1:9000) ten times a second and on every step:
    {"steps", "cadence" (steps/min), "distance" (m), "stride" (m/step), "calibrated",
     "last_step" (id of the newest step), "status"}
"""

import argparse
import json
import socket
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import config
import dataset
from imu_stream import ImuStream
from model import load_checkpoint
from stepcount import StepDetector

DEFAULT_STRIDE_M = 0.7
CADENCE_WINDOW_S = 10.0


def load_stride():
    if config.CALIBRATION_PATH.exists():
        return json.loads(config.CALIBRATION_PATH.read_text())["stride_m"], True
    return DEFAULT_STRIDE_M, False


class LiveCounter:
    def __init__(self, model_path=config.MODELS_DIR / "step_cnn_all.pt"):
        if not model_path.exists():
            raise SystemExit(f"No trained model at {model_path}: run train.py first.")
        self.model, self.ckpt = load_checkpoint(model_path)
        self.detector = StepDetector(self.ckpt["threshold"])
        self.buf_t = deque(maxlen=self.ckpt["window"])
        self.buf_x = deque(maxlen=self.ckpt["window"])
        self.since_eval = 0
        self.steps = []

    def reset(self):
        self.detector = StepDetector(self.ckpt["threshold"])
        self.steps = []

    @torch.no_grad()
    def feed(self, batch):
        """Add (t, sample) pairs; returns how many new steps were confirmed."""
        new = 0
        ck = self.ckpt
        for t, x in batch:
            self.buf_t.append(t)
            self.buf_x.append(x)
            self.since_eval += 1
            if len(self.buf_x) < ck["window"] or self.since_eval < config.LIVE_HOP:
                continue
            self.since_eval = 0
            w = dataset.normalize(np.array(self.buf_x), ck["mean"], ck["std"])[:, ck["channels"]]
            p = torch.sigmoid(self.model(torch.from_numpy(w.T[None].copy()))).item()
            if self.detector.update(self.buf_t[ck["center"]], p) is not None:
                self.steps.append(time.time())
                new += 1
        return new

    def cadence(self):
        """Steps per minute over the last 10 s; 0 once you've stopped for 2 s."""
        now = time.time()
        recent = [s for s in self.steps if now - s <= CADENCE_WINDOW_S]
        if len(recent) < 2 or now - recent[-1] > 2.0:
            return 0.0
        return 60.0 * (len(recent) - 1) / (recent[-1] - recent[0])


def replay_batches(name):
    s = dataset.load_session(name)
    start = time.time()
    for i in range(0, len(s.t), 10):
        while time.time() - start < s.t[i] - s.t[0]:
            yield []
            time.sleep(0.01)
        yield [(s.t[j], s.x[j]) for j in range(i, min(i + 10, len(s.t)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=config.BOARD_HOST)
    ap.add_argument("--replay", metavar="SESSION")
    ap.add_argument("--model", default=str(config.MODELS_DIR / "step_cnn_all.pt"))
    args = ap.parse_args()

    counter = LiveCounter(Path(args.model))
    stride, calibrated = load_stride()
    game = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stream = None
    if args.replay:
        batches, status = replay_batches(args.replay), (lambda: "replay")
    else:
        stream = ImuStream(args.board)
        batches, status = stream.batches(), (lambda: stream.status)
    print(f"Counting steps (stride {stride:.2f} m{'' if calibrated else ', uncalibrated'}). Ctrl+C to stop.")

    last_send = 0.0
    try:
        for batch in batches:
            new = counter.feed(batch)
            now = time.time()
            if new or now - last_send >= 0.1:
                last_send = now
                n = len(counter.steps)
                msg = {"steps": n, "cadence": round(counter.cadence(), 1), "distance": round(n * stride, 2),
                       "stride": stride, "calibrated": calibrated, "last_step": n, "status": status()}
                game.sendto(json.dumps(msg).encode(), (config.GAME_HOST, config.GAME_PORT))
                sys.stdout.write(f"\r  steps {n:5d}   cadence {msg['cadence']:5.1f}/min   "
                                 f"distance {msg['distance']:7.1f} m   IMU: {status()[:30]:30s}")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
