"""Count fingers live from the webcam with the trained network.

    python live.py              PyTorch model (models/finger_cnn.pt)
    python live.py --numpy      the numpy copy the UNO Q will run (models/finger_cnn.npz)
    python live.py --camera 1   another camera

Keys: f shows or hides the feature maps (what each conv layer sees), q or ESC quits.
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

import camera
import config
import npnet
from model import load_checkpoint

WINDOW = "finger count (f feature maps, q quit)"
MAPS_WINDOW = "feature maps: conv1 (8), conv2 (16), conv3 (32)"


def mosaic(maps, tile=64, cols=16):
    """All three layers' feature maps as one image, each map scaled to its own brightest value."""
    rows = []
    for m in maps:
        tiles = []
        for k in range(m.shape[0]):
            a = m[k]
            a = (a / (a.max() + 1e-6) * 255).astype(np.uint8)
            tiles.append(cv2.applyColorMap(cv2.resize(a, (tile, tile), interpolation=cv2.INTER_NEAREST),
                                           cv2.COLORMAP_MAGMA))
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        for r in range(0, len(tiles), cols):
            rows.append(np.hstack(tiles[r:r + cols]))
        rows.append(np.full((6, tile * cols, 3), 80, np.uint8))
    return np.vstack(rows[:-1])


def draw(frame, inp, probs, fps):
    view = frame.copy()
    x, y, side = camera.crop_box(frame)
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    color = (0, 200, 0) if conf > 0.8 else (0, 200, 255)
    cv2.rectangle(view, (x, y), (x + side, y + side), color, 2)

    shown = cv2.normalize(inp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    view[8:168, 8:168] = cv2.cvtColor(cv2.resize(shown, (160, 160), interpolation=cv2.INTER_NEAREST),
                                      cv2.COLOR_GRAY2BGR)
    cv2.putText(view, str(pred), (view.shape[1] - 110, 110), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 12)
    cv2.putText(view, str(pred), (view.shape[1] - 110, 110), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 5)
    cv2.putText(view, f"{conf:.0%} sure  {fps:.0f} fps", (view.shape[1] - 230, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for c, p in zip(config.CLASSES, probs):
        y0 = 200 + c * 26
        cv2.putText(view, str(c), (10, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(view, (30, y0), (30 + int(p * 150), y0 + 18), (255, 160, 60), -1)
        cv2.rectangle(view, (30, y0), (180, y0 + 18), (255, 255, 255), 1)
    return view


def make_runner(use_numpy=False):
    """A function that takes one network input and returns (probabilities, feature maps)."""
    if use_numpy:
        weights = npnet.load_weights(config.NUMPY_WEIGHTS_PATH)
        return lambda inp: npnet.forward(weights, inp, return_maps=True)

    model, _ = load_checkpoint()

    def run(inp):
        with torch.no_grad():
            logits, maps = model(torch.from_numpy(inp)[None, None], return_maps=True)
        return torch.softmax(logits, 1)[0].numpy(), [m[0].numpy() for m in maps]
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--numpy", action="store_true", help="run npnet.py instead of PyTorch")
    ap.add_argument("--smooth", type=int, default=5, help="average the probabilities over this many frames")
    args = ap.parse_args()

    run = make_runner(args.numpy)
    cap = camera.open_camera(args.camera)
    recent = deque(maxlen=args.smooth)
    show_maps, last = False, time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        inp = camera.frame_to_input(frame)
        probs, maps = run(inp)
        recent.append(probs)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
        last = now
        cv2.imshow(WINDOW, draw(frame, inp, np.mean(recent, axis=0), fps))
        if show_maps:
            cv2.imshow(MAPS_WINDOW, mosaic(maps))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            show_maps = not show_maps
            if not show_maps:
                cv2.destroyWindow(MAPS_WINDOW)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
