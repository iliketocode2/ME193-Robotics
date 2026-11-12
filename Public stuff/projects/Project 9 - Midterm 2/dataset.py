"""Loading the collected images, splitting them, and augmenting them during training."""

import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import camera
import config


def find_images(data_dir=config.DATA_DIR):
    """[(path, label, session)] for every PNG under data_dir/<label>/."""
    items = []
    for c in config.CLASSES:
        for p in sorted((Path(data_dir) / str(c)).glob("*.png")):
            items.append((p, c, p.stem.rsplit("_", 1)[0]))
    return items


def split(items, val_frac=0.2, seed=0):
    """Hold out whole sessions for validation where possible.

    Images from one burst are nearly identical, so a random split would put near-copies of the same photo
    in both sets and make validation look better than it is. For each label with two or more sessions, its
    newest session becomes validation. A label with only one session falls back to a random split.
    """
    rng = random.Random(seed)
    by_label = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_label[it[1]][it[2]].append(it)
    train, val, notes = [], [], []
    for label, sessions in sorted(by_label.items()):
        names = sorted(sessions)
        if len(names) >= 2:
            for n in names[:-1]:
                train += sessions[n]
            val += sessions[names[-1]]
        else:
            imgs = sessions[names[0]][:]
            rng.shuffle(imgs)
            k = max(1, int(len(imgs) * val_frac))
            val += imgs[:k]
            train += imgs[k:]
            notes.append(label)
    if notes:
        print(f"Labels {notes} have one session only, so their validation images are a random 20% "
              "(expect validation accuracy to look too good for them). Collect a second session to fix this.")
    return train, val


def load_arrays(items):
    """Images as a (N, 1, 64, 64) float tensor in [0, 1] and labels as a (N,) long tensor."""
    x = np.stack([camera.to_small(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p, _, _ in items])
    y = np.array([c for _, c, _ in items])
    return torch.from_numpy(x[:, None]), torch.from_numpy(y).long()


def standardize(x):
    """Torch version of camera.standardize, applied to each image in a batch."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    return (x - mean) / (std + 1e-6)


def augment(x, max_deg=20, max_shift=0.12, scale=(0.85, 1.15), flip=True, gen=None):
    """A different random rotation, shift, zoom, mirror flip and brightness/contrast change for every image.

    The number of fingers does not change when the hand turns, moves, comes closer, or switches from left
    to right, so each altered copy is a new, correctly labeled training example.
    x: (B, 1, H, W) in [0, 1].
    """
    b = x.shape[0]
    rand = lambda lo, hi: torch.rand(b, generator=gen) * (hi - lo) + lo
    ang = rand(-max_deg, max_deg) * math.pi / 180
    s = rand(*scale)
    mirror = torch.where(torch.rand(b, generator=gen) < 0.5, -1.0, 1.0) if flip else torch.ones(b)
    tx, ty = rand(-max_shift, max_shift) * 2, rand(-max_shift, max_shift) * 2   # grid coords span -1..1

    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos * mirror, -sin, tx], 1),
                         torch.stack([sin * mirror, cos, ty], 1)], 1)          # (B, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    contrast = rand(0.7, 1.3).view(b, 1, 1, 1)
    bright = rand(-0.15, 0.15).view(b, 1, 1, 1)
    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * contrast + mean + bright
    out = out + torch.randn(out.shape, generator=gen) * 0.02
    return out.clamp(0, 1)
