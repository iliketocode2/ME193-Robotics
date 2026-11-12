"""FingerNet's forward pass in plain numpy, for the UNO Q (no PyTorch there).

The math is the same as model.py, written out by hand:
  convolution  slide each filter over the image and take a weighted sum at every position
  ReLU         max(0, x)
  max pooling  keep the largest value in each 2 x 2 block
  linear       weighted sum of all 2048 pooled values, one set of weights per class
  softmax      exp(score) / sum of exp(scores), so the 6 outputs add up to 1

train.py checks that this file and PyTorch give the same probabilities.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

LAYER_NAMES = ["conv1", "conv2", "conv3"]


def load_weights(path):
    with np.load(path) as f:
        return {k: f[k].astype(np.float32) for k in f.files}


def conv2d(x, w, b):
    """x: (C_in, H, W), w: (C_out, C_in, k, k), zero padding keeps H and W the same."""
    k = w.shape[-1]
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    windows = sliding_window_view(xp, (k, k), axis=(1, 2))        # (C_in, H, W, k, k)
    return np.einsum("chwij,ocij->ohw", windows, w, optimize=True) + b[:, None, None]


def relu(x):
    return np.maximum(x, 0)


def maxpool2(x):
    c, h, w = x.shape
    return x[:, :h // 2 * 2, :w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def forward(weights, img, return_maps=False):
    """img: standardized (64, 64) float32. Returns the 6 class probabilities."""
    x = img[None].astype(np.float32)
    maps = []
    for name in LAYER_NAMES:
        x = maxpool2(relu(conv2d(x, weights[f"{name}_w"], weights[f"{name}_b"])))
        maps.append(x)
    logits = weights["fc_w"] @ x.reshape(-1) + weights["fc_b"]
    probs = softmax(logits)
    return (probs, maps) if return_maps else probs
