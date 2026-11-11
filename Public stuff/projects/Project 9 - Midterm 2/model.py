"""FingerNet: three convolution layers, each followed by ReLU and 2x2 max pooling, then one linear layer.

    input  1 x 64 x 64   grayscale image
    conv1  8 filters, 5 x 5   -> 8 x 64 x 64  -> pool -> 8 x 32 x 32
    conv2  16 filters, 3 x 3  -> 16 x 32 x 32 -> pool -> 16 x 16 x 16
    conv3  32 filters, 3 x 3  -> 32 x 16 x 16 -> pool -> 32 x 8 x 8
    flatten 2048 numbers -> linear -> 6 scores, one per finger count (0 to 5)

Cross-entropy loss turns the 6 scores into probabilities with softmax and penalizes a low probability
on the right answer.
"""

import torch
from torch import nn
import torch.nn.functional as F

import config

LAYERS = [  # (name, in_channels, out_channels, kernel)
    ("conv1", 1, 8, 5),
    ("conv2", 8, 16, 3),
    ("conv3", 16, 32, 3),
]


class FingerNet(nn.Module):
    def __init__(self, n_classes=len(config.CLASSES), img_size=config.IMG_SIZE, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(i, o, k, padding=k // 2) for _, i, o, k in LAYERS)
        side = img_size // 2 ** len(LAYERS)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(LAYERS[-1][2] * side * side, n_classes)

    def forward(self, x, return_maps=False):
        maps = []
        for conv in self.convs:
            x = F.max_pool2d(F.relu(conv(x)), 2)
            maps.append(x)
        logits = self.fc(self.dropout(torch.flatten(x, 1)))
        return (logits, maps) if return_maps else logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, model, **info):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "img_size": config.IMG_SIZE, **info}, path)


def load_checkpoint(path=config.MODEL_PATH):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FingerNet(img_size=ckpt["img_size"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def export_numpy(model, path=config.NUMPY_WEIGHTS_PATH):
    """Save the weights as plain arrays for npnet.py (the UNO Q runs the network without PyTorch)."""
    arrays = {}
    for (name, *_), conv in zip(LAYERS, model.convs):
        arrays[f"{name}_w"] = conv.weight.detach().numpy()
        arrays[f"{name}_b"] = conv.bias.detach().numpy()
    arrays["fc_w"] = model.fc.weight.detach().numpy()
    arrays["fc_b"] = model.fc.bias.detach().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez(path, **arrays)
