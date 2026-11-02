"""The step network: a small 1D CNN over a 1 s multichannel IMU window."""

import torch
from torch import nn

import config


class StepNet(nn.Module):
    """Input (batch, channels, 100). Output one logit: is there a step at sample 70 of the window?"""

    def __init__(self, in_channels=6, window=config.WINDOW):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(32 * (window // 4), 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def save_checkpoint(path, model, **info):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), **info}, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = StepNet(len(ckpt["channels"]), ckpt["window"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt
