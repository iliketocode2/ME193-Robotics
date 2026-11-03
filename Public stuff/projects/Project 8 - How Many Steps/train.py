"""Train the step network on labeled sessions.

    python train.py                     all six IMU channels -> models/step_cnn_all.pt
    python train.py --channels accel    accelerometer only   -> models/step_cnn_accel.pt
"""

import argparse
import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import config
import dataset
import stepcount
from model import StepNet, save_checkpoint


def tune_threshold(model, ckpt, sessions):
    probs = [stepcount.session_probs(model, ckpt, s) for s in sessions]
    best = (0.5, -1.0)
    for th in np.arange(0.2, 0.91, 0.05):
        hits = fps = misses = 0
        for s, (times, p) in zip(sessions, probs):
            h, f, m = stepcount.match_events(stepcount.detect_steps(times, p, th), s.t[s.events > 0])
            hits, fps, misses = hits + h, fps + f, misses + m
        f1 = 2 * hits / max(2 * hits + fps + misses, 1)
        if f1 > best[1]:
            best = (round(float(th), 2), f1)
    return best


def plot_history(history, path, title):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(epochs, history["train_loss"], label="train")
    a.plot(epochs, history["val_loss"], label="validation")
    a.set(xlabel="epoch", ylabel="BCE loss", title="Loss")
    a.legend(frameon=False)
    b.plot(epochs, np.array(history["val_acc"]) * 100, color="#30a46c")
    b.set(xlabel="epoch", ylabel="% windows correct", title="Validation accuracy")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train(channel_set="all", epochs=30, seed=0, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sessions = {s.name: s for s in dataset.load_sessions()}
    split = dataset.split_sessions(sessions.values(), seed)
    if not split["train"] or not split["val"]:
        raise SystemExit("Need at least 2 labeled sessions to train (one is held out for validation). "
                         "Label a session with label_tool.py.")
    train_s = [sessions[n] for n in split["train"]]
    val_s = [sessions[n] for n in split["val"]]
    channels = config.CHANNEL_SETS[channel_set]
    mean, std = dataset.norm_stats(train_s)

    xtr, ytr = dataset.build_arrays(train_s, mean, std, channels)
    xva, yva = dataset.build_arrays(val_s, mean, std, channels)
    xtr, ytr, xva, yva = map(torch.from_numpy, (xtr, ytr, xva, yva))
    pos = float(ytr.mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3)))

    model = StepNet(len(channels))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = (np.inf, None)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xtr))
        total = 0.0
        for idx in order.split(256):
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(b) for b in xva.split(2048)])
            val_loss = loss_fn(logits, yva).item()
            val_acc = ((logits > 0).float() == yva).float().mean().item()
        history["train_loss"].append(total / len(xtr))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_loss < best[0]:
            best = (val_loss, copy.deepcopy(model.state_dict()))
        if not quiet:
            print(f"epoch {epoch + 1:2d}  train {total / len(xtr):.4f}  val {val_loss:.4f}  acc {val_acc:.1%}")

    model.load_state_dict(best[1])
    model.eval()
    ckpt = {"channels": channels, "channel_set": channel_set, "mean": mean, "std": std,
            "window": config.WINDOW, "center": config.CENTER, "split": split, "history": history}
    ckpt["threshold"], ckpt["val_f1"] = tune_threshold(model, ckpt, val_s)
    path = config.MODELS_DIR / f"step_cnn_{channel_set}.pt"
    save_checkpoint(path, model, **ckpt)
    plot_history(history, config.PLOTS_DIR / f"training_curves_{channel_set}.png",
                 f"Step CNN ({channel_set} channels)")
    if not quiet:
        print(f"saved {path.name}: threshold {ckpt['threshold']}, validation step F1 {ckpt['val_f1']:.3f}, "
              f"{len(split['train'])}/{len(split['val'])}/{len(split['test'])} train/val/test sessions")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", choices=list(config.CHANNEL_SETS), default="all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.channels, args.epochs, args.seed)
