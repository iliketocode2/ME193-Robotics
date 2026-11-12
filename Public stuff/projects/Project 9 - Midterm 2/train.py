"""Train FingerNet on the collected images.

    python train.py                  augmentation on -> models/finger_cnn.pt and models/finger_cnn.npz
    python train.py --no-augment     same network without augmentation -> models/finger_cnn_noaug.pt,
                                     to measure how much augmentation helps
    python train.py --epochs 60      train longer

Writes plots to plots/ and a summary to results/.
"""

import argparse
import copy
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import config
import dataset
import npnet
from model import FingerNet, count_params, export_numpy, save_checkpoint


def evaluate(model, x, y, loss_fn, batch=256):
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits.append(model(dataset.standardize(x[i:i + batch])))
    logits = torch.cat(logits)
    return loss_fn(logits, y).item(), (logits.argmax(1) == y).float().mean().item(), logits.argmax(1)


def confusion(y_true, y_pred, n=len(config.CLASSES)):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)


def plot_history(h, path, title):
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.plot(ep, h["train_loss"], label="training (augmented)" if h["augment"] else "training")
    a.plot(ep, h["val_loss"], label="validation")
    a.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    a.legend(frameon=False)
    b.plot(ep, np.array(h["train_acc"]) * 100, label="training")
    b.plot(ep, np.array(h["val_acc"]) * 100, label="validation")
    b.set(xlabel="epoch", ylabel="% images correct", title="Accuracy", ylim=(0, 101))
    b.legend(frameon=False)
    for ax in (a, b):
        style(ax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(m, path, title):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(m, cmap="Blues")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set(xlabel="network's answer", ylabel="true fingers", title=title,
           xticks=range(len(config.CLASSES)), yticks=range(len(config.CLASSES)))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_augmentations(x, y, path, per_class=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    fig, axes = plt.subplots(len(config.CLASSES), per_class + 1, figsize=(1.4 * (per_class + 1), 1.45 * 6))
    for r, c in enumerate(config.CLASSES):
        idx = (y == c).nonzero().flatten()
        for a in axes[r]:
            a.axis("off")
        if len(idx) == 0:
            continue
        img = x[idx[0]:idx[0] + 1]
        axes[r, 0].imshow(img[0, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title(f"{c}: original", fontsize=8)
        aug = dataset.augment(img.repeat(per_class, 1, 1, 1), gen=gen)
        for k in range(per_class):
            axes[r, k + 1].imshow(aug[k, 0], cmap="gray", vmin=0, vmax=1)
    fig.suptitle("Augmentation: each row is one photo and five random altered copies")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_filters(model, path):
    w = model.convs[0].weight.detach()[:, 0]
    fig, axes = plt.subplots(1, w.shape[0], figsize=(1.3 * w.shape[0], 1.6))
    for k, a in enumerate(axes):
        a.imshow(w[k], cmap="RdBu_r", vmin=-w.abs().max(), vmax=w.abs().max())
        a.set_title(f"filter {k + 1}", fontsize=8)
        a.axis("off")
    fig.suptitle("conv1's 8 learned 5 x 5 filters (red = positive weight, blue = negative)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_feature_maps(model, img, label, path):
    with torch.no_grad():
        logits, maps = model(dataset.standardize(img[None]), return_maps=True)
    probs = torch.softmax(logits, 1)[0]
    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(4, 16, height_ratios=[2.4, 1, 1, 1], hspace=0.45, top=0.86, bottom=0.03)
    ax = fig.add_subplot(gs[0, :3])
    ax.imshow(img[0], cmap="gray")
    ax.set_title(f"input ({label} finger{'' if label == 1 else 's'})", fontsize=9)
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 4:])
    ax.bar(config.CLASSES, probs.numpy() * 100, color="#4c78a8")
    ax.set(ylabel="%", title="softmax output", ylim=(0, 100))
    style(ax)
    for row, (m, name) in enumerate(zip(maps, ["conv1 + pool: 8 maps, 32 x 32", "conv2 + pool: 16 maps, 16 x 16",
                                               "conv3 + pool: 32 maps, 8 x 8 (first 16)"]), start=1):
        for k in range(min(16, m.shape[1])):
            a = fig.add_subplot(gs[row, k])
            a.imshow(m[0, k], cmap="magma")
            a.axis("off")
            if k == 0:
                a.set_title(name, fontsize=8, loc="left")
    fig.suptitle("What each layer passes on (bright = a strong ReLU output)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_mistakes(x, y, pred, path, n=12):
    wrong = (pred != y).nonzero().flatten()[:n]
    if len(wrong) == 0:
        return
    fig, axes = plt.subplots(1, len(wrong), figsize=(1.5 * len(wrong), 1.9), squeeze=False)
    for a, i in zip(axes[0], wrong):
        a.imshow(x[i, 0], cmap="gray")
        a.set_title(f"true {y[i].item()}\nsaid {pred[i].item()}", fontsize=8)
        a.axis("off")
    fig.suptitle("Validation images the network got wrong")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def numpy_check(model, x, n=20):
    """Largest difference between PyTorch's and npnet.py's probabilities on n images."""
    weights = npnet.load_weights(config.NUMPY_WEIGHTS_PATH)
    model.eval()
    worst = 0.0
    with torch.no_grad():
        for i in range(min(n, len(x))):
            xs = dataset.standardize(x[i:i + 1])
            pt = torch.softmax(model(xs), 1)[0].numpy()
            npy = npnet.forward(weights, xs[0, 0].numpy())
            worst = max(worst, float(np.abs(pt - npy).max()))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    augment = not args.no_augment
    tag = "" if augment else "_noaug"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    items = dataset.find_images(args.data)
    if not items:
        raise SystemExit(f"No images in {args.data}. Run collect.py first.")
    train_items, val_items = dataset.split(items, seed=args.seed)
    xt, yt = dataset.load_arrays(train_items)
    xv, yv = dataset.load_arrays(val_items)
    per_class = np.bincount(yt.numpy(), minlength=len(config.CLASSES))
    print(f"{len(xt)} training images {per_class.tolist()}, {len(xv)} validation images "
          f"{np.bincount(yv.numpy(), minlength=len(config.CLASSES)).tolist()}")

    model = FingerNet()
    print(f"FingerNet has {count_params(model):,} weights. Augmentation {'on' if augment else 'off'}.")
    class_w = torch.tensor(per_class.sum() / np.maximum(per_class, 1) / len(per_class), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    val_loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "augment": augment}
    best = (-1.0, float("inf"), None, 0)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(xt), generator=gen)
        tot_loss = tot_right = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb, yb = xt[idx], yt[idx]
            if augment:
                xb = dataset.augment(xb, gen=gen)
            logits = model(dataset.standardize(xb))
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()          # backpropagation: the chain rule through every layer
            opt.step()               # gradient descent step (Adam adjusts the step size per weight)
            tot_loss += loss.item() * len(idx)
            tot_right += (logits.argmax(1) == yb).sum().item()
        sched.step()
        v_loss, v_acc, _ = evaluate(model, xv, yv, val_loss_fn)
        hist["train_loss"].append(tot_loss / len(xt))
        hist["train_acc"].append(tot_right / len(xt))
        hist["val_loss"].append(v_loss)
        hist["val_acc"].append(v_acc)
        if v_acc > best[0] or (v_acc == best[0] and v_loss < best[1]):
            best = (v_acc, v_loss, copy.deepcopy(model.state_dict()), ep)
        print(f"epoch {ep:3d}  train loss {hist['train_loss'][-1]:.3f} acc {hist['train_acc'][-1]:.1%}   "
              f"val loss {v_loss:.3f} acc {v_acc:.1%}")

    model.load_state_dict(best[2])
    v_loss, v_acc, v_pred = evaluate(model, xv, yv, val_loss_fn)
    cm = confusion(yv.numpy(), v_pred.numpy())
    print(f"\nBest epoch {best[3]}: validation accuracy {v_acc:.1%} ({int(cm.trace())}/{cm.sum()}), "
          f"{time.time() - t0:.0f} s of training")
    print("Confusion matrix (rows = true fingers, columns = network's answer):")
    print(cm)

    config.PLOTS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    model_path = config.MODELS_DIR / f"finger_cnn{tag}.pt"
    save_checkpoint(model_path, model, classes=config.CLASSES, val_acc=v_acc, best_epoch=best[3],
                    history=hist, augment=augment)
    label = "with augmentation" if augment else "without augmentation"
    plot_history(hist, config.PLOTS_DIR / f"training_curves{tag}.png", f"FingerNet training, {label}")
    plot_confusion(cm, config.PLOTS_DIR / f"confusion{tag}.png", f"Validation, {v_acc:.0%} correct")
    plot_mistakes(xv, yv, v_pred, config.PLOTS_DIR / f"mistakes{tag}.png")
    summary = {"augment": augment, "val_acc": v_acc, "val_loss": v_loss, "best_epoch": best[3],
               "epochs": args.epochs, "n_train": len(xt), "n_val": len(xv), "params": count_params(model),
               "confusion": cm.tolist()}

    if augment:
        export_numpy(model)
        summary["numpy_max_prob_diff"] = numpy_check(model, xv)
        print(f"npnet.py matches PyTorch to within {summary['numpy_max_prob_diff']:.1e} (probability)")
        plot_augmentations(xt, yt, config.PLOTS_DIR / "augmentation_examples.png")
        plot_filters(model, config.PLOTS_DIR / "conv1_filters.png")
        i = int((yv == 3).nonzero().flatten()[0]) if (yv == 3).any() else 0
        plot_feature_maps(model, xv[i], int(yv[i]), config.PLOTS_DIR / "feature_maps.png")
    (config.RESULTS_DIR / f"train_summary{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {model_path.name}, plots in plots/, summary in results/")


if __name__ == "__main__":
    main()
