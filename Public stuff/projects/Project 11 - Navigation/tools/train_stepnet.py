"""Retrain Project 8's step network on clean, consistent labels; export it for the board.

Project 8's labels mixed two conventions (some sessions marked every footstep,
some every stride), so its network counted each differently. Here every
recording is labeled the same way by the foot tracker: one step per stride of
the instrumented foot, at the start of the stance that follows a real swing.
The network sees four mount-independent channels instead of raw axes.

Needs PyTorch (my_env_cpu):
    my_env_cpu/Scripts/python tools/train_stepnet.py                      Project 8 + simulated walks
    my_env_cpu/Scripts/python tools/train_stepnet.py --data p8 sim export.json   + walks from the board
    my_env_cpu/Scripts/python tools/train_stepnet.py --plots              also save label check plots

Writes unoq_app/python/models/stepnet.npz (copy unoq_app to the board) and
results/stepnet.md + results/stepnet_training.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "unoq_app" / "python")]

import sensor_data as sd        # noqa: E402
import stepnet_np               # noqa: E402

WINDOW, CENTER = 100, 70
TOL = 7                 # a window is positive if a step lies within 70 ms of its center sample
TRAIN_HOP = 2
LIVE_HOP = 4            # the board evaluates a window every 4 samples (25 per second)
MIN_GAP_S = 0.6         # never two strides of one foot closer than this (a brisk walking stride is ~0.9 s)
MATCH_S = 0.2           # a detected step counts as correct within 200 ms of a label
OUT = ROOT / "unoq_app" / "python" / "models" / "stepnet.npz"
RESULTS = ROOT / "results"


class StepNet(nn.Module):
    """Project 8's architecture, 4 input channels."""

    def __init__(self, in_channels=4, window=WINDOW):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32 * (window // 4), 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def windows(chans, centers):
    """(N, 4, WINDOW) windows whose sample CENTER is each center (zero-padded at the edges)."""
    pad = np.pad(chans, ((WINDOW, WINDOW), (0, 0)))
    idx = centers[:, None] + WINDOW - CENTER + np.arange(WINDOW)[None, :]
    return pad[idx].transpose(0, 2, 1).astype(np.float32)


def dataset(recs, mean, std, hop):
    xs, ys = [], []
    for r in recs:
        _, chans, _, _ = r.tracked()
        chans = (chans - mean) / std
        steps = np.round((sd.step_times(r) - r.t[0]) * 100).astype(int)
        centers = np.arange(0, len(r.t), hop)
        y = np.zeros(len(centers), np.float32)
        for s in steps:
            y[np.abs(centers - s) <= TOL] = 1.0
        xs.append(windows(chans, centers))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def detect(probs_fn, rec, threshold):
    """Run the live chain over a recording: windows every LIVE_HOP samples -> debouncer."""
    _, chans, _, _ = rec.tracked()
    centers = np.arange(0, len(rec.t), LIVE_HOP)
    probs = probs_fn(chans, centers)
    det = stepnet_np.StepDetector(threshold, MIN_GAP_S)
    out = [s for c, p in zip(centers, probs) if (s := det.update(rec.t[c], p)) is not None]
    if (s := det.update(np.inf, -1.0)) is not None:
        out.append(s)
    return np.array(out)


def match(pred, true):
    true = list(np.sort(true))
    hits = 0
    for p in np.sort(pred):
        if true:
            j = int(np.argmin(np.abs(np.array(true) - p)))
            if abs(true[j] - p) <= MATCH_S:
                true.pop(j)
                hits += 1
    return hits, len(pred) - hits, len(true)


def fold(conv, bn):
    s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
    w = conv.weight.detach().numpy() * s[:, None, None]
    b = (conv.bias.detach().numpy() - bn.running_mean.numpy()) * s + bn.bias.detach().numpy()
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["p8", "sim"])
    ap.add_argument("--val", nargs="+", default=["trial50_20261102-223545", "sim10", "sim11"],
                    help="recordings held out to pick the threshold and report counts")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    recs = [r for r in sd.load(args.data) if len(sd.step_times(r)) > 0 or r.activity == "still"]
    if args.plots:
        sd.check_plots(recs, RESULTS / "label_checks")
    val = [r for r in recs if r.name in args.val]
    train = [r for r in recs if r.name not in args.val]
    print(f"train: {len(train)} recordings, {sum(len(sd.step_times(r)) for r in train)} steps; "
          f"held out: {[r.name for r in val]}")

    allc = np.concatenate([r.tracked()[1] for r in train])
    mean, std = allc.mean(0), allc.std(0) + 1e-6
    xtr, ytr = dataset(train, mean, std, TRAIN_HOP)
    xva, yva = dataset(val, mean, std, TRAIN_HOP)
    pos = ytr.mean()
    print(f"{len(xtr)} training windows, {pos:.1%} positive")

    model = StepNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xv, yv = torch.from_numpy(xva), torch.from_numpy(yva)
    hist = {"train": [], "val": []}
    best = (1e9, None)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xv), yv).item()
        hist["train"].append(total / len(xt))
        hist["val"].append(vl)
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in model.state_dict().items()})
        print(f"epoch {epoch + 1:2d}  train {hist['train'][-1]:.3f}  val {vl:.3f}")
    model.load_state_dict(best[1])
    model.eval()

    def torch_probs(chans, centers):
        x = torch.from_numpy(windows((chans - mean) / std, centers))
        with torch.no_grad():
            return torch.sigmoid(torch.cat([model(b) for b in x.split(1024)])).numpy()

    # threshold: best step F1 on the held-out recordings, through the live chain
    best_t, best_f1 = 0.5, -1
    for th in np.arange(0.3, 0.91, 0.05):
        h = fp = ms = 0
        for r in val:
            a, b, c = match(detect(torch_probs, r, th), sd.step_times(r))
            h, fp, ms = h + a, fp + b, ms + c
        f1 = 2 * h / max(2 * h + fp + ms, 1)
        if f1 > best_f1:
            best_t, best_f1 = float(th), f1

    # export with batch norm folded in
    f = model.features
    c1w, c1b = fold(f[0], f[1])
    c2w, c2b = fold(f[4], f[5])
    c3w, c3b = fold(f[8], f[9])
    hd = model.head
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, c1_w=c1w, c1_b=c1b, c2_w=c2w, c2_b=c2b, c3_w=c3w, c3_b=c3b,
             f1_w=hd[2].weight.detach().numpy(), f1_b=hd[2].bias.detach().numpy(),
             f2_w=hd[4].weight.detach().numpy()[0], f2_b=hd[4].bias.detach().numpy()[0],
             mean=mean.astype(np.float32), std=std.astype(np.float32), window=WINDOW, center=CENTER,
             threshold=best_t, min_gap=MIN_GAP_S)

    # the numpy port must agree with PyTorch
    net = stepnet_np.StepNet(OUT)
    r = val[0]
    chans = r.tracked()[1]
    centers = np.arange(0, len(r.t), 37)[:40]
    tp = torch_probs(chans, centers)
    npp = np.array([net.prob(chans[max(0, c - CENTER):c - CENTER + WINDOW]) if c >= CENTER and c - CENTER + WINDOW <= len(chans)
                    else np.nan for c in centers])
    ok = ~np.isnan(npp)
    diff = np.abs(tp[ok] - npp[ok]).max()
    print(f"numpy vs torch: max difference {diff:.2e} over {ok.sum()} windows")
    assert diff < 1e-4, "numpy port disagrees with PyTorch"

    # report
    rows = []
    for r in recs:
        pred = detect(torch_probs, r, best_t)
        true = sd.step_times(r)
        h, fp, ms = match(pred, true)
        human = ""
        if r.source == "p8":
            meta = json.loads((sd.P8_DIR / f"{r.name}.json").read_text())
            human = str(meta.get("true_steps") or "")
        sim_true = str(len(r.truth_strides)) if r.truth_strides else ""
        rows.append((r.name, "held out" if r in val else "train", len(true), len(pred), h, fp, ms, human, sim_true))
    RESULTS.mkdir(exist_ok=True)
    lines = ["# Step network (retrained)", "",
             f"Threshold {best_t:.2f} (best held-out F1 {best_f1:.3f}), min gap {MIN_GAP_S} s, "
             f"numpy/PyTorch max difference {diff:.1e}.", "",
             "Counts are strides of the instrumented foot (one per two footsteps).", "",
             "| Recording | Set | Tracker labels | Network | Matched | Extra | Missed | Human footsteps (P8) | True strides (sim) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {' | '.join(str(c) for c in row)} |" for row in rows]
    (RESULTS / "stepnet.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist["train"], label="train")
    ax.plot(hist["val"], label="held out")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "stepnet_training.png", dpi=90)
    print(f"\nsaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
