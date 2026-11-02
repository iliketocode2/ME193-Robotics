"""Render the end-to-end explainer animation to media/how_it_works.mp4.

    python explainer.py                         uses data/sample/walk_project7 (real wearable data)
    python explainer.py --session walk_2026...  uses one of your labeled sessions instead

Everything shown is computed here: real IMU windows, the real StepNet
architecture and its activations (forward hooks), real gradients from
loss.backward(), a short real training run for the loss and weight updates,
and the debounced step detector. If models/step_cnn_all.pt exists, its trained
weights drive the forward-pass and step-detection scenes.
"""

import argparse
import subprocess

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from torch import nn

import config
import dataset
import stepcount
from model import StepNet, load_checkpoint

FPS = 20
BG, PANEL, INK, MUTED = "#0f1115", "#1a1d24", "#eceef2", "#8b93a1"
CH_COLORS = ["#ff6b6b", "#51cf66", "#4dabf7", "#ffa94d", "#cc5de8", "#20c997"]
ACCENT, GOOD = "#ffa94d", "#51cf66"
SCENES = [("Raw IMU signals", 7), ("Windowing", 8), ("The network", 7), ("Forward pass", 9),
          ("Loss", 8), ("Backpropagation", 11), ("Step detection", 10), ("The game", 7)]
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#333844"})


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Story:
    def __init__(self, session_name):
        folder = config.SESSIONS_DIR if session_name else config.SAMPLE_DIR
        self.session = dataset.load_session(session_name or "walk_project7", folder)
        s = self.session
        self.true_steps = s.t[s.events > 0] - s.t[0]
        self.t = s.t - s.t[0]

        ckpt_path = config.MODELS_DIR / "step_cnn_all.pt"
        self.trained_model = ckpt_path.exists()
        self.train_demo()
        if self.trained_model:
            self.model, self.ckpt = load_checkpoint(ckpt_path)
        self.capture_forward()
        times, self.probs = stepcount.session_probs(self.model, self.ckpt, s)
        self.prob_t = times - s.t[0]
        self.pred_steps = stepcount.detect_steps(self.prob_t, self.probs, self.ckpt["threshold"])

    def train_demo(self):
        s = self.session
        torch.manual_seed(0)
        mean, std = dataset.norm_stats([s])
        ckpt = {"channels": config.CHANNEL_SETS["all"], "mean": mean, "std": std, "window": config.WINDOW,
                "center": config.CENTER, "threshold": 0.5}
        x, y = dataset.build_arrays([s], mean, std, ckpt["channels"])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        model = StepNet(6)
        layers = [model.features[0], model.features[4], model.features[8], model.head[2], model.head[4]]
        pos = float(y.mean())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3)))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        self.losses, self.grads, self.kernels = [], [], []
        snap_at = {0, 5, 15, 40, 80, 149}
        for it in range(150):
            model.train()
            idx = torch.randint(0, len(x), (128,))
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            self.grads.append([float(l.weight.grad.norm()) for l in layers])
            if it in snap_at:
                self.kernels.append((it, model.features[0].weight.detach()[:, :, :].mean(1).numpy().copy()))
            opt.step()
            self.losses.append(loss.item())
        model.eval()
        self.model, self.ckpt = model, ckpt
        self.demo_xy = (x, y)

    def capture_forward(self):
        s, ck = self.session, self.ckpt
        x = dataset.normalize(s.x, ck["mean"], ck["std"])[:, ck["channels"]]
        centers = np.arange(0, len(s.t), 2)
        wins = torch.from_numpy(dataset.make_windows(x, centers, ck["window"], ck["center"]))
        labels = dataset.window_targets(s.events, centers)
        with torch.no_grad():
            p = torch.sigmoid(self.model(wins)).numpy()
        pick = int(np.argmax(np.where(labels > 0, p, -1))) if labels.any() else int(np.argmax(p))
        self.win_center = centers[pick]
        self.win_label = float(labels[pick])
        acts = {}
        hooks = [m.register_forward_hook(lambda m, i, o, k=k: acts.__setitem__(k, o.detach()[0].numpy()))
                 for k, m in {"c1": self.model.features[2], "p1": self.model.features[3],
                              "c2": self.model.features[6], "p2": self.model.features[7],
                              "c3": self.model.features[10], "d1": self.model.head[3]}.items()]
        inp = wins[pick:pick + 1]
        with torch.no_grad():
            logit = self.model(inp)
        for h in hooks:
            h.remove()
        self.window = inp[0].numpy()
        self.prob = float(torch.sigmoid(logit))
        self.loss = float(nn.functional.binary_cross_entropy(torch.tensor([self.prob]),
                                                             torch.tensor([self.win_label])))
        self.layers = [("Input", "6 x 100", self.window), ("Conv k7 + ReLU", "16 x 100", acts["c1"]),
                       ("Max pool", "16 x 50", acts["p1"]), ("Conv k5 + ReLU", "32 x 50", acts["c2"]),
                       ("Max pool", "32 x 25", acts["p2"]), ("Conv k5 + ReLU", "32 x 25", acts["c3"]),
                       ("Dense + ReLU", "32", acts["d1"][:, None]), ("Output", "1", np.array([[self.prob]]))]
        self.n_params = sum(p.numel() for p in self.model.parameters())


def title(fig, n, name, sub):
    fig.text(0.04, 0.93, f"{n}. {name}", fontsize=26, weight="bold", color=INK)
    fig.text(0.04, 0.885, sub, fontsize=13, color=MUTED)


def footer(fig, k):
    for i, (name, _) in enumerate(SCENES):
        x = 0.04 + i * 0.118
        on = i == k
        fig.add_artist(Rectangle((x, 0.035), 0.11, 0.006, transform=fig.transFigure,
                                     color=ACCENT if on else "#333844"))
        fig.text(x, 0.05, name, fontsize=8.5, color=INK if on else MUTED)


def signal_axes(fig, story, rect, reveal=1.0, window=None):
    ax = fig.add_axes(rect, facecolor=PANEL)
    s = story.session
    end = story.t[-1] * reveal
    keep = story.t <= end
    off = 0
    for c in range(6):
        v = s.x[:, c]
        v = (v - v.mean()) / (v.std() + 1e-9)
        ax.plot(story.t[keep], v[keep] * 0.35 - off, color=CH_COLORS[c], lw=0.9)
        ax.text(-0.3, -off, config.CHANNELS[c], color=CH_COLORS[c], ha="right", va="center", fontsize=10)
        off += 1.6
    for st in story.true_steps[story.true_steps <= end]:
        ax.plot([st], [1.0], marker="v", color=INK, ms=6)
    if window is not None:
        a, b = window
        ax.add_patch(Rectangle((a, -off + 0.6), b - a, off + 0.9, fill=True, color=ACCENT, alpha=0.18))
        ax.add_patch(Rectangle((a, -off + 0.6), b - a, off + 0.9, fill=False, ec=ACCENT, lw=2))
        c = a + config.CENTER / config.SAMPLE_HZ
        ax.plot([c, c], [-off + 0.6, 1.5], color=ACCENT, lw=1.5, ls="--")
    ax.set_xlim(-0.2, story.t[-1])
    ax.set_ylim(-off + 0.5, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    return ax


def scene_raw(fig, story, u):
    title(fig, 1, "Raw IMU signals", "The MPU-6050 on the shoe streams 3 accelerations + 3 rotation rates, 100 times a second."
          "  ▼ = a step")
    signal_axes(fig, story, [0.07, 0.14, 0.88, 0.68], reveal=0.08 + 0.92 * ease(u))


def scene_window(fig, story, u):
    title(fig, 2, "Windowing", "A 1-second window (100 samples x 6 channels) slides along. The question for each window:"
          " is there a step at sample 70?")
    start = (story.t[-1] - 1.2) * ease(u)
    signal_axes(fig, story, [0.07, 0.14, 0.55, 0.68], window=(start, start + 1.0))
    s = story.session
    i0 = int(start * config.SAMPLE_HZ)
    x = dataset.normalize(s.x, story.ckpt["mean"], story.ckpt["std"])
    w = x[i0:i0 + config.WINDOW].T
    ax = fig.add_axes([0.68, 0.38, 0.28, 0.4], facecolor=PANEL)
    ax.imshow(w, aspect="auto", cmap="magma", vmin=-2.5, vmax=2.5)
    ax.axvline(config.CENTER, color=ACCENT, lw=2, ls="--")
    ax.set_yticks(range(6), config.CHANNELS)
    ax.set_xlabel("sample in window")
    ax.set_title("what the network sees (normalized)", color=INK, fontsize=11)
    near = np.abs(story.true_steps - (start + config.CENTER / config.SAMPLE_HZ)) <= config.EVENT_TOLERANCE / 100
    lab = 1 if near.any() else 0
    fig.text(0.68, 0.27, f"label = {lab}", fontsize=24, weight="bold", color=GOOD if lab else MUTED)
    fig.text(0.68, 0.22, "step within 70 ms of sample 70" if lab else "no step at sample 70", fontsize=12, color=MUTED)


def block_positions():
    xs = np.linspace(0.07, 0.86, 8)
    heights = [0.18, 0.34, 0.3, 0.42, 0.36, 0.36, 0.22, 0.08]
    return [(x, 0.5 - h / 2, 0.075, h) for x, h in zip(xs, heights)]


def draw_blocks(fig, story, show=8, active=-1, heat=False, grads=None):
    pos = block_positions()
    for i, ((name, shape, act), (x, y, w, h)) in enumerate(zip(story.layers, pos)):
        if i >= show:
            break
        on = i <= active
        color = ACCENT if i == active else ("#2d3340" if not on else "#3b4a5e")
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                          transform=fig.transFigure, fc=color, ec=ACCENT if on else "#4a5160",
                                          lw=1.5, alpha=0.95))
        if heat and on:
            ax = fig.add_axes([x + 0.006, y + 0.008, w - 0.012, h - 0.016])
            ax.set_zorder(5)
            a = act if act.ndim == 2 else act[None, :]
            ax.imshow(a, aspect="auto", cmap="magma" if i < 7 else "viridis",
                      vmin=0 if i else -2.5, vmax=max(float(np.abs(a).max()), 1e-6) if i else 2.5)
            ax.axis("off")
        fig.text(x + w / 2, y + h + 0.03, name, ha="center", fontsize=10, color=INK)
        fig.text(x + w / 2, y - 0.04, shape, ha="center", fontsize=10, color=MUTED)
        if i < len(pos) - 1 and i + 1 < show:
            fig.add_artist(FancyArrowPatch((x + w + 0.004, 0.5), (pos[i + 1][0] - 0.004, 0.5),
                                               transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
                                               color=ACCENT if i < active else "#4a5160", lw=1.5))
        if grads is not None and grads[i] is not None:
            g = grads[i]
            fig.add_artist(Rectangle((x, y - 0.12), w * g, 0.02, transform=fig.transFigure, color="#ff6b6b"))
    return pos


def scene_network(fig, story, u):
    title(fig, 3, "The network", f"A small 1D convolutional neural net: {story.n_params:,} learnable weights."
          "  Convolutions slide little pattern detectors along time.")
    draw_blocks(fig, story, show=1 + int(ease(u * 1.3) * 7.99))


def scene_forward(fig, story, u):
    title(fig, 4, "Forward pass", "One window flows left to right. Each layer's real activations light up"
          " (brighter = stronger response).")
    k = int(ease(u * 1.15) * 7.99)
    draw_blocks(fig, story, active=k, heat=True)
    if k == 7:
        fig.text(0.5, 0.15, f"P(step at sample 70) = {story.prob:.2f}", ha="center", fontsize=22, weight="bold",
                 color=GOOD if story.prob > 0.5 else ACCENT)


def scene_loss(fig, story, u):
    title(fig, 5, "Loss", "Training compares the prediction with the label. Binary cross-entropy is small when the"
          " network is confidently right.")
    p, y = story.prob, story.win_label
    fig.text(0.07, 0.7, f"prediction  p = {p:.2f}", fontsize=20)
    fig.text(0.07, 0.62, f"label         y = {y:.0f}", fontsize=20)
    fig.text(0.07, 0.5, r"$L = -[\,y \log p + (1-y)\log(1-p)\,]$", fontsize=22)
    fig.text(0.07, 0.4, f"L = {story.loss:.3f}", fontsize=24, weight="bold", color=ACCENT)
    ax = fig.add_axes([0.5, 0.18, 0.44, 0.6], facecolor=PANEL)
    n = max(2, int(len(story.losses) * ease(u)))
    ax.plot(np.arange(n), story.losses[:n], color=ACCENT, lw=1.5)
    ax.set_xlim(0, len(story.losses))
    ax.set_ylim(0, max(story.losses) * 1.05)
    ax.set_xlabel("training step (batch of 128 windows)")
    ax.set_ylabel("average loss")
    ax.set_title("loss falls as the network learns", color=INK, fontsize=12)


def scene_backprop(fig, story, u):
    title(fig, 6, "Backpropagation", "The loss's gradient flows backward through every layer (red bars = real gradient"
          " size), then each weight takes a small step downhill.")
    g = np.array(story.grads[0])
    g = g / g.max()
    reach = 7 - int(ease(min(u * 2, 1)) * 7.99)
    per_block = [None, g[0], None, g[1], None, g[2], g[3], g[4]]
    shown = [v if (v is not None and i >= reach) else None for i, v in enumerate(per_block)]
    pos = draw_blocks(fig, story, active=-1, grads=shown)
    for i in range(7, max(reach, 0), -1):
        a, b = pos[i], pos[i - 1]
        fig.add_artist(FancyArrowPatch((a[0] - 0.004, 0.3), (b[0] + b[2] + 0.004, 0.3), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=14, color="#ff6b6b", lw=1.5))
    if u > 0.5:
        v = ease((u - 0.5) / 0.5)
        j = min(int(v * (len(story.kernels) - 1) + 0.5), len(story.kernels) - 1)
        it, k = story.kernels[j]
        ax = fig.add_axes([0.07, 0.765, 0.24, 0.075])
        ax.imshow(k, aspect="auto", cmap="coolwarm", vmin=-0.4, vmax=0.4)
        ax.set_title(f"first-layer weights after {it} updates", color=INK, fontsize=10, pad=3)
        ax.set_yticks([])
        ax.set_xticks([])
        fig.text(0.36, 0.79, "w  ←  w  −  learning rate × ∂L/∂w", fontsize=17, color=INK)


def scene_detect(fig, story, u):
    title(fig, 7, "Step detection", "The network scores every window. Each bump above the threshold becomes one step"
          " (debounced: at least 0.25 s apart).")
    end = story.t[-1] * (0.05 + 0.95 * ease(u))
    ax = fig.add_axes([0.07, 0.48, 0.88, 0.3], facecolor=PANEL)
    a = np.linalg.norm(story.session.x[:, :3], axis=1)
    keep = story.t <= end
    ax.plot(story.t[keep], a[keep], color=CH_COLORS[2], lw=0.9)
    ax.set_xlim(0, story.t[-1])
    ax.set_ylabel("|accel| (g)")
    ax.set_xticks([])
    ax2 = fig.add_axes([0.07, 0.16, 0.88, 0.28], facecolor=PANEL)
    k = story.prob_t <= end
    ax2.plot(story.prob_t[k], story.probs[k], color=ACCENT, lw=1.4)
    ax2.axhline(story.ckpt["threshold"], color=MUTED, ls="--", lw=1)
    found = story.pred_steps[story.pred_steps <= end]
    ax2.scatter(found, np.interp(found, story.prob_t, story.probs), color=GOOD, s=40, zorder=3)
    for st in story.true_steps[story.true_steps <= end]:
        ax2.plot([st], [1.08], marker="v", color=INK, ms=5)
    ax2.set_xlim(0, story.t[-1])
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("P(step)")
    fig.text(0.8, 0.83, f"steps: {len(found)}", fontsize=26, weight="bold", color=GOOD)
    if not story.trained_model:
        fig.text(0.07, 0.8, "illustration: this network was trained on this same clip, so it matches by heart."
                 " Real accuracy comes from evaluate.py on unseen walks.", fontsize=10, color=ACCENT)


def scene_game(fig, story, u):
    title(fig, 8, "The game", "live.py sends steps, cadence and distance (steps x stride) over UDP to a LÖVE game:"
          " every step moves the walker.")
    n = len(story.pred_steps)
    shown = int(ease(u) * n)
    stride = 0.7
    ax = fig.add_axes([0.07, 0.2, 0.86, 0.5], facecolor="#9ccdf2")
    ax.axhspan(-1, 0, color="#5b8c55")
    ax.axhspan(0, 0.15, color="#dcc59e")
    dist = shown * stride
    for m in range(0, int(n * stride) + 12):
        ax.plot([m, m], [0, 0.06 if m % 10 else 0.15], color="#6b5a43", lw=1)
    for i in range(shown):
        ax.plot(i * stride, 0.07 + (0.03 if i % 2 else -0.03), marker="o", color="#6b4f35", ms=5)
    swing = 0.25 if shown % 2 else -0.25
    ax.plot([dist - swing, dist, dist + swing], [0.15, 0.45, 0.15], color="#2b2f3a", lw=4)
    ax.plot([dist, dist + 0.05], [0.45, 0.85], color="#3e63dd", lw=5)
    ax.plot([dist + swing * 0.8, dist + 0.04, dist - swing * 0.8], [0.5, 0.78, 0.5], color="#f2c6a0", lw=3)
    ax.plot(dist + 0.07, 1.0, marker="o", color="#f2c6a0", ms=16)
    ax.set_xlim(dist - 6, dist + 10)
    ax.set_ylim(-0.3, 1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(0.07, 0.76, f"{shown} steps   x   {stride} m   =   {dist:.1f} m", fontsize=24, weight="bold", color=INK)


DRAW = [scene_raw, scene_window, scene_network, scene_forward, scene_loss, scene_backprop, scene_detect, scene_game]


def render(story, path, still_path):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    frames = [(k, f / (sec * FPS - 1)) for k, (_, sec) in enumerate(SCENES) for f in range(sec * FPS)]
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = FFMpegWriter(fps=FPS, bitrate=2400, extra_args=["-pix_fmt", "yuv420p"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(path), dpi=100):
        for k, u in frames:
            fig.clf()
            DRAW[k](fig, story, u)
            footer(fig, k)
            if k == 3 and abs(u - 1.0) < 1e-9:
                fig.savefig(still_path, facecolor=BG, dpi=100)
            writer.grab_frame(facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="a labeled session in data/sessions (default: the bundled sample)")
    args = ap.parse_args()
    story = Story(args.session)
    out = config.MEDIA_DIR / "how_it_works.mp4"
    render(story, out, config.MEDIA_DIR / "architecture.png")
    gif = config.MEDIA_DIR / "how_it_works_preview.gif"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", "22", "-t", "9",
                    "-i", str(out), "-vf", "fps=8,scale=640:-1:flags=lanczos", str(gif)], check=True)
    print(f"wrote {out.name}, {gif.name} and architecture.png"
          + ("" if story.trained_model else " (no trained model yet: weights come from a short demo training run)"))


if __name__ == "__main__":
    main()
