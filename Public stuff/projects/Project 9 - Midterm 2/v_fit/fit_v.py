"""Fit the V data with gradient descent in Python, the same math as V Fit Gradient Descent.xlsx.

Three models, all trained by hand-written gradient descent (numpy only, no autograd):
    1. a straight line            y_hat = m*u + b
    2. one ReLU neuron            y_hat = ReLU(m*u + b)
    3. a layer of two ReLU neurons  y_hat = v1*ReLU(w1*u + c1) + v2*ReLU(w2*u + c2) + d

u is x normalized to mean 0 and standard deviation 1, as in the spreadsheet. The starting weights and learning
rates match the spreadsheet, so the first rows printed here should match its 5 Iterations, ReLU Neuron and
ReLU Layer tabs.

    python v_fit/fit_v.py     -> prints the first 5 iterations and writes v_fit/v_fit.png and v_fit/v_loss.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EPOCHS = 2000
SHOW_LOSS = 60    # the loss plot shows this many iterations; nothing changes after that


def load():
    data = np.loadtxt(HERE / "v_data.tsv", skiprows=1)
    x, y = data[:, 1], data[:, 2]
    return x, y, x.mean(), x.std()


def relu(z):
    return np.maximum(0.0, z)


def train_line(u, y, m=15.0, b=0.0, lr=0.2, epochs=EPOCHS):
    n, losses = len(u), []
    for _ in range(epochs + 1):
        e = m * u + b - y                       # error of every point
        losses.append(np.mean(e ** 2))          # MSE loss
        dm = 2 / n * np.sum(e * u)              # chain rule: d(y_hat)/dm = u
        db = 2 / n * np.sum(e)                  # d(y_hat)/db = 1
        m, b = m - lr * dm, b - lr * db
    return (m, b), losses


def train_relu(u, y, m=15.0, b=0.0, lr=0.2, epochs=EPOCHS):
    n, losses = len(u), []
    for _ in range(epochs + 1):
        z = m * u + b
        on = (z > 0).astype(float)              # ReLU's slope: 1 where the neuron is on, 0 where it is off
        e = relu(z) - y
        losses.append(np.mean(e ** 2))
        dm = 2 / n * np.sum(e * on * u)
        db = 2 / n * np.sum(e * on)
        m, b = m - lr * dm, b - lr * db
    return (m, b), losses


def layer(p, u):
    w1, c1, w2, c2, v1, v2, d = p
    return v1 * relu(w1 * u + c1) + v2 * relu(w2 * u + c2) + d


def train_layer(u, y, p=(0.5, 0.2, -0.8, 0.1, 2.0, 3.0, 0.0), lr=0.01, epochs=EPOCHS):
    w1, c1, w2, c2, v1, v2, d = p
    n, losses = len(u), []
    for _ in range(epochs + 1):
        z1, z2 = w1 * u + c1, w2 * u + c2
        h1, h2 = relu(z1), relu(z2)
        on1, on2 = (z1 > 0).astype(float), (z2 > 0).astype(float)
        e = v1 * h1 + v2 * h2 + d - y
        losses.append(np.mean(e ** 2))
        g = 2 / n * e                           # dL/d(y_hat) for every point
        # backpropagation: the output weights see the neuron outputs, the hidden weights see v * ReLU' * input
        grads = (np.sum(g * v1 * on1 * u), np.sum(g * v1 * on1),
                 np.sum(g * v2 * on2 * u), np.sum(g * v2 * on2),
                 np.sum(g * h1), np.sum(g * h2), np.sum(g))
        w1, c1, w2, c2, v1, v2, d = (a - lr * da for a, da in zip((w1, c1, w2, c2, v1, v2, d), grads))
    return (w1, c1, w2, c2, v1, v2, d), losses


def main():
    x, y, mean, std = load()
    u = (x - mean) / std

    (m, b), line_loss = train_line(u, y)
    (rm, rb), relu_loss = train_relu(u, y)
    p, layer_loss = train_layer(u, y)

    print("First 5 iterations (compare with the spreadsheet):")
    print(f"{'iter':>4}  {'line':>10}  {'1 ReLU':>10}  {'2 ReLU':>10}")
    for i in range(6):
        print(f"{i:>4}  {line_loss[i]:10.2f}  {relu_loss[i]:10.2f}  {layer_loss[i]:10.2f}")
    print(f"After {EPOCHS} iterations: line {line_loss[-1]:.2f}, 1 ReLU {relu_loss[-1]:.2f}, "
          f"2 ReLU {layer_loss[-1]:.2f}  (noise alone gives about 4)")
    w1, c1, w2, c2, v1, v2, d = p
    print(f"2 ReLU neurons turn on at x = {mean - c1 / w1 * std:.1f} and x = {mean - c2 / w2 * std:.1f} "
          "(the V's corner is at 60)")

    xs = np.linspace(20, 100, 400)
    us = (xs - mean) / std
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=14, color="0.55", label="V data (120 points)")
    ax.plot(xs, m * us + b, lw=2, label="straight line")
    ax.plot(xs, relu(rm * us + rb), lw=2, ls="--", label="1 ReLU neuron")
    ax.plot(xs, layer(p, us), lw=2.5, label="2 ReLU neurons")
    ax.set(xlabel="x", ylabel="y", title=f"Fits after {EPOCHS} iterations of gradient descent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "v_fit.png", dpi=130)

    fig, ax = plt.subplots(figsize=(8, 5))
    for losses, name in ((line_loss, "straight line"), (relu_loss, "1 ReLU neuron"), (layer_loss, "2 ReLU neurons")):
        ax.plot(losses[:SHOW_LOSS + 1], lw=2, marker="o", ms=3, label=name)
    ax.axvspan(0, 5, color="0.9", label="the 5 iterations in the spreadsheet")
    ax.set(yscale="log", xlabel="iteration (epoch)", ylabel="MSE loss (log scale)",
           title=f"Loss during the first {SHOW_LOSS} iterations (it stays flat after that)")
    ax.set_xlim(0, SHOW_LOSS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "v_loss.png", dpi=130)
    print("Wrote v_fit/v_fit.png and v_fit/v_loss.png")


if __name__ == "__main__":
    main()
