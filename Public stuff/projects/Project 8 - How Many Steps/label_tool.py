"""Mark steps on a plotted session.

    python label_tool.py                    newest session
    python label_tool.py walk_20261101-1200 a specific session

Left click: add a step   Right click: remove the nearest step
P: suggest this foot's landings (acceleration peaks)
M: add the other foot's steps halfway between consecutive marks
[ / ]: shift every mark 20 ms earlier / later (undo key-press reaction lag)
C: clear   S: save   Q: quit
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import config
import dataset

COLORS = ["#e5484d", "#30a46c", "#3e63dd"]


def suggest_landings(t, x):
    """Peaks of the acceleration magnitude, at most one per 0.6 s."""
    a = np.linalg.norm(x[:, :3], axis=1)
    a = np.convolve(a - a.mean(), np.ones(5) / 5, mode="same")
    floor = a.mean() + 0.5 * a.std()
    peaks = [i for i in range(1, len(a) - 1) if a[i] > floor and a[i] >= a[i - 1] and a[i] > a[i + 1]]
    keep = []
    for i in sorted(peaks, key=lambda i: -a[i]):
        if all(abs(i - j) > 0.6 * config.SAMPLE_HZ for j in keep):
            keep.append(i)
    return sorted(round(float(t[i]), 3) for i in keep)


class Labeler:
    def __init__(self, name):
        self.name = name
        self.session = dataset.load_session(name)
        self.meta = self.session.meta
        self.steps = sorted(self.meta.get("steps", []))
        self.saved = True
        t, x = self.session.t, self.session.x
        self.fig, (self.ax_a, self.ax_g) = plt.subplots(2, 1, sharex=True, figsize=(13, 6))
        for c in range(3):
            self.ax_a.plot(t, x[:, c], color=COLORS[c], lw=0.8, label=config.CHANNELS[c])
            self.ax_g.plot(t, x[:, 3 + c], color=COLORS[c], lw=0.8, label=config.CHANNELS[3 + c])
        self.ax_a.set_ylabel("acceleration (g)")
        self.ax_g.set_ylabel("rotation (deg/s)")
        self.ax_g.set_xlabel("time (s)")
        for ax in (self.ax_a, self.ax_g):
            ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
        self.lines = []
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    def redraw(self):
        for ln in self.lines:
            ln.remove()
        self.lines = [ax.axvline(s, color="black", lw=1, alpha=0.6) for s in self.steps
                      for ax in (self.ax_a, self.ax_g)]
        star = "" if self.saved else " *"
        self.fig.suptitle(f"{self.name} ({self.meta.get('activity')}): {len(self.steps)} steps{star}   "
                          "click add / right-click remove   P peaks  M midpoints  [ ] shift  C clear  S save")
        self.fig.canvas.draw_idle()

    def on_click(self, ev):
        if ev.inaxes is None or self.fig.canvas.toolbar.mode:
            return
        if ev.button == 1:
            self.steps = sorted(self.steps + [round(ev.xdata, 3)])
        elif ev.button == 3 and self.steps:
            self.steps.pop(int(np.argmin([abs(s - ev.xdata) for s in self.steps])))
        self.saved = False
        self.redraw()

    def on_key(self, ev):
        if ev.key == "p":
            self.steps = suggest_landings(self.session.t, self.session.x)
        elif ev.key == "m" and len(self.steps) > 1:
            gaps = np.diff(self.steps)
            typical = np.median(gaps)
            mids = [round(a + g / 2, 3) for a, g in zip(self.steps, gaps) if g < 1.6 * typical]
            self.steps = sorted(self.steps + mids)
        elif ev.key in ("[", "]"):
            d = -0.02 if ev.key == "[" else 0.02
            self.steps = [round(s + d, 3) for s in self.steps]
        elif ev.key == "c":
            self.steps = []
        elif ev.key == "s":
            self.meta.update(steps=self.steps, labeled=True)
            if not self.meta.get("trial"):
                self.meta["true_steps"] = len(self.steps)
            dataset.save_meta(self.name, self.meta)
            self.saved = True
        else:
            return
        if ev.key != "s":
            self.saved = False
        self.redraw()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?")
    args = ap.parse_args()
    name = args.session
    if name is None:
        files = sorted(config.SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            raise SystemExit("No sessions yet: record one with collect.py.")
        name = files[-1].stem
    plt.rcParams["keymap.save"] = ["ctrl+s"]
    plt.rcParams["keymap.back"] = ["left", "backspace"]
    plt.rcParams["keymap.pan"] = []
    labeler = Labeler(name)
    plt.show()
    return labeler


if __name__ == "__main__":
    main()
