"""Plot how training went: mean reward and each reward term against iteration.

    python plot_training.py runs/train-20261114-120000          # a run folder that runIt.py brought back
    python plot_training.py runs/train-20261114-120000 media/training_curves.png

Reads the TensorBoard event files mjlab wrote under <run>/logs/. Needs `tensorboard` and `matplotlib`
(on your laptop: ../.venv/bin/pip install tensorboard matplotlib).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if len(sys.argv) < 2:
  sys.exit(__doc__)
run = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "training_curves.png"

events = sorted(run.rglob("events.out.tfevents.*"))
if not events:
  sys.exit(f"no TensorBoard files under {run} (look for logs/rsl_rl/beam_drive_straight/<date>/)")
acc = EventAccumulator(str(events[-1].parent), size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags()["scalars"]


def series(tag):
  ev = acc.Scalars(tag)
  return [e.step for e in ev], [e.value for e in ev]


reward_tag = next((t for t in tags if t.endswith("mean_reward")), None)
term_tags = sorted(t for t in tags if "Episode_Reward/" in t)
length_tag = next((t for t in tags if t.endswith("mean_episode_length")), None)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
if reward_tag:
  axes[0].plot(*series(reward_tag), color="tab:orange")
axes[0].set_title("Mean reward per episode"); axes[0].set_xlabel("iteration")
for t in term_tags:
  axes[1].plot(*series(t), label=t.split("/")[-1])
axes[1].set_title("Reward terms (per episode)"); axes[1].set_xlabel("iteration"); axes[1].legend(fontsize=8)
if length_tag:
  axes[2].plot(*series(length_tag), color="tab:green")
axes[2].set_title("Mean episode length (steps; 500 = full 10 s)"); axes[2].set_xlabel("iteration")
for a in axes:
  a.grid(alpha=0.3)
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120)
print(f"wrote {out}")
print("scalars in this run:", ", ".join(tags))
