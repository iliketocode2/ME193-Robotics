"""Train the Beam-Drive-Straight task. Same as mjlab's own `train`, but it registers our task first.

    python train.py Beam-Drive-Straight --env.scene.num-envs 4096 --agent.max-iterations 1000
"""
import sys

import beam_task  # noqa: F401  (registers "Beam-Drive-Straight")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
