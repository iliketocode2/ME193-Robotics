"""Watch a trained Beam-Drive-Straight policy in mjlab's viewer (needs a screen and a working mjlab install).

    python play.py Beam-Drive-Straight --agent zero          # beams do nothing: checks the robot loads
    python play.py Beam-Drive-Straight --checkpoint-file <run folder>/model_999.pt
"""
import sys

import beam_task  # noqa: F401  (registers "Beam-Drive-Straight")
from mjlab.scripts.play import main

if __name__ == "__main__":
  sys.exit(main())
