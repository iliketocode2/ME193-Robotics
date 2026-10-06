# Q-Tables — Silly Walks

A Double Motor with a leg on each side learns to walk straight using a
**Q-table**, the simplest kind of reinforcement learning. The Double Motor's
built-in IMU gives yaw; the table learns which left/right speed difference to
use for each amount of drift.

## Run it

From the repo root, with `my_env` activated:

```
cd "Public stuff/projects/q-tables"
python silly_walk_qlearn.py train   # learn — saves q_table.json after every episode
python silly_walk_qlearn.py run     # walk with the learned table, no exploring
```

**Training:** before each episode the script waits for Enter so you can set
the walker back at the start and aim it. Type `q` + Enter (or Ctrl+C) to
quit — the motors stop and the table is saved.

**Run:** no prompts. It connects, then walks for `RUN_STEPS` steps (or until
it drifts past `ABORT_YAW`, or you press Ctrl+C), then stops. Aim the walker
before you start the script. It needs a `q_table.json` from training first. Running `train` again resumes from `q_table.json`;
delete that file to start from scratch.

Set `CARD_SERIAL` / `CARD_COLOR` at the top of the script to match your
Connection Card (defaults: green, 1126).

## How it works

**States (rows)** — yaw error, sorted into 9 bins:

| State | Yaw error | Meaning |
| --- | --- | --- |
| 0 | < −20° | Strong left drift |
| 1 | −20° .. −10° | Significant left drift |
| 2 | −10° .. −5° | Moderate left drift |
| 3 | −5° .. −2° | Slight left drift |
| **4** | **−2° .. +2°** | **On target** |
| 5 | +2° .. +5° | Slight right drift |
| 6 | +5° .. +10° | Moderate right drift |
| 7 | +10° .. +20° | Significant right drift |
| 8 | > +20° | Strong right drift |

**Actions (columns)** — speed difference added on top of `BASE_SPEED`:
`left = BASE_SPEED − diff/2`, `right = BASE_SPEED + diff/2`, with
`diff ∈ {−40, −20, 0, +20, +40}`. The average speed never changes.

**Reward** — after each action: +1 if |error| < 2°, 0 if < 10°, −1 otherwise.

**Update** — `Q(s,a) = (1 − α)·Q(s,a) + α·(r + γ·max Q(s'))`

**Explore vs. exploit** — with probability ε pick a random action, otherwise
the best one in the row. ε starts at 1.0 and is multiplied by 0.98 every step
(down to a floor of 0.05), so after about 150 steps (~5 episodes) it's mostly
using the table.

## Tuning

| Constant | What it does |
| --- | --- |
| `STEP_TIME` | How long each action runs before checking yaw. Legs make the yaw wobble every stride, so set this to about one full stride, or the rewards will mostly be gait noise. |
| `LEG_HOME_LEFT`, `LEG_HOME_RIGHT` | Absolute motor angle (0–359) each leg turns to before every walk, so every episode starts from the same point in the stride. Pick angles where the robot stands steadily. |
| `BASE_SPEED` | Average walking speed. |
| `FORWARD` | Set to `-1` if the walker goes backward. |
| `ALPHA`, `GAMMA` | Learning rate and discount. |
| `EPSILON_DECAY`, `EPSILON_MIN` | How quickly exploration drops off. |
| `ABORT_YAW` | An episode ends early if the walker turns more than this. |

**Yaw units:** the Double Motor's IMU reports yaw in **tenths of a degree**
(900 = 90°). The script divides by 10. Watch out for this if you use
`dm.yaw()` elsewhere.

**Mirrored legs?** If your build swaps left and right, the action labels
("right turn" / "left turn") will be backwards, but learning still works. The
table only learns which column helps in each row; the names don't matter. Look
at the "best" column in the printed table: drift to the left (states 0–3)
should learn one sign and drift to the right (states 5–8) the other.
