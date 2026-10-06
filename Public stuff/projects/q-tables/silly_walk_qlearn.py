"""
Silly Walks -- teach a two-legged Double Motor walker to go straight with a Q-table.

The Double Motor has a leg on each side. Its built-in IMU reports yaw, and a
Q-table learns which left/right speed difference to apply for each amount of
drift so the walker heads straight (yaw = 0).

    python silly_walk_qlearn.py train   # learn (explore + exploit), saves q_table.json
    python silly_walk_qlearn.py run     # walk using the learned table only (no exploring)

Training saves after every episode, so you can Ctrl+C and pick up later --
the next `train` resumes from q_table.json (delete it to start over).

States  (rows):    9 yaw-error bins, see BIN_EDGES.
Actions (columns): 5 speed differences, see ACTIONS.
    left_speed  = BASE_SPEED - diff // 2
    right_speed = BASE_SPEED + diff // 2
Reward:  +1 if |error| < 2 deg, 0 if < 10 deg, -1 otherwise.
Update:  Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (r + gamma * max Q(s'))
"""

import bisect
import json
import math
import os
import random
import sys
import time

import legoeducation as le

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "useful libraries"))
from lelib import doubleMotor  # noqa: E402

# ── Hardware ─────────────────────────────────────────────────────────────────
CARD_SERIAL = 1126                  # serial printed on your Connection Card
CARD_COLOR = le.LEGO_COLOR_GREEN

LEG_HOME_LEFT = 0    # absolute motor angle (0-359) each leg returns to before every walk,
LEG_HOME_RIGHT = 0   # so every episode starts from the same point in the stride
LEG_HOME_SPEED = 20

BASE_SPEED = 40      # average leg speed (%) -- stays fixed no matter the correction
FORWARD = 1          # set to -1 if your walker walks backward with positive speeds

# ── Q-table shape ────────────────────────────────────────────────────────────
# Yaw error bin edges (deg). 8 edges -> 9 states:
#   0: < -20   1: -20..-10   2: -10..-5   3: -5..-2   4: -2..+2 (goal)
#   5: +2..+5  6: +5..+10    7: +10..+20  8: > +20
BIN_EDGES = [-20, -10, -5, -2, 2, 5, 10, 20]
N_STATES = len(BIN_EDGES) + 1

# Speed difference (right - left, %) for each action.
#   A0 -40: hard right   A1 -20: gentle right   A2 0: straight
#   A3 +20: gentle left  A4 +40: hard left
ACTIONS = [-40, -20, 0, 20, 40]
N_ACTIONS = len(ACTIONS)

# ── Learning ─────────────────────────────────────────────────────────────────
ALPHA = 0.1            # learning rate: how much a new experience overrides the old Q
GAMMA = 0.9            # discount: how much future reward matters vs. this step's
EPSILON_START = 1.0    # start fully exploring
EPSILON_DECAY = 0.98   # multiply epsilon by this after every step
EPSILON_MIN = 0.05     # always keep a little exploration while training

# ── Timing / safety ──────────────────────────────────────────────────────────
STEP_TIME = 0.4          # seconds each action runs before we look at the result.
                         # Legs wobble the yaw every stride -- aim for ~one full stride.
STEPS_PER_EPISODE = 30   # training episode length
RUN_STEPS = 75           # length of the walk in `run` mode (~30 s at STEP_TIME = 0.4)
ABORT_YAW = 60           # |yaw| beyond this ends the episode (walker is way off course)

Q_FILE = os.path.join(os.path.dirname(__file__), "q_table.json")


# ── Q-learning pieces ────────────────────────────────────────────────────────

def yaw_to_state(yaw_error):
    """Which row of the table this yaw error falls in (0..8)."""
    return bisect.bisect_right(BIN_EDGES, yaw_error)


def reward(yaw_error):
    """+1 on target, 0 acceptable, -1 off course (checked in that order)."""
    if abs(yaw_error) < 2:
        return 1.0
    if abs(yaw_error) < 10:
        return 0.0
    return -1.0


def choose_action(q, state, epsilon):
    """Epsilon-greedy: random action with probability epsilon, else the best-known one."""
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)                       # explore
    row = q[state]
    best = max(row)
    return random.choice([a for a, v in enumerate(row) if v == best])  # exploit (ties broken randomly)


def update_q(q, state, action, r, next_state, terminal=False):
    """Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (r + gamma * max Q(s')).
    If the walk ended here (terminal), there is no future to add."""
    best_next = 0.0 if terminal else max(q[next_state])
    q[state][action] = (1 - ALPHA) * q[state][action] + ALPHA * (r + GAMMA * best_next)


# ── Save / load / show ───────────────────────────────────────────────────────

def load_q():
    """Load the table from q_table.json, or start a fresh all-zeros table."""
    if os.path.exists(Q_FILE):
        with open(Q_FILE) as f:
            data = json.load(f)
        if data.get("actions") != ACTIONS or data.get("bin_edges") != BIN_EDGES:
            sys.exit(f"{Q_FILE} was trained with different ACTIONS/BIN_EDGES -- "
                     "delete it (or rename it) to start a new table.")
        print(f"Loaded {Q_FILE} (epsilon={data['epsilon']:.3f}, steps={data['steps']})")
        return data["q"], data["epsilon"], data["steps"]
    print("No saved table -- starting from zeros.")
    return [[0.0] * N_ACTIONS for _ in range(N_STATES)], EPSILON_START, 0


def save_q(q, epsilon, steps):
    with open(Q_FILE, "w") as f:
        json.dump({"actions": ACTIONS, "bin_edges": BIN_EDGES,
                   "epsilon": epsilon, "steps": steps, "q": q}, f, indent=2)


def print_q_table(q):
    labels = ["< -20", "-20..-10", "-10..-5", "-5..-2", "-2..+2",
              "+2..+5", "+5..+10", "+10..+20", "> +20"]
    print("\n state  yaw err   " + "".join(f"{d:+5d}%  " for d in ACTIONS) + " best")
    for s, row in enumerate(q):
        best = ACTIONS[row.index(max(row))]
        print(f"  {s}   {labels[s]:>9}  " + "".join(f"{v:+6.2f} " for v in row) + f" {best:+d}%")
    print()


# ── Robot helpers ────────────────────────────────────────────────────────────

def read_yaw_error(dm):
    """Yaw error in degrees, wrapped to -180..180 (0 = heading we reset to).
    Positive = drifted right (clockwise)."""
    yaw_deg = dm.yaw() / 10          # the IMU reports tenths of a degree
    return (yaw_deg + 180) % 360 - 180


def wait_for_yaw(dm, timeout=3.0):
    """The IMU reading is NaN until the first notification arrives -- wait for it."""
    end = time.time() + timeout
    while math.isnan(dm.yaw()):
        if time.time() > end:
            raise RuntimeError("No yaw reading from the Double Motor's IMU.")
        time.sleep(0.05)


def check_connected(dm):
    """A BLE drop doesn't raise -- commands just stop working and yaw goes stale.
    Bail out instead of training on frozen readings."""
    if not dm.connected:
        raise ConnectionError("Lost connection to the Double Motor.")


def home_legs(dm):
    """Turn both legs to their home angles (one side at a time, waiting for each)."""
    dm.motor_run_to_absolute_position(LEG_HOME_LEFT, motor=le.MOTOR_LEFT, speed=LEG_HOME_SPEED)
    dm.motor_run_to_absolute_position(LEG_HOME_RIGHT, motor=le.MOTOR_RIGHT, speed=LEG_HOME_SPEED)


def apply_action(dm, action):
    """Drive both legs: same average speed, `diff` split between the sides."""
    diff = ACTIONS[action]
    left = max(-100, min(100, FORWARD * (BASE_SPEED - diff // 2)))
    right = max(-100, min(100, FORWARD * (BASE_SPEED + diff // 2)))
    dm.movement_move_tank(left, right, blocking=False)


def run_episode(dm, q, epsilon, steps, learn, n_steps=STEPS_PER_EPISODE):
    """One walk from a fresh heading. Returns (epsilon, steps, total_reward)."""
    check_connected(dm)
    home_legs(dm)                        # same leg pose every time...
    dm.reset_heading()                   # ...then zero the heading (homing can twist the body)
    time.sleep(0.3)                      # let the reset show up in the next IMU report
    state = yaw_to_state(read_yaw_error(dm))
    total = 0.0

    for step in range(n_steps):
        check_connected(dm)
        action = choose_action(q, state, epsilon if learn else 0.0)   # 3. choose
        apply_action(dm, action)                                      # 4. execute
        time.sleep(STEP_TIME)

        err = read_yaw_error(dm)                                      # 5. observe
        next_state = yaw_to_state(err)
        r = reward(err)
        aborted = abs(err) > ABORT_YAW
        total += r

        if learn:
            update_q(q, state, action, r, next_state, terminal=aborted)  # 6. update
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)       # 7. decay
            steps += 1

        print(f"  step {step:2d}  yaw {err:+6.1f}  state {state}->{next_state}  "
              f"action {ACTIONS[action]:+3d}%  reward {r:+.0f}  eps {epsilon:.3f}")

        if aborted:
            print(f"  |yaw| > {ABORT_YAW} deg -- ending episode early.")
            break
        state = next_state                                            # 8. repeat from s'

    dm.movement_stop()
    return epsilon, steps, total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode not in ("train", "run"):
        sys.exit("usage: python silly_walk_qlearn.py [train|run]")
    learn = mode == "train"
    if not learn and not os.path.exists(Q_FILE):
        sys.exit(f"No {Q_FILE} yet -- run `python silly_walk_qlearn.py train` first.")

    q, epsilon, steps = load_q()
    print_q_table(q)

    dm = doubleMotor()
    print("Connecting to Double Motor...")
    dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
    if not dm.connected:
        sys.exit("Could not connect to the Double Motor.")

    try:
        wait_for_yaw(dm)
        if not learn:
            # Run mode: no prompts -- walk straight away using the learned table.
            print(f"Walking for up to {RUN_STEPS} steps (Ctrl+C to stop)...")
            _, _, total = run_episode(dm, q, epsilon, steps, learn=False, n_steps=RUN_STEPS)
            print(f"Done -- total reward {total:+.0f}")
            return
        episode = 0
        while True:
            cmd = input(f"\nEpisode {episode}: aim the walker, press Enter to go (q + Enter to quit): ")
            if cmd.strip().lower() == "q":
                break
            epsilon, steps, total = run_episode(dm, q, epsilon, steps, learn)
            print(f"Episode {episode} done -- total reward {total:+.0f}")
            if learn:
                save_q(q, epsilon, steps)
                print_q_table(q)
            episode += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # Each cleanup step on its own, so one failing doesn't skip the rest.
        for cleanup in (dm.movement_stop,   # not lelib's dm.stop(), which only stops one motor
                        dm.disconnect):
            try:
                cleanup()
            except Exception as e:
                print(f"Cleanup error ({cleanup.__name__}): {e}")
        if learn:
            save_q(q, epsilon, steps)
            print(f"Saved {Q_FILE}")


if __name__ == "__main__":
    main()
