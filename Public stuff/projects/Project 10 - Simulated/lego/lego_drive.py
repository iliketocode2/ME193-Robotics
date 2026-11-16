"""Drive the real LEGO two-leg car with the policy trained in MuJoCo (or with a fixed stepping pattern).

Runs on your laptop and talks to the LEGO Education Double Motor over Bluetooth, like the other ME193 projects.
Install once (in a normal Windows/Mac terminal, not WSL: WSL has no Bluetooth):
    pip install legoeducation numpy
Keep lelib.py and policy.npz in this folder. Then:

    python lego_drive.py test               # 1. check each leg's direction and measure the step speed
    python lego_drive.py open_loop 5        # 2. both legs step nonstop for 5 s (compare with check_model.py)
    python lego_drive.py policy 10          # 3. the trained policy drives for 10 s

Before every run: put the car on the floor facing the way you want it to go, with BOTH LEGS LEVEL (horizontal,
pointing forward and back). The program calls that leg angle 0 and that heading 0, the same as the simulation.

What the program does, 50 times a second (the same loop as beam_task.py in the simulation):
  1. read the Double Motor's gyro (yaw) and both motor encoders (leg angles)
  2. build the 13 numbers the network was trained on: heading, turn rate, leg angles, leg speeds, step state
  3. run the network (numpy, no torch): one number per leg, above 0 = "start your next 135-degree step now"
  4. for each leg that is told to step and is not already stepping: move its target 135 degrees further forward
Every run is logged to lego_run_<mode>.csv (time, yaw, leg angles, network outputs).
"""
import csv
import math
import sys
import time
from pathlib import Path

import legoeducation as le
import numpy as np

from lelib import doubleMotor

# ---- settings to check on your car (lego_drive.py test tells you what to put here) ---------------------------
CARD_SERIAL = None          # your connection card's serial number, or None for the first Double Motor found
CARD_COLOR = None           # e.g. le.LEGO_COLOR_BLUE, or None
# Which way the motor's position counter goes when a leg moves FORWARD (its bottom end sweeping backward, which
# pushes the car forward). +1 or -1 per side. The two sides of a Double Motor face opposite ways, so they usually differ.
FORWARD_SIGN = {le.MOTOR_LEFT: -1, le.MOTOR_RIGHT: +1}
STEP_SPEED = 70             # motor speed for a step, in percent. The simulation assumes 12 rad/s (about 115 rpm, a
                            # 135-degree step in 0.2 s); `test` measures yours. Match the two (here or MAX_BEAM_SPEED).
YAW_SIGN = -1               # the Double Motor's yaw is + clockwise (turning right); the simulation's is + to the left
LOOP_HZ = 50                # decisions per second, as in training (Bluetooth may not keep up; the loop prints its rate)
STEP_DONE_DEG = 10          # a leg counts as "stepping" until it is within this many degrees of its target
# ---------------------------------------------------------------------------------------------------------------

STEP_DEG = 135
N_INPUTS = 13               # heading 2, turn rate 1, leg angles 4, leg speeds 2, step state 4 (see run())
SIDES = (le.MOTOR_LEFT, le.MOTOR_RIGHT)      # the policy's order: [left, right]
NAMES = {le.MOTOR_LEFT: "left", le.MOTOR_RIGHT: "right"}
HERE = Path(__file__).resolve().parent


class Policy:
    """The trained network, from policy.npz (written by make_video.py): normalize, 3 layers, ELU in between."""

    def __init__(self, path=HERE / "policy.npz"):
        if not path.exists():
            sys.exit(f"{path.name} is missing: copy it here from the make_video.py run folder (runs/make_video-*/).")
        p = np.load(path)
        self.mean, self.std, self.eps = p["obs_mean"], p["obs_std"], float(p["obs_eps"])
        self.layers = [(p["w0"], p["b0"]), (p["w1"], p["b1"]), (p["w2"], p["b2"])]
        self.n_inputs = self.layers[0][0].shape[1]
        if self.n_inputs != N_INPUTS:
            sys.exit(f"policy.npz expects {self.n_inputs} inputs but this program builds {N_INPUTS}: it comes from an "
                     f"older training run. Use the policy.npz from the latest make_video.py run.")
        if list(p["leg_order"]) != ["left_beam", "right_beam"]:
            sys.exit(f"policy.npz has legs in order {list(p['leg_order'])}; this program expects [left, right]")
        if abs(float(p["step_angle_deg"]) - STEP_DEG) > 1e-3:
            sys.exit(f"policy.npz was trained with {float(p['step_angle_deg'])}-degree steps, not {STEP_DEG}")
        print(f"policy.npz: {self.n_inputs} inputs; trained for steps at {float(p['step_speed_rad_s']):.1f} rad/s")

    def __call__(self, obs):
        x = (obs - self.mean) / (self.std + self.eps)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))   # ELU
        return x


class Car:
    """The Double Motor: leg angles (deg, 0 = level, + = forward), yaw, and the 135-degree step commands."""

    def __init__(self):
        self.dm = doubleMotor()
        print("Connecting to the Double Motor (turn it on)...")
        self.dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        for side in SIDES:
            self.dm.motor_set_end_state(le.MOTOR_END_STATE_HOLD, motor=side)   # hold the leg between steps
        self.zero()

    def zero(self):
        """Call this leg angle 0 and this heading 0 (the car must be standing with both legs level)."""
        for side in SIDES:
            self.dm.motor_reset_relative_position(motor=side, position=0)
        self.dm.reset_heading()
        time.sleep(0.3)
        for _ in range(50):     # wait until the first gyro and motor readings have arrived
            if not math.isnan(self.dm.yaw()) and not any(math.isnan(self.dm.motor[s].position) for s in SIDES):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("no readings from the Double Motor's gyro/encoders")
        self.goal = {side: 0 for side in SIDES}     # where each leg's current step ends, in forward degrees

    def leg_deg(self, side):
        return FORWARD_SIGN[side] * float(self.dm.motor[side].position)

    def yaw_rad(self):
        """Heading in radians, + = turned left, like the simulation."""
        return YAW_SIGN * math.radians(self.dm.yaw())

    def stepping(self, side):
        return self.leg_deg(side) < self.goal[side] - STEP_DONE_DEG

    def step(self, side):
        """Start the next 135-degree step of this leg (forward), then hold."""
        self.goal[side] += STEP_DEG
        self.dm.motor_run_to_relative_position(int(FORWARD_SIGN[side] * self.goal[side]), motor=side, speed=STEP_SPEED,
                                               blocking=False)

    def stop(self):
        try:
            for side in SIDES:
                self.dm.motor_stop(motor=side)
        finally:
            self.dm.disconnect()


def run(car, mode, seconds, policy=None):
    """The control loop. mode 'policy' asks the network; 'open_loop' steps both legs whenever they are free."""
    dt_target = 1.0 / LOOP_HZ
    prev = {"t": time.monotonic(), "yaw": car.yaw_rad(), "legs": [car.leg_deg(s) for s in SIDES]}
    turn_rate, leg_speed = 0.0, np.zeros(2)
    log = []
    start = prev["t"]
    ticks = 0
    print(f"\n  time   heading (deg)   steps left, right   loop rate")
    steps = {s: 0 for s in SIDES}
    while (now := time.monotonic()) - start < seconds:
        yaw = car.yaw_rad()
        legs = np.array([car.leg_deg(s) for s in SIDES])
        dt = max(now - prev["t"], 1e-3)
        # rates from differences, lightly smoothed (Bluetooth readings arrive in steps)
        turn_rate = 0.5 * turn_rate + 0.5 * (yaw - prev["yaw"]) / dt
        leg_speed = 0.5 * leg_speed + 0.5 * np.radians(legs - np.array(prev["legs"])) / dt
        q = np.radians(legs)
        stepping = np.array([float(car.stepping(s)) for s in SIDES])
        remaining = np.radians(np.maximum([car.goal[s] - legs[i] for i, s in enumerate(SIDES)], 0.0))

        if mode == "policy":
            # the same 13 numbers, in the same order, as the actor observations in beam_task.py
            obs = np.concatenate([
                [math.sin(yaw), math.cos(yaw)],          # heading
                [turn_rate],                              # turn rate (rad/s, + = left)
                np.sin(q), np.cos(q),                     # leg angles: sin left, sin right, cos left, cos right
                leg_speed,                                # leg speeds (rad/s)
                stepping, remaining,                      # step state
            ]).astype(np.float32)
            action = policy(obs)
        else:
            action = np.ones(2)                           # open loop: always "step"

        for i, side in enumerate(SIDES):
            if action[i] > 0 and not stepping[i]:
                car.step(side)
                steps[side] += 1
        log.append([round(now - start, 3), round(math.degrees(yaw), 2), *np.round(legs, 1), *np.round(action, 3),
                    steps[SIDES[0]], steps[SIDES[1]]])
        prev = {"t": now, "yaw": yaw, "legs": list(legs)}
        ticks += 1
        if ticks % LOOP_HZ == 0:
            rate = LOOP_HZ / max(now - log[-LOOP_HZ][0] - start, 1e-3) if len(log) >= LOOP_HZ else 0
            print(f"  {now - start:5.1f}  {math.degrees(yaw):12.1f}   {steps[SIDES[0]]:6d}, {steps[SIDES[1]]:4d}   {rate:6.1f} Hz")
        sleep = dt_target - (time.monotonic() - now)
        if sleep > 0:
            time.sleep(sleep)

    out = HERE / f"lego_run_{mode}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "heading_deg", "left_leg_deg", "right_leg_deg", "out_left", "out_right", "steps_left", "steps_right"])
        w.writerows(log)
    print(f"\nDone: {steps[SIDES[0]]} left steps, {steps[SIDES[1]]} right steps, final heading "
          f"{math.degrees(car.yaw_rad()):.1f} deg. Measure how far it went forward and sideways. Log: {out.name}")


def test(car):
    """Check FORWARD_SIGN for each leg, and time a 135-degree step at STEP_SPEED."""
    print("\nEach leg will turn 45 degrees, one at a time. Watch which way it goes.")
    for side in SIDES:
        start = car.dm.motor[side].position
        car.dm.motor_run_to_relative_position(int(start + FORWARD_SIGN[side] * 45), motor=side, speed=30)
        time.sleep(0.5)
        ans = input(f"Did the {NAMES[side]} leg's BOTTOM end swing BACKWARD (the way that pushes the car forward)? [y/n] ")
        if ans.strip().lower().startswith("n"):
            print(f"  -> set FORWARD_SIGN[le.MOTOR_{NAMES[side].upper()}] = {-FORWARD_SIGN[side]} at the top of this file")
        else:
            print(f"  -> FORWARD_SIGN for {NAMES[side]} is right ({FORWARD_SIGN[side]:+d})")
    print(f"\nTiming one 135-degree step of each leg at STEP_SPEED = {STEP_SPEED}%...")
    for side in SIDES:
        start = car.dm.motor[side].position
        t0 = time.monotonic()
        car.dm.motor_run_to_relative_position(int(start + FORWARD_SIGN[side] * STEP_DEG), motor=side, speed=STEP_SPEED)
        t = time.monotonic() - t0
        print(f"  {NAMES[side]}: {t:.2f} s (includes the Bluetooth round trip) -> about {math.radians(STEP_DEG) / t:.1f} rad/s")
    print("The simulation assumes 12 rad/s (0.2 s per step). If yours is much slower, raise STEP_SPEED, or set\n"
          "MAX_BEAM_SPEED in robot_model.py to your number and retrain.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "policy"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else (5.0 if mode == "open_loop" else 10.0)
    if mode not in ("test", "open_loop", "policy"):
        sys.exit(__doc__)
    policy = Policy() if mode == "policy" else None
    car = Car()
    try:
        if mode == "test":
            test(car)
        else:
            input("Car on the floor, facing forward, both legs level? Press Enter to start (Ctrl-C stops). ")
            car.zero()
            run(car, mode, seconds, policy)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        car.stop()


if __name__ == "__main__":
    main()
