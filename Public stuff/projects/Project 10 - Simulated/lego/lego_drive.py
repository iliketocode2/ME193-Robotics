"""Drive the real LEGO two-leg car with the policy trained in MuJoCo (or with a fixed stepping pattern).

Runs on your laptop and talks to the LEGO Education Double Motor over Bluetooth, like the other ME193 projects.
Install once (in a normal Windows/Mac terminal, not WSL: WSL has no Bluetooth):
    pip install legoeducation numpy
Keep lelib.py and policy.npz in this folder. Then:

    python lego_drive.py gyro               # 0. measure RATE_SCALE (turn the car one full turn by hand)
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
import threading
import time
from pathlib import Path

import legoeducation as le
import numpy as np

from lelib import doubleMotor

# ---- settings to check on your car (lego_drive.py test tells you what to put here) ---------------------------
CARD_SERIAL = 1126          # your connection card's serial number, or None for the first Double Motor found
CARD_COLOR = le.LEGO_COLOR_GREEN           # e.g. le.LEGO_COLOR_BLUE, or None
# Which way the motor's position counter goes when a leg moves FORWARD (its bottom end sweeping backward, which
# pushes the car forward). +1 or -1 per side. The two sides of a Double Motor face opposite ways, so they usually differ.
FORWARD_SIGN = {le.MOTOR_LEFT: +1, le.MOTOR_RIGHT: -1}
STEP_SPEED = 88             # motor speed for a step, in percent. The simulation assumes 6 rad/s (about 57 rpm, a
                            # 135-degree step in 0.4 s); `test` measures yours. Match the two (here or MAX_BEAM_SPEED).
YAW_SIGN = +1               # the Double Motor's yaw is + clockwise (turning right); the simulation's is + to the left
YAW_FACE = None             # which face of the Double Motor points UP on the car (le.DEVICE_FACE_TOP, ..._FRONT, ...).
                            # The gyro only measures turning on the floor around that face. None = use the face the
                            # Double Motor reports as up when the run starts. `lego_drive.py gyro` shows both.
HEADING_FROM = "rate"       # "rate": add up the gyro's turn rate around the vertical (works however the Double Motor
                            # is mounted). "yaw": the Double Motor's own yaw angle, which flips by 180 degrees when the
                            # Double Motor stands on its end, as on this car.
RATE_SCALE = 0.1            # heading degrees per (raw gyro unit x second), + = left. `lego_drive.py gyro` measures it:
                            # turn the car one full turn LEFT by hand and it prints the number to put here.
LOOP_HZ = 50                # decisions per second, as in training (Bluetooth may not keep up; the loop prints its rate)
STEP_DONE_DEG = 10          # a leg counts as "stepping" until it is within this many degrees of its target
# ---------------------------------------------------------------------------------------------------------------

STEP_DEG = 135
N_INPUTS = 13               # heading 2, turn rate 1, leg angles 4, leg speeds 2, step state 4 (see run())
SIDES = (le.MOTOR_LEFT, le.MOTOR_RIGHT)      # the policy's order: [left, right]
NAMES = {le.MOTOR_LEFT: "left", le.MOTOR_RIGHT: "right"}
HERE = Path(__file__).resolve().parent
FACE_NAMES = {le.DEVICE_FACE_TOP: "top", le.DEVICE_FACE_FRONT: "front", le.DEVICE_FACE_RIGHT: "right",
              le.DEVICE_FACE_BOTTOM: "bottom", le.DEVICE_FACE_BACK: "back", le.DEVICE_FACE_LEFT: "left"}


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


class Heading(threading.Thread):
    """Heading from the gyro's turn rate. The Double Motor sends a new IMU reading every few tens of milliseconds;
    this thread catches each one, takes the part of the rotation around the vertical (the direction gravity pulls,
    from the accelerometer while standing still), and adds rate x time. Unlike the Double Motor's yaw angle, this
    does not depend on how the Double Motor is mounted."""

    def __init__(self, dm):
        super().__init__(daemon=True)
        self.dm = dm
        self.up = None              # unit vector, in the Double Motor's axes, that points up
        self.bias = 0.0             # the vertical rate reading while standing still (raw units)
        self.raw = 0.0              # sum of (vertical rate - bias) x seconds, in raw units
        self.lock = threading.Lock()
        self.start()

    def _vectors(self):
        imu = self.dm.imu_device
        return imu, np.array([imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ], float), \
            np.array([imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ], float)

    def zero(self, settle=0.6):
        """Measure which way is up and the gyro's resting offset (the car must be standing still), then heading 0."""
        accs, rates, last, t_end = [], [], None, time.monotonic() + settle
        while time.monotonic() < t_end:
            imu, acc, gyro = self._vectors()
            if imu is not last and np.all(np.isfinite(acc)) and np.all(np.isfinite(gyro)):
                accs.append(acc)
                rates.append(gyro)
                last = imu
            time.sleep(0.005)
        if not accs:
            raise RuntimeError("no IMU readings from the Double Motor")
        acc = np.mean(accs, axis=0)
        with self.lock:
            self.up = acc / np.linalg.norm(acc)
            self.bias = float(np.mean(rates, axis=0) @ self.up)
            self.raw = 0.0

    def degrees(self, scale=None):
        with self.lock:
            return self.raw * (RATE_SCALE if scale is None else scale)

    def run(self):
        last, last_t = None, None
        while True:
            imu, _, gyro = self._vectors()
            if imu is not last:
                now = time.monotonic()
                with self.lock:
                    if self.up is not None and last_t is not None and np.all(np.isfinite(gyro)):
                        dt = min(now - last_t, 0.1)      # a gap longer than this is a dropout, not a turn
                        self.raw += (float(gyro @ self.up) - self.bias) * dt
                last, last_t = imu, now
            time.sleep(0.002)


class Car:
    """The Double Motor: leg angles (deg, 0 = level, + = forward), yaw, and the 135-degree step commands."""

    def __init__(self):
        self.dm = doubleMotor()
        print("Connecting to the Double Motor (turn it on)...")
        self.dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        for side in SIDES:
            self.dm.motor_set_end_state(le.MOTOR_END_STATE_HOLD, motor=side)   # hold the leg between steps
        self.heading = Heading(self.dm)
        self.zero()

    def zero(self):
        """Call this leg angle 0 and this heading 0 (the car must be standing with both legs level)."""
        for side in SIDES:
            self.dm.motor_reset_relative_position(motor=side, position=0)
        if HEADING_FROM == "yaw":
            self.set_yaw_face()
        self.dm.reset_heading()
        self.heading.zero()
        time.sleep(0.3)
        for _ in range(50):     # wait until the first gyro and motor readings have arrived
            if not math.isnan(self.dm.yaw()) and not any(math.isnan(self.dm.motor[s].position) for s in SIDES):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("no readings from the Double Motor's gyro/encoders")
        self.goal = {side: 0 for side in SIDES}     # where each leg's current step ends, in forward degrees

    def set_yaw_face(self):
        """Make the gyro's yaw the car's turning on the floor. If the Double Motor is mounted on its side or end,
        its default yaw axis is horizontal, so "yaw" swings back and forth as the legs rock the body."""
        imu = self.dm.imu_device
        for _ in range(20):                       # the first IMU reading can take a moment to arrive
            if isinstance(imu.orientation, int) and 0 <= imu.orientation <= 5:
                break
            time.sleep(0.1)
            imu = self.dm.imu_device
        up = YAW_FACE if YAW_FACE is not None else imu.orientation
        if not (isinstance(up, int) and 0 <= up <= 5):
            print("  (could not read which face is up; leaving the gyro's yaw face as it is)")
            return
        if imu.yawFace != up:
            print(f"  gyro yaw face: {FACE_NAMES.get(imu.yawFace, imu.yawFace)} -> {FACE_NAMES[up]} (the face pointing up)")
            self.dm.imu_set_yaw_face(up)

    def leg_deg(self, side):
        return FORWARD_SIGN[side] * float(self.dm.motor[side].position)

    def yaw_rad(self):
        """Heading in radians, + = turned left, like the simulation."""
        if HEADING_FROM == "rate":
            return math.radians(self.heading.degrees())
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
            if self.dm.connected:
                for side in SIDES:
                    self.dm.motor_stop(motor=side)
        finally:
            self._quiet_pending()
            self.dm.disconnect()

    def _quiet_pending(self):
        """Each step is sent with blocking=False, so legoeducation keeps a future waiting for that move's
        "finished" reply. A step that is cut short by the next one, or still running at the end, never gets one.
        disconnect() then fails those futures with ConnectionError("Device disconnected unexpectedly"), and since
        nothing waits on them, asyncio prints "Future exception was never retrieved". That is our own planned
        disconnect, not a dropped connection, so mark those futures as handled first."""
        from legoeducation import _platform
        loop = _platform._persistent_loop
        futures = [f for dq in list(self.dm._pending_responses.values()) for f in list(dq)]
        if not futures or loop is None or loop.is_closed():
            return

        def mark():
            for f in futures:
                f.add_done_callback(lambda f: f.cancelled() or f.exception())

        loop.call_soon_threadsafe(mark)    # runs on the library's event loop before disconnect() fails them


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
    last_step, warned = start, False
    while (now := time.monotonic()) - start < seconds:
        if not car.dm.connected:
            # a real Bluetooth drop: the readings would freeze at their last values, so stop here
            print(f"\nBluetooth connection lost at {now - start:.1f} s. Check the Double Motor's battery and keep "
                  f"the laptop within a couple of meters. The log up to here is saved.")
            break
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
                last_step = now
        if mode == "policy" and now - last_step > 1.5 and not warned:
            # Standing still nothing changes, so the network keeps getting the same inputs and keeps saying no.
            print(f"  ({now - start:.1f} s: the policy has stopped stepping, at heading {math.degrees(yaw):.0f} deg. "
                  f"Training only started within {math.degrees(0.4):.0f} deg of straight.)")
            warned = True
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
    print("The simulation assumes 6 rad/s (0.4 s per step). If yours is much slower, raise STEP_SPEED, or set\n"
          "MAX_BEAM_SPEED in robot_model.py to your number and retrain.")


def gyro(car, seconds):
    """Measure RATE_SCALE, then show the heading live while you move the car by hand."""
    imu = car.dm.imu_device
    print(f"\nThe Double Motor's own angles at rest: yaw {imu.yaw}, pitch {imu.pitch}, roll {imu.roll} "
          f"(a pitch or roll near +-90 means it stands on its end, where its yaw angle is unreliable)")
    input("\n1. Put the car on the floor and keep it still. Mark where its nose points. Press Enter.")
    car.zero()
    input("2. Now turn it slowly LEFT (counterclockwise from above) one full turn, back to the mark, "
          "and press Enter.")
    raw = car.heading.degrees(scale=1.0)
    if abs(raw) < 1e-6:
        print("   The gyro did not register the turn: no IMU readings are arriving.")
        return
    scale = 360.0 / raw
    print(f"   Set RATE_SCALE = {scale:.5f} at the top of this file (it is {RATE_SCALE} now).")
    print(f"   With the current RATE_SCALE that turn read {raw * RATE_SCALE:.0f} deg; it should read +360.")

    print("\n3. Live check for the next few seconds, using the measured number: turn left (should go up),\n"
          "   turn right (down), and rock the nose up and down (should barely change).\n")
    car.zero()
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        imu = car.dm.imu_device
        print(f"   heading {car.heading.degrees(scale=scale):7.1f} deg   (Double Motor yaw {imu.yaw:5.0f})   ", end="\r")
        time.sleep(0.1)
    print()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "policy"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else {"open_loop": 5.0, "gyro": 20.0}.get(mode, 10.0)
    if mode not in ("test", "open_loop", "policy", "gyro"):
        sys.exit(__doc__)
    policy = Policy() if mode == "policy" else None
    car = Car()
    try:
        if mode == "test":
            test(car)
        elif mode == "gyro":
            gyro(car, seconds)
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
