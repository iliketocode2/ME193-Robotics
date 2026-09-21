"""
AprilTag centering -- computer webcam edition
================================================

Same straight-line PD centering as apriltag_center.py, but the camera
and the tag have swapped roles:

    - The AprilTag is taped to the Double Motor car itself, not
      mounted on a wall.
    - The camera is this computer's own webcam (via camlib.pick_camera()
      -- no phone/stream setup at all), sitting still and watching the
      car drive back and forth parallel to the screen.

The goal is unchanged: drive straight (both wheels the same signed
speed, no turning) until the tag's centroid is horizontally centered
in the camera's frame -- i.e. the car always stops in the middle of
your computer's camera view.

The PD math is the same policy() as apriltag_center.py. If you already
tuned Kp/Kd there, treat those as a starting point only -- this
camera's resolution and field of view are different from the phone
stream's, so the same gains will likely need retuning here.

What happens with no tag detected
-----------------------------------
Motors stop immediately -- no "last known command," no search/spin
behavior, same as apriltag_center.py.

Setup:
    1. Print an AprilTag from the 36h11 family and tape it to the car
       so it's visible face-on as the car drives parallel to the
       screen (roughly camera height).
    2. Run this script -- it'll prompt you to pick a camera if more
       than one is connected.
    3. Fill in / re-tune Kp/Kd in policy() below for this camera.

Run:
    my_env/Scripts/python "Public stuff/projects/apriltag_center_webcam.py"
"""
import os
import sys
import threading
import time

import cv2
import numpy as np

# lelib.py/camlib.py live in the shared "useful libraries" folder, not
# next to this script -- add it to sys.path so the imports below
# resolve no matter where this script is run from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "useful libraries"))

import legoeducation as le
from lelib import doubleMotor
from camlib import pick_camera

# --- Hardware placeholder ---------------------------------------------------
# None = connect to the first advertising Double Motor. Fine for solo
# testing, but ambiguous with more than one Double Motor nearby -- set
# a real card_serial/card_color (see your Connection Card) for the demo.
CARD_SERIAL = 1126  # <-- fill in with your Double Motor card's serial number
CARD_COLOR = le.LEGO_COLOR_GREEN

# --- AprilTag ----------------------------------------------------------------
APRILTAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11
TARGET_TAG_ID = None  # None = react to any detected tag; set an int to track one specific ID

# --- Controller tuning --------------------------------------------------------
MAX_SPEED = 40    # top motor speed as a percent (0-100)
DEADZONE_PX = 12  # +/- pixel band around center that reads as "already centered"

# Motor mounting may be mirrored side-to-side on this chassis, so
# "clockwise" on one side can drive that wheel backward even when both
# wheels are commanded the same signed speed. Flip whichever one goes
# the wrong way once you watch the car actually move (both wheels
# should always turn the SAME physical direction here -- there's no
# turning in this script, only straight driving).
INVERT_LEFT_MOTOR = False
INVERT_RIGHT_MOTOR = True

# Sign of the whole drive command: which direction (positive vs
# negative "drive") actually moves the car toward a positive pixel
# error. Flip this -- not the two flags above -- if the car drives the
# right physical direction relative to itself but the wrong direction
# relative to the tag's on-screen error.
INVERT_DRIVE_DIRECTION = False

# Firmware-level ramp (0-100): how fast the hub itself is allowed to
# speed up/slow down toward a newly commanded speed, independent of
# what policy() outputs. Lower = gentler ramp = less jerk, slower to
# react. Start low if the phone mount/tag attachment is fragile.
# car.run() drives through legoeducation's "movement" commands, not raw
# per-motor ones, so this must be set via movement_set_acceleration() --
# motor_set_acceleration() is a separate command channel that car.run()
# never touches.
MOTOR_ACCEL = 10
MOTOR_DECEL = 10

# How often (seconds) the motor-command thread checks for a new target
# and, if it changed, sends it. Decoupled from camera frame rate on
# purpose -- see _MotorDriver below.
MOTOR_COMMAND_PERIOD_S = 0.02


class _MotorDriver:
    """Sends Double Motor speed commands from a dedicated thread.

    Every legoeducation call -- motor_run, movement_move, etc. --
    ultimately blocks its calling thread until the BLE write completes
    (see legoeducation/_platform.py's _run_sync: it always waits on a
    concurrent.futures.Future, regardless of that call's own
    blocking= kwarg, which only affects whether the *coroutine* also
    waits for a completion ack). If the main thread issues those calls,
    every BLE round-trip stalls cap.read()/cv2.imshow() along with it,
    slowing the camera feed down.

    This class fixes that by owning the car and running its own loop on
    a background thread: the vision loop just calls set_target() (a
    plain attribute write under a lock, effectively instant), and this
    thread notices the change and sends it whenever it's ready to,
    without the video loop ever waiting on it.
    """

    def __init__(self, car, period_s=MOTOR_COMMAND_PERIOD_S):
        self._car = car
        self._period_s = period_s
        self._lock = threading.Lock()
        self._target = 0
        self._sent = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_target(self, target):
        with self._lock:
            self._target = target

    def _run(self):
        while not self._stop_event.is_set():
            with self._lock:
                target = self._target
            if target != self._sent:
                try:
                    self._car.run(target)
                    self._sent = target
                except Exception as e:
                    print(f"Motor command failed: {e}")
            self._stop_event.wait(self._period_s)

    def stop(self):
        """Stop the background thread. Doesn't itself stop the motors --
        call car.motor_stop() separately once this has returned."""
        self._stop_event.set()
        self._thread.join(timeout=1)


def detect_tag(detector, frame):
    """Return (corners[4,2], centroid[x,y]) for the first matching tag in
    this frame, or (None, None) if no (matching) tag is detected."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None

    for tag_corners, tag_id in zip(corners, ids.flatten()):
        if TARGET_TAG_ID is not None and tag_id != TARGET_TAG_ID:
            continue
        pts = tag_corners.reshape(4, 2)
        return pts, pts.mean(axis=0)
    return None, None


def policy(centroid, frame_width, prev_error, dt):
    """The controller: straight-line PD centering the tag horizontally.

    Returns (drive, error) where drive is clamped to [-1, 1] and error
    is this frame's raw pixel offset (pass it back in as prev_error on
    the next call so the D term has something to diff against).
    """
    error = centroid[0] - frame_width / 2
    if abs(error) < DEADZONE_PX:
        error = 0
    error_rate = (error - prev_error) / dt if dt > 0 else 0.0

    # ---------------- PD CONTROL -- FILL IN / RE-TUNE ----------------
    Kp = 0.0075  # <-- proportional gain
    Kd = 0.0001  # <-- derivative gain
    drive = Kp * error + Kd * error_rate
    # -------------------------------------------------------------------

    drive = max(-1.0, min(1.0, drive))
    return drive, error


def main():
    car = doubleMotor()
    cap = None
    driver = None

    try:
        car.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        if not car.connected:
            raise ConnectionError("Double Motor did not connect.")
        car.movement_set_acceleration(MOTOR_ACCEL, MOTOR_DECEL)
        driver = _MotorDriver(car)

        cap, _ = pick_camera()

        dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

        prev_error = 0.0
        last_time = time.monotonic()

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed -- camera may have disconnected.")
                break

            h, w = frame.shape[:2]

            now = time.monotonic()
            dt = now - last_time
            last_time = now

            corners, centroid = detect_tag(detector, frame)

            if corners is None:
                # No tag visible this frame -- stop, don't coast on the
                # last command (see module docstring). Routed through
                # the driver like every other command so this doesn't
                # block the vision loop either.
                driver.set_target(0)
                prev_error = 0.0
            else:
                drive, prev_error = policy(centroid, w, prev_error, dt)
                if INVERT_DRIVE_DIRECTION:
                    drive = -drive
                # Both wheels get the SAME signed speed -- straight
                # driving, no differential steering (see module docstring).
                target = int(max(-MAX_SPEED, min(MAX_SPEED, MAX_SPEED * drive)))
                driver.set_target(target)

                cv2.polylines(frame, [corners.astype(int)], isClosed=True, color=(0, 0, 255), thickness=3)
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"centroid=({cx},{cy}) error={prev_error:.0f}",
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)  # centering reference
            cv2.imshow("AprilTag centering (webcam) -- press q to stop", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop the motor-command thread first so it can't race the hard
        # stop below, then hard-stop independent of whatever the last
        # frame read -- explicit MOTOR_BOTH (lelib's stop() only stops
        # motor index 0).
        if driver is not None:
            driver.stop()
        car.motor_stop(motor=le.MOTOR_BOTH, blocking=True)
        car.disconnect()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
