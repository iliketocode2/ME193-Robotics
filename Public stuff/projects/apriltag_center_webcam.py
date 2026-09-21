"""
AprilTag centering (drive straight only) -- laptop webcam version
====================================================================

Same as apriltag_center.py, but reads from the laptop's built-in/USB
webcam instead of an iPhone MJPEG stream. Everything about the control
logic is unchanged -- only the camera source setup is different.

Physical setup assumed here:

    - The AprilTag is fastened to the SIDE of the LEGO car (not the
      front), facing the laptop.
    - The laptop's webcam sits off to the side, roughly level with the
      tag, looking at the plane the car drives along -- i.e. the car's
      path is parallel to the laptop, same as the wall-mounted-tag
      case in the original script, just with the tag/camera roles
      swapped (tag moves, camera is stationary).
    - The car does NOT turn. It only drives straight forward/backward
      along that path. Moving forward/backward is what slides the
      tag's centroid left/right in the webcam frame.
    - The car is a plain Double Motor -- no pan platform, no
      Controller, nothing else connected.

Goal: drive straight (both wheels the same signed speed, no
differential steering) until the AprilTag's centroid is horizontally
centered in the webcam frame, driven by a PD controller.

What happens with no tag detected
-----------------------------------
Both motors stop immediately -- no "last known command," no
search/spin behavior.

Setup:
    1. Plug in / confirm your laptop's webcam works (Camera app, etc.)
       and note its index -- 0 is almost always the built-in camera;
       if you have an external USB webcam too, it may bump the built-in
       one to a different index or take 0 itself. If the wrong camera
       opens, try WEBCAM_INDEX = 1, 2, etc.
    2. Print an AprilTag from the 36h11 family and fasten it to the
       side of the car, facing the laptop, at roughly webcam height.
    3. Set the car up so it can drive back and forth along a path
       parallel to the laptop, with the tag staying in view the whole
       time.
    4. Run this script once first just to check the tag is detected
       and centroid/error look right in the preview window (press q
       to quit) -- e.g. with the motors unplugged/disconnected -- before
       trusting it to drive the car.
    5. Fill in / tune Kp/Kd in policy() below if needed.

Run:
    my_env/Scripts/python "Public stuff/projects/apriltag_center_webcam.py"
"""
import os
import sys
import time

import cv2
import numpy as np

# lelib.py lives in the shared "useful libraries" folder, not next to
# this script -- add it to sys.path so the import below resolves no
# matter where this script is run from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "useful libraries"))

import legoeducation as le
from lelib import doubleMotor

# --- Camera source -- laptop webcam -----------------------------------------
# 0 = default/built-in webcam. Bump to 1, 2, ... if you have multiple
# cameras and the wrong one opens.
WEBCAM_INDEX = 0

# On Windows, cv2.CAP_DSHOW usually opens faster and more reliably than
# the default backend. Set to None to let OpenCV pick automatically
# (e.g. on macOS/Linux, where CAP_DSHOW doesn't apply).
CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else None

# Requested capture resolution -- lower can mean less latency, which
# matters more here than image quality. Set either to None to leave it
# at the camera's default.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# None = no rotation; otherwise cv2.ROTATE_90_CLOCKWISE /
# cv2.ROTATE_90_COUNTERCLOCKWISE / cv2.ROTATE_180. Check the preview
# window first -- moving the car left/right should pan the tag
# left/right in frame, not up/down, since policy() below only reads
# horizontal pixel offset.
FRAME_ROTATION = None

# --- Hardware placeholder ---------------------------------------------------
# None = connect to the first advertising Double Motor. Fine for solo
# testing, but ambiguous with more than one Double Motor nearby -- set
# a real card_serial/card_color (see your Connection Card) for the demo.
CARD_SERIAL = 3664  # <-- fill in with your Double Motor card's serial number
CARD_COLOR = le.LEGO_COLOR_RED

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
# what policy() outputs. This is the real fix for a jerky/violent car --
# without it, every new motor_run() speed is applied instantly, so a
# noisy one-frame derivative spike (see policy()'s Kd term) snaps the
# motor straight to it. Lower = gentler ramp = less jerk, slower to
# react. Start low if the setup is fragile; raw legoeducation method,
# not wrapped by lelib.
MOTOR_ACCEL = 10
MOTOR_DECEL = 10


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

    # ---------------- PD CONTROL -- FILL IN ----------------
    Kp = 0.0075  # <-- proportional gain
    Kd = 0.0001  # <-- derivative gain
    drive = Kp * error + Kd * error_rate
    # --------------------------------------------------------

    drive = max(-1.0, min(1.0, drive))
    return drive, error


def main():
    car = doubleMotor()
    cap = None

    try:
        car.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        if not car.connected:
            raise ConnectionError("Double Motor did not connect.")
        car.motor_set_acceleration(MOTOR_ACCEL, MOTOR_DECEL, motor=le.MOTOR_BOTH)

        if CAMERA_BACKEND is not None:
            cap = cv2.VideoCapture(WEBCAM_INDEX, CAMERA_BACKEND)
        else:
            cap = cv2.VideoCapture(WEBCAM_INDEX)

        if FRAME_WIDTH is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        if FRAME_HEIGHT is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {WEBCAM_INDEX!r}. Try a "
                "different WEBCAM_INDEX (0, 1, 2, ...), and make sure no "
                "other app (Zoom, Camera, etc.) is already using the camera."
            )

        dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

        prev_error = 0.0
        last_time = time.monotonic()
        previous_target = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed -- webcam may have disconnected.")
                break

            if FRAME_ROTATION is not None:
                frame = cv2.rotate(frame, FRAME_ROTATION)
            h, w = frame.shape[:2]

            now = time.monotonic()
            dt = now - last_time
            last_time = now

            corners, centroid = detect_tag(detector, frame)

            if corners is None:
                car.motor_stop(motor=le.MOTOR_BOTH, blocking=True)
                target = 0
                prev_error = 0.0
                previous_target = 0.0
            else:
                drive, prev_error = policy(centroid, w, prev_error, dt)
                if INVERT_DRIVE_DIRECTION:
                    drive = -drive
                # Both wheels get the SAME signed speed -- straight
                # driving, no differential steering (see module docstring).
                target = int(max(-MAX_SPEED, min(MAX_SPEED, MAX_SPEED * drive)))

                cv2.polylines(frame, [corners.astype(int)], isClosed=True, color=(0, 0, 255), thickness=3)
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"centroid=({cx},{cy}) error={prev_error:.0f}",
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)  # centering reference
            cv2.imshow("AprilTag centering (webcam) -- press q to stop", frame)

            if previous_target == target:
                pass
            else:
                car.run(target)
                previous_target = target

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Hard stop independent of whatever the last frame read --
        # explicit MOTOR_BOTH (lelib's stop() only stops motor index 0).
        car.motor_stop(motor=le.MOTOR_BOTH, blocking=True)
        car.disconnect()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()