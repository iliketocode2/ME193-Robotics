"""
AprilTag centering (drive straight only)
==========================================

Simplified sibling of apriltag_parking.py. Assumptions here are
different from that script:

    - The camera is mounted 90 degrees off the car's front (looking out
      to the side, not straight ahead), and the car sits parallel to
      the wall/plane the tag is mounted on. That means the car doesn't
      need to turn at all -- sliding straight forward/backward along
      that wall is what moves the tag left/right in frame.
    - The car is a plain Double Motor -- no pan platform, no
      Controller, nothing else connected.

So the only goal is: drive straight (both wheels the same signed
speed, no differential steering) until the AprilTag's centroid is
horizontally centered in frame. No turning, no distance/area logic at
all -- just forward/backward drive, driven by a PD controller.

The PD math itself is left for you to fill in -- see the
"PD CONTROL -- FILL IN" block inside policy() below. error (pixels off
center) and error_rate (pixels/sec) are already computed for you.

What happens with no tag detected
-----------------------------------
Both motors stop immediately -- no "last known command," no search/spin
behavior, same as apriltag_parking.py.

Setup:
    1. Get the iPhone stream working first via apriltag_stream_test.py
       -- this script uses the same STREAM_URL.
    2. Print an AprilTag from the 36h11 family and mount it on the
       wall/plane the car will drive parallel to, at camera height.
    3. Fill in Kp/Kd in policy() below.

Run:
    my_env/Scripts/python "Public stuff/projects/apriltag_center.py"
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

# --- Camera source -- same URL validated in apriltag_stream_test.py --------
STREAM_URL = "http://10.243.67.39:8080/stream.jpeg"  # <-- update if the phone's IP changes

# None = no rotation; otherwise cv2.ROTATE_90_CLOCKWISE /
# cv2.ROTATE_90_COUNTERCLOCKWISE / cv2.ROTATE_180. Verify with
# apriltag_stream_test.py first -- panning the camera left/right should
# pan the image left/right, not up/down, since policy() below only
# reads horizontal pixel offset.
FRAME_ROTATION = None

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
# what policy() outputs. This is the real fix for a jerky/violent car --
# without it, every new motor_run() speed is applied instantly, so a
# noisy one-frame derivative spike (see policy()'s Kd term) snaps the
# motor straight to it. Lower = gentler ramp = less jerk, slower to
# react. Start low if the phone mount is fragile; raw legoeducation
# method, not wrapped by lelib.
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
    Kp = 0.0075 # <-- proportional gain
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

        cap = cv2.VideoCapture(STREAM_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open stream at {STREAM_URL!r}. Re-verify it with "
                "apriltag_stream_test.py first -- this script assumes that "
                "part already works."
            )

        dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

        prev_error = 0.0
        last_time = time.monotonic()
        previous_target = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed -- stream may have dropped.")
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
            cv2.imshow("AprilTag centering -- press q to stop", frame)

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
