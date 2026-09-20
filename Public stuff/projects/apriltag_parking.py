"""
AprilTag parking
=================

Drives a LEGO Education Double Motor "car" up to a printed AprilTag and
centers itself on it, using an iPhone camera stream (mounted vertically
on the car) instead of a laptop webcam -- see apriltag_stream_test.py,
which validates the same STREAM_URL in isolation.

The policy (controller)
------------------------
Two independent proportional (P) controllers, one per axis, re-derived
fresh every frame from the tag's current position/size -- no memory of
past frames beyond the last commanded speed:

    x_error = tag_centroid_x - frame_center_x   (0 if within DEADZONE_PX)
    turn    = STEER_GAIN   * x_error / (frame_width / 2)   -- normalized to [-1, 1]
    forward = FORWARD_GAIN * (TARGET_TAG_AREA - tag_area)  -- raw pixel^2 units

Note turn operates on a [-1, 1]-normalized horizontal fraction while
forward operates on raw pixel^2 area error -- the two gains aren't on
the same scale, so don't expect STEER_GAIN and FORWARD_GAIN to mean
"the same amount of response" numerically.

turn steers toward centering the tag horizontally; forward drives
toward a target apparent tag size (a stand-in for target distance,
since a bigger tag means the car is closer). Combined via the same
arcade-drive mixing used elsewhere in this repo:

    target_left  = forward + turn
    target_right = forward - turn

Why this can overshoot and come back ("spring-loaded")
--------------------------------------------------------
A pure P controller's output is proportional to error -- exactly
Hooke's law, F = -kx, treating position error like a spring's
displacement. A mass on a spring with too little damping doesn't
settle at rest, it oscillates around the target with decaying (or even
growing) amplitude. This car has real physical momentum, and there's
inherent camera -> detection -> BLE latency in this loop (the "force"
gets applied late relative to the "displacement" it was measured at) --
both are exactly the ingredients that make a P-only controller overshoot
a setpoint in practice, not just in theory. STEER_GAIN/FORWARD_GAIN are
deliberately exposed so you can tune this directly: raise them ("stiffer
spring") to make the overshoot-and-return more pronounced, lower them
to damp it out.

What happens with no tag detected
-----------------------------------
Both motors stop immediately -- no "last known command," no search/spin
behavior. This is a deliberate, simple choice (see this project's
Notion page for the reasoning), not an oversight.

Camera pan platform (manual, not part of the AprilTag policy)
----------------------------------------------------------------
The phone sits on a rotating platform driven by its own Single Motor,
independent of the Double Motor "car" above. That's manual only --
there's no autonomous pan-seeking behavior here, just a Controller's
left joystick teleoperating the platform every frame so a human can
re-aim the camera at the tag if it drifts out of frame. All three
devices (Double Motor, Single Motor, Controller) connect using the
*same* card serial/color -- one physical Connection Card tapped to all
three, per this project's hardware setup.

Setup:
    1. Get the iPhone stream working first via apriltag_stream_test.py
       -- this script uses the same STREAM_URL.
    2. Print an AprilTag from the 36h11 family and stick it on a wall
       or box at the car's camera height.
    3. Calibrate TARGET_TAG_AREA below: run this script, note the live
       "area=" readout in the video overlay at the distance you want
       the car to stop, and set TARGET_TAG_AREA to that value.

Run:
    my_env/Scripts/python "Public stuff/projects/apriltag_parking.py"
"""
import os
import sys

import cv2
import numpy as np

# lelib.py/camlib.py live in the shared "useful libraries" folder, not next
# to this script -- add it to sys.path so the imports below resolve no
# matter where this script is run from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "useful libraries"))

import legoeducation as le
from lelib import doubleMotor, singleMotor, controller

# --- Camera source -- same URL validated in apriltag_stream_test.py --------
STREAM_URL = "http://10.243.67.39:8080/stream.jpeg"  # <-- update if the phone's IP changes

# Some streaming apps send the camera's native sensor orientation without
# correcting for how the phone is physically mounted, so a vertically-
# mounted phone can produce a frame rotated 90 degrees from upright. This
# is NOT just cosmetic here: policy()'s "turn" reads horizontal pixel
# offset, so an unrotated frame could have it actually reacting to the
# tag's real-world VERTICAL position instead. Match whatever value you
# settled on in apriltag_stream_test.py (verify there first: panning the
# phone left/right should pan the image left/right, not up/down).
# None = no rotation; otherwise cv2.ROTATE_90_CLOCKWISE /
# cv2.ROTATE_90_COUNTERCLOCKWISE / cv2.ROTATE_180.
FRAME_ROTATION = None  # phone remounted landscape -- re-verify with the pan test if this changes again

# --- Hardware placeholder ---------------------------------------------------
# All three devices below (Double Motor "car", Single Motor "pan platform",
# Controller) are set up to connect via the *same* Connection Card -- fill
# in the one card's serial/color here rather than a separate constant per
# device. None = connect to the first advertising device of that type. Fine
# for solo testing, but ambiguous with more than one of the same device type
# nearby -- set a real card_serial/card_color (see your Connection Card) for
# the demo.
CARD_SERIAL = 3664  # <-- fill in with your shared Connection Card's serial number
CARD_COLOR = le.LEGO_COLOR_RED

# Camera pan platform's joystick teleop -- left stick drives the Single
# Motor directly at a speed proportional to how far it's pushed (same
# convention as single_motor_controller.py). Flip if the platform spins
# the wrong way relative to stick direction once you watch it move.
INVERT_PAN_MOTOR = False

# --- AprilTag ----------------------------------------------------------------
APRILTAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11
TARGET_TAG_ID = None  # None = react to any detected tag; set an int to track one specific ID

# --- Controller (the "policy") tuning ----------------------------------------
MAX_SPEED = 50          # top motor speed as a percent (0-100)
STEER_GAIN = 0.1      # turn response to horizontal centering error -- raise for a stiffer "spring"
FORWARD_GAIN = 0.00006  # drive response to tag-size (distance) error -- same idea, other axis
TARGET_TAG_AREA = 18000  # pixels^2 -- CALIBRATE THIS (see Setup step 3 above)
DEADZONE_PX = 12        # +/- pixel band around center that reads as "already centered"

# Motor mounting may be mirrored side-to-side on this chassis, so
# "clockwise" on one side can drive that wheel backward. Flip whichever
# one goes the wrong way once you watch the car actually move.
INVERT_LEFT_MOTOR = False
INVERT_RIGHT_MOTOR = True

# Separate from the two flags above: those calibrate each wheel's own
# forward/backward sign, but turning direction depends on which wheel
# gets the +turn vs -turn term, which is an independent calibration.
# If forward/backward is correct but turning left/right comes out
# backward, flip this -- NOT the INVERT_*_MOTOR flags above (that would
# re-break forward/backward, since those apply the same way regardless
# of whether the command came from driving straight or turning).
INVERT_TURN_DIRECTION = True


def _drive_wheel(car, motor_side, speed, invert, blocking=False):
    """Set one Double Motor wheel to a signed speed (-100..100).

    lelib's run_left()/run_right() hardcode counter-clockwise-only
    rotation, so bidirectional per-wheel control goes through the raw
    legoeducation motor_run() directly instead -- its speed= kwarg sets
    the speed and starts rotation in a single BLE command.
    """
    if invert:
        speed = -speed
    direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if speed >= 0
                 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
    car.motor_run(direction=direction, motor=motor_side, speed=abs(speed), blocking=blocking)

def _drive_pan_motor(pan_motor, speed, invert):
    """Set the camera pan platform's Single Motor to a signed speed
    (-100..100), or stop it at 0.

    Non-blocking for the same reason as _drive_wheel(): this gets called
    from inside the vision loop, so a manual joystick nudge shouldn't
    stall frame processing waiting on a BLE ack.
    """
    if invert:
        speed = -speed
    if speed == 0:
        pan_motor.motor_stop(blocking=False)
        return
    direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if speed >= 0
                 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
    pan_motor.motor_run(direction=direction, speed=abs(speed), blocking=False)

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


def policy(centroid, area, frame_width):
    """The controller: map the tag's current position/size to a signed
    (forward, turn) command pair, each clamped to [-1, 1]. See the
    module docstring for why a P-only controller like this can
    overshoot and swing back around the setpoint on real hardware."""
    x_error = centroid[0] - frame_width / 2
    if abs(x_error) < DEADZONE_PX:
        x_error = 0
    turn = max(-1.0, min(1.0, STEER_GAIN * x_error / (frame_width / 2)))

    size_error = TARGET_TAG_AREA - area
    forward = max(-1.0, min(1.0, FORWARD_GAIN * size_error))

    return forward, turn


def main():
    car = doubleMotor()
    pan_motor = singleMotor()
    ctl = controller()

    # cap/detector, and all three connect() calls, are done *inside* the
    # try below (not here) so that a failure partway through setup (e.g.
    # car connects fine but the Controller doesn't) still reaches the
    # finally block and disconnects whatever already connected, instead
    # of leaking a BLE connection.
    cap = None

    try:
        car.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        if not car.connected:
            raise ConnectionError("Double Motor did not connect.")

        pan_motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        if not pan_motor.connected:
            raise ConnectionError("Single Motor (camera pan platform) did not connect.")

        ctl.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        if not ctl.connected:
            raise ConnectionError("Controller did not connect.")

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

        last_left, last_right = 0, 0
        last_pan_speed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed -- stream may have dropped.")
                break

            if FRAME_ROTATION is not None:
                frame = cv2.rotate(frame, FRAME_ROTATION)
            h, w = frame.shape[:2]

            corners, centroid = detect_tag(detector, frame)

            if corners is None:
                # No tag visible this frame -- stop, don't coast on the
                # last command (see module docstring).
                target_left, target_right = 0, 0
            else:
                area = cv2.contourArea(corners.astype(np.float32))
                forward, turn = policy(centroid, area, w)
                if INVERT_TURN_DIRECTION:
                    turn = -turn
                target_left = int(max(-MAX_SPEED, min(MAX_SPEED, MAX_SPEED * (forward + turn))))
                target_right = int(max(-MAX_SPEED, min(MAX_SPEED, MAX_SPEED * (forward - turn))))

                cv2.polylines(frame, [corners.astype(int)], isClosed=True, color=(0, 0, 255), thickness=3)
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"centroid=({cx},{cy}) area={area:.0f}",
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)  # centering reference
            cv2.imshow("AprilTag parking -- press q to stop", frame)

            if target_left != last_left:
                _drive_wheel(car, le.MOTOR_LEFT, target_left, INVERT_LEFT_MOTOR)
                last_left = target_left
            if target_right != last_right:
                _drive_wheel(car, le.MOTOR_RIGHT, target_right, INVERT_RIGHT_MOTOR)
                last_right = target_right

            # Manual camera pan platform -- independent of the AprilTag
            # policy above, polled every frame off the Controller's left
            # stick. Released stick reads as 0 (stop), matching
            # single_motor_controller.py's convention.
            pan_speed = 0 if ctl.left_released() else ctl.left_position()
            if pan_speed != last_pan_speed:
                _drive_pan_motor(pan_motor, pan_speed, INVERT_PAN_MOTOR)
                last_pan_speed = pan_speed

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Hard stop independent of whatever the last frame read --
        # explicit MOTOR_BOTH (lelib's stop() only stops motor index 0).
        car.motor_stop(motor=le.MOTOR_BOTH, blocking=True)
        pan_motor.motor_stop(blocking=True)
        car.disconnect()
        pan_motor.disconnect()
        ctl.disconnect()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
