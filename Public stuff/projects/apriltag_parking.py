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

Camera pan platform: SEARCH -> ALIGN -> DRIVE
------------------------------------------------
The phone sits on a rotating platform driven by its own Single Motor,
independent of the Double Motor "car." Unlike a plain teleop rig, this
script runs a 3-state pipeline every frame:

    SEARCH  -- the car sits still. The platform is teleoperated by the
               Controller's left joystick so a human can hunt for the
               tag. The instant detect_tag() sees it -> ALIGN.

    ALIGN   -- fully autonomous, no forward motion. Two P-controllers
               run concurrently:
                 * the pan motor keeps the tag centered in-frame
                   (vision P-control, PAN_STEER_GAIN);
                 * the car turns in place, using the pan platform's own
                   accumulated relative rotation (pan_motor.motor.position,
                   in degrees) as its error signal -- that reading IS the
                   bearing from the car's forward direction to the tag,
                   because the pan platform's zero point is calibrated
                   to "dead ahead" once at startup (see Setup below).
               Running both loops at once is what produces the
               counter-rotation: as the car turns, the tag drifts in the
               pan camera's view, and the pan loop corrects by rotating
               the platform back toward center -- an automatic,
               equal-and-opposite counter-rotation relative to the car's
               own turn. No IMU/yaw math needed. Once the pan platform's
               angle is back within ALIGN_TOLERANCE_DEG of zero (car is
               now facing the tag directly) -> DRIVE. Losing the tag
               anywhere in ALIGN -> back to SEARCH.

    DRIVE   -- unchanged from before: the pan platform is held
               stationary (its job is done) and the car drives using the
               blended forward+turn policy() below. Losing the tag ->
               back to SEARCH.

All three devices (Double Motor, Single Motor, Controller) connect using
the *same* card serial/color -- one physical Connection Card tapped to
all three, per this project's hardware setup.

Setup:
    1. Get the iPhone stream working first via apriltag_stream_test.py
       -- this script uses the same STREAM_URL.
    2. Print an AprilTag from the 36h11 family and stick it on a wall
       or box at the car's camera height.
    3. Calibrate TARGET_TAG_AREA below: run this script, note the live
       "area=" readout in the video overlay at the distance you want
       the car to stop, and set TARGET_TAG_AREA to that value.
    4. Before pressing run, manually point the camera platform dead
       ahead so it matches the car's own front -- main() zeroes the
       platform's relative-position counter once at startup (see
       pan_motor.motor_reset_relative_position() below) under the
       assumption that "0" means "facing the same way as the car."
       Skipping this makes ALIGN turn toward the wrong heading.

Run:
    my_env/Scripts/python "Public stuff/projects/apriltag_parking.py"
"""
import os
import sys
import threading

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

# --- ALIGN state tuning (autonomous pan-vision + car turn-to-heading) ------
# Pan motor's vision P-gain during ALIGN -- same shape as STEER_GAIN's
# horizontal-centering term but a separate constant, since this one
# drives the pan motor's speed directly instead of contributing to the
# car's turn command, and the two motors have very different speed/
# torque characteristics. Reuses DEADZONE_PX (same camera, same notion
# of "close enough to centered").
PAN_STEER_GAIN = 0.1
PAN_MAX_SPEED = 40  # cap on the pan motor's autonomous (ALIGN) speed

# ALIGN's car-turn P-gain: maps pan_motor.motor.position (degrees the
# pan platform has had to rotate away from "dead ahead" to keep the tag
# centered) into a normalized turn command for the car. Raise for a
# more aggressive turn per degree of pan error.
ALIGN_TURN_GAIN = 0.02

# ALIGN is a deliberately slow, cautious turn-in-place (see module
# docstring) -- capped well below MAX_SPEED so the counter-rotating pan
# vision loop has time to keep tracking the tag while the car turns,
# instead of the car swinging past it.
ALIGN_MAX_SPEED = 20

# How close pan_motor.motor.position must be to 0 (degrees) before the
# car is considered "facing the tag" and control hands off to DRIVE.
# Too tight and ordinary pan-loop jitter around center may never
# satisfy it; too loose and DRIVE starts while still visibly off-axis.
ALIGN_TOLERANCE_DEG = 5

# Separate from INVERT_TURN_DIRECTION above (which calibrates
# pixel-error-sign -> wheel-turn-sign for DRIVE's steering): this
# calibrates pan-POSITION-error-sign -> wheel-turn-sign for ALIGN's
# turn-to-heading. A different pair of signals that could need opposite
# calibration even on the same physical wheels -- flip this (not
# INVERT_TURN_DIRECTION) if the car turns away from the tag instead of
# toward it during ALIGN.
INVERT_ALIGN_TURN_DIRECTION = False

# Firmware-level ramp (0-100) for each motor, set once at startup, so
# neither motor snaps instantly to a newly commanded speed -- helps
# both ALIGN's "slowly turn" ask and general phone-mount safety. Both
# _drive_wheel and _drive_pan_motor dispatch through the raw per-motor
# motor_run() channel, so motor_set_acceleration() (not
# movement_set_acceleration(), which is a separate command channel for
# the higher-level "movement" API this script doesn't use) is correct.
MOTOR_ACCEL = 10
MOTOR_DECEL = 10
PAN_ACCEL = 10
PAN_DECEL = 10

# How often (seconds) each background motor-command thread checks for a
# new target and, if it changed, sends it. Decoupled from camera frame
# rate on purpose -- see _MotorDriver below.
MOTOR_COMMAND_PERIOD_S = 0.02


class _MotorDriver:
    """Sends motor commands from a dedicated background thread.

    Every legoeducation call -- motor_run, motor_stop, etc. -- blocks
    its calling thread until the BLE write completes (see
    legoeducation/_platform.py's _run_sync: it always waits on a
    concurrent.futures.Future, regardless of that call's own blocking=
    kwarg, which only affects whether the *coroutine* also waits for a
    completion ack). If the vision-loop thread issued those calls
    directly, every BLE round-trip would stall cap.read()/cv2.imshow()
    along with it.

    Generic over what "a command" means via send_fn(target): the vision
    loop just calls set_target() (a plain attribute write under a lock,
    effectively instant), and this thread notices the change and sends
    it whenever it's ready to. Both the car (target = a (left, right)
    tuple) and the pan motor (target = a plain speed) reuse this same
    class instead of bespoke threading code per device.
    """

    def __init__(self, send_fn, initial_target, period_s=MOTOR_COMMAND_PERIOD_S):
        self._send_fn = send_fn
        self._period_s = period_s
        self._lock = threading.Lock()
        self._target = initial_target
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
                    self._send_fn(target)
                    self._sent = target
                except Exception as e:
                    print(f"Motor command failed: {e}")
            self._stop_event.wait(self._period_s)

    def stop(self):
        """Stop the background thread. Doesn't itself stop the motors --
        call motor_stop() separately once this has returned."""
        self._stop_event.set()
        self._thread.join(timeout=1)


def _drive_wheel(car, motor_side, speed, invert):
    """Set one Double Motor wheel to a signed speed (-100..100).

    lelib's run_left()/run_right() hardcode counter-clockwise-only
    rotation, so bidirectional per-wheel control goes through the raw
    legoeducation motor_run() directly instead -- its speed= kwarg sets
    the speed and starts rotation in a single BLE command. Always
    blocking now -- safe because this is only ever called from a
    _MotorDriver's own background thread, never directly from the
    vision loop.
    """
    if invert:
        speed = -speed
    direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if speed >= 0
                 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
    car.motor_run(direction=direction, motor=motor_side, speed=abs(speed))

def _drive_pan_motor(pan_motor, speed, invert):
    """Set the camera pan platform's Single Motor to a signed speed
    (-100..100), or stop it at 0. Always blocking -- see _drive_wheel."""
    if invert:
        speed = -speed
    if speed == 0:
        pan_motor.motor_stop()
        return
    direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if speed >= 0
                 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
    pan_motor.motor_run(direction=direction, speed=abs(speed))

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


STATE_SEARCH, STATE_ALIGN, STATE_DRIVE = "SEARCH", "ALIGN", "DRIVE"


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
    car_driver = None
    pan_driver = None

    try:
        car.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        if not car.connected:
            raise ConnectionError("Double Motor did not connect.")
        car.motor_set_acceleration(MOTOR_ACCEL, MOTOR_DECEL, motor=le.MOTOR_BOTH)

        pan_motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        if not pan_motor.connected:
            raise ConnectionError("Single Motor (camera pan platform) did not connect.")
        pan_motor.motor_set_acceleration(PAN_ACCEL, PAN_DECEL)
        # Zero the pan platform's relative-position counter here, under
        # the documented assumption (see module docstring/Setup) that
        # the camera is manually pointed dead ahead, matching the car's
        # own front, at this moment. This is what makes
        # pan_motor.motor.position usable later as "the car's heading
        # error relative to the tag" during ALIGN.
        pan_motor.motor_reset_relative_position()

        ctl.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        if not ctl.connected:
            raise ConnectionError("Controller did not connect.")

        car_driver = _MotorDriver(
            lambda t: (_drive_wheel(car, le.MOTOR_LEFT, t[0], INVERT_LEFT_MOTOR),
                       _drive_wheel(car, le.MOTOR_RIGHT, t[1], INVERT_RIGHT_MOTOR)),
            initial_target=(0, 0))
        pan_driver = _MotorDriver(
            lambda t: _drive_pan_motor(pan_motor, t, INVERT_PAN_MOTOR),
            initial_target=0)

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

        state = STATE_SEARCH

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed -- stream may have dropped.")
                break

            if FRAME_ROTATION is not None:
                frame = cv2.rotate(frame, FRAME_ROTATION)
            h, w = frame.shape[:2]

            corners, centroid = detect_tag(detector, frame)
            tag_visible = corners is not None
            if tag_visible:
                area = cv2.contourArea(corners.astype(np.float32))

                cv2.polylines(frame, [corners.astype(int)], isClosed=True, color=(0, 0, 255), thickness=3)
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"centroid=({cx},{cy}) area={area:.0f}",
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Each branch re-checks tag_visible itself and falls back to
            # SEARCH inline, so exactly one state's motor commands are
            # ever issued per frame (see module docstring for the
            # SEARCH -> ALIGN -> DRIVE pipeline this implements).
            if state == STATE_SEARCH:
                car_driver.set_target((0, 0))
                # Manual camera pan platform, polled every frame off the
                # Controller's left stick. Released stick reads as 0
                # (stop), matching single_motor_controller.py's convention.
                pan_speed = 0 if ctl.left_released() else ctl.left_position()
                pan_driver.set_target(pan_speed)
                if tag_visible:
                    state = STATE_ALIGN

            elif state == STATE_ALIGN:
                if not tag_visible:
                    car_driver.set_target((0, 0))
                    pan_driver.set_target(0)
                    state = STATE_SEARCH
                else:
                    # Pan motor: vision P-control keeps the tag centered
                    # in-frame -- same shape as policy()'s turn term.
                    x_error = centroid[0] - w / 2
                    if abs(x_error) < DEADZONE_PX:
                        x_error = 0
                    pan_cmd = max(-1.0, min(1.0, PAN_STEER_GAIN * x_error / (w / 2)))
                    pan_driver.set_target(int(max(-PAN_MAX_SPEED, min(PAN_MAX_SPEED, PAN_MAX_SPEED * pan_cmd))))

                    # Car: turn-in-place P-control using the pan
                    # platform's own accumulated rotation as the error --
                    # see module docstring for why running both loops
                    # concurrently produces the counter-rotation effect.
                    pan_angle = pan_motor.motor.position
                    turn = max(-1.0, min(1.0, ALIGN_TURN_GAIN * pan_angle))
                    if INVERT_ALIGN_TURN_DIRECTION:
                        turn = -turn
                    target_left = int(max(-ALIGN_MAX_SPEED, min(ALIGN_MAX_SPEED, ALIGN_MAX_SPEED * turn)))
                    car_driver.set_target((target_left, -target_left))  # pure turn-in-place

                    if abs(pan_angle) < ALIGN_TOLERANCE_DEG:
                        pan_driver.set_target(0)
                        state = STATE_DRIVE

            elif state == STATE_DRIVE:
                pan_driver.set_target(0)  # platform's job is done -- hold it stationary
                if not tag_visible:
                    car_driver.set_target((0, 0))
                    state = STATE_SEARCH
                else:
                    forward, turn = policy(centroid, area, w)
                    if INVERT_TURN_DIRECTION:
                        turn = -turn
                    car_driver.set_target((
                        int(max(-MAX_SPEED, min(MAX_SPEED, MAX_SPEED * (forward + turn)))),
                        int(max(-MAX_SPEED, min(MAX_SPEED, MAX_SPEED * (forward - turn)))),
                    ))

            cv2.putText(frame, f"state={state}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # live debug readout
            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)  # centering reference
            cv2.imshow("AprilTag parking -- press q to stop", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop both motor-command threads first so neither can race the
        # hard stop below, then hard-stop independent of whatever the
        # last frame read -- explicit MOTOR_BOTH (lelib's stop() only
        # stops motor index 0).
        if car_driver is not None:
            car_driver.stop()
        if pan_driver is not None:
            pan_driver.stop()
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
