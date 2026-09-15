"""
Finger-controlled LEGO race car
================================

Two raised hands in front of the webcam drive a LEGO Education Double
Motor "car": your RIGHT hand's finger count (0-5, a fist is 0) sets the
overall speed, and your LEFT hand's pointing direction (wrist -> palm,
as a compass: up=forward, down=backward, left/right=turn) steers it.
The two are combined with "arcade drive" mixing -- the same forward+turn
-> left/right-wheel-speed scheme used on most differential-drive robots
-- so pointing straight up drives both wheels forward, pointing left
spins the car left, and a diagonal blends the two smoothly. Either hand
missing = stop, not "last known speed".

Assignment questions
---------------------
How is Python talking to the LEGO hardware?
    Through legoeducation (installed in my_env), which talks to the
    Hub over Bluetooth Low Energy (BLE, via the `bleak` library).
    lelib.py's doubleMotor subclasses legoeducation's DoubleMotor, so
    every call here (motor_set_speed, motor_run, motor_stop, ...) is
    serialized into an RPC message and sent to the hub firmware over a
    BLE GATT characteristic.

Is it synchronous or asynchronous?
    BLE commands are synchronous/blocking by default: each call waits
    for a firmware acknowledgment before returning. This script passes
    blocking=False on the per-frame drive commands in _drive_wheel()
    so the camera/MediaPipe loop isn't throttled by BLE round-trip
    latency, but the final motor_stop() uses blocking=True so it is
    guaranteed to reach the hub before the program exits.

How was it trained, and what are the limitations?
    No model is trained here. MediaPipe's HandLandmarker is Google's
    pretrained hand-pose model (21 landmarks/hand); this script only
    runs inference on it, plus two hand-crafted heuristics on top of
    the landmarks (see _count_extended_fingers and _pointing_direction).
    Limitations: single 2D camera view loses hands under occlusion, low
    light, or fast motion; the four-finger "extended" check assumes the
    hand is roughly upright facing the camera (it compares each
    fingertip's height to its own knuckle, which breaks down if the
    hand is rotated sideways or tilted toward/away from the camera);
    the thumb uses a different, orientation-agnostic heuristic
    (distance from the pinky's base) since its extension direction
    isn't vertical like the other four; the pointing direction is the
    wrist-to-palm vector, not true arm/forearm pose, so it only
    captures wrist orientation, not full arm posture; correct left/
    right handedness depends on the frame being mirrored (handled
    below by flipping every frame); and there's inherent camera +
    inference + BLE latency between a gesture and the wheels
    responding.

Run:
    my_env/Scripts/python "Public stuff/useful libraries/arm_race_control.py"
"""

import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import legoeducation as le
from lelib import doubleMotor
from camlib import pick_camera
from piviz import send_telemetry  # optional: live dashboard on a Raspberry Pi, see piviz.py

# --- Hardware placeholder ---------------------------------------------------
# None = connect to the first advertising Double Motor. Fine for solo
# testing, but ambiguous with more than one Double Motor nearby -- set
# a real card_serial/card_color (see your Connection Card) for the demo.
CAR_CARD_SERIAL = 1126  # <-- fill in with your Double Motor card's serial number
CAR_CARD_COLOR = le.LEGO_COLOR_GREEN

# --- Control tuning ----------------------------------------------------------
MAX_SPEED = 95  # top motor speed as a percent (0-100), at 5 fingers + full-forward pointing

# Motor mounting is mirrored side-to-side on the chassis, so "clockwise"
# on one side can drive that wheel backward. Flip whichever one goes the
# wrong way once you watch the car actually move.
INVERT_LEFT_MOTOR = False
INVERT_RIGHT_MOTOR = True

# --- MediaPipe hand-landmark model -------------------------------------------
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
# Stored outside "Public stuff/" (repo root's .gitignore ignores everything
# except Public stuff/**) so this ~8MB download never lands in git.
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "hand_landmarker.task"
)


def _ensure_model():
    """Download the pretrained hand-landmark model on first run."""
    path = os.path.abspath(_MODEL_PATH)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading MediaPipe hand landmark model...")
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


# MediaPipe's 21 hand landmarks, by index (only the ones used below).
_WRIST = 0
_THUMB_IP, _THUMB_TIP = 3, 4
_INDEX_PIP, _INDEX_TIP = 6, 8
_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_TIP = 9, 10, 12
_RING_PIP, _RING_TIP = 14, 16
_PINKY_MCP, _PINKY_PIP, _PINKY_TIP = 17, 18, 20


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _count_extended_fingers(landmarks):
    """Count extended fingers (0-5) from one hand's 21 MediaPipe landmarks.

    Index/middle/ring/pinky: "extended" if the fingertip sits above (a
    smaller image-y than) its own PIP joint -- reliable as long as the
    hand is roughly upright facing the camera, which matches how this
    script is used (hand raised toward the webcam).

    Thumb: its extension direction is sideways, not vertical, and which
    side depends on handedness and hand rotation -- rather than branch
    on that, compare each joint's distance from the pinky's base (17).
    The tip ends up farther from that point than the IP joint only when
    the thumb is actually splayed out from the palm, regardless of
    which hand or which way it's rotated.
    """
    tips_pips = ((_INDEX_TIP, _INDEX_PIP), (_MIDDLE_TIP, _MIDDLE_PIP),
                 (_RING_TIP, _RING_PIP), (_PINKY_TIP, _PINKY_PIP))
    count = sum(1 for tip, pip in tips_pips if landmarks[tip].y < landmarks[pip].y)

    pinky_base = landmarks[_PINKY_MCP]
    if _dist(landmarks[_THUMB_TIP], pinky_base) > _dist(landmarks[_THUMB_IP], pinky_base):
        count += 1
    return count


def _pointing_direction(landmarks):
    """Unit-vector (forward, turn) for the direction a hand is pointing,
    read as a compass: up=forward (+1), down=backward (-1), right=turn
    right (+1), left=turn left (-1); a diagonal blends both.

    Uses the wrist -> middle-finger-knuckle vector (the palm's own
    orientation) rather than a fingertip, so it stays stable even when
    the fingers are curled into a fist.
    """
    wrist = landmarks[_WRIST]
    palm_forward = landmarks[_MIDDLE_MCP]
    dx = palm_forward.x - wrist.x
    dy = palm_forward.y - wrist.y
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, 0.0
    return -dy / length, dx / length  # (forward, turn)


def _drive_wheel(car, motor_side, speed, invert, blocking=False):
    """Set one Double Motor wheel to a signed speed (-100..100).

    lelib's run_left()/run_right() hardcode counter-clockwise-only
    rotation, so bidirectional per-wheel control goes through the raw
    legoeducation motor_run() directly instead -- its speed= kwarg sets
    the speed and starts rotation in a single BLE command, so a separate
    motor_set_speed() call first isn't needed.
    """
    if invert:
        speed = -speed
    direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if speed >= 0
                 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
    car.motor_run(direction=direction, motor=motor_side, speed=abs(speed), blocking=blocking)


def main():
    model_path = _ensure_model()

    car = doubleMotor()
    car.connect(card_serial=CAR_CARD_SERIAL, card_color=CAR_CARD_COLOR)

    # cap/landmarker are opened *inside* the try below (not here) so that a
    # failure in either -- e.g. pick_camera() finding no camera, or a
    # corrupted hand_landmarker.task -- still reaches the finally block and
    # disconnects the Double Motor instead of leaking the BLE connection.
    cap = None
    landmarker = None

    try:
        if not car.connected:
            raise ConnectionError("Double Motor did not connect.")

        cap, start_ms = pick_camera()

        landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
            )
        )

        last_left, last_right = 0, 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view so on-screen L/R match the driver's own hands
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000) - start_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Either hand missing = stop, not "last known speed" -- only set once
            # both a speed (right hand) and a direction (left hand) are seen below.
            right_speed = None
            right_fingers = 0
            left_direction = None  # (forward, turn), each in [-1, 1]

            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                side = handedness[0].category_name  # "Left" or "Right"
                wrist = landmarks[_WRIST]
                wx, wy = int(wrist.x * w), int(wrist.y * h)

                if side == "Right":
                    fingers = _count_extended_fingers(landmarks)
                    right_fingers = fingers
                    right_speed = (fingers / 5) * MAX_SPEED
                    cv2.circle(frame, (wx, wy), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"Right: {fingers} fingers -> {right_speed:+.0f}%",
                                (wx + 12, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    left_direction = _pointing_direction(landmarks)
                    palm = landmarks[_MIDDLE_MCP]
                    px, py = int(palm.x * w), int(palm.y * h)
                    # Extend the wrist->palm vector well past the hand so the
                    # pointing direction reads clearly on screen.
                    ex, ey = wx + (px - wx) * 4, wy + (py - wy) * 4
                    cv2.arrowedLine(frame, (wx, wy), (int(ex), int(ey)), (255, 0, 0), 3, tipLength=0.3)
                    cv2.putText(frame, f"Left: fwd {left_direction[0]:+.2f} turn {left_direction[1]:+.2f}",
                                (wx + 12, wy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            forward, turn = left_direction if left_direction is not None else (0.0, 0.0)

            if right_speed is not None and left_direction is not None:
                # Arcade-drive mixing: forward moves both wheels together,
                # turn spins them apart -- the standard forward+turn ->
                # left/right-wheel-speed scheme for a differential-drive robot.
                target_left = int(max(-MAX_SPEED, min(MAX_SPEED, right_speed * (forward + turn))))
                target_right = int(max(-MAX_SPEED, min(MAX_SPEED, right_speed * (forward - turn))))
            else:
                target_left, target_right = 0, 0

            # Optional fun add-on -- see piviz.py. Best-effort/non-blocking,
            # so a missing Raspberry Pi never affects the control loop above.
            send_telemetry(right_fingers, right_speed or 0.0, forward, turn)

            cv2.imshow("Finger Race Control -- press q to stop", frame)

            if target_left != last_left:
                _drive_wheel(car, le.MOTOR_LEFT, target_left, INVERT_LEFT_MOTOR)
                last_left = target_left
            if target_right != last_right:
                _drive_wheel(car, le.MOTOR_RIGHT, target_right, INVERT_RIGHT_MOTOR)
                last_right = target_right

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Hard stop independent of whatever the last camera frame read --
        # explicit MOTOR_BOTH (lelib's stop() only stops motor index 0).
        car.motor_stop(motor=le.MOTOR_BOTH, blocking=True)
        car.disconnect()
        # cap/landmarker may still be None if pick_camera() or the
        # landmarker failed to set up -- the motor cleanup above must still
        # run in that case, but there's nothing here to release/close.
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
