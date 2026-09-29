"""
Arm-controlled LEGO race car
============================

Two raised hands in front of the webcam drive a LEGO Education Double
Motor "car": left wrist controls the left wheel, right wrist controls
the right wheel. Raise a hand above the on-screen center line to drive
that wheel forward, lower it below the line to reverse; how far the
wrist is from the line sets the speed.

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
    runs inference on it. Limitations: single 2D camera view loses
    hands under occlusion, low light, or fast motion; only wrist
    height is used, so there's no elbow/shoulder posture or gesture
    vocabulary beyond "which hand, how high"; correct left/right
    handedness depends on the frame being mirrored (handled below by
    flipping every frame); and there's inherent camera + inference +
    BLE latency between a gesture and the wheels responding.

Run:
    my_env/Scripts/python "Public stuff/projects/Project 1 - Pose Race/arm_race_control.py"
"""

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

# --- Hardware placeholder ---------------------------------------------------
# None = connect to the first advertising Double Motor. Fine for solo
# testing, but ambiguous with more than one Double Motor nearby -- set
# a real card_serial/card_color (see your Connection Card) for the demo.
CAR_CARD_SERIAL = None
CAR_CARD_COLOR = None

# --- Control tuning ----------------------------------------------------------
MAX_SPEED = 70       # top motor speed as a percent (0-100)
CENTER_Y_FRAC = 0.5  # 0 = top of frame, 1 = bottom; the "zero speed" line
DEADZONE_FRAC = 0.06 # +/- band around the center line that reads as 0

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
# except Public stuff/**) so this ~8MB download never lands in git. Three
# levels up from this script (Public stuff/projects/Project 1 - Pose
# Race/) reaches the repo root.
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "hand_landmarker.task"
)


def _ensure_model():
    """Download the pretrained hand-landmark model on first run."""
    path = os.path.abspath(_MODEL_PATH)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading MediaPipe hand landmark model...")
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


def _wheel_speed(wrist_y_frac):
    """Map a wrist's normalized vertical position (0=top, 1=bottom) to a
    signed motor speed in [-MAX_SPEED, MAX_SPEED]."""
    offset = CENTER_Y_FRAC - wrist_y_frac  # positive = hand above center
    if abs(offset) < DEADZONE_FRAC:
        return 0
    span = CENTER_Y_FRAC if offset < 0 else (1 - CENTER_Y_FRAC)
    speed = (offset / span) * MAX_SPEED
    return max(-MAX_SPEED, min(MAX_SPEED, int(speed)))


def _drive_wheel(car, motor_side, speed, invert, blocking=False):
    """Set one Double Motor wheel to a signed speed (-100..100).

    lelib's run_left()/run_right() hardcode counter-clockwise-only
    rotation, so bidirectional per-wheel control goes through the raw
    legoeducation motor_set_speed/motor_run calls directly instead.
    """
    if invert:
        speed = -speed
    direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if speed >= 0
                 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
    car.motor_set_speed(abs(speed), motor=motor_side, blocking=blocking)
    car.motor_run(direction=direction, motor=motor_side, speed=abs(speed), blocking=blocking)


def main():
    model_path = _ensure_model()

    car = doubleMotor()
    car.connect(card_serial=CAR_CARD_SERIAL, card_color=CAR_CARD_COLOR)

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

    try:
        if not car.connected:
            raise ConnectionError("Double Motor did not connect.")

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

            # No hand visible for a side = that wheel gets 0, not "last known speed".
            target_left, target_right = 0, 0
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                wrist = landmarks[0]  # landmark 0 = wrist in MediaPipe's 21-point hand model
                speed = _wheel_speed(wrist.y)
                side = handedness[0].category_name  # "Left" or "Right"
                if side == "Left":
                    target_left = speed
                else:
                    target_right = speed

                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"{side} {speed:+d}%", (cx + 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.line(frame, (0, int(h * CENTER_Y_FRAC)), (w, int(h * CENTER_Y_FRAC)),
                      (0, 0, 255), 1)
            cv2.imshow("Arm Race Control -- press q to stop", frame)

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
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
