# Project 1 — Pose Race

A LEGO Education Double Motor "car" driven by raising your hands in
front of a webcam — left wrist controls the left wheel, right wrist
controls the right wheel.

## Files

- **`single_motor_controller.py`** — a foundational exercise from
  earlier in this assignment sequence: drive a single LEGO motor with a
  Controller's left joystick (push up/down to spin forward/backward,
  proportional to how far the stick is pushed). No camera, no
  MediaPipe — this is the "can I talk to one motor over Bluetooth at
  all" building block before adding vision-based control.

- **`arm_race_control.py`** — the real Pose Race controller. Uses
  Google's pretrained MediaPipe HandLandmarker (21 landmarks per hand,
  no training done here — this script only runs inference) to track
  both wrists in the webcam feed. Each wrist's height relative to an
  on-screen center line sets that side's wheel speed: raise a hand
  above the line to drive that wheel forward, lower it below to
  reverse, and how far the wrist is from the line sets the speed
  (with a deadzone around center so it's not twitchy). The frame is
  mirrored so left/right handedness matches what you see on screen.

## Setup

```bash
python -m venv my_env
my_env\Scripts\activate.bat        # Windows
source my_env/bin/activate         # macOS/Linux
pip install --upgrade pip
pip install legoeducation opencv-python mediapipe
```

- `lelib.py` / `camlib.py` (from `Public stuff/useful libraries/`) and
  `legoeducation` need to be importable from wherever you run these
  scripts — `single_motor_controller.py` is written to have `lelib.py`
  copied into the same folder as the script (see its own docstring);
  check `arm_race_control.py`'s imports resolve for your setup before
  running it (e.g. run it with `Public stuff/useful libraries/` on your
  `PYTHONPATH`, or add the same `sys.path` fix used in the Project 2
  scripts).
- Fill in the `MOTOR_CARD_*`/`CONTROLLER_CARD_*` (single motor) or
  `CAR_CARD_*` (arm control) placeholders with the color/serial printed
  on your LEGO Connection Card(s).
- `arm_race_control.py` downloads MediaPipe's hand-landmark model
  (~8MB) to `models/` at the repo root on first run — that's outside
  `Public stuff/`, so it never lands in git.
- No card needed for the webcam itself — `camlib.pick_camera()` will
  prompt you to pick one if more than one is connected.

## Run

```bash
my_env/Scripts/python "Public stuff/projects/Project 1 - Pose Race/single_motor_controller.py"
my_env/Scripts/python "Public stuff/projects/Project 1 - Pose Race/arm_race_control.py"
```

## Biggest failure / limitation

MediaPipe's hand tracking only sees a single 2D camera view, so hands
get lost under occlusion, in low light, or during fast motion — and
since only wrist *height* drives the car (no elbow/shoulder posture or
gesture vocabulary), the control scheme is intentionally simple but
also easy to fool (e.g. leaning forward changes apparent wrist height
without you meaning to speed up). There's also inherent
camera-capture + MediaPipe-inference + BLE latency between raising a
hand and the wheels actually responding, same class of lag documented
in Project 2's control loops.
