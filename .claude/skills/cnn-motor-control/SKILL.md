---
name: cnn-motor-control
description: Roadmap and design guidance for training a CNN to drive the LEGO Education motors in this repo (behavioral cloning / vision- or sensor-based control). Use when designing data-collection scripts, choosing a model architecture, writing a training loop, or wiring a trained model into a real-time control loop against lelib.py's doubleMotor/singleMotor. Load the lego-hardware skill alongside this one for exact API calls.
---

# CNN-based motor control — design roadmap

The class goal is advanced motor control of the LEGO Education motors
using a CNN, built on top of `lelib.py` (device control) and
`camlib.py`/`miclib.py` (sensor input). This skill is about the
*pipeline design* — data collection, architecture choice, training,
and closed-loop deployment — not the hardware API itself (see
`lego-hardware` for that).

No training pipeline exists in this repo yet. Don't assume a data
format, model file, or script name from a prior conversation — check
`Public stuff/useful libraries/` for what actually exists before
building on top of something that may not be there.

## 1. Decide what the CNN actually consumes and predicts, explicitly

Don't default to "camera image in, motor command out" without stating
it — there are several reasonable designs given what's in this repo,
and they have very different data-collection and latency implications:

| Input | Output | Notes |
|---|---|---|
| Webcam frame (`camlib`) | Tank-drive speeds (`speed_left`, `speed_right`) | Classic behavioral cloning / end-to-end vision control. Needs a CNN over images (conv stack, not an MLP) — this is the "advanced motor control" reading most in line with "CNN." |
| Webcam frame | Discrete action class (forward/left/right/stop) | Simpler than continuous regression; easier to collect balanced training data for, easier to sanity-check, less precise. |
| `colorSensor.raw_reading()` (7 floats, already 0–1 normalized) | motor command | Not actually a *convolutional* problem — a small MLP fits this better than a CNN. Say so if the user asks for a CNN here; a CNN over a 7-value non-spatial vector is architecture-mismatched, not "more advanced." |
| Stacked frames / short video clip | motor command | Adds temporal context (useful for a moving robot) at the cost of more complex data collection and a 3D-conv or CNN+RNN/temporal-pooling architecture. Only reach for this once single-frame control works. |

Ask the user which of these (or something else) they mean before
building a data collection script — "CNN for motor control" is
underspecified across this table, and the labeling scheme changes
completely depending on the answer.

## 2. Data collection design

Whatever the input modality, the label side needs **synchronized**
ground truth. Practical pattern using this repo's libraries:

```python
import time, csv
from lelib import doubleMotor, controller
from camlib import pick_camera
import cv2

dm, ctrl = doubleMotor(), controller()
dm.connect(card_serial=None)
ctrl.connect(card_serial=None)
cap, start_ms = pick_camera()

with open("run_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_path", "t_ms", "speed_left", "speed_right"])
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            left, right = ctrl.left_position(), ctrl.right_position()
            dm.movement_move_tank(left, right)     # human drives, CNN later imitates
            path = f"frames/{i:06d}.jpg"
            cv2.imwrite(path, frame)
            writer.writerow([path, int(time.time() * 1000) - start_ms, left, right])
            i += 1
            time.sleep(0.05)   # ~20 Hz — match this to inference-loop rate, see §4
    finally:
        dm.stop()
        dm.disconnect()
        ctrl.disconnect()
        cap.release()
```

Key design points to get right, and why:

- **Label = the human/controller command at the moment the frame was
  captured**, not the motor's *actual* speed a moment later — behavioral
  cloning imitates the controller, and `ctrl.left_position()`/
  `right_position()` is what's driving `movement_move_tank`. If instead
  you want to imitate observed motor behavior, log `dm.motor[0].speed`/
  `dm.motor[1].speed` (`MotorNotification.speed`) instead — decide
  which and be consistent, don't mix.
- **Timestamp everything** (`start_ms` from `camlib`) so frame and
  label are provably from the same instant, not just "close enough."
- **Balance the dataset deliberately** — a human driving mostly
  straight produces mostly "go straight" labels; a CNN trained on that
  alone won't learn to turn. Either collect deliberate turn-heavy runs
  or note the imbalance so training accounts for it (weighted loss,
  oversampling), rather than being surprised when the trained model
  never turns.
- **Collect under conditions the model will actually run in** — same
  lighting, same track/environment. A CNN generalizes poorly to visual
  conditions absent from training data; this matters more here than
  algorithm choice.

## 3. Architecture guidance

- Small conv stack (a handful of conv+pool blocks feeding into a small
  dense head) is plenty for a single low-res frame (`camlib`'s
  `width=640, height=480` or smaller — don't default to 1280×720 for
  training input, it's slower to train and rarely helps at this task
  scale).
- Output layer matches the label design from §1: 2 linear outputs
  (tanh-scaled to −100..100) for continuous tank-drive regression, or
  a softmax over discrete actions for the classification framing.
- Don't reach for a large pretrained backbone (ResNet/EfficientNet
  etc.) by default for a small single-track dataset — it's likely to
  overfit or just be unnecessary compute for the problem size. A small
  from-scratch CNN is the reasonable default; only escalate if it
  clearly underfits.
- If temporal context matters (§1's last row), prefer a simple
  frame-stack (channel-concat last N frames) over a full 3D-conv or
  recurrent model as the first thing to try — much less data-hungry.

## 4. Real-time inference loop — hardware constraints that shape the design

This is where CNN-control code differs most from a normal ML deployment,
because the target is a live BLE-connected robot:

- **BLE round-trip latency is real** — motor commands and sensor reads
  go over Bluetooth, not a local bus. A tight polling loop faster than
  the device's own notification rate wastes time; match your loop
  period to what `device_notification_delay` on `connect()` actually
  delivers rather than assuming sub-millisecond responsiveness.
- **Use `blocking=False` motor calls in a control loop, not blocking
  ones.** `movement_move_tank(..., blocking=True)` (the default) waits
  for a completion response before your next inference step can run —
  fine for a scripted maneuver, wrong for a loop meant to react every
  frame. For continuous CNN-driven control, prefer commands that set a
  standing state (`movement_move_tank` naturally returns once the tank
  command is acknowledged, not once motion "finishes" — that's fine for
  a control loop already; the trap is `_for_degrees`/`_for_time` calls,
  which block until that motion completes and starve the next
  inference step).
- **`set_notification_callback`** (raw `legoeducation`, both classes)
  gives you an event-driven hook instead of polling `.motor`/`.sensor`
  attributes — consider it once the loop is real and you're tuning for
  latency, not for the first working version.
- **Always keep a hard stop path independent of model output.** A
  runaway or NaN prediction driving `movement_move_tank` at ±100
  indefinitely is a real failure mode with a physical robot. Concretely:
  wrap the inference→command call in a try/except that calls
  `dm.stop()` on any exception, clamp predicted speeds to a sane range
  before sending them, and give the human a keyboard-interrupt path
  (`try/finally: dm.stop(); dm.disconnect()`, same pattern as every
  other script in this repo) that's always reachable.
- **Inference latency budgets the loop rate.** If a forward pass takes
  40ms, don't design a 100Hz control loop around it — measure actual
  inference time on the deployment machine first, then pick a loop
  period comfortably above it.

## 5. Dependencies — verify, don't assume

`my_env` currently has only `legoeducation` + `bleak` installed — no
`torch`/`tensorflow`/`opencv`/`numpy` yet (see CLAUDE.md). Before
writing training code:

- Actually run the install and read the output; don't assume success.
- Python 3.14 (this venv's interpreter) is new enough that ML
  framework wheels may not exist yet for it — check, and surface the
  problem to the user rather than quietly switching interpreters or
  substituting a different library.
- Training likely happens on a laptop, inference likely happens
  live against BLE hardware — these can be different environments
  (e.g. train in a notebook/Colab, export a small model, run inference
  in `my_env` where `legoeducation` lives). Don't assume both must be
  the same venv; ask if it's unclear which the user wants for a given
  script.

## 6. What to build first

Recommended order, each a working checkpoint before the next:

1. Data collection script (§2) producing a labeled frame dataset from
   a human driving the robot with `controller`.
2. Offline training script/notebook on the collected data — verify the
   model overfits a small subset before trusting it on the full set.
3. Offline evaluation (predicted vs. logged label, not yet on hardware).
4. Closed-loop inference script (§4) with the hard-stop safeguards,
   tested at low speed first.

Don't skip straight to step 4 — a model that's never been evaluated
offline, driving a real robot, is a debugging environment with much
worse feedback than a training script.
