---
name: lego-hardware
description: Ground-truth API reference for this repo's LEGO Education BLE stack — the lelib.py wrapper (singleMotor/doubleMotor/controller/colorSensor), the raw legoeducation package it wraps, and the camlib.py/miclib.py device pickers. Use whenever writing, reviewing, or debugging code that connects to or controls LEGO Education hardware (motors, controller, color sensor), or that opens a camera/microphone for a robotics script in this repo.
---

# LEGO Education hardware — ground-truth reference

This is verified against the **actually installed** `legoeducation`
v1.1.1 in `my_env/Lib/site-packages/legoeducation/` and the project's
own `Public stuff/useful libraries/lelib.py`, not against training-data
memory or upstream docs (which may be for a different version). If a
method call isn't in this file, grep the source before assuming it
exists:

```bash
grep -n "def " "my_env/Lib/site-packages/legoeducation/device.py"
grep -n "def " "my_env/Lib/site-packages/legoeducation/basic_device.py"
```

## The two layers

```
your script
   │
   ├─ from lelib import singleMotor, doubleMotor, controller, colorSensor
   │      ↑ project wrapper — USE THIS for device control
   │
   └─ import legoeducation as le
          ↑ raw package — use for constants + methods lelib doesn't wrap
              (lelib classes subclass le classes, so raw methods still work
               on a lelib instance: dm.movement_move_tank(...) is a raw
               DoubleMotor method called on a lelib doubleMotor instance)
```

`lelib.py`'s classes are deliberately **lowercase-first** —
`singleMotor`, `doubleMotor`, `controller`, `colorSensor` — the
opposite of raw `legoeducation`'s `SingleMotor`, `DoubleMotor`,
`Controller`, `ColorSensor`. Both exist in every script; don't "fix"
the casing.

---

## Connecting (all four device types)

```python
from lelib import singleMotor, doubleMotor, controller, colorSensor
import legoeducation as le

sm = singleMotor()
sm.connect(card_serial=3664, card_color=le.LEGO_COLOR_RED)   # specific card
dm = doubleMotor()
dm.connect(card_serial=None)                                  # first device found, no card
```

- `lelib`'s `connect()` retries up to 5 times (1s apart) on a
  "not ready" error before raising `ConnectionError` — this is why
  scripts here use `lelib`'s `connect()` rather than calling
  `le.SingleMotor().connect()` directly (no retry there).
- `card_serial=None` (or omitted for `doubleMotor`) connects to the
  first advertising device of that type — convenient, ambiguous with
  multiple identical devices nearby.
- Always follow with `try: ... finally: motor.stop(); motor.disconnect()`.
- `.connected` is a bool attribute (not a method) on every device.

---

## `lelib.singleMotor`

| Method | Behavior |
|---|---|
| `spin(rotations=1)` | `motor_run_for_degrees(rotations * 360)` — full/partial turns, blocking. |
| `run(speed=50)` | Continuous spin. Sign sets direction (+ = clockwise). Calls `motor_set_speed` + `motor_run`. |
| `set_speed(speed)` | Sets default speed for future `motor_run()`-style calls. |
| `stop()` | `motor_stop()`. |
| `card_serial()` / `card_tapped()` | See "Card reading" below. |

Raw `le.SingleMotor` methods also available directly on a `singleMotor`
instance (all take `blocking: bool = True` and, where relevant,
`motor: int = 0` — ignored for single motor, required for double):

- `motor_set_speed(speed, *, motor=0, blocking=True)` — speed −100..100.
- `motor_run(*, direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, motor=0, speed=UNCHANGED, blocking=True)`
- `motor_run_for_time(time_ms, *, direction=..., motor=0, speed=UNCHANGED, blocking=True)`
- `motor_run_for_degrees(degrees, *, direction=..., motor=0, speed=UNCHANGED, blocking=True)`
- `motor_run_to_relative_position(position, *, motor=0, speed=UNCHANGED, blocking=True)` — relative to last reset.
- `motor_reset_relative_position(*, motor=0, position=0, blocking=True)` — zero (or set) the relative encoder.
- `motor_run_to_absolute_position(position, *, direction=le.MOTOR_MOVE_DIRECTION_SHORTEST, motor=0, speed=UNCHANGED, blocking=True)`
- `motor_set_duty_cycle(duty_cycle, *, motor=0, blocking=True)` — raw drive, bypasses speed control.
- `motor_stop(*, motor=0, blocking=True)`
- `motor_set_end_state(end_state, *, motor=0, blocking=True)` — `le.MOTOR_END_STATE_{COAST,BRAKE,HOLD,DEFAULT,CONTINUE,SMART_COAST,SMART_BRAKE}`.
- `motor_set_acceleration(acceleration, deceleration, *, motor=0, blocking=True)` — both 0..100.
- `done(motor=None) -> bool` — true if no motor commands still in flight; poll this after a `blocking=False` call.
- `light_color(color, *, pattern=0, intensity=100, blocking=True)` — button LED; `le.LEGO_COLOR_*`, `le.LIGHT_PATTERN_*`.
- `beep(*, pattern=le.SOUND_PATTERN_BEEP_SINGLE, frequency=440, count=1, blocking=True)`, `stop_beep(*, blocking=True)`.
- `.motor` — a `MotorNotification` with live fields: `.position` (relative, degrees, signed 32-bit), `.absolutePosition` (0-359), `.speed` (−100..100), `.power`, `.motorState` (`le.MOTOR_STATE_*`: READY/RUNNING/STALLED/HOLDING/...), `.gesture` (`le.MOTOR_GESTURE_*`).
- `set_notification_callback(callback)` — register a function called on every hardware notification, for a non-polling control loop.

`speed=UNCHANGED` (i.e. omit the kwarg) means "keep whatever speed was
last set" — pass an explicit `speed=` if you need a specific one for
that single call.

---

## `lelib.doubleMotor`

Two drive motors (`le.MOTOR_LEFT` / `le.MOTOR_RIGHT` / `le.MOTOR_BOTH`
= 0/1/2) + an onboard IMU. Subclasses `singleMotor`'s per-motor API
(all methods above work with `motor=le.MOTOR_LEFT` etc.) plus:

| `lelib` method | Behavior |
|---|---|
| `move_steps(step=1)` | `movement_move_for_degrees(-180 * step)` — one "step" = 180° of wheel rotation. |
| `run(speed=50)` | Continuous forward/backward drive (sign = direction) via `movement_move`. |
| `run_time(time=2000)` | `movement_move_for_time(time)` — drive for `time` ms then stop. |
| `run_left(degrees=None)` / `run_right(degrees=None)` | One side only; continuous if `degrees=None`, else `motor_run_for_degrees`. |
| `turn_left(degrees=90)` / `turn_right(degrees=90)` | Spin in place via `movement_turn_for_degrees` (IMU-confirmed). |
| `set_speed(speed)` | Sets speed for left motor, right motor, *and* movement commands together. |
| `set_speed_left(speed)` / `set_speed_right(speed)` | Per-side only. |
| `stop()` | `motor_stop()` (both motors, current end-state applies). |
| `reset_heading()` | `imu_reset_yaw_axis(0)` — zero yaw at current orientation; call once before straight-line/heading control. |
| `yaw()` | `float(imu_device.yaw)` — degrees since last `reset_heading()`. + = clockwise drift. |
| `gyro_z()` | `float(imu_device.gyroscopeZ)` — raw angular velocity, deg/s-ish, + = clockwise. |

Raw `le.DoubleMotor` methods also available (movement-level, act on
both wheels together — **cannot** be called inside batch mode):

- `movement_move(*, direction=le.MOVEMENT_DIRECTION_FORWARD, speed=UNCHANGED, blocking=True)`
- `movement_move_for_time(time_ms, *, direction=..., speed=UNCHANGED, blocking=True)`
- `movement_move_for_degrees(degrees, *, direction=le.MOVEMENT_MOVE_DIRECTION_FORWARD, speed=UNCHANGED, blocking=True)`
- `movement_move_tank(speed_left, speed_right, *, blocking=True)` — **the main method for joystick/AI-driven control**, independent per-side speed −100..100.
- `movement_move_tank_for_degrees(degrees, *, speed_left=50, speed_right=50, blocking=True)` — runs until the faster wheel hits `degrees`.
- `movement_turn_for_degrees(degrees, *, direction=le.MOVEMENT_TURN_DIRECTION_LEFT, speed=UNCHANGED, blocking=True)`
- `movement_stop(*, blocking=True)`, `movement_set_speed(speed, *, blocking=True)`
- `movement_set_end_state(end_state, *, blocking=True)`, `movement_set_acceleration(accel, decel, *, blocking=True)`
- `movement_set_turn_steering(steering, *, blocking=True)` — 0..100, balances left/right during turns.
- `imu_set_yaw_face(yaw_face, *, blocking=True)` — `le.DEVICE_FACE_{TOP,FRONT,RIGHT,BOTTOM,BACK,LEFT}`.
- `imu_reset_yaw_axis(value=0, *, blocking=True)`
- `.motor` is a **list of two** `MotorNotification` (index 0=left, 1=right) on `doubleMotor` (vs. a single object on `singleMotor`).
- `.imu_device` — `ImuDeviceNotification`: `.orientation`, `.yawFace`, `.yaw`, `.pitch`, `.roll`, `.accelerometerX/Y/Z`, `.gyroscopeX/Y/Z`.
- `.imu_gesture` — `ImuGestureNotification.gesture` (`le.MOTION_GESTURE_*`: TAPPED, DOUBLE_TAPPED, COLLISION, SHAKE, FREEFALL).

`MOTOR_LEFT`/`MOTOR_RIGHT`/`MOTOR_BOTH` matter: passing `motor=le.MOTOR_BOTH`
to a per-motor call (e.g. `motor_stop(motor=le.MOTOR_BOTH)`) is valid on
`doubleMotor` only (raises on `singleMotor`, which only has motor index 0).

---

## `lelib.controller`

```python
ctrl = controller()
ctrl.connect(card_serial=2279)
left, right = ctrl.left_position(), ctrl.right_position()   # -100..100 each
dm.movement_move_tank(left, right)
```

| Method | Returns |
|---|---|
| `left_position()` / `right_position()` | `int`, −100 (full back) .. +100 (full forward) — `sensor.leftPercent`/`rightPercent`. |
| `left_up()` / `left_down()` / `left_released()` | bool, sign of `leftPercent`. |
| `right_up()` / `right_down()` / `right_released()` | bool, sign of `rightPercent`. |
| `drive(dm, t=100)` | Blocking loop: `t` iterations × 100ms, tank-driving `dm` from both sticks. Good for a quick demo, not for anything needing a stop condition mid-loop — write your own loop instead when you need that. |

`.sensor` is a `ControllerNotification`: `.leftPercent`, `.rightPercent`
(both int8, −100..100), `.leftAngle`, `.rightAngle` (raw stick angle,
int16). `Controller` has no motor/movement methods — read-only device.

---

## `lelib.colorSensor`

| Method | Returns |
|---|---|
| `detect_color()` | `str` — one of `No color, Red, Yellow, Blue, Teal, Green, Purple, White, Magenta, Orange, Azure`. |
| `reflection()` | `int` 0–255, raw `sensor.reflection`. |
| `raw_rgb()` | `(r, g, b)` tuple, each 0–65535 (`sensor.rawRed/Green/Blue`). |
| `raw_reading()` | `dict` with **all channels normalized to 0–1** — `rawRed`, `rawGreen`, `rawBlue` (÷65535), `reflection` (÷255), `hue` (÷65535), `saturation` (÷255), `value` (÷255). **This is the feature vector to use for any classifier/ML input** — already normalized, no extra scaling needed. |

`Purple`/`Teal`/`White` are detectable by the sensor
(`SENSOR_DETECTABLE_COLORS`); `Magenta`/`Orange`/`Azure` are Connection
Card colors, not sensor-detectable colors, even though `detect_color()`'s
string table includes them for completeness — don't expect the sensor
to actually report those against a physical surface.

---

## Card reading (all four device types — `_CardReader` mixin)

Every device can read a Connection Card resting on its NFC pad:

```python
serial = dm.card_serial()     # current serial, 0 = no card present
tapped = dm.card_tapped()     # serial ONCE per new tap, else None — call in a poll loop
```

`card_tapped()` tracks the last-seen serial internally — call it once
per loop iteration, don't call it twice expecting the same answer (the
second call in the same "tap" returns `None`).

---

## Constants (`import legoeducation as le`)

Never use raw numbers/strings for these — always the named constant:

- **Colors**: `le.LEGO_COLOR_{NOCOLOR,RED,YELLOW,BLUE,TEAL,GREEN,PURPLE,WHITE,MAGENTA,ORANGE,AZURE}` (values 0–10, App-aligned — matches `colorSensor.detect_color()`'s numeric mapping exactly).
- **Motor index**: `le.MOTOR_LEFT` (0), `le.MOTOR_RIGHT` (1), `le.MOTOR_BOTH` (2, DoubleMotor only).
- **Motor move direction**: `le.MOTOR_MOVE_DIRECTION_{CLOCKWISE,COUNTERCLOCKWISE,SHORTEST,LONGEST}`.
- **Movement direction** (DoubleMotor driving): `le.MOVEMENT_DIRECTION_{FORWARD,BACKWARD,LEFT,RIGHT}`, `le.MOVEMENT_MOVE_DIRECTION_{FORWARD,BACKWARD}`, `le.MOVEMENT_TURN_DIRECTION_{LEFT,RIGHT}`.
- **Motor end state**: `le.MOTOR_END_STATE_{DEFAULT,COAST,BRAKE,HOLD,CONTINUE,SMART_COAST,SMART_BRAKE}`.
- **Device face** (yaw reference): `le.DEVICE_FACE_{TOP,FRONT,RIGHT,BOTTOM,BACK,LEFT}`.
- **Light pattern**: `le.LIGHT_PATTERN_{SOLID,BREATHE,PULSE,SHORT_BLINK,LONG_BLINK,DOUBLE_BLINK}`.
- **Sound pattern**: `le.SOUND_PATTERN_BEEP_{SINGLE,DOUBLE,TRIPLE,UP_MIDDLE_DOWN}`.
- **Motor state** (`.motor.motorState`): `le.MOTOR_STATE_{READY,RUNNING,STALLED,CMD_ABORTED,REGULATION_ERROR,MOTOR_DISCONNECTED,HOLDING,DC_RUNNING,NOT_ALLOWED_TO_RUN}`.
- **Motion gesture** (IMU, `.imu_gesture.gesture`): `le.MOTION_GESTURE_{NO_GESTURE,TAPPED,DOUBLE_TAPPED,COLLISION,SHAKE,FREEFALL}`.
- **Motor gesture** (`.motor.gesture`): `le.MOTOR_GESTURE_{NO_GESTURE,SLOW_CLOCKWISE,FAST_CLOCKWISE,SLOW_COUNTERCLOCKWISE,FAST_COUNTERCLOCKWISE,WIGGLED}`.

Full canonical list: `my_env/Lib/site-packages/legoeducation/__init__.py`'s
`__all__` — grep it if a constant you expect isn't listed above.

---

## Batch mode (raw `legoeducation`, not wrapped by `lelib`)

`_BasicDevice` supports batching multiple motor commands to fire
together — relevant for tight sync (e.g. simultaneous but different
per-motor commands). Not used anywhere in this repo's current scripts,
and `connect()`/`disconnect()`/movement-level commands **cannot** run
inside a batch. Grep `basic_device.py` for `_batch_mode` before
introducing this — it's an advanced/uncommon path here.

---

## `camlib.pick_camera()` — for vision/CNN input

```python
from camlib import pick_camera
cap, start_ms = pick_camera()          # default 1280x720
cap, start_ms = pick_camera(width=640, height=480)   # lower res = faster inference
```

Shows a numbered thumbnail-preview picker (press the number key), opens
the chosen `cv2.VideoCapture` warmed up and ready, returns it plus
`start_ms` (epoch ms at open time — use as the timestamp base for
MediaPipe `detect_for_video`, or any per-frame elapsed-time need).
macOS-oriented (`system_profiler` for names, `CAP_AVFOUNDATION`
backend) but degrades gracefully elsewhere (generic `Camera N` names,
default backend). Raises `RuntimeError` if no camera / can't open —
don't swallow that silently in a script meant to run standalone.

**Not currently installed in `my_env`** (`cv2`/`numpy` missing) — see
CLAUDE.md's environment section before assuming this import works.

## `miclib.pick_mic()` — for audio input

```python
from miclib import pick_mic
device_index = pick_mic()   # int or None (system default); prompts interactively
```

Never raises — falls back to `None` (system default) on no devices, no
input, or invalid input. Pass straight to `sd.InputStream(device=...)`.
Also not currently installed in `my_env` (`sounddevice` missing).

---

## Common mistakes to catch in review

- Using `le.SingleMotor()`/`le.DoubleMotor()`/etc. directly instead of
  `lelib`'s wrappers, without a stated reason.
- Forgetting `motor=le.MOTOR_LEFT/RIGHT/BOTH` on a per-motor call to a
  `doubleMotor` (defaults to `motor=0` i.e. left — often not the intent).
- Calling `movement_*` methods (DoubleMotor-only, whole-robot driving)
  on a `singleMotor` — they don't exist there; only `motor_*` per-motor
  calls do.
- Polling `card_tapped()` expecting it to re-fire for a card that's
  still resting on the sensor — it only fires once per new serial.
  Use `card_serial()` if you need the current value every call.
- Treating `raw_reading()`'s dict values as raw sensor units — they're
  pre-normalized to 0–1; don't divide again.
- Missing `.stop()` before `.disconnect()` on a motor device — leaves
  it spinning after the script exits/crashes if end-state isn't coast.
- Hardcoding a webcam/mic index instead of `pick_camera()`/`pick_mic()`.
