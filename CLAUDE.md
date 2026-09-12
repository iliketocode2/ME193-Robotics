# CLAUDE.md — ME193 Robotics (LEGO Education BLE + CNN Motor Control)

This file is read automatically by Claude Code when working in this
repository. It covers (1) the repo layout and dev environment, (2) the
**real** API stack used in this codebase — which is a wrapper library,
not the raw `legoeducation` package — and (3) the project's direction:
building a CNN for advanced motor control on top of this hardware.

For the full API reference, see the `lego-hardware` skill
(`.claude/skills/lego-hardware/SKILL.md`) — it's loaded automatically
for hardware/motor questions. For CNN/ML pipeline work, see the
`cnn-motor-control` skill. Project subagents `lego-hardware-expert` and
`cnn-motor-control-architect` are available for implementation tasks in
each area; `hardware-code-reviewer` is a fresh-context reviewer to run
after implementing a change, before treating it as done (see §7).

---

## ⚠️ Critical: this is NOT SPIKE Prime, and it's NOT raw `legoeducation` either

This project uses LEGO Education's **BLE Python API**
(`pip install legoeducation`, currently v1.1.1 in `my_env/`) — a
completely different library from LEGO SPIKE Prime's `hub` module
(`from hub import port`, `motor_pair`, etc.), which is what most LLMs
default to when asked for "LEGO Python code."

**But this repo does not call `legoeducation` directly.** Every real
script in `Public stuff/useful libraries/` goes through **`lelib.py`**,
a thin wrapper that all class code is written against:

```python
from lelib import singleMotor, doubleMotor, controller, colorSensor
```

Rules for Claude Code:

1. If you catch yourself writing `import hub`, `from spike import ...`,
   or `motor_pair` — stop. That's SPIKE Prime.
2. If you catch yourself writing `le.SingleMotor()`, `le.DoubleMotor()`,
   `le.Controller()`, or `le.ColorSensor()` directly in a student-facing
   script — stop and use `lelib.py`'s `singleMotor`, `doubleMotor`,
   `controller`, `colorSensor` instead (note the lowercase-first
   class names — this is the opposite convention from raw
   `legoeducation`, and it's deliberate/load-bearing here, not a typo).
3. Still `import legoeducation as le` alongside `lelib` — you need it
   for constants (`le.LEGO_COLOR_RED`, `le.MOTOR_MOVE_DIRECTION_*`,
   `le.MOTOR_LEFT`, etc.) and for lower-level calls `lelib.py` doesn't
   wrap (see the skill for the full list: absolute-position moves,
   light/sound, batch mode, notification callbacks, IMU face select).

Why this matters: `lelib.py` classes **subclass** the raw `legoeducation`
classes (`class singleMotor(_CardReader, le.SingleMotor)`), so every
raw method is still callable on a `lelib` instance — `lelib` only adds
convenience methods and retry-on-connect logic on top. Writing against
raw `le.SingleMotor()` instead of `lelib.singleMotor()` isn't wrong in
the sense of crashing, but it throws away the retry logic, card-tap
tracking, and the simpler method names every other script in this repo
uses — so it reads as inconsistent/incorrect in review here.

---

## 1. Repo layout

```
ME193-Robotics/
├── CLAUDE.md                        (this file)
├── README.md
├── .gitignore                       ⚠️ only syncs "Public stuff/" — see below
├── my_env/                          local venv, gitignored, already exists
└── Public stuff/                    the only folder tracked in git
    ├── README.md
    └── useful libraries/
        ├── README.md                setup + "how to prompt an AI" for students
        ├── lelib.py                 ★ the wrapper everything imports
        ├── LELIB.md                 lelib.py reference (student-facing)
        ├── camlib.py                webcam picker (for CV/CNN input)
        ├── CAMLIB.md
        ├── miclib.py                microphone picker (for audio input)
        ├── MICLIB.md
        ├── main.py                  template: color-sensor + controller dispatch
        ├── single_motor_controller.py   template: joystick → single motor
        ├── test_devices.py          hardware smoke test (connects one of each device)
        └── test_devices.md
```

**Important:** `.gitignore` at the repo root is `*` plus explicit
un-ignores for `.gitignore`, `README.md`, and `Public stuff/**`. That
means **only `Public stuff/` is version-controlled** — anything you
create outside it (scratch scripts, notebooks, model checkpoints,
datasets) never reaches GitHub unless you deliberately add an
un-ignore rule for it. Keep that in mind when the CNN work starts
producing training data and model files — decide deliberately whether
those belong in `Public stuff/` (shared with the class) or should stay
local/ignored (large binaries, personal datasets).

---

## 2. Environment Setup

A virtual environment named `my_env/` **already exists** in this repo
(Windows, Python 3.14.0) with `legoeducation` and its dependency
`bleak` installed. Check for `my_env/` before creating a new one.

### Windows (PowerShell / cmd) — this machine

```powershell
python -m venv my_env
my_env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install legoeducation
```

### macOS / Linux (bash)

```bash
python3 -m venv my_env
source my_env/bin/activate
pip install --upgrade pip
pip install legoeducation
```

### Verify

```bash
my_env/Scripts/python -c "import legoeducation as le; print(le.__version__, le.__file__)"
```

If this fails, fix the environment before writing code that assumes it
works — check `where python` / `which python` resolves inside
`my_env`, not system Python.

### ⚠️ Adding CNN/ML dependencies (PyTorch, TensorFlow, OpenCV, etc.)

`my_env` currently has **only** `legoeducation` + `bleak` installed —
`camlib.py`'s `cv2`/`numpy` imports and `miclib.py`'s `sounddevice`
import are **not yet satisfied** in this environment, and neither is
any ML framework. Before writing CNN training code:

1. `pip install`, don't assume — actually run the install and read the
   output.
2. **Python 3.14 is very new.** PyTorch, TensorFlow, and OpenCV
   historically lag several months behind a new CPython release before
   publishing wheels for it. Do not assume `pip install torch` will
   succeed on this interpreter — try it, and if there's no wheel for
   `cp314` yet, tell the user rather than silently downgrading Python
   or picking an unrelated package. Options if it fails: a second venv
   pinned to Python 3.11/3.12 for training, or CPU-only builds, or
   waiting for upstream support — this is a decision for the user, not
   one to make silently.
3. Keep training/inference dependencies in a `requirements.txt` (or
   similar) inside `Public stuff/` if the class should share it, since
   nothing outside `Public stuff/` is tracked.

---

## 3. Ground truth for the `legoeducation` API

Don't guess method names or trust training-data memory of LEGO
libraries — inspect what's actually installed, since this package is
young and versioned independently of what any doc site says:

```bash
my_env/Scripts/python -m pip show legoeducation
my_env/Scripts/python -c "import legoeducation as le; help(le)"
my_env/Scripts/python -c "import legoeducation as le; help(le.SingleMotor)"
```

Or read the source directly (fast, exact, no BLE hardware needed):
`my_env/Lib/site-packages/legoeducation/device.py` (SingleMotor,
DoubleMotor, Controller, ColorSensor — all public methods have full
docstrings with examples), `basic_device.py` (connect/disconnect/
search/light_color/beep/notifications, shared by all four classes),
`rpc_message.py` (notification field layouts, all constants),
`color_map.py` (LEGO_COLOR_* numeric values and firmware translation).

The upstream doc repo (`https://github.com/LEGO/legoeducation` —
lowercase, per `pip show`'s Home-page field; the casing in older notes
may be stale) may also have per-device `.md` pages, but the installed
source is the authoritative reference for whatever version is actually
in `my_env` — prefer it when the two disagree.

The full distilled API reference (both `lelib.py` and raw
`legoeducation`) lives in the **`lego-hardware` skill** — load it for
any hardware-writing task rather than re-deriving this from scratch
each time.

---

## 4. Coding conventions for this repo

- `from lelib import singleMotor, doubleMotor, controller, colorSensor`
  for device control; `import legoeducation as le` for constants and
  any raw method `lelib` doesn't wrap.
- Every script follows connect → check `.connected` → do work (in
  `try`/`finally`) → `.stop()` motors → `.disconnect()`, so BLE
  devices are released even on error (see `test_devices.py` and
  `single_motor_controller.py` for the pattern).
- `card_color` / `card_serial` are never hardcoded deep in logic — keep
  them as named constants near the top of the script (see `main.py`),
  since every student's hardware has different values printed on their
  physical Connection Card. Passing `card_serial=None` connects to the
  first advertising device of that type with no card needed — fine for
  quick tests (`test_devices.py`), but ambiguous with multiple devices
  of the same type nearby.
- Prefer `lelib`'s `spin()` / `move_steps()` / `run_time()` /
  `turn_left()` / `turn_right()`, or raw `motor_run_for_degrees` /
  `motor_run_for_time`, over manual `run()` + `time.sleep()` when an
  exact amount of movement is the goal — less drift, less error-prone.
- Reference `le.MOTOR_MOVE_DIRECTION_*`, `le.LEGO_COLOR_*`,
  `le.MOTOR_LEFT/RIGHT/BOTH`, etc. by name, never as raw magic
  numbers — see the `lego-hardware` skill for the full constant list.
- `camlib.pick_camera()` / `miclib.pick_mic()` for any script that
  opens a webcam or microphone — don't hardcode device indices, they
  aren't stable across machines or reboots.

---

## 5. Where this is headed: CNN-based motor control

The end goal is training a CNN for advanced motor control of the
LEGO Education motors — likely vision-based (camera → CNN → motor
command) given `camlib.py` is already in the shared library set, but
could also mean a CNN over the color sensor's `raw_reading()` feature
vector, or a hybrid. See the **`cnn-motor-control`** skill for the
detailed roadmap: data collection design (what to log alongside
camera frames — `MotorNotification.position/speed/power`, controller
joystick state, IMU yaw — as training targets or auxiliary signals),
candidate architectures, the real-time inference loop's constraints
(BLE round-trip latency, blocking vs. `blocking=False` motor commands,
`set_notification_callback` for a non-polling control loop), and
safety (always have a hard motor-stop path independent of model
output).

Don't start implementing a training pipeline without loading that
skill first — it captures decisions (frame/label pairing strategy,
loop timing budget) that shouldn't be re-derived ad hoc per script.

---

## 6. Sanity check before finishing any hardware task

- [ ] Uses `lelib` classes (`singleMotor`/`doubleMotor`/`controller`/
      `colorSensor`) for device control, not raw `le.SingleMotor()`
      etc., unless there's a specific reason to bypass the wrapper
      (say so if you do).
- [ ] `import legoeducation as le` is present if any constant is used.
- [ ] Connects, checks `.connected`, and disconnects in `finally`
      (motors stopped before disconnect).
- [ ] `card_color` / `card_serial` are clearly-marked placeholders (or
      `None` with a comment explaining that's intentional) — never a
      guessed real value.
- [ ] Every method call matches something verified in
      `my_env/Lib/site-packages/legoeducation/device.py` /
      `basic_device.py` or `lelib.py` — not assumed from memory.
- [ ] If the task touches ML/CNN work: dependency availability was
      actually checked (not assumed) against Python 3.14, and any
      data/model artifacts outside `Public stuff/` are gitignored on
      purpose, not by accident.

---

## 7. Verification, review, and guardrails already in place

This repo has no automated test suite that can run unattended —
hardware requires physical BLE devices in range. Three things fill
that gap instead of relying on "looks done":

- **`.claude/settings.json` hooks**: any `.py` file Claude writes or
  edits is auto-syntax-checked with `my_env`'s Python (`py_compile`) —
  a real error is fed back immediately, not discovered later. A
  separate hook refuses edits under `my_env/` (vendored dependency,
  read-only), and another blocks a `git add`/`git commit` that would
  stage a file over 5MB (no Git LFS is set up here, and the CNN work
  will start producing camera-frame datasets/model checkpoints).
- **`hardware-code-reviewer` subagent**: run this after implementing or
  changing a hardware or CNN-pipeline script, before calling it done or
  committing — it reviews the diff in a fresh context against this
  file's and the skills' verified constraints (API correctness, motor
  targeting, connect/disconnect lifecycle, blocking-vs-non-blocking
  loops, CNN safety non-negotiables) rather than the reasoning that
  produced the change.
- **Plan mode** for anything that touches multiple files or an
  unfamiliar part of the codebase (e.g. wiring up the first CNN data
  collection pipeline) — explore and plan before editing. For a small,
  clearly-scoped hardware tweak, skip straight to implementing.

None of this replaces actually running a script against the physical
robot at least once before calling a change done — say explicitly when
that hasn't happened yet.
