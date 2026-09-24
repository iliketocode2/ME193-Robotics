# Project 3 — World Cup

A LEGO Education "car" (Double Motor drive + a Single Motor "shield" arm
with a cardboard square taped on) driven entirely by whistling into a
laptop microphone — pitch controls speed/turn, short whistle rhythms
trigger a shield toggle or a "goal" call — coordinated with an opponent
robot over MQTT for a ball-vs-goalie match on topic `ME193/Rogers`.

## Two virtual environments — read this first

This project **must use PyAudio** (assignment requirement), and PyAudio
has no prebuilt wheel for Python 3.14 (only through `cp313`) — building it
from source also fails here because the PortAudio C headers aren't
installed. Rather than fight that, this project runs in its own venv:

| Venv | Python | Used for |
|---|---|---|
| `my_env/` | 3.14 | Everything else in this repo (unchanged). |
| `my_env_audio/` | 3.13 | **Only this project.** `pyaudio`, `numpy`, `matplotlib`, `paho-mqtt`, and `legoeducation` all installed here too — this venv is fully self-contained for `calibrate.py`/`world_cup.py`. |

Setup (already done once on this machine; included here so it's
reproducible elsewhere):

```powershell
winget install Python.Python.3.13
py -3.13 -m venv my_env_audio
my_env_audio\Scripts\python -m pip install --upgrade pip
my_env_audio\Scripts\python -m pip install legoeducation pyaudio numpy matplotlib paho-mqtt
```

Every command below runs through `my_env_audio`, **not** `my_env`.

## Files

- **`whistle_policy.py`** — pure signal-processing + decision logic (FFT
  pitch detection, noise gating, gesture recognition). No hardware, no I/O
  — this is what `test_whistle_policy.py` exercises directly.
- **`test_whistle_policy.py`** — synthetic-signal self-check (sine waves,
  chirps, white noise, pulse trains) for the policy above. No microphone or
  BLE hardware needed; run this any time the policy logic changes.
- **`pyaudio_mic.py`** — microphone device picker built on PyAudio's own
  device-enumeration API (mirrors the shared `miclib.pick_mic()`'s UX, but
  `miclib.py` itself is `sounddevice`-based, which doesn't satisfy this
  assignment's "use PyAudio" requirement).
- **`songs.py`** — `DEATH_SONG` / `SUCCESS_SONG` note lists and
  `play_song()`, which plays them on the LEGO hub's own speaker via
  `beep()`.
- **`calibrate.py`** — guided calibration: measures room noise, then your
  lowest and highest comfortable whistle notes, and writes
  `whistle_config.json`.
- **`world_cup.py`** — the match script: connects the Double Motor, Single
  Motor (shield), and Color Sensor (front light sensor); subscribes to
  `ME193/Rogers`; runs the live whistle-control loop; shows the live
  waveform/spectrum/decision on screen; handles the fail/goal MQTT protocol
  and songs.

## Run order

```powershell
my_env_audio\Scripts\python "Public stuff\projects\Project 3 - World Cup\test_whistle_policy.py"
my_env_audio\Scripts\python "Public stuff\projects\Project 3 - World Cup\calibrate.py"
my_env_audio\Scripts\python "Public stuff\projects\Project 3 - World Cup\world_cup.py"
```

Before running `world_cup.py`:

- Fill in `ROLE` (`"ball"` or `"goalie"`, assigned match day) and
  `TEAM_NAME` at the top of the file.
- Fill in `CAR_CARD_SERIAL`/`CAR_CARD_COLOR`,
  `SHIELD_CARD_SERIAL`/`SHIELD_CARD_COLOR`,
  `SENSOR_CARD_SERIAL`/`SENSOR_CARD_COLOR` from your three LEGO Connection
  Cards (Double Motor, Single Motor, Color Sensor).
- Calibrate `PROXIMITY_REFLECTION_THRESHOLD` on-site against the actual
  opponent robot and match-day lighting — the default (200) is an
  unverified placeholder, not something to trust as-is.
- Watch the car actually drive once and flip `INVERT_LEFT_MOTOR` /
  `INVERT_RIGHT_MOTOR` if a wheel spins the wrong way (same convention as
  Project 1's `arm_race_control.py`).
- Run `calibrate.py` in the same room, with the same mic, close to match
  time — ambient noise and your own whistle range are what it's tuned
  against.

## The required write-up

### Describe the policy — how does it make decisions?

Every ~46ms audio block (2048 samples @ 44.1kHz) goes through
`whistle_policy.PitchDetector`: a Hann-windowed FFT finds the spectral peak
bin (skipping DC), giving a raw pitch estimate, which is then smoothed
across blocks with an exponential moving average to remove frame-to-frame
jitter. The peak's magnitude divided by the spectrum's mean magnitude — the
**tonal ratio** — measures how "whistle-like" (narrowband) vs. "noise-like"
(broadband) the block is; see the noise-masking answer below for why.

If a block passes both the tonal-ratio gate and an RMS noise floor, its
smoothed frequency is classified into one of three **calibrated** bands
(from `calibrate.py`, derived from *your* actual whistle range, not a fixed
guess):

- **STOP band** (your lowest comfortable whistle ± 150 Hz) → speed = 0.
- **FORWARD band** (your highest comfortable whistle ± 150 Hz) → drive
  forward, speed scaling linearly from 30% to 90% across the band.
- **TURN band** (everything in between) → turn only, no forward motion.
  Since a single frequency is one-dimensional, it can't by itself encode a
  left/right choice — so turn *direction* comes from the pitch's recent
  **slope** (Hz/second over the last ~300ms): a rising glissando turns
  right, a falling one turns left, and a nearly flat pitch in this band is
  treated as "no clear direction" and produces a small forward creep
  instead of turning randomly.

On top of that continuous mapping, two **rhythmic** gestures are
recognized from sequences of short (<0.4s) tonal pulses, so the same
whistling channel carries both continuous proportional control and
discrete commands:

- **Three short pulses in the FORWARD band within 2 seconds** → the "made
  it in the goal" command (publishes to MQTT, plays the success song).
- **Two short pulses in the STOP band within 1.2 seconds** → toggles the
  cardboard shield (Single Motor swings between retracted/deployed). Using
  a *different* band than the goal gesture (rather than just a different
  pulse count) means a 3-pulse goal attempt can never accidentally fire the
  2-pulse shield toggle partway through — see
  `test_goal_and_shield_gestures_dont_cross_contaminate` in
  `test_whistle_policy.py`.

### What does your code do if no whistle is detected?

It does **not** hold the last command forever, and it does **not** cut the
motors instantly either. If no tonal block has been seen for longer than
`silence_timeout_s` (0.4s — short lapses, like a breath between whistles,
are tolerated), the current forward/turn command is ramped **linearly down
to zero** over `ramp_time_s` (0.3s) rather than snapped to a stop. This
avoids two failure modes: instantly killing the car on every tiny gap in
the whistle (frustrating and jerky to drive), and — the more important
one — a car that silently coasts at its last commanded speed indefinitely
if the whistler stops for any reason (out of breath, dropped the whistle,
etc.), which would be an actual safety problem on a moving robot. This is
independent of, and in addition to, the hard `try/except -> car.stop()`
wrapped around every control step in `world_cup.py`'s audio callback, and
the `finally: car.stop(); car.disconnect()` that runs on any exit.

### How did you try to mask out unwanted noise?

Three layers, from cheapest to most targeted:

1. **Tonal-ratio gate.** A clean whistle is a narrow spectral spike; talking,
   footsteps, motor whir, and ambient room noise are broadband. Comparing
   the FFT peak's magnitude to the spectrum's *mean* magnitude
   (peak/mean) separates the two cleanly — empirically, white noise at
   this block size tops out around a ratio of ~4, while a clean tone hits
   ~450, so the default gate (6.0) has a wide margin on both sides.
2. **Calibrated ambient RMS floor.** `calibrate.py` records a few seconds
   of silence and sets the floor to 2× the loudest ambient block observed,
   so a block also has to actually be *loud enough*, not just tonal —
   guards against a faint high-pitched electronic whine or hum being
   mistaken for an intentional whistle.
3. **EMA smoothing across blocks.** Even once a block passes both gates,
   the reported frequency is smoothed with an exponential moving average
   rather than trusted block-to-block, so a single noisy FFT estimate
   doesn't jerk the turn-direction slope calculation or the reported HUD
   frequency around.

## MQTT protocol on `ME193/Rogers`

Uses `mqttlib.MQTTClient` (`test.mosquitto.org`, `qos=0`, no retain — see
`MQTTLIB.md`'s own reasoning for why a stale retained game message would be
actively wrong here).

- **`"start"`** — plain text (not JSON), sent by the instructor. Gates
  whether whistle commands actually drive the motors; before it arrives,
  the live plot still runs (so you can see your whistle being decoded) but
  the car doesn't move.
- **`{"event": "fail", "team": "ball"}`** — published by the ball itself
  the moment its front sensor detects the opponent within
  `PROXIMITY_REFLECTION_THRESHOLD`. The ball stops and plays `DEATH_SONG`
  locally; the goalie, on receiving this, plays `SUCCESS_SONG`.
- **`{"event": "goal", "team": "ball"}`** — published by the ball the
  moment its whistle policy fires the 3-pulse goal gesture. The ball plays
  `SUCCESS_SONG` locally; the goalie, on receiving this, plays
  `DEATH_SONG`.
- Messages whose `"team"` matches your own `ROLE` are ignored — the public
  broker echoes every publish back to the publisher too (see
  `MQTTLIB.md`/`mqtt_chat.py`), so without this filter a robot would
  re-trigger its own event from its own echo.

**This schema must be agreed with the opponent team before the match** —
the assignment explicitly requires this, and nothing here enforces it
automatically. If the opponent's script uses different field names or
values, neither side's fail/goal reactions will fire.

## Hardware / concurrency notes

- **`legoeducation`'s synchronous calls block their *calling* thread on a
  real BLE round-trip regardless of their own `blocking=` kwarg** — verified
  in `legoeducation/_platform.py`'s `_run_sync_cpython` (every call submits
  to a background asyncio loop and then does `wrapper_future.result()`,
  which blocks). `blocking=False` only skips waiting for a completion
  *response* on top of that, it doesn't make the call non-blocking. That
  means BLE calls cannot live on PyAudio's callback thread, whose ~46ms
  block period leaves no room for a BLE round-trip — doing so risks
  PortAudio reporting input overflow / dropping audio. This is a stricter
  version of the exact lesson Project 2 documented after motor commands
  stalled its camera loop.
- So there are three threads with a clear division of labor: **PyAudio's
  callback thread** (`audio_callback`) does only DSP (`policy.update()`,
  pure numpy/FFT) and writes results into a lock-guarded shared dict;
  **`mqttlib`'s network thread** (`on_mqtt_message`) only reads/writes that
  same dict; a **dedicated `run_control_loop()` thread**, polling every
  `CONTROL_LOOP_PERIOD_S` (50ms — comfortably above any single BLE
  round-trip), is the *only* place that ever calls `movement_move_tank`,
  `motor_run_for_degrees`, `beep()` (via `play_song`), or `mqtt.publish()`.
- The main thread only runs a `matplotlib` `FuncAnimation` that reads the
  same lock-guarded state to draw the waveform, spectrum (with the
  calibrated bands shaded), and a HUD text of the current decision — it
  never touches BLE, MQTT, or audio directly.
- **`lelib.py`'s `doubleMotor.stop()` only stops the LEFT motor** —
  verified: it calls the raw `motor_stop()` with no `motor=` kwarg, which
  defaults to motor index 0 (left) in `legoeducation/device.py`
  (`DEFAULT_MOTOR = 0`). This is a real bug in the *shared* library (not
  modified here — it would affect every project in this repo that calls
  `dm.stop()` expecting both wheels to stop), so `world_cup.py`
  deliberately never calls it: every stop here uses the raw
  `car.movement_stop()` instead, which is the correct whole-robot
  counterpart to `movement_move_tank()`.
- Every control step in `run_control_loop()` is wrapped in `try/except`
  that calls `car.movement_stop()` on any error, and the whole hardware
  section in `main()` is in a
  `try/finally: car.movement_stop(); shield.stop(); <disconnect all three devices>`.

## Status

The DSP/policy logic (`test_whistle_policy.py`) passes under
`my_env_audio`, and `world_cup.py`/`calibrate.py` import and syntax-check
cleanly. **This has not yet been run against the physical car, shield
motor, and color sensor** — that still needs to happen (with real
Connection Card values filled in and a calibrated proximity threshold)
before treating this as match-ready, per this repo's usual rule that
nothing is "done" until it's actually run against the hardware at least
once.
