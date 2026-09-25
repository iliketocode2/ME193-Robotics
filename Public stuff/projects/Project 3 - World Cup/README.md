# Project 3 — World Cup

A LEGO Education "car" (Double Motor drive + a Single Motor "shield" arm
with a cardboard square taped on) driven entirely by whistling into a
laptop microphone — pitch continuously sets speed and turn (a gradient:
closer to your lowest whistle turns further left, closer to your highest
turns further right), and a 3-pulse whistle rhythm calls a "goal" —
coordinated with an opponent robot over MQTT for a ball-vs-goalie match on
topic `ME193/Rogers`.

A second person on a second computer can help drive the same physical car:
one computer ("drive") is the one actually Bluetooth-connected to the
robot; a second computer ("shield co-pilot") runs the exact same script
with its own microphone and continuously relays its own pitch gradient
over a dedicated MQTT control topic, which the drive computer applies to
the Single Motor shield's position in real time — the same gradient
control style as steering, just relayed over MQTT instead of BLE. See
"Two-computer setup" below.

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
- **`songs.py`** — `DEATH_SONG` / `SUCCESS_SONG` note lists, and `play_both()`,
  which plays them on **two** outputs at once: the LEGO hub's own speaker
  (`play_song()`, via `beep()`) and the computer's speaker (`play_computer_song()`,
  via stdlib `winsound.Beep()`) on a background thread, so the cue is audible
  even if you're not standing right next to the hub's small buzzer, and so a
  working computer speaker confirms the trigger fired independent of any
  hub/BLE audio issue.
- **`calibrate.py`** — guided calibration: measures room noise, then your
  lowest and highest comfortable whistle notes, and writes
  `whistle_config.json`.
- **`world_cup.py`** — the match script. Opens a small setup dialog first
  (game role, "drive" vs. "shield co-pilot", team name), then either: runs
  the full flow (connects the Double Motor, Single Motor, Color Sensor;
  subscribes to `ME193/Rogers` + the team's control topic; drives; handles
  the fail/goal MQTT protocol and songs) if you picked **drive**, or a much
  lighter no-hardware flow that just whistles a shield toggle to the drive
  computer if you picked **shield co-pilot**. Both modes show the live
  waveform/spectrum/decision dashboard.

## Run order

```powershell
my_env_audio\Scripts\python "Public stuff\projects\Project 3 - World Cup\test_whistle_policy.py"
my_env_audio\Scripts\python "Public stuff\projects\Project 3 - World Cup\calibrate.py"
my_env_audio\Scripts\python "Public stuff\projects\Project 3 - World Cup\world_cup.py"
```

Before running `world_cup.py` (on the **drive** computer, i.e. the one
actually Bluetooth-connected to the robot):

- Fill in `CAR_CARD_SERIAL`/`CAR_CARD_COLOR`,
  `SHIELD_CARD_SERIAL`/`SHIELD_CARD_COLOR`,
  `SENSOR_CARD_SERIAL`/`SENSOR_CARD_COLOR` from your three LEGO Connection
  Cards (Double Motor, Single Motor, Color Sensor). (Role and team name are
  no longer hardcoded here — see below.)
- Calibrate `PROXIMITY_REFLECTION_THRESHOLD` on-site against the actual
  opponent robot and match-day lighting — the default (70) is an
  unverified placeholder, not something to trust as-is.
- Watch the car actually drive once and flip `INVERT_LEFT_MOTOR` /
  `INVERT_RIGHT_MOTOR` if a wheel spins the wrong way (same convention as
  Project 1's `arm_race_control.py`).
- Verify the shield's absolute position 0 actually corresponds to
  "retracted" on the physical arm (`set_shield_position()`'s comment in
  `world_cup.py`) — `motor_run_to_absolute_position()`'s "0" is whatever the
  hub's own internal zero reference is, not necessarily where the arm
  happened to be pointed at connect time. Adjust `SHIELD_SWING_DEGREES`
  or manually re-zero the arm if "retracted"/"deployed" come out backwards
  or offset.
- Run `calibrate.py` in the same room, with the same mic, close to match
  time — ambient noise and your own whistle range are what it's tuned
  against. If using a shield co-pilot, they should run their own
  `calibrate.py` too, against their own mic/room.

Running `world_cup.py` (either computer) first opens a small setup dialog:
pick **game role** (Ball/Goalie), pick **this computer controls**
(Drive/Shield co-pilot), and type a **team name**. Everything else follows
from those three choices — see "Two-computer setup" below.

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
  forward, speed scaling linearly from 20% to 55% across the band (kept
  deliberately gentle — see "motor speed" below).
- **TURN band** (everything in between) → turn only, no forward motion, as
  a **gradient of pitch position within the band**: the low edge of the
  band is full left, the high edge is full right, the exact center is
  straight, and everywhere in between scales linearly. So a steady whistle
  a little above the stop band turns gently left, and a steady whistle a
  little below the forward band turns sharply right — the turn amount
  tracks *where* you're whistling, not how you got there (earlier versions
  of this policy used the pitch's *slope*/rate of change instead, which
  meant you had to actively slide your pitch to keep turning; a straight
  gradient is easier to hold a specific turn amount with).

On top of that continuous mapping, a **rhythmic** gesture is recognized
from sequences of short (<0.4s) tonal pulses in the forward band, so the
same whistling channel carries both continuous proportional control and a
discrete command:

- **Three short pulses in the FORWARD band within 2 seconds** → the "made
  it in the goal" command (publishes to MQTT, plays the success song).

The shield (Single Motor) uses the identical gradient idea, just from a
**second, independent whistle** — see "Two-computer setup": a co-pilot's
own turn-band gradient continuously sets the shield's position (their
lowest-band pitch retracts it, their highest-band pitch fully deploys it)
instead of driving the car, relayed over MQTT rather than computed locally.

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
   doesn't jerk the turn/shield gradient or the reported HUD frequency
   around.

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

## Two-computer setup

Only one computer can hold the actual Bluetooth connections to the robot's
three devices, but a second person can still help control it with their own
whistle from a second computer:

- **Drive** (pick this on the computer physically paired with the robot):
  runs the full flow from the rest of this README. Also subscribes to a
  second, team-scoped MQTT topic, `<MQTT_TOPIC>/control/<team name>` (e.g.
  `ME193/Rogers/control/Cucurella`) — separate from the public game-event
  topic so it doesn't collide with other teams' traffic, same
  crosstalk-avoidance reasoning `MQTTLIB.md` gives for topic names in
  general. A `{"shield_gradient": <-100..100>}` message on that topic sets
  the shield's absolute position (linearly, -100 = fully retracted, +100 =
  fully deployed) every time one arrives — continuous, not a one-shot toggle.
- **Shield co-pilot** (pick this on the second computer): no BLE connection
  at all — it just opens its own microphone, runs the same whistle policy,
  and continuously publishes `{"shield_gradient": <cmd.turn_bias>}` to that
  same control topic (about 20x/second, matching the control loop rate) —
  literally the same turn-band gradient value the drive computer would use
  for steering, just relayed instead of applied locally. Its dashboard
  still shows its own waveform/spectrum/decision and a live MQTT feed
  (though the continuous shield-gradient traffic itself isn't logged there
  — at 20 messages/second it would drown out everything else useful in a
  10-entry rolling log; the current commanded shield position is shown
  directly in the decision panel instead), and mirrors the match's
  LIVE/WAITING/GAME OVER status (read-only — it never itself starts/ends
  the match or plays a song, since it has no local hardware to act with).
  With no co-pilot connected, the drive car's shield defaults to fully
  retracted (matching the "sensor must be open" rule) rather than sitting
  at some arbitrary position.

**Both computers must type the exact same team name** in the setup dialog —
that's what makes the control topic match up between them. A shield
co-pilot's own game-role choice only affects its own dashboard's display;
the actual match outcome (fail/goal, songs, publishing to `MQTT_TOPIC`)
is entirely owned by whichever instance is running as **drive**.

This is a generic capability, not "goalie only" or "ball only" — either
role's robot can have a remote shield co-pilot, since a shield is generic
hardware, not part of the ball/goalie game logic itself.

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
  **`mqttlib`'s network thread** (`on_mqtt_message` and, in drive mode,
  `on_control_message`) only reads/writes that same dict; a **dedicated
  `run_control_loop()` thread**, polling every `CONTROL_LOOP_PERIOD_S`
  (50ms — comfortably above any single BLE round-trip), is the *only* place
  that ever calls `movement_move_tank`, `motor_run_to_absolute_position`
  (the shield, via `set_shield_position`), `beep()` (via `play_both`), or
  `mqtt.publish()`. `play_both()` itself spawns one
  extra short-lived background thread so the computer-speaker song and the
  hub-speaker song play at the same time instead of back-to-back. The
  shield co-pilot mode mirrors this same split with much less work per
  thread (`copilot_audio_callback`/`run_copilot_loop`) since there's no BLE
  hardware involved on that instance at all.
- The main thread only runs a `matplotlib` `FuncAnimation` (a dashboard: game
  status, waveform, spectrum with the calibrated bands shaded, the drive
  command actually sent to the car, the front sensor reading vs. its
  proximity threshold, the whistle decision, and a live MQTT sent/received
  feed) that only *reads* the same lock-guarded state — it never touches
  BLE, MQTT, or audio directly.
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

The DSP/policy logic (`test_whistle_policy.py`, including the gradient-turn
tests: low/high/center-of-band and a partial-gradient check) passes under
`my_env_audio`, and `world_cup.py`/`calibrate.py` import and syntax-check
cleanly. The dashboard (both "drive" and "shield" modes) has been rendered
headlessly and exercised for one frame each without error, the new message
handlers (`on_control_message`, `on_mqtt_message_readonly`) have been
unit-checked directly against `shared` state, and `set_shield_position()`'s
gradient-to-degrees math has been checked directly (-100→0°, 0→45°,
100→90°). **None of this has been run against the physical car, shield
motor, and color sensor, and the setup dialog and the two-computer
control-topic relay have not been click-tested by a human or run across two
real machines** — that all still needs to happen (with real Connection Card
values filled in and a calibrated proximity threshold) before treating this
as match-ready, per this repo's usual rule that nothing is "done" until
it's actually run against the hardware at least once. In particular, the
shield's absolute-position-0-means-retracted assumption and the new lower
motor speeds (20-55% instead of 30-90%) still need a real drive to confirm
they feel right.
