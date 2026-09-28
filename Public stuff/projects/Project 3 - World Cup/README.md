# Project 3 — World Cup

A LEGO Education "car" (Double Motor drive + a Single Motor "shield" arm)
controlled by **fixed pitches played from a phone tone-generator app** into
a laptop microphone, coordinated with an opponent robot over MQTT for a
ball-vs-goalie match on topic `ME193/Rogers`.

Two computers, two phones:

- **Drive computer** — Bluetooth-connected to the robot. Its phone plays
  the drive tones (Double Motor).
- **Shield co-pilot computer** — no robot connection. Its phone plays the
  shield tones (Single Motor); the co-pilot relays them to the drive
  computer over MQTT.

## The tones

| Computer | Tone | Does |
|---|---|---|
| Drive | **1000 Hz** | forward |
| Drive | **1250 Hz** | turn left (in place) |
| Drive | **1500 Hz** | turn right (in place) |
| Drive | **1750 Hz** | hold ≥ 0.5 s → "goal" (ball only) |
| Drive | *no tone* | stop |
| Shield | **4250 Hz** | shield up |
| Shield | **5600 Hz** | shield down (stays where it was last put) |

A tone counts if it's within ±100 Hz of these, so play the exact number.
The shield tones are all higher than the drive tones, and each computer
only listens for its own tones, so the two phones can play at the same
time without either computer reacting to the other one. The frequencies
were also picked so that no 2nd/3rd/4th harmonic of a drive tone lands
within 200 Hz of any command. A loud phone speaker distorts a little, and
without this spacing it could, for example, trigger "goal" or the shield.

Frequencies, the ±100 Hz width, and the noise gate live at the top of
`tone_policy.py`. Speeds, shield swing, and goal-hold time live at the top
of `world_cup.py`.

## Two virtual environments — read this first

This project **must use PyAudio** (assignment requirement), and PyAudio has
no prebuilt wheel for Python 3.14, so this project runs in its own venv:

| Venv | Python | Used for |
|---|---|---|
| `my_env/` | 3.14 | Everything else in this repo. |
| `my_env_audio/` | 3.13 | **Only this project** (`pyaudio`, `numpy`, `matplotlib`, `paho-mqtt`, `legoeducation`). |

```powershell
winget install Python.Python.3.13
py -3.13 -m venv my_env_audio
my_env_audio\Scripts\python -m pip install --upgrade pip
my_env_audio\Scripts\python -m pip install legoeducation pyaudio numpy matplotlib paho-mqtt
```

## Files

- **`tone_policy.py`** — FFT tone detection: finds the strongest peak in
  this computer's frequency range, checks it's a clean tone and not noise,
  and names the command. No hardware, no I/O.
- **`test_tone_policy.py`** — synthetic-signal self-check for the above (no
  mic or robot needed). Run it any time `tone_policy.py` changes.
- **`world_cup.py`** — the match script (setup dialog, motors, MQTT, dashboard).
- **`pyaudio_mic.py`** — microphone picker built on PyAudio.
- **`songs.py`** — win/lose melodies, played on the hub and the computer at once.

## Run

```powershell
my_env_audio\Scripts\python "Public stuff\projects\Project 3 - World Cup\test_tone_policy.py"
my_env_audio\Scripts\python "Public stuff\projects\Project 3 - World Cup\world_cup.py"
```

Run `world_cup.py` on **both** computers. The setup dialog asks for the game
role (Ball/Goalie), which computer this is (Drive / Shield co-pilot), your
team name (**must be identical on both computers**), and optionally your
opponent's team name.

Before a match, on the drive computer:

- Fill in the `*_CARD_SERIAL` / `*_CARD_COLOR` values in `world_cup.py`.
- **Start with the shield arm DOWN.** Wherever the arm is when the script
  connects counts as "down"; "up" swings it `SHIELD_UP_DEGREES` (90°) from
  there. Make that negative if it swings the wrong way.
- Watch the car drive once and flip `INVERT_LEFT_MOTOR` /
  `INVERT_RIGHT_MOTOR` if a wheel spins the wrong way.
- Calibrate `PROXIMITY_REFLECTION_THRESHOLD` (default 70, on a ~0-100
  scale) against the real opponent robot and room lighting.
- Motors only move once the match is **LIVE** (`start` on MQTT, or press
  **s** in the dashboard to test locally; **r** resets).
- On Windows, turn off the mic's "audio enhancements" / noise suppression
  (Sound settings → the mic → Advanced). Some laptops treat a steady tone
  as background noise and fade it out after a second or two.

The dashboard shows the live spectrum with this computer's command bands
highlighted (green) and the other computer's greyed out, the detected peak
and its tonal ratio, the current command, what the wheels/shield were last
sent, and the MQTT feed. If the phone is playing but the command says
NONE, look at the ratio: below the gate means the tone is too quiet or the
room too loud — move the phone closer to the mic or turn it up.

## The required write-up

### Describe the policy — how does it make decisions?

Every ~46 ms audio block (2048 samples at 44.1 kHz) gets a Hann-windowed
FFT. Then each of **this computer's own** command bands (±100 Hz around
each tone) is checked separately. A band "hears" its command only if:

- the loudest point inside it is a real peak, not the band's edge (an edge
  maximum is just the spill-over from a strong tone outside the band), and
- it passes the noise gate below.

If several bands pass, the loudest one wins.

A new command takes effect after it's heard in **2 blocks in a row**, so a
single stray block can't jerk the robot. The drive computer maps the
command straight to wheel speeds: forward = both wheels at 60 %,
left/right = wheels opposite at 40 % (turn in place). Motor commands are
only sent to the robot when the command **changes**, not every loop tick.

The shield co-pilot latches: the last up/down tone it heard sticks until
the other is played. It publishes `{"shield": "up"|"down"}` on change (and
re-sends every second in case a message is lost). The drive computer moves
the Single Motor to 0° or 90° from its starting position when that changes.

Goal: holding the 1750 Hz tone for 0.5 s publishes the goal message and
plays the success song (ball only). Needing a held, dedicated tone means a
drive tone that drops out can't be mistaken for a goal.

### What does your code do if no tone is detected?

Dropouts shorter than 0.3 s (`HOLD_S`) are ignored: the current command
keeps going, so a brief glitch doesn't stutter the car. After 0.3 s of no
tone, the command becomes "none" and the car stops. The car never keeps
driving on a command nobody is playing. The shield stays where it was last
put, since it's a position, not a motion. If the microphone stops delivering
audio at all (device unplugged, laptop asleep), the last reading expires
after 0.5 s and the car stops. Separately, any error in the drive part of
the control loop stops the car, and the car and shield are always stopped
before disconnecting on exit.

### How did you try to mask out unwanted noise?

1. **Only its own bands.** Each computer only checks the narrow bands
   around its own tones. That ignores low room rumble, most of a voice,
   and the other team member's phone, even when the other phone is louder.
2. **Tonal-ratio gate.** A phone tone is one sharp spike. Talking, motors,
   and room noise are spread across the spectrum. The peak must be at least
   **8×** the spectrum's *median* level (`TONAL_RATIO_GATE`). White noise
   never gets above ~4; a tone quieter than the room noise still clears 8.
   The median is used instead of the mean, so the other phone's spike
   doesn't raise the bar.
3. **Real peak + confirmation.** The peak has to be an actual peak inside
   the band, not spill-over from a nearby sound. A new command also has to
   hold for 2 blocks in a row before it's used.

## MQTT protocol on `ME193/Rogers`

Uses `mqttlib.MQTTClient` (`test.mosquitto.org`, `qos=0`, no retain).

- **`"start"`** — plain text, from the instructor. Nothing moves before it.
- **`{"event": "fail", "team": "<TEAM_NAME>"}`** — the ball publishes this
  when its front sensor sees the opponent (reflection ≥ threshold), stops,
  and plays `DEATH_SONG`.
- **`{"event": "goal", "team": "<TEAM_NAME>"}`** — the ball publishes this
  on the goal tone and plays `SUCCESS_SONG`.
- If a message's `team` is the opponent's name you typed in the setup
  dialog, you react the opposite way (their fail = you win, their goal =
  you lose). Your own echoed messages and other teams' messages are
  ignored. **Agree on both team names with your opponent before the match.**
- Shield relay, team-scoped: `ME193/Rogers/control/<TEAM_NAME>` carries
  `{"shield": "up"|"down"}` from the co-pilot to the drive computer.

## Hardware / concurrency notes

- `legoeducation` calls block their calling thread on a real BLE
  round-trip even with `blocking=False`, so **no BLE calls happen in the
  PyAudio callback**. The audio callback only does the FFT. One control
  loop thread (every 50 ms) does all motor/sensor/song calls and game-event
  publishes. The MQTT thread only updates shared state, and the dashboard
  only reads it.
- `lelib`'s `doubleMotor.stop()` only stops the left motor (it calls
  `motor_stop()` with the default motor index 0), so this script always
  uses `car.movement_stop()` instead.

## Status

`test_tone_policy.py` passes. It covers every tone, each computer ignoring
the other's tones (including when both play at once and the other phone is
louder), harmonics, noise-only input, a tone quieter than the noise,
dropouts, stray blocks, switching, heavily distorted tones (harmonics as
loud as the tone), and loud tones just outside a band. The drive and co-pilot control loops
have also been exercised with fake motors/MQTT to confirm the commands they
send. **Not yet run against the physical robot or a real phone + mic** —
that still needs to happen, including checking the speeds, turn direction,
and shield swing direction feel right.
