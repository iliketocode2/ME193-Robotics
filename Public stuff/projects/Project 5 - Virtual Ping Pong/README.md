# Project 5: Rogers Cup Virtual Ping-Pong

You play Wii Sports-style table tennis against a virtual opponent. You hold a
**LEGO Double Motor** as your paddle. The webcam cuts you out of the room and
places you at a 3D table.

| Course skill | What it does in the game | Where |
|---|---|---|
| **Pose** (MediaPipe) | Your wrist position becomes the paddle position. A hit only counts if the paddle is on the ball. | `vision.py` |
| **IMU** (Double Motor) | Gyro magnitude spikes become swings. A hit only counts if you swing while the ball is in your zone. | `paddle_imu.py` |
| **AprilTags** | Hold up a tag to pick your opponent, then press ENTER to start. The opponent sets the ball speed. | `vision.py`, `opponents.py` |
| **MQTT** | Your record number of continuous hits is posted live as a float. | `pingpong.py` |
| **Motors** | They buzz on a hit and rumble on a miss. The hub light shows green, red or yellow. | `paddle_imu.py` |
| **Commentary booth** | Two commentators: Ray calls the play instantly from a script, and Sonia, written live by a **local open-source AI model** on the laptop's GPU, gives colour commentary. | `commentary.py`, `ai_commentator.py` |
| **Stadium crowd** | Real recordings of football crowds: ambience, cheers, roars, applause, gasps and chants. | `web/audio.js`, `web/audio/` |

## Opponents
| Tag | Opponent | Level | Ball speed | Personality |
|---|---|---|---|---|
| 0 | Pip the Penguin | Easy | 3.0 m/s | Goofy and cheerful, hits slow loopy lobs, misses a lot |
| 1 | Coach Rita | Medium | 3.9 m/s | Competitive coach who gives tips and aims for the corners |
| 2 | Viktor the Wall | Hard | 4.9 m/s | A silent robot whose every shot curves, and who almost never misses |

Print the cards in `tags/`. To regenerate them, run `make_tags.py`. Keep the white border when you cut them out.

## Setup
```
my_env/Scripts/pip install websockets openvino-genai huggingface_hub
my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/ai_commentator.py" --setup
```
- `--setup` downloads the AI model, Qwen3-4B (about 2.2 GB), into the gitignored repo-root `models/` folder. Skip it and the game still runs, just with Ray alone.
- The first run downloads `pose_landmarker_lite.task` into the repo-root `models/` folder, which is gitignored.
- The browser page loads Three.js and the font from a CDN, so you need internet (the MQTT broker needs it anyway).
- **Use Microsoft Edge for the best voices.** Edge has natural-sounding British voices: "Ryan" for Ray and "Sonia" for Sonia. Chrome falls back to its Google UK voices.

## Run
```
my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/pingpong.py"
```
1. The Double Motor connects. Put your Connection Card values in `PADDLE_CARD_SERIAL`/`PADDLE_CARD_COLOR` at the top of `pingpong.py`. With `None` it connects to the first Double Motor it finds.
2. Pick a camera in the camlib window.
3. The game opens at http://localhost:8193/.
4. Hold up an opponent's tag. Their card lights up, and it stays picked after you lower the tag. Click **Start match** or press **ENTER**.
5. Stand about 2 m from the camera with your shoulders visible. Hold the motor in your right hand, or set `RIGHT_HANDED = False` in `vision.py`.

**Hold the motor with nothing attached to its shafts. They spin for the haptic buzz.**

Everything can be clicked on screen. Each button also has a keyboard shortcut:

| On screen | Key | Action |
|---|---|---|
| **Start match** | ENTER | Play the opponent whose tag you held up |
| **🏠 Home** (top) | — | Quit the current match. Click twice, so a stray click can't end your game. |
| **🏠 Home** (game over) | ENTER | Back to opponent select |
| **↻ Rematch** (game over) | R | Same opponent again |
| **🎙 Commentary** | M | Both commentators on/off (remembered between sessions) |
| **👥 Crowd** | C | Crowd noise on/off (remembered between sessions) |
| **🤖 Sonia** badge | — | Shows the AI booth status: *warming up* (about 20 s at launch), *on air*, or *offline* (hover for the reason) |

The game-over buttons unlock after 2.5 s, so a swing or keypress right at match point can't skip the result screen.

| Key only | Action |
|---|---|
| D | Debug overlay: live gyro magnitude vs. swing threshold, vision fps, frame age, paddle position |
| [ / ] | Make swing detection more / less sensitive |
| SPACE | Swing (only with `--sim`) |
| ESC | Quit the whole program (motors stop and BLE disconnects) |

**Sound starts after your first click or key press.** Browsers block audio until then.

Options:
- `--sim` plays without the Double Motor, using SPACE to swing.
- `--no-mqtt` turns off publishing.
- `--no-ai` runs without the AI commentator.

### Calibrating the swing
The hub's raw gyro units aren't documented, so `SWING_GYRO_THRESHOLD` in
`paddle_imu.py` is a starting guess.
1. Press **D** and swing.
2. The bar should go green on a real swing and stay yellow when you're just holding or moving the motor.
3. Tune it live with `[` / `]`.
4. Copy the number shown into `SWING_GYRO_THRESHOLD`.

## Rules
- **Hitting:** the opponent's shot bounces on your side and enters your hit zone. You score a hit if your paddle is on the ball (pose) **and** you swing (IMU) while it's in the zone. A swing up to 0.25 s early still counts.
- **Miss messages:** when you miss, the game tells you which signal failed: "Move your paddle to the ball!" (pose), "Swing the paddle!" (IMU), or "Step into the camera view!".
- **Aiming:** where the ball sits on your paddle, and your sideways hand speed, steer your return. A harder swing returns it faster.
- **Scoring:** games go to 11, win by 2, and the serve alternates every 2 points. To serve, just swing.
- **Streak:** your hits in a row. It resets only when **you** miss. The opponent's errors don't break it.
- **Record:** your best streak this session.

## Commentary booth
Like a TV broadcast, there are two commentators. Both are covered by `test_commentary.py`, which doesn't need the model.

### Ray: play-by-play (scripted, instant)
`commentary.Announcer` turns game events into lines from a script, so they're instant and never wait on anything.

| When | Example |
|---|---|
| Match start | "Welcome to the Rogers Cup! Today's match: you, versus Coach Rita." Plus an intro for each opponent |
| Every point | Why it ended, then the score: "Caught flat-footed! Seven, four, to Coach Rita." |
| Key moments | "Deuce!", "Game point, you!", "What a comeback!" |
| During rallies | Long rallies ("Ten shots and counting!"), streaks ("That's ten in a row!"), smashes |
| Records | Called when the point ends: "And a new personal best: seven in a row!" |
| Game over | The final result |

- **Delivery:** each line has an excitement level. Hyped lines are spoken higher and faster.
- **No talking over himself:** point calls cut in immediately, and optional chatter is dropped while he's still talking.

### Sonia: colour commentary (local AI)
Sonia's lines are written live by **Qwen3-4B**, an open-source language model. It runs locally on the laptop's Intel Arc GPU through Intel's OpenVINO GenAI, so it's free, needs no API key and no cloud, and no game data leaves the machine.

**What she knows:** `commentary.MatchStory` tracks what a real analyst would notice:
- momentum ("Rita has won 3 in a row")
- repeated mistakes ("caught out of position 3 times")
- rally length, comebacks, game point and deuce
- your streak and the opponent's personality

**When she's asked:** `commentary.Booth` asks her for a line at the match start, after interesting points (not every point; there's a cooldown), and at game over. Each request gets a random *focus* so she doesn't fall into one groove: a tactical tip, the crowd, the opponent's reaction, footwork, a joke, and so on. Her recent lines are passed along so she doesn't repeat herself.

**Example lines from test matches:**
- "Our player's feet are locked in place, but Viktor's spin is already on the other side of the net."
- "The silence in the stands is louder than Viktor's next shot."

**Safety net:** every AI line is filtered before it's spoken (`sanitize_line`):
- first sentence only, at most 20 words (she's asked for 12)
- no emojis or markdown
- misspelled names fixed
- dropped if it states the score (Ray already did) or is too close to one of her recent lines

**The game never waits for her:**
- **Separate process:** the model runs as `ai_commentator.py --worker`, talking JSON over stdin/stdout. Model loading and text generation can never stall the camera, the game tick or Bluetooth.
- **Late lines are dropped:** a line that arrives after the match has moved on (the next point ended, or the rally is 4+ shots in) is thrown away.
- **Offline is fine:** if the model is missing or crashes, the game carries on with Ray alone.
- **Dead ball only:** Sonia thinks only between points, on menus and at game over; see "Performance" below.
- **Speed (measured on this laptop, Core Ultra 9 288V, Arc 140V):**
  - Warm-up: about 10–25 s at launch, done while you pick a camera.
  - Lines: about 1.3 s each on an idle machine, 2.5–4 s with the game running.

**Benchmark and model choice:**
- Check this machine with `ai_commentator.py --bench`.
- Qwen3-4B won a comparison against Qwen3-1.7B (faster but repetitive) and Phi-4-mini (slower and muddled).
- To try another model, set `PINGPONG_AI_MODEL` (for example `OpenVINO/Qwen3-1.7B-int4-ov`) and rerun `--setup`.

### Voices and captions
- Both commentators share one speech queue, so they never talk over each other. Ray's point calls cut in; Sonia waits her turn.
- Every line also appears as a caption labelled **Ray** or **Sonia**.

## Stadium crowd
The crowd is made of real recordings: about 2.5 MB of trimmed clips in `web/audio/crowd/`.
- **Building the clips:** `make_crowd_audio.py` downloads the source recordings, trims them, matches loudness and builds seamless loops, using the ffmpeg bundled with `pip install imageio-ffmpeg`. It only needs re-running to change clips.
- **Sources:** Wikimedia Commons and OpenGameArt. Licences are public domain, CC0, CC BY 4.0 and CC BY-SA 4.0. Every author and licence is listed in `web/audio/CREDITS.md`.

| Layer | What it does |
|---|---|
| Ambience | A real football stadium (Austria vs Sweden, Vienna) plus big-crowd chatter. It hushes when a rally starts and swells between points. |
| Reactions | Cheers that grow with the rally (small → big → roar), applause, "ooh"s at near-misses and "aww"s when you lose a point. Each plays with random pitch and pan, and big ones are doubled, so it never sounds like one clip on repeat. |
| Chants | Real supporter chants: terrace singing, rhythmic clapping, a call-and-response. They start on a run of points, a big streak, game point, the match start or a win (at most one every 15 s), and fade out when the next rally starts. |
| Mixing | The crowd ducks under both commentators' voices. If the recordings can't load, a synthesized crowd stands in. |

## MQTT
- **Topic:** `ME193/Rogers/WilliamGoldman`
- **Payload:** the record as a bare float string, e.g. `7.0`.
- **When:** every time the record goes up, as a 1 Hz heartbeat, and once more on exit.
- **Broker:** `test.mosquitto.org` (via `mqttlib.py`).

Watch it with `projects/mqtt_chat/mqtt_test.py`, or any MQTT client subscribed to that topic.

## How it fits together
```
webcam -> vision thread (pose + segmentation + AprilTag) --+
Double Motor IMU -> imu thread (swings; sends haptics) ----+-> game tick 60 Hz (game_logic.py)
                                                           |     -> MQTT record
                                                           |     -> Ray (commentary.Announcer)
                                                           |     -> Booth -> AI worker process -> Sonia
                                                           +-> websocket -> browser (Three.js, web/)
```
- **Python is authoritative.** `game_logic.py` holds the rules and physics. It has no hardware code and is covered by `test_game_logic.py`:
  ```
  my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_game_logic.py"
  my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_commentary.py"
  my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_ai_worker.py"
  ```
- **The AI worker** (`ai_commentator.py --worker`) is a separate process. The game sends it requests and gate changes, and polls for answers without blocking. See "Commentary booth" and "Performance".
- **The browser only draws.** It gets 60 Hz game state plus, for each frame pose runs on, a small JPEG and that frame's raw person mask. A GPU shader uses the mask as transparency to cut you out. It sends back ENTER, D, [ ], SPACE and ESC.
- **Low camera latency:**
  - **No backlog:** a grab thread reads the webcam continuously and keeps only the newest frame, so frames never queue up behind slow pose detection.
  - **Matched images:** your cut-out uses the same frame pose ran on, so the cut-out edges and the virtual paddle stay on your body.
  - **Full-rate inset:** the select screen's camera inset gets every frame.
  - **Checking it:** the D overlay shows "frame age" (how old the frame behind the current paddle position is). It should stay under ~100 ms.
- **Bluetooth stays on one thread.** All BLE reads and commands happen on the IMU thread, so Bluetooth never stalls the camera or the game.
- **IMU update rate:** the hub's notifications are raised from 100 ms to 30 ms with `device_notification_request(30)`. At 10 Hz, fast swings slipped between readings.
- **Paddle/hand alignment:** `SHOULDER_WORLD` in `vision.py` is used both to map your hand to table coordinates and to scale your cut-out in 3D. That's why the virtual paddle lands on your real hand.

## Performance
The camera (MediaPipe on the CPU), the 3D graphics (WebGL on the GPU) and Sonia's AI (an LLM on the GPU) all share **one laptop chip**: one power budget, one memory bus, one GPU. Anything that wastes that budget shows up as camera lag. The rules below came out of profiling the full system (game server, AI, camera pipeline and the real page in GPU Edge) second by second.

| Rule | Why (measured) |
|---|---|
| **Sonia only thinks when the ball is dead.** The game holds a gate that closes the moment a rally starts; a line in progress is cancelled within one token. | While she generated during play, camera fps fell from 30 to ~22; during her warm-up, to ~17 with 4 s frame lag. Gated: **30 fps** in every rally second. |
| **The match start waits up to 25 s for her warm-up.** Pressing Start again starts at once; otherwise the match starts with Ray alone after 25 s. | Her one-time GPU compile must never overlap play. |
| **Muting commentary (🎙) switches her off entirely.** | Nothing is heard, so nothing is generated. |
| **1 ms Windows timer for the game loop.** | Windows' default 15.6 ms timer made the "60 Hz" loop run at 32 Hz with 100–150 ms hitches. Now 60 Hz, worst gap about 20 ms. |
| **The game process runs at above-normal priority and opts out of power throttling** (prefers the performance cores). | Worst tick gap 91 → 19 ms, worst frame age 142 → 87 ms. |
| **The browser draws at most 60 fps at 1× pixel ratio.** A governor drops shadows, then resolution, if frames stutter. | Uncapped 120 fps at 2× cost the camera about 8 fps. 1.25× caused 50–400 ms GPU stalls. |
| **The ball is extrapolated between server updates** with its own velocity. | Smooth motion even if an update arrives late. |
| **A short pause between points** (2.8 s, and 1.6 s before the opponent serves). | Room for Ray's call, the crowd, and Sonia's line before the next rally. |

**Measured result:** fake 30 fps camera, live AI, real page at 2× pixel density, three 100 s matches. Every rally second had:
- camera **30 fps**
- frame age median **35–40 ms**, max under 60 ms
- game loop **60 Hz**
- zero browser stalls

That matches the run with the AI switched off.

**Checking it on your machine:**
- Press **D** in game. The overlay shows render fps and quality, server tick rate, camera fps and frame age, and Sonia's state ("ball dead: may think" / "RALLY: AI paused").
- Run `pingpong.py --perf-log` to write a per-second CSV to the gitignored repo-root `logs/` folder.
