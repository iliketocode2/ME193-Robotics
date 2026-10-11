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
| **Arduino UNO Q** | The board's `ip-weather` app publishes the live weather at Tufts over MQTT; the arena's sky, light, rain/snow, wind and floodlights match it, and the commentators talk about it. | `weather.py`, `web/weather.js`, `ArduinoApps/ip-weather` |

## Opponents
| Tag | Opponent | Level | Ball speed | Personality |
|---|---|---|---|---|
| 0 | Pip the Penguin | Easy | 3.0 m/s | Goofy and cheerful, hits slow loopy lobs, misses a lot |
| 1 | Coach Rita | Medium | 3.9 m/s | Competitive coach who gives tips and aims for the corners |
| 2 | Viktor the Wall | Hard | 4.9 m/s | A silent robot whose every shot curves, and who almost never misses |

Print the cards in `tags/`. To regenerate them, run `make_tags.py`. Keep the white border when you cut them out.

## Setup
```
my_env/Scripts/pip install websockets openvino-genai huggingface_hub kokoro-onnx
my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/ai_commentator.py" --setup
my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/tts.py" --setup
```
- `ai_commentator.py --setup` downloads the AI model, Qwen3-4B (about 2.2 GB), into the gitignored repo-root `models/` folder. Skip it and the game still runs, just with Ray alone.
- `tts.py --setup` downloads the natural-voice model, Kokoro-82M (about 350 MB), and renders every line Ray can say (about 1,200 clips, 150 MB in `models/tts/`; 30–45 min the first time, and later runs only render what's new). Skip it and the commentators use the browser's voices.
- **Weather:** deploy the UNO Q's `ip-weather` app (`python tools/deploy.py ip-weather` in the ArduinoApps repo). The board only needs internet; it talks to the game through the public MQTT broker. Without it the PC fetches the same weather itself.
- The first run downloads `pose_landmarker_lite.task` into the repo-root `models/` folder, which is gitignored.
- The browser page loads Three.js and the font from a CDN, so you need internet (the MQTT broker needs it anyway).
- **Browser voices (fallback only):** without `tts.py --setup`, Edge's "Ryan" and "Sonia" voices are the best fallback; Chrome uses its Google UK voices.

## Run
```
my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/pingpong.py"
```
1. The Double Motor connects. Put your Connection Card values in `PADDLE_CARD_SERIAL`/`PADDLE_CARD_COLOR` at the top of `pingpong.py`. With `None` it connects to the first Double Motor it finds.
2. Pick a camera in the camlib window.
3. The game opens at http://localhost:8193/.
4. Hold up an opponent's tag. Their card lights up, and it stays picked after you lower the tag. (While it's up, Sonia already writes her pre-match lines.) Click **Start match** or press **ENTER**.
5. **The pre-match show** (at most 15 s): Ray and Sonia introduce the match, the players and the local weather while the camera sweeps the sky. Press **ENTER** or **Skip** to go straight to the countdown.
6. Stand about 2 m from the camera with your shoulders visible. Hold the motor in your right hand, or set `RIGHT_HANDED = False` in `vision.py`.

**Hold the motor with nothing attached to its shafts. They spin for the haptic buzz.**

Everything can be clicked on screen. Each button also has a keyboard shortcut:

| On screen | Key | Action |
|---|---|---|
| **Start match** | ENTER | Play the opponent whose tag you held up |
| **Skip ▸** (pre-match show) | ENTER | Skip the rest of the show (a second ENTER within 0.8 s is ignored, so a double-press doesn't skip it) |
| **🏠 Home** (top) | — | Quit the current match. Click twice, so a stray click can't end your game. |
| **🏠 Home** (game over) | ENTER | Back to opponent select |
| **↻ Rematch** (game over) | R | Same opponent again |
| **🎙 Commentary** | M | Both commentators on/off (remembered between sessions) |
| **👥 Crowd** | C | Crowd noise on/off (remembered between sessions) |
| **🤖 Sonia** badge | — | Shows the AI booth status: *warming up* (about 20 s at launch), *on air*, or *offline* (hover for the reason) |
| **🌧 48°F · UNO Q** badge | — | The live weather and where it came from: *UNO Q*, *PC (UNO Q silent)* or *demo* (hover for details) |

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
- `--no-weather` keeps the default sunny arena.
- `--weather-demo clear|rain|storm|snow|fog|night|windy` fixes the weather, to try the visuals (or check performance) on demand.

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
| Pre-match show | "Welcome to the Rogers Cup! It's you versus Coach Rita." "Live from Tufts, on a rainy evening. Fifty-two degrees. Sonia?" |
| Every point | Why it ended, then the score: "Caught flat-footed! Seven, four, to Coach Rita." |
| Key moments | "Deuce!", "Game point, you!", "What a comeback!" |
| During rallies | Long rallies ("Ten shots and counting!"), streaks ("That's ten in a row!"), smashes |
| Records | Called when the point ends: "And a new personal best: seven in a row!" |
| Game over | The final result |
| Weather changes | Between points, when it has really changed: "And here comes the rain.", "The floodlights are on as night falls." |

- **Delivery:** each line has an excitement level. Hyped lines are spoken higher and faster.
- **No talking over himself:** point calls cut in immediately, and optional chatter is dropped while he's still talking.

### Sonia: colour commentary (local AI)
Sonia's lines are written live by **Qwen3-4B**, an open-source language model. It runs locally on the laptop's Intel Arc GPU through Intel's OpenVINO GenAI, so it's free, needs no API key and no cloud, and no game data leaves the machine.

**What she knows:** `commentary.MatchStory` tracks what a real analyst would notice:
- momentum ("Rita has won 3 in a row")
- repeated mistakes ("caught out of position 3 times")
- rally length, comebacks, game point and deuce
- your streak and the opponent's personality

**When she's asked:** `commentary.Booth` asks for her pre-match show lines while you pick an opponent, then for lines after interesting points (not every point; there's a cooldown), and at game over. Each request gets a random *focus* so she doesn't fall into one groove: a tactical tip, the crowd, the opponent's reaction, footwork, a joke, and so on. Her recent lines are passed along so she doesn't repeat herself.

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
- **Late lines are dropped:** a line about a point is fresh until the next point ends. It may play over the next rally (TV analysts do), but once the next point is over it's thrown away.
- **Offline is fine:** if the model is missing or crashes, the game carries on with Ray alone.
- **Dead ball only:** Sonia *writes* only between points, on menus and at game over (her *voice* may be synthesized any time: it's CPU-only); see "Performance" below.
- **Speed (measured on this laptop, Core Ultra 9 288V, Arc 140V):**
  - Warm-up: about 10–25 s at launch, done while you pick a camera.
  - Lines: about 1.3 s each on an idle machine, 2.5–4 s with the game running.

**Benchmark and model choice:**
- Check this machine with `ai_commentator.py --bench`.
- Qwen3-4B won a comparison against Qwen3-1.7B (faster but repetitive) and Phi-4-mini (slower and muddled).
- To try another model, set `PINGPONG_AI_MODEL` (for example `OpenVINO/Qwen3-1.7B-int4-ov`) and rerun `--setup`.

### Voices and captions
Both commentators speak with **Kokoro-82M**, a small open-source neural voice model (`tts.py`), because the browser's built-in voices sound robotic. Ray is `bm_george` and Sonia is `bf_emma`, both British; change `VOICES` in `tts.py` to try others.

| | How | In-game cost |
|---|---|---|
| **Ray** | Every sentence he can say is rendered once by `tts.py --setup`. A call like "Caught flat-footed! Four, two, to Pip." plays two clips back to back. | None. The browser just plays files. |
| **Sonia** | Her line is written live, so it's voiced live. The AI worker synthesizes it on 2 CPU threads right after the text -- even mid-rally (CPU only, measured harmless). The line is held until its audio is ready, and it's dropped if the match has moved on by then. | About 3 s of 2 CPU cores per line. Measured: see "Performance". |

- **Speed:** ~0.85 s of CPU per second of speech on this laptop. 2 threads is as fast as 8, so it leaves the camera's cores alone. The Arc GPU can't run Kokoro (OpenVINO's GPU plugin lacks one of its ops), and ONNX Runtime was 2–3× slower than OpenVINO.
- **Fallback:** a line with no clip (a rare number above 100, a Sonia line the worker couldn't voice in time, or no setup) uses the browser's voice. A call never mixes two voices.
- Both commentators share one speech queue, so they never talk over each other. Ray's point calls cut in; Sonia waits her turn.
- **Sharing the airtime:** on a point Sonia has been asked about, Ray calls only the score ("Four, two, to Pip.") and leaves the colour to her. If his next point call comes while she's finishing a sentence, he waits for her.
- **Speech never fails silently:** Edge's "Natural" voices stream from an online service. If a line hasn't started within 2.5 s (slow or blocked network, stuck speech engine), it's replayed with an installed Windows voice (David / Zira, or George / Hazel), and those are used for the rest of the session. Press **D** to see the voices in use, lines spoken, and any failure.
- Every line also appears as a caption labelled **Ray** or **Sonia**, even if the speech fails. The **D** overlay counts the lines heard from each ("heard: Ray 12 · Sonia 4").

### The pre-match show
After ENTER, before the countdown: at most 15 s (9 s for a rematch, 6 s with commentary muted), skippable.

| | Who | What |
|---|---|---|
| 1 | Ray | Welcome, naming the opponent |
| 2 | Ray | The weather: "Live from Tufts, on a grey night. Forty-seven degrees." (+ strong wind), handing over with "Sonia?" if her line is ready |
| 3 | Sonia | The opponent, given the conditions -- or Ray's opponent intro if her line isn't ready |
| 4 | Sonia | What our player must do -- or Ray on the player ("Coach Rita won the last meeting.") |
| 5 | Ray | "Let's play!" (if it still fits) |

- **Sonia's lines are written ahead:** while your tag is up on the menu (the ball is dead there anyway) she writes and voices both, so they're ready the moment you press ENTER. Switching tags keeps each opponent's lines; ENTER drops the others' from her queue.
- **Gives her time to warm up:** if she's still loading at launch, Ray says "Sonia is still finding her seat in the booth." and covers for her; she joins as soon as she's ready.
- `commentary.ShowDirector` runs the script and times it from the clips' real lengths; a line that wouldn't finish in time is left out and the show ends as soon as the script is done. Lower-third cards (title, weather, opponent, player) and a slow camera sweep of the sky dress it; a progress bar on **Skip** shows the time left.

## Live weather (Arduino UNO Q)
The board's **ip-weather** app (`C:\Users\goldm\GitHub Projects\ArduinoApps\ip-weather`) fetches the current weather for Tufts from Open-Meteo every 5 minutes, scrolls it on its LED matrix (`48F CLOUD`, then its IP), and publishes it over MQTT:

- **Topic:** `ME193/WilliamGoldman/weather` -- JSON (`weather_payload.py`), **retained**, so the game gets the latest reading the moment it subscribes; re-published every 60 s as a heartbeat. `.../weather/status` is `online`/`offline`.
- **Fields:** temperature and feels-like (°F), humidity, WMO weather code, precipitation/rain/showers/snowfall, cloud cover, wind speed/gusts (mph) and direction, day/night, visibility, sunrise/sunset.
- **The game (`weather.py`):** subscribes while it runs. No reading from the board within 5 s, a reading over 30 min old, or the board silent for 3 min (it re-subscribes) -> the PC fetches the same data from Open-Meteo itself. Payloads are validated and clamped (it's a public broker). `test_weather.py` checks the board's payload code against the game's.

What you see (`web/weather.js`) -- changes ease in over a few seconds, mid-match too:

| Weather | In the arena |
|---|---|
| Sun / cloud | Sky gradient and sun position from the real sunrise/sunset; drifting clouds as thick as the cloud cover; storm clouds dark |
| Night | Dark sky, stars and the moon (with its phase) when clear; floodlight towers switch on, with light beams in rain and fog |
| Rain / drizzle / showers / storm | Rain streaks slanted by the wind, wet darker ground, rain sound; storms add lightning (at most one strike every 4 s, dimmed during rallies) and thunder |
| Snow | Drifting flakes, the ground whitening |
| Fog / mist | Haze that hides the far stands (never closer than 6.5 m: the table and opponent stay clear) |
| Wind | Pennants on the stands stream and flap with the speed and gusts, clouds drift, rain slants; wind sound |

The commentators: Ray reports it in the show and calls notable changes between points (a change must hold for a minute; at most one call every 3 minutes); Sonia's show lines use it, and mid-match she sometimes riffs on notable weather.

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
- **Weather in:** `ME193/WilliamGoldman/weather` from the UNO Q (see "Live weather").
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
  my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_weather.py"
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
| **The pre-match show covers her warm-up** (and she writes her show lines on the menu). | Her one-time GPU compile shouldn't overlap play; a skipped show while she's still loading is the one exception. |
| **Natural voices: Ray is pre-rendered; Sonia is voiced on 2 CPU threads, any time (also mid-rally).** | Real pose pipeline at 30 fps: still **30 fps** with her voice synthesizing non-stop (far more than real play); frame age +5 ms (median 27 → 33 ms). Kokoro is no faster on 8 threads than on 2. Before this, gating her voice too meant most of her lines missed their moment. |
| **Weather is built once and only re-tinted.** Rain/snow is one instanced draw animated in the vertex shader; no new lights, fog types or shader variants after startup (they'd recompile and stall). Effects are cut before shadows when the governor steps down, and it ignores menu/show seconds and seconds when the AI is busy. | Full system with a storm (6000 drops, lightning) or snow: camera **30 fps** every second in play, frame age max 58 / 50 ms, game loop 60 Hz, page 60 fps, quality stayed *high* -- the same as with no weather. |
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
