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

## Opponents
| Tag | Opponent | Level | Ball speed | Personality |
|---|---|---|---|---|
| 0 | Pip the Penguin | Easy | 3.0 m/s | Goofy and cheerful, hits slow loopy lobs, misses a lot |
| 1 | Coach Rita | Medium | 3.9 m/s | Competitive coach who gives tips and aims for the corners |
| 2 | Viktor the Wall | Hard | 4.9 m/s | A silent robot whose every shot curves, and who almost never misses |

Print the cards in `tags/`. To regenerate them, run `make_tags.py`. Keep the white border when you cut them out.

## Setup
```
my_env/Scripts/pip install websockets      # once (everything else is already in my_env)
```
- The first run downloads `pose_landmarker_lite.task` into the repo-root `models/` folder, which is gitignored.
- The browser page loads Three.js and the font from a CDN, so you need internet (the MQTT broker needs it anyway).

## Run
```
my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/pingpong.py"
```
1. The Double Motor connects. Put your Connection Card values in `PADDLE_CARD_SERIAL`/`PADDLE_CARD_COLOR` at the top of `pingpong.py`. With `None` it connects to the first Double Motor it finds.
2. Pick a camera in the camlib window.
3. The game opens at http://localhost:8193/.
4. Hold up an opponent's tag. Their card lights up, and it stays picked after you lower the tag. Press **ENTER**.
5. Stand about 2 m from the camera with your shoulders visible. Hold the motor in your right hand, or set `RIGHT_HANDED = False` in `vision.py`.

**Hold the motor with nothing attached to its shafts. They spin for the haptic buzz.**

| Key (browser) | Action |
|---|---|
| ENTER | Confirm the opponent / play again after game over |
| D | Debug overlay: live gyro magnitude vs. swing threshold, vision fps, paddle position |
| [ / ] | Make swing detection more / less sensitive |
| SPACE | Swing (only with `--sim`) |
| ESC | Quit (motors stop and BLE disconnects) |

Options:
- `--sim` plays without the Double Motor, using SPACE to swing.
- `--no-mqtt` turns off publishing.

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
                                                           +-> websocket -> browser (Three.js, web/)
```
- **Python is authoritative.** `game_logic.py` holds the rules and physics. It has no hardware code and is covered by `test_game_logic.py`:
  ```
  my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_game_logic.py"
  ```
- **The browser only draws.** It gets 60 Hz game state plus, for each frame pose runs on, a small JPEG and that frame's raw person mask. A GPU shader uses the mask as transparency to cut you out. It sends back ENTER, D, [ ], SPACE and ESC.
- **Low camera latency:**
  - **No backlog:** a grab thread reads the webcam continuously and keeps only the newest frame, so frames never queue up behind slow pose detection.
  - **Matched images:** your cut-out uses the same frame pose ran on, so the cut-out edges and the virtual paddle stay on your body.
  - **Full-rate inset:** the select screen's camera inset gets every frame.
  - **Checking it:** the D overlay shows "frame age" (how old the frame behind the current paddle position is). It should stay under ~100 ms.
- **Bluetooth stays on one thread.** All BLE reads and commands happen on the IMU thread, so Bluetooth never stalls the camera or the game.
- **IMU update rate:** the hub's notifications are raised from 100 ms to 30 ms with `device_notification_request(30)`. At 10 Hz, fast swings slipped between readings.
- **Paddle/hand alignment:** `SHOULDER_WORLD` in `vision.py` is used both to map your hand to table coordinates and to scale your cut-out in 3D. That's why the virtual paddle lands on your real hand.
