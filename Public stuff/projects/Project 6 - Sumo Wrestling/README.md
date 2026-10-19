# Project 6 – Sumo Wrestling

## Starting everything (match day)

The system runs in three places, so there is no single command. Start them in this order:

1. **Phone:** open the IP-camera app and start streaming.
2. **UNO Q:** plug it into the battery pack and turn on the double motor and color sensor. That's all, once the relay autostart is installed (below). The App Lab default app starts the fan-speed reporter, and the `sumo-relay` service connects to the LEGO devices by itself within about 30 s of boot. The dashboard's **Robot**, **Motor** and **Color** pills turn green when it's connected.
   - **One-time setup** (board on USB, in the board terminal):
     ```
     cd ~/ArduinoApps/<your app>/python && bash install_autostart.sh
     ```
     Our app folder is `ble-probe`. It asks for the `arduino` user's password once (for `sudo`) and installs a systemd service. The service starts the relay at every boot, waits for Wi-Fi and Bluetooth, keeps retrying until the LEGO devices are on, and restarts the relay if it ever crashes.
   - **Check on it:** `journalctl -u sumo-relay -f` shows the relay's live log. `sudo systemctl stop sumo-relay` / `start sumo-relay` stop and start it.
   - **Running it by hand instead** (`bash run_on_host.sh`): first `sudo systemctl stop sumo-relay`. Two relays would fight over the LEGO devices.
   - **Remove the autostart:** `bash install_autostart.sh --remove`.
3. **Laptop:** in PowerShell from the repo root, start the brain:
   ```
   my_env\Scripts\python "Public stuff/projects/Project 6 - Sumo Wrestling/sumo_brain.py"
   ```
   The dashboard opens at http://localhost:8293. Check that the status pills are green before the match.
4. **Before each match, with the ring EMPTY** (no robots, no hands), click **Snapshot empty arena**. After that, only what changes from the snapshot can become the opponent. Take a new one if the phone moves or the lighting changes. Then put both robots in. The video header should read `field: closed tape loop` (or `field: partial, NN% tape` if the tape has gaps). If it says "no field", tune **Calibrate → HSV** until the tape is found.

**To stop:** press Ctrl-C on the laptop. The robot stops on its own within 0.6 s whenever the laptop stops sending commands. Unplugging the battery stops the board.

The relay runs on the board's own Linux (as the `sumo-relay` service), not inside App Lab, because App Lab's container can't reach Bluetooth (see "Setup and run order", step 3).

---

Two robots in a ring of blue tape (not necessarily a perfect circle). The first one pushed fully out loses and has to stop and play a sad song.

Our robot is a LEGO Education **double motor**. A LEGO **color sensor** is mounted on its rear, looking down. An **Arduino UNO Q** rides on top. An **overhead phone camera** streams to the laptop, which runs the vision and a **behavior tree**. A web dashboard shows everything and lets us step in.

```
 phone (IP camera app) ──MJPEG──▶ laptop: sumo_brain.py ──────────────▶ web dashboard
                                   │ vision.py: blue-tape field + AprilTag #6 (us) + opponent (tag / YOLO)
                                   │ behavior_tree.py (15 Hz)              ▲ E-stop, start/pause,
                                   │                                       │ WASD, click target,
                                   ▼                                       │ strategy
                         MQTT  test.mosquitto.org
          ME193/<TEAM>/sumo/cmd ▼            ▲ .../sensor  .../status
                     UNO Q (unoq_app/): relay.py ──BLE──▶ LEGO double motor (tank drive, beeps)
                                        relay.py ◀─BLE── LEGO color sensor (rear, facing floor)
                                        sketch.ino ◀─tach─ Arctic P12 PRO fan (on 5V, always on)
```

## Files

| File | What it does |
|---|---|
| `sumo_brain.py` | **Run this.** Camera → vision → behavior tree → MQTT commands, plus the dashboard server (http://localhost:8293). |
| `behavior_tree.py` | The decision policy: a priority behavior tree (see below). |
| `vision.py`, `field.py` | Finds the field as the space the blue tape **encloses**, in any shape (no circle assumed); stored as center + distance-to-tape in each of 180 directions. AprilTag pose for us (#6: position + heading). The opponent is found from **any other AprilTag** first (instant and reliable), then the YOLO model, then change detection. |
| `config.py` | Laptop settings: phone URL, our tag id, opponent tag families (`OPPONENT_TAG_DICTIONARIES`), HSV defaults, ports, timing. |
| `unoq_app/` | UNO Q App Lab app: `python/main.py`, `python/relay.py` (MQTT↔BLE), `python/sumo_topics.py` (**shared MQTT schema + card IDs**), `sketch/sketch.ino` (fan RPM from the tach wire). |
| `unoq_app/python/ble_probe_unoq.py` | Step 1: checks whether the UNO Q can reach the LEGO devices over Bluetooth. |
| `ble_relay_laptop.py` | Fallback: runs the same relay on the laptop if the UNO Q can't do BLE. |
| `songs_bridge.py` | Win/lose songs: Project 3's mp3s on the laptop, plus the same note lists beeped on the double motor. |
| `make_tag.py` → `tags/` | Our AprilTag (tag36h11 **#6**). The `_print.png` version prints at exactly 7 cm. |
| `capture_frames.py`, `train_sumo.py` | Collect frames → label in Roboflow → train the local YOLO model. |
| `test_sumo_offline.py` | No-hardware test of vision (circular, rectangular, lopsided and broken-tape fields; opponent tags) and every behavior-tree branch, using synthetic frames. |
| `web/` | Dashboard (plain HTML/CSS/JS, no build step). |

## The field (any shape)

The tape isn't assumed to be a circle. `vision.py` finds the space the blue tape encloses, and `field.py` stores it as a center plus the **distance to the tape in each direction** (a radial profile). That handles ovals, rectangles, wobbly hand-taped loops and dents.

- Every robot gets a **normalized radius** `r`: 0 at the center, **1 at the tape in that direction**, more than 1 outside. All behavior-tree thresholds ("too close to the edge", "pushed out") are values of `r`, so they work for any field size, shape or camera height.
- Positions, headings and clicked targets stay in **camera pixels**, so turn angles are exact on any shape. Distances such as "close enough to push" are divided by the field's size.
- **Closed vs. partial:** if the tape forms a closed loop, its exact outline is used (`field: closed tape loop`). If the tape has gaps (taped badly, or covered by a robot), the outline is estimated from the pieces that are visible (`field: partial, NN% tape`).
- A new field must be seen for a few frames before it's believed, and it's dropped about 1 s after the tape disappears. At match start the field **locks** so robots covering the tape can't change it. **Calibrate → Re-detect field** unlocks it.
- The dashboard draws the real outline, plus the edge limit and "out" line as the same outline scaled toward the center (the exact `r` lines the brain uses).

## Opponent tags

If the opponent has an AprilTag on top, it's the most reliable cue: it is confirmed on the first frame. **Any** tag inside the field other than our #6 is treated as the opponent. Tags outside the field (a robot waiting its turn) are ignored and drawn in grey. The course uses tag36h11. If a classmate printed another family, add it to `OPPONENT_TAG_DICTIONARIES` in `config.py` (each extra family costs a few ms per frame).

## Fan wiring (Arctic P12 PRO, 4-pin PC fan, no motor driver)

The fan runs from the UNO Q's 5V pin and is always on while the board has power. The UNO Q only *measures* its speed.

| Fan connector pin (key/notch side = pin 1 edge) | Function | Connect to |
|---|---|---|
| 1 | GND | UNO Q **GND** |
| 2 | +12 V supply | UNO Q **5V** |
| 3 | Tach / sense (2 pulses per turn) | UNO Q **D2** (`TACH_PIN` in `sketch.ino`). No resistor needed; the sketch uses the internal pull-up. |
| 4 | PWM control | **Leave unconnected.** A floating PWM line means full speed. |

- Identify the wires by **position in the connector**, not by color. Arctic cables are often all black.
- The P12 PRO is a 12 V fan, so at 5 V it spins well below its rated speed. Check that it starts reliably from a standstill at 5 V. The dashboard's **Fan** pill shows the measured RPM, and a red pill at "0 rpm" means it stalled.
- Don't wire pin 4 to an Arduino pin. The fan pulls that line up internally, and it doesn't need it.

## Setup and run order

All laptop commands run from the repo root, using `my_env` (Python 3.14 with opencv, ultralytics, websockets and paho-mqtt already installed).

1. **Fill in the placeholders.**
   - `unoq_app/python/sumo_topics.py`: `TEAM` and both devices' card serials/colors.
   - `config.py`: `PHONE_STREAM_URL`, the phone app's MJPEG URL (as in Project 2).
2. **Print the tag.** Use `tags/sumo_tag36h11_id6_print.png` at 100% scale. Mount it flat on top with the arrow pointing to the robot's **front** (the pushing side; the color sensor is at the back).
3. **BLE probe (decides where the relay runs).** Deploy `unoq_app` (`python tools/deploy.py <app>`, see `UNOQ code/DevelopingUNOQAppsInVSCode.md`), then:
   - **(a) Inside the App Lab container: tested, does NOT work.** `unoq_probe_app/` showed the container has the adapter (`hci0`) but no D-Bus socket, so bleak can't scan. That's why `RELAY_BLE = False` in `main.py`.
   - **(b) On the UNO Q host (the plan now).** From a terminal on the board (App Lab's terminal or SSH): `cd` into the app's `python/` folder, then run `bash run_on_host.sh ble_probe_unoq.py --drive` (robot on blocks). If that passes, run `bash run_on_host.sh` to start the relay. The first run creates `~/sumo-venv`.
   - **(c) Fallback on the laptop.** Set `RELAY_BLE = False` and run `my_env\Scripts\python "Public stuff/projects/Project 6 - Sumo Wrestling/ble_relay_laptop.py"`.
4. **Offline check:**
   `my_env\Scripts\python "Public stuff/projects/Project 6 - Sumo Wrestling/test_sumo_offline.py"` → should print `ALL PASSED`.
5. **Brain and dashboard:**
   `my_env\Scripts\python "Public stuff/projects/Project 6 - Sumo Wrestling/sumo_brain.py"`
   - Turn **Dry run** on first. Move the robots around by hand and watch "Next action" and the tree.
   - Click a spot to check that the robot would turn the right way. If turns are mirrored, set `CAMERA_MIRRORED` in `config.py`. If the real wheels turn wrong, set `FLIP_DRIVE` / `SWAP_SIDES` in `relay.py`.
   - Open **Calibrate** and tune the HSV sliders until the blue field outline hugs the inner edge of the tape. The header should say "field: closed tape loop"; "partial" means the tape has gaps the camera can't bridge.
6. **Vision model (optional but better).** If the opponent has no AprilTag and there's no model, the brain uses **change detection**: the opponent is whatever differs from the empty-arena snapshot (or the floor color) and is robot-sized.
   - Run `capture_frames.py`.
   - Upload the frames from `runs/sumo_frames/` to Roboflow (app.roboflow.com, workspace `william-goldman`) and label the opponent cars. Our current model is the `lego-car-detection` project, class `Opponent`. Labeling our own robot too is optional; vision ignores whichever box holds our tag.
   - Export as **YOLOv8** with "Resize to 512×512 (Stretch)". Unzip the export into its own folder here (ours is `Lego Car Detection.v1.yolov8/`) and set `DATASET_DIR` in `train_sumo.py`.
   - **Train with `my_env_cpu`, not `my_env`:**
     `my_env_cpu\Scripts\python "Public stuff/projects/Project 6 - Sumo Wrestling/train_sumo.py"`
     Windows **Smart App Control** blocks the Intel-GPU PyTorch DLLs in `my_env` (`WinError 4551`). `my_env_cpu` is a gitignored venv with CPU-only PyTorch, ultralytics, legoeducation, websockets and paho-mqtt. 30 images took about 37 min.
   - Training writes `<repo>/models/sumo_yolo.pt` **and** `sumo_yolo.onnx` (both gitignored). The brain, still run with plain `my_env`, loads the `.onnx` with **OpenVINO on the Intel NPU** (about 16 ms of model time per frame, no PyTorch involved). It falls back to the GPU, then the CPU, then change detection. The video header shows which one is in use.

## MQTT schema (qos 0, never retained)

Here `T = ME193/<TEAM>/sumo`. The source of truth is `unoq_app/python/sumo_topics.py`.

| Topic | Direction | Payload |
|---|---|---|
| `T/cmd` | laptop → robot, 15 Hz | `{"seq","l","r","mode"}`. `l`/`r` are −100…100 tank speeds. The robot stops if none arrive for 0.6 s. |
| `T/song` | laptop → robot | `{"song":"win"\|"lose"}` |
| `T/sensor` | robot → laptop, ~20 Hz | `{"color","name","reflection","hue","sat","val","rgb","ts"}` |
| `T/status` | robot → laptop, 1 Hz | `{"motor","color","uptime","host","cmd_fresh"}` |
| `T/unoq` | UNO Q → laptop, 1 Hz | `{"fan_rpm","relay_ble","ts"}`: fan speed from the tach wire |
| `T/result` | laptop | `{"event":"lose"\|"win","team"}` |
| `T/chat` | dashboard | plain text |
| `ME193/Rogers` | referee | `start` (a `[topic]` prefix is OK) → the match goes live |

## Safety layers (independent of the policy)

- **Robot watchdog:** `relay.py` stops the motors when `T/cmd` is stale for 0.6 s, so a dead laptop or dropped Wi-Fi means the robot stops.
- **E-stop:** the dashboard button or the Space key, at the top of the behavior tree.
- **Stop method:** every stop uses `movement_stop()`. Note that `lelib.doubleMotor.stop()` only stops the **left** motor, because it calls `motor_stop()` with the default `motor=0`.
- **Brain errors:** if the brain's tick crashes, it publishes an explicit stop and keeps the UI alive.
- **Relay robustness:** a mid-match reconnect never blocks driving: drive commands are non-blocking, the color sensor only reconnects while the robot is idle, junk MQTT payloads on our public topics are ignored, and stopping the App Lab app (SIGTERM) runs `movement_stop()` + disconnect.

## How this meets the assignment

The write-up (policy, sensors, teleoperation, strengths and weaknesses) is on our Notion page. This is the checklist against the requirements:

| Requirement | How we meet it |
|---|---|
| Built with LEGO hardware | LEGO Education double motor (tank drive) + LEGO color sensor. |
| UNO Q as some kind of sensor, over MQTT | The UNO Q reads the fan's tach wire and publishes `fan_rpm` on `T/unoq`. It also runs `relay.py`, which publishes the color sensor's readings on `T/sensor` and robot health on `T/status`. |
| ≤ 1 kg, fits in a 30 cm cube | TODO: weigh and measure the final build (with the fan and UNO Q on). |
| One of the four decision-making policies | A **behavior tree** (`behavior_tree.py`), ticked at 15 Hz. |
| At least two sensors in the decision inputs | Overhead phone camera (field, us, opponent) + rear color sensor (tape). Both feed the tree's conditions. |
| At least one autonomous sensor | Both the camera and the color sensor are read with no human involved. |
| At least one sensor via teleoperation | The dashboard: E-stop, start/pause, WASD, click-to-target on the camera view, strategy slider. |
| Shared control based on sensor fusion | A clicked target or the strategy slider tells the robot *what* to do. The camera + color-sensor fusion still decides *when it's safe*: "Escape edge" outranks "Go to target" and "Attack" in the tree, and the slider moves the edge limit the fused check uses. |
| Pushed out → stop and play a sad song | "Pushed out → LOSE" needs the camera **and** the color sensor to agree, then stops and plays the sad song (laptop mp3 + motor beeps). |
