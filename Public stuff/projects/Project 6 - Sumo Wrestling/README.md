# Project 6 – Sumo Wrestling

## Starting everything (match day)

The system runs in three places, so there is no single command. Start them in this order:

1. **Phone:** open the IP-camera app and start streaming.
2. **UNO Q, in two parts:**
   - In **App Lab**, press **Run** on the sumo app. This part only reports the fan's speed (`RELAY_BLE = False`).
   - In the **board terminal** (App Lab's terminal or SSH), start the relay, which talks to the motor and color sensor:
     ```
     cd ~/ArduinoApps/<your app>/python && bash run_on_host.sh
     ```
     Our app folder is `ble-probe`. Wait for `[relay] double motor connected` and `[relay] color sensor connected`, then leave this terminal open.
3. **Laptop:** in PowerShell from the repo root, start the brain:
   ```
   my_env\Scripts\python "Public stuff/projects/Project 6 - Sumo Wrestling/sumo_brain.py"
   ```
   The dashboard opens at http://localhost:8293. Check that the status pills are green before the match.

**To stop:** press Ctrl-C in the board terminal and on the laptop. The robot also stops on its own within 0.6 s whenever the laptop stops sending commands.

The relay runs on the board's host, not inside App Lab, because App Lab's container can't reach Bluetooth (see "Setup and run order", step 3).

---

Two robots in a ring of blue tape. The first one pushed fully out loses and has to stop and play a sad song.

Our robot is a LEGO Education **double motor**. A LEGO **color sensor** is mounted on its rear, looking down. An **Arduino UNO Q** rides on top. An **overhead phone camera** streams to the laptop, which runs the vision and a **behavior tree**. A web dashboard shows everything and lets us step in.

```
 phone (IP camera app) ──MJPEG──▶ laptop: sumo_brain.py ──────────────▶ web dashboard
                                   │ vision.py: YOLO robots + AprilTag #6 (us) + blue ring
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
| `vision.py`, `field.py` | Ring ellipse fit (HSV), AprilTag pose (position + heading), YOLO / tag-only opponent detection, and ring-normalized coordinates. |
| `config.py` | Laptop settings: phone URL, tag id, HSV defaults, ports, timing. |
| `unoq_app/` | UNO Q App Lab app: `python/main.py`, `python/relay.py` (MQTT↔BLE), `python/sumo_topics.py` (**shared MQTT schema + card IDs**), `sketch/sketch.ino` (fan RPM from the tach wire). |
| `unoq_app/python/ble_probe_unoq.py` | Step 1: checks whether the UNO Q can reach the LEGO devices over Bluetooth. |
| `ble_relay_laptop.py` | Fallback: runs the same relay on the laptop if the UNO Q can't do BLE. |
| `songs_bridge.py` | Win/lose songs: Project 3's mp3s on the laptop, plus the same note lists beeped on the double motor. |
| `make_tag.py` → `tags/` | Our AprilTag (tag36h11 **#6**). The `_print.png` version prints at exactly 7 cm. |
| `capture_frames.py`, `train_sumo.py` | Collect frames → label in Roboflow → train the local YOLO model. |
| `test_sumo_offline.py` | No-hardware test of vision and every behavior-tree branch, using synthetic frames. |
| `web/` | Dashboard (plain HTML/CSS/JS, no build step). |

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
   - Open **Calibrate** and tune the HSV sliders until the blue ellipse hugs the tape.
6. **Vision model (optional but better).** Until it exists, the brain runs in **tag-only mode**: the opponent is the biggest non-floor blob near the ring.
   - Run `capture_frames.py`.
   - Upload the frames from `runs/sumo_frames/` to Roboflow and label one class, `robot`, on every robot including ours.
   - Export as **YOLOv8** and unzip the export into this folder. Set `DATASET_DIR` in `train_sumo.py` and run it.
   - The weights go to `<repo>/models/sumo_yolo.pt`, which is gitignored and picked up on the next start.

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

## Draft answers for the Notion page

**The policy.** It is a **behavior tree**, one of the four algorithms from class. The root is a priority *selector* ticked 15 times a second, and each child is a *sequence* of a condition and an action. The first child whose condition holds drives the motors that tick. In priority order:
1. E-stop
2. Manual drive
3. Robot link down
4. Wait for start
5. Pushed out → stop and play the sad song
6. Opponent out → play the win song
7. Escape edge
8. Go to the clicked target
9. Attack (align, then charge)
10. Guard when the camera is lost
11. Patrol the center

Because higher branches pre-empt lower ones, staying in the ring always beats attacking. The dashboard highlights which branch is running.

**Autonomous sensors.**
- **Overhead phone camera.** It finds the ring (blue HSV mask + ellipse fit), our robot's position and heading (AprilTag #6), and the opponent (YOLO model trained from Roboflow labels). Everything is converted to "ring coordinates", where the tape is radius 1.
- **LEGO color sensor** on the rear, facing the floor. It sees the tape directly.

These are fused in two ways. "Escape edge" fires on *either* the camera (too far out) or the sensor (blue under the rear). "We lost" needs the camera to say we're out *and* the sensor to have crossed the tape (or the camera to be very sure), so one bad reading can't end the match. With no camera, the sensor still guards the back edge.

**Teleoperation / shared control** from the dashboard:
- E-stop
- Manual start/pause/reset, in case the referee's `start` is missed
- WASD manual driving, which overrides the policy while held
- Click-to-target: tell the robot where to go, for example to flank
- A strategy slider from defensive to aggressive, which changes charge speed, how close to the edge we're allowed to go, and the alignment tolerance
- Dry-run mode
- Live HSV calibration
- A chat box on our MQTT channel

**Strengths and weaknesses.**
- *Strengths:*
  - The camera gives the robot a full-field view, so it always knows where the center and the opponent are.
  - It keeps pushing even near the edge when the opponent is between it and the tape.
  - The rear sensor catches being shoved backward, the most common way to lose, even with no camera.
  - Watchdogs make every failure stop the robot.
  - The fan adds some downforce for traction, and its RPM is reported live so we know it's spinning.
- *Weaknesses:*
  - There is only one color sensor, on the back, so the front and sides are blind to the tape without the camera.
  - The loop (camera → laptop → MQTT → UNO Q → BLE) adds roughly 150–300 ms of latency. Fast opponents get a head start.
  - The phone has to stay still. The ring fit locks at match start, so if the phone is bumped after that, the ring has to be re-fit.
  - Tag-only mode can mistake shadows or clutter for the opponent until the YOLO model is trained.
  - The robot is a light double-motor chassis (limit 1 kg), so a heavier opponent may out-push it head-on.
