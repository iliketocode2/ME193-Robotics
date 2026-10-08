# Project 4 — Door to Door (YOLO minifig → MQTT → Arduino UNO Q)

A laptop webcam finds a green LEGO minifig with a YOLO object detector
that we trained ourselves. It publishes the minifig's position and a
drive command over MQTT. The Arduino UNO Q then:
1. lights a dot on its LED matrix at the minifig's scaled position, and
2. drives its DC motors forward or backward until the minifig is in the
   middle of the screen.

The blue minifig uses the same pipeline on its own topic.

## Files

| File | What it does |
|---|---|
| `Green LEGO Minifig Detection.v1-…yolov8/` | Dataset: 37 webcam photos (26 train / 7 valid / 4 test), boxes drawn in Roboflow, exported as YOLOv8 at 512×512. |
| `train_minifig.py` | Trains the model **locally** with Ultralytics. It fine-tunes COCO-pretrained `yolo11n` on the dataset, prints test-split precision/recall/mAP, and saves `<repo>/models/minifig_yolo.pt`. It uses the Intel Arc GPU automatically when the `+xpu` PyTorch build is installed (about 8 s/epoch, versus 35–120 s/epoch on the CPU, which throttles under sustained load), and falls back to the CPU otherwise. `DATASETS` can list several Roboflow exports (e.g. a green one and a blue one): they're merged into one dataset with classes matched by name, every label is checked, and box counts per class and split are printed before training. Final val/test scores are computed on the CPU and broken down per class. |
| `minifig_tracker.py` | Live loop: webcam → YOLO → policy → MQTT. Quit with `q`, `Esc`, or by closing the window; any of these sends a final stop first. |
| `tracker_ui.py` | The dashboard window (drawing only, no logic). It shows the live feed with detection boxes, the stop zone and direction arrows; a position track; a status card per color (status, error, confidence, motor command); a preview of the UNO Q LED matrix; and the exact MQTT payloads being sent. |

Roboflow was used only for labeling and export. Training, inference, and the control policy are all in this repo's Python.

## Setup / run

```bash
my_env\Scripts\python -m pip install ultralytics      # pulls in torch (CPU build)
# Optional, much faster training on an Intel Arc laptop GPU (replaces the CPU torch build):
my_env\Scripts\python -m pip install torch==2.14.1+xpu torchvision==0.29.1+xpu --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
my_env\Scripts\python "Public stuff/projects/Project 4 - Door to Door/train_minifig.py"
my_env\Scripts\python "Public stuff/projects/Project 4 - Door to Door/minifig_tracker.py"
```

`runs/` (training plots) and `models/` (weights) sit at the repo root,
outside `Public stuff/`. That keeps them gitignored on purpose.

## MQTT messages

- **Broker:** `test.mosquitto.org`, port `1883` (the `mqttlib.py` default)
- **Topics:** `ME193/Will-Courtland/green` and `ME193/Will-Courtland/blue`
- **Rate:** 10 messages/s on **each** topic, all the time, even when
  nothing is detected.
- **Payload:** compact JSON. Every value is an integer except `cmd`.

Detected:
```json
{"seen":1,"x":712,"y":455,"col":9,"row":3,"err":212,"speed":-55,"cmd":"backward","conf":87}
```
Not detected (and also the final message sent when the script quits):
```json
{"seen":0,"x":-1,"y":-1,"col":-1,"row":-1,"err":0,"speed":0,"cmd":"stop","conf":0}
```

| Field | Range | Meaning |
|---|---|---|
| `seen` | 0 / 1 | 1 if this color's minifig is currently detected |
| `x`, `y` | 0–1000 (−1 if not seen) | Box center in thousandths of the frame width/height. (0,0) is top-left. The image is **not** mirrored. |
| `col` | 0–12 (−1) | Which LED column to light: `x * 13 / 1000` |
| `row` | 0–7 (−1) | Which LED row to light: `y * 8 / 1000` |
| `err` | −500…500 | `x − 500`. Positive means the minifig is right of center. |
| `speed` | −70…70 | Suggested signed motor speed (percent). 0 means stop. Positive means forward. |
| `cmd` | `"forward"` / `"backward"` / `"stop"` | The sign of `speed`, as a word |
| `conf` | 0–100 | YOLO confidence, in percent |

**Arduino side, suggested approach:**
- Parse the payload with ArduinoJson.
- If `seen==1`, light pixel (`col`, `row`). If `seen==0`, clear the matrix.
- Drive the motors with `speed`, mapped to PWM, with its sign setting direction.
- **Add a watchdog:** stop the motors if no message has arrived for about 0.5 s. That covers the laptop crashing or Wi-Fi dropping, since then no "stop" message ever comes.
- If you'd rather run your own controller on the UNO Q, ignore `speed`/`cmd` and use `err` directly.

## The policy (how it decides)

This is proportional control on the horizontal offset, with a deadband and hysteresis:

1. YOLO returns boxes. For each color, keep the single highest-confidence box at ≥ 0.50 confidence.
2. Compute `err = x − 500`, the distance from the screen's vertical center line.
3. If `|err| ≤ 40` (within 4% of center), the minifig has **arrived** and `speed = 0`.
   Once arrived, it only starts moving again if `|err| > 80`. Without
   that gap, detection jitter right at the edge of the band would make
   the car twitch between stop and go.
4. Otherwise, `speed = ±clamp(0.25·|err|, 30, 70)`:
   - It goes faster when the minifig is far from center and slows down as it closes in.
   - Speed never drops below 30, because DC motors stall at low PWM.
   - The sign is chosen to move the minifig *toward* the center.
     `FORWARD_MOVES_RIGHT` in `minifig_tracker.py` sets which way
     "forward" moves the minifig on screen. Flip it if the car runs away
     from center.

Green and blue each keep their own policy state, so both cars can run at once.

## What happens when no minifig is detected

- If the detection disappears for only one or two frames (under 0.25 s),
  the last position is reused, so normal YOLO flicker doesn't jerk the car.
- After 0.25 s with no detection, the script sends `seen:0`, `speed:0`, `cmd:"stop"`.
  The car **stops and waits**. It does not search or coast on its last command.
- On quit (`q`, Ctrl+C, or a crash), it sends a final stop on both topics.
- Until the model has a blue class, the blue topic only ever sends the `seen:0` stop message.

## How good is the model? (fill in after training/testing)

- **Test-split metrics** (from `train_minifig.py` output): precision __, recall __, mAP@50 __, mAP@50-95 __
- **Caveat:** the test split is only 4 images, so these numbers are a rough sign, not a measurement.
- **Things to try confusing it with:**
  - the blue minifig, and other colors
  - green non-minifig objects (a green LEGO brick, a green cup)
  - a minifig partly hidden by fingers
  - a different room or different lighting
  - a minifig much farther from the camera than in the dataset
- **Likely weakness:** with only one class trained, "green minifig" may
  partly mean "green blob" or "anything minifig-shaped." Adding the blue
  class (and other-color minifigs as unlabeled negatives) forces the
  model to use both color and shape.
