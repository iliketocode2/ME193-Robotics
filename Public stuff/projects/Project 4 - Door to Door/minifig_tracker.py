"""
minifig_tracker.py -- find LEGO minifigs with YOLO and publish their
position + a drive command over MQTT for the Arduino UNO Q.

Pipeline, every camera frame:
    webcam (camlib.pick_camera) -> YOLO (models/minifig_yolo.pt)
      -> best detection per color -> policy -> JSON over MQTT

One topic per minifig color:
    ME193/Will-Courtland/green
    ME193/Will-Courtland/blue

Each topic gets its own independent policy state, so the green and blue
cars can be driven at the same time once the model knows both classes.
A color the model wasn't trained on yet (blue, for now) still gets
"seen":0 / stop heartbeats, so the Arduino side can be written and
tested against it today.

Message (JSON, integers only so the Arduino doesn't need float parsing),
published PUBLISH_HZ times per second whether or not anything changed:

    {"seen":1,"x":712,"y":455,"col":9,"row":3,"err":212,
     "speed":-55,"cmd":"backward","conf":87}

See README.md in this folder for what every field means.

Keys: q / Esc (or closing the window) = quit -- sends a final stop on every topic first.
The dashboard drawing lives in tracker_ui.py.

Run:
    my_env\\Scripts\\python "Public stuff/projects/Project 4 - Door to Door/minifig_tracker.py"
"""

import json
import os
import sys
import time

import cv2
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "useful libraries"))

from camlib import pick_camera
from mqttlib import MQTTClient
from tracker_ui import Dashboard

# ----- Model -------------------------------------------------------------
MODEL_PATH = os.path.join(REPO_ROOT, "models", "minifig_yolo.pt")  # written by train_minifig.py
IMG_SIZE = 512        # the dataset was exported as 512x512 *stretched*, so frames are
                      # stretched the same way before inference (see detect())
CONF_THRESH = 0.50    # ignore detections below this confidence

# ----- MQTT --------------------------------------------------------------
# Class names are matched by the color word they contain, so
# "green LEGO minifig" -> green topic, "blue LEGO minifig" -> blue topic.
TOPICS = {
    "green": "ME193/Will-Courtland/green",
    "blue": "ME193/Will-Courtland/blue",
}
PUBLISH_HZ = 10       # heartbeat rate -- the Arduino should stop if messages stop arriving

# ----- UNO Q LED matrix --------------------------------------------------
LED_COLS = 13         # UNO Q matrix is 13 wide x 8 tall
LED_ROWS = 8

# ----- Policy (proportional control on horizontal offset) ----------------
# Positions are in thousandths of the frame: x = 0 (left edge) .. 1000 (right edge).
CENTER = 500
DEADBAND_ENTER = 40   # |err| <= this -> "arrived", stop
DEADBAND_EXIT = 80    # once stopped, must drift past this to start again (no chatter)
KP = 0.25             # speed per unit err (err 400 -> speed 100 before clamping)
MIN_SPEED = 30        # DC motors stall below some PWM -- never command less than this when moving
MAX_SPEED = 70
# Which way does "forward" move the minifig ON SCREEN? Depends on how the
# car is placed in front of the camera. If the car drives away from center
# instead of toward it, flip this.
FORWARD_MOVES_RIGHT = True
LOST_GRACE_S = 0.25   # keep using the last detection this long through a 1-frame dropout


class ColorPolicy:
    """Per-color state: last detection, and whether we're inside the deadband."""

    def __init__(self, color, topic):
        self.color = color
        self.topic = topic
        self.last_det = None        # (x, y, conf, box): x/y in 0..1000, box = normalized (x1, y1, x2, y2)
        self.last_seen_t = 0.0
        self.arrived = False

    def update(self, det, now):
        if det is not None:
            self.last_det = det
            self.last_seen_t = now
        elif now - self.last_seen_t > LOST_GRACE_S:
            self.last_det = None

    def message(self):
        if self.last_det is None:
            # Nothing detected: the policy is simply "stop and wait".
            self.arrived = False
            return {"seen": 0, "x": -1, "y": -1, "col": -1, "row": -1, "err": 0,
                    "speed": 0, "cmd": "stop", "conf": 0}

        x, y, conf = self.last_det[:3]
        err = x - CENTER                        # + means minifig is right of center

        deadband = DEADBAND_EXIT if self.arrived else DEADBAND_ENTER
        if abs(err) <= deadband:
            self.arrived = True
            speed = 0
        else:
            self.arrived = False
            # To bring a right-of-center minifig back to center it must move left.
            direction = -1 if FORWARD_MOVES_RIGHT else 1
            magnitude = min(MAX_SPEED, max(MIN_SPEED, KP * abs(err)))
            speed = int(round(direction * (1 if err > 0 else -1) * magnitude))

        cmd = "stop" if speed == 0 else ("forward" if speed > 0 else "backward")
        return {
            "seen": 1,
            "x": x,
            "y": y,
            "col": min(LED_COLS - 1, x * LED_COLS // 1000),
            "row": min(LED_ROWS - 1, y * LED_ROWS // 1000),
            "err": err,
            "speed": speed,
            "cmd": cmd,
            "conf": int(round(conf * 100)),
        }


def color_of(class_name):
    """'green LEGO minifig' -> 'green'; None if no known color word."""
    name = class_name.lower()
    for color in TOPICS:
        if color in name:
            return color
    return None


def detect(model, frame, class_colors):
    """Run YOLO on one frame; return {color: (x, y, conf, box)} with the
    single highest-confidence box per color. x/y = box center in 0..1000;
    box = (x1, y1, x2, y2) normalized 0..1, used only for drawing."""
    # Stretch (not letterbox) to 512x512, exactly like the Roboflow export,
    # so the minifig has the same aspect ratio it had during training.
    # Normalized box coordinates map straight back to the original frame.
    small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    result = model.predict(small, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]

    best = {}
    for (cx, cy, _, _), box, cls, conf in zip(result.boxes.xywhn.tolist(),
                                              result.boxes.xyxyn.tolist(),
                                              result.boxes.cls.tolist(),
                                              result.boxes.conf.tolist()):
        color = class_colors.get(int(cls))
        if color is None:
            continue
        if color not in best or conf > best[color][2]:
            best[color] = (int(round(cx * 1000)), int(round(cy * 1000)), conf, tuple(box))
    return best


WINDOW = "Minifig Tracker"


def open_window(ui):
    """Resizable window, initially scaled down if the dashboard is taller
    than the screen (common on laptops with 125-150% display scaling)."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    scale = 1.0
    try:
        import ctypes
        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        scale = min(1.0, 0.95 * screen_w / ui.width, 0.85 * screen_h / ui.height)
    except (AttributeError, OSError):
        pass   # not Windows -- just open at full size
    cv2.resizeWindow(WINDOW, int(ui.width * scale), int(ui.height * scale))


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"No model at {MODEL_PATH} -- run train_minifig.py first.")
    model = YOLO(MODEL_PATH)

    class_colors = {}
    for idx, name in model.names.items():
        color = color_of(name)
        if color is None:
            print(f"Warning: class '{name}' has no color word from {list(TOPICS)} -- ignored.")
        else:
            class_colors[idx] = color
    for color in TOPICS:
        if color not in class_colors.values():
            print(f"Note: model has no '{color}' class yet -- {TOPICS[color]} will only send seen=0/stop.")

    policies = [ColorPolicy(c, t) for c, t in TOPICS.items()]
    trained = set(class_colors.values())
    cap, _ = pick_camera()

    ui = Dashboard(center=CENTER, deadband=DEADBAND_ENTER, led_cols=LED_COLS, led_rows=LED_ROWS,
                   broker="test.mosquitto.org", publish_hz=PUBLISH_HZ,
                   model_name=os.path.basename(MODEL_PATH))
    open_window(ui)
    fps, infer_ms, sent = 0.0, 0.0, 0
    payloads = {p.color: "" for p in policies}

    with MQTTClient() as client:
        try:
            last_pub = 0.0
            last_frame_t = time.time()
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed.")
                    break

                now = time.time()
                dets = detect(model, frame, class_colors)
                done = time.time()
                infer_ms = 0.9 * infer_ms + 0.1 * (done - now) * 1000      # smoothed for display
                fps = 0.9 * fps + 0.1 / max(1e-6, done - last_frame_t)
                last_frame_t = done
                for p in policies:
                    p.update(dets.get(p.color), now)
                messages = {p.color: p.message() for p in policies}

                if now - last_pub >= 1.0 / PUBLISH_HZ:
                    last_pub = now
                    for p in policies:
                        payloads[p.color] = json.dumps(messages[p.color], separators=(",", ":"))
                        client.publish(p.topic, payloads[p.color])
                        sent += 1

                entries = [{"color": p.color, "topic": p.topic, "msg": messages[p.color],
                            "payload": payloads[p.color], "trained": p.color in trained,
                            "box": p.last_det[3] if p.last_det else None} for p in policies]
                cv2.imshow(WINDOW, ui.render(frame, entries,
                                             {"fps": fps, "infer_ms": infer_ms, "sent": sent}))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):   # q or Esc
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # window closed with X
                    break
        finally:
            # Always leave the cars stopped, even on a crash or Ctrl+C.
            stop = {"seen": 0, "x": -1, "y": -1, "col": -1, "row": -1, "err": 0,
                    "speed": 0, "cmd": "stop", "conf": 0}
            for p in policies:
                client.publish(p.topic, json.dumps(stop, separators=(",", ":")))
            time.sleep(0.2)   # let the final publishes flush before disconnecting
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
