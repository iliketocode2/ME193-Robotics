"""Finger-count lock: the Linux side of the UNO Q.

Reads the USB webcam, runs FingerNet in numpy (npnet.py), and feeds the readings and button presses to
lock_policy.py, the same files the computer version uses. It sends the LED pattern and the latch position to
the sketch over the Bridge, and the sketch does the motor and switch timing.

config.py, camera.py, npnet.py, lock_policy.py and models/finger_cnn.npz are copies made by uno_q/sync_app.py.
Edit the originals in the project folder and run that script again, never these copies.

On the computer this file also runs without the board (python main.py inside this folder). It prints what it
would send, which checks that the camera and the network work before uploading.
"""

import glob
import queue
import re
import sys
import time
from collections import deque

import cv2
import numpy as np

import camera
import config
import npnet
import viewer

try:   # only OpenCV errors in the log, not a warning for every /dev/video node that isn't a camera
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass
from lock_policy import LockPolicy

try:
    from arduino.app_utils import App, Bridge
    ON_BOARD = True
except ImportError:
    ON_BOARD = False

# None keeps the latch open until the button is pressed. On the box the lid can't close while the arm points
# at the catch, so locking by itself after a few seconds would block a lid that is still up. Set a number of
# seconds to lock by itself, like lock.py on the computer.
RELOCK_AFTER_S = None
SMOOTH = 5            # average the probabilities over this many frames, as lock.py does
KEEPALIVE_S = 1.0     # resend the LEDs this often so the sketch knows this side is running
PRINT_EVERY_S = 2.0   # how often to print the current reading to the log
CAMERA_RETRY_S = 5.0  # how often to look for a camera when there is none


# Camera ---------------------------------------------------------------------------------------------------------

class OpenCvCamera:
    def __init__(self, cap, name):
        self.cap, self.name = cap, name

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None


class ArduinoCamera:
    """App Lab's own camera class, used if OpenCV can't open /dev/video* inside the app."""

    def __init__(self):
        from arduino.app_peripherals.camera import Camera
        self.cam = Camera(resolution=(config.FRAME_W, config.FRAME_H))
        self.cam.start()
        self.name = "arduino.app_peripherals.camera"

    def read(self):
        img = self.cam.capture()
        if img is None:
            return None
        frame = np.asarray(img)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame


def try_opencv(index):
    # Open by number: this OpenCV build can't open "/dev/videoN" by name.
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    for _ in range(5):              # the first frames can be empty while the camera wakes up
        ok, frame = cap.read()
        if ok and frame is not None and frame.ndim == 3:
            return OpenCvCamera(cap, f"OpenCV camera {index}")
    cap.release()
    return None


def open_camera():
    """The first camera that gives a color frame, or None. The UNO Q also has /dev/video nodes for its video
    decoder, and the webcam's number can change after a reboot, so try them all."""
    numbers = sorted({int(n) for n in (re.sub(r"\D", "", p) for p in glob.glob("/dev/video*")) if n})
    for index in numbers or [config.CAMERA_INDEX, 0, 1, 2]:
        cam = try_opencv(index)
        if cam:
            return cam
    if ON_BOARD:
        try:
            return ArduinoCamera()
        except Exception as e:
            if not str(e) == last_error[0]:
                last_error[0] = str(e)
                print("no camera:", e, flush=True)
    return None


# Board link -----------------------------------------------------------------------------------------------------

presses = queue.Queue()   # Bridge callbacks run on their own thread, so they only queue things up
printed = {}
latch_state = [None]      # last report from the sketch: 0 locked, 1 open, 2 fault


inputs = {"button": None, "locked": None, "open": None}   # raw states from the sketch, for the live view
press_count = [None]
status_error = [""]


def poll_status():
    """Ask the sketch for the button, the switches and the latch. A change in its press counter is a press."""
    if not ON_BOARD:
        return
    try:
        status = Bridge.call("get_status")
    except Exception as e:
        if str(e) != status_error[0]:
            status_error[0] = str(e)
            print("get_status failed:", e, flush=True)
        return
    if status is None:
        if status_error[0] != "None":
            status_error[0] = "None"
            print("get_status returned nothing; is the new sketch on the board?", flush=True)
        return
    status = int(status)
    inputs.update(button=bool(status & 1), locked=bool(status & 2), open=bool(status & 4))
    latch_state[0] = (status >> 3) & 3
    count = (status >> 8) & 0xFF
    if press_count[0] is not None:
        for _ in range((count - press_count[0]) % 256):
            print("button pressed", flush=True)
            presses.put(time.monotonic())
    press_count[0] = count


def on_latch(state):
    latch_state[0] = state
    print("latch:", {0: "locked", 1: "open", 2: "FAULT, a limit switch did not click", 3: "moving"}.get(state, state),
          flush=True)


def send(name, value):
    if not ON_BOARD:
        if printed.get(name) != value:      # skip the keepalive repeats
            printed[name] = value
            print(f"[no board] {name}({value})", flush=True)
        return
    try:
        Bridge.call(name, value)
    except Exception as e:          # the sketch may still be starting
        print(f"Bridge.call({name!r}) failed: {e}", flush=True)


def led_mask(leds):
    mask = sum(1 << i for i, lit in enumerate(leds["digits"]) if lit)
    if leds["status"] == "green":
        mask |= 1 << 3
    elif leds["status"] == "red":
        mask |= 1 << 4
    return mask


# Main loop ------------------------------------------------------------------------------------------------------

weights = npnet.load_weights(config.NUMPY_WEIGHTS_PATH)
cam = None                # opened in loop(), so the LEDs and the live view work even with no camera
cam_tried = [-1e9]
bad_reads = [0]
last_error = [""]
policy = LockPolicy(open_s=float("inf") if RELOCK_AFTER_S is None else RELOCK_AFTER_S)
recent = deque(maxlen=SMOOTH)
sent = {"mask": None, "latch": None, "at": 0.0}
last_print = [0.0]
live = viewer.Viewer()
fps = [0.0, time.monotonic()]   # smoothed frames per second, time of the last frame


def no_camera(now, why):
    """Keep the board going without a picture: LEDs off, the live view says what is wrong."""
    if live.watching:
        live.publish(viewer.message_frame(why))
    if now - sent["at"] > KEEPALIVE_S:
        sent["mask"], sent["at"] = 0, now
        send("set_leds", 0)
    time.sleep(0.2)


def loop():
    global cam
    now = time.monotonic()
    if cam is None:
        if now - cam_tried[0] < CAMERA_RETRY_S:
            return no_camera(now, "No camera. Plug the webcam into the hub.")
        cam_tried[0] = now
        cam = open_camera()
        if cam is None:
            return no_camera(now, "No camera. Plug the webcam into the hub.")
        print("camera:", cam.name, flush=True)

    frame = cam.read()
    if frame is None:
        bad_reads[0] += 1
        if bad_reads[0] > 20:       # unplugged: drop it and look again
            print("camera stopped giving frames", flush=True)
            cam, bad_reads[0] = None, 0
        time.sleep(0.05)
        return
    bad_reads[0] = 0
    now = time.monotonic()

    poll_status()
    events = []
    while not presses.empty():
        presses.get()
        recent.clear()              # start the attempt from fresh readings
        events += policy.press(now)

    inp = camera.frame_to_input(frame)
    recent.append(npnet.forward(weights, inp))
    avg = np.mean(recent, axis=0)
    count, conf = int(np.argmax(avg)), float(avg.max())
    events += policy.update(count, conf, now)

    dt = now - fps[1]
    fps[0], fps[1] = 0.9 * fps[0] + 0.1 / max(dt, 1e-3), now
    if live.watching:
        live.publish(viewer.draw(frame, inp, policy, count, conf, now, latch_state[0], fps[0], inputs))

    for event in events:
        print(f"{event:8s} state={policy.state} digits={len(policy.entered)} fails={policy.fails}", flush=True)
    if now - last_print[0] > PRINT_EVERY_S:
        last_print[0] = now
        print(f"reading {count} ({conf:.0%})  {policy.state}: {policy.message}", flush=True)

    latch = 1 if policy.unlocked else 0
    if latch != sent["latch"]:
        sent["latch"] = latch
        send("set_latch", latch)
    mask = led_mask(policy.leds(now))
    if mask != sent["mask"] or now - sent["at"] > KEEPALIVE_S:
        sent["mask"], sent["at"] = mask, now
        send("set_leds", mask)


if ON_BOARD:
    Bridge.provide("latch", on_latch)
    App.run(user_loop=loop)
else:
    print("Running without the board. Ctrl+C to stop.")
    while True:
        loop()
