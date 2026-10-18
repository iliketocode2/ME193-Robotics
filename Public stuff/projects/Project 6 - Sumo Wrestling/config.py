"""
config.py -- laptop-side settings for the sumo brain.

The MQTT schema and robot card IDs are NOT here: they live in
unoq_app/python/sumo_topics.py, the one file both the robot and the laptop
import, and are re-exported below so laptop code can just `from config import ...`.
"""

import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))

from sumo_topics import *  # noqa: E402,F401,F403  (TEAM, TOPIC_*, BROKER_*, songs, timeouts)

# ---- camera ----------------------------------------------------------------
# Phone IP-camera app (same as Project 2: e.g. "SimpleIPCamera" MJPEG stream).
# TODO: put your phone's URL here. A plain integer (0, 1, ...) uses a local
# webcam instead; None asks with camlib.pick_camera() at startup.
PHONE_STREAM_URL = "http://192.168.1.183:8080/stream.mjpg"
FRAME_ROTATION = None            # e.g. cv2.ROTATE_90_CLOCKWISE if the phone is mounted sideways
PROCESS_WIDTH = 960              # frames are resized to this width before vision
# True if the phone app mirrors the picture (text on the tag reads backwards).
# A mirrored image flips left/right turns, so the brain would steer the wrong way.
CAMERA_MIRRORED = False

# ---- AprilTag on top of OUR robot --------------------------------------------
APRILTAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11
OUR_TAG_ID = 6                   # "Project 6"; printed by make_tag.py
TAG_PRINT_CM = 7.0               # printed black-square size (white border is extra)
# The tag's "up" (top edge in the printed image) points to the robot's FRONT
# when mounted as the README shows. If you glued it rotated, fix it here.
TAG_HEADING_OFFSET_DEG = 0.0

# ---- YOLO model (Roboflow-labelled, trained locally by train_sumo.py) --------
MODEL_PATH = os.path.join(REPO_ROOT, "models", "sumo_yolo.pt")   # gitignored
YOLO_IMG_SIZE = 512
YOLO_CONF = 0.45

# ---- ring detection (blue tape) -----------------------------------------------
# OpenCV HSV: H 0-179. Blue painter's/electrical tape is usually ~H 95-130.
# Tune live from the UI's "Calibrate" panel; values there are not saved here.
HSV_LOW = (95, 90, 50)
HSV_HIGH = (130, 255, 255)

# ---- web UI -----------------------------------------------------------------
HTTP_PORT = 8293
WS_PORT = HTTP_PORT + 1          # web/app.js assumes page port + 1
VIDEO_JPEG_QUALITY = 70
VIDEO_MAX_FPS = 15

# ---- control loop -------------------------------------------------------------
TICK_HZ = 15                     # behavior tree tick + T/cmd publish rate
CAMERA_STALE_S = 0.6             # no fresh vision for this long -> "camera lost"
ROBOT_STALE_S = 2.5              # no T/status for this long -> robot link down
SENSOR_STALE_S = 0.8             # no T/sensor for this long -> ignore color sensor
UNOQ_STALE_S = 3.0               # no T/unoq for this long -> fan RPM unknown
