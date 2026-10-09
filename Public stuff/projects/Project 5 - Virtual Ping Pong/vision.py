"""
Camera side of the game, run on its own thread so slow frames never stall
the 60 Hz game tick:

  * MediaPipe Pose (Tasks API, like Project 1's hand landmarker) -> where
    your paddle hand is, mapped into table coordinates, plus a person
    segmentation mask used to cut you out of the webcam image.
  * AprilTags (cv2.aruco tag36h11, like Project 2) -> which opponent card
    you're holding up on the select screen.

Everything the rest of the program needs is published as one snapshot
(VisionState) under a lock.
"""

import os
import threading
import time
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from opponents import TAG_TO_OPPONENT

# --- Player ------------------------------------------------------------------
RIGHT_HANDED = True          # which wrist holds the Double Motor paddle

# --- Pose -> table mapping (meters) --------------------------------------------
# Your paddle x = body position + hand offset from your shoulders. The hand
# offset is measured in shoulder-widths, so it doesn't matter how far from
# the camera you stand. SHOULDER_WORLD is also how wide your shoulders are
# drawn in the 3D scene, so the virtual paddle lines up with your real hand
# in the cut-out video.
BODY_X_GAIN = 1.6            # walk across the whole frame -> +/-0.8 m
SHOULDER_WORLD = 0.36        # one shoulder-width of hand movement = this many meters
SHOULDER_Y = 0.45            # your shoulders' height above the table top
PADDLE_X_LIMIT = 1.0
PADDLE_Y_RANGE = (-0.1, 0.95)
SMOOTHING = 0.75             # EMA weight on the newest sample (1 = no smoothing)
MIN_VISIBILITY = 0.5

# --- AprilTag ------------------------------------------------------------------
APRILTAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11
TAG_HOLD_S = 0.5             # tag must be steady this long before it selects

# --- Video stream to the browser ---------------------------------------------
STREAM_SIZE = (320, 240)
STREAM_QUALITY = 70
POSE_INPUT = (320, 240)      # pose runs on a half-size copy: ~10% faster, same landmarks
MASK_SIZE = (160, 120)       # person mask sent raw (no encoding); the browser's GPU upsamples it

# MediaPipe pose model, stored in the gitignored repo-root models/ folder
# (same pattern as Project 1's hand_landmarker.task).
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models",
                           "pose_landmarker_lite.task")

# Pose landmark indices (MediaPipe's 33-point body model)
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16


def ensure_model():
    """Download the pretrained pose model on first run."""
    path = os.path.abspath(_MODEL_PATH)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading MediaPipe pose landmark model...")
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


@dataclass
class VisionState:
    paddle_x: float = 0.0
    paddle_y: float = 0.3
    paddle_vx: float = 0.0
    visible: bool = False
    # for lining the cut-out video up in 3D (mirrored, normalized image coords)
    shoulder_u: float = 0.5
    shoulder_v: float = 0.4
    shoulder_w: float = 0.25
    tag_opponent: str | None = None
    fps: float = 0.0
    frame_t: float = 0.0


def pose_to_table(lm, aspect):
    """Pose landmarks (already mirrored) -> (paddle_x, paddle_y, shoulder_u,
    shoulder_v, shoulder_w) or None if the needed points aren't visible.
    `aspect` = frame width / height, so x and y offsets share units."""
    wrist = lm[R_WRIST if RIGHT_HANDED else L_WRIST]
    ls, rs = lm[L_SHOULDER], lm[R_SHOULDER]
    if min(wrist.visibility, ls.visibility, rs.visibility) < MIN_VISIBILITY:
        return None
    su, sv = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
    sw = abs(ls.x - rs.x)
    if sw < 0.03:
        return None  # side-on to the camera -- can't measure shoulder width
    rel_x = (wrist.x - su) / sw
    rel_y = (sv - wrist.y) / (sw * aspect)       # + = hand above shoulders
    px = (su - 0.5) * BODY_X_GAIN + rel_x * SHOULDER_WORLD
    py = SHOULDER_Y + rel_y * SHOULDER_WORLD
    px = max(-PADDLE_X_LIMIT, min(PADDLE_X_LIMIT, px))
    py = max(PADDLE_Y_RANGE[0], min(PADDLE_Y_RANGE[1], py))
    return px, py, su, sv, sw


class _Mirrored:
    """A landmark with x flipped, so 'your right' is on screen-right like a mirror."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, l):
        self.x, self.y, self.visibility = 1.0 - l.x, l.y, l.visibility


class Vision:
    """Owns the camera. Two threads:

      grab   -- reads the camera as fast as it delivers and hands the newest
                frame to the vision thread. It also encodes every frame for
                the select screen's camera inset, so that runs at full
                camera rate.
      vision -- pose + AprilTag on the newest frame only (anything that
                arrives meanwhile replaces it, never queues -- a queue of old
                frames was the original lag). For gameplay it sends THAT frame
                plus its person mask, so the cut-out edges and the virtual
                paddle line up exactly with your body.

    read() returns the latest VisionState; latest_images() returns
    (video_id, video_jpeg, cut_id, cut_jpeg, mask_bytes)."""

    def __init__(self, cap, start_ms):
        self.cap = cap
        self.start_ms = start_ms
        self.want_tags = True           # main loop turns this on only on the select screen
        self._state = VisionState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._new_frame = threading.Condition()
        self._raw = None                # (frame, capture_time) -- newest only
        self._frame_jpeg = None         # every camera frame, mirrored, STREAM_SIZE (select screen)
        self._frame_id = 0
        self._cut_jpeg = None           # the frame pose last ran on (gameplay cut-out)
        self._mask = None               # its person mask, MASK_SIZE uint8, mirrored, top row first
        self._cut_id = 0
        self._tag_seen = None
        self._tag_since = 0.0
        self._tag_outlines = []         # [(mirrored corner pts, BGR color)] drawn on the video by grab
        landmarker_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=ensure_model()),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=True,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(landmarker_opts)
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY),
            cv2.aruco.DetectorParameters())
        self._threads = [threading.Thread(target=self._grab, name="grab", daemon=True),
                         threading.Thread(target=self._run, name="vision", daemon=True)]

    def start(self):
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # a hint; not every backend honors it
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        for t in self._threads:
            t.join(timeout=2)
        self.landmarker.close()

    def read(self):
        with self._lock:
            return VisionState(**self._state.__dict__)

    def latest_images(self):
        """(video_id, video_jpeg, cut_id, cut_jpeg, mask_bytes)."""
        with self._lock:
            return self._frame_id, self._frame_jpeg, self._cut_id, self._cut_jpeg, self._mask

    # ------------------------------------------------------------------ threads
    def _grab(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._new_frame:
                self._raw = (frame, time.time())
                self._new_frame.notify()

            if not self.want_tags:      # the full-rate video is only for the select screen
                continue
            small = cv2.flip(cv2.resize(frame, STREAM_SIZE, interpolation=cv2.INTER_AREA), 1)
            scale = STREAM_SIZE[0] / frame.shape[1]
            for pts, color in self._tag_outlines:
                cv2.polylines(small, [(pts * scale).astype(np.int32)], True, color, 2)
            ok_j, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
            if ok_j:
                with self._lock:
                    self._frame_jpeg = jpeg.tobytes()
                    self._frame_id += 1

    @staticmethod
    def _encode_small(frame):
        small = cv2.flip(cv2.resize(frame, STREAM_SIZE, interpolation=cv2.INTER_AREA), 1)
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        return jpeg.tobytes() if ok else None

    def _run(self):
        last_t = time.time()
        last_ts = -1
        fps = 0.0
        prev_px = None
        smooth = None
        while not self._stop.is_set():
            with self._new_frame:
                while self._raw is None and not self._stop.is_set():
                    self._new_frame.wait(0.1)
                if self._raw is None:
                    continue
                frame, captured = self._raw
                self._raw = None          # anything that arrives while we work replaces, never queues
            now = time.time()
            h, w = frame.shape[:2]

            # Detect on the UN-mirrored frame so MediaPipe's left/right labels
            # match your real hands; mirror the results afterwards.
            rgb = cv2.cvtColor(cv2.resize(frame, POSE_INPUT, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            ts = max(int(captured * 1000) - self.start_ms, last_ts + 1)
            last_ts = ts
            result = self.landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            st = VisionState()
            mask = None
            if result.pose_landmarks:
                lm = [_Mirrored(l) for l in result.pose_landmarks[0]]
                mapped = pose_to_table(lm, w / h)
                if mapped:
                    if smooth is None:
                        smooth = list(mapped)
                    else:
                        smooth = [SMOOTHING * m + (1 - SMOOTHING) * s for m, s in zip(mapped, smooth)]
                    px, py, su, sv, sw = smooth
                    dt = max(now - last_t, 1e-3)
                    st.paddle_vx = 0.0 if prev_px is None else (px - prev_px) / dt
                    prev_px = px
                    st.paddle_x, st.paddle_y = px, py
                    st.shoulder_u, st.shoulder_v, st.shoulder_w = su, sv, sw
                    st.visible = True
                else:
                    prev_px = None
                if result.segmentation_masks:
                    m = np.squeeze(result.segmentation_masks[0].numpy_view())
                    m = cv2.resize(m, MASK_SIZE, interpolation=cv2.INTER_AREA)
                    mask = (np.fliplr(m) * 255).astype(np.uint8).tobytes()
            else:
                prev_px = None

            # AprilTag opponent pick (select screen only -- saves CPU in game)
            if self.want_tags:
                # detect on the RAW frame -- a mirror-flipped AprilTag is not a valid tag
                st.tag_opponent = self._detect_tag(frame, now)
            else:
                self._tag_seen = None
                self._tag_outlines = []

            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3)
            last_t = now
            st.fps, st.frame_t = fps, captured

            cut_jpeg = None if self.want_tags else self._encode_small(frame)   # not needed on the select screen

            with self._lock:
                self._state = st
                if cut_jpeg:
                    self._cut_jpeg = cut_jpeg
                    self._mask = mask if mask is not None else bytes(MASK_SIZE[0] * MASK_SIZE[1])
                    self._cut_id += 1

    def _detect_tag(self, frame, now):
        """Return the opponent key once the same tag has been held TAG_HOLD_S.
        `frame` is the raw (un-mirrored) camera image -- a mirrored AprilTag
        isn't a valid tag. Outlines (mirrored to match the screen) are left
        in self._tag_outlines for the grab thread to draw on the video."""
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        seen = None
        outlines = []
        if ids is not None:
            w = frame.shape[1]
            for c, tag_id in zip(corners, ids.flatten()):
                key = TAG_TO_OPPONENT.get(int(tag_id))
                color = (60, 220, 90) if key else (80, 80, 255)
                pts = c.reshape(4, 2).copy()
                pts[:, 0] = (w - 1) - pts[:, 0]            # mirror to match the displayed view
                outlines.append((pts, color))
                if key and seen is None:
                    seen = key
        self._tag_outlines = outlines
        if seen != self._tag_seen:
            self._tag_seen, self._tag_since = seen, now
        if seen and now - self._tag_since >= TAG_HOLD_S:
            return seen
        return None
