"""
vision.py -- one camera frame -> where everything is.

Three detectors, each doing what it's best at:
  * Ring:   HSV blue mask -> biggest blue contour(s) -> cv2.fitEllipse.
            Smoothed while waiting for the match, LOCKED once it starts
            (robots driving over the tape would otherwise bend the fit).
  * Us:     AprilTag tag36h11 id OUR_TAG_ID on top of our robot gives our
            position AND heading (the tag's top edge points to our front).
  * Robots: the Roboflow-labelled YOLO model (train_sumo.py) finds every
            robot. The box containing our tag is us; the best other box is
            the opponent.
            If the model file doesn't exist yet, "tag-only mode" finds the
            opponent as the largest non-floor, non-tape blob inside the ring
            -- good enough to develop and test everything else first.

Coordinates in the returned dict are pixels of the PROCESSED frame
(resized to PROCESS_WIDTH), which is also the frame sent to the UI.
"""

import math
import os
import threading
import time

import cv2
import numpy as np

from config import (APRILTAG_DICTIONARY, HSV_HIGH, HSV_LOW, MODEL_PATH, OUR_TAG_ID,
                    TAG_HEADING_OFFSET_DEG, YOLO_CONF, YOLO_IMG_SIZE)
from field import Ring

RING_SMOOTH = 0.25               # EMA weight of a new ellipse fit while unlocked
RING_MIN_AREA_FRAC = 0.002       # ignore blue specks smaller than this fraction of the frame
BLOB_MIN_AREA_FRAC = 0.004       # tag-only mode: smallest blob that counts as a robot
BLOB_MAX_AREA_FRAC = 0.06        # ...and the biggest (bigger = a patch of different floor, not a robot)
BLOB_SEARCH_SCALE = 1.5          # look this far past the tape, so a pushed-out opponent is still seen
OUR_BODY_TAG_SIZES = 1.6         # tag-only mode: our robot ~ this many tag widths from the tag center
OPP_TRACK_TTL_S = 1.0            # forget the opponent track after this long unseen
OPP_TRACK_MAX_JUMP = 0.35        # max move between frames, as a fraction of the ring's size


class Vision:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY)
        self.tags = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.model = None
        self.mode = "tag-only (no model yet)"
        if os.path.exists(MODEL_PATH):
            from ultralytics import YOLO
            self.model = YOLO(MODEL_PATH)
            self.mode = f"YOLO {os.path.basename(MODEL_PATH)}"
        self.hsv_lock = threading.Lock()
        self.hsv_low, self.hsv_high = np.array(HSV_LOW), np.array(HSV_HIGH)
        self.ring_fit = None             # smoothed (cx, cy, w, h, angle)
        self.ring_locked = False
        self.opp_track = None            # (x, y, time) of the opponent we're following

    # ------------------------------------------------------------ settings
    def set_hsv(self, low, high):
        with self.hsv_lock:
            self.hsv_low, self.hsv_high = np.array(low), np.array(high)

    def get_hsv(self):
        with self.hsv_lock:
            return self.hsv_low.tolist(), self.hsv_high.tolist()

    def lock_ring(self, locked):
        self.ring_locked = locked

    def reset_ring(self):
        self.ring_fit = None

    def reset_opponent_track(self):
        self.opp_track = None

    # ------------------------------------------------------------ detectors
    def _blue_mask(self, hsv):
        with self.hsv_lock:
            low, high = self.hsv_low, self.hsv_high
        mask = cv2.inRange(hsv, low, high)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def _fit_ring(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_area = RING_MIN_AREA_FRAC * mask.shape[0] * mask.shape[1]
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not contours:
            return None
        # the ring is usually the longest blue outline; tape gaps split it into
        # several arcs, so keep every arc at least 15% as long as the longest
        lengths = [cv2.arcLength(c, True) for c in contours]
        longest = max(lengths)
        pts = np.vstack([c for c, n in zip(contours, lengths) if n >= 0.15 * longest])
        if len(pts) < 20:
            return None
        (cx, cy), (w, h), ang = cv2.fitEllipse(pts)
        if min(w, h) < 0.15 * min(mask.shape) or max(w, h) / max(min(w, h), 1) > 4:
            return None                  # degenerate: not a ring
        return np.array([cx, cy, w, h, ang], float)

    def _update_ring(self, fit):
        if self.ring_locked and self.ring_fit is not None:
            return
        if fit is None:
            return
        if self.ring_fit is None:
            self.ring_fit = fit
            return
        # A near-circle can come back as (w, h, a) or as the same ellipse
        # (h, w, a +- 90); put the new fit in the same form as the old one,
        # then blend the angle the short way round (angles wrap at 180).
        fit = fit.copy()
        d_ang = (fit[4] - self.ring_fit[4] + 90) % 180 - 90
        if abs(d_ang) > 45:
            fit[2], fit[3] = fit[3], fit[2]
            fit[4] = (fit[4] - 90) % 180
            d_ang = (fit[4] - self.ring_fit[4] + 90) % 180 - 90
        old_ang = self.ring_fit[4]
        self.ring_fit = self.ring_fit + RING_SMOOTH * (fit - self.ring_fit)
        self.ring_fit[4] = (old_ang + RING_SMOOTH * d_ang) % 180

    def _find_us(self, gray):
        corners, ids, _ = self.tags.detectMarkers(gray)
        if ids is None:
            return None
        for c, i in zip(corners, ids.flatten()):
            if int(i) != OUR_TAG_ID:
                continue
            pts = c.reshape(4, 2)            # TL, TR, BR, BL of the printed tag
            center = pts.mean(axis=0)
            top_mid, bot_mid = (pts[0] + pts[1]) / 2, (pts[3] + pts[2]) / 2
            hx, hy = (float(v) for v in top_mid - bot_mid)       # plain floats: this ends up in JSON
            if TAG_HEADING_OFFSET_DEG:
                a = math.radians(TAG_HEADING_OFFSET_DEG)
                hx, hy = hx * math.cos(a) - hy * math.sin(a), hx * math.sin(a) + hy * math.cos(a)
            n = math.hypot(hx, hy) or 1.0
            size = float(np.linalg.norm(pts[0] - pts[1]))
            return {"x": float(center[0]), "y": float(center[1]), "hx": hx / n, "hy": hy / n,
                    "size": size, "corners": pts.round(1).tolist()}
        return None

    def _yolo_robots(self, frame):
        res = self.model.predict(frame, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            out.append({"box": [x1, y1, x2, y2], "x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "conf": conf})
        return out

    def _blob_robots(self, frame, hsv, blue, ring, us):
        """Tag-only mode: robot-sized blobs in/around the ring that differ
        from the ring's floor color and aren't tape. Runs at half resolution
        (a robot is still dozens of pixels across) to stay fast."""
        if ring is None:
            return []
        H, W = blue.shape
        small = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        blue_s = cv2.resize(blue, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        h, w = blue_s.shape
        c, ax = (ring.cx / 2, ring.cy / 2), (ring.w / 2, ring.h / 2)
        inside = np.zeros((h, w), np.uint8)
        cv2.ellipse(inside, (c, (ax[0] * 0.97, ax[1] * 0.97), ring.angle), 255, -1)
        search = np.zeros((h, w), np.uint8)
        cv2.ellipse(search, (c, (ax[0] * BLOB_SEARCH_SCALE, ax[1] * BLOB_SEARCH_SCALE), ring.angle), 255, -1)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        floor_px = lab[(inside > 0) & (blue_s == 0)]
        if not len(floor_px):
            return []
        floor = np.median(floor_px[::7], axis=0)                       # subsampled: plenty for a median
        dist2 = ((lab - floor) ** 2).sum(axis=2)                       # squared color distance (no sqrt)
        no_tape = cv2.dilate(blue_s, np.ones((5, 5), np.uint8)) == 0    # tape edges blur into the floor
        objects = ((dist2 > 28 ** 2) & (search > 0) & no_tape).astype(np.uint8) * 255
        if us is not None:
            cv2.circle(objects, (int(us["x"] / 2), int(us["y"] / 2)), int(us["size"] * OUR_BODY_TAG_SIZES / 2), 0, -1)
        objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
        contours, _ = cv2.findContours(objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not BLOB_MIN_AREA_FRAC * h * w <= area <= BLOB_MAX_AREA_FRAC * h * w:
                continue
            x, y, bw, bh = (2 * v for v in cv2.boundingRect(cnt))         # back to full-res pixels
            out.append({"box": [x, y, x + bw, y + bh], "x": x + bw / 2, "y": y + bh / 2,
                        "conf": min(1.0, area / (0.02 * h * w))})
        return out

    # ------------------------------------------------------------ main entry
    def process(self, frame):
        t0 = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue = self._blue_mask(hsv)
        self._update_ring(self._fit_ring(blue))
        ring = Ring(*self.ring_fit) if self.ring_fit is not None else None

        us = self._find_us(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        robots = self._yolo_robots(frame) if self.model else self._blob_robots(frame, hsv, blue, ring, us)

        # which robot box is us? the one that contains our tag (or is nearest to it)
        if us is not None and robots:
            def tag_dist(r):
                x1, y1, x2, y2 = r["box"]
                inside = x1 <= us["x"] <= x2 and y1 <= us["y"] <= y2
                return 0.0 if inside else math.hypot(r["x"] - us["x"], r["y"] - us["y"])
            mine = min(robots, key=tag_dist)
            if tag_dist(mine) < us["size"] * 2:
                mine["ours"] = True
                us["box"] = mine["box"]
        others = [r for r in robots if not r.get("ours")]

        # Opponent = ONE tracked robot. Stick with whatever is nearest the last
        # opponent position (it may legitimately be pushed out of the ring);
        # with no recent track, only pick up a robot that is inside the ring,
        # so bystanders/waiting robots outside are never chosen in the first place.
        opp = None
        now = time.monotonic()
        track = self.opp_track if self.opp_track and now - self.opp_track[2] < OPP_TRACK_TTL_S else None
        if others and ring is not None:
            if track:
                jump = OPP_TRACK_MAX_JUMP * max(ring.w, ring.h)
                near = [r for r in others if math.hypot(r["x"] - track[0], r["y"] - track[1]) < jump]
                if near:
                    opp = min(near, key=lambda r: math.hypot(r["x"] - track[0], r["y"] - track[1]))
            if opp is None:
                inside = [r for r in others if ring.radius_at(r["x"], r["y"]) < 1.0]
                if inside:
                    opp = max(inside, key=lambda r: r["conf"])
        if opp is not None:
            opp = dict(opp)
            self.opp_track = (opp["x"], opp["y"], now)

        return {
            "ring": ring, "us": us, "opp": opp, "robots": robots,
            "blue_frac": float(np.count_nonzero(blue)) / blue.size,
            "ms": (time.perf_counter() - t0) * 1000.0,
        }
