"""
vision.py -- one camera frame -> where everything is.

  * Ring:   HSV blue mask -> candidate ellipses (cv2.fitEllipse) that must be
            VALIDATED: blue tape has to cover a real share of the ellipse's
            outline, the fitted points have to hug it, and the shape has to be
            plausibly round. A new ring must show up in several frames in a
            row before it's believed; with no valid ring for RING_LOST_S it is
            dropped (no blue arc -> no field). Once the match starts the ring
            is LOCKED, since robots driving over the tape hide parts of it.
  * Us:     AprilTag tag36h11 id OUR_TAG_ID on top of our robot gives our
            position AND heading (the tag's top edge points to our front).
  * Robots: candidates come from the YOLO model if it exists (train_sumo.py),
            otherwise from "change detection": what differs from an
            EMPTY-ARENA SNAPSHOT (UI button), or, without a snapshot, from the
            ring's median floor color. Either way a candidate must be
            robot-sized (measured against our 7 cm tag when visible), solid,
            not tape, and not us.
  * Opponent: one tracked candidate that started INSIDE the ring and has
            been seen in OPP_CONFIRM_HITS frames, so a one-frame shadow or a
            hand reaching in never becomes "the opponent".

Coordinates in the returned dict are pixels of the PROCESSED frame
(resized to PROCESS_WIDTH).
"""

import math
import os
import threading
import time

import cv2
import numpy as np

from config import (APRILTAG_DICTIONARY, HSV_HIGH, HSV_LOW, MODEL_ONNX_PATH, MODEL_PATH, OUR_TAG_ID,
                    TAG_HEADING_OFFSET_DEG, YOLO_CONF, YOLO_DEVICES, YOLO_IMG_SIZE)
from field import Ring

# ---- ring ------------------------------------------------------------------
RING_MIN_AREA_FRAC = 0.0015      # ignore blue specks smaller than this fraction of the frame
RING_SAMPLES = 180               # points sampled along a candidate ellipse
RING_MIN_COVERAGE = 0.35         # share of the (visible) outline that must be blue tape
RING_MIN_INLIERS = 0.70          # share of the fitted blue points that must lie on the ellipse
RING_INLIER_TOL = 0.15           # |normalized radius - 1| counted as "on the ellipse"
RING_MIN_VISIBLE = 0.5           # share of the ellipse outline that must be inside the frame
RING_MAX_INSIDE_BLUE = 0.25      # share of points INSIDE the ring allowed to be blue (it's tape, not a disc)
RING_MIN_SIZE = 0.25             # smaller axis >= this fraction of the frame's smaller side
RING_MAX_ASPECT = 2.5            # camera tilt squashes the circle, but not more than this
RING_CONFIRM_FRAMES = 3          # a new/different ring must be seen this many frames in a row
RING_LOST_S = 1.0                # unlocked ring with no valid fit for this long -> no field
RING_SMOOTH = 0.25               # EMA weight of a new fit for the current ring

# ---- robots ----------------------------------------------------------------
BLOB_SEARCH_SCALE = 1.4          # look this far past the tape, so a pushed-out opponent is still seen
OUR_BODY_TAG_SIZES = 1.8         # our robot ~ this many tag widths from the tag center (cut out of blobs)
CHANGE_THRESH_REF = 24           # color distance vs. the empty-arena snapshot that counts as "changed"
CHANGE_THRESH_FLOOR = 34         # stricter threshold vs. median floor color (no snapshot)
L_WEIGHT = 0.5                   # brightness counts half: shadows change brightness, not color
SIZE_MIN_TAGS, SIZE_MAX_TAGS = 1.1, 7.0     # robot's longer side, in tag widths
SIZE_MIN_RING, SIZE_MAX_RING = 0.06, 0.45   # ...or, with no tag, in ring diameters
MIN_SOLIDITY = 0.55              # blob area / convex hull area (cars are chunky, shadows/streaks aren't)
MIN_FILL = 0.30                  # blob area / bounding box area
MAX_ASPECT = 3.5
YOLO_MIN_CHANGE = 0.15           # with a snapshot: a YOLO box must cover at least this much change
OPP_TRACK_TTL_S = 1.0            # forget the opponent track after this long unseen...
OPP_TRACK_TTL_FRAMES = 3         # ...or this many frame intervals, whichever is longer (slow laptops)
OPP_TRACK_MAX_JUMP = 0.30        # max move between frames, as a fraction of the ring's size
OPP_CONFIRM_HITS = 3             # frames a candidate must be seen before it is "the opponent"


def _norm_radius(fit, xs, ys):
    cx, cy, w, h, ang = fit
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    dx, dy = xs - cx, ys - cy
    return np.hypot((dx * c + dy * s) / max(w / 2, 1e-6), (-dx * s + dy * c) / max(h / 2, 1e-6))


def _ellipse_points(fit, n=RING_SAMPLES):
    cx, cy, w, h, ang = fit
    a = math.radians(ang)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    u, v = np.cos(t) * w / 2, np.sin(t) * h / 2
    return cx + u * math.cos(a) - v * math.sin(a), cy + u * math.sin(a) + v * math.cos(a)


def _align(fit, ref):
    """Express `fit` the same way as `ref` (an ellipse can be (w,h,a) or (h,w,a+90))."""
    fit = fit.copy()
    d = (fit[4] - ref[4] + 90) % 180 - 90
    if abs(d) > 45:
        fit[2], fit[3] = fit[3], fit[2]
        fit[4] = (fit[4] - 90) % 180
    return fit


def _similar(a, b):
    b = _align(b, a)
    mean_axis = (a[2] + a[3]) / 4
    return (math.hypot(a[0] - b[0], a[1] - b[1]) < 0.15 * mean_axis
            and 0.8 < b[2] / a[2] < 1.25 and 0.8 < b[3] / a[3] < 1.25)


class _OpenVinoYolo:
    """The trained YOLO model (exported to ONNX by train_sumo.py) run with Intel
    OpenVINO -- no PyTorch needed, so Windows Smart App Control's block on
    my_env's PyTorch DLLs doesn't matter. Tries YOLO_DEVICES in order; on this
    laptop the NPU does a frame in ~16 ms (GPU ~25 ms, CPU ~88 ms, PyTorch-on-CPU
    ~330 ms) and leaves the CPU free for the camera and change detection."""

    def __init__(self, path):
        import openvino as ov
        core = ov.Core()
        model = core.read_model(path)
        _, _, self.in_h, self.in_w = (int(d) for d in model.inputs[0].shape)
        errors = []
        for want in YOLO_DEVICES:
            for dev in [d for d in core.available_devices if d.split(".")[0] == want]:
                try:
                    self.compiled = core.compile_model(model, dev, {"PERFORMANCE_HINT": "LATENCY"})
                    self.device = dev
                    self.req = self.compiled.create_infer_request()
                    self.out = self.compiled.outputs[0]
                    return
                except Exception as e:
                    errors.append(f"{dev}: {e}")
        raise RuntimeError("no OpenVINO device could run the model: " + "; ".join(errors))

    def start(self, frame):
        """Kick off inference on the accelerator and return immediately, so the
        CPU can do ring/tag work meanwhile; collect() gets the boxes."""
        self.req.wait()                                     # finish any job whose result nobody collected
        self.shape = frame.shape[:2]
        # STRETCH to the model size: the Roboflow export stretched 960x540 -> 512x512,
        # so cars must look stretched the same way here (padding would change their shape)
        x = cv2.resize(frame, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        self.req.start_async({0: x})

    def collect(self, conf):
        """-> [(x1, y1, x2, y2, conf)] in frame pixels, for the frame given to start()."""
        self.req.wait()
        H, W = self.shape
        p = self.req.get_output_tensor(0).data[0]          # (4 + classes, N): cx, cy, w, h, scores
        scores = p[4:].max(axis=0)
        keep = scores > conf
        if not keep.any():
            return []
        cx, cy, bw, bh = p[:4, keep]
        sx, sy = W / self.in_w, H / self.in_h
        boxes = np.stack([(cx - bw / 2) * sx, (cy - bh / 2) * sy, bw * sx, bh * sy], axis=1)
        idx = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), scores[keep].tolist(), conf, 0.5)).flatten()
        return [(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 0] + boxes[i, 2]),
                 float(boxes[i, 1] + boxes[i, 3]), float(scores[keep][i])) for i in idx]


class _UltralyticsYolo:
    """Fallback: the .pt model through ultralytics/PyTorch (needs my_env_cpu on this
    laptop, ~330 ms/frame on CPU). Same stretched input as the OpenVINO path."""

    def __init__(self, path):
        from ultralytics import YOLO
        self.model = YOLO(path)
        self.device = "torch"

    def start(self, frame):
        self.frame = frame

    def collect(self, conf):
        frame = self.frame
        H, W = frame.shape[:2]
        x = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE), interpolation=cv2.INTER_AREA)
        res = self.model.predict(x, imgsz=YOLO_IMG_SIZE, conf=conf, verbose=False)[0]
        sx, sy = W / YOLO_IMG_SIZE, H / YOLO_IMG_SIZE
        return [(x1 * sx, y1 * sy, x2 * sx, y2 * sy, c)
                for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())]


def _load_detector():
    """OpenVINO + ONNX first, then ultralytics + .pt; None = change detection."""
    for path, cls in ((MODEL_ONNX_PATH, _OpenVinoYolo), (MODEL_PATH, _UltralyticsYolo)):
        if not os.path.exists(path):
            continue
        try:
            det = cls(path)
            return det, f"YOLO {os.path.basename(path)} on {det.device}"
        except Exception as e:      # e.g. Smart App Control blocking PyTorch DLLs (OSError)
            print(f"[vision] can't use {os.path.basename(path)}: {type(e).__name__}: {e}")
    return None, "change detection (no usable model)"


class Vision:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY)
        self.tags = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.model, self.mode = _load_detector()
        print(f"[vision] robot detection: {self.mode}")
        self.hsv_lock = threading.Lock()
        self.hsv_low, self.hsv_high = np.array(HSV_LOW), np.array(HSV_HIGH)
        self.ring_fit = None             # current (cx, cy, w, h, angle)
        self.ring_seen_t = 0.0
        self.ring_pending = None         # (fit, frames in a row) for a ring not yet believed
        self.ring_quality = None         # tape coverage of the last accepted fit
        self.ring_locked = False
        self.opp_track = None            # {"x", "y", "t", "hits"}
        self.frame_dt = 0.1              # smoothed time between processed frames
        self.last_frame_t = None
        self.reference = None            # blurred half-res Lab image of the empty arena
        self.reference_t = None
        self.want_reference = False

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
        self.ring_fit, self.ring_pending = None, None

    def reset_opponent_track(self):
        self.opp_track = None

    def request_reference(self):
        """Take the next frame as the empty-arena snapshot (called from the UI thread)."""
        self.want_reference = True

    def clear_reference(self):
        self.reference, self.reference_t = None, None

    # ------------------------------------------------------------ ring
    def _blue_mask(self, hsv):
        with self.hsv_lock:
            low, high = self.hsv_low, self.hsv_high
        mask = cv2.inRange(hsv, low, high)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def _score_ellipse(self, fit, pts, near_tape):
        """Return tape coverage if `fit` is a believable ring for these points, else None."""
        H, W = near_tape.shape
        cx, cy, w, h, _ = fit
        if not (np.isfinite(fit).all() and min(w, h) >= RING_MIN_SIZE * min(H, W)
                and max(w, h) / max(min(w, h), 1) <= RING_MAX_ASPECT
                and max(w, h) <= 3 * max(H, W)):
            return None
        r = _norm_radius(fit, pts[:, 0].astype(float), pts[:, 1].astype(float))
        if np.mean(np.abs(r - 1) < RING_INLIER_TOL) < RING_MIN_INLIERS:
            return None
        xs, ys = _ellipse_points(fit)
        visible = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if visible.mean() < RING_MIN_VISIBLE:
            return None
        coverage = float(np.mean(near_tape[ys[visible].astype(int), xs[visible].astype(int)] > 0))
        if coverage < RING_MIN_COVERAGE:
            return None
        # a tape ring is HOLLOW: inside it is floor, not blue (rejects blue boxes, blue robots, ...)
        inner = []
        for scale in (0.4, 0.7):
            ix, iy = _ellipse_points((cx, cy, w * scale, h * scale, fit[4]), 60)
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            inner.append(near_tape[iy[ok].astype(int), ix[ok].astype(int)] > 0)
        inner = np.concatenate(inner)
        if inner.size and inner.mean() > RING_MAX_INSIDE_BLUE:
            return None
        return coverage

    def _fit_ring(self, mask):
        H, W = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv2.contourArea(c) >= RING_MIN_AREA_FRAC * H * W]
        if not contours:
            return None, None
        contours.sort(key=lambda c: cv2.arcLength(c, True), reverse=True)
        longest = cv2.arcLength(contours[0], True)
        # candidates: all big arcs together (tape gaps split the ring), plus
        # each big arc alone (in case other blue things polluted the union)
        candidates = [np.vstack([c for c in contours if cv2.arcLength(c, True) >= 0.15 * longest])]
        candidates += [c for c in contours[:4] if cv2.arcLength(c, True) >= 0.1 * (H + W)]
        k = max(5, int(0.012 * min(H, W)) | 1)
        near_tape = cv2.dilate(mask, np.ones((k, k), np.uint8))
        best, best_cov = None, None
        for cand in candidates:
            pts = cand.reshape(-1, 2)
            if len(pts) < 20:
                continue
            (cx, cy), (w, h), ang = cv2.fitEllipse(pts.astype(np.float32))
            fit = np.array([cx, cy, w, h, ang], float)
            # one refinement pass: refit on the points that actually lie on it
            r = _norm_radius(fit, pts[:, 0].astype(float), pts[:, 1].astype(float))
            on = pts[np.abs(r - 1) < 0.2]
            if len(on) >= 20 and len(on) < len(pts):
                (cx, cy), (w, h), ang = cv2.fitEllipse(on.astype(np.float32))
                fit, pts = np.array([cx, cy, w, h, ang], float), on
            cov = self._score_ellipse(fit, pts, near_tape)
            if cov is not None and (best_cov is None or cov > best_cov):
                best, best_cov = fit, cov
        return best, best_cov

    def _update_ring(self, fit, quality, now):
        if self.ring_locked and self.ring_fit is not None:
            return
        if fit is not None and self.ring_fit is not None and _similar(self.ring_fit, fit):
            fit = _align(fit, self.ring_fit)                          # same ring: smooth it
            d_ang = (fit[4] - self.ring_fit[4] + 90) % 180 - 90
            old_ang = self.ring_fit[4]
            self.ring_fit = self.ring_fit + RING_SMOOTH * (fit - self.ring_fit)
            self.ring_fit[4] = (old_ang + RING_SMOOTH * d_ang) % 180
            self.ring_seen_t, self.ring_pending, self.ring_quality = now, None, quality
            return
        if fit is not None:                                           # new or different ring:
            n = self.ring_pending[1] + 1 if self.ring_pending and _similar(self.ring_pending[0], fit) else 1
            self.ring_pending = (fit, n)                              # believe it only if it persists
            if n >= RING_CONFIRM_FRAMES:
                self.ring_fit, self.ring_seen_t, self.ring_pending, self.ring_quality = fit, now, None, quality
                return
        else:
            self.ring_pending = None
        if self.ring_fit is not None and now - self.ring_seen_t > RING_LOST_S:
            self.ring_fit, self.ring_quality = None, None             # no blue arc -> no field

    # ------------------------------------------------------------ us
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

    # ------------------------------------------------------------ robots
    @staticmethod
    def _small_lab(frame):
        H, W = frame.shape[:2]
        small = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), (5, 5), 0).astype(np.float32)

    def _change_mask(self, lab, blue_s, ring):
        """Half-res mask of pixels that look like 'something is here'."""
        if self.reference is not None and self.reference.shape == lab.shape:
            base, thresh = self.reference, CHANGE_THRESH_REF
        else:
            inside = np.zeros(blue_s.shape, np.uint8)
            cv2.ellipse(inside, ((ring.cx / 2, ring.cy / 2), (ring.w / 2 * 0.95, ring.h / 2 * 0.95), ring.angle), 255, -1)
            floor_px = lab[(inside > 0) & (blue_s == 0)]
            if not len(floor_px):
                return None
            base, thresh = np.median(floor_px[::7], axis=0), CHANGE_THRESH_FLOOR
        d = lab - base
        dist2 = L_WEIGHT * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2
        return (dist2 > thresh ** 2).astype(np.uint8) * 255

    def _size_ok(self, bw, bh, ring, us):
        side = max(bw, bh)
        if us is not None:
            return SIZE_MIN_TAGS * us["size"] <= side <= SIZE_MAX_TAGS * us["size"]
        d = (ring.w + ring.h) / 2
        return SIZE_MIN_RING * d <= side <= SIZE_MAX_RING * d

    def _candidates(self, frame, blue, ring, us):
        H, W = blue.shape
        lab = self._small_lab(frame)
        blue_s = cv2.resize(blue, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        change = self._change_mask(lab, blue_s, ring)
        search = np.zeros(blue_s.shape, np.uint8)
        s = BLOB_SEARCH_SCALE
        cv2.ellipse(search, ((ring.cx / 2, ring.cy / 2), (ring.w / 2 * s, ring.h / 2 * s), ring.angle), 255, -1)
        no_tape = cv2.dilate(blue_s, np.ones((7, 7), np.uint8)) == 0
        if change is not None:
            change = change & search & (no_tape.astype(np.uint8) * 255)

        out = []
        if self.model is not None:
            for x1, y1, x2, y2, conf in self.model.collect(YOLO_CONF):   # started in process()
                if not self._size_ok(x2 - x1, y2 - y1, ring, us):
                    continue
                if self.reference is not None and change is not None:
                    roi = change[int(y1) // 2:int(y2) // 2 + 1, int(x1) // 2:int(x2) // 2 + 1]
                    if roi.size == 0 or np.count_nonzero(roi) / roi.size < YOLO_MIN_CHANGE:
                        continue                         # nothing actually there vs. the empty arena
                out.append({"box": [x1, y1, x2, y2], "x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "conf": conf})
            return out

        if change is None:
            return out
        if us is not None:
            cv2.circle(change, (int(us["x"] / 2), int(us["y"] / 2)), int(us["size"] * OUR_BODY_TAG_SIZES / 2), 0, -1)
        change = cv2.morphologyEx(change, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        change = cv2.morphologyEx(change, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(change, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if area < 30 or not self._size_ok(2 * bw, 2 * bh, ring, us):
                continue
            hull = cv2.contourArea(cv2.convexHull(cnt)) or 1.0
            solidity, fill = area / hull, area / max(bw * bh, 1)
            if solidity < MIN_SOLIDITY or fill < MIN_FILL or max(bw, bh) / max(min(bw, bh), 1) > MAX_ASPECT:
                continue
            x, y, bw, bh = 2 * x, 2 * y, 2 * bw, 2 * bh                  # back to full-res pixels
            out.append({"box": [x, y, x + bw, y + bh], "x": x + bw / 2, "y": y + bh / 2,
                        "conf": round(min(1.0, solidity * fill * 1.6), 2)})
        return out

    def _track_opponent(self, others, ring, now):
        ttl = max(OPP_TRACK_TTL_S, OPP_TRACK_TTL_FRAMES * self.frame_dt)
        track = self.opp_track if self.opp_track and now - self.opp_track["t"] < ttl else None
        match = None
        if track and others:
            jump = OPP_TRACK_MAX_JUMP * max(ring.w, ring.h)
            dist = lambda r: math.hypot(r["x"] - track["x"], r["y"] - track["y"])  # noqa: E731
            near = [r for r in others if dist(r) < jump]
            if near:
                match = min(near, key=dist)
        if match:
            track.update(x=match["x"], y=match["y"], t=now, hits=min(track["hits"] + 1, OPP_CONFIRM_HITS + 2))
        else:
            if track:
                track["hits"] = max(0, track["hits"] - 2)    # misses count double: lost tracks yield fast
            if not track or track["hits"] < OPP_CONFIRM_HITS:
                # new tracks only start INSIDE the ring: bystanders outside are never chosen
                inside = [r for r in others if ring.radius_at(r["x"], r["y"]) < 1.0]
                if inside:
                    best = max(inside, key=lambda r: r["conf"])
                    track = {"x": best["x"], "y": best["y"], "t": now, "hits": 1}
                    match = best
        self.opp_track = track
        if match and track and track["hits"] >= OPP_CONFIRM_HITS:
            return dict(match)
        return None

    # ------------------------------------------------------------ main entry
    def process(self, frame):
        t0 = time.perf_counter()
        now = time.monotonic()
        if self.last_frame_t is not None:
            self.frame_dt = 0.8 * self.frame_dt + 0.2 * min(now - self.last_frame_t, 5.0)
        self.last_frame_t = now
        if self.want_reference:
            self.reference, self.reference_t, self.want_reference = self._small_lab(frame), now, False

        if self.model is not None:
            self.model.start(frame)               # NPU works on this frame while the CPU does the rest
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue = self._blue_mask(hsv)
        fit, quality = self._fit_ring(blue)
        self._update_ring(fit, quality, now)
        ring = Ring(*self.ring_fit) if self.ring_fit is not None else None

        us = self._find_us(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        robots = self._candidates(frame, blue, ring, us) if ring is not None else []
        if ring is None and self.model is not None:
            self.model.collect(YOLO_CONF)         # no field: drain the started inference, ignore it

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
        opp = self._track_opponent(others, ring, now) if ring is not None else None
        if opp is None and ring is None:
            self.opp_track = None

        return {
            "ring": ring, "us": us, "opp": opp, "robots": robots,
            "ring_quality": self.ring_quality, "ring_pending": self.ring_pending is not None,
            "reference_age": (now - self.reference_t) if self.reference_t else None,
            "ms": (time.perf_counter() - t0) * 1000.0,
        }
