"""
vision.py -- one camera frame -> where everything is.

  * Field:  the space the blue tape ENCLOSES, whatever its shape (field.py
            stores it as a center + distance-to-tape per direction; no circle
            or ellipse is assumed).
            1. Closed loop: blue mask -> small gaps closed -> the largest
               HOLE inside a blue component is the field, boundary exact.
            2. Broken loop (tape gap, robot sitting on the tape): the outline
               around the big blue arcs, accepted only if tape covers at
               least FIELD_MIN_COVERAGE of that outline and the inside isn't
               blue. Anything else -> no field.
            A new field must show up in several frames in a row before it's
            believed; with no valid field for FIELD_LOST_S it is dropped. Once
            the match starts the field is LOCKED (robots hide the tape).
  * Us:     AprilTag tag36h11 id OUR_TAG_ID on top of our robot gives our
            position AND heading (the tag's top edge points to our front).
  * Robots: candidates come from the YOLO model if it exists (train_sumo.py),
            otherwise from "change detection": what differs from an
            EMPTY-ARENA SNAPSHOT (UI button), or, without a snapshot, from the
            field's median floor color. Either way a candidate must be
            robot-sized (measured against our 7 cm tag when visible), solid,
            not tape, and not us.
  * Opponent: any other team's AprilTag first; otherwise one tracked
            candidate that started INSIDE the field and has been seen in
            OPP_CONFIRM_HITS frames, so a one-frame shadow or a hand reaching
            in never becomes "the opponent".

Coordinates in the returned dict are pixels of the PROCESSED frame
(resized to PROCESS_WIDTH).
"""

import math
import os
import threading
import time

import cv2
import numpy as np

from config import (APRILTAG_DICTIONARY, HSV_HIGH, HSV_LOW, MODEL_ONNX_PATH, MODEL_PATH,
                    OPPONENT_TAG_DICTIONARIES, OUR_TAG_ID, TAG_HEADING_OFFSET_DEG, YOLO_CONF, YOLO_DEVICES,
                    YOLO_IMG_SIZE)
from field import Field

# ---- field -----------------------------------------------------------------
FIELD_GAP_CLOSE = 0.02           # bridge tape gaps up to ~2% of the frame (morphological close)
FIELD_MIN_AREA_FRAC = 0.05       # the field must cover at least this much of the frame
ARC_MIN_AREA_FRAC = 0.0015       # blue pieces smaller than this are ignored (broken-loop mode)
FIELD_MIN_COVERAGE = 0.5         # broken loop: share of the outline that must be blue tape
FIELD_MIN_VISIBLE = 0.5          # share of the outline that must be inside the frame
FIELD_MAX_INSIDE_BLUE = 0.25     # share of points INSIDE the field allowed to be blue (tape, not a blob)
FIELD_MIN_SOLIDITY = 0.6         # area / convex-hull area: dents are fine, a spiky star isn't a field
FIELD_MAX_ASPECT = 4.0           # longest / shortest center-to-tape distance
FIELD_CONFIRM_FRAMES = 3         # a new/different field must be seen this many frames in a row
FIELD_LOST_S = 1.0               # unlocked field with no valid detection for this long -> no field
FIELD_SMOOTH = 0.25              # EMA weight of a new detection of the current field

# ---- robots ----------------------------------------------------------------
BLOB_SEARCH_SCALE = 1.4          # look this far past the tape, so a pushed-out opponent is still seen
OUR_BODY_TAG_SIZES = 1.8         # our robot ~ this many tag widths from the tag center (cut out of blobs)
CHANGE_THRESH_REF = 24           # color distance vs. the empty-arena snapshot that counts as "changed"
CHANGE_THRESH_FLOOR = 34         # stricter threshold vs. median floor color (no snapshot)
L_WEIGHT = 0.5                   # brightness counts half: shadows change brightness, not color
SIZE_MIN_TAGS, SIZE_MAX_TAGS = 1.1, 7.0     # robot's longer side, in tag widths
SIZE_MIN_FIELD, SIZE_MAX_FIELD = 0.06, 0.45  # ...or, with no tag, in field diameters
MIN_SOLIDITY = 0.55              # blob area / convex hull area (cars are chunky, shadows/streaks aren't)
MIN_FILL = 0.30                  # blob area / bounding box area
MAX_ASPECT = 3.5
YOLO_MIN_CHANGE = 0.15           # with a snapshot: a YOLO box must cover at least this much change
OPP_TRACK_TTL_S = 1.0            # forget the opponent track after this long unseen...
OPP_TRACK_TTL_FRAMES = 3         # ...or this many frame intervals, whichever is longer (slow laptops)
OPP_TRACK_MAX_JUMP = 0.30        # max move between frames, as a fraction of the field's diameter
OPP_CONFIRM_HITS = 3             # frames a candidate must be seen before it is "the opponent"
OPP_TAG_BOX_TAGS = 2.5           # tagged opponent with no detected robot box: assume a car this many tags wide


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
        CPU can do field/tag work meanwhile; collect() gets the boxes."""
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
        # ours is APRILTAG_DICTIONARY #OUR_TAG_ID; every family listed is searched for opponent tags
        families = [APRILTAG_DICTIONARY] + [f for f in OPPONENT_TAG_DICTIONARIES if f != APRILTAG_DICTIONARY]
        self.tag_detectors = [(f, cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(f),
                                                         cv2.aruco.DetectorParameters())) for f in families]
        self.model, self.mode = _load_detector()
        print(f"[vision] robot detection: {self.mode}")
        self.hsv_lock = threading.Lock()
        self.hsv_low, self.hsv_high = np.array(HSV_LOW), np.array(HSV_HIGH)
        self.field = None                # current Field (any shape)
        self.field_seen_t = 0.0
        self.field_pending = None        # (Field, frames in a row) for a field not yet believed
        self.field_quality = None        # 1.0 for a closed tape loop, else tape coverage of the outline
        self.field_kind = None           # "closed" | "partial"
        self.field_locked = False
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

    def lock_field(self, locked):
        self.field_locked = locked

    def reset_field(self):
        self.field, self.field_pending = None, None

    def reset_opponent_track(self):
        self.opp_track = None

    def request_reference(self):
        """Take the next frame as the empty-arena snapshot (called from the UI thread)."""
        self.want_reference = True

    def clear_reference(self):
        self.reference, self.reference_t = None, None

    # ------------------------------------------------------------ field
    def _blue_mask(self, hsv):
        with self.hsv_lock:
            low, high = self.hsv_low, self.hsv_high
        mask = cv2.inRange(hsv, low, high)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    @staticmethod
    def _plausible(field, contour, H, W):
        """Shape sanity for a candidate field (any shape, but field-like)."""
        if field is None or field.area < FIELD_MIN_AREA_FRAC * H * W:
            return False
        hull_area = cv2.contourArea(cv2.convexHull(np.asarray(contour, np.float32).reshape(-1, 1, 2)))
        if hull_area <= 0 or field.area / hull_area < FIELD_MIN_SOLIDITY:
            return False
        if field.radii.max() / max(field.radii.min(), 1e-6) > FIELD_MAX_ASPECT:
            return False
        return True

    @staticmethod
    def _outline_checks(field, near_tape):
        """Broken-loop mode: share of the outline covered by tape, or None if the
        outline isn't mostly visible or the inside is blue (a blob, not a border)."""
        H, W = near_tape.shape

        def samples(k):
            pts = field.polygon(k)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
            return pts[ok].astype(int), ok.mean()

        edge, visible = samples(1.0)
        if visible < FIELD_MIN_VISIBLE:
            return None
        inner = np.vstack([samples(k)[0] for k in (0.4, 0.7)])
        if len(inner) and np.mean(near_tape[inner[:, 1], inner[:, 0]] > 0) > FIELD_MAX_INSIDE_BLUE:
            return None
        return float(np.mean(near_tape[edge[:, 1], edge[:, 0]] > 0))

    def _find_field(self, mask):
        """-> (Field, quality, kind): kind "closed" (tape loop, exact shape) or
        "partial" (outline of broken tape, quality = tape coverage); or (None, None, None)."""
        H, W = mask.shape
        k = max(5, int(FIELD_GAP_CLOSE * min(H, W)) | 1)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

        # 1. closed loop: the field is a HOLE of a blue component = the space the tape encloses
        contours, hier = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        best = None
        if hier is not None:
            for c, h in zip(contours, hier[0]):
                if h[3] < 0 or cv2.contourArea(c) < FIELD_MIN_AREA_FRAC * H * W:
                    continue                                     # not a hole, or too small
                f = Field.from_contour(c)
                if self._plausible(f, c, H, W) and (best is None or f.area > best.area):
                    best = f
        if best is not None:
            return best, 1.0, "closed"

        # 2. broken loop: outline (convex hull) around the big blue arcs
        arcs = [c for c in cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                if cv2.contourArea(c) >= ARC_MIN_AREA_FRAC * H * W]
        if not arcs:
            return None, None, None
        arcs.sort(key=lambda c: cv2.arcLength(c, True), reverse=True)
        longest = cv2.arcLength(arcs[0], True)
        # all big arcs together (tape gaps split the loop), and each alone (other blue junk)
        candidates = [np.vstack([c for c in arcs if cv2.arcLength(c, True) >= 0.15 * longest])] + arcs[:3]
        near_tape = cv2.dilate(mask, np.ones((k, k), np.uint8))
        best, best_cov = None, None
        for cand in candidates:
            hull = cv2.convexHull(cand)
            f = Field.from_contour(hull)
            if not self._plausible(f, hull, H, W):
                continue
            cov = self._outline_checks(f, near_tape)
            if cov is not None and cov >= FIELD_MIN_COVERAGE and (best_cov is None or cov > best_cov):
                best, best_cov = f, cov
        return (best, best_cov, "partial") if best is not None else (None, None, None)

    def _update_field(self, found, quality, kind, now):
        if self.field_locked and self.field is not None:
            return
        if found is not None and self.field is not None and self.field.similar(found):
            self.field = self.field.blend(found, FIELD_SMOOTH)          # same field: smooth it
            self.field_seen_t, self.field_pending = now, None
            self.field_quality, self.field_kind = quality, kind
            return
        if found is not None:                                           # new or different field:
            n = self.field_pending[1] + 1 if self.field_pending and self.field_pending[0].similar(found) else 1
            self.field_pending = (found, n)                             # believe it only if it persists
            if n >= FIELD_CONFIRM_FRAMES:
                self.field, self.field_seen_t, self.field_pending = found, now, None
                self.field_quality, self.field_kind = quality, kind
                return
        else:
            self.field_pending = None
        if self.field is not None and now - self.field_seen_t > FIELD_LOST_S:
            self.field, self.field_quality, self.field_kind = None, None, None   # no tape -> no field

    # ------------------------------------------------------------ us
    @staticmethod
    def _tag_pose(pts, tag_id, family, heading_offset=0.0):
        center = pts.mean(axis=0)
        top_mid, bot_mid = (pts[0] + pts[1]) / 2, (pts[3] + pts[2]) / 2   # TL, TR, BR, BL of the printed tag
        hx, hy = (float(v) for v in top_mid - bot_mid)                    # plain floats: this ends up in JSON
        if heading_offset:
            a = math.radians(heading_offset)
            hx, hy = hx * math.cos(a) - hy * math.sin(a), hx * math.sin(a) + hy * math.cos(a)
        n = math.hypot(hx, hy) or 1.0
        return {"x": float(center[0]), "y": float(center[1]), "hx": hx / n, "hy": hy / n,
                "size": float(np.linalg.norm(pts[0] - pts[1])), "corners": pts.round(1).tolist(),
                "id": tag_id, "family": family}

    def _find_tags(self, gray):
        """-> (us, [opponent tags]). Ours = OUR tag family + OUR_TAG_ID; any other
        tag (any id, any family in OPPONENT_TAG_DICTIONARIES) is an opponent's."""
        us, theirs = None, []
        for family, detector in self.tag_detectors:
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            for c, i in zip(corners, ids.flatten()):
                pts = c.reshape(4, 2)
                if family == APRILTAG_DICTIONARY and int(i) == OUR_TAG_ID:
                    us = self._tag_pose(pts, int(i), family, TAG_HEADING_OFFSET_DEG)
                else:
                    theirs.append(self._tag_pose(pts, int(i), family))
        return us, theirs

    def _opponent_from_tags(self, tags, robots, field, now):
        """An opponent's AprilTag beats every other cue: pick it (same in/out-of-field
        and tracking rules as _track_opponent) and confirm it immediately."""
        ttl = max(OPP_TRACK_TTL_S, OPP_TRACK_TTL_FRAMES * self.frame_dt)
        track = self.opp_track if self.opp_track and now - self.opp_track["t"] < ttl else None
        tag = None
        if track:
            jump = OPP_TRACK_MAX_JUMP * 2 * field.scale
            near = [t for t in tags if math.hypot(t["x"] - track["x"], t["y"] - track["y"]) < jump]
            if near:
                tag = min(near, key=lambda t: math.hypot(t["x"] - track["x"], t["y"] - track["y"]))
        if tag is None:
            inside = [t for t in tags if field.radius_at(t["x"], t["y"]) < 1.0]
            if inside:      # several tagged robots in the field: the one closest to its center
                tag = min(inside, key=lambda t: field.radius_at(t["x"], t["y"]))
        if tag is None:
            return None
        # the robot's extent: a detected robot box containing the tag, else a box around the tag
        box = next((r["box"] for r in robots if r["box"][0] <= tag["x"] <= r["box"][2]
                    and r["box"][1] <= tag["y"] <= r["box"][3]), None)
        if box is not None:
            for r in robots:
                if r["box"] is box:
                    r["tag_id"] = tag["id"]
        else:
            half = tag["size"] * OPP_TAG_BOX_TAGS / 2
            box = [tag["x"] - half, tag["y"] - half, tag["x"] + half, tag["y"] + half]
        self.opp_track = {"x": tag["x"], "y": tag["y"], "t": now, "hits": OPP_CONFIRM_HITS + 2}
        return {"box": box, "x": tag["x"], "y": tag["y"], "conf": 1.0, "tag_id": tag["id"],
                "corners": tag["corners"]}

    # ------------------------------------------------------------ robots
    @staticmethod
    def _small_lab(frame):
        H, W = frame.shape[:2]
        small = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), (5, 5), 0).astype(np.float32)

    def _change_mask(self, lab, blue_s, field):
        """Half-res mask of pixels that look like 'something is here'."""
        if self.reference is not None and self.reference.shape == lab.shape:
            base, thresh = self.reference, CHANGE_THRESH_REF
        else:
            inside = field.mask(blue_s.shape, 0.95, downscale=2)       # the field's real outline
            floor_px = lab[(inside > 0) & (blue_s == 0)]
            if not len(floor_px):
                return None
            base, thresh = np.median(floor_px[::7], axis=0), CHANGE_THRESH_FLOOR
        d = lab - base
        dist2 = L_WEIGHT * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2
        return (dist2 > thresh ** 2).astype(np.uint8) * 255

    def _size_ok(self, bw, bh, field, us):
        side = max(bw, bh)
        if us is not None:
            return SIZE_MIN_TAGS * us["size"] <= side <= SIZE_MAX_TAGS * us["size"]
        d = 2 * field.scale                                    # equal-area diameter of the field
        return SIZE_MIN_FIELD * d <= side <= SIZE_MAX_FIELD * d

    def _candidates(self, frame, blue, field, us):
        H, W = blue.shape
        lab = self._small_lab(frame)
        blue_s = cv2.resize(blue, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        change = self._change_mask(lab, blue_s, field)
        search = field.mask(blue_s.shape, BLOB_SEARCH_SCALE, downscale=2)   # field outline, grown 40%
        no_tape = cv2.dilate(blue_s, np.ones((7, 7), np.uint8)) == 0
        if change is not None:
            change = change & search & (no_tape.astype(np.uint8) * 255)

        out = []
        if self.model is not None:
            for x1, y1, x2, y2, conf in self.model.collect(YOLO_CONF):   # started in process()
                if not self._size_ok(x2 - x1, y2 - y1, field, us):
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
            if area < 30 or not self._size_ok(2 * bw, 2 * bh, field, us):
                continue
            hull = cv2.contourArea(cv2.convexHull(cnt)) or 1.0
            solidity, fill = area / hull, area / max(bw * bh, 1)
            if solidity < MIN_SOLIDITY or fill < MIN_FILL or max(bw, bh) / max(min(bw, bh), 1) > MAX_ASPECT:
                continue
            x, y, bw, bh = 2 * x, 2 * y, 2 * bw, 2 * bh                  # back to full-res pixels
            out.append({"box": [x, y, x + bw, y + bh], "x": x + bw / 2, "y": y + bh / 2,
                        "conf": round(min(1.0, solidity * fill * 1.6), 2)})
        return out

    def _track_opponent(self, others, field, now):
        ttl = max(OPP_TRACK_TTL_S, OPP_TRACK_TTL_FRAMES * self.frame_dt)
        track = self.opp_track if self.opp_track and now - self.opp_track["t"] < ttl else None
        match = None
        if track and others:
            jump = OPP_TRACK_MAX_JUMP * 2 * field.scale
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
                # new tracks only start INSIDE the field: bystanders outside are never chosen
                inside = [r for r in others if field.radius_at(r["x"], r["y"]) < 1.0]
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
        found, quality, kind = self._find_field(blue)
        self._update_field(found, quality, kind, now)
        field = self.field

        us, their_tags = self._find_tags(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        robots = self._candidates(frame, blue, field, us) if field is not None else []
        if field is None and self.model is not None:
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
        opp = None
        if field is not None:
            opp = self._opponent_from_tags(their_tags, others, field, now) if their_tags else None
            if opp is None:                        # no usable opponent tag: model / change detection
                opp = self._track_opponent(others, field, now)
        if opp is None and field is None:
            self.opp_track = None

        return {
            "field": field, "us": us, "opp": opp, "robots": robots, "their_tags": their_tags,
            "field_quality": self.field_quality, "field_kind": self.field_kind,
            "field_pending": self.field_pending is not None,
            "reference_age": (now - self.reference_t) if self.reference_t else None,
            "ms": (time.perf_counter() - t0) * 1000.0,
        }
