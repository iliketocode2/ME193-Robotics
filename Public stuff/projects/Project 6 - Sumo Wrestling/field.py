"""
field.py -- the playing field, whatever shape the blue tape makes.

The field is NOT assumed to be a circle. vision.py finds the space the blue
tape encloses and hands its boundary here. The field is stored as its center
plus a RADIAL PROFILE: the distance from the center to the tape in each of
N_BINS directions. That captures ovals, squares, wobbly hand-taped loops,
dents -- any shape you can see the whole edge of from its center
("star-shaped", which every reasonable sumo ring is).

Normalized radius r of a point = its distance from the center / the distance
to the tape IN THAT DIRECTION:

    r = 0     -> center of the field
    r = 1     -> at the tape (inner edge), in every direction
    r > 1     -> outside

so every behavior-tree threshold ("too close to the edge" = r > 0.75,
"pushed out" = r > 1.15) means the same thing on a long narrow field as on a
round one. Positions, headings and bearings stay in real camera pixels, so
steering angles are exact; field.scale (the radius of a circle with the
field's area) turns pixel distances into "field sizes".

Bearings: signed degrees from our heading to a target, in (-180, 180].
Positive = target is to the robot's RIGHT (seen from above).
"""

import math

import cv2
import numpy as np

from config import CAMERA_MIRRORED

N_BINS = 180                     # directions in the radial profile (every 2 degrees)
_ANGLES = np.linspace(-np.pi, np.pi, N_BINS, endpoint=False)


class Field:
    def __init__(self, cx, cy, radii):
        self.cx, self.cy = float(cx), float(cy)
        self.radii = np.asarray(radii, float)              # distance to the tape per direction bin
        pts = self.polygon(1.0)
        self.area = float(cv2.contourArea(pts.astype(np.float32)))
        self.scale = math.sqrt(self.area / math.pi)         # radius of a circle with the same area

    # ------------------------------------------------------------ building
    @classmethod
    def from_contour(cls, contour):
        """Field from a closed boundary (Nx2 pixels): center = its centroid,
        radius per direction = distance to the NEAREST boundary point that way."""
        pts = np.asarray(contour, float).reshape(-1, 2)
        m = cv2.moments(pts.astype(np.float32))
        if m["m00"] <= 0:
            return None
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # densify so every direction bin gets boundary points
        dense = []
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            n = max(1, int(np.hypot(*(b - a)) // 2))
            dense.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        pts = np.vstack(dense)
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        bins = ((ang + np.pi) / (2 * np.pi) * N_BINS).astype(int) % N_BINS
        radii = np.full(N_BINS, np.inf)
        np.minimum.at(radii, bins, d)
        have = np.isfinite(radii)
        if have.sum() < N_BINS // 3:
            return None
        if not have.all():                                  # fill empty directions, wrapping round
            idx = np.arange(N_BINS)
            radii = np.interp(idx, idx[have], radii[have], period=N_BINS)
        return cls(cx, cy, radii)

    def blend(self, other, w):
        """EMA toward another measurement of the same field (vision smoothing)."""
        return Field(self.cx + w * (other.cx - self.cx), self.cy + w * (other.cy - self.cy),
                     self.radii + w * (other.radii - self.radii))

    def similar(self, other, tol=0.12):
        return (math.hypot(self.cx - other.cx, self.cy - other.cy) < 1.5 * tol * self.scale
                and float(np.mean(np.abs(self.radii - other.radii))) < tol * self.scale)

    # ------------------------------------------------------------ geometry
    def _radius_toward(self, ang):
        f = (np.asarray(ang) + np.pi) / (2 * np.pi) * N_BINS - 0.5      # bin centers at +0.5
        i0 = np.floor(f).astype(int) % N_BINS
        t = f - np.floor(f)
        return self.radii[i0] * (1 - t) + self.radii[(i0 + 1) % N_BINS] * t

    def radius_at(self, x, y):
        """Normalized radius: 0 center, 1 on the tape, >1 outside (any shape)."""
        dx, dy = x - self.cx, y - self.cy
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return 0.0
        return d / max(float(self._radius_toward(math.atan2(dy, dx))), 1e-6)

    def radius_at_many(self, xs, ys):
        dx, dy = np.asarray(xs, float) - self.cx, np.asarray(ys, float) - self.cy
        return np.hypot(dx, dy) / np.maximum(self._radius_toward(np.arctan2(dy, dx)), 1e-6)

    def polygon(self, k=1.0):
        """Boundary scaled by k toward the center (k=0.75 -> the r=0.75 line). Nx2 pixels."""
        a = _ANGLES + np.pi / N_BINS                        # bin centers
        return np.stack([self.cx + k * self.radii * np.cos(a), self.cy + k * self.radii * np.sin(a)], axis=1)

    def mask(self, shape, k=1.0, downscale=1):
        """uint8 mask (255 inside the r<k region) for an image of `shape` (h, w)."""
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(self.polygon(k) / downscale).astype(np.int32)], 255)
        return m

    @property
    def center(self):
        return self.cx, self.cy

    def as_dict(self):
        poly = self.polygon(1.0)[::2]                       # 90 points is plenty for drawing
        return {"cx": self.cx, "cy": self.cy, "scale": self.scale,
                "poly": np.round(poly, 1).tolist()}


def bearing_deg(heading, origin, target):
    """Signed angle from unit vector `heading` (at `origin`) to `target`, all in
    camera pixels. + = target on the robot's right."""
    tx, ty = target[0] - origin[0], target[1] - origin[1]
    if tx == 0 and ty == 0:
        return 0.0
    hx, hy = heading
    # image y points DOWN, so a positive cross product is clockwise on screen =
    # a right turn for a robot seen from above.
    ang = math.degrees(math.atan2(hx * ty - hy * tx, hx * tx + hy * ty))
    return -ang if CAMERA_MIRRORED else ang


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0
