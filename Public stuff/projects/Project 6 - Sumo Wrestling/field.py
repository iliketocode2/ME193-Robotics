"""
field.py -- ring geometry: image pixels <-> "ring coordinates".

The overhead phone sees the tape circle as an ellipse (camera tilt, imperfect
tape). cv2.fitEllipse gives ((cx, cy), (w, h), angle). Ring coordinates undo
that: translate to the ellipse center, rotate by -angle, and scale each axis
by its semi-axis, so the tape is ALWAYS the unit circle:

    r = 0     -> dead center of the ring
    r = 1     -> on the tape
    r > 1     -> outside

That makes every threshold in the behavior tree ("too close to the edge",
"pushed out") one number that works for any ring size or camera height.
Directions (our heading, the direction to the opponent) are mapped through
the same linear part of the transform.

Bearings: signed degrees from our heading to a target, in (-180, 180].
Positive = target is to the robot's RIGHT (seen from above).
"""

import math

from config import CAMERA_MIRRORED


class Ring:
    def __init__(self, cx, cy, w, h, angle):
        self.cx, self.cy, self.w, self.h, self.angle = cx, cy, w, h, angle
        a = math.radians(angle)
        self._cos, self._sin = math.cos(a), math.sin(a)
        self._ax, self._ay = max(w / 2.0, 1e-6), max(h / 2.0, 1e-6)

    @classmethod
    def from_fit(cls, fit):
        (cx, cy), (w, h), angle = fit
        return cls(cx, cy, w, h, angle)

    def as_dict(self):
        return {"cx": self.cx, "cy": self.cy, "w": self.w, "h": self.h, "angle": self.angle}

    # pixel vector -> ring vector (linear part only)
    def _vec_to_ring(self, dx, dy):
        u = dx * self._cos + dy * self._sin
        v = -dx * self._sin + dy * self._cos
        return u / self._ax, v / self._ay

    def to_ring(self, x, y):
        return self._vec_to_ring(x - self.cx, y - self.cy)

    def dir_to_ring(self, dx, dy):
        u, v = self._vec_to_ring(dx, dy)
        n = math.hypot(u, v) or 1.0
        return u / n, v / n

    def to_pixel(self, u, v):
        dx, dy = u * self._ax, v * self._ay
        return (self.cx + dx * self._cos - dy * self._sin,
                self.cy + dx * self._sin + dy * self._cos)

    def radius_at(self, x, y):
        return math.hypot(*self.to_ring(x, y))


def bearing_deg(heading, origin, target):
    """Signed angle from unit vector `heading` (at `origin`) to `target`,
    all in ring coordinates. + = target on the robot's right."""
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
