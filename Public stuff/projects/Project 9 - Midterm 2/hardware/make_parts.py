"""Laser-cut parts for the lock box and the camera stand, written as DXF files (millimeters).

    python hardware/make_parts.py

Writes hardware/dxf/parts/*.dxf (one file per part), hardware/dxf/sheet_*.dxf (the parts packed onto 300 x 300 mm
sheets of 3 mm plywood) and preview images in hardware/preview/. Every size is a setting below, so measure the
motor and the plywood, change the numbers, and run it again.

Layers: CUT (red) is cut through, ENGRAVE (blue) is a light score line for lining things up.

World coordinates for the box: x runs left to right as you face the front, y runs from the front face (0) to the
back, z runs up from the bottom of the floor. The box is W x D, the walls are H tall, and the lid sits on top.
"""

import math
from pathlib import Path

import numpy as np

# Material ----------------------------------------------------------------------------------------------------------
T = 3.0                 # plywood thickness; measure yours with calipers, sheets are often 2.8 to 3.2
FINGER = 12.0           # target finger-joint width
SHEET = 300.0           # sheet size for the packed files (a 12 x 12 in sheet with a little margin)

# Box ---------------------------------------------------------------------------------------------------------------
W, D, H = 127.0, 127.0, 105.0       # 5 x 5 in footprint; walls 105 tall, plus the 3 mm lid = 108 mm (4.25 in)
LED_Z = 62.0                         # height of the LED row on the front face
LED_D = 5.0                          # hole for a 5 mm LED
DIGIT_LED_X = (80.0, 90.0, 100.0)    # three yellow digit LEDs
GREEN_LED_X, RED_LED_X = 110.0, 118.0
BUTTON_X, BUTTON_Z, BUTTON_D = 72.0, 35.0, 12.2   # 12 mm panel-mount push button
CABLE_HOLE_D = 20.0                  # back wall (power) and left wall (webcam)
FALSE_FLOOR_Z = 38.0                 # electronics live under this removable floor

# Motor: Geartisan ZGB37RG (37 mm gearbox, off-center output shaft). CHECK THESE WITH CALIPERS.
GEARBOX_D = 37.0
MOTOR_LEN = 70.0        # gearbox face to the back of the motor, including the solder tabs
SHAFT_OFFSET = 7.0      # distance from the gearbox center to the output shaft
BOSS_D = 12.0           # raised ring around the shaft on the gearbox face
SHAFT_D, SHAFT_FLAT = 6.0, 5.5   # D-shaft: diameter, and the distance from the flat to the far side
SCREW_PCD, SCREW_D = 31.0, 3.4   # M3 mounting holes on this circle around the gearbox center
SCREW_ANGLES = (0, 60, 120, 180, 240, 300)

# Latch layout (world coordinates, mm)
TOWER_X = 60.0          # the motor tower fills the front-right corner: x from here to the right wall,
TOWER_Y = 62.0          # and y from the front wall to here
SHAFT_X, SHAFT_Y = 100.0, 32.0
SHELF_Z = 80.0          # bottom of the shelf the motor hangs from
ARM_TIP, ARM_TAIL, ARM_W, HUB_R = 27.0, 20.0, 10.0, 9.0
ARM_Z = SHELF_Z + T + 3.0           # bottom of the 2-layer arm (the boss holds it 2 to 3 mm above the shelf)
CATCH_Y = SHAFT_Y - 22.0            # center plane of the catch that hangs from the lid
OVERSHOOT_DEG = 8.0                 # how far past "locked" the arm might coast
# Limit switches (KW11-3Z style, 20 x 10 x 6.4 mm), as (x0, y0, x1, y1) on the shelf
SWITCH_LOCKED = (84.0, 40.0, 94.0, 60.0)   # the arm's tail presses it when the arm points at the front
SWITCH_OPEN = (62.0, 38.0, 82.0, 48.0)     # the arm presses it when it points left

# Camera stand ------------------------------------------------------------------------------------------------------
CAM_HEIGHT = 254.0      # lens above the base, 10 in
CAM_DROP = 20.0         # how far the lens hangs below the arm; measure your webcam on its clip
BASE_W, BASE_D = 200.0, 220.0
HAND_X, HAND_Y = 100.0, 92.0        # point under the camera
COL = 30.0                          # square column, outside size

OUT = Path(__file__).resolve().parent
RED, BLUE = 1, 5                    # DXF colors


# Geometry helpers --------------------------------------------------------------------------------------------------

def odd_count(length):
    n = max(3, int(round(length / FINGER)))
    return n if n % 2 else n + 1


def finger_notches(length, notch_even, band, along_u, at):
    """Rectangles to remove along one edge. band=(lo, hi) is the strip across the edge."""
    n = odd_count(length)
    seg = length / n
    rects = []
    for k in range(n):
        if (k % 2 == 0) == notch_even:
            a, b = k * seg, (k + 1) * seg
            rects.append((a, band[0], b, band[1]) if along_u else (band[0], a, band[1], b))
    return rects


def region_loops(add, sub):
    """Outline of (union of add rectangles) minus (union of sub rectangles), as closed loops."""
    add = [tuple(round(c, 4) for c in r) for r in add]     # so 8.9233...797 and 8.9233...804 are one line
    sub = [tuple(round(c, 4) for c in r) for r in sub]
    us = sorted({r[0] for r in add + sub} | {r[2] for r in add + sub})
    vs = sorted({r[1] for r in add + sub} | {r[3] for r in add + sub})

    def inside(r, u, v):
        return r[0] < u < r[2] and r[1] < v < r[3]

    nu, nv = len(us) - 1, len(vs) - 1
    filled = np.zeros((nu, nv), bool)
    for i in range(nu):
        for j in range(nv):
            u, v = (us[i] + us[i + 1]) / 2, (vs[j] + vs[j + 1]) / 2
            filled[i, j] = any(inside(r, u, v) for r in add) and not any(inside(r, u, v) for r in sub)

    def f(i, j):
        return 0 <= i < nu and 0 <= j < nv and filled[i, j]

    nxt = {}
    for i in range(nu):
        for j in range(nv):
            if not filled[i, j]:
                continue
            p00, p10, p11, p01 = (us[i], vs[j]), (us[i + 1], vs[j]), (us[i + 1], vs[j + 1]), (us[i], vs[j + 1])
            if not f(i, j - 1): nxt.setdefault(p00, []).append(p10)
            if not f(i + 1, j): nxt.setdefault(p10, []).append(p11)
            if not f(i, j + 1): nxt.setdefault(p11, []).append(p01)
            if not f(i - 1, j): nxt.setdefault(p01, []).append(p00)

    loops = []
    while nxt:
        start = next(iter(nxt))
        loop, p = [start], start
        while True:
            q = nxt[p].pop()
            if not nxt[p]:
                del nxt[p]
            if q == start:
                break
            loop.append(q)
            p = q
        loops.append(simplify(loop))
    return loops


def simplify(loop):
    out = []
    n = len(loop)
    for k in range(n):
        a, b, c = loop[k - 1], loop[k], loop[(k + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            out.append(b)
    return out


def circle_pts(cx, cy, r, n=64):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


class Part:
    """A flat part. frame = (origin xyz, u axis index, v axis index); the thickness goes along the third axis."""

    def __init__(self, name, frame=None, qty=1, group="box", color=(0.85, 0.72, 0.52)):
        self.name, self.frame, self.qty, self.group, self.color = name, frame, qty, group, color
        self.add, self.sub = [], []
        self.circles = []          # (cx, cy, d), cut
        self.polys = []            # extra closed loops to cut (non-rectangular outlines and holes)
        self.engrave = []          # closed loops to score

    def uv_rect(self, box):
        o, ua, va = self.frame
        return (box[ua][0] - o[ua], box[va][0] - o[va], box[ua][1] - o[ua], box[va][1] - o[va])

    def body(self, box):
        self.add.append(self.uv_rect(box))

    def cut_loops(self):
        loops = region_loops(self.add, self.sub) if self.add else []
        return loops + self.polys

    def to_world(self, loop, w):
        o, ua, va = self.frame
        na = 3 - ua - va
        pts = []
        for u, v in loop:
            p = [0.0, 0.0, 0.0]
            p[ua], p[va], p[na] = o[ua] + u, o[va] + v, o[na] + w
            pts.append(p)
        return pts


def joint(tabbed, slotted, box):
    """A tab on one part sitting in a matching slot in another. box is the tab in world coordinates."""
    tabbed.add.append(tabbed.uv_rect(box))
    slotted.sub.append(slotted.uv_rect(box))


# The box -----------------------------------------------------------------------------------------------------------

def box_parts():
    P = {}
    floor = P["floor"] = Part("floor", ((0, 0, 0), 0, 1))
    floor.body(((0, W), (0, D), (0, T)))
    floor.sub += finger_notches(W, True, (0, T), True, 0) + finger_notches(W, True, (D - T, D), True, 0)
    floor.sub += finger_notches(D, True, (0, T), False, 0) + finger_notches(D, True, (W - T, W), False, 0)

    for name, y0 in (("front", 0.0), ("back", D - T)):
        p = P[name] = Part(name + " wall", ((0, y0, 0), 0, 2))
        p.body(((0, W), (y0, y0 + T), (0, H)))
        p.sub += finger_notches(W, False, (0, T), True, 0)
        p.sub += finger_notches(H, False, (0, T), False, 0) + finger_notches(H, False, (W - T, W), False, 0)
    for name, x0 in (("left", 0.0), ("right", W - T)):
        p = P[name] = Part(name + " wall", ((x0, 0, 0), 1, 2))
        p.body(((x0, x0 + T), (0, D), (0, H)))
        p.sub += finger_notches(D, False, (0, T), True, 0)
        p.sub += finger_notches(H, True, (0, T), False, 0) + finger_notches(H, True, (D - T, D), False, 0)

    front, back, left, right = P["front"], P["back"], P["left"], P["right"]
    for x in DIGIT_LED_X + (GREEN_LED_X, RED_LED_X):
        front.circles.append((x, LED_Z, LED_D))
    front.circles.append((BUTTON_X, BUTTON_Z, BUTTON_D))
    back.circles.append((30.0, 20.0, CABLE_HOLE_D))          # 12 V and USB-C power in
    left.circles.append((100.0, 20.0, CABLE_HOLE_D))         # webcam cable from the stand

    # Motor tower: a partition on the left of it and a removable back panel.
    part = P["partition"] = Part("tower partition", ((TOWER_X - T, 0, 0), 1, 2))
    part.body(((TOWER_X - T, TOWER_X), (T, TOWER_Y + T), (T, H - 2)))
    for z0 in (15, 45):
        joint(part, front, ((TOWER_X - T, TOWER_X), (0, T), (z0, z0 + 15)))
    for y0 in (12, 40):
        joint(part, floor, ((TOWER_X - T, TOWER_X), (y0, y0 + 12), (0, T)))

    tb = P["tower_back"] = Part("tower back (removable)", ((0, TOWER_Y, 0), 0, 2))
    tb.body(((TOWER_X, W - T), (TOWER_Y, TOWER_Y + T), (T, H - 2)))
    for x0 in (68, 96):
        joint(tb, floor, ((x0, x0 + 12), (TOWER_Y, TOWER_Y + T), (0, T)))
    tb.circles.append((110.0, 18.0, 14.0))                   # wires from the tower to the electronics

    shelf = P["shelf"] = Part("motor shelf", ((0, 0, SHELF_Z), 0, 1))
    shelf.body(((TOWER_X, W - T), (18.0, TOWER_Y), (SHELF_Z, SHELF_Z + T)))
    for y0 in (24, 46):
        joint(shelf, right, ((W - T, W), (y0, y0 + 10), (SHELF_Z, SHELF_Z + T)))
        joint(shelf, part, ((TOWER_X - T, TOWER_X), (y0, y0 + 10), (SHELF_Z, SHELF_Z + T)))
    gx, gy = SHAFT_X, SHAFT_Y + SHAFT_OFFSET                 # gearbox center, behind the shaft
    shelf.circles.append((SHAFT_X, SHAFT_Y, BOSS_D + 0.5))
    screws = [(gx + SCREW_PCD / 2 * math.cos(math.radians(a)), gy + SCREW_PCD / 2 * math.sin(math.radians(a)))
              for a in SCREW_ANGLES]
    for sx, sy in screws:
        shelf.circles.append((sx, sy, SCREW_D))
    shelf.circles.append((116.0, 56.0, 8.0))                 # limit-switch wires
    for r in (SWITCH_LOCKED, SWITCH_OPEN):                   # score where the switches go
        shelf.engrave.append([(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])])

    # Spacers lift the switches to the arm's height, with room for the motor screw heads underneath.
    for key, r in (("spacer_locked", SWITCH_LOCKED), ("spacer_open", SWITCH_OPEN)):
        sp = P[key] = Part(f"switch spacer ({key.split('_')[1]})", ((r[0], r[1], SHELF_Z + T), 0, 1))
        sp.add = [(0, 0, r[2] - r[0], r[3] - r[1])]
        for sx, sy in screws:              # square pocket over any screw head under the spacer
            u, v = sx - r[0], sy - r[1]
            sp.sub.append((u - 3.5, v - 3.5, u + 3.5, v + 3.5))

    # Catch: hangs from the lid; the arm swings through its window.
    x0w, x1w = arm_window()
    z0w, z1w = ARM_Z - 1.5, ARM_Z + 2 * T + 1.5
    catch = P["catch"] = Part("lid catch", ((0, CATCH_Y - T / 2, 0), 0, 2))
    cx0, cx1 = x0w - 7, x1w + 7
    catch.body(((cx0, cx1), (CATCH_Y - T / 2, CATCH_Y + T / 2), (z0w - 5, H)))
    catch.sub.append(catch.uv_rect(((x0w, x1w), (0, 1), (z0w, z1w))))

    lid = P["lid"] = Part("lid", ((0, 0, H), 0, 1))
    lid.body(((0, W), (0, D), (H, H + T)))
    for a in (cx0 + 3, cx1 - 13):
        joint(catch, lid, ((a, a + 10), (CATCH_Y - T / 2, CATCH_Y + T / 2), (H, H + T)))

    # Removable false floor over the electronics, resting on cleats glued to the walls.
    ff = P["false_floor"] = Part("false floor", ((0, 0, FALSE_FLOOR_Z), 0, 1))
    ff.add = [(T + 0.5, T + 0.5, W - T - 0.5, D - T - 0.5)]
    ff.sub = [(TOWER_X - T - 0.5, 0, W, TOWER_Y + T + 0.5)]
    ff.circles.append((25.0, 100.0, 22.0))                   # finger hole to lift it out
    cleat = P["cleat"] = Part("cleat (cut 6)", None, qty=6)   # glue 2 on the left wall, 2 on the back wall,
    cleat.add = [(0, 0, 30, 8)]                               # 1 on the partition, 1 on the tower back

    # Latch arm: two identical layers glued together on the D-shaft.
    arm = P["arm"] = Part("latch arm (cut 2)", None, qty=2)
    arm.polys = [arm_outline(), d_hole()]
    return P


def arm_outline(n=360):
    """Stadium from -TAIL to +TIP along +u, width ARM_W, joined to a round hub. Star-shaped about the shaft."""
    r = ARM_W / 2

    def inside(u, v):
        c = min(max(u, -ARM_TAIL + r), ARM_TIP - r)
        return (u - c) ** 2 + v ** 2 <= r * r or u * u + v * v <= HUB_R ** 2

    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        lo, hi = 0.0, ARM_TIP + 5
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if inside(mid * math.cos(a), mid * math.sin(a)) else (lo, mid)
        pts.append((lo * math.cos(a), lo * math.sin(a)))
    return pts


def d_hole(n=48):
    """D-shaped hole for the shaft, flat side toward -v."""
    r = SHAFT_D / 2 + 0.05
    flat = SHAFT_FLAT - r            # distance from center to the flat
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        x, y = r * math.cos(a), r * math.sin(a)
        pts.append((x, max(y, -flat)))
    return simplify(pts)


def arm_points(alpha_deg):
    """World x, y of points filling the arm at angle alpha (0 = pointing at the front, 90 = pointing left)."""
    a = math.radians(alpha_deg)
    du = np.array([-math.sin(a), -math.cos(a)])
    dv = np.array([du[1], -du[0]])
    s = np.arange(-ARM_TAIL, ARM_TIP + 0.01, 0.25)
    t = np.arange(-ARM_W / 2, ARM_W / 2 + 0.01, 0.25)
    S, Tt = np.meshgrid(s, t)
    r = ARM_W / 2
    c = np.clip(S, -ARM_TAIL + r, ARM_TIP - r)
    ok = (S - c) ** 2 + Tt ** 2 <= r * r
    S, Tt = S[ok], Tt[ok]
    return SHAFT_X + S * du[0] + Tt * dv[0], SHAFT_Y + S * du[1] + Tt * dv[1]


def arm_window(margin=1.5):
    """x range the arm passes through at the catch plate, over its whole swing."""
    xs = []
    for alpha in np.arange(-OVERSHOOT_DEG, 95.0, 0.5):
        x, y = arm_points(alpha)
        hit = np.abs(y - CATCH_Y) <= T / 2 + 0.3
        xs.extend(x[hit])
    return min(xs) - margin, max(xs) + margin


# The camera stand --------------------------------------------------------------------------------------------------

def stand_parts():
    P = {}
    top = T + CAM_HEIGHT + CAM_DROP          # underside of the arm
    base = P["base"] = Part("stand base", ((0, 0, 0), 0, 1), group="stand")
    base.body(((0, BASE_W), (0, BASE_D), (0, T)))
    base.engrave += [[(HAND_X - 15, HAND_Y - 0.25), (HAND_X + 15, HAND_Y - 0.25), (HAND_X + 15, HAND_Y + 0.25),
                      (HAND_X - 15, HAND_Y + 0.25)],
                     [(HAND_X - 0.25, HAND_Y - 15), (HAND_X + 0.25, HAND_Y - 15), (HAND_X + 0.25, HAND_Y + 15),
                      (HAND_X - 0.25, HAND_Y + 15)]]

    cx0, cx1 = HAND_X - COL / 2, HAND_X + COL / 2
    cy1 = BASE_D
    cy0 = cy1 - COL
    arm = P["cam_arm"] = Part("camera arm", ((0, 0, top), 0, 1), group="stand")
    arm.body(((cx0 - 5, cx1 + 5), (HAND_Y - 17, cy1), (top, top + T)))
    arm.circles.append((HAND_X, HAND_Y, 6.5))              # 1/4 in tripod screw
    for dx in (-18, 18):                                   # zip-tie slots
        arm.sub.append((HAND_X + dx - 1.25, HAND_Y - 3, HAND_X + dx + 1.25, HAND_Y + 3))

    for name, y0 in (("front", cy0), ("back", cy1 - T)):
        p = P["col_" + name] = Part(f"column {name}", ((0, y0, 0), 0, 2), group="stand")
        p.body(((cx0, cx1), (y0, y0 + T), (T, top)))
        joint(p, base, ((HAND_X - 5, HAND_X + 5), (y0, y0 + T), (0, T)))
        joint(p, arm, ((HAND_X - 5, HAND_X + 5), (y0, y0 + T), (top, top + T)))
    for name, x0 in (("left", cx0), ("right", cx1 - T)):
        p = P["col_" + name] = Part(f"column {name} side", ((x0, 0, 0), 1, 2), group="stand")
        p.body(((x0, x0 + T), (cy0 + T, cy1 - T), (T, top)))
        joint(p, base, ((x0, x0 + T), (cy0 + 7, cy0 + 17), (0, T)))

    # Gusset under the arm, glued to the column's front face.
    g = P["gusset"] = Part("arm gusset", ((HAND_X - T / 2, 0, 0), 1, 2), group="stand")
    gy0 = cy0 - 60
    g.polys = [[(gy0, top), (gy0 + 10, top), (gy0 + 10, top + T), (gy0 + 22, top + T), (gy0 + 22, top),
                (gy0 + 32, top), (gy0 + 32, top + T), (gy0 + 44, top + T), (gy0 + 44, top), (cy0, top),
                (cy0, top - 27), (cy0 + T, top - 27), (cy0 + T, top - 42), (cy0, top - 42), (cy0, top - 60)]]
    for a in (gy0 + 10, gy0 + 32):
        arm.sub.append(arm.uv_rect(((HAND_X - T / 2, HAND_X + T / 2), (a, a + 12), (top, top + T))))
    P["col_front"].sub.append(P["col_front"].uv_rect(((HAND_X - T / 2, HAND_X + T / 2), (cy0, cy0 + T),
                                                      (top - 42, top - 27))))
    return P


# DXF output --------------------------------------------------------------------------------------------------------

def dxf_text(entities):
    """entities: list of ("loop", pts, color, layer) or ("circle", (x, y, d), color, layer). DXF R12."""
    out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "9", "$INSUNITS", "70", "4",
           "0", "ENDSEC", "0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER", "70", "2",
           "0", "LAYER", "2", "CUT", "70", "0", "62", str(RED), "6", "CONTINUOUS",
           "0", "LAYER", "2", "ENGRAVE", "70", "0", "62", str(BLUE), "6", "CONTINUOUS",
           "0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for kind, data, color, layer in entities:
        if kind == "circle":
            x, y, d = data
            out += ["0", "CIRCLE", "8", layer, "62", str(color), "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0",
                    "40", f"{d / 2:.4f}"]
        else:
            out += ["0", "POLYLINE", "8", layer, "62", str(color), "66", "1", "70", "1"]
            for x, y in data:
                out += ["0", "VERTEX", "8", layer, "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0"]
            out += ["0", "SEQEND", "8", layer]
    out += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(out) + "\n"


def part_entities(p, dx=0.0, dy=0.0, flip=False):
    """The part's entities moved so its bounding box starts at (dx, dy); flip turns it 90 degrees (no mirroring)."""
    loops = p.cut_loops()
    allpts = [pt for lp in loops for pt in lp]
    u0, v0 = min(q[0] for q in allpts), min(q[1] for q in allpts)
    v1 = max(q[1] for q in allpts)

    def tr(u, v):
        return ((v1 - v) + dx, (u - u0) + dy) if flip else ((u - u0) + dx, (v - v0) + dy)

    ents = [("loop", [tr(*q) for q in lp], RED, "CUT") for lp in loops]
    ents += [("circle", (*tr(c[0], c[1]), c[2]), RED, "CUT") for c in p.circles]
    ents += [("loop", [tr(*q) for q in lp], BLUE, "ENGRAVE") for lp in p.engrave]
    return ents


def bbox(p):
    pts = [q for lp in p.cut_loops() for q in lp]
    return max(q[0] for q in pts) - min(q[0] for q in pts), max(q[1] for q in pts) - min(q[1] for q in pts)


def pack(parts, gap=4.0, margin=5.0):
    """Shelf packing onto SHEET x SHEET sheets. Returns a list of sheets, each a list of (part, x, y, flip)."""
    items = []
    for p in parts:
        w, h = bbox(p)
        flip = h > w            # lay long parts down so the rows stay short
        for _ in range(p.qty):
            items.append((p, *((h, w) if flip else (w, h)), flip))
    items.sort(key=lambda it: (-it[2], -it[1]))
    # Each sheet is a stack of rows; a row is [y, height, next free x]. Small parts drop into the first row,
    # on any sheet, that still has room for them, so leftover space gets used.
    sheets, rows = [], []
    for p, w, h, flip in items:
        for k, sheet_rows in enumerate(rows):
            spot = next((r for r in sheet_rows if r[2] + w <= SHEET - margin and h <= r[1]), None)
            if spot is None:
                top = sheet_rows[-1][0] + sheet_rows[-1][1] + gap
                if top + h <= SHEET - margin:
                    spot = [top, h, margin]
                    sheet_rows.append(spot)
            if spot is not None:
                sheets[k].append((p, spot[2], spot[0], flip))
                spot[2] += w + gap
                break
        else:
            rows.append([[margin, h, margin + w + gap]])
            sheets.append([(p, margin, margin, flip)])
    return sheets


# Previews ----------------------------------------------------------------------------------------------------------

def preview_3d(parts, path, title, motor=False, hide=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    for key, p in parts.items():
        if p.frame is None or key in hide:
            continue
        loops = p.cut_loops()
        outer = max(loops, key=lambda lp: abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(lp, lp[1:] + lp[:1]))))
        for lp in (outer,):
            for w in (0.0, T):
                poly = Poly3DCollection([p.to_world(lp, w)], facecolor=p.color, edgecolor="k", linewidths=0.3,
                                        alpha=0.55)
                ax.add_collection3d(poly)
    if motor:
        gx, gy = SHAFT_X, SHAFT_Y + SHAFT_OFFSET
        th = np.linspace(0, 2 * np.pi, 40)
        for z in (SHELF_Z - MOTOR_LEN, SHELF_Z):
            ax.plot(gx + GEARBOX_D / 2 * np.cos(th), gy + GEARBOX_D / 2 * np.sin(th), z, color="0.3")
        x, y = arm_points(0)
        ax.scatter(x[::7], y[::7], ARM_Z + T, s=1, color="tab:red")
    lim = max(W, D, H) if motor else max(BASE_W, BASE_D, CAM_HEIGHT + 40)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_zlim(0, lim)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.view_init(elev=28, azim=-60)
    ax.set_title(title)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def preview_latch(path):
    """Top view of the shelf: the motor, both arm positions, the catch window and the switches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_patch(Rectangle((T, 0), W - 2 * T, T, color="0.6"))
    ax.text(W / 2, -4, "front wall", ha="center", va="top", fontsize=8)
    ax.add_patch(Rectangle((TOWER_X, 18), W - T - TOWER_X, TOWER_Y - 18, fill=False, ls="--", color="0.4"))
    ax.text(TOWER_X + 1, TOWER_Y + 1, "shelf", fontsize=8, color="0.3")
    gx, gy = SHAFT_X, SHAFT_Y + SHAFT_OFFSET
    ax.add_patch(Circle((gx, gy), GEARBOX_D / 2, fill=False, color="0.3", ls=":"))
    ax.text(gx + 12, gy - 14, "gearbox\n(under shelf)", fontsize=7, color="0.3")
    x0w, x1w = arm_window()
    ax.add_patch(Rectangle((x0w - 7, CATCH_Y - T / 2), x1w - x0w + 14, T, color="tab:brown"))
    ax.add_patch(Rectangle((x0w, CATCH_Y - T / 2), x1w - x0w, T, color="white", ec="tab:brown"))
    ax.text(x0w - 8, CATCH_Y, "catch\nwindow", ha="right", va="center", fontsize=8, color="tab:brown")
    for r, label in ((SWITCH_LOCKED, "switch:\nlocked"), (SWITCH_OPEN, "switch:\nopen")):
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], color="tab:blue", alpha=0.4))
        ax.text((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, label, ha="center", va="center", fontsize=7)
    for alpha, color, label in ((0, "tab:red", "locked"), (90, "tab:green", "open")):
        x, y = arm_points(alpha)
        ax.scatter(x, y, s=0.5, color=color, alpha=0.4)
        ax.plot([], [], color=color, lw=4, label=f"arm {label}")
    ax.plot(SHAFT_X, SHAFT_Y, "k+", ms=10)
    ax.set_xlim(TOWER_X - 10, W + 2); ax.set_ylim(-8, TOWER_Y + 8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm), front at the bottom")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Latch from above")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def preview_sheets(sheets, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, sheet in enumerate(sheets, 1):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, SHEET, SHEET, 0, 0], [0, 0, SHEET, SHEET, 0], color="0.7")
        for p, x, y, flip in sheet:
            for kind, data, color, _ in part_entities(p, x, y, flip):
                c = "tab:red" if color == RED else "tab:blue"
                if kind == "circle":
                    ax.add_patch(plt.Circle(data[:2], data[2] / 2, fill=False, color=c, lw=0.6))
                else:
                    xs, ys = zip(*(data + data[:1]))
                    ax.plot(xs, ys, color=c, lw=0.6)
            w, h = bbox(p)
            cx, cy = (x + h / 2, y + w / 2) if flip else (x + w / 2, y + h / 2)
            ax.text(cx, cy, p.name, ha="center", va="center", fontsize=5.5, color="0.25")
        ax.set_aspect("equal"); ax.set_xlim(-5, SHEET + 5); ax.set_ylim(-5, SHEET + 5)
        ax.set_title(f"{prefix} sheet {k} ({SHEET:.0f} x {SHEET:.0f} mm)")
        fig.savefig(OUT / "preview" / f"{prefix}_sheet_{k}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    (OUT / "dxf" / "parts").mkdir(parents=True, exist_ok=True)
    (OUT / "preview").mkdir(exist_ok=True)
    box, stand = box_parts(), stand_parts()
    for group, parts in (("box", box), ("stand", stand)):
        for key, p in parts.items():
            (OUT / "dxf" / "parts" / f"{group}_{key}.dxf").write_text(dxf_text(part_entities(p)))
        sheets = pack(list(parts.values()))
        for k, sheet in enumerate(sheets, 1):
            ents = [e for p, x, y, flip in sheet for e in part_entities(p, x, y, flip)]
            (OUT / "dxf" / f"{group}_sheet_{k}.dxf").write_text(dxf_text(ents))
        preview_sheets(sheets, group)
        print(f"{group}: {sum(p.qty for p in parts.values())} pieces on {len(sheets)} sheet(s)")

    preview_3d(box, OUT / "preview" / "box_3d.png", "Lock box, lid and false floor hidden", motor=True,
               hide=("lid", "false_floor"))
    preview_3d(stand, OUT / "preview" / "stand_3d.png", "Camera stand")
    preview_latch(OUT / "preview" / "latch_top.png")
    x0w, x1w = arm_window()
    print(f"catch window x {x0w:.1f} to {x1w:.1f} mm; arm tip clears the front wall by "
          f"{SHAFT_Y - ARM_TIP - T:.1f} mm; motor bottom sits {SHELF_Z - MOTOR_LEN - T:.1f} mm above the floor")


if __name__ == "__main__":
    main()
