"""Bottom tray for the shoe-mounted UNO Q, made to fit case_top_embossed.stl.

    pip install manifold3d numpy matplotlib
    python hardware/make_case.py            writes case_tray.stl, preview PNGs, and a fit report

Stack, from the shoe up (z = 0 is the bottom of the tray, flat on the print bed):

    0    - 2.4   floor, with a pocket for the MPU-6050 breakout sunk 1.2 mm into it
    2.4  - 7.4   standoffs at the four UNO mounting holes (space for the board's
                 bottom-side connectors and the IMU)
    7.4  - 9.0   the UNO Q (68.58 x 53.34 mm)
    7.4  - 14.0  the lid: its rim surrounds the board and rests on a ledge inside the
                 tray; the tray walls come up around it, leaving the lid 0.8 mm proud

Openings: the USB-C side (matching the lid's gap, sized for a plug body), and a
channel on the opposite side, at the lid's second gap, that takes the IMU cable from
under the board up to the board's top surface. A groove in the floor runs from the
IMU pocket to that channel.

Strap loops: both short ends, at shoe level, sloping smoothly up into the wall like
watch lugs. Each has a 26 x 4 mm slot with rounded ends and softened edges, so a
20-24 mm watch band or 1-inch masking tape threads through without catching. One
band through both slots and back over the lid holds the whole stack down.

Board coordinates: origin at the board's center, +x away from the USB-C edge,
z up from the shoe. Every dimension is a constant below.
"""

import struct
from pathlib import Path

import manifold3d as m3
import numpy as np

HERE = Path(__file__).resolve().parent
LID_STL = HERE.parent / "case_top_embossed.stl"
SEG = 64

# ---- the board and the existing lid (measured from case_top_embossed.stl) ----
BOARD_L, BOARD_W, BOARD_T = 68.58, 53.34, 1.6
LID_L, LID_W, LID_H = 71.58, 56.34, 6.6      # outer size; the rim is 1.0 mm thick
LID_RIM_INNER_L, LID_RIM_INNER_W = 69.58, 54.34
LID_CENTER_IN_STL = (125.0, 110.0)
USB_GAP_Y = (6.9, 15.55)                      # lid's USB-C gap, board coordinates
RIGHT_GAP_Y = (-15.25, -9.3)                  # lid's other gap (probably Qwiic)
# Uno-standard mounting holes, converted to board-centered coordinates. CHECK THESE
# against your board before printing (lay the board on a printout of preview_top.png).
HOLES = [(13.97, 2.54), (15.24, 50.8), (66.04, 7.62), (66.04, 35.56)]
HOLES = [(x - BOARD_L / 2, y - BOARD_W / 2) for x, y in HOLES]

# ---- the tray ----
FLOOR = 2.4
STANDOFF_H = 5.0            # board bottom sits this far above the floor
STANDOFF_D = 6.0
PILOT_D = 2.6               # for M3 self-tapping screws (2.5 mm for M2.5 into a tapped hole)
FIT = 0.2                   # clearance between the lid and the tray wall, per side
WALL = 2.0
LID_PROUD = 0.8             # lid sticks up this much above the tray walls, to grab it
CORNER_R = 4.0
BOTTOM_CHAMFER = 0.6

IMU_L, IMU_W, IMU_T = 21.2, 16.4, 1.6        # GY-521 breakout; change for another board
IMU_CLEAR = 0.3
IMU_DEPTH = 1.2                               # pocket depth into the floor
IMU_CENTER = (14.0, -12.3)                    # under the board, lined up with the cable channel
GROOVE_W = 4.0                                # wire groove from pocket to channel
CHANNEL_W = 7.0                               # cable channel width (y)
CHANNEL_DEPTH = 2.2                           # how far past the lid rim it cuts (x); leaves a 1 mm wall
USB_CUT_W, USB_CUT_FROM_Z = 12.0, 7.6         # USB-C plug body

SLOT_LEN, SLOT_W = 26.0, 4.0
EAR_GAP = 3.0               # from the outer wall to the slot
EAR_BAR = 5.0               # solid bar beyond the slot
EAR_T = 3.0                 # bar thickness
EAR_W = SLOT_LEN + 12.0     # width of the outer bar
EAR_CORNER_R = 6.0          # plan-view rounding of the lug's outer corners
LUG_FILLET = 8.0            # plan-view fillet where the lug flares into the wall
EAR_BLEND_H = 7.0           # height where the lug meets the wall (its top arcs down to EAR_T)
SLOT_SOFTEN = 0.8           # edge chamfer around the slot

# derived heights
BOARD_Z = FLOOR + STANDOFF_H
LID_TOP = BOARD_Z + LID_H
TRAY_TOP = LID_TOP - LID_PROUD
UPPER_L, UPPER_W = LID_L + 2 * FIT, LID_W + 2 * FIT
OUTER_L, OUTER_W = UPPER_L + 2 * WALL, UPPER_W + 2 * WALL


# ---------- helpers ----------

def rrect(l, w, r):
    """Rounded rectangle cross-section centered on the origin."""
    r = min(r, l / 2 - 1e-3, w / 2 - 1e-3)
    if r <= 0:
        return m3.CrossSection.square((l, w), center=True)
    return m3.CrossSection.square((l - 2 * r, w - 2 * r), center=True).offset(
        r, m3.JoinType.Round, circular_segments=SEG)


def prism(section, z0, z1, x=0.0, y=0.0):
    return m3.Manifold.extrude(section, z1 - z0).translate((x, y, z0))


def box(x0, x1, y0, y1, z0, z1):
    return m3.Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def stadium(length, width):
    """Slot shape: long axis along y, fully round ends."""
    return rrect(width, length, width / 2)


# ---------- parts ----------

def tray_body():
    body = m3.Manifold.hull(m3.Manifold.batch_hull([
        prism(rrect(OUTER_L - 2 * BOTTOM_CHAMFER, OUTER_W - 2 * BOTTOM_CHAMFER, CORNER_R - BOTTOM_CHAMFER), 0, 0.01),
        prism(rrect(OUTER_L, OUTER_W, CORNER_R), BOTTOM_CHAMFER, TRAY_TOP),
    ]))
    lower = box(-LID_RIM_INNER_L / 2, LID_RIM_INNER_L / 2, -LID_RIM_INNER_W / 2, LID_RIM_INNER_W / 2,
                FLOOR, BOARD_Z + 0.01)
    upper = box(-UPPER_L / 2, UPPER_L / 2, -UPPER_W / 2, UPPER_W / 2, BOARD_Z, TRAY_TOP + 1)
    return body - lower - upper


def ear(side):
    """Strap lug on the +x end (side=+1) or -x end (side=-1)."""
    wall_x = OUTER_L / 2
    tip_len = EAR_GAP + SLOT_W + EAR_BAR
    tip_x = wall_x + tip_len

    # plan view: the lug's rounded rectangle, joined to the tray wall with smooth concave
    # fillets (union, then grow and shrink by LUG_FILLET: a morphological "closing")
    tray_plan = rrect(OUTER_L, OUTER_W, CORNER_R)
    lug_rect = rrect(tip_len + 6.0, EAR_W, EAR_CORNER_R).translate((tip_x - (tip_len + 6.0) / 2, 0))
    joined = (tray_plan + lug_rect).offset(LUG_FILLET, m3.JoinType.Round, circular_segments=SEG) \
                                   .offset(-LUG_FILLET, m3.JoinType.Round, circular_segments=SEG)
    half = m3.CrossSection.square((tip_len + 20, OUTER_W + 40)).translate((wall_x - 3.0, -(OUTER_W + 40) / 2))
    plan = joined ^ half
    lug = prism(plan, 0, EAR_BLEND_H)

    # side view: one smooth arc from EAR_BLEND_H at the wall down to EAR_T at the tip,
    # cut by a long cylinder lying along y
    rise = EAR_BLEND_H - EAR_T
    d = tip_x - (wall_x - 1.0)
    r = (d * d + rise * rise) / (2 * rise)
    arc = m3.Manifold.cylinder(OUTER_W + 60, r, r, 4 * SEG).rotate((90, 0, 0)) \
        .translate((wall_x - 1.0, (OUTER_W + 60) / 2, EAR_BLEND_H - r))
    lug = lug ^ arc

    sx = wall_x + EAR_GAP + SLOT_W / 2
    slot = prism(stadium(SLOT_LEN, SLOT_W), -1, EAR_BLEND_H + 1, x=sx)
    s = SLOT_SOFTEN
    soften_bottom = m3.Manifold.batch_hull([
        prism(stadium(SLOT_LEN + 2 * s, SLOT_W + 2 * s), -0.01, 0.0, x=sx),
        prism(stadium(SLOT_LEN, SLOT_W), s, s + 0.01, x=sx)])
    soften_top = m3.Manifold.batch_hull([
        prism(stadium(SLOT_LEN, SLOT_W), EAR_T + 0.6, EAR_T + 0.61, x=sx),
        prism(stadium(SLOT_LEN + 2 * s, SLOT_W + 2 * s), EAR_BLEND_H + 1, EAR_BLEND_H + 1.01, x=sx)])
    lug = lug - slot - soften_bottom - soften_top
    # keep the lug out of the tray's inside
    lug = lug - box(-OUTER_L / 2 + WALL / 2, OUTER_L / 2 - WALL / 2, -OUTER_W, OUTER_W, FLOOR, 100)
    return lug if side > 0 else lug.mirror((1, 0, 0))


def standoffs():
    posts = [m3.Manifold.cylinder(STANDOFF_H + 0.01, STANDOFF_D / 2, STANDOFF_D / 2, SEG).translate((x, y, FLOOR - 0.01))
             for x, y in HOLES]
    return m3.Manifold.batch_boolean(posts, m3.OpType.Add)


def cuts():
    out = []
    for x, y in HOLES:      # pilot holes: down through the post into the floor, not through the bottom
        out.append(m3.Manifold.cylinder(STANDOFF_H + 1.0 + 0.02, PILOT_D / 2, PILOT_D / 2, SEG)
                   .translate((x, y, FLOOR - 1.0)))
    ix, iy = IMU_CENTER
    pl, pw = IMU_L + 2 * IMU_CLEAR, IMU_W + 2 * IMU_CLEAR
    out.append(prism(rrect(pl, pw, 0.6), FLOOR - IMU_DEPTH, FLOOR + 1, x=ix, y=iy))
    # wire groove from the pocket to the cable channel
    out.append(box(ix + pl / 2 - 1, LID_RIM_INNER_L / 2 + 0.5, iy - GROOVE_W / 2, iy + GROOVE_W / 2,
                   FLOOR - IMU_DEPTH, FLOOR + 1))
    # cable channel, up the +x wall at the lid's second gap
    cy = (RIGHT_GAP_Y[0] + RIGHT_GAP_Y[1]) / 2
    out.append(box(LID_RIM_INNER_L / 2 - 0.5, LID_RIM_INNER_L / 2 + CHANNEL_DEPTH, cy - CHANNEL_W / 2,
                   cy + CHANNEL_W / 2, FLOOR - IMU_DEPTH, TRAY_TOP + 1))
    # USB-C opening in the -x wall
    uy = (USB_GAP_Y[0] + USB_GAP_Y[1]) / 2
    out.append(box(-OUTER_L / 2 - 1, -LID_RIM_INNER_L / 2 + 0.5, uy - USB_CUT_W / 2, uy + USB_CUT_W / 2,
                   USB_CUT_FROM_Z, TRAY_TOP + 1))
    return m3.Manifold.batch_boolean(out, m3.OpType.Add)


def make_tray():
    tray = m3.Manifold.batch_boolean([tray_body(), standoffs(), ear(+1), ear(-1)], m3.OpType.Add)
    return tray - cuts()


# ---------- the lid, the board and the IMU, for the fit check ----------

def read_stl(path):
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    tri = np.frombuffer(b, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                        offset=84, count=n)["v"].astype(float)
    uniq, inv = np.unique(np.round(tri.reshape(-1, 3), 4), axis=0, return_inverse=True)
    return m3.Manifold(m3.Mesh(vert_properties=uniq.astype(np.float32),
                               tri_verts=inv.reshape(-1, 3).astype(np.uint32)))


def placed_lid():
    """The lid as printed (plate down) -> flipped over onto the board."""
    cx, cy = LID_CENTER_IN_STL
    return read_stl(LID_STL).rotate((180, 0, 0)).translate((-cx, cy, LID_TOP))


def board():
    return box(-BOARD_L / 2, BOARD_L / 2, -BOARD_W / 2, BOARD_W / 2, BOARD_Z, BOARD_Z + BOARD_T)


def imu():
    ix, iy = IMU_CENTER
    z0 = FLOOR - IMU_DEPTH
    return box(ix - IMU_L / 2, ix + IMU_L / 2, iy - IMU_W / 2, iy + IMU_W / 2, z0, z0 + IMU_T + 1.0)  # + chips


def write_stl(manifold, path):
    mesh = manifold.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    t = np.asarray(mesh.tri_verts)
    tri = v[t]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(t), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = normals, tri
    with open(path, "wb") as f:
        f.write(b"case_tray.stl from make_case.py".ljust(80, b" "))
        f.write(struct.pack("<I", len(t)))
        f.write(rec.tobytes())


def fit_report(tray, lid, brd, sensor):
    def overlap(a, b):
        return (a ^ b).volume()
    lines = [
        f"tray: {OUTER_L + 2 * (EAR_GAP + SLOT_W + EAR_BAR):.1f} x {OUTER_W:.1f} mm with lugs "
        f"({OUTER_L:.1f} x {OUTER_W:.1f} body), {TRAY_TOP:.1f} mm tall; lid top at {LID_TOP:.1f} mm",
        f"tray volume {tray.volume() / 1000:.1f} cm^3, genus {tray.genus()}, status {tray.status().name}",
        f"overlap lid/tray   {overlap(lid, tray):.3f} mm^3  (should be 0)",
        f"overlap board/tray {overlap(brd, tray):.3f} mm^3  (should be 0: board rests on the standoffs)",
        f"overlap board/lid  {overlap(brd, lid):.3f} mm^3  (should be 0)",
        f"overlap IMU/tray   {overlap(sensor, tray):.3f} mm^3  (should be 0)",
        f"overlap IMU/board  {overlap(sensor, brd):.3f} mm^3  (should be 0)",
        f"lid rim rests at z = {lid.bounding_box()[2]:.2f} on the ledge at z = {BOARD_Z:.2f}",
        f"gap from IMU top to board bottom: {BOARD_Z - (FLOOR - IMU_DEPTH + IMU_T + 1.0):.1f} mm",
        f"wall left outside the cable channel: {OUTER_L / 2 - (LID_RIM_INNER_L / 2 + CHANNEL_DEPTH):.2f} mm; "
        f"gap beside the board edge for the wires: {LID_RIM_INNER_L / 2 + CHANNEL_DEPTH - BOARD_L / 2:.1f} mm",
    ]
    return "\n".join(lines)


def preview(tray, lid, path_top, path_3d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def tris(man):
        mesh = man.to_mesh()
        return np.asarray(mesh.vert_properties)[:, :3][np.asarray(mesh.tri_verts)]

    t = tris(tray)
    fig, ax = plt.subplots(figsize=(9, 6))
    up = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2] > 1e-9
    z = t[up][:, :, 2].mean(1)
    o = np.argsort(z)
    pc = PolyCollection(t[up][o][:, :, :2], array=z[o], cmap="viridis", edgecolor="none")
    ax.add_collection(pc)
    ax.add_patch(plt.Rectangle((-BOARD_L / 2, -BOARD_W / 2), BOARD_L, BOARD_W, fill=False, ls="--", color="r",
                               label="UNO Q outline"))
    for x, y in HOLES:
        ax.add_patch(plt.Circle((x, y), 1.6, fill=False, color="r"))
    ax.annotate("USB-C", (-OUTER_L / 2, np.mean(USB_GAP_Y)), xytext=(-OUTER_L / 2 - 12, np.mean(USB_GAP_Y) + 14),
                arrowprops={"arrowstyle": "->"})
    ax.annotate("IMU pocket", IMU_CENTER, xytext=(IMU_CENTER[0] - 10, IMU_CENTER[1] + 22),
                arrowprops={"arrowstyle": "->"})
    ax.annotate("cable channel", (OUTER_L / 2 - 1, np.mean(RIGHT_GAP_Y)),
                xytext=(OUTER_L / 2 - 20, np.mean(RIGHT_GAP_Y) - 18), arrowprops={"arrowstyle": "->"})
    span = OUTER_L / 2 + EAR_GAP + SLOT_W + EAR_BAR + 3
    ax.set_xlim(-span, span)
    ax.set_ylim(-OUTER_W / 2 - 4, OUTER_W / 2 + 4)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("mm")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(pc, ax=ax, label="height above shoe (mm)", shrink=0.8)
    ax.set_title("Tray from above (red: board outline and mounting holes)")
    fig.tight_layout()
    fig.savefig(path_top, dpi=110)
    plt.close(fig)

    fig = plt.figure(figsize=(15, 6.5))
    light = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])
    views = [([(tray, (0.55, 0.62, 0.78))], "tray"),
             ([(tray, (0.55, 0.62, 0.78)), (lid.translate((0, 0, 22)), (0.92, 0.92, 0.9))], "exploded: lid above tray"),
             ([(tray, (0.55, 0.62, 0.78)), (lid, (0.92, 0.92, 0.9))], "assembled")]
    zmax = 42
    for k, (parts, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        polys, colors = [], []
        for man, color in parts:                # one collection, so faces sort together
            tt = tris(man)
            nrm = np.cross(tt[:, 1] - tt[:, 0], tt[:, 2] - tt[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            shade = 0.35 + 0.65 * np.clip(nrm @ light, 0, 1)
            polys.append(tt)
            colors.append(np.outer(shade, color))
        ax.add_collection3d(Poly3DCollection(np.concatenate(polys), facecolors=np.concatenate(colors),
                                             edgecolor="none"))
        ax.set_xlim(-span, span)
        ax.set_ylim(-OUTER_W / 2 - 2, OUTER_W / 2 + 2)
        ax.set_zlim(0, zmax)
        ax.set_box_aspect((2 * span, OUTER_W + 4, zmax))
        ax.view_init(30, -60)
        ax.set_axis_off()
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path_3d, dpi=110)
    plt.close(fig)


def main():
    tray = make_tray()
    lid = placed_lid()
    report = fit_report(tray, lid, board(), imu())
    write_stl(tray, HERE / "case_tray.stl")
    preview(tray, lid, HERE / "preview_top.png", HERE / "preview_3d.png")
    (HERE / "fit_report.txt").write_text(report + "\n")
    print(report)
    print(f"wrote {HERE / 'case_tray.stl'}")


if __name__ == "__main__":
    main()
