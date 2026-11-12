"""Builds hardware/assembly_viewer.html, an interactive 3D view of the box and stand, from make_parts.py.

    python hardware/export_viewer.py

The page is viewer_template.html with the part geometry dropped in, so after changing a size in make_parts.py,
run this again and the viewer matches the new DXFs.
"""

import json
from pathlib import Path

import make_parts as m

HERE = Path(__file__).resolve().parent


def area(loop):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(loop, loop[1:] + loop[:1]))) / 2


def r(v):
    return round(float(v), 3)


def part_data(key, p, group, sheet):
    loops = p.cut_loops()
    outer = max(loops, key=area)
    holes = [lp for lp in loops if lp is not outer]
    w, h = m.bbox(p)
    return {
        "key": f"{group}_{key}", "name": p.name, "group": group, "qty": p.qty,
        "frame": None if p.frame is None else [[r(c) for c in p.frame[0]], p.frame[1], p.frame[2]],
        "outer": [[r(x), r(y)] for x, y in outer],
        "holes": [[[r(x), r(y)] for x, y in lp] for lp in holes],
        "circles": [[r(c[0]), r(c[1]), r(c[2])] for c in p.circles],
        "size": [r(w), r(h)], "sheet": sheet,
    }


def main():
    out = {"T": m.T, "W": m.W, "D": m.D, "H": m.H, "parts": [],
           "latch": {"shaft": [m.SHAFT_X, m.SHAFT_Y], "armZ": m.ARM_Z, "shelfZ": m.SHELF_Z,
                     "gearbox": [m.SHAFT_X, m.SHAFT_Y + m.SHAFT_OFFSET, m.GEARBOX_D, m.MOTOR_LEN],
                     "switchLocked": m.SWITCH_LOCKED, "switchOpen": m.SWITCH_OPEN, "catchY": m.CATCH_Y},
           "front": {"ledZ": m.LED_Z, "digitX": m.DIGIT_LED_X, "greenX": m.GREEN_LED_X, "redX": m.RED_LED_X,
                     "button": [m.BUTTON_X, m.BUTTON_Z]},
           "falseFloorZ": m.FALSE_FLOOR_Z,
           "stand": {"hand": [m.HAND_X, m.HAND_Y], "camHeight": m.CAM_HEIGHT, "camDrop": m.CAM_DROP,
                     "base": [m.BASE_W, m.BASE_D]}}
    for group, parts in (("box", m.box_parts()), ("stand", m.stand_parts())):
        sheet_of = {}
        for k, sheet in enumerate(m.pack(list(parts.values())), 1):
            for p, *_ in sheet:
                sheet_of[id(p)] = k
        for key, p in parts.items():
            out["parts"].append(part_data(key, p, group, sheet_of[id(p)]))
    html = (HERE / "viewer_template.html").read_text(encoding="utf-8")
    html = html.replace("/*PART_DATA*/null", json.dumps(out, separators=(",", ":")))
    (HERE / "assembly_viewer.html").write_text(html, encoding="utf-8")
    print("wrote", HERE / "assembly_viewer.html", f"({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
