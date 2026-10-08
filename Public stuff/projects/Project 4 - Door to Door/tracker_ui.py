"""
tracker_ui.py -- the dashboard window for minifig_tracker.py.

Pure drawing code: it takes a camera frame plus the per-color state the
tracker already computed (the MQTT messages, detection boxes, timing
stats) and returns one BGR image to cv2.imshow(). No detection, policy,
or MQTT logic lives here.

Shapes are drawn with OpenCV (anti-aliased). Text goes through Pillow,
because OpenCV's built-in Hershey fonts can't render a modern UI font.
Pillow is slow per call, so each distinct string is rasterized ONCE into
an alpha mask and cached; every frame after that just alpha-blends the
cached mask with numpy. That keeps the whole dashboard to a few ms per
frame, so it doesn't eat into YOLO's frame budget.

Colors are BGR tuples throughout (OpenCV order).
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----- Palette (BGR) -----------------------------------------------------
BG = (24, 20, 17)            # #111418 page background
PANEL = (38, 32, 28)         # #1c2026 cards
PANEL_HI = (56, 49, 43)      # #2b313a tracks, inactive bars
BORDER = (66, 58, 52)        # #343a42 card outlines
TEXT = (245, 240, 235)       # #ebf0f5
MUTED = (160, 150, 140)      # #8c96a0
DIM = (110, 102, 95)         # #5f666e
OK = (153, 211, 52)          # #34d399 arrived
INFO = (250, 165, 96)        # #60a5fa driving
WARN = (36, 191, 251)        # #fbbf24 no target
LIVE_RED = (113, 113, 248)   # #f87171
LED_ON = (255, 175, 70)      # UNO Q matrix LEDs are blue
LED_OFF = (58, 46, 38)

ACCENTS = {"green": (94, 197, 34), "blue": (246, 130, 59)}   # #22c55e, #3b82f6

# ----- Layout ------------------------------------------------------------
PAD = 20
HEADER_H = 64
VIDEO_BOX = (820, 480)       # camera feed is fit (not stretched) inside this box
SIDEBAR_W = 380
CARD_R = 14                  # corner radius
COLOR_CARD_H = 180
TRACK_CARD_H = 84
PAYLOAD_CARD_H = 78
LED_CELL = 22

_FONT_FILES = {
    "regular": ["segoeui.ttf", "Arial.ttf", "DejaVuSans.ttf"],
    "semibold": ["seguisb.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"],
    "bold": ["segoeuib.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"],
    "mono": ["consola.ttf", "Menlo.ttc", "DejaVuSansMono.ttf"],
}


# ----- Drawing primitives ------------------------------------------------
def _rrect_raw(img, x1, y1, x2, y2, r, color, thickness):
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    t, aa = thickness, cv2.LINE_AA
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, t, aa)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, t, aa)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, t, aa)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, t, aa)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, t, aa)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, t, aa)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, aa)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, t, aa)


def _blend(img, x1, y1, x2, y2, alpha, draw):
    """Run draw(layer, dx, dy) on a copy of a region, then alpha-blend it back."""
    h, w = img.shape[:2]
    ax1, ay1 = max(0, x1 - 2), max(0, y1 - 2)
    ax2, ay2 = min(w, x2 + 3), min(h, y2 + 3)
    if ax2 <= ax1 or ay2 <= ay1:
        return
    roi = img[ay1:ay2, ax1:ax2]
    layer = roi.copy()
    draw(layer, -ax1, -ay1)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)


def rrect(img, x1, y1, x2, y2, r, color, thickness=-1, alpha=1.0):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if alpha >= 1.0:
        _rrect_raw(img, x1, y1, x2, y2, r, color, thickness)
    else:
        _blend(img, x1, y1, x2, y2, alpha,
               lambda L, dx, dy: _rrect_raw(L, x1 + dx, y1 + dy, x2 + dx, y2 + dy, r, color, thickness))


def circle(img, cx, cy, r, color, alpha=1.0):
    cx, cy, r = int(cx), int(cy), int(r)
    if alpha >= 1.0:
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        _blend(img, cx - r, cy - r, cx + r, cy + r, alpha,
               lambda L, dx, dy: cv2.circle(L, (cx + dx, cy + dy), r, color, -1, cv2.LINE_AA))


def dashed_vline(img, x, y1, y2, color, dash=8, gap=6):
    y = y1
    while y < y2:
        cv2.line(img, (x, y), (x, min(y + dash, y2)), color, 1, cv2.LINE_AA)
        y += dash + gap


def corner_box(img, x1, y1, x2, y2, color, t=3):
    """Corner brackets instead of a full rectangle -- reads cleaner over video."""
    L = int(max(8, min(22, (x2 - x1) / 3, (y2 - y1) / 3)))
    for (cx, cy, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + sx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + sy * L), color, t, cv2.LINE_AA)


class Dashboard:
    def __init__(self, center, deadband, led_cols, led_rows, broker, publish_hz, model_name):
        self.center = center
        self.deadband = deadband
        self.led_cols = led_cols
        self.led_rows = led_rows
        self.broker = broker
        self.publish_hz = publish_hz
        self.model_name = model_name

        self.left_w = VIDEO_BOX[0]
        self.width = PAD + self.left_w + PAD + SIDEBAR_W + PAD
        left_h = HEADER_H + 16 + VIDEO_BOX[1] + 16 + TRACK_CARD_H + 16 + PAYLOAD_CARD_H + PAD
        led_card_h = 44 + led_rows * LED_CELL + 34
        side_h = HEADER_H + 16 + 2 * COLOR_CARD_H + 2 * 16 + led_card_h + PAD
        self.height = max(left_h, side_h)

        self._fonts = {}
        self._texts = []
        self._glyphs = {}         # (string, size, weight, anchor) -> (dx, dy, alpha mask)
        self._video_mask = None   # cached rounded-corner mask for the current video size
        self._bg = np.full((self.height, self.width, 3), BG, np.uint8)

    # ----- text --------------------------------------------------------
    def _font(self, size, weight):
        key = (size, weight)
        if key not in self._fonts:
            font = None
            for name in _FONT_FILES[weight]:
                try:
                    font = ImageFont.truetype(name, size)
                    break
                except OSError:
                    continue
            self._fonts[key] = font or ImageFont.load_default(size)
        return self._fonts[key]

    def text(self, x, y, s, size=14, color=TEXT, weight="regular", anchor="ls"):
        """Queue text; anchor is Pillow's (e.g. 'ls' left-baseline, 'lm' left-middle, 'rm', 'mm')."""
        self._texts.append((int(x), int(y), str(s), size, weight, color, anchor))

    def text_w(self, s, size=14, weight="regular"):
        return int(self._font(size, weight).getlength(str(s)))

    def _glyph(self, s, size, weight, anchor):
        key = (s, size, weight, anchor)
        g = self._glyphs.get(key)
        if g is None:
            if len(self._glyphs) > 3000:      # changing numbers/payloads -- don't grow forever
                self._glyphs.clear()
            font = self._font(size, weight)
            x0, y0, x1, y1 = font.getbbox(s, anchor=anchor)
            mask = Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 0)
            ImageDraw.Draw(mask).text((-x0, -y0), s, font=font, fill=255, anchor=anchor)
            g = (x0, y0, np.asarray(mask, np.float32)[..., None] / 255.0)
            self._glyphs[key] = g
        return g

    def _flush_text(self, canvas):
        H, W = canvas.shape[:2]
        for x, y, s, size, weight, color, anchor in self._texts:
            dx, dy, a = self._glyph(s, size, weight, anchor)
            x1, y1 = x + dx, y + dy
            x2, y2 = x1 + a.shape[1], y1 + a.shape[0]
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            a = a[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
            roi = canvas[cy1:cy2, cx1:cx2]
            roi[:] = (roi * (1.0 - a) + np.array(color, np.float32) * a).astype(np.uint8)
        self._texts.clear()
        return canvas

    # ----- widgets -----------------------------------------------------
    def card(self, img, x, y, w, h):
        rrect(img, x, y, x + w, y + h, CARD_R, PANEL)
        rrect(img, x, y, x + w, y + h, CARD_R, BORDER, thickness=1)

    def pill(self, img, x_right, cy, label, color, dot=True):
        """A status pill, right-aligned at x_right, vertically centered on cy."""
        tw = self.text_w(label, 11, "bold")
        w = tw + (30 if dot else 20)
        x1 = x_right - w
        rrect(img, x1, cy - 12, x_right, cy + 12, 12, color, alpha=0.18)
        if dot:
            circle(img, x1 + 13, cy, 4, color)
        self.text(x_right - 10, cy, label, 11, color, "bold", "rm")
        return x1

    def chip(self, img, x, cy, label, value, dot_color=None):
        """Header chip: [dot] LABEL value, left-aligned at x. Returns its right edge."""
        lw = self.text_w(label, 11, "semibold")
        vw = self.text_w(value, 13, "semibold")
        dot_w = 16 if dot_color else 0
        w = 14 + dot_w + lw + 8 + vw + 14
        rrect(img, x, cy - 15, x + w, cy + 15, 15, PANEL)
        rrect(img, x, cy - 15, x + w, cy + 15, 15, BORDER, thickness=1)
        cx = x + 14
        if dot_color:
            circle(img, cx + 4, cy, 4, dot_color)
            cx += dot_w
        self.text(cx, cy, label, 11, MUTED, "semibold", "lm")
        self.text(cx + lw + 8, cy, value, 13, TEXT, "semibold", "lm")
        return x + w

    # ----- sections ----------------------------------------------------
    def _header(self, img, stats):
        x = PAD
        cy = HEADER_H // 2 + 6
        rrect(img, x, cy - 16, x + 32, cy + 16, 9, ACCENTS["green"])
        cv2.circle(img, (x + 16, cy), 7, BG, 2, cv2.LINE_AA)        # little "target" logo
        cv2.circle(img, (x + 16, cy), 2, BG, -1, cv2.LINE_AA)
        self.text(x + 46, cy - 2, "Minifig Tracker", 20, TEXT, "bold", "ls")
        self.text(x + 46, cy + 15, f"Project 4 · Door to Door  ·  {self.model_name}", 12, MUTED, "regular", "ls")

        # Chips laid out right-to-left from the right edge
        items = [
            ("YOLO", f"{stats['infer_ms']:.0f} ms", None),
            ("FPS", f"{stats['fps']:.1f}", None),
            ("SENT", f"{stats['sent']:,}", None),
            ("MQTT", f"{self.broker} · {self.publish_hz} Hz", OK),
        ]
        widths = []
        for label, value, dot in items:
            widths.append(14 + (16 if dot else 0) + self.text_w(label, 11, "semibold") + 8
                          + self.text_w(value, 13, "semibold") + 14)
        xr = self.width - PAD
        for (label, value, dot), w in zip(items, widths):
            xr -= w
            self.chip(img, xr, cy, label, value, dot)
            xr -= 10

    def _video(self, img, frame, entries):
        bx, by = PAD, HEADER_H + 16
        bw, bh = VIDEO_BOX

        fh, fw = frame.shape[:2]
        scale = min(bw / fw, bh / fh)
        dw, dh = int(fw * scale), int(fh * scale)
        ox, oy = bx + (bw - dw) // 2, by          # top-aligned; spare height goes below the cards
        view = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)

        # Overlays are drawn at display resolution so lines stay crisp.
        vx = lambda xn: int(xn * dw / 1000)        # 0..1000 -> video px
        vy = lambda yn: int(yn * dh / 1000)
        cxp = vx(self.center)
        band = vx(self.deadband)
        rrect(view, cxp - band, 0, cxp + band, dh, 0, OK, alpha=0.12)
        dashed_vline(view, cxp, 0, dh, (230, 230, 230))

        any_seen = False
        for e in entries:
            m, accent = e["msg"], ACCENTS.get(e["color"], TEXT)
            if not m["seen"]:
                continue
            any_seen = True
            px, py = vx(m["x"]), vy(m["y"])
            if e["box"] is not None:
                x1, y1, x2, y2 = e["box"]
                x1, y1, x2, y2 = int(x1 * dw), int(y1 * dh), int(x2 * dw), int(y2 * dh)
                rrect(view, x1, y1, x2, y2, 6, accent, alpha=0.10)
                corner_box(view, x1, y1, x2, y2, accent)
                label = f"{e['color'].upper()}  {m['conf']}%"
                lw = self.text_w(label, 12, "bold") + 18
                ly = y1 - 30 if y1 > 34 else y2 + 6
                rrect(view, x1, ly, x1 + lw, ly + 24, 8, accent)
                # view is pasted into img below at (ox, oy), so offset the text
                self.text(ox + x1 + 9, oy + ly + 12, label, 12, BG, "bold", "lm")
            if m["speed"] != 0:   # arrow: which way the car is pushing the minifig
                length = min(abs(cxp - px), 110)
                tip_x = px + (length if cxp > px else -length)
                cv2.arrowedLine(view, (px, py), (tip_x, py), accent, 3, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(view, (px, py), 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(view, (px, py), 6, accent, -1, cv2.LINE_AA)

        # Paste the video with rounded corners: copy it all, then put the
        # background back in just the four small corner squares.
        if self._video_mask is None or self._video_mask.shape != (CARD_R, CARD_R):
            mask = np.zeros((2 * CARD_R, 2 * CARD_R), np.uint8)
            cv2.circle(mask, (CARD_R, CARD_R), CARD_R, 255, -1, cv2.LINE_AA)
            self._video_mask = (mask[:CARD_R, :CARD_R] < 128)   # True = outside the curve
        roi = img[oy:oy + dh, ox:ox + dw]
        roi[:] = view
        m, r = self._video_mask, CARD_R
        for sl_y, sl_x, mm in ((slice(0, r), slice(0, r), m),
                               (slice(0, r), slice(dw - r, dw), m[:, ::-1]),
                               (slice(dh - r, dh), slice(0, r), m[::-1, :]),
                               (slice(dh - r, dh), slice(dw - r, dw), m[::-1, ::-1])):
            roi[sl_y, sl_x][mm] = BG
        rrect(img, ox, oy, ox + dw - 1, oy + dh - 1, CARD_R, BORDER, thickness=1)

        # LIVE + resolution chips on top of the video
        rrect(img, ox + 12, oy + 12, ox + 76, oy + 38, 13, (0, 0, 0), alpha=0.55)
        circle(img, ox + 26, oy + 25, 4, LIVE_RED)
        self.text(ox + 36, oy + 25, "LIVE", 11, TEXT, "bold", "lm")
        res = f"{fw}×{fh}"
        rw = self.text_w(res, 11, "semibold") + 22
        rrect(img, ox + dw - 12 - rw, oy + 12, ox + dw - 12, oy + 38, 13, (0, 0, 0), alpha=0.55)
        self.text(ox + dw - 12 - rw // 2, oy + 25, res, 11, TEXT, "semibold", "mm")
        if not any_seen:
            msg = "Looking for a minifig…  (cars stopped)"
            mw = self.text_w(msg, 14, "semibold") + 36
            mx = ox + (dw - mw) // 2
            rrect(img, mx, oy + dh - 58, mx + mw, oy + dh - 22, 18, (0, 0, 0), alpha=0.6)
            self.text(ox + dw // 2, oy + dh - 40, msg, 14, TEXT, "semibold", "mm")
        return oy + dh

    def _track(self, img, y, entries):
        x, w = PAD, self.left_w
        self.card(img, x, y, w, TRACK_CARD_H)
        self.text(x + 18, y + 24, "HORIZONTAL POSITION", 11, MUTED, "semibold")
        self.text(x + w - 18, y + 24, f"stop zone ±{self.deadband}", 11, DIM, "semibold", "rs")

        tx1, tx2, ty = x + 24, x + w - 24, y + 48
        to_px = lambda v: int(tx1 + (tx2 - tx1) * v / 1000)
        rrect(img, tx1, ty - 4, tx2, ty + 4, 4, PANEL_HI)
        rrect(img, to_px(self.center - self.deadband), ty - 8, to_px(self.center + self.deadband), ty + 8,
              5, OK, alpha=0.30)
        cv2.line(img, (to_px(self.center), ty - 12), (to_px(self.center), ty + 12), TEXT, 2, cv2.LINE_AA)
        for v, label, anchor in ((0, "LEFT", "ls"), (self.center, "CENTER", "ms"), (1000, "RIGHT", "rs")):
            self.text(to_px(v), y + TRACK_CARD_H - 10, label, 10, DIM, "semibold", anchor)
        for e in entries:
            m = e["msg"]
            if m["seen"]:
                px = to_px(m["x"])
                cv2.circle(img, (px, ty), 10, BG, -1, cv2.LINE_AA)
                cv2.circle(img, (px, ty), 8, ACCENTS.get(e["color"], TEXT), -1, cv2.LINE_AA)
        return y + TRACK_CARD_H

    def _payloads(self, img, y, entries):
        x, w = PAD, self.left_w
        self.card(img, x, y, w, PAYLOAD_CARD_H)
        self.text(x + 18, y + 22, "LAST MQTT PAYLOADS", 11, MUTED, "semibold")
        for i, e in enumerate(entries[:2]):
            ry = y + 44 + i * 20
            circle(img, x + 22, ry, 4, ACCENTS.get(e["color"], TEXT))
            self.text(x + 34, ry, e["topic"].rsplit("/", 1)[-1], 12, TEXT, "semibold", "lm")
            self.text(x + 90, ry, e["payload"], 12, MUTED if e["msg"]["seen"] else DIM, "mono", "lm")

    def _color_card(self, img, x, y, e):
        w, h = SIDEBAR_W, COLOR_CARD_H
        m, accent = e["msg"], ACCENTS.get(e["color"], TEXT)
        self.card(img, x, y, w, h)
        rrect(img, x, y + 18, x + 4, y + h - 18, 2, accent)            # accent stripe

        if not e["trained"]:
            status, scol = "NOT TRAINED", DIM
        elif not m["seen"]:
            status, scol = "NO TARGET", WARN
        elif m["speed"] == 0:
            status, scol = "ARRIVED", OK
        else:
            status, scol = "DRIVING", INFO

        circle(img, x + 26, y + 28, 7, accent)
        self.text(x + 42, y + 28, f"{e['color'].capitalize()} minifig", 17, TEXT, "semibold", "lm")
        self.pill(img, x + w - 18, y + 28, status, scol)
        self.text(x + 20, y + 54, e["topic"], 11, DIM, "mono", "lm")

        # Stats row
        seen = m["seen"]
        cols = [("POSITION", f"{m['x']}, {m['y']}" if seen else "—"),
                ("ERROR", f"{m['err']:+d}" if seen else "—"),
                ("CONF", f"{m['conf']}%" if seen else "—")]
        cw = (w - 40) // 3
        for i, (label, value) in enumerate(cols):
            cx = x + 20 + i * cw
            self.text(cx, y + 82, label, 10, MUTED, "semibold")
            self.text(cx, y + 106, value, 18, TEXT if seen else DIM, "semibold")
        # confidence bar under CONF
        bx1, bx2 = x + 20 + 2 * cw, x + w - 20
        rrect(img, bx1, y + 114, bx2, y + 118, 2, PANEL_HI)
        if seen:
            rrect(img, bx1, y + 114, bx1 + int((bx2 - bx1) * m["conf"] / 100), y + 118, 2, accent)

        # Motor: bidirectional bar, center = stop, left = backward, right = forward
        self.text(x + 20, y + 142, "MOTOR", 10, MUTED, "semibold")
        if not e["trained"]:
            self.text(x + w - 20, y + 142, f"add a '{e['color']}' class and retrain", 11, DIM, "regular", "rs")
        else:
            word = m["cmd"].upper() + (f"  {abs(m['speed'])}%" if m["speed"] else "")
            self.text(x + w - 20, y + 142, word, 12, scol if m["speed"] else MUTED, "bold", "rs")
        mx1, mx2, my = x + 20, x + w - 20, y + 158
        mid = (mx1 + mx2) // 2
        rrect(img, mx1, my - 5, mx2, my + 5, 5, PANEL_HI)
        if m["speed"]:
            end = mid + int((mx2 - mid) * m["speed"] / 100)
            rrect(img, min(mid, end), my - 5, max(mid, end), my + 5, 5, INFO)
        cv2.line(img, (mid, my - 9), (mid, my + 9), TEXT, 2, cv2.LINE_AA)
        self.text(mx1, y + h - 4, "BACK", 9, DIM, "semibold", "ls")
        self.text(mx2, y + h - 4, "FWD", 9, DIM, "semibold", "rs")
        return y + h

    def _led_card(self, img, x, y, entries):
        w = SIDEBAR_W
        grid_w, grid_h = self.led_cols * LED_CELL, self.led_rows * LED_CELL
        h = 44 + grid_h + 34
        self.card(img, x, y, w, h)
        self.text(x + 20, y + 26, "UNO Q LED MATRIX", 11, MUTED, "semibold")
        self.text(x + w - 20, y + 26, f"{self.led_cols} × {self.led_rows}", 11, DIM, "semibold", "rs")

        gx, gy = x + (w - grid_w) // 2, y + 40
        rrect(img, gx - 8, gy - 6, gx + grid_w + 8, gy + grid_h + 6, 10, (14, 12, 10))
        lit = {(e["msg"]["col"], e["msg"]["row"]) for e in entries if e["msg"]["seen"]}
        for r in range(self.led_rows):
            for c in range(self.led_cols):
                cx, cy = gx + c * LED_CELL + LED_CELL // 2, gy + r * LED_CELL + LED_CELL // 2
                if (c, r) in lit:
                    circle(img, cx, cy, LED_CELL // 2 + 2, LED_ON, alpha=0.30)   # glow
                    circle(img, cx, cy, 6, LED_ON)
                    circle(img, cx - 1, cy - 1, 2, (255, 230, 200))
                else:
                    circle(img, cx, cy, 5, LED_OFF)

        parts = [f"{e['color']}: col {e['msg']['col']}, row {e['msg']['row']}"
                 for e in entries if e["msg"]["seen"]]
        self.text(x + w // 2, y + h - 12, "   ·   ".join(parts) if parts else "no dot lit",
                  11, MUTED if parts else DIM, "regular", "ms")

    # ----- public ------------------------------------------------------
    def render(self, frame, entries, stats):
        """entries: list of dicts with keys color, topic, msg, payload, box
        (normalized x1,y1,x2,y2 or None), trained. stats: fps, infer_ms, sent."""
        img = self._bg.copy()
        self._header(img, stats)
        y = self._video(img, frame, entries)
        y = self._track(img, y + 16, entries)
        self._payloads(img, y + 16, entries)

        sx, sy = PAD + self.left_w + PAD, HEADER_H + 16
        for e in entries[:2]:
            sy = self._color_card(img, sx, sy, e) + 16
        self._led_card(img, sx, sy, entries)
        return self._flush_text(img)
