"""Renders the stop-action movie about how FingerNet works, and the voice-over script that goes with it.

    python movie/make_movie.py              # writes movie/nn_movie.mp4 and movie/voiceover_script.md
    python movie/make_movie.py --preview    # also saves one PNG per step to movie/preview/

Everything on screen comes from the real trained network (models/finger_cnn.npz) and real training photos,
run through npnet.py exactly as the lock runs them. The video is silent, 1280 x 720 at 8 frames a second.
Each step is drawn once and held for a few frames with a one-pixel wobble, which gives the stop-action look.

The narration for each scene lives in NARRATION below. A scene is held long enough for its narration at
140 words a minute, and the script's timestamps are written from the same frame counts, so they match.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import camera  # noqa: E402
import config  # noqa: E402
import npnet  # noqa: E402

W, H = 1280, 720
FPS = 8
WORDS_PER_MIN = 150
EXAMPLE_CODE = (2, 4, 1)      # shown in the last scene; the real code stays secret

PAPER = (247, 243, 232)
INK = (38, 38, 46)
SOFT = (120, 118, 112)
RED = (200, 58, 46)
BLUE = (46, 96, 186)
GREEN = (46, 150, 76)
YELLOW = (232, 178, 30)
DIM = (210, 205, 195)

rng = np.random.default_rng(7)

# Narration, one entry per scene, in the order the scenes play. First person, read by the student.
NARRATION = {
    "title": (
        "This is my finger-count lock. A webcam looks down at my hand, and a small neural network counts my "
        "fingers, from zero to five. Here's how it works."),
    "input": (
        "The program first keeps the center square, turns it gray and shrinks it to "
        "64 by 64 pixels, so it's just a grid of numbers from 0 to 255. Then it subtracts the average "
        "brightness and divides by the spread, so a bright room and a dim room look alike."),
    "conv": (
        "The first layer is a convolution. A filter is a 5 by 5 grid of weights the network learned. It "
        "slides over the image, and at each spot it multiplies the 25 pixels by the 25 weights, adds them up "
        "and adds a bias. A patch that matches the pattern gives a big positive number, in red, and the "
        "opposite pattern gives a negative one, in blue. All the spots together make a feature map."),
    "relu": (
        "There are eight filters, so one photo becomes eight feature maps, each picking out something "
        "different. Then ReLU keeps the positive numbers and turns every negative into zero."),
    "pool": (
        "Next comes max pooling. A 2 by 2 window steps across the map and keeps the biggest of each four "
        "numbers. That halves the size to 32 by 32, keeps the strong responses, and makes small shifts of my "
        "hand matter less."),
    "deeper": (
        "Convolution, ReLU and pooling happen three times. The second layer's 16 filters each look at all "
        "eight maps before them, so they can combine edges into pieces of fingers. The third layer's 32 "
        "filters tend to respond to whole fingers."),
    "output": (
        "Then the 32 small maps become one line of 2048 numbers. Each count has its own 2048 weights, and "
        "multiplying and adding gives six scores. Softmax turns them into probabilities that add up to one. "
        "The tallest bar is the answer, and its height is how sure the network is."),
    "heat": (
        "Since the last layer comes right after the third convolution, I can split each score back onto the "
        "8 by 8 grid. Red shows what pushed the network toward its answer. Often that's the fingers, and "
        "sometimes it's the empty space beside them."),
    "train": (
        "The weights start out random, so at first the network just guesses. The loss is minus the log of the probability it gave the right answer. Backpropagation uses the chain "
        "rule, like my V-fit spreadsheet, to find how each of the 18,310 weights changes the loss, and "
        "gradient descent nudges them all downhill. Each training photo is also randomly rotated, shifted, "
        "zoomed, flipped and brightened. In the end it got 98 percent of the test photos right, though they came "
        "from the same session as the training photos."),
    "lock": (
        "Finally, the lock. I press the button and take my hand away so it reads zero. Then I hold up a "
        "count for a second at 80 percent or more, and a yellow LED lights. After three digits the whole code "
        "is checked at once, and the right code turns on the green LED and swings the latch open."),
    "end": "",
}

SCENE_TITLES = {
    "title": "Title", "input": "From camera to numbers", "conv": "Convolution", "relu": "Eight filters and ReLU",
    "pool": "Max pooling", "deeper": "Going deeper", "output": "From maps to an answer",
    "heat": "Where it looked", "train": "How it learned", "lock": "From counts to the lock", "end": "End card",
}


# Drawing helpers ------------------------------------------------------------------------------------------------

_fonts = {}
FONT_FILES = {"hand": "segoepr.ttf", "handb": "segoeprb.ttf", "mono": "consola.ttf", "monob": "consolab.ttf",
              "sans": "segoeui.ttf"}


def font(size, kind="hand"):
    key = (size, kind)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.truetype(str(Path("C:/Windows/Fonts") / FONT_FILES[kind]), size)
        except OSError:
            _fonts[key] = ImageFont.load_default(size)
    return _fonts[key]


def make_paper():
    noise = np.random.default_rng(1).normal(0, 2.5, (H, W, 1))
    blotch = cv2.GaussianBlur(np.random.default_rng(2).normal(0, 1, (H, W)).astype(np.float32), (0, 0), 40)
    blotch = blotch / (np.abs(blotch).max() + 1e-6) * 5
    img = np.array(PAPER, np.float32)[None, None] + noise + blotch[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


PAPER_IMG = make_paper()


def wline(d, p0, p1, fill=INK, width=3, wob=1.0):
    (x0, y0), (x1, y1) = p0, p1
    n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 35))
    pts = [(x0 + (x1 - x0) * t + rng.normal(0, wob), y0 + (y1 - y0) * t + rng.normal(0, wob))
           for t in np.linspace(0, 1, n + 1)]
    d.line(pts, fill=fill, width=width, joint="curve")


def wrect(d, box, outline=INK, width=3, fill=None):
    x0, y0, x1, y1 = box
    if fill is not None:
        d.rectangle(box, fill=fill)
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        wline(d, a, b, outline, width)


def arrow(d, p0, p1, fill=INK, width=4):
    wline(d, p0, p1, fill, width)
    (x0, y0), (x1, y1) = p0, p1
    ang = np.arctan2(y1 - y0, x1 - x0)
    head = [(x1, y1), (x1 - 18 * np.cos(ang - 0.45), y1 - 18 * np.sin(ang - 0.45)),
            (x1 - 18 * np.cos(ang + 0.45), y1 - 18 * np.sin(ang + 0.45))]
    d.polygon(head, fill=fill)


def text(d, xy, s, size=28, fill=INK, kind="hand", anchor="la"):
    d.text(xy, s, font=font(size, kind), fill=fill, anchor=anchor)


def para(d, xy, s, size=26, maxw=1100, fill=INK, kind="hand", gap=8, anchor_x="l"):
    """Wrapped text. Returns the y just below it."""
    f = font(size, kind)
    lines, cur = [], ""
    for word in s.split():
        trial = (cur + " " + word).strip()
        if d.textlength(trial, font=f) > maxw and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    x, y = xy
    for line in lines:
        if anchor_x == "c":
            d.text((x, y), line, font=f, fill=fill, anchor="ma")
        else:
            d.text((x, y), line, font=f, fill=fill)
        y += size + gap
    return y


def page(title=None):
    img = PAPER_IMG.copy()
    d = ImageDraw.Draw(img)
    if title:
        text(d, (56, 26), title, 46, INK, "handb")
        tw = d.textlength(title, font=font(46, "handb"))
        wline(d, (56, 92), (56 + tw + 20, 92), RED, 4, wob=1.5)
    return img, d


def caption(d, s, y=640, size=26):
    para(d, (W // 2, y), s, size, 1150, INK, "hand", anchor_x="c")


# Colors for arrays

def gray_rgb(a, lo=None, hi=None):
    lo = a.min() if lo is None else lo
    hi = a.max() if hi is None else hi
    g = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255
    return np.repeat(g[..., None], 3, axis=2).astype(np.uint8)


def div_rgb(a, m=None):
    m = np.abs(a).max() + 1e-9 if m is None else m
    t = np.clip(a / m, -1, 1)[..., None]
    white = np.array([250, 248, 240], np.float32)
    red = np.array([205, 55, 45], np.float32)
    blue = np.array([45, 95, 190], np.float32)
    out = np.where(t >= 0, white + (red - white) * t, white + (blue - white) * (-t))
    return out.astype(np.uint8)


def heat_rgb(a, m=None):
    m = a.max() + 1e-9 if m is None else m
    return div_rgb(np.clip(a, 0, None), m)


def put(img, d, rgb, xy, size, border=True, width=3):
    pil = Image.fromarray(np.ascontiguousarray(rgb)).resize(size, Image.NEAREST)
    img.paste(pil, (int(xy[0]), int(xy[1])))
    if border:
        wrect(d, (xy[0] - 2, xy[1] - 2, xy[0] + size[0] + 1, xy[1] + size[1] + 1), INK, width)


def number_grid(d, x, y, cell, arr, fmt, bg=None, size=14, kind="mono", lines=True):
    n, m = arr.shape
    for i in range(n):
        for j in range(m):
            box = (x + j * cell, y + i * cell, x + (j + 1) * cell, y + (i + 1) * cell)
            fill = tuple(int(v) for v in bg[i, j]) if bg is not None else None
            if fill is not None:
                d.rectangle(box, fill=fill)
            if lines:
                d.rectangle(box, outline=(150, 145, 135), width=1)
            lum = 0.3 * fill[0] + 0.59 * fill[1] + 0.11 * fill[2] if fill is not None else 255
            d.text((box[0] + cell / 2, box[1] + cell / 2), fmt(arr[i, j]), font=font(size, kind),
                   fill=(255, 255, 255) if lum < 120 else INK, anchor="mm")
    wrect(d, (x, y, x + m * cell, y + n * cell), INK, 3)


def bars(d, x, y, probs, w=300, h=200, frac=1.0, size=20, color=BLUE, best_color=RED):
    """Six bars for counts 0-5. (x, y) is the bottom left corner."""
    bw = w / 6
    best = int(np.argmax(probs))
    d.line([(x - 6, y), (x + w + 6, y)], fill=INK, width=3)
    for k, p in enumerate(probs):
        ph = h * p * frac
        x0 = x + k * bw + 6
        c = best_color if k == best else color
        if ph > 1:
            wrect(d, (x0, y - ph, x0 + bw - 12, y), c, 2, fill=c)
        text(d, (x0 + (bw - 12) / 2, y + 6), str(k), size, INK, "hand", "ma")
        if frac >= 1 and p >= 0.01:
            text(d, (x0 + (bw - 12) / 2, y - ph - 6), f"{p:.0%}", size - 4, INK, "mono", "md")


# The movie ------------------------------------------------------------------------------------------------------

class Movie:
    def __init__(self, path, preview_dir=None):
        self.path = path
        self.preview_dir = preview_dir
        self.frames = 0
        self.scenes = []          # (key, start_frame, end_frame)
        self.current = None
        self.last = None
        self.step_n = 0
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                 "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                 "-preset", "medium", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
            self.writer = None
            self.encoder = "ffmpeg libx264 (H.264, yuv420p)"
        except Exception:
            self.proc = None
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
            self.encoder = "OpenCV mp4v"

    def _write(self, arr):
        if self.proc:
            self.proc.stdin.write(arr.tobytes())
        else:
            self.writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def hold(self, img, seconds=None, frames=None, name=None):
        n = frames if frames is not None else max(1, round(seconds * FPS))
        arr = np.asarray(img.convert("RGB"))
        for _ in range(n):
            dx, dy = rng.integers(-1, 2, 2)
            self._write(np.ascontiguousarray(np.roll(arr, (int(dy), int(dx)), axis=(0, 1))))
        self.last = img
        if self.preview_dir and name:
            self.step_n += 1
            img.save(self.preview_dir / f"{self.current}_{self.step_n:02d}_{name}.png")

    def scene(self, key):
        self.end_scene()
        self.current, self.start = key, self.frames
        self.step_n = 0
        print(f"scene {key} ...", flush=True)

    def end_scene(self):
        if self.current is None:
            return
        words = len(NARRATION.get(self.current, "").split())
        need = round((words / WORDS_PER_MIN * 60 + 0.8) * FPS) if words else 0
        if self.frames - self.start < need:
            self.hold(self.last, frames=need - (self.frames - self.start))
        self.scenes.append((self.current, self.start, self.frames))
        self.current = None

    def close(self):
        self.end_scene()
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.writer.release()


# The network, layer by layer ------------------------------------------------------------------------------------

WTS = npnet.load_weights(config.NUMPY_WEIGHTS_PATH)


def run_net(inp, wts=WTS):
    x = inp[None].astype(np.float32)
    out = {"x": x}
    for name in ("conv1", "conv2", "conv3"):
        pre = npnet.conv2d(x, wts[f"{name}_w"], wts[f"{name}_b"])
        r = npnet.relu(pre)
        x = npnet.maxpool2(r)
        out[name] = (pre, r, x)
    out["flat"] = x.reshape(-1)
    out["logits"] = wts["fc_w"] @ out["flat"] + wts["fc_b"]
    out["probs"] = npnet.softmax(out["logits"])
    return out


def load_photo(path):
    saved = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = camera.to_small(saved)
    inp = camera.standardize(small)
    return {"path": path, "saved": saved, "small": small, "inp": inp, "net": run_net(inp)}


def pick_examples():
    """For each count, the photo (out of an even sample) the network is surest about. Plus an empty view."""
    best = {}
    for c in config.CLASSES:
        files = sorted((config.DATA_DIR / str(c)).glob("*.png"))
        sample = files[:: max(1, len(files) // 40)]
        photos = [load_photo(p) for p in sample]
        right = [ph for ph in photos if int(np.argmax(ph["net"]["probs"])) == c] or photos
        unsure = [ph for ph in right if ph["net"]["probs"][c] < 0.97]     # so the bars show the other counts
        best[c] = max(unsure or right, key=lambda ph: ph["net"]["probs"][c])
        if c == 0:
            flat = min(files, key=lambda f: cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)[20:120, 20:140].std())
            best["empty"] = load_photo(flat)
    return best


# Scenes ---------------------------------------------------------------------------------------------------------

def scene_title(mv, ex):
    mv.scene("title")
    img, d = page()
    text(d, (W // 2, 170), "How the finger-count lock sees", 64, INK, "handb", "mm")
    wline(d, (250, 215), (1030, 215), RED, 5, 2)
    put(img, d, gray_rgb(ex[3]["saved"].astype(np.float32)), (200, 300), (220, 220))
    arrow(d, (460, 410), (560, 410))
    wrect(d, (580, 330, 760, 490), INK, 4, fill=(236, 230, 214))
    text(d, (670, 385), "neural", 30, INK, "handb", "mm")
    text(d, (670, 425), "network", 30, INK, "handb", "mm")
    arrow(d, (780, 410), (880, 410))
    text(d, (980, 410), "3", 120, RED, "handb", "mm")
    text(d, (980, 500), "fingers (0 to 5)", 26, SOFT, "hand", "mm")
    caption(d, "A photo goes in, a count from 0 to 5 comes out. Three counts in the right order open the box.",
            600, 28)
    mv.hold(img, 4, name="title")


def scene_input(mv, ex):
    mv.scene("input")
    ph = ex[3]
    # 1. camera frame with the center square. The area outside the square is only suggested (the saved photos
    # keep just the square), so it is drawn faded.
    img, d = page("1. From camera to numbers")
    fw, fh = 480, 360
    side = int(fh * config.CROP_FRAC)
    sq = cv2.resize(ph["saved"], (side, side), interpolation=cv2.INTER_AREA)
    px, py = (fw - side) // 2, (fh - side) // 2
    frame = cv2.copyMakeBorder(sq, py, fh - side - py, px, fw - side - px, cv2.BORDER_REPLICATE)
    frame = cv2.GaussianBlur(frame, (0, 0), 6).astype(np.float32)
    frame = frame * 0.55 + 255 * 0.45
    frame[py:py + side, px:px + side] = sq
    fx, fy = 60, 150
    put(img, d, gray_rgb(frame, 0, 255), (fx, fy), (fw, fh))
    wrect(d, (fx + px, fy + py, fx + px + side, fy + py + side), RED, 4)
    text(d, (fx, fy + fh + 14), "camera frame, 640 x 480", 24, SOFT)
    text(d, (fx + px + side - 4, fy + py + 8), "center square", 22, RED, "hand", "ra")
    caption(d, "Keep only the center square, where the hand goes.")
    mv.hold(img, 3, name="frame")

    # 2. square, gray, 160 -> 64 x 64
    x0 = fx + fw + 50
    arrow(d, (fx + fw + 6, 330), (x0 - 6, 330))
    put(img, d, gray_rgb(ph["saved"].astype(np.float32), 0, 255), (x0, 230), (200, 200))
    text(d, (x0 + 100, 445), "gray", 24, SOFT, "hand", "ma")
    arrow(d, (x0 + 210, 330), (x0 + 270, 330))
    small = ph["small"]
    put(img, d, gray_rgb(small, 0, 1), (x0 + 280, 170), (320, 320))
    text(d, (x0 + 440, 505), "64 x 64 pixels", 24, SOFT, "hand", "ma")
    img2 = img.copy()
    d2 = ImageDraw.Draw(img2)
    d2.rectangle((0, 600, W, H), fill=PAPER)
    caption(d2, "Turn it gray and shrink it to 64 x 64. Now it's just 4,096 numbers.")
    mv.hold(img2, 3.5, name="small")

    # 3. a zoomed 8 x 8 patch as numbers, then standardized
    best, pos = -1, (0, 0)
    for i in range(0, 33, 4):          # the upper part, where the fingers are
        for j in range(0, 57, 4):
            s = small[i:i + 8, j:j + 8].std()
            if s > best:
                best, pos = s, (i, j)
    i0, j0 = pos
    raw = (small[i0:i0 + 8, j0:j0 + 8] * 255).round()
    std = ph["inp"][i0:i0 + 8, j0:j0 + 8]
    mean, spread = small.mean() * 255, small.std() * 255

    def zoom_page(values, fmt, colors, title_line, cap):
        im, dd = page("1. From camera to numbers")
        put(im, dd, gray_rgb(small, 0, 1), (60, 150), (384, 384))
        wrect(dd, (60 + j0 * 6, 150 + i0 * 6, 60 + (j0 + 8) * 6, 150 + (i0 + 8) * 6), RED, 4)
        arrow(dd, (60 + (j0 + 8) * 6 + 8, 150 + (i0 + 4) * 6), (520, 330), RED, 3)
        text(dd, (540, 125), title_line, 26, INK, "handb")
        number_grid(dd, 540, 170, 54, values, fmt, colors, 17)
        caption(dd, cap)
        return im

    im = zoom_page(raw, lambda v: f"{int(v)}", gray_rgb(raw, 0, 255), "8 x 8 patch: brightness 0 to 255",
                   "Zoom in on a finger edge: every pixel is a number from 0 (black) to 255 (white).")
    mv.hold(im, 4, name="raw")
    im = zoom_page(std, lambda v: f"{v:+.1f}", div_rgb(std, 2.5), "standardized: (pixel - mean) / spread",
                   f"This photo's mean is {mean:.0f} and its spread is {spread:.0f}. After standardizing, "
                   "darker than average is negative (blue) and brighter is positive (red).")
    mv.hold(im, 4.5, name="std")


def scene_conv(mv, ex):
    mv.scene("conv")
    ph = ex[3]
    pre1 = ph["net"]["conv1"][0]
    f = int(np.argmax([pre1[k].std() for k in range(8)]))
    w = WTS["conv1_w"][f, 0]
    b = float(WTS["conv1_b"][f])
    fmap = pre1[f]
    m = np.abs(fmap).max()
    xp = np.pad(ph["inp"], 2)
    IX, IY, S = 40, 150, 5          # input at 5 px per pixel = 320 px
    MX, MY = 920, 150               # feature map, same scale

    def base(filled=None, win=None, cap=None, title="2. Convolution: one filter"):
        im, dd = page(title)
        put(im, dd, div_rgb(ph["inp"], 2.5), (IX, IY), (320, 320))
        text(dd, (IX + 160, IY + 335), "input (64 x 64)", 24, SOFT, "hand", "ma")
        dd.text((645, 120), "filter weights (5 x 5)", font=font(24, "handb"), fill=INK)
        number_grid(dd, 645, 160, 44, w, lambda v: f"{v:+.2f}", div_rgb(w), 13)
        if filled is not None:
            shown = np.full((64, 64, 3), 232, np.uint8)
            mask = np.zeros(64 * 64, bool)
            mask[:filled] = True
            mask = mask.reshape(64, 64)
            shown[mask] = div_rgb(fmap, m)[mask]
            put(im, dd, shown, (MX, MY), (320, 320))
            text(dd, (MX + 160, MY + 335), "feature map (64 x 64)", 24, SOFT, "hand", "ma")
        if win is not None:
            i, j = win
            wrect(dd, (IX + (j - 2) * S, IY + (i - 2) * S, IX + (j + 3) * S, IY + (i + 3) * S), INK, 3)
        if cap:
            caption(dd, cap)
        return im, dd

    im, dd = base(cap="A filter is 25 weights the network learned. Red weights like bright pixels, blue ones "
                      "like dark pixels.")
    mv.hold(im, 4, name="filter")

    inner = fmap[4:40, 4:60]
    hi = np.unravel_index(np.argmax(inner), inner.shape)
    lo = np.unravel_index(np.argmin(inner), inner.shape)
    for (i, j), word in (((hi[0] + 4, hi[1] + 4), "matches the pattern: big and positive"),
                         ((lo[0] + 4, lo[1] + 4), "the opposite pattern: negative")):
        patch = xp[i:i + 5, j:j + 5]
        total = float((patch * w).sum())
        im, dd = base(filled=None, win=(i, j))
        number_grid(dd, 385, 160, 44, patch, lambda v: f"{v:+.1f}", div_rgb(patch, 2.5), 13)
        text(dd, (625, 270), "x", 40, INK, "handb", "mm")
        text(dd, (385 + 110, 390), "pixels under the filter", 20, SOFT, "hand", "ma")
        arrow(dd, (IX + (j + 3) * S + 4, IY + i * S), (380, 200), INK, 3)
        val = total + b
        dd.text((385, 450), f"multiply each pair and add: {total:+.2f}", font=font(24, "mono"), fill=INK)
        dd.text((385, 485), f"plus the bias:              {b:+.2f}", font=font(24, "mono"), fill=INK)
        dd.text((385, 525), f"= {val:+.2f}", font=font(34, "monob"), fill=RED if val > 0 else BLUE)
        text(dd, (385, 570), word, 24, RED if val > 0 else BLUE, "hand")
        assert abs(val - fmap[i, j]) < 1e-3
        mv.hold(im, 4.5, name=f"step_{'pos' if val > 0 else 'neg'}")

    # fill the map: the first row cell by cell, then faster
    sched = [8 * (k + 1) for k in range(8)] + [64 * r for r in (2, 3, 4)] + list(range(64 * 8, 4097, 64 * 4))
    sched[-1] = 4096
    for n in sched:
        last = n - 1
        im, dd = base(filled=n, win=(last // 64, last % 64),
                      cap="Slide to every position and write each answer down: that builds the feature map.")
        mv.hold(im, frames=1 if n > 64 else 2)
    im, dd = base(filled=4096, cap="Red where the filter's pattern shows up, blue where the opposite does.")
    mv.hold(im, 3, name="map")


def scene_relu(mv, ex):
    mv.scene("relu")
    ph = ex[3]
    pre1, r1, _ = ph["net"]["conv1"]
    m = np.abs(pre1).max()

    def grid(after):
        im, dd = page("3. Eight filters, then ReLU")
        for k in range(8):
            x = 52 + k * 150
            put(im, dd, div_rgb(WTS["conv1_w"][k, 0]), (x + 29, 130), (70, 70), width=2)
            maps = r1[k] if after else pre1[k]
            put(im, dd, div_rgb(maps, m), (x, 240), (128, 128), width=2)
        text(dd, (40, 160), "filters", 20, SOFT, "hand", "ra")
        if after:
            text(dd, (W // 2, 420), "ReLU:  max(0, x)", 40, INK, "handb", "mm")
            caption(dd, "ReLU keeps the positive numbers and turns every negative (blue) into 0.", 470)
        else:
            caption(dd, "8 filters give 8 feature maps from the same photo, each one picking out something "
                        "different.", 420)
        return im

    mv.hold(grid(False), 5, name="maps")
    mv.hold(grid(True), 5, name="relu")


def pool_fmt(v):
    if np.isnan(v):
        return ""
    if v == 0:
        return "0"
    return f"{v:.2f}"[1:] if v < 0.1 else f"{v:.1f}"


def scene_pool(mv, ex):
    mv.scene("pool")
    ph = ex[3]
    _, r1, p1 = ph["net"]["conv1"]
    best, f, pos = -1, 0, (0, 0)
    for k in range(8):                 # the filter and corner with the most going on, on the fingers
        for i in range(0, 33, 2):
            for j in range(0, 57, 2):
                patch = r1[k, i:i + 8, j:j + 8]
                s = patch.std() * (np.count_nonzero(patch) > 24)
                if s > best:
                    best, f, pos = s, k, (i, j)
    rm = r1[f]
    i0, j0 = pos
    patch = rm[i0:i0 + 8, j0:j0 + 8]
    pooled = patch.reshape(4, 2, 4, 2).max(axis=(1, 3))
    m = rm.max()
    GX, GY, C = 380, 140, 54
    OX, OY, OC = 900, 230, 60

    def frame(k, cap):
        im, dd = page("4. Max pooling")
        put(im, dd, heat_rgb(rm, m), (40, 160), (256, 256))
        wrect(dd, (40 + j0 * 4, 160 + i0 * 4, 40 + (j0 + 8) * 4, 160 + (i0 + 8) * 4), INK, 3)
        text(dd, (168, 430), "after ReLU", 22, SOFT, "hand", "ma")
        number_grid(dd, GX, GY, C, patch, pool_fmt, heat_rgb(patch, m), 15)
        shown = np.full((4, 4, 3), 232, np.uint8)
        vals = np.full((4, 4), np.nan)
        for q in range(k):
            a, bb = divmod(q, 4)
            shown[a, bb] = heat_rgb(pooled, m)[a, bb]
            vals[a, bb] = pooled[a, bb]
        number_grid(dd, OX, OY, OC, vals, pool_fmt, shown, 16)
        for q in range(k, 16):     # blank the cells not filled yet
            a, bb = divmod(q, 4)
            dd.rectangle((OX + bb * OC + 1, OY + a * OC + 1, OX + (bb + 1) * OC - 1, OY + (a + 1) * OC - 1),
                         fill=(232, 228, 218))
        if 0 < k <= 16:
            a, bb = divmod(k - 1, 4)
            wrect(dd, (GX + 2 * bb * C, GY + 2 * a * C, GX + 2 * (bb + 1) * C, GY + 2 * (a + 1) * C), RED, 5)
            blk = patch[2 * a:2 * a + 2, 2 * bb:2 * bb + 2]
            ii, jj = np.unravel_index(np.argmax(blk), blk.shape)
            cx, cy = GX + (2 * bb + jj + 0.5) * C, GY + (2 * a + ii + 0.5) * C
            d_el = 24
            dd.ellipse((cx - d_el, cy - d_el, cx + d_el, cy + d_el), outline=RED, width=3)
            arrow(dd, (GX + 8 * C + 10, GY + (2 * a + 1) * C), (OX - 10, OY + (a + 0.5) * OC), RED, 3)
        text(dd, (OX + 2 * OC, OY + 4 * OC + 16), "4 x 4", 24, SOFT, "hand", "ma")
        caption(dd, cap, 610)
        return im

    mv.hold(frame(0, "Take an 8 x 8 corner of one feature map."), 2.5, name="patch")
    for k in range(1, 17):
        cap = ("Each 2 x 2 block keeps only its biggest number." if k <= 4
               else "The window steps across without overlapping.")
        mv.hold(frame(k, cap), 1.4 if k <= 4 else 0.375, name="step" if k in (1, 16) else None)

    im, dd = page("4. Max pooling")
    put(im, dd, heat_rgb(rm, m), (140, 150), (384, 384))
    text(dd, (332, 550), "64 x 64", 30, INK, "handb", "ma")
    arrow(dd, (560, 342), (700, 342))
    text(dd, (630, 300), "max pool", 26, INK, "hand", "mm")
    put(im, dd, heat_rgb(p1[f], m), (740, 246), (192, 192))
    text(dd, (836, 550), "32 x 32", 30, INK, "handb", "ma")
    caption(dd, "Half the width, half the height. The strong spots survive, and a small shift of the hand "
                "changes less.", 610)
    mv.hold(im, 4, name="halved")


def stack(d, img, x, y, n, s, label, sub, front=None):
    """A stack of n maps drawn as offset squares, front map at (x, y), size s."""
    off = 3 if n > 1 else 0
    for k in range(min(n, 32) - 1, -1, -1):
        dx, dy = k * off, -k * off
        box = (x + dx, y + dy, x + dx + s, y + dy + s)
        d.rectangle(box, fill=(236, 230, 214), outline=INK, width=2)
    if front is not None:
        put(img, d, front, (x, y), (s, s), width=2)
    text(d, (x + s / 2, y + s + 18), label, 26, INK, "handb", "ma")
    para(d, (x + s / 2, y + s + 54), sub, 20, 230, SOFT, "hand", 4, "c")


def scene_deeper(mv, ex):
    mv.scene("deeper")
    net = ex[3]["net"]
    p1, p2, p3 = net["conv1"][2], net["conv2"][2], net["conv3"][2]
    im, dd = page("5. Going deeper")
    base_y = 330
    stack(dd, im, 40, base_y - 80, 1, 180, "1 x 64 x 64", "the photo",
          div_rgb(ex[3]["inp"], 2.5))
    arrow(dd, (240, base_y), (330, base_y))
    stack(dd, im, 345, base_y - 40, 8, 128, "8 x 32 x 32", "layer 1: edges, light and dark",
          heat_rgb(p1[int(np.argmax(p1.std(axis=(1, 2))))]))
    arrow(dd, (510, base_y), (620, base_y))
    stack(dd, im, 640, base_y - 10, 16, 84, "16 x 16 x 16", "layer 2: pieces of fingers",
          heat_rgb(p2[int(np.argmax(p2.std(axis=(1, 2))))]))
    arrow(dd, (800, base_y), (900, base_y))
    stack(dd, im, 925, base_y + 10, 32, 56, "32 x 8 x 8", "layer 3: whole fingers",
          heat_rgb(p3[int(np.argmax(p3.std(axis=(1, 2))))]))
    caption(dd, "Each round is convolution, ReLU and pooling. The maps get smaller while there are more of "
                "them.", 620)
    mv.hold(im, 6, name="stacks")

    im, dd = page("5. Going deeper")
    text(dd, (60, 120), "layer 2: 16 maps, 16 x 16", 28, INK, "handb")
    for k in range(16):
        a, bb = divmod(k, 4)
        put(im, dd, heat_rgb(p2[k], p2.max()), (60 + bb * 112, 170 + a * 100), (92, 92), width=2)
    text(dd, (600, 120), "layer 3: 32 maps, 8 x 8", 28, INK, "handb")
    for k in range(32):
        a, bb = divmod(k, 8)
        put(im, dd, heat_rgb(p3[k], p3.max()), (600 + bb * 82, 170 + a * 100), (68, 68), width=2)
    caption(dd, "Every layer-2 filter looks at all 8 maps before it, so it can combine edges into finger "
                "shapes. Layer 3 combines those again.", 590)
    mv.hold(im, 6, name="maps")


def scene_output(mv, ex):
    mv.scene("output")
    net = ex[3]["net"]
    flat = net["flat"]
    fc = WTS["fc_w"]
    im, dd = page("6. From maps to an answer")
    for k in range(32):
        a, bb = divmod(k, 16)
        put(im, dd, heat_rgb(net["conv3"][2][k], flat.max()), (60 + bb * 72, 130 + a * 70), (56, 56), width=2)
    arrow(dd, (W // 2, 280), (W // 2, 330))
    strip = heat_rgb(flat, flat.max()).reshape(2, 1024, 3)
    put(im, dd, strip, (128, 350), (1024, 60))
    text(dd, (W // 2, 425), "one line of 32 x 8 x 8 = 2048 numbers", 26, INK, "handb", "ma")
    caption(dd, "Lay the 32 maps out as one long line of numbers.", 620)
    mv.hold(im, 4, name="flatten")

    im, dd = page("6. From maps to an answer")
    put(im, dd, strip, (128, 120), (1024, 40))
    text(dd, (128, 168), "the 2048 numbers", 20, SOFT)
    text(dd, (128, 205), "each count has its own 2048 weights:", 24, INK, "handb")
    for c in range(6):
        y = 245 + c * 52
        text(dd, (110, y + 18), f"count {c}", 22, INK, "hand", "rm")
        put(im, dd, div_rgb(fc[c]).reshape(1, 2048, 3), (128, y), (820, 36), width=2)
        s = net["logits"][c]
        dd.text((990, y + 18), f"score {s:+6.1f}", font=font(24, "mono"), fill=RED if c == 3 else INK,
                anchor="lm")
    caption(dd, "Multiply each number by its weight and add them up: one score per count.", 610)
    mv.hold(im, 5, name="scores")

    for c in (1, 2, 3, 4, 5):
        probs = ex[c]["net"]["probs"]
        for frac in (0.25, 0.5, 0.75, 1.0):
            im, dd = page("6. From maps to an answer")
            text(dd, (W // 2, 120), "softmax:  probability = e^score / (sum of e^scores)", 28, INK, "mono",
                 "ma")
            put(im, dd, gray_rgb(ex[c]["saved"].astype(np.float32), 0, 255), (150, 200), (300, 300))
            arrow(dd, (480, 350), (560, 350))
            bars(dd, 600, 520, probs, 480, 300, frac, 24)
            text(dd, (840, 560), "count", 22, SOFT, "hand", "ma")
            if frac == 1.0:
                text(dd, (300, 530), f"answer: {int(np.argmax(probs))}  ({probs.max():.0%} sure)", 30, RED,
                     "handb", "ma")
            mv.hold(im, frames=1 if frac < 1 else 16, name=f"bars{c}" if frac == 1 else None)


def contrib_map(net, c):
    a = net["conv3"][2]
    return (WTS["fc_w"][c].reshape(a.shape) * a).sum(axis=0)


def overlay(saved, cmap_8, size):
    photo = cv2.resize(saved, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    heat = cv2.resize(np.clip(cmap_8, 0, None), (size, size), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat / (heat.max() + 1e-9), 0, 1)[..., None]
    base = np.repeat(photo[..., None], 3, axis=2) * 255
    red = np.array([230, 40, 30], np.float32)
    out = base * (1 - 0.65 * heat) + red * 0.65 * heat
    return out.astype(np.uint8)


def scene_heat(mv, ex):
    mv.scene("heat")
    ph = ex[3]
    c = int(np.argmax(ph["net"]["probs"]))
    cm = contrib_map(ph["net"], c)
    im, dd = page("7. Where it looked")
    put(im, dd, gray_rgb(ph["saved"].astype(np.float32), 0, 255), (60, 150), (300, 300))
    arrow(dd, (380, 300), (440, 300))
    number_grid(dd, 460, 150, 37.5, cm, lambda v: f"{v:.1f}".replace("-0.0", "0.0"), div_rgb(cm), 11)
    arrow(dd, (780, 300), (840, 300))
    put(im, dd, overlay(ph["saved"], cm, 300), (860, 150), (300, 300))
    text(dd, (610, 470), f"score for {c}, split over the 8 x 8 grid", 22, SOFT, "hand", "ma")
    caption(dd, "Weights times layer-3 values, added up over the 32 maps at each spot. Red pushed the "
                "network toward its answer.", 560)
    mv.hold(im, 6, name="one")

    im, dd = page("7. Where it looked")
    for k, cc in enumerate((1, 2, 3, 4, 5)):
        e = ex[cc]
        pc = int(np.argmax(e["net"]["probs"]))
        put(im, dd, overlay(e["saved"], contrib_map(e["net"], pc), 210), (55 + k * 240, 200), (210, 210))
        text(dd, (160 + k * 240, 430), f"read as {pc}", 26, INK, "handb", "ma")
    caption(dd, "Often the red is on the fingers. Sometimes it's the empty space beside them, since a gap "
                "where a finger isn't is evidence too.", 540)
    mv.hold(im, 5, name="all")


def augment_np(small, r):
    """One random copy with the same ranges as dataset.augment: rotate up to 20 degrees, zoom 85-115%,
    shift up to 12%, mirror half the time, contrast 0.7-1.3 and brightness -0.15 to +0.15."""
    m = cv2.getRotationMatrix2D((32, 32), r.uniform(-20, 20), r.uniform(0.85, 1.15))
    m[:, 2] += r.uniform(-0.12, 0.12, 2) * 64
    out = cv2.warpAffine(small, m, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if r.random() < 0.5:
        out = out[:, ::-1]
    mean = out.mean()
    out = (out - mean) * r.uniform(0.7, 1.3) + mean + r.uniform(-0.15, 0.15)
    return np.clip(out, 0, 1)


def scene_train(mv, ex):
    mv.scene("train")
    ph = ex[3]
    r = np.random.default_rng(0)
    rand_w = {}
    for name in ("conv1", "conv2", "conv3", "fc"):
        wshape = WTS[f"{name}_w"].shape
        bound = 1 / np.sqrt(np.prod(wshape[1:]))          # PyTorch's default starting range
        rand_w[f"{name}_w"] = r.uniform(-bound, bound, wshape).astype(np.float32)
        rand_w[f"{name}_b"] = r.uniform(-bound, bound, WTS[f"{name}_b"].shape).astype(np.float32)
    p_rand = run_net(ph["inp"], rand_w)["probs"]
    p_true = ph["net"]["probs"]

    im, dd = page("8. How it learned")
    put(im, dd, gray_rgb(ph["saved"].astype(np.float32), 0, 255), (60, 170), (240, 240))
    text(dd, (180, 425), "the right answer: 3", 24, INK, "handb", "ma")
    text(dd, (580, 130), "random weights", 28, INK, "handb", "ma")
    bars(dd, 400, 450, p_rand, 360, 250, 1.0, 20)
    text(dd, (580, 500), f"loss = -log({p_rand[3]:.2f}) = {-np.log(p_rand[3]):.2f}", 24, INK, "mono", "ma")
    text(dd, (1010, 130), "after training", 28, INK, "handb", "ma")
    bars(dd, 830, 450, p_true, 360, 250, 1.0, 20)
    text(dd, (1010, 500), f"loss = -log({p_true[3]:.2f}) = {-np.log(p_true[3]):.2f}", 24, INK, "mono", "ma")
    caption(dd, "The loss is small when the network gives the right answer a high probability.", 600)
    mv.hold(im, 7, name="loss")

    im, dd = page("8. How it learned")
    text(dd, (W // 2, 160), "new weight = weight - learning rate x slope of the loss", 34, INK, "mono", "ma")
    # the loss as a bowl over one weight, and a ball stepping down it
    def to_xy(xv):
        return 380 + 230 * xv, 520 - 220 * xv * xv
    dd.line([to_xy(xv) for xv in np.linspace(-1, 1, 60)], fill=INK, width=3)
    steps = (0.9, 0.62, 0.4, 0.22, 0.1)
    for k, xv in enumerate(steps):
        px, py = to_xy(xv)
        dd.ellipse((px - 10, py - 18, px + 10, py + 2), fill=RED if k == 0 else (232, 150, 130))
        if k + 1 < len(steps):
            arrow(dd, (px - 8, py - 26), (to_xy(steps[k + 1])[0] + 8, to_xy(steps[k + 1])[1] - 26), SOFT, 2)
    text(dd, (380, 545), "one weight", 22, SOFT, "hand", "ma")
    text(dd, (130, 300), "loss", 22, SOFT, "hand", "ma")
    para(dd, (720, 260), "Backpropagation is the chain rule from my V-fit spreadsheet, run backwards through "
                         "every layer. It finds the slope of the loss for each of the 18,310 weights, and "
                         "gradient descent steps every weight downhill.", 26, 500)
    mv.hold(im, 8, name="descent")

    if True:
        aug = [augment_np(ph["small"], np.random.default_rng(10 + k)) for k in range(5)]
        im, dd = page("8. How it learned")
        put(im, dd, gray_rgb(ph["small"], 0, 1), (50, 200), (190, 190))
        text(dd, (145, 410), "original", 24, INK, "handb", "ma")
        for k in range(5):
            put(im, dd, gray_rgb(aug[k], 0, 1), (290 + k * 195, 200), (175, 175), width=2)
        text(dd, (770, 410), "five random copies: rotated, shifted, zoomed, flipped, brighter or darker", 22,
             SOFT, "hand", "ma")
        caption(dd, "The count of fingers doesn't change, so every copy is a new labeled example.", 560)
        mv.hold(im, 5, name="augment")

    summ = json.loads((config.RESULTS_DIR / "train_summary.json").read_text())
    im, dd = page("8. How it learned")
    curves = config.PLOTS_DIR / "training_curves.png"
    if curves.exists():
        cimg = Image.open(curves).convert("RGB")
        cimg.thumbnail((700, 430))
        im.paste(cimg, (50, 130))
        wrect(dd, (48, 128, 52 + cimg.width, 132 + cimg.height), INK, 3)
    y = 150
    for line in (f"{summ['n_train']} training photos", f"{summ['n_val']} test photos",
                 f"{summ['epochs']} passes, best at {summ['best_epoch']}", f"{summ['params']:,} weights",
                 f"{summ['val_acc']:.0%} of test photos right"):
        text(dd, (800, y), line, 30, INK, "handb")
        y += 56
    para(dd, (800, y + 10), "The test photos come from the same session as the training photos, so the real "
                            "box is a little less accurate.", 22, 440, SOFT)
    mv.hold(im, 6, name="results")


def draw_lock(view, reading, conf, progress, digits, blink_next, status, msg, arm_deg):
    im, dd = page("9. From counts to the lock")
    put(im, dd, gray_rgb(view.astype(np.float32), 0, 255), (60, 140), (300, 300))
    text(dd, (210, 455), "camera", 22, SOFT, "hand", "ma")
    if reading is not None:
        text(dd, (210, 490), f"reads {reading}  ({conf:.0%})", 28, GREEN if conf >= 0.8 else SOFT, "handb", "ma")
    wrect(dd, (60, 535, 360, 560), INK, 2)
    if progress > 0:
        dd.rectangle((62, 537, 62 + 296 * progress, 558), fill=YELLOW)
    text(dd, (210, 570), "hold for 1 s", 18, SOFT, "hand", "ma")

    # front panel
    wrect(dd, (420, 160, 860, 420), INK, 4, fill=(226, 205, 168))
    text(dd, (630, 175), "front of the box", 22, INK, "hand", "ma")
    for k in range(3):
        on = digits > k or (blink_next and k == digits)
        cx = 500 + k * 70
        dd.ellipse((cx - 22, 260, cx + 22, 304), fill=YELLOW if on else (120, 112, 96), outline=INK, width=2)
    dd.ellipse((718, 260, 762, 304), fill=GREEN if status == "green" else (120, 112, 96), outline=INK, width=2)
    dd.ellipse((778, 260, 822, 304), fill=RED if status == "red" else (120, 112, 96), outline=INK, width=2)
    text(dd, (570, 320), "digits", 20, INK, "hand", "ma")
    text(dd, (770, 320), "open   wrong", 20, INK, "hand", "ma")

    # latch from above
    cx, cy = 1050, 300
    wrect(dd, (900, 160, 1200, 440), INK, 3)
    text(dd, (1050, 450), "latch, from above", 22, SOFT, "hand", "ma")
    dd.rectangle((960, 390, 1140, 402), fill=SOFT)
    text(dd, (1050, 408), "catch", 18, SOFT, "hand", "ma")
    a = np.radians(arm_deg)
    tip = (cx + 95 * np.cos(a), cy + 95 * np.sin(a))
    dd.line([(cx, cy), tip], fill=(160, 110, 60), width=22)
    dd.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=INK)
    caption(dd, msg, 610)
    return im


def scene_lock(mv, ex):
    mv.scene("lock")
    empty = ex["empty"]["saved"]
    pe = ex["empty"]["net"]["probs"]
    locked, opened = 90, 180
    mv.hold(draw_lock(empty, None, 0, 0, 0, False, "off", "Locked. Press the button to start.", locked), 2,
            name="start")
    for k, digit in enumerate(EXAMPLE_CODE):
        for blink in (True, False, True):
            mv.hold(draw_lock(empty, 0, pe[0], 0, k, blink, "off",
                              "Hand away: the network has to read 0 first.", locked), 0.375)
        ph = ex[digit]
        p = ph["net"]["probs"]
        for prog in (0.25, 0.5, 0.75, 1.0):
            mv.hold(draw_lock(ph["saved"], digit, p[digit], prog, k, True, "off",
                              f"An example code: {', '.join(map(str, EXAMPLE_CODE))}. "
                              f"Hold {digit} for one second at 80% or more.", locked), 0.375)
        mv.hold(draw_lock(ph["saved"], digit, p[digit], 1.0, k + 1, False, "off", f"Digit {k + 1} accepted.",
                          locked), 1.0, name=f"digit{k + 1}")
    for ang in (locked, 120, 150, opened):
        mv.hold(draw_lock(empty, None, 0, 0, 3, False, "green", "Right code: the motor swings the latch open.",
                          ang), 0.5)
    mv.hold(draw_lock(empty, None, 0, 0, 3, False, "green", "Open!", opened), 2.5, name="open")


def scene_end(mv):
    mv.scene("end")
    im, dd = page()
    text(dd, (W // 2, 250), "Finger-count lock", 64, INK, "handb", "mm")
    wline(dd, (380, 295), (900, 295), RED, 5, 2)
    text(dd, (W // 2, 370), "ME193 Midterm 2", 34, INK, "hand", "mm")
    text(dd, (W // 2, 430), "PyTorch to train, numpy on the Arduino UNO Q", 28, SOFT, "hand", "mm")
    text(dd, (W // 2, 490), "github.com/iliketocode2/ME193-Robotics", 26, SOFT, "mono", "mm")
    mv.hold(im, 4, name="end")


# Script ---------------------------------------------------------------------------------------------------------

def stamp(frame):
    s = frame / FPS
    return f"{int(s // 60)}:{s % 60:04.1f}"


def write_script(scenes, total):
    lines = ["# Voice-over script for the stop-action movie", "",
             f"The video `nn_movie.mp4` is {stamp(total)} long, silent, at {FPS} frames a second. Each scene "
             "below gives its start and end time and the words to read over it. Every scene is long enough to "
             f"read its part at about {WORDS_PER_MIN} words a minute, with a short pause at the end.", "",
             "## Recording tips", "",
             "- Play the video full screen on one monitor and record yourself on Zoom (or any recorder), "
             "starting the recording at the same moment you press play.",
             "- Read each part when its timestamp comes up. If you finish early, stop and wait for the next "
             "scene, so the voice stays lined up with the pictures.",
             "- To put the audio on the video, drop both into Clipchamp or iMovie, or run this from the "
             "`movie` folder (it needs ffmpeg; the copy inside the Python environment works too):", "",
             "```",
             "ffmpeg -i nn_movie.mp4 -i voiceover.m4a -map 0:v -map 1:a -c:v copy -c:a aac -shortest "
             "nn_movie_voiced.mp4",
             "```", "",
             "- A quiet room and a headset mic sound better than a laptop mic.", "",
             "## Script", ""]
    for key, a, b in scenes:
        words = NARRATION.get(key, "")
        lines.append(f"### {stamp(a)} to {stamp(b)}: {SCENE_TITLES[key]}")
        lines.append("")
        lines.append(words if words else "(no narration, let the end card play)")
        lines.append("")
    (HERE / "voiceover_script.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="save one PNG per step to movie/preview/")
    args = ap.parse_args()

    preview = None
    if args.preview:
        preview = HERE / "preview"
        preview.mkdir(exist_ok=True)
        for p in preview.glob("*.png"):
            p.unlink()

    ex = pick_examples()
    for c in (1, 2, 3, 4, 5):
        assert int(np.argmax(ex[c]["net"]["probs"])) == c, f"no correctly read example for {c}"
    check = npnet.forward(WTS, ex[3]["inp"])
    assert np.allclose(check, ex[3]["net"]["probs"], atol=1e-5)

    mv = Movie(HERE / "nn_movie.mp4", preview)
    scene_title(mv, ex)
    scene_input(mv, ex)
    scene_conv(mv, ex)
    scene_relu(mv, ex)
    scene_pool(mv, ex)
    scene_deeper(mv, ex)
    scene_output(mv, ex)
    scene_heat(mv, ex)
    scene_train(mv, ex)
    scene_lock(mv, ex)
    scene_end(mv)
    mv.close()

    times = [{"scene": k, "title": SCENE_TITLES[k], "start_s": a / FPS, "end_s": b / FPS} for k, a, b in mv.scenes]
    (HERE / "scene_times.json").write_text(json.dumps(times, indent=2))
    write_script(mv.scenes, mv.frames)
    print(f"wrote {HERE / 'nn_movie.mp4'}: {mv.frames} frames, {stamp(mv.frames)}, encoder {mv.encoder}")
    for t in times:
        print(f"  {stamp(t['start_s'] * FPS)}-{stamp(t['end_s'] * FPS)}  {t['title']}")


if __name__ == "__main__":
    main()
