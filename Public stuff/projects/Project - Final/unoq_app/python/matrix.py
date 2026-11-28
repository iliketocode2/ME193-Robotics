"""Text and icons for the UNO Q's 8 x 13 LED matrix.

Everything here is pure Python + numpy so it can be tested on a laptop.
A "frame" is an 8 x 13 uint8 array of brightness values (0-7); main.py
turns it into board bytes with arduino.app_utils.Frame and sends it to the
sketch over the Bridge.

Text uses the classic 5 x 7 LCD font (one byte per column, bit 0 = top row,
bit 7 = descender row). Glyphs are trimmed to their used columns so short
words fit: "Hi" is 9 columns wide and sits still in the middle of the
matrix; anything wider than 13 columns scrolls right to left.
"""

import numpy as np

ROWS = 8
COLS = 13
ON = 7  # brightness of a lit pixel (0-7)

# 5 x 7 font for ASCII 0x20 (space) .. 0x7E (~), five column bytes per glyph.
_FONT_HEX = (
    "0000000000 00005F0000 0007000700 147F147F14 242A7F2A12 2313086462 3649562050 0008070300"
    " 001C224100 0041221C00 2A1C7F1C2A 08083E0808 0080703000 0808080808 0000606000 2010080402"
    " 3E5149453E 00427F4000 7249494946 2141494D33 1814127F10 2745454539 3C4A494931 4121110907"
    " 3649494936 464949291E 0000140000 0040340000 0008142241 1414141414 0041221408 0201590906"
    " 3E415D594E 7C1211127C 7F49494936 3E41414122 7F4141413E 7F49494941 7F09090901 3E41415173"
    " 7F0808087F 00417F4100 2040413F01 7F08142241 7F40404040 7F021C027F 7F0408107F 3E4141413E"
    " 7F09090906 3E4151215E 7F09192946 2649494932 03017F0103 3F4040403F 1F2040201F 3F4038403F"
    " 6314081463 0304780403 6159494D43 007F414141 0204081020 004141417F 0402010204 4040404040"
    " 0003070800 2054547840 7F28444438 3844444428 384444287F 3854545418 00087E0902 18A4A49C78"
    " 7F08040478 00447D4000 2040403D00 7F10284400 00417F4000 7C04780478 7C08040478 3844444438"
    " FC18242418 18242418FC 7C08040408 4854545424 04043F4424 3C4040207C 1C2040201C 3C4030403C"
    " 4428102844 4C9090907C 4464544C44 0008364100 0000770000 0041360800 0201020402"
)
FONT = {}
for _i, _g in enumerate(_FONT_HEX.split()):
    FONT[chr(0x20 + _i)] = [int(_g[j:j + 2], 16) for j in range(0, 10, 2)]
assert len(FONT) == 95, len(FONT)

# 8 x 13 icons, one string per row ("#" = lit).
ICONS = {
    "heart": [
        "...##...##...",
        "..####.####..",
        "..#########..",
        "..#########..",
        "...#######...",
        "....#####....",
        ".....###.....",
        "......#......",
    ],
    "smile": [
        "....#####....",
        "...#.....#...",
        "..#.#...#.#..",
        "..#.......#..",
        "..#.#...#.#..",
        "..#..###..#..",
        "...#.....#...",
        "....#####....",
    ],
    "sad": [
        "....#####....",
        "...#.....#...",
        "..#.#...#.#..",
        "..#.......#..",
        "..#..###..#..",
        "..#.#...#.#..",
        "...#.....#...",
        "....#####....",
    ],
    "check": [
        ".............",
        "..........#..",
        ".........#...",
        "........#....",
        "..#....#.....",
        "...#..#......",
        "....##.......",
        ".............",
    ],
    "x": [
        "...#.....#...",
        "....#...#....",
        ".....#.#.....",
        "......#......",
        ".....#.#.....",
        "....#...#....",
        "...#.....#...",
        ".............",
    ],
    "arrow_up": [
        "......#......",
        ".....###.....",
        "....#.#.#....",
        "...#..#..#...",
        "......#......",
        "......#......",
        "......#......",
        "......#......",
    ],
    "arrow_down": [
        "......#......",
        "......#......",
        "......#......",
        "......#......",
        "...#..#..#...",
        "....#.#.#....",
        ".....###.....",
        "......#......",
    ],
    "arrow_left": [
        ".............",
        "...#.........",
        "..#..........",
        ".#########...",
        "..#..........",
        "...#.........",
        ".............",
        ".............",
    ],
    "arrow_right": [
        ".............",
        ".........#...",
        "..........#..",
        "...#########.",
        "..........#..",
        ".........#...",
        ".............",
        ".............",
    ],
}
ICON_NAMES = sorted(ICONS)


def blank():
    return np.zeros((ROWS, COLS), dtype=np.uint8)


def icon_frame(name):
    frame = blank()
    for r, row in enumerate(ICONS[name]):
        for c, ch in enumerate(row):
            if ch == "#":
                frame[r, c] = ON
    return frame


def _glyph_columns(ch):
    """Column bytes for one character, trimmed to the columns it uses."""
    cols = FONT.get(ch, FONT["?"])
    if ch == " ":
        return [0, 0, 0]
    while cols and cols[0] == 0:
        cols = cols[1:]
    while cols and cols[-1] == 0:
        cols = cols[:-1]
    return cols


def text_strip(text):
    """Render text into an 8-row bitmap as wide as it needs to be."""
    columns = []
    for i, ch in enumerate(text):
        if i:
            columns.append(0)  # one blank column between letters
        columns.extend(_glyph_columns(ch))
    strip = np.zeros((ROWS, max(1, len(columns))), dtype=np.uint8)
    for c, byte in enumerate(columns):
        for r in range(ROWS):
            if byte >> r & 1:
                strip[r, c] = ON
    return strip


class TextScene:
    """Static text if it fits on the matrix, otherwise a right-to-left scroll.

    frame(t) gives the frame to show t seconds after the scene started.
    shown_once(t) is True once every letter has been on screen, so a
    command like "type Hi and then tell me you're done" only reports done
    after the user could actually read it.
    """

    SCROLL_COLS_PER_S = 9.0

    def __init__(self, text):
        self.text = text
        self.strip = text_strip(text)
        self.width = self.strip.shape[1]
        self.scrolls = self.width > COLS
        if not self.scrolls:
            self._static = blank()
            left = (COLS - self.width) // 2
            self._static[:, left:left + self.width] = self.strip
        else:
            gap = np.zeros((ROWS, COLS), dtype=np.uint8)
            # Enter from the right edge, leave through the left, repeat.
            self._loop = np.concatenate([gap, self.strip], axis=1)

    def frame(self, t):
        if not self.scrolls:
            return self._static
        offset = int(t * self.SCROLL_COLS_PER_S) % self._loop.shape[1]
        window = np.roll(self._loop, -offset, axis=1)[:, :COLS]
        return window.copy()

    def duration_once(self):
        """Seconds until the last letter has fully crossed the screen."""
        if not self.scrolls:
            return 0.0
        return (COLS + self.width) / self.SCROLL_COLS_PER_S

    def shown_once(self, t):
        return t >= self.duration_once()


def ascii_art(frame):
    return "\n".join("".join("#" if v else "." for v in row) for row in frame)


if __name__ == "__main__":
    print(ascii_art(TextScene("Hi").frame(0)))
    print()
    print(ascii_art(text_strip("ABCDEFGHIJKLM")))
    print(ascii_art(text_strip("NOPQRSTUVWXYZ")))
    print(ascii_art(text_strip("abcdefghijklm")))
    print(ascii_art(text_strip("nopqrstuvwxyz")))
    print(ascii_art(text_strip("0123456789 !?.,:-+")))
    print(ascii_art(text_strip("yaw=-12.5 deg (C) [x] {y} @#$%&*/\\'\"<>=^_`|~;")))
    for name in ICON_NAMES:
        print(name)
        print(ascii_art(icon_frame(name)))
