"""Draws the lock's wiring diagram to hardware/preview/wiring.png.

    python hardware/wiring.py

The pin numbers here are the ones the UNO Q sketch will use, so change both together.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "preview" / "wiring.png"

# (UNO Q pin, wire color, what it connects to)
PINS = [
    ("D9 (PWM)", "tab:purple", "ENA"),
    ("D7", "tab:orange", "IN1"),
    ("D8", "tab:brown", "IN2"),
    ("D2", "tab:cyan", "locked switch"),
    ("D4", "tab:olive", "open switch"),
    ("D12", "tab:pink", "button"),
    ("D3", "goldenrod", "digit LED 1"),
    ("D5", "goldenrod", "digit LED 2"),
    ("D6", "goldenrod", "digit LED 3"),
    ("D10", "green", "green LED"),
    ("D11", "red", "red LED"),
    ("GND", "black", "common ground"),
]


def block(ax, x, y, w, h, title, lines=(), color="#eef3f8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec="0.3"))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=10, weight="bold")
    for k, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.65 - 0.32 * k, s, ha="center", va="top", fontsize=7.5, color="0.25")


def wire(ax, pts, color, lw=1.8):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")


def main():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 20); ax.set_ylim(0, 13); ax.axis("off")

    # UNO Q with its pins down the right side
    block(ax, 0.5, 2.0, 3.6, 9.5, "Arduino UNO Q", ["3.3 V logic on the", "UNO headers"])
    pin_y = {}
    for k, (pin, color, _) in enumerate(PINS):
        y = 10.2 - 0.68 * k
        pin_y[pin] = y
        ax.plot(4.1, y, "o", color=color, ms=5)
        ax.text(3.95, y, pin, ha="right", va="center", fontsize=8)

    # Power for the UNO Q and the webcam
    block(ax, 0.5, 0.1, 3.6, 1.5, "USB-C hub with PD", ["5 V 3 A adapter in,", "webcam in, UNO Q out"],
          color="#f3f3f3")
    wire(ax, [(2.3, 1.6), (2.3, 2.0)], "0.3", 3)
    block(ax, 4.6, 0.1, 2.4, 1.5, "USB webcam", ["on the stand, 10 in", "above the hand"], color="#f3f3f3")
    wire(ax, [(4.1, 0.85), (4.6, 0.85)], "0.3", 3)

    color_of = dict((pin, c) for pin, c, _ in PINS)

    # Motor driver, motor, 12 V supply
    block(ax, 10.0, 7.7, 4.0, 4.3, "HW-095 (L298N)", color="#fdf1e3")
    ax.text(12.0, 8.25, "ENA jumper off, 5V-EN jumper on", ha="center", fontsize=7.5, color="0.25")
    ax.text(12.0, 7.95, "leave its 5V terminal unconnected", ha="center", fontsize=7.5, color="0.25")
    for k, name in enumerate(("ENA", "IN1", "IN2")):
        pin = ("D9 (PWM)", "D7", "D8")[k]
        y = pin_y[pin]
        ax.text(10.15, y, name, ha="left", va="center", fontsize=8)
        wire(ax, [(4.1, y), (10.0, y)], color_of[pin])
    for name, y in (("12V", 11.0), ("GND", 10.4), ("OUT1", 9.4), ("OUT2", 8.8)):
        ax.text(13.85, y, name, ha="right", va="center", fontsize=8)

    block(ax, 15.6, 10.2, 3.9, 1.9, "12 V supply", ["1 A or more", "(barrel adapter or bench supply)"],
          color="#fdecea")
    wire(ax, [(14.0, 11.0), (15.6, 11.0)], "red", 2.5)
    wire(ax, [(14.0, 10.4), (15.6, 10.4)], "black", 2.5)
    ax.text(14.8, 11.15, "+12 V", ha="center", fontsize=7.5, color="red")

    block(ax, 15.6, 7.6, 3.9, 2.0, "12 V gearmotor", ["Geartisan ZGB37RG", "swap its two wires if it",
                                                       "turns the wrong way"], color="#fdecea")
    wire(ax, [(14.0, 9.4), (15.6, 9.4)], "0.4", 2.5)
    wire(ax, [(14.0, 8.8), (15.6, 8.8)], "0.4", 2.5)

    # Common ground: UNO Q GND to the driver's GND
    gy = pin_y["GND"]
    wire(ax, [(4.1, gy), (9.6, gy), (9.6, 7.45), (14.6, 7.45), (14.6, 10.4)], "black", 2.2)
    ax.text(12.0, 7.25, "common ground: the UNO Q and the 12 V side share GND", ha="center", fontsize=7.5)

    # Inputs: each switch or button sits between its pin and GND and uses the internal pull-up
    block(ax, 10.0, 4.5, 4.0, 2.55, "Inputs", color="#e8f5ec")
    ax.text(12.0, 4.65, "other side of each one goes to GND", ha="center", fontsize=7.5, color="0.25")
    for k, (pin, label) in enumerate((("D2", "locked switch (C and NO)"), ("D4", "open switch (C and NO)"),
                                      ("D12", "push button"))):
        y = (6.3, 5.8, 5.1)[k]
        x_bend = 6.0 + 0.3 * k
        wire(ax, [(4.1, pin_y[pin]), (x_bend, pin_y[pin]), (x_bend, y), (10.0, y)], color_of[pin])
        ax.text(10.15, y, label, ha="left", va="center", fontsize=8)

    # LEDs: pin, 150 ohm resistor, LED, GND
    block(ax, 10.0, 0.1, 9.5, 4.1, "Front-panel LEDs",
          ["pin -> 150 ohm -> LED long leg; short leg -> GND (about 8 mA at 3.3 V)"], color="#fffbe6")
    leds = (("D3", "digit 1 (yellow)"), ("D5", "digit 2 (yellow)"), ("D6", "digit 3 (yellow)"),
            ("D10", "open (green)"), ("D11", "wrong code or lockout (red)"))
    for k, (pin, label) in enumerate(leds):
        y = 2.95 - 0.6 * k
        x_bend = 7.2 + 0.25 * k
        wire(ax, [(4.1, pin_y[pin]), (x_bend, pin_y[pin]), (x_bend, y), (11.6, y)], color_of[pin])
        ax.add_patch(plt.Rectangle((11.6, y - 0.1), 0.8, 0.2, fc="tan", ec="0.3"))
        ax.text(12.0, y + 0.13, "150", ha="center", va="bottom", fontsize=6.5)
        wire(ax, [(12.4, y), (12.9, y)], color_of[pin])
        ax.plot(13.05, y, marker=">", ms=9, color=color_of[pin], mec="0.2")
        ax.text(13.4, y, label, fontsize=8, va="center")

    ax.set_title("Finger-count lock wiring", fontsize=14, weight="bold")
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
