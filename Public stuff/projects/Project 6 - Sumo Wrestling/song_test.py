"""
song_test.py -- quick hardware test of the win / lose songs, laptop only.

The laptop connects DIRECTLY to the LEGO double motor and color sensor over
Bluetooth (no UNO Q, no MQTT, no camera) and opens a small window:

    Arrow keys   drive (hold to move, release to stop)
    Blue tape    color sensor reads blue -> robot stops, SAD song plays
                 (laptop mp3 + beeps on the double motor), driving locks
    W / button   WIN song
    R / button   reset (unlock driving after a loss)
    Space        stop
    close window stops the motors and disconnects

Songs are Project 3's (songs_bridge loads its songs.py): waka_waka_final.mp3 /
jb_sorry.mp3 on the laptop, SUCCESS_SONG / DEATH_SONG beeped on the motor.
Driving uses the same forward/turn correction as the brain (REVERSE_DRIVE in
config.py or a saved drive calibration), so "up" is the robot's front.

BEFORE RUNNING: the LEGO devices accept only one connection, so stop the UNO Q
relay first (board terminal: sudo systemctl stop sumo-relay -- or just power
the UNO Q off). Card serials come from unoq_app/python/sumo_topics.py.

Run:
    my_env\\Scripts\\python "Public stuff/projects/Project 6 - Sumo Wrestling/song_test.py"
"""

import threading
import time
import tkinter as tk

import legoeducation as le

import config  # noqa: F401  (puts unoq_app/python on sys.path for lelib + sumo_topics)
import drive_calibration as dcal
from lelib import colorSensor, doubleMotor
from songs_bridge import songs                   # Project 3's songs.py
from sumo_topics import COLOR_CARD_COLOR, COLOR_CARD_SERIAL, MOTOR_CARD_COLOR, MOTOR_CARD_SERIAL

DRIVE_SPEED = 50          # % for up/down
TURN_SPEED = 40           # % per wheel for left/right
LOOP_HZ = 20
BLUE_SAMPLES = 2          # consecutive blue readings needed (ignores a single glitch)
COLOR_NAMES = {le.LEGO_COLOR_NOCOLOR: "none", le.LEGO_COLOR_RED: "red", le.LEGO_COLOR_YELLOW: "yellow",
               le.LEGO_COLOR_BLUE: "blue", le.LEGO_COLOR_TEAL: "teal", le.LEGO_COLOR_GREEN: "green",
               le.LEGO_COLOR_PURPLE: "purple", le.LEGO_COLOR_WHITE: "white"}
SWATCH = {"blue": "#2563eb", "red": "#dc2626", "yellow": "#facc15", "green": "#16a34a", "teal": "#14b8a6",
          "purple": "#7c3aed", "white": "#f1f5f9", "none": "#334155"}


class Hardware(threading.Thread):
    """Owns ALL Bluetooth calls (one thread, like relay.py). The window only
    sets the flags below; this loop does the driving, sensing and songs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.held = set()                  # arrow keys currently held
        self.want_win = self.want_reset = False
        self.stop_event = threading.Event()
        self.status = "connecting..."
        self.color = "?"
        self.state = "connecting"          # connecting | ready | lost | error
        self.cal = dcal.load()

    def command(self):
        with self.lock:
            k = set(self.held)
        fwd = ("Up" in k) - ("Down" in k)
        turn = ("Right" in k) - ("Left" in k)
        l = fwd * DRIVE_SPEED + turn * TURN_SPEED
        r = fwd * DRIVE_SPEED - turn * TURN_SPEED
        return dcal.to_motor((l, r), self.cal)

    def run(self):
        motor, sensor = doubleMotor(), colorSensor()
        try:
            self.status = "connecting double motor..."
            motor.connect(card_serial=MOTOR_CARD_SERIAL, card_color=MOTOR_CARD_COLOR)
            self.status = "connecting color sensor..."
            sensor.connect(COLOR_CARD_SERIAL, card_color=COLOR_CARD_COLOR)
            sensor.device_notification_request(30)
            self.status, self.state = "connected - drive with the arrow keys", "ready"
            applied, blue_run = None, 0
            while not self.stop_event.is_set():
                # --- color sensor -> lose
                c = sensor.sensor.color
                name = COLOR_NAMES.get(c, "?") if c == c else "?"          # c != c when NaN (no reading yet)
                self.color = name
                blue_run = blue_run + 1 if name == "blue" else 0
                if blue_run >= BLUE_SAMPLES and self.state == "ready":
                    motor.movement_stop()
                    applied = (0, 0)
                    self.state, self.status = "lost", "BLUE! sad song... (press R to reset)"
                    songs.play_lose(motor)                     # laptop mp3 + DEATH_SONG beeps (blocking ~2 s)
                    self.status = "LOST - sad song played. Press R to reset."
                # --- win button
                with self.lock:
                    win, reset = self.want_win, self.want_reset
                    self.want_win = self.want_reset = False
                if win:
                    motor.movement_stop()
                    applied = (0, 0)
                    self.status = "WIN song..."
                    songs.play_win(motor)                      # laptop mp3 + SUCCESS_SONG beeps
                    self.status = "WIN song played" if self.state == "ready" else self.status
                if reset:
                    songs.stop_computer_mp3()
                    self.state, self.status = "ready", "reset - drive with the arrow keys"
                    blue_run = 0
                # --- driving (locked after a loss)
                target = self.command() if self.state == "ready" else (0, 0)
                if target != applied:
                    if target == (0, 0):
                        motor.movement_stop(blocking=False)
                    else:
                        motor.movement_move_tank(*target, blocking=False)
                    applied = target
                time.sleep(1 / LOOP_HZ)
        except Exception as e:
            self.state, self.status = "error", f"ERROR: {e}"
        finally:
            for dev, stop in ((motor, motor.movement_stop), (sensor, None)):
                try:
                    if dev.connected:
                        if stop:
                            stop()
                        dev.disconnect()
                except Exception:
                    pass
            songs.stop_computer_mp3()


def main():
    hw = Hardware()
    hw.start()

    root = tk.Tk()
    root.title("Sumo song test")
    root.configure(bg="#0f172a", padx=18, pady=16)
    font = ("Segoe UI", 11)
    tk.Label(root, text="Sumo song test", font=("Segoe UI", 16, "bold"), fg="#e2e8f0", bg="#0f172a").pack(anchor="w")
    status = tk.Label(root, text="", font=font, fg="#94a3b8", bg="#0f172a", wraplength=380, justify="left")
    status.pack(anchor="w", pady=(4, 12))
    row = tk.Frame(root, bg="#0f172a")
    row.pack(anchor="w", pady=(0, 12))
    swatch = tk.Label(row, width=4, height=2, bg="#334155")
    swatch.pack(side="left")
    color = tk.Label(row, text="color: ?", font=("Segoe UI", 13, "bold"), fg="#e2e8f0", bg="#0f172a")
    color.pack(side="left", padx=10)
    buttons = tk.Frame(root, bg="#0f172a")
    buttons.pack(anchor="w")

    def win():
        with hw.lock:
            hw.want_win = True

    def reset():
        with hw.lock:
            hw.want_reset = True

    tk.Button(buttons, text="Play WIN song (W)", font=font, command=win).pack(side="left")
    tk.Button(buttons, text="Reset (R)", font=font, command=reset).pack(side="left", padx=8)
    tk.Label(root, text="Arrows = drive · Space = stop · blue under the sensor = sad song",
             font=("Segoe UI", 9), fg="#64748b", bg="#0f172a").pack(anchor="w", pady=(12, 0))

    def key_down(e):
        if e.keysym in ("Up", "Down", "Left", "Right"):
            with hw.lock:
                hw.held.add(e.keysym)
        elif e.keysym.lower() == "w":
            win()
        elif e.keysym.lower() == "r":
            reset()
        elif e.keysym == "space":
            with hw.lock:
                hw.held.clear()

    def key_up(e):
        with hw.lock:
            hw.held.discard(e.keysym)

    def focus_lost(_):
        with hw.lock:                       # never keep driving after alt-tab
            hw.held.clear()

    root.bind("<KeyPress>", key_down)
    root.bind("<KeyRelease>", key_up)
    root.bind("<FocusOut>", focus_lost)

    def refresh():
        status.config(text=hw.status, fg="#f87171" if hw.state in ("lost", "error") else "#94a3b8")
        color.config(text=f"color: {hw.color}")
        swatch.config(bg=SWATCH.get(hw.color, "#334155"))
        root.after(100, refresh)

    def close():
        hw.stop_event.set()
        hw.join(timeout=3)                  # lets it stop the motors + disconnect
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    refresh()
    root.focus_force()
    root.mainloop()


if __name__ == "__main__":
    main()
