"""Virtual devices (PyScript glue).

"+ Add virtual device" in the Devices card adds a pretend Controller, Double Motor,
Single Motor or Color Sensor next to (or instead of) real hardware, so a student
without a full kit can still record data, train and run. A virtual device looks
like a connected one to the rest of the page: it is in state.devices, appears in
every Input/Output dropdown, exposes `state` channels with the same names as the
real thing, and answers run_output() (set_speed, set_speedL, set_speedR, stop).

The numbers (key -> stick targets, ramping, motor position) live in
virtual_logic.py, which is unit-tested with plain Python.

Virtual controller keyboard (works whenever you are not typing in a box):
    arrow keys  drive like a car (up/down forward/back, left/right turn)
    W / S       left stick          I / K   right stick
"""
import asyncio
import time

from pyscript import document, window  # noqa: F401  (window kept for parity with other modules)
from pyscript.ffi import create_proxy

import state
from state import get_id
import Device
import virtual_logic as vl

TICK_S = 0.05

_pressed: set = set()      # normalized keys currently held
_uid = 0
_loop_running = False
_listeners_installed = False


# ── Base ─────────────────────────────────────────────────────────────────────

class VirtualBase:
    kind = ""
    base_name = ""

    def __init__(self, name: str, uid: int):
        self.name = name
        self.uid = uid
        self.state: dict = {}
        self.plots: list = []
        self.plot_vars: list = []
        self.battery = None
        self.is_virtual = True

    # What this device can be told to do. Real-device methods that don't apply are no-ops.
    def get_out_list(self):
        return []

    def set_speed(self, speed): pass
    def set_speedL(self, speed): pass
    def set_speedR(self, speed): pass
    def set_light(self, variable, value): pass
    def set_beep(self, variable, value): pass
    def stop(self): pass

    # Per-kind pieces
    def body_html(self) -> str: return ""
    def bind(self): pass
    def tick(self, dt: float): pass

    def el(self, suffix: str):
        return get_id(f"vdev-{self.uid}-{suffix}")

    def read(self, var):
        return self.state.get(var, 0)

    def push_plots(self):
        """Like a real device's notification: feed every live plot attached to this device."""
        for graph, var in zip(self.plots, self.plot_vars):
            val = self.read(var)
            if val is None:
                continue
            Device.show_reading(graph.id, val)
            graph.updatePlot(graph.addPoints(1, [val]))

    def build_widget(self, container):
        card = document.createElement("div")
        card.className = "vdev-card"
        card.id = f"vdev-{self.uid}"
        card.innerHTML = (
            f'<div class="vdev-header"><span class="vdev-title">{self.name}</span>'
            f'<button class="vdev-remove" id="vdev-{self.uid}-remove" title="Remove this virtual device">&times;</button></div>'
            f'<div class="vdev-body">{self.body_html()}</div>'
        )
        container.appendChild(card)
        self.el("remove").addEventListener("click", create_proxy(lambda evt: remove(self)))
        self.bind()


def _spin_blade(blade, speed):
    """Same look as the original virtual motor: spin rate follows speed."""
    if not blade:
        return
    seconds = vl.spin_seconds(speed)
    if seconds is None:
        blade.style.animationPlayState = "paused"
        return
    blade.style.animationPlayState = "running"
    blade.style.animationDuration = f"{seconds}s"
    blade.style.animationDirection = "reverse" if speed < 0 else "normal"


def _motor_html(uid: int, side: str, label: str) -> str:
    return (
        f'<div class="virtual-motor"><div class="virtual-motor-body">'
        f'<div class="virtual-motor-blade" id="vdev-{uid}-blade-{side}"></div></div>'
        f'<span class="virtual-motor-label">{label} <span id="vdev-{uid}-val-{side}">0</span></span></div>'
    )


# ── Controller ───────────────────────────────────────────────────────────────

class VirtualController(VirtualBase):
    kind = "controller"
    base_name = "Virtual Controller"

    def __init__(self, name, uid):
        super().__init__(name, uid)
        self.state = {"LeftPercent": 0, "RightPercent": 0}
        self.values = {"left": 0.0, "right": 0.0}
        self.dragging = {"left": False, "right": False}

    def body_html(self):
        def col(side, label):
            return (f'<div class="virtual-slider-col"><div class="vslider-wrap">'
                    f'<input type="range" class="vslider" id="vdev-{self.uid}-{side}" min="-100" max="100" value="0" step="1" />'
                    f'</div><label class="virtual-slider-label" for="vdev-{self.uid}-{side}">{label}</label>'
                    f'<span class="virtual-slider-value" id="vdev-{self.uid}-{side}-value">0</span></div>')
        return (f'<div class="virtual-slider-pair">{col("left", "Left")}{col("right", "Right")}</div>'
                '<p class="vdev-help"><b>Keyboard</b><br>'
                'Arrow keys: drive (up/down = forward/back, left/right = turn)<br>'
                'W / S: left stick &nbsp; I / K: right stick</p>')

    def bind(self):
        for side in ("left", "right"):
            slider = self.el(side)

            def on_input(evt, side=side):
                self.dragging[side] = True
                self.set_stick(side, float(evt.target.value), from_slider=True)

            def on_release(evt, side=side):
                self.dragging[side] = False     # tick() springs it back to centre

            slider.addEventListener("input", create_proxy(on_input))
            for name in ("change", "pointerup", "pointercancel"):
                slider.addEventListener(name, create_proxy(on_release))

    def set_stick(self, side, value, from_slider=False):
        value = max(-100.0, min(100.0, float(value)))
        shown = int(round(value))
        self.values[side] = value
        self.state["LeftPercent" if side == "left" else "RightPercent"] = shown
        if not from_slider:
            slider = self.el(side)
            if slider:
                slider.value = str(shown)
        label = self.el(f"{side}-value")
        if label:
            label.textContent = str(shown)

    def tick(self, dt):
        keyed = vl.has_control_key(_pressed)
        targets = dict(zip(("left", "right"), vl.key_targets(_pressed) if keyed else (0.0, 0.0)))
        for side in ("left", "right"):
            if self.dragging[side]:
                continue
            current = self.values[side]
            new = vl.ramp(current, targets[side], dt)
            if new != current:
                self.set_stick(side, new)


# ── Motors ───────────────────────────────────────────────────────────────────

class VirtualDoubleMotor(VirtualBase):
    """Two wheels. Mirrored like the real chassis: both wheels drive forward when
    LeftSpeed and RightSpeed have opposite signs (the example networks rely on this)."""
    kind = "double_motor"
    base_name = "Virtual Double Motor"

    def __init__(self, name, uid):
        super().__init__(name, uid)
        self.state = {
            "LeftPosition": 0.0, "RightPosition": 0.0, "LeftAngle": 0.0, "RightAngle": 0.0,
            "LeftSpeed": 0, "RightSpeed": 0, "LeftPower": 0, "RightPower": 0,
            "BothSpeed": 0,
        }

    def get_out_list(self):
        return ["LeftSpeed", "RightSpeed", "BothSpeed"]

    def body_html(self):
        return f'<div class="vdev-wheels">{_motor_html(self.uid, "left", "Left")}{_motor_html(self.uid, "right", "Right")}</div>'

    def _set_side(self, side, speed):
        self.state[f"{side}Speed"] = speed
        self.state[f"{side}Power"] = speed
        self.state["BothSpeed"] = (self.state["LeftSpeed"] - self.state["RightSpeed"]) / 2
        label = self.el(f"val-{side.lower()}")
        if label:
            label.textContent = str(speed)
        _spin_blade(self.el(f"blade-{side.lower()}"), speed)

    def set_speed(self, speed):
        speed = vl.clamp_speed(speed)
        self._set_side("Left", speed)
        self._set_side("Right", -speed)

    def set_speedL(self, speed):
        self._set_side("Left", vl.clamp_speed(speed))

    def set_speedR(self, speed):
        self._set_side("Right", vl.clamp_speed(speed))

    def stop(self):
        self._set_side("Left", 0)
        self._set_side("Right", 0)

    def tick(self, dt):
        for side in ("Left", "Right"):
            speed = self.state[f"{side}Speed"]
            if speed:
                pos = vl.advance_position(self.state[f"{side}Position"], speed, dt)
                self.state[f"{side}Position"] = pos
                self.state[f"{side}Angle"] = vl.wrap_angle(pos)


class VirtualSingleMotor(VirtualBase):
    kind = "single_motor"
    base_name = "Virtual Single Motor"

    def __init__(self, name, uid):
        super().__init__(name, uid)
        self.state = {"Position": 0.0, "Speed": 0, "Angle": 0.0, "Power": 0}

    def get_out_list(self):
        return ["Speed"]

    def body_html(self):
        return _motor_html(self.uid, "main", "Speed")

    def set_speed(self, speed):
        speed = vl.clamp_speed(speed)
        self.state["Speed"] = speed
        self.state["Power"] = speed
        label = self.el("val-main")
        if label:
            label.textContent = str(speed)
        _spin_blade(self.el("blade-main"), speed)

    def stop(self):
        self.set_speed(0)

    def tick(self, dt):
        speed = self.state["Speed"]
        if speed:
            pos = vl.advance_position(self.state["Position"], speed, dt)
            self.state["Position"] = pos
            self.state["Angle"] = vl.wrap_angle(pos)


# ── Sensor ───────────────────────────────────────────────────────────────────

class VirtualColorSensor(VirtualBase):
    """Reflection only: slide it to change what the sensor "sees" (0 = dark, 100 = bright)."""
    kind = "color_sensor"
    base_name = "Virtual Color Sensor"

    def __init__(self, name, uid):
        super().__init__(name, uid)
        self.state = {"Reflection": 50}

    def body_html(self):
        return (f'<div class="vdev-sensor"><div class="vdev-swatch" id="vdev-{self.uid}-swatch"></div>'
                f'<input type="range" class="vdev-hslider" id="vdev-{self.uid}-slider" min="0" max="100" value="50" step="1" />'
                f'<span class="virtual-motor-label">Reflection <span id="vdev-{self.uid}-val">50</span></span>'
                f'<span class="vdev-help-faint">0 = dark, 100 = bright</span></div>')

    def bind(self):
        self._show(50)

        def on_input(evt):
            self.state["Reflection"] = int(float(evt.target.value))
            self._show(self.state["Reflection"])

        self.el("slider").addEventListener("input", create_proxy(on_input))

    def _show(self, value):
        label = self.el("val")
        if label:
            label.textContent = str(value)
        swatch = self.el("swatch")
        if swatch:
            g = int(round(value * 2.55))
            swatch.style.background = f"rgb({g},{g},{g})"


KINDS = {cls.kind: cls for cls in (VirtualController, VirtualDoubleMotor, VirtualSingleMotor, VirtualColorSensor)}


# ── Adding / removing / the shared update loop ───────────────────────────────

def virtual_devices() -> list:
    return [d for d in state.devices if getattr(d, "is_virtual", False)]

def add(kind: str):
    global _uid
    cls = KINDS.get(kind)
    container = get_id("virtual-devices")
    if cls is None or container is None:
        return None
    import sync
    _uid += 1
    name = Device._dedupe_name(cls.base_name, [d.name for d in state.devices])
    dev = cls(name, _uid)
    state.devices.append(dev)
    dev.build_widget(container)
    install_listeners()
    _ensure_loop()
    sync.refresh_device_dropdowns()
    return dev

_status_generation = 0

def announce(text: str, seconds: float = 3.0):
    """Show a short confirmation next to the Add button, then clear it."""
    global _status_generation
    _status_generation += 1
    mine = _status_generation
    el = get_id("add-virtual-status")
    if el:
        el.textContent = text

    async def clear():
        await asyncio.sleep(seconds)
        if mine == _status_generation and el:      # a newer message keeps its own timer
            el.textContent = ""

    asyncio.ensure_future(clear())

def remove(dev):
    import sync
    try:
        dev.stop()
    except Exception as e:
        print("Could not stop " + dev.name + ": " + str(e))
    card = get_id(f"vdev-{dev.uid}")
    if card:
        card.remove()
    if dev in state.devices:
        state.devices.remove(dev)
    sync.refresh_device_dropdowns()

def _ensure_loop():
    global _loop_running
    if not _loop_running:
        _loop_running = True
        asyncio.ensure_future(_loop())

async def _loop():
    global _loop_running
    last = time.monotonic()
    try:
        while virtual_devices():
            now = time.monotonic()
            dt = min(now - last, 0.2)
            last = now
            for dev in list(virtual_devices()):
                dev.tick(dt)
                dev.push_plots()
            await asyncio.sleep(TICK_S)
    finally:
        _loop_running = False


# ── Keyboard ─────────────────────────────────────────────────────────────────

_TEXT_INPUT_TYPES = ("text", "number", "search", "email", "password", "tel", "url")

def _typing_in_field(el) -> bool:
    """Don't steal keys from a text box (arrows edit a number box)."""
    try:
        tag = (el.tagName or "").upper()
        if tag == "TEXTAREA":
            return True
        if tag == "INPUT":
            return (el.type or "text").lower() in _TEXT_INPUT_TYPES
        return bool(el.isContentEditable)
    except Exception:
        return False

def _on_keydown(evt):
    if evt.ctrlKey or evt.metaKey or evt.altKey:      # leave Cmd/Ctrl shortcuts alone
        return
    if not any(isinstance(d, VirtualController) for d in virtual_devices()):
        return
    if _typing_in_field(evt.target):
        return
    key = vl.normalize_key(evt.key)
    if key in vl.STICK_KEYS:
        _pressed.add(key)
        evt.preventDefault()                           # no page scroll / dropdown change

def _on_keyup(evt):
    _pressed.discard(vl.normalize_key(evt.key))

def _on_blur(evt=None):
    _pressed.clear()                                   # never leave a key "stuck" down

def _on_change(evt):
    """After picking from a dropdown it keeps focus, and arrow keys would then change the
    dropdown instead of driving. Drop the focus so the keyboard goes back to the sticks."""
    try:
        if (evt.target.tagName or "").upper() == "SELECT":
            evt.target.blur()
    except Exception:
        pass

def install_listeners():
    global _listeners_installed
    if _listeners_installed:
        return
    _listeners_installed = True
    window.addEventListener("keydown", create_proxy(_on_keydown))
    window.addEventListener("keyup", create_proxy(_on_keyup))
    window.addEventListener("blur", create_proxy(_on_blur))
    document.addEventListener("change", create_proxy(_on_change))
