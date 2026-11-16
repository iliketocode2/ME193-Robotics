"""
AnnaCode Table Cloner - main.py  (PyScript)

Behavior cloning with a lookup table, for a car made of a Double Motor and three
Color Sensors (left, front, right).

  1. Connect the hubs (each is its own Bluetooth device) and a Controller.
  2. Press Record and drive with the joysticks. Each tenth of a second the page notes
     which sensors see an object (one of 8 patterns) and what you did (Forward, Left,
     Right, Back) and adds a tally to the table.
  3. Press Run. The car looks up its pattern and does the most-tallied action.
     Push a stick to take over; with "Add my corrections" on, what you do is tallied
     too (the DAgger idea), so the table improves where the car actually goes wrong.

All the decisions are in table_logic.py (pure, unit-tested). This file is the wiring.
"""
import asyncio
import json
import time

import legoeducation as le
from pyscript import document, when, window

import state
from state import get_id
import Device
import sync
import power
import table_logic as tl
import teleop_logic as tele

TICK_S = 0.1          # one table sample / one motor update
RESEND_TICKS = 5      # repeat an unchanged motor command every 0.5 s in case a write was dropped
STORAGE_KEY = "annacode-table-cloner"

table = tl.BehaviorTable()
mode = "idle"                 # "idle" | "record" | "run"
sent = None                   # (left, right, ticks since sent) last motor command
table_version = 0             # bumped when counts change, so the table HTML is only rebuilt then
drawn_version = -1
last_pattern = None
step_trigger = tl.StepTrigger()   # step mode: one joystick push = one move
step_move = None              # step mode: the move the hub is carrying out: {action, clock}


# ── Saving ───────────────────────────────────────────────────────────────────

def save_table():
    try:
        window.localStorage.setItem(STORAGE_KEY, json.dumps(table.to_dict()))
    except Exception as e:
        print("Could not save the table: " + str(e))

def load_table():
    global table
    try:
        raw = window.localStorage.getItem(STORAGE_KEY)
        if raw:
            table = tl.BehaviorTable.from_dict(json.loads(raw))
    except Exception as e:
        print("Could not load the saved table: " + str(e))


# ── Reading the world ────────────────────────────────────────────────────────

def settings() -> dict:
    def number(elem_id, default):
        el = get_id(elem_id)
        try:
            return float(el.value) if el and str(el.value).strip() != "" else default
        except (ValueError, TypeError):
            return default

    def checked(elem_id, default=False):
        el = get_id(elem_id)
        return bool(el.checked) if el else default

    bright = get_id("object-is")
    return {
        "threshold": number("threshold", tl.DEFAULT_THRESHOLD),
        "bright": (bright.value if bright else "bright") == "bright",
        "speed": number("speed", tl.DEFAULT_SPEED),
        "reverse": checked("reverse"),
        "correct": checked("add-corrections", True),
        "step": checked("step-mode"),
        "car_length": number("car-length", tl.DEFAULT_CAR_LENGTH_DEG),
    }

def read_sensors() -> list:
    """Reflection from the Left, Front and Right sensors (None if unassigned or the link dropped)."""
    out = []
    for sel_id in sync.SENSOR_SELECTS:
        sel = get_id(sel_id)
        dev = Device.device_by_name(sel.value) if sel and sel.value else None
        if dev is None or Device.link_dropped(dev):
            out.append(None)
        else:
            out.append(dev.state.get("Reflection"))
    return out

def find_controller():
    for dev in state.devices:
        dev_state = getattr(dev, "state", None)
        if isinstance(dev_state, dict) and "LeftPercent" in dev_state:
            return dev
    return None

def read_sticks() -> tuple[float, float]:
    ctrl = find_controller()
    if ctrl is None or Device.link_dropped(ctrl):   # a lost controller keeps stale values: treat as centred
        return 0.0, 0.0
    return tele.stick_values(ctrl.state)

def motor_device():
    sel = get_id(sync.MOTOR_SELECT)
    return Device.device_by_name(sel.value) if sel and sel.value else None


# ── Driving ──────────────────────────────────────────────────────────────────

def drive(left: int, right: int):
    """Send (LeftSpeed, RightSpeed), skipping repeats of an unchanged command."""
    global sent
    dev = motor_device()
    if dev is None:
        return
    if sent is not None and sent[0] == left and sent[1] == right and sent[2] < RESEND_TICKS:
        sent = (left, right, sent[2] + 1)
        return
    Device.run_output("LeftSpeed", dev.name, left)
    Device.run_output("RightSpeed", dev.name, right)
    sent = (left, right, 0)

def halt():
    global sent
    sent = None
    dev = motor_device()
    if dev is not None and not Device.link_dropped(dev):
        try:
            dev.stop()
        except Exception as e:
            print("Could not stop the motors: " + str(e))

def human_commands(sl, sr, s):
    return (tele.command_for("LeftSpeed", sl, sr, s["speed"], s["reverse"]),
            tele.command_for("RightSpeed", sl, sr, s["speed"], s["reverse"]))


# ── One tick ─────────────────────────────────────────────────────────────────

def tick():
    global table_version, last_pattern
    s = settings()
    readings = read_sensors()
    bits = [tl.detect(r, s["threshold"], s["bright"]) for r in readings]
    pattern = tl.pattern_index(*bits)
    last_pattern = pattern

    sl, sr = read_sticks()
    human = (tl.classify_step if s["step"] else tl.classify_action)(sl, sr)
    predicted, source = table.predict(pattern)
    driver = "none"

    if s["step"]:
        driver = step_tick(s, pattern, sl, sr, predicted, source)
    elif mode == "record":
        driver = "you"
        if human is not None:
            table.add(pattern, human)
            table_version += 1
            drive(*human_commands(sl, sr, s))
        else:
            drive(0, 0)
    elif mode == "run":
        if human is not None:
            driver = "you"
            drive(*human_commands(sl, sr, s))
            if s["correct"]:
                table.add(pattern, human)
                table_version += 1
        else:
            driver = "table"
            drive(*tl.motor_commands(predicted, s["speed"], s["reverse"]))
    else:
        if human is not None:           # idle: drive freely without recording
            drive(*human_commands(sl, sr, s))
        elif sent is not None and (sent[0] or sent[1]):
            drive(0, 0)

    render_live(readings, bits, pattern, s, human, predicted, source, driver)
    if step_move is not None:
        set_text("live-human", step_move["action"] + " (moving)")
    render_table(pattern)


# ── Step mode: whole moves instead of continuous driving ─────────────────────

def send_step(action: str, s: dict) -> bool:
    """Ask the Double Motor for one move (forward/back one car length, or a 90 degree turn)
    without waiting for it to finish. Returns False if nothing could be sent."""
    plan = tl.step_plan(action, s["car_length"], s["reverse"])
    dev = motor_device()
    if plan is None or dev is None or Device.link_dropped(dev) or getattr(dev, "hub", None) is None:
        return False
    kind, direction, amount = plan
    try:
        if kind == "move":
            way = le.MOVEMENT_MOVE_DIRECTION_FORWARD if direction == "forward" else le.MOVEMENT_MOVE_DIRECTION_BACKWARD
            dev.hub.movement_move_for_degrees(amount, direction=way, speed=int(s["speed"]), blocking=False)
        else:
            way = le.MOVEMENT_TURN_DIRECTION_LEFT if direction == "left" else le.MOVEMENT_TURN_DIRECTION_RIGHT
            dev.hub.movement_turn_for_degrees(amount, direction=way, speed=int(s["speed"]), blocking=False)
    except Exception as e:
        print(f"Could not start the {action} move: {e}")
        return False
    return True

def start_step(action: str, s: dict):
    global step_move, sent
    sent = None
    if send_step(action, s):
        step_move = {"action": action, "clock": tl.MoveClock(time.monotonic())}

def step_tick(s, pattern, sl, sr, predicted, source) -> str:
    """One tick of step mode. A joystick push is one move (and, while recording, one example for
    the table); while running, the table picks a move whenever the car is standing still and you
    are not pushing. Returns who is driving, for the display."""
    global step_move, table_version
    if step_move is not None:
        dev = motor_device()
        dev_state = dev.state if dev is not None else {}
        if dev is None or Device.link_dropped(dev) or step_move["clock"].done(
                time.monotonic(), dev_state.get("LeftSpeed"), dev_state.get("RightSpeed")):
            step_move = None
        return "moving"
    fired = step_trigger.update(sl, sr)
    if fired is not None:
        if mode == "record" or (mode == "run" and s["correct"]):
            table.add(pattern, fired)
            table_version += 1
        start_step(fired, s)
        return "you"
    if mode == "run" and source != "none" and predicted != "Stop":
        start_step(predicted, s)
        return "table"
    return "none"


# ── Drawing ──────────────────────────────────────────────────────────────────

def set_text(elem_id, text):
    el = get_id(elem_id)
    if el:
        el.textContent = text

def render_live(readings, bits, pattern, s, human, predicted, source, driver):
    for key, reading, on in zip(("left", "front", "right"), readings, bits):
        set_text(f"read-{key}", "-" if reading is None else str(int(reading)))
        el = get_id(f"sense-{key}")
        if el:
            el.classList.toggle("on", on)
            el.classList.toggle("missing", reading is None)
    set_text("live-pattern", tl.pattern_name(pattern))
    set_text("live-human", human or "not driving")
    if source == "none":
        set_text("live-table", "Stop (table is empty)")
    else:
        set_text("live-table", predicted + (" (borrowed from a similar pattern)" if source == "nearest" else ""))
    set_text("live-total", f"{table.total()} examples")

    problems = []
    if all(r is None for r in readings):
        problems.append("No color sensors assigned.")
    if motor_device() is None:
        problems.append("No Double Motor selected.")
    if find_controller() is None:
        problems.append("No controller connected, so there is nothing to learn from.")
    set_text("setup-note", " ".join(problems))

def render_table(pattern):
    global drawn_version
    body = get_id("table-body")
    if body is None:
        return
    if drawn_version != table_version:
        drawn_version = table_version
        rows = ""
        for p in range(tl.N_PATTERNS):
            l, f, r = tl.pattern_bits(p)
            cells = "".join(
                f'<span class="dot{" on" if on else ""}" title="{name}"></span>'
                for name, on in zip(tl.SENSORS, (l, f, r)))
            top = max(table.counts[p].values()) or 1
            bars = "".join(
                f'<td class="cnt"><div class="bar"><div class="fill a-{a}" '
                f'style="height:{round(100 * table.counts[p][a] / top)}%"></div></div>'
                f'<span class="num">{table.counts[p][a]}</span></td>'
                for a in tl.ACTIONS if a != "Stop")
            best = table.best(p)
            learned = (f'<span class="chip a-{best}">{best}</span>'
                       f'<span class="conf">{round(100 * table.confidence(p))}%</span>'
                       if best else '<span class="none">no examples</span>')
            rows += (f'<tr id="row-{p}"><td class="pat"><span class="dots">{cells}</span>'
                     f'<span class="pname">{tl.pattern_name(p)}</span></td>{bars}'
                     f'<td class="learned">{learned}</td></tr>')
        body.innerHTML = rows
        save_table()
    for p in range(tl.N_PATTERNS):
        row = get_id(f"row-{p}")
        if row:
            row.classList.toggle("current", p == pattern)


# ── Buttons ──────────────────────────────────────────────────────────────────

def set_mode(new_mode: str):
    global mode, sent, step_move
    mode = new_mode
    state.is_running = (mode == "run")                 # what power.py (Save power) watches
    state.is_recording = (mode == "record")
    sent = None
    step_move = None
    step_trigger.reset()
    halt()
    asyncio.ensure_future(power.reconcile())           # sensor lights follow the mode
    for name in ("record", "run"):
        btn = get_id(f"{name}-btn")
        if btn:
            active = (mode == name)
            btn.classList.toggle("active", active)
            btn.textContent = {"record": "Stop recording" if active else "Record",
                               "run": "Stop" if active else "Run table"}[name]
    set_text("mode-label", {"idle": "Idle: you can drive, nothing is learned",
                            "record": "Recording: drive the car and the table learns",
                            "run": "Running: the table drives. Push a stick to take over"}[mode])
    document.body.classList.toggle("running", mode == "run")
    document.body.classList.toggle("recording", mode == "record")

async def switch_mode(target: str):
    """Stop (back to idle), or start Record/Run. Starting waits until the sensors are really
    reading (Save power may have their lights off), so the first rows and the first motor
    commands are never based on a sleeping sensor's stale reading."""
    if mode == target:
        set_mode("idle")
        return
    if mode != "idle":
        set_mode("idle")
    # Pretend the mode is already on so power.py wakes the sensors, wait for them, then start.
    state.is_running = (target == "run")
    state.is_recording = (target == "record")
    await power.ensure_ready()
    set_mode(target)

@when("click", "#record-btn")
async def on_record(evt):
    await switch_mode("record")

@when("click", "#run-btn")
async def on_run(evt):
    await switch_mode("run")

@when("change", "#save-power")
def on_save_power(evt):
    asyncio.ensure_future(power.reconcile())

@when("click", "#clear-btn")
def on_clear(evt):
    global table_version
    if table.total() and not window.confirm("Forget everything the table has learned?"):
        return
    table.clear()
    table_version += 1

@when("click", "#add-device-btn")
async def on_add_device(evt):
    await Device.create_new_device()

@when("click", "#close-ble-warning-btn")
def on_close_warning(evt):
    get_id("ble-warning-banner").classList.add("hidden")

@when("change", ".device-pick")
def on_pick(evt):
    sent_reset()

def sent_reset():
    global sent
    sent = None


# ── Boot ─────────────────────────────────────────────────────────────────────

async def main_loop():
    while True:
        try:
            tick()
        except Exception as e:             # keep the loop alive; one bad tick shouldn't freeze the page
            print("tick failed: " + str(e))
        await asyncio.sleep(TICK_S)

def boot():
    get_id("loading-splash").style.display = "none"
    get_id("page-wrap").style.display = "flex"
    if not hasattr(window.navigator, "bluetooth"):
        get_id("ble-warning-banner").classList.remove("hidden")
    load_table()
    set_mode("idle")
    sync.refresh_device_dropdowns()
    power.start()
    asyncio.ensure_future(main_loop())

boot()
