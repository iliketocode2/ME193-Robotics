"""Line Follower page: connect buttons, two oscilloscopes, and the shared `robot` the tabs will use.

The control tabs (Bang bang, PD, ...) will read `robot.light.reflection` and call
`robot.motor.drive(left, right)`; nothing here depends on which tab is showing.
"""

import asyncio
import html
import inspect
import re
import time

from pyscript import document, window
from pyscript.ffi import create_proxy, to_js

import bangbang
import json
import lqr
import mpc
import nnet
import pd
import qtable
import steering
import textwrap
import webhub
from bangbang import clamp
from devices import Device
from track import Tracker

WINDOW_S = 10   # seconds of history on the scopes

def make_scope(canvas_id, **opts):
    # a Python dict must be converted to a plain JS object before JS can read its keys
    return window.Scope.new(canvas_id, to_js(opts, dict_converter=window.Object.fromEntries))


scope_light = make_scope("scope-light", min=0, max=100, window=WINDOW_S,
                         traces=[{"name": "light", "color": "#b45309"}])
scope_motor = make_scope("scope-motor", min=-100, max=100, window=WINDOW_S,
                         traces=[{"name": "left", "color": "#0369a1"},
                                 {"name": "right", "color": "#be185d"}])


class Robot:
    """The two devices the control methods work with."""
    motor: Device | None = None
    light: Device | None = None


robot = Robot()
SLOTS = {
    "motor": {"kind": "Double Motor", "card": "card-motor", "button": "btn-motor", "label": "label-motor", "batt": "batt-motor",
              "title": "Double Motor <small>(speed, forward = +)</small>"},
    "light": {"kind": "Color Sensor", "card": "card-light", "button": "btn-light", "label": "label-light", "batt": "batt-light",
              "title": "Color Sensor <small>(reflection, 0–100)</small>"},
}


def el(id_):
    return document.getElementById(id_)


def set_status(text):
    el("status").textContent = text


def display_name(dev):
    """The hub's own name (it already says 'Color Sensor' or 'Double Motor')."""
    return dev.name or dev.kind


def show_chip(slot, state, text=None):
    """state: 'off' | 'busy' | 'on' | 'lost'. The plot is open while connected (or lost)."""
    s = SLOTS[slot]
    card = el(s["card"])
    card.className = "card scope-card " + ("closed" if state in ("off", "busy") else state)
    el(s["label"]).innerHTML = s["title"] if state in ("off", "busy") else (text or s["kind"])
    if state in ("off", "busy"):
        el(s["batt"]).textContent = ""
    button = el(s["button"])
    button.textContent = {"off": "Connect", "busy": "Connecting…", "on": "Disconnect", "lost": "Reconnect"}[state]
    button.disabled = state == "busy"
    el("btn-stop").disabled = not (robot.motor and robot.motor.connected)


def show_battery(slot, dev):
    if dev.battery is None:
        return
    box = el(SLOTS[slot]["batt"])
    box.textContent = f"🔋 {dev.battery}%"
    box.className = "battery" + (" low" if dev.battery <= 20 else "")


# -- calibration (Calibrate tab) -------------------------------------------------------------------
cal = {"on": None, "off": None}      # average reflection over the line / over the floor
cal_samples = None                    # a list while a measurement is running


def load_calibration():
    try:
        for k in cal:
            v = window.localStorage.getItem("lf-cal-" + k)
            cal[k] = float(v) if v not in (None, "") else None
    except Exception:
        pass


def calibrated():
    return cal["on"] is not None and cal["off"] is not None and abs(cal["on"] - cal["off"]) >= 5


def show_calibration():
    fmt = lambda v: "–" if v is None else f"{v:.0f}"
    el("cal-on-val").textContent = fmt(cal["on"])
    el("cal-off-val").textContent = fmt(cal["off"])
    if calibrated():
        mid = bangbang.target(cal["on"], cal["off"])
        el("cal-mid").textContent = f"{mid:.0f}"
        note = f"On line {cal['on']:.0f} · off line {cal['off']:.0f} · target {mid:.0f}"
    else:
        el("cal-mid").textContent = "–"
        note = ("On and off the line read too alike - recalibrate on the Calibrate tab."
                if cal["on"] is not None and cal["off"] is not None
                else "Not calibrated yet: do it on the Calibrate tab.")
    for tab in TABS:
        box = el(tab.prefix + "-cal")
        if box is not None:
            box.textContent = note
    refresh_buttons()


async def measure(which):
    """Average the sensor over one second and keep it as the 'on' or 'off' line value."""
    global cal_samples
    if robot.light is None or not robot.light.connected:
        return
    el("cal-" + which).disabled = True
    cal_samples = []
    await asyncio.sleep(1.0)
    samples, cal_samples = cal_samples, None
    if samples:
        cal[which] = sum(samples) / len(samples)
        try:
            window.localStorage.setItem("lf-cal-" + which, str(cal[which]))
        except Exception:
            pass
    else:
        set_status("No light readings arrived - is the sensor connected?")
    show_calibration()


# -- code card: the real controller source, colored ------------------------------------------
_TOKENS = re.compile(
    r"(?P<com>#[^\n]*)"
    r"|(?P<str>f?\"[^\"\n]*\"|f?'[^'\n]*')"
    r"|(?P<kw>\b(?:def|return|if|elif|else|not|and|or|in|for|while|import|from|True|False|None|lambda)\b)"
    r"|(?P<num>\b\d+(?:\.\d+)?\b)"
    r"|(?P<fn>\b[A-Za-z_]\w*(?=\())")


def highlight(source):
    """Python source -> HTML with <span class="kw|fn|str|num|com"> around each token."""
    def color(m):
        return f'<span class="{m.lastgroup}">{m.group()}</span>'
    return _TOKENS.sub(color, html.escape(source, quote=False))


LOOP_TEXT = '''# every time the color sensor reports a reading:
left, right = {call}
motor.drive(left, right)


'''


class RunTab:
    """A control tab: Start/Stop, its settings, a Plotly run plot and a code card.

    make_step() is called at every Start and returns step(reflection, now, settings) -> (left,
    right); a controller that remembers things (PD) builds fresh state there. Optional extras:
    modes = {name: make_step} adds more Start-like buttons (id "<prefix>-<name>"), checks adds
    tick boxes, refresh() is called about twice a second while running and once at the end."""

    def __init__(self, prefix, settings, make_step, code, call, info=None,
                 modes=None, checks=None, refresh=None, timed=False):
        self.prefix = prefix
        self.settings = settings            # {name: input id suffix}; numbers
        self.checks = checks or {}          # {name: tick box id suffix}
        self.values = {name: 0.0 for name in settings}
        self.values["right"] = True
        self.factories = {"run": make_step, **(modes or {})}
        self.mode = None
        self.step = None
        self.running = False
        self.cmd = (0, 0)
        self.sent = 0.0
        self.refresh = refresh
        self.refreshed = 0.0
        self.timed = timed          # a "run" lasts the seconds in <prefix>-secs (0 = until Stop)
        self.deadline = None
        self.plot = window.RunPlot.new("plot-" + prefix)     # Plotly: zoom after a run
        self.code = code if isinstance(code, (list, tuple)) else [code]
        self.call = call
        self.info = info          # optional: values -> text shown next to the settings
        self.labels = {}
        for mode in self.factories:
            button = el(f"{prefix}-{mode}")
            self.labels[mode] = button.textContent
            button.addEventListener("click", create_proxy(lambda evt, mode=mode: self.toggle(mode)))
        for id_ in list(settings.values()) + list(self.checks.values()) + ["side"]:
            el(f"{prefix}-{id_}").addEventListener("input", create_proxy(self.read_settings))
        self.read_settings()
        self.show_code()

    def show_code(self):
        try:
            source = "\n\n".join(textwrap.dedent(inspect.getsource(part)) for part in self.code)
        except Exception:
            source = "# (could not read the controller source)"
        box = el(self.prefix + "-code")
        if box is not None:             # (the Calibrate tab's measuring card has no code card)
            box.innerHTML = highlight(LOOP_TEXT.format(call=self.call) + source)

    def read_settings(self, evt=None):
        for name, id_ in self.settings.items():
            try:
                self.values[name] = max(0.0, float(el(f"{self.prefix}-{id_}").value))
            except ValueError:
                pass
        for name, id_ in self.checks.items():
            self.values[name] = bool(el(f"{self.prefix}-{id_}").checked)
        self.values["right"] = el(self.prefix + "-side").value == "right"
        if self.timed and not self.running and "run" in self.labels:
            el(self.prefix + "-run").textContent = self.label("run")
        if self.info is not None:
            el(self.prefix + "-info").textContent = self.info(self.values)

    def label(self, mode):
        """The button's resting text: 'Run 20 s' on a timed tab, otherwise what the page gave it."""
        if self.timed and mode == "run":
            seconds = self.values.get("secs", 0)
            return f"Run {seconds:g} s" if seconds > 0 else "Start"
        return self.labels[mode]

    def can_run(self):
        return bool(calibrated() and robot.light and robot.light.connected
                    and robot.motor and robot.motor.connected)

    def refresh_buttons(self):
        ok = self.can_run() or self.running
        for mode in self.factories:
            el(f"{self.prefix}-{mode}").disabled = not ok or (self.running and mode != self.mode)

    def toggle(self, mode="run"):
        if self.running:
            if mode == self.mode:
                self.stop()
        elif self.can_run():
            for other in TABS:
                other.stop()
            self.read_settings()
            step = self.factories[mode]()
            if step is None:                 # the mode could not start and said why
                return
            self.step, self.mode, self.running = step, mode, True
            seconds = self.values.get("secs", 0) if self.timed and mode == "run" else 0
            self.deadline = time.monotonic() + seconds if seconds > 0 else None
            self.plot.start()
            button = el(f"{self.prefix}-{mode}")
            button.textContent = "Stop"
            button.className = "go running"
            self.refresh_buttons()

    def stop(self):
        self.plot.stop()            # freeze the plot so the run can be studied
        if self.running or self.cmd != (0, 0):
            self.running = False
            self.cmd = (0, 0)
            if robot.motor is not None:
                robot.motor.stop()
        self.deadline = None
        for mode in self.labels:
            button = el(f"{self.prefix}-{mode}")
            button.textContent = self.label(mode)
            button.className = "go"
        self.mode = None
        self.refresh_buttons()
        if self.refresh is not None:
            self.refresh()

    def on_reading(self, reflection):
        """One control step per light reading: pick the wheel speeds, send them if new (or every
        quarter second regardless, in case a command got lost), and plot everything."""
        if not self.running:
            return
        now = time.monotonic()
        if self.deadline is not None and now >= self.deadline:
            self.stop()
            set_status(f"Run finished ({self.values['secs']:g} s). Press {self.label('run')} to go again.")
            return
        cmd = self.step(reflection, now, self.values)
        if cmd != self.cmd or now - self.sent > 0.25:
            self.cmd, self.sent = cmd, now
            robot.motor.drive(*cmd)
            webhub.drop_stale_commands()
        self.plot.push(bangbang.target(cal["on"], cal["off"]), reflection, self.cmd[0], self.cmd[1])
        if self.refresh is not None and now - self.refreshed > 0.5:
            self.refreshed = now
            self.refresh()


def bang_bang_step():
    return lambda reflection, now, v: bangbang.decide(
        reflection, cal["on"], cal["off"], v["right"], v["base"], v["turn"])


def pd_step():
    controller = pd.PD()
    return lambda reflection, now, v: controller.decide(
        reflection, now, cal["on"], cal["off"], v["right"], v["base"], v["kp"], v["kd"])


def lqr_step():
    controller = lqr.LQR()
    return lambda reflection, now, v: controller.decide(
        reflection, now, cal["on"], cal["off"], v["right"], v["base"],
        v["q_error"], v["q_rate"], v["effort"])


def lqr_info(v):
    k_error, k_rate = lqr.gains(v["q_error"], v["q_rate"], v["effort"])
    b = lqr.PLANT_GAIN
    omega = (b * k_error) ** 0.5                       # how fast the loop responds, rad/s
    zeta = b * k_rate / (2 * omega) if omega else 0    # damping: ~0.7 is quick and calm
    text = f"K = [{k_error:.3f}, {k_rate:.3f}] · b = {b:.0f} · ω = {omega:.1f} rad/s · ζ = {zeta:.2f}"
    if omega > 8:
        text += " · fast: Bluetooth delay may make it wobble, try a larger R"
    return text


# -- MPC ----------------------------------------------------------------------------------------
def mpc_step():
    controller = mpc.MPC()
    return lambda reflection, now, v: controller.decide(
        reflection, now, cal["on"], cal["off"], v["right"], v["base"],
        v["q_error"], v["q_rate"], v["effort"], v["horizon"], v["max_steer"])


def mpc_info(v):
    return f"looks {max(1, min(30, int(v['horizon']))) * mpc.STEP:.2f} s ahead · b = {lqr.PLANT_GAIN:.0f}"


# -- Q-table ------------------------------------------------------------------------------------
learner = qtable.QLearner()
ERROR_NAMES = ["far off the line", "off the line", "near the target", "on the line", "far on the line"]
RATE_NAMES = ["moving off", "steady", "moving on"]


def qtable_step():
    learner.restart()                # keep the table, forget the last move
    return lambda reflection, now, v: learner.decide(
        reflection, now, cal["on"], cal["off"], v["right"], v["base"],
        v["turn"], v["alpha"], v["gamma"], v["epsilon"], v["learning"])


def show_qtable():
    values = [v for row in learner.q for v in row]
    low, high = min(values), max(values)
    head = "".join(f"<th>{a:+g}</th>" for a in qtable.ACTIONS)
    rows = []
    for state, row in enumerate(learner.q):
        best = max(row)
        cells = []
        for value in row:
            fraction = 0.5 if high == low else (value - low) / (high - low)
            shade = f"background: hsl({120 * fraction:.0f}, 45%, {88 - 18 * fraction:.0f}%)" if high != low else ""
            mark = " best" if value == best and high != low else ""
            cells.append(f'<td class="q{mark}" style="{shade}">{value:.2f}</td>')
        rows.append(f"<tr><th>{ERROR_NAMES[state // 3]}</th><th>{RATE_NAMES[state % 3]}</th>{''.join(cells)}</tr>")
    el("ql-table").innerHTML = (f"<table><tr><th>error</th><th>rate</th>{head}</tr>{''.join(rows)}</table>")
    el("ql-info").textContent = f"{learner.steps} decisions so far"


def reset_qtable(evt=None):
    if not any(tab.running for tab in TABS):
        learner.__init__()
        show_qtable()


# -- Neural net ---------------------------------------------------------------------------------
nn = {"net": None, "data": [], "training": False, "note": "not trained yet"}


def load_net():
    try:
        text = window.localStorage.getItem("lf-net")
        if text:
            nn["net"] = nnet.Net.from_json(text)
            nn["note"] = "net loaded from last time"
    except Exception:
        pass


def nn_show():
    el("nn-info").textContent = f"{len(nn['data'])} examples · {nn['note']}"


def nn_record_step():
    """Drive with the PD teacher and keep every (error, rate, steering) it produces."""
    tracker = Tracker()

    def step(reflection, now, v):
        error, rate = tracker.update(reflection, now, cal["on"], cal["off"])
        steer = nnet.teacher(error, rate, v["kp"], v["kd"])
        nn["data"].append((error, rate, max(-1.0, min(1.0, steer / 100))))
        del nn["data"][:-3000]
        steer = steer if v["right"] else -steer
        return clamp(v["base"] + steer), clamp(v["base"] - steer)
    return step


def nn_run_step():
    if nn["net"] is None:
        set_status("Train the network first (record some examples, then press Train).")
        return None
    policy = nnet.Policy(nn["net"])
    return lambda reflection, now, v: policy.decide(
        reflection, now, cal["on"], cal["off"], v["right"], v["base"])


async def nn_train(evt=None):
    tab = TABS_BY_NAME["nn"]
    if nn["training"] or tab.running:
        return
    if len(nn["data"]) < 30:
        set_status("Record some examples first (press Record and let the PD teacher drive).")
        return
    tab.read_settings()
    hidden = max(1, min(32, int(tab.values["hidden"])))
    if nn["net"] is None or len(nn["net"].w2) != hidden:
        nn["net"] = nnet.Net(hidden)
    nn["training"] = True
    el("nn-train").disabled = True
    epochs = max(1, int(tab.values["epochs"]))
    for epoch, loss in enumerate(nnet.train_steps(nn["net"], list(nn["data"]), epochs, tab.values["lr"]), 1):
        nn["note"] = f"training: epoch {epoch}/{epochs}, loss {loss:.5f}"
        nn_show()
        await asyncio.sleep(0)
    nn["note"] = f"trained ({epochs} epochs, final loss {loss:.5f})"
    try:
        window.localStorage.setItem("lf-net", nn["net"].to_json())
    except Exception:
        pass
    nn["training"] = False
    el("nn-train").disabled = False
    nn_show()


def nn_clear(evt=None):
    nn["data"].clear()
    nn_show()


# -- Spin test (Calibrate tab): how fast can the car turn on the spot? --------------------------
spin = {"running": False, "gyro": {"left": None, "right": None}}


def stop_spin():
    spin["running"] = False


def number(id_, default):
    try:
        return float(el(id_).value)
    except ValueError:
        return default


async def run_spin(direction):
    """Spin on the spot for a while (left wheel forward and right wheel back is a turn to the
    right) while averaging the hub's gyro: the fastest the car can turn."""
    motor = robot.motor
    if spin["running"] or motor is None or not motor.connected:
        return
    stop_all()
    speed = max(0.0, min(100.0, number("spin-speed", 100)))
    seconds = max(2.0, number("spin-time", 10))
    sign = 1 if direction == "right" else -1
    samples = []
    spin["running"] = True
    refresh_buttons()
    motor.drive(sign * speed, -sign * speed)
    start = time.monotonic()
    while spin["running"] and time.monotonic() - start < seconds and motor.connected:
        if time.monotonic() - start > 1.0 and motor.gyro_z is not None:      # skip the speed-up
            samples.append(motor.gyro_z)
        await asyncio.sleep(0.05)
    spin["gyro"][direction] = abs(sum(samples) / len(samples)) if samples else None   # degrees/s
    motor.stop()
    spin["running"] = False
    refresh_buttons()
    show_spin()


def show_spin():
    rates = []
    for direction in ("left", "right"):
        gyro = spin["gyro"][direction]
        el(f"spin-gyro-{direction}").textContent = "–" if gyro is None else f"{gyro * 0.017453:.1f} rad/s ({gyro:.0f}°/s)"
        if gyro is not None:
            rates.append(gyro * 0.017453)
    if rates:
        mean = sum(rates) / len(rates)
        el("spin-summary").textContent = f"Maximum turn rate ≈ {mean:.1f} rad/s ({mean * 57.296:.0f}°/s, {mean / 6.2832:.2f} turns per second)"
    else:
        el("spin-summary").textContent = ""


# -- Estimating steering authority (Calibrate tab) -------------------------------------------------
bstate = {"finder": None, "result": None}


def b_step():
    """Steer with Kp only, so the car wobbles across the edge, and time the wobble."""
    tracker, finder = Tracker(), steering.PeriodFinder()
    bstate["finder"], bstate["result"] = finder, None

    def step(reflection, now, v):
        error, _ = tracker.update(reflection, now, cal["on"], cal["off"])
        finder.update(now, error)
        steer = v["kp"] * error if v["right"] else -v["kp"] * error
        return clamp(v["base"] + steer), clamp(v["base"] - steer)
    return step


ZETA = 0.707            # damping the suggested gains aim for: quick, with little overshoot


def sig(value):
    """Two significant figures, as text for an input box."""
    return f"{float(f'{value:.2g}'):g}"


def suggest_defaults():
    """Set the gain boxes so each controller responds at the target speed (rad/s) for the
    measured steering authority b. For e'' = -b u:
      PD:      Kp = omega**2 / b * 100 and Kd = 2 * zeta * omega / b * 100 (percent steering)
      LQR/MPC: omega = sqrt(b * K1) with K1 = sqrt(Qe / R), so R = Qe * b**2 / omega**4.
    The neural net's PD teacher gets the PD gains; the Q-table's Turn is 1.5 x PD's Kp."""
    omega = max(1.0, number("b-omega", 10.0))
    b = lqr.PLANT_GAIN
    # in seconds: a well-damped (zeta ~ 0.7) response settles in about 4 / (zeta * omega)
    el("b-omega-time").textContent = f"≈ {1.8 / omega:.2f} s to get most of the way back to the edge, {4 / (ZETA * omega):.2f} s to settle on it"
    for name in ("lqr", "mpc"):
        tab = TABS_BY_NAME[name]
        tab.read_settings()
        el(f"{name}-r").value = sig(max(tab.values["q_error"], 1e-6) * b ** 2 / omega ** 4)
        tab.read_settings()
    # Q-table: its biggest move should be about the steering a full error needs (1.5 x PD's Kp)
    turn = sig(1.5 * omega ** 2 / b * 100)
    if el("ql-turn").value != turn:
        el("ql-turn").value = turn
        TABS_BY_NAME["ql"].read_settings()
        if learner.steps and not TABS_BY_NAME["ql"].running:
            reset_qtable()                  # a table learned with other moves no longer fits
            set_status("Q-table reset: the steering size changed with b or the response speed.")
    for name in ("pd", "nn"):
        el(f"{name}-kp").value = sig(omega ** 2 / b * 100)
        el(f"{name}-kd").value = sig(2 * ZETA * omega / b * 100)
        TABS_BY_NAME[name].read_settings()


def show_b_current():
    el("b-current").textContent = f"{lqr.PLANT_GAIN:.0f}"
    suggest_defaults()          # the gains follow b, and read_settings refreshes the info lines


def b_show():
    finder = bstate["finder"]
    kp = TABS_BY_NAME["b"].values["kp"] if "b" in TABS_BY_NAME else 0
    result = steering.estimate(finder, kp) if finder is not None else None
    bstate["result"] = result
    el("b-use").disabled = result is None
    if finder is None:
        text = ""
    elif result is None:
        text = f"Waiting for a steady wobble ({max(0, len(finder.rises) - 1)} cycles so far, need 3)…"
    else:
        period, b, cycles, unevenness, amplitude = result
        text = f"Period {period:.2f} s over {cycles} cycles → b ≈ {b:.0f}"
        if unevenness > 0.25:
            text += " (the wobble is uneven: wait for it to settle, or adjust Kp)"
        elif amplitude > 0.9:
            text += " (swings reach the sensor's limits: lower Kp for a cleaner reading)"
    el("b-result").textContent = text


def use_b(evt=None):
    if bstate["result"] is None:
        return
    lqr.PLANT_GAIN = float(round(bstate["result"][1]))
    try:
        window.localStorage.setItem("lf-b", str(lqr.PLANT_GAIN))
    except Exception:
        pass
    show_b_current()


def reset_b(evt=None):
    lqr.PLANT_GAIN = float(lqr.DEFAULT_PLANT_GAIN)
    try:
        window.localStorage.removeItem("lf-b")
    except Exception:
        pass
    show_b_current()


def load_b():
    try:
        text = window.localStorage.getItem("lf-b")
        if text:
            lqr.PLANT_GAIN = float(text)
    except Exception:
        pass


TABS = []   # filled just below, once the page is wired
TABS.append(RunTab("bb", {"base": "base", "turn": "turn"}, bang_bang_step, bangbang.decide,
                   "decide(reflection, on_line, off_line, follow_right, base, turn)"))
TABS.append(RunTab("pd", {"base": "base", "kp": "kp", "kd": "kd"}, pd_step, pd.PD.decide,
                   "controller.decide(reflection, now, on_line, off_line, follow_right, base, kp, kd)"))
TABS.append(RunTab("lqr", {"base": "base", "q_error": "qe", "q_rate": "qd", "effort": "r"}, lqr_step,
                   [lqr.gains, lqr.LQR.decide],
                   "controller.decide(reflection, now, on_line, off_line, follow_right, base, q_error, q_rate, effort)",
                   info=lqr_info))
TABS.append(RunTab("mpc", {"base": "base", "q_error": "qe", "q_rate": "qd", "effort": "r",
                           "horizon": "n", "max_steer": "max"}, mpc_step,
                   [mpc.build, mpc.solve, mpc.MPC.decide],
                   "controller.decide(reflection, now, on_line, off_line, follow_right, base, q_error, q_rate, effort, horizon, max_steer)",
                   info=mpc_info))
TABS.append(RunTab("ql", {"secs": "secs", "base": "base", "turn": "turn", "alpha": "alpha", "gamma": "gamma", "epsilon": "eps"},
                   qtable_step, [qtable.state_of, qtable.QLearner.decide],
                   "learner.decide(reflection, now, on_line, off_line, follow_right, base, turn, alpha, gamma, epsilon, learning)",
                   checks={"learning": "learn"}, refresh=show_qtable, timed=True))
TABS.append(RunTab("nn", {"base": "base", "hidden": "hidden", "epochs": "epochs", "lr": "lr", "kp": "kp", "kd": "kd"},
                   nn_run_step, [nnet.Net.forward, nnet.Policy.decide, nnet.train_steps],
                   "policy.decide(reflection, now, on_line, off_line, follow_right, base)",
                   modes={"rec": nn_record_step}, refresh=nn_show))
TABS.append(RunTab("b", {"base": "base", "kp": "kp"}, b_step,
                   [steering.PeriodFinder.update, steering.estimate], "", refresh=b_show))
TABS_BY_NAME = {tab.prefix: tab for tab in TABS}


def stop_all():
    stop_spin()
    for tab in TABS:
        tab.stop()


def refresh_buttons():
    light_ok = bool(robot.light and robot.light.connected)
    for which in ("on", "off"):
        el("cal-" + which).disabled = not light_ok
    for tab in TABS:
        tab.refresh_buttons()
    motor_ok = bool(robot.motor and robot.motor.connected)
    for direction in ("left", "right"):
        el("spin-" + direction).disabled = not motor_ok or spin["running"]
    el("spin-stop").disabled = not spin["running"]


# -- live data -> scopes ----------------------------------------------------------------------
def on_light(dev):
    show_battery("light", dev)
    if dev.reflection is not None:
        scope_light.push(0, dev.reflection)
        el("read-light").textContent = f"{dev.reflection:.0f}"
        if cal_samples is not None:
            cal_samples.append(dev.reflection)
        for tab in TABS:
            tab.on_reading(dev.reflection)


def on_motor(dev):
    show_battery("motor", dev)
    if dev.left is not None:
        scope_motor.push(0, dev.left)
        el("read-left").textContent = f"{dev.left:.0f}"
    if dev.right is not None:
        scope_motor.push(1, dev.right)
        el("read-right").textContent = f"{dev.right:.0f}"


HANDLERS = {"motor": on_motor, "light": on_light}
KIND_SLOT = {"Double Motor": "motor", "Color Sensor": "light"}


def make_drop_handler(slot):
    def dropped(dev):
        show_chip(slot, "lost", f"{display_name(dev)} (lost)")
        set_status(f"{SLOTS[slot]['kind']} lost its Bluetooth link.")
        stop_all()
        refresh_buttons()
    return dropped


# -- buttons ----------------------------------------------------------------------------------
async def toggle(slot):
    dev = getattr(robot, slot)
    if dev is not None and dev.connected:        # Disconnect
        await dev.disconnect()
        setattr(robot, slot, None)
        show_chip(slot, "off")
        set_status("")
        stop_all()
        refresh_buttons()
        return
    if dev is not None:                          # lost link: clean up, then connect again
        await dev.disconnect()
        setattr(robot, slot, None)
    dev = Device(SLOTS[slot]["kind"])
    dev.notify_ms = requested_ms()
    show_chip(slot, "busy")
    set_status(f"Pick the {SLOTS[slot]['kind']} (or the Double Motor / Color Sensor, either one) in the Bluetooth window…")
    if not await dev.connect():
        show_chip(slot, "off")
        set_status(f"Could not connect the {SLOTS[slot]['kind']}.")
        return

    # The hub may not be the one this button was for: put it where it belongs.
    actual = KIND_SLOT[dev.kind]
    current = getattr(robot, actual)
    if current is not None and current.connected:
        window.alert(f"The {dev.kind} is already connected. Disconnect it first to connect another one.")
        await dev.disconnect()
        show_chip(slot, "off")
        set_status("")
        return
    dev.on_data = HANDLERS[actual]
    dev.on_drop = make_drop_handler(actual)
    setattr(robot, actual, dev)
    show_chip(actual, "on", display_name(dev))
    show_battery(actual, dev)
    if actual == slot:
        set_status("")
    else:
        show_chip(slot, "off")
        set_status(f"That was the {dev.kind}, so it is connected there. "
                   f"Press Connect on the {SLOTS[slot]['kind']} when you are ready.")
    refresh_buttons()


def requested_ms():
    try:
        return int(el("rate-select").value)
    except ValueError:
        return 20


def change_rate(evt=None):
    for slot in SLOTS:
        dev = getattr(robot, slot)
        if dev is not None and dev.connected:
            dev.set_rate(requested_ms())
    set_status(f"Asked the devices for a reading every {requested_ms()} ms.")


async def rate_monitor():
    """Twice a second, show how many readings per second really arrive from each device."""
    while True:
        try:
            show_rates()
        except Exception as exc:          # never let a display glitch end the readout
            print(f"rate readout: {exc}")
        await asyncio.sleep(0.5)


def show_rates():
    asked = 1000 / requested_ms()
    for slot in SLOTS:
        dev = getattr(robot, slot)
        box = el("rate-" + slot)
        hz = dev.measured_hz() if dev is not None and dev.connected else None
        if dev is None or not dev.connected:
            box.textContent = ""
            box.className = "battery"
        elif hz is None:
            box.textContent = "📶 no data"
            box.className = "battery slow"
        else:
            box.textContent = f"📶 {hz:.0f} Hz"
            box.className = "battery" + (" slow" if hz < 0.8 * asked else "")


def stop_motors(evt=None):
    stop_all()
    if robot.motor is not None:
        robot.motor.stop()
        set_status("Motors stopped.")


for _slot, _s in SLOTS.items():
    el(_s["button"]).addEventListener("click", create_proxy(lambda evt, slot=_slot: asyncio.ensure_future(toggle(slot))))
    el(_s["button"]).disabled = False
el("btn-stop").addEventListener("click", create_proxy(stop_motors))
el("rate-select").addEventListener("change", create_proxy(change_rate))
asyncio.ensure_future(rate_monitor())
el("cal-on").addEventListener("click", create_proxy(lambda evt: asyncio.ensure_future(measure("on"))))
el("cal-off").addEventListener("click", create_proxy(lambda evt: asyncio.ensure_future(measure("off"))))
# stop any run when another tab is opened
el("tabs").addEventListener("click", create_proxy(lambda evt: stop_all()))
el("spin-left").addEventListener("click", create_proxy(lambda evt: asyncio.ensure_future(run_spin("left"))))
el("spin-right").addEventListener("click", create_proxy(lambda evt: asyncio.ensure_future(run_spin("right"))))
el("spin-stop").addEventListener("click", create_proxy(lambda evt: stop_all()))
el("b-omega").addEventListener("input", create_proxy(lambda evt: show_b_current()))
el("b-use").addEventListener("click", create_proxy(use_b))
el("b-reset").addEventListener("click", create_proxy(reset_b))
load_b()
show_b_current()
el("ql-reset").addEventListener("click", create_proxy(reset_qtable))
el("nn-train").addEventListener("click", create_proxy(lambda evt: asyncio.ensure_future(nn_train())))
el("nn-clear").addEventListener("click", create_proxy(nn_clear))
load_net()
nn_show()
show_qtable()
load_calibration()
show_calibration()

# stop the wheels if the page is closed or reloaded
window.addEventListener("beforeunload", create_proxy(lambda evt: stop_motors()))

set_status("Ready. Connect the Double Motor and the Color Sensor.")
