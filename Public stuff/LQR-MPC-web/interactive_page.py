"""The Interactive page: a Python port of the original page's JavaScript.

Every tab recomputes its simulation with lqrmpc.py when a slider moves. The
"Run on real robot" buttons and the calibration panel talk to the LEGO Education
hubs over Web Bluetooth through hardware.py.
"""

import asyncio
import json
import time

import numpy as np
from pyscript import document, window

import hardware as hw
import lqrmpc as m
from ui import *

cs = hw.ColorSensorDevice()
dm = hw.DoubleMotorDevice()

# -- calibration: the generic reference build until you measure your own -----------
CAL = dict(hw.DEFAULT_CAL)
CAL_SOURCE = "the reference build (not calibrated yet)"
try:
    saved = window.localStorage.getItem("lqrmpc-cal")
    if saved:
        CAL.update(json.loads(saved))
        CAL_SOURCE = "your calibration (saved in this browser)"
except Exception:
    pass
pending = dict(CAL)           # edits made by the "Redo calibration" panel, applied on Save

scenario = {"v": 150.0, "R": 80.0}
gains = {"qe": 1.0, "qth": 0.0, "r": 25.0}
real = {}                      # last real-robot run for each tab
recompute = {}                 # tab name -> function
state = {"running": False, "stop": False}


def aspect(id):
    return float(el(id).dataset.aspect or 0.42)


def sim_series(log, col, cls="sim", label="simulated"):
    return dict(xs=log[:, 0], ys=log[:, col], cls=cls, label=label)


def real_series(rows, key="e"):
    return dict(xs=[r["t"] for r in rows], ys=[r[key] for r in rows], cls="real", label="real robot")


def summarize_real(rows):
    e = np.array([r["e"] for r in rows])
    return (float(np.abs(e).max()), 100 * float(np.mean([r["saturated"] for r in rows])),
            float(np.mean([r["v"] for r in rows])))


def real_readout(prefix, name, e_max, with_v=False):
    rows = real.get(name)
    if not rows:
        return
    peak, blind, mean_v = summarize_real(rows)
    items = [("real peak |e|", f"{fmt(peak)} mm", "bad" if peak > e_max else "ok"),
             ("real blind", f"{fmt(blind, 1)}%", "bad" if blind > 0 else "ok")]
    if with_v:
        items.append(("real mean v", f"{fmt(mean_v, 0)} mm/s", ""))
    el(prefix + "-hw-readout").innerHTML = readout(items)


# -- scenario panel (speed and turn radius), shared by the PD, LQR and MPC tabs --------
def scenario_panel(prefix):
    return f'''<h3>Scenario (shared)</h3>
    <div class="slider-row"><label>forward speed v <span class="val" id="{prefix}-v-val"></span></label>
      <input type="range" id="{prefix}-v" min="50" max="300" step="5"></div>
    <div class="slider-row"><label>turn radius R <span class="val" id="{prefix}-R-val"></span></label>
      <input type="range" id="{prefix}-R" min="20" max="400" step="5"></div>
    <div class="shared-note" id="{prefix}-vmax-note"></div>'''


def wire_scenario(prefix, on_change):
    def set_v(v):
        scenario["v"] = v
        sync_scenario(prefix)
        on_change()

    def set_r(r):
        scenario["R"] = r
        sync_scenario(prefix)
        on_change()

    slider(prefix + "-v", lambda x: f"{fmt(x, 0)} mm/s", scenario["v"], set_v)
    slider(prefix + "-R", lambda x: f"{fmt(x, 0)} mm", scenario["R"], set_r)


def sync_scenario(except_prefix):
    for p in ("pd", "lqr", "mpc"):
        if p == except_prefix:
            continue
        el(p + "-v").value, el(p + "-R").value = scenario["v"], scenario["R"]
        el(p + "-v-val").textContent = f"{fmt(scenario['v'], 0)} mm/s"
        el(p + "-R-val").textContent = f"{fmt(scenario['R'], 0)} mm"
    for p in ("pd", "lqr", "mpc"):
        if p != except_prefix:
            recompute[p]()


# -- tab 1: bang-bang ---------------------------------------------------------------------
bb = {"v": 100.0, "w": 2.0}


def bb_recompute():
    dt, e_max = CAL["dt"], CAL["e_max_mm"]
    e, theta, prev, crossings, t_max = e_max * 0.7, 0.0, None, 0, 6.0
    ts, es = [], []
    for i in range(int(t_max / dt)):
        sign = 1.0 if e > 0 else -1.0
        if prev is not None and sign != prev:
            crossings += 1
        prev = sign
        omega = -bb["w"] * sign
        ts.append(i * dt); es.append(e)
        e, theta = m.step(e, theta, bb["v"], omega, 0.0, dt)
    series = [dict(xs=ts, ys=es, cls="sim", label="simulated")]
    if "bangbang" in real:
        series.append(real_series(real["bangbang"]))
    el("bb-plot-e").innerHTML = chart(series, "t [s]", "e [mm]", hlines=[e_max, -e_max], aspect=aspect("bb-plot-e"))
    el("bb-readout").innerHTML = readout([("wobble frequency", f"{fmt(crossings / (2 * t_max))} Hz", ""),
                                          ("sign changes (6 s)", str(crossings), "")])
    rows = real.get("bangbang")
    if rows:
        flips, last = 0, None
        for r in rows:
            sg = 1 if r["e"] > 0 else -1
            if last is not None and sg != last:
                flips += 1
            last = sg
        dur = max(rows[-1]["t"], 1e-6)
        peak = max(abs(r["e"]) for r in rows)
        el("bb-hw-readout").innerHTML = readout([
            ("real wobble frequency", f"{fmt(flips / (2 * dur))} Hz", ""),
            ("real peak |e|", f"{fmt(peak)} mm", "bad" if peak > e_max else "ok")])


def init_bb():
    slider("bb-v", lambda x: f"{fmt(x, 0)} mm/s", bb["v"], lambda x: (bb.update(v=x), bb_recompute()))
    slider("bb-w", lambda x: f"{fmt(x, 2)} rad/s", bb["w"], lambda x: (bb.update(w=x), bb_recompute()))
    recompute["bb"] = bb_recompute
    wire_hw("bb", "bangbang", lambda: {"v": bb["v"], "omega_mag": bb["w"]})


# -- tab 2: P / PD --------------------------------------------------------------------------
pd = {"kp": 0.15, "kd": 0.0}


def pd_recompute():
    dt, e_max = CAL["dt"], CAL["e_max_mm"]
    track = m.turn_track(scenario["R"])
    log = m.run_fixed_gain(track, scenario["v"], pd["kp"], pd["kd"], dt, e_max)
    peak, blind, _, _ = m.stats(log, e_max)
    t0, t1 = m.turn_times(track, scenario["v"])
    series = [sim_series(log, 2)]
    if "pd" in real:
        series.append(real_series(real["pd"]))
    el("pd-plot-e").innerHTML = chart(series, "t [s]", "e [mm]", hlines=[e_max, -e_max], vspan=(t0, t1),
                                      aspect=aspect("pd-plot-e"))
    e_ss = -(1 / scenario["R"]) * scenario["v"] / max(pd["kp"], 1e-6)
    el("pd-readout").innerHTML = readout([
        ("peak |e|", f"{fmt(peak)} mm", "bad" if peak > e_max else "ok"),
        ("predicted e_ss in turn", f"{fmt(e_ss)} mm", ""),
        ("blind (sensor clipped)", f"{fmt(blind, 1)}%", "bad" if blind > 0 else "ok")])
    real_readout("pd", "pd", e_max)


def init_pd():
    el("scenario-panel-pd").innerHTML = scenario_panel("pd")
    slider("pd-kp", lambda x: fmt(x, 3), pd["kp"], lambda x: (pd.update(kp=x), pd_recompute()))
    slider("pd-kd", lambda x: fmt(x, 2), pd["kd"], lambda x: (pd.update(kd=x), pd_recompute()))
    recompute["pd"] = pd_recompute
    wire_scenario("pd", pd_recompute)
    wire_hw("pd", "pd", lambda: {"v": scenario["v"], "kp": pd["kp"], "kd": pd["kd"]})


# -- tab 3: LQR -------------------------------------------------------------------------------
def lqr_recompute():
    dt, e_max, v, R = CAL["dt"], CAL["e_max_mm"], scenario["v"], scenario["R"]
    q = (gains["qe"], gains["qth"], gains["r"])
    ke_a, kth_a = m.lqr_gains_analytic(v, *q)
    ke, kth = m.lqr_gains_discrete(float(v), dt, *q)
    wn = float(np.sqrt(v * ke_a))
    zeta = kth_a / (2 * wn)
    track = m.turn_track(R)
    log = m.run_fixed_gain(track, v, ke, kth, dt, e_max)
    peak, blind, _, _ = m.stats(log, e_max)
    t0, t1 = m.turn_times(track, v)
    series = [sim_series(log, 2)]
    if "lqr" in real:
        series.append(real_series(real["lqr"]))
    el("lqr-plot-e").innerHTML = chart(series, "t [s]", "e [mm]", hlines=[e_max, -e_max], vspan=(t0, t1),
                                       aspect=aspect("lqr-plot-e"))
    el("lqr-track-plot").innerHTML = track_plot(track, log, e_max)
    el("lqr-gains-readout").innerHTML = readout([
        ("k_e (discrete, used)", fmt(ke, 4), ""), ("k_θ (discrete, used)", fmt(kth, 4), ""),
        ("damping ratio ζ (continuous)", fmt(zeta, 3), ""), ("natural freq ω_n", f"{fmt(wn)} rad/s", "")])
    e_ss = -(1 / R) * v / ke
    ok = abs(e_ss) <= e_max
    el("lqr-readout").innerHTML = readout([
        ("peak |e|", f"{fmt(peak)} mm", "bad" if peak > e_max else "ok"),
        ("predicted e_ss = −κv/k_e", f"{fmt(e_ss)} mm", ""),
        ("verdict", "ok" if ok else "BLIND", "ok" if ok else "bad"),
        ("v_max = k_e·e_max·R at this R", f"{fmt(ke * e_max * R, 0)} mm/s", ""),
        ("blind (sensor clipped)", f"{fmt(blind, 1)}%", "bad" if blind > 0 else "ok")])
    real_readout("lqr", "lqr", e_max)


def shared_note():
    n = el("mpc-shared-gains")
    if n is not None:
        n.textContent = (f"Using Q_e={fmt(gains['qe'])}, Q_θ={fmt(gains['qth'])}, "
                         f"R={fmt(gains['r'], 1)} from the LQR tab.")


def gains_changed(key):
    def f(x):
        gains[key] = x
        lqr_recompute()
        shared_note()
        mpc_recompute()
    return f


def init_lqr():
    el("scenario-panel-lqr").innerHTML = scenario_panel("lqr")
    slider("lqr-qe", lambda x: fmt(x), gains["qe"], gains_changed("qe"))
    slider("lqr-qth", lambda x: fmt(x), gains["qth"], gains_changed("qth"))
    slider("lqr-r", lambda x: fmt(x, 1), gains["r"], gains_changed("r"))
    recompute["lqr"] = lqr_recompute
    wire_scenario("lqr", lqr_recompute)
    wire_hw("lqr", "lqr", lambda: {"v": scenario["v"], "qe": gains["qe"], "qth": gains["qth"], "r": gains["r"]})


# -- tab 5: MPC --------------------------------------------------------------------------------
mpc = {"n": 15, "log_qv": -2.0}


def mpc_recompute():
    if "mpc-plot-e" not in recompute_ready:
        return
    dt, e_max, v = CAL["dt"], CAL["e_max_mm"], scenario["v"]
    cands = tuple(c for c in m.MPC_CANDIDATES if c <= CAL["wheel_max_mms"] * 0.95)
    qv = 10 ** mpc["log_qv"]
    track = m.turn_track(scenario["R"])
    run_m = m.run_mpc(track, dt, e_max, 14.0, cands, int(mpc["n"]), v, gains["qe"], gains["qth"], gains["r"], qv)
    ke, kth = m.lqr_gains_discrete(float(v), dt, gains["qe"], gains["qth"], gains["r"])
    run_l = m.run_fixed_gain(track, v, ke, kth, dt, e_max)
    t0, t1 = m.turn_times(track, v)
    e_series = [sim_series(run_m, 2, "mpc", "MPC (sim)"), sim_series(run_l, 2, "lqrd", "LQR (sim)")]
    v_series = [sim_series(run_m, 4, "mpc", "MPC (sim)"), sim_series(run_l, 4, "lqrd", "LQR (sim)")]
    if "mpc" in real:
        e_series.append(real_series(real["mpc"]))
        v_series.append(real_series(real["mpc"], "v"))
    el("mpc-plot-e").innerHTML = chart(e_series, "t [s]", "e [mm]", hlines=[e_max, -e_max], vspan=(t0, t1),
                                       aspect=aspect("mpc-plot-e"))
    el("mpc-plot-v").innerHTML = chart(v_series, "t [s]", "v [mm/s]", vspan=(t0, t1), aspect=aspect("mpc-plot-v"))
    pm, bm, lm, vm = m.stats(run_m, e_max)
    pl, bl, ll, vl = m.stats(run_l, e_max)
    el("mpc-readout").innerHTML = readout([
        ("MPC peak |e|", f"{fmt(pm)} mm", "bad" if pm > e_max else "ok"),
        ("LQR peak |e|", f"{fmt(pl)} mm", "bad" if pl > e_max else "ok"),
        ("MPC blind", f"{fmt(bm, 1)}%", "bad" if bm > 0 else "ok"),
        ("LQR blind", f"{fmt(bl, 1)}%", "bad" if bl > 0 else "ok"),
        ("MPC lap time", f"{fmt(lm)} s", ""), ("LQR lap time", f"{fmt(ll)} s", ""),
        ("MPC mean v", f"{fmt(vm, 0)} mm/s", ""), ("LQR mean v", f"{fmt(vl, 0)} mm/s", "")])
    # convergence: an N-step solve against the infinite-horizon LQR gain
    x = (4.0, 0.1)
    A, B = m.discrete_model(v, dt)
    Q, Rm = np.diag([gains["qe"], gains["qth"]]) * dt, np.array([[gains["r"] * dt]])
    K, _ = m.dlqr(A, B, Q, Rm)
    w_inf = float((-K @ np.array([[x[0]], [x[1]]]))[0, 0])
    w_n = m.horizon_gain(max(1, round(mpc["n"])), x, v, dt, gains["qe"], gains["qth"], gains["r"])
    el("conv-readout").innerHTML = readout([
        (f"ω₀ at N={int(mpc['n'])}", f"{fmt(w_n, 6)} rad/s", ""),
        ("ω₀ at N=∞ (LQR)", f"{fmt(w_inf, 6)} rad/s", ""),
        ("gap", f"{abs(w_n - w_inf):.2e}", "")])
    real_readout("mpc", "mpc", e_max, with_v=True)


recompute_ready = set()


def init_mpc():
    el("scenario-panel-mpc").innerHTML = scenario_panel("mpc")
    shared_note()
    slider("mpc-n", lambda x: str(int(x)), mpc["n"], lambda x: (mpc.update(n=round(x)), mpc_recompute()))
    slider("mpc-qv", lambda x: fmt(10 ** x, 3), mpc["log_qv"], lambda x: (mpc.update(log_qv=x), mpc_recompute()))
    recompute["mpc"] = mpc_recompute
    recompute_ready.add("mpc-plot-e")
    wire_scenario("mpc", mpc_recompute)
    wire_hw("mpc", "mpc", lambda: {"v": scenario["v"], "qe": gains["qe"], "qth": gains["qth"], "r": gains["r"],
                                    "n": mpc["n"], "qv": 10 ** mpc["log_qv"]})


def recompute_all():
    bb_recompute()
    for f in ("pd", "lqr", "mpc"):
        recompute[f]()


# -- running on the robot -------------------------------------------------------------------------
HW_PREFIXES = []


def set_hw_status(prefix, text, cls=""):
    s = el(prefix + "-hw-status")
    s.textContent = text
    s.className = "hw-status" + (" " + cls if cls else "")


def refresh_hw():
    ready = cs.connected and dm.connected
    for p in HW_PREFIXES:
        btn = el(p + "-hw-run")
        btn.disabled = state["running"] or not ready or not window.navigator.bluetooth
        el(p + "-hw-stop").disabled = not state["running"]
        if state["running"]:
            continue
        if not window.navigator.bluetooth:
            set_hw_status(p, "This browser has no Web Bluetooth. Use Chrome or Edge.", "err")
        elif not ready:
            set_hw_status(p, "Connect the sensor and the motors on the \"Your robot\" tab first.", "err")
        else:
            set_hw_status(p, "Ready.", "ok")


def wire_hw(prefix, name, get_params):
    HW_PREFIXES.append(prefix)
    run_btn = el(prefix + "-hw-run")
    stop = document.createElement("button")
    stop.id, stop.className, stop.textContent = prefix + "-hw-stop", "btn secondary", "■ Stop"
    stop.disabled = True
    run_btn.after(stop)

    async def go(*_):
        duration = float(el(prefix + "-hw-duration").value or 8)
        if not window.confirm(f"This drives the real robot for {duration:g}s using the {name} controller. "
                              f"Make sure the track is clear. Continue?"):
            return
        p = {"v": 150.0, "omega_mag": 2.0, "kp": 0.15, "kd": 0.0, "qe": 1.0, "qth": 0.0, "r": 25.0,
             "n": 15, "qv": 0.01}
        p.update(get_params())
        ctl = hw.make_controller(name, p, CAL)
        bot = hw.LineFollower(dm, cs, CAL)
        state["running"], state["stop"] = True, False
        refresh_hw()
        set_hw_status(prefix, f"Running {name} on the robot for {duration:g}s…")
        rows = []
        try:
            bot.start()
            async for k, t in bot.loop(duration, lambda: state["stop"] or not (dm.connected and cs.connected)):
                e, theta, sat = bot.read()
                v, omega = ctl(e, theta)
                left, right = bot.drive(v, omega)
                rows.append({"t": t, "e": e, "theta": theta, "v": v, "omega": omega,
                             "saturated": bool(sat), "left": left, "right": right})
            msg, cls = f"Done -- {len(rows)} samples captured.", "ok"
        except Exception as exc:
            msg, cls = f"Run failed: {exc}", "err"
        finally:
            bot.stop()
            state["running"] = False
        if len(rows) > 2:
            real[name] = rows
            recompute_all()
        refresh_hw()
        set_hw_status(prefix, msg, cls)

    run_btn.addEventListener("click", create_proxy(lambda e: asyncio.ensure_future(go())))
    stop.addEventListener("click", create_proxy(lambda e: state.update(stop=True)))


# -- tab 0: your robot (connect, calibrate, oscilloscope) ------------------------------------------------
def render_cal():
    el("cal-subtitle").textContent = (
        f"Using {CAL_SOURCE} — e_max={CAL['e_max_mm']:g}mm, track width={CAL['track_width_mm']:g}mm, "
        f"wheel max={CAL['wheel_max_mms']:g}mm/s, dt={CAL['dt'] * 1000:.0f}ms")
    rows = [("reflect_black", f"{CAL['reflect_black']:g}"), ("reflect_white", f"{CAL['reflect_white']:g}"),
            ("contrast (white − black)", fmt(CAL["reflect_white"] - CAL["reflect_black"], 1)),
            ("e_max_mm (sensor window, half-width)", f"{CAL['e_max_mm']:.1f}mm"),
            ("track_width_mm", f"{CAL['track_width_mm']:.1f}mm"),
            ("wheel_max_mms (speed at 100%)", f"{CAL['wheel_max_mms']:.0f}mm/s"),
            ("dt (control period)", f"{CAL['dt'] * 1000:.0f}ms")]
    el("cal-table").innerHTML = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    banner = el("cal-banner")
    banner.innerHTML = "" if "saved" in CAL_SOURCE else (
        '<div class="banner">No calibration saved in this browser yet, so this page shows generic '
        'reference-build numbers. Connect your robot below and use "Redo calibration" to measure your own.</div>')


def set_recal_status(text, cls=""):
    s = el("recal-status")
    s.textContent = text
    s.className = "hw-status" + (" " + cls if cls else "")


def show(id, on=True):
    el(id).style.display = "block" if on else "none"


def refresh_connection():
    for dev, key in ((cs, "cs"), (dm, "dm")):
        el(f"conn-{key}-status").textContent = "connected" if dev.connected else "not connected"
        el(f"conn-{key}").disabled = dev.connected
    if cs.connected and el("recal-step-start").style.display != "none":
        show("recal-step-start", False)
        show("recal-step-sensor", True)
        set_recal_status("Sensor connected.", "ok")
    refresh_hw()


hw.hooks["log"] = lambda text: set_recal_status(text)
hw.hooks["changed"] = refresh_connection

scope = {"t0": None, "buf": []}


def reflection():
    v = cs.sensor.reflection if cs.connected else None
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def draw_scope():
    hl = [CAL["reflect_black"], CAL["reflect_white"]]
    buf = scope["buf"]
    el("scope-plot").innerHTML = chart(
        [dict(xs=[b[0] for b in buf], ys=[b[1] for b in buf], cls="scope", label="")],
        "t [s]", "reflection", hlines=hl, ydomain=(0, 100), aspect=aspect("scope-plot"))


async def scope_loop():
    draw_scope()
    while True:
        r = reflection()
        el("recal-live").textContent = "--" if r is None else f"{r:.0f}"
        if r is not None:
            now = time.time()
            scope["t0"] = scope["t0"] or now
            t = now - scope["t0"]
            scope["buf"] = [b for b in scope["buf"] if t - b[0] <= 12] + [(t, r)]
            draw_scope()
        await asyncio.sleep(0.1)


async def sample(which):
    for i in ("recal-sample-black", "recal-sample-white"):
        el(i).disabled = True
    out = el(f"recal-{which}-result")
    vals = []
    for i in range(60):
        out.textContent = f"sampling… {3 - i * 0.05:.1f} s"
        r = reflection()
        if r is not None:
            vals.append(r)
        await asyncio.sleep(0.05)
    if vals:
        pending["reflect_" + which] = float(np.mean(vals))
        out.textContent = f"{pending['reflect_' + which]:.0f}"
    else:
        out.textContent = "no readings"
    for i in ("recal-sample-black", "recal-sample-white"):
        el(i).disabled = False


def set_window(*_):
    span = float(el("recal-span").value or 12)
    pending["e_max_mm"] = span / 2.0
    el("recal-window-result").textContent = f"e_max = {span / 2:.1f} mm"


async def to_motors():
    if not dm.connected:
        set_recal_status("Pick the Double Motor in the Bluetooth dialog…")
        await dm.connect_web()
    if dm.connected:
        show("recal-step-sensor", False)
        show("recal-step-motors", True)
        el("recal-trackwidth").value = pending["track_width_mm"]
        set_recal_status("Motors connected.", "ok")


async def drive_test():
    el("recal-drive-test").disabled = True
    set_recal_status("Driving straight for 2 s at 100%. Measure how far it went.")
    dm.movement_move_tank(100, 100, blocking=False)
    await asyncio.sleep(2.0)
    dm.movement_stop(blocking=False)
    el("recal-drive-test").disabled = False
    set_recal_status("Enter the distance it travelled, then save.", "ok")


def save_cal(*_):
    global CAL_SOURCE
    dist = el("recal-distance").value
    width = el("recal-trackwidth").value
    if dist:
        pending["wheel_max_mms"] = float(dist) / 2.0
    if width:
        pending["track_width_mm"] = float(width)
    CAL.update(pending)
    CAL_SOURCE = "your calibration (saved in this browser)"
    try:
        window.localStorage.setItem("lqrmpc-cal", json.dumps(CAL))
    except Exception:
        pass
    show("recal-step-motors", False)
    show("recal-done", True)
    set_recal_status("Saved.", "ok")
    render_cal()
    draw_scope()
    recompute_all()


def init_robot():
    render_cal()
    el("recal-span").value = CAL["e_max_mm"] * 2
    on("conn-cs", "click", lambda e: asyncio.ensure_future(cs.connect_web()))
    on("conn-dm", "click", lambda e: asyncio.ensure_future(dm.connect_web()))
    on("recal-connect-sensor", "click", lambda e: asyncio.ensure_future(cs.connect_web()))
    on("recal-sample-black", "click", lambda e: asyncio.ensure_future(sample("black")))
    on("recal-sample-white", "click", lambda e: asyncio.ensure_future(sample("white")))
    on("recal-set-window", "click", set_window)
    on("recal-to-motors", "click", lambda e: asyncio.ensure_future(to_motors()))
    on("recal-drive-test", "click", lambda e: asyncio.ensure_future(drive_test()))
    on("recal-save", "click", save_cal)
    set_recal_status("Connect the sensor over Bluetooth to start.")
    asyncio.ensure_future(scope_loop())


init_robot()
init_bb()
init_pd()
init_lqr()
init_mpc()
recompute_all()
refresh_connection()
el("loading-banner").style.display = "none"
