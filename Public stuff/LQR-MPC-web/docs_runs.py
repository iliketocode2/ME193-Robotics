"""The "run it yourself" boxes in the notes.

Each box runs, in your browser, what the matching command would have printed on a
laptop: the report at the bottom of sim.py, or one of the run_*.py scripts. The
scripts normally drive a real robot, so here they run on the simulator's model of
it (same controllers, same print format). To run on your own robot, use the
Interactive page.
"""

import contextlib
import io
import re
import traceback

import numpy as np
from pyodide.ffi import create_proxy
from pyscript import document, window

import lqrmpc as m
from ui import chart, el, slider

BAR = "=" * 74


# -- the report at the bottom of sim.py ------------------------------------------------
def _summarize(name, log, e_max=m.E_MAX):
    t, s, e, theta, v, omega = log.T
    lost = np.abs(e) > e_max
    return (f"{name:26s} peak |e| = {np.abs(e).max():6.2f} mm   "
            f"blind {100 * lost.mean():5.1f}% of the run   "
            f"lap {t[-1] + (t[1] - t[0]):5.2f} s   "
            f"mean v {v.mean():6.1f} mm/s")


def sim_sections():
    out = {}
    V, E_MAX = m.V_REF, m.E_MAX
    k_e, k_th = m.lqr_gains_analytic(V)

    lines = [BAR, "1. LQR GAINS  (Q = diag(%g, %g), R = %g, v = %g mm/s)" % (m.Q_E, m.Q_TH, m.R_W, V), BAR]
    wn = np.sqrt(V * k_e)
    zeta = k_th / (2 * wn)
    lines += [f"  analytic (continuous):  k_e = {k_e:.4f} rad/s/mm   k_theta = {k_th:.4f} 1/s",
              f"  natural frequency wn  = {wn:.3f} rad/s",
              f"  damping ratio  zeta   = {zeta:.4f}   (= 1/sqrt(2) whenever q_th = 0)",
              f"  settle 4/(zeta*wn)    = {4 / (zeta * wn):.2f} s = {4 / (zeta * wn) * V:.0f} mm of travel", ""]
    for dt in (0.02, 0.05, 0.10, 0.20):
        kd_e, kd_th = m.lqr_gains_discrete(V, dt)
        A, B = m.discrete_model(V, dt)
        eig = np.linalg.eigvals(A - B @ np.array([[kd_e, kd_th]]))
        lines.append(f"  discrete dt={dt:4.2f}s:  k_e = {kd_e:.4f}  k_theta = {kd_th:.4f}"
                     f"   |closed-loop poles| = {np.abs(eig)[0]:.3f}")
    out[1] = "\n".join(lines)

    r_black, r_white = 12.0, 85.0
    span = r_white - r_black
    lines = [BAR, f"2. THE THREE SENSOR STATES  (black={r_black:g}, white={r_white:g}, theta = 0)", BAR,
             f"  {'sensor state':26s} {'refl':>5s} {'s':>6s} {'e mm':>7s} {'w rad/s':>9s} {'vL %':>7s} {'vR %':>7s}"]
    for name, f in [("fully ON the line", 0.0), ("edge, drifted onto line", 0.35), ("edge, on target", 0.5),
                    ("edge, drifted off line", 0.65), ("fully OFF the line", 1.0)]:
        refl = r_black + f * span
        s_norm = (refl - r_black) / span
        e = float(np.clip(2 * E_MAX * (s_norm - 0.5), -E_MAX, E_MAX))
        omega = -k_e * e
        v_l, v_r = m.wheel_speeds(V, omega)
        lines.append(f"  {name:26s} {refl:5.1f} {s_norm:6.3f} {e:+7.2f} {omega:+9.3f} "
                     f"{m.to_percent(v_l):7.1f} {m.to_percent(v_r):7.1f}")
    out[2] = "\n".join(lines)

    lines = [BAR, "3. WHY LQR BREAKS ON SHARP TURNS:  e_ss = -kappa*v/k_e", BAR,
             f"  {'turn radius':>12s} {'kappa 1/mm':>11s} {'e_ss @150':>10s} {'verdict':>9s} "
             f"{f'max v for |e|<{E_MAX:g}mm':>19s}"]
    for radius in (np.inf, 300.0, 150.0, 80.0, 40.0):
        kappa = 0.0 if np.isinf(radius) else 1.0 / radius
        e_ss = -kappa * V / k_e
        v_lim = np.inf if kappa == 0 else E_MAX * k_e / kappa
        ok = "ok" if abs(e_ss) <= E_MAX else "BLIND"
        rad = "straight" if np.isinf(radius) else f"{radius:.0f} mm"
        lim = "any" if np.isinf(v_lim) else f"{v_lim:.0f} mm/s"
        lines.append(f"  {rad:>12s} {kappa:11.5f} {e_ss:+9.2f} {ok:>10s} {lim:>19s}")
    out[3] = "\n".join(lines)

    lines = [BAR, "4. ONE MPC SOLVE, entering the R=80mm turn (s=390mm, e=0, theta=0)", BAR]
    _, table = m.mpc_choose_speed(0.0, 0.0, 390.0, m.DEMO_TRACK, m.CANDIDATES)
    best_v = min(table, key=lambda row: row[2])[0]
    lines.append(f"  {'candidate v':>12s} {'peak |e| over horizon':>23s} {'cost J':>10s}   choice")
    for v, peak, cost in table:
        flag = "  <-- chosen" if v == best_v else ("  (violates window)" if peak > E_MAX else "")
        lines.append(f"  {v:9.0f} mm/s {peak:19.2f} mm {cost:10.2f}{flag}")
    out[4] = "\n".join(lines)

    log_lqr, _ = m.run_lqr(m.DEMO_TRACK)
    log_mpc = m.run_mpc(m.DEMO_TRACK)
    out[5] = "\n".join([BAR, "5. HEAD TO HEAD on 400mm straight -> 90deg R=80mm turn -> 400mm straight", BAR,
                        "  " + _summarize("LQR, fixed v = 150", log_lqr),
                        "  " + _summarize("MPC, v scheduled", log_mpc),
                        f"  MPC speed profile: min {log_mpc[:, 4].min():.0f} mm/s, max {log_mpc[:, 4].max():.0f} mm/s"])

    A, B = m.discrete_model(V)
    Q, R = np.diag([m.Q_E, m.Q_TH]) * m.DT, np.array([[m.R_W * m.DT]])
    K_inf, _ = m.dlqr(A, B, Q, R)
    x = np.array([[4.0], [0.1]])
    u_lqr = float((-K_inf @ x)[0, 0])
    lines = [BAR, "6. MPC WITH NO CONSTRAINTS AND A LONG HORIZON *IS* LQR", BAR,
             "  test state: e = 4.0 mm, theta = 0.10 rad",
             f"  {'horizon N':>10s} {'first move omega_0 [rad/s]':>28s} {'gap to LQR':>12s}"]
    for N in (1, 2, 5, 10, 20, 50, 200):
        u_N = m.horizon_gain(N, (4.0, 0.1))
        lines.append(f"  {N:10d} {u_N:28.6f} {abs(u_N - u_lqr):12.2e}")
    lines.append(f"  {'infinite':>10s} {u_lqr:28.6f} {0.0:12.2e}   <-- the LQR gain K")
    out[6] = "\n".join(lines)
    return out


def run_sim(args):
    """python3 code/sim.py  (args: 'all' or a section number)"""
    sections = sim_sections()
    note = ("No code/calibration.json here -- using the generic reference-build numbers quoted in the docs.\n")
    arg = (args or "all").strip()
    if arg == "all":
        return note + "\n" + "\n\n".join(sections[i] for i in range(1, 7))
    return sections[int(arg)]


# -- the run_*.py scripts, on the simulator's model of the robot -----------------------------
class _ConstantCurvature:
    """MPC preview with no map: 'whatever I am turning now, I will keep turning'."""

    def __init__(self, kappa):
        self._kappa = kappa

    def kappa(self, _s):
        return self._kappa


def _lap_track(length):
    """The demo track (straight, 90 degree turn, straight) repeated until it is longer than the run."""
    seg = m.DEMO_TRACK.segments
    reps = int(length / m.DEMO_TRACK.length) + 2
    return m.Track(list(seg) * reps)


def _floats(args, defaults):
    vals = [float(x) for x in args.split()] if args and args.strip() else []
    return vals + defaults[len(vals):]


def _simulate_script(name, args):
    dt, e_max = m.DT, m.E_MAX
    lines = []
    if name == "bangbang":
        duration, v, omega_mag = _floats(args, [20.0, 100.0, 2.0])
        lines += [f"Bang-bang: v = {v:.0f} mm/s   omega = +-{omega_mag:.2f} rad/s   dt = {dt * 1000:.0f} ms",
                  "Control law: omega = -omega_mag * sign(e)  (one bit of the error, not its size)\n"]
    elif name == "pd":
        duration, v, kp, kd = _floats(args, [20.0, 150.0, 0.15, 0.0])
        lines += [f"{'PD' if kd else 'P'} control: v = {v:.0f} mm/s   kp = {kp:.3f}   kd = {kd:.3f}   dt = {dt * 1000:.0f} ms",
                  "Control law: omega = -kp*e - kd*theta\n"]
    elif name == "lqr":
        duration, v = _floats(args, [20.0, 150.0])
        qe, qth, r = m.Q_E, m.Q_TH, m.R_W
        k_e, k_th = m.lqr_gains_discrete(v, dt, qe, qth, r)
        k_e_c, k_th_c = m.lqr_gains_analytic(v, qe, qth, r)
        lines += [f"Q = diag({qe}, {qth})   R = {r}   v = {v:.0f} mm/s   dt = {dt * 1000:.0f} ms",
                  f"  continuous-time gains : k_e = {k_e_c:.4f}  k_theta = {k_th_c:.4f}",
                  f"  discrete-time gains   : k_e = {k_e:.4f}  k_theta = {k_th:.4f}  <- used",
                  f"  predicted offset on an R = 200 mm curve: {-(1 / 200) * v / k_e:+.1f} mm "
                  f"(window is +-{e_max:.1f} mm)\n"]
    elif name == "mpc":
        (duration,) = _floats(args, [20.0])
        cands = m.CANDIDATES
        horizon = 15
        lines += [f"MPC: {len(cands)} candidate speeds, horizon {horizon} steps = {horizon * dt:.2f} s",
                  f"hard-ish constraint: |e| <= {e_max:.1f} mm over the whole horizon\n"]
    else:
        raise ValueError(name)
    lines.append("(Simulated: this is the controller from the robot script, run on the simulator's model of the robot "
                 "on the demo track repeated. Nothing is logged. To run it on your own robot, use the Interactive page.)\n")

    track = _lap_track(duration * 220.0)
    s = e = theta = 0.0
    kappa_hat, v_cmd = 0.0, m.V_REF
    blind, peak, crossings, sign_prev = 0, 0.0, 0, None
    n = int(duration / dt)
    for k in range(n):
        t = k * dt
        measured = m.coerce(e, e_max)
        saturated = abs(e) >= 0.96 * e_max
        extra = ""
        if name == "bangbang":
            sign = 1.0 if measured > 0 else -1.0
            omega, v_cmd = -omega_mag * sign, v
            if sign_prev is not None and sign != sign_prev:
                crossings += 1
            sign_prev = sign
        elif name == "pd":
            omega, v_cmd = -kp * measured - kd * theta, v
        elif name == "lqr":
            omega, v_cmd = -k_e * measured - k_th * theta, v
        else:
            v_cmd, table = m.mpc_choose_speed(measured, theta, 0.0, _ConstantCurvature(kappa_hat), cands, horizon, dt, e_max)
            kd_e, kd_th = m.lqr_gains_discrete(float(v_cmd), dt)
            omega = -kd_e * measured - kd_th * theta
            kappa_hat = 0.8 * kappa_hat + 0.2 * (omega / max(v_cmd, 20.0))
            radius = 1e9 if abs(kappa_hat) < 1e-6 else 1.0 / abs(kappa_hat)
            rejected = sum(1 for _, pk, _ in table if pk > e_max)
            extra = (radius, rejected, len(cands))
        left = int(round(m.to_percent(m.wheel_speeds(v_cmd, omega)[0])))
        right = int(round(m.to_percent(m.wheel_speeds(v_cmd, omega)[1])))
        blind += saturated
        peak = max(peak, abs(measured))
        if k % 10 == 0:
            tag = "  BLIND" if saturated else ""
            if name == "bangbang":
                lines.append(f"  t={t:5.2f}s  e={measured:+6.2f}mm  omega={omega:+6.2f}rad/s  motors={left:+4d}/{right:+4d}{tag}")
            elif name == "mpc":
                radius, rejected, total = extra
                lines.append(f"  t={t:5.2f}s  e={measured:+6.2f}mm  R~{min(radius, 9999):6.0f}mm  v={v_cmd:5.0f}mm/s  "
                             f"omega={omega:+6.2f}rad/s  {rejected}/{total} speeds infeasible{tag}")
            else:
                lines.append(f"  t={t:5.2f}s  e={measured:+6.2f}mm  theta={theta:+6.3f}rad  "
                             f"omega={omega:+6.2f}rad/s  motors={left:+4d}/{right:+4d}{tag}")
        e, theta = m.step(e, theta, v_cmd, omega, track.kappa(s), dt)
        s += v_cmd * dt

    pct = 100.0 * blind / max(n, 1)
    if name == "bangbang":
        wobble = crossings / (2.0 * duration) if duration > 0 else 0.0
        lines.append(f"\npeak |e| = {peak:.2f} mm, blind on {pct:.1f}% of samples, "
                     f"wobble ~ {wobble:.2f} Hz ({crossings} sign changes in {duration:.1f}s)")
    else:
        lines.append(f"\npeak |e| = {peak:.2f} mm, blind on {pct:.1f}% of samples")
    if name == "pd":
        lines.append("Record this (kp, kd, behavior) triple, then try another -- there is no formula yet for which pair is best.")
    return "\n".join(lines)


RUNNERS = {"sim": run_sim,
           "bangbang": lambda a: _simulate_script("bangbang", a),
           "pd": lambda a: _simulate_script("pd", a),
           "lqr": lambda a: _simulate_script("lqr", a),
           "mpc": lambda a: _simulate_script("mpc", a)}




# -- editing sim.py in the page --------------------------------------------------------------
ORIGINAL_SOURCE = open("sim.py").read()
source = {"text": ORIGINAL_SOURCE}


def run_source(text):
    """Run the (possibly edited) sim.py as `python3 code/sim.py` would, and return what it printed."""
    buf = io.StringIO()
    namespace = {"__name__": "__main__", "__file__": "sim.py"}
    ok = True
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(text, "sim.py", "exec"), namespace)
    except BaseException:
        ok = False
        buf.write("\n" + traceback.format_exc())
    return buf.getvalue(), ok


def pick_section(output, n):
    """The part of the report for section n (the sections start with a line of '=' and 'N. TITLE')."""
    heads = [(mm.start(), int(mm.group(2))) for mm in re.finditer(r"(?m)^(={20,})\n(\d)\. ", output)]
    for i, (pos, num) in enumerate(heads):
        if num == n:
            end = heads[i + 1][0] if i + 1 < len(heads) else len(output)
            return output[pos:end].rstrip()
    return output


def wire_editors():
    boxes = document.querySelectorAll(".pyrun")
    areas = []
    for i in range(boxes.length):
        area = boxes.item(i).querySelector("textarea")
        if area is not None:
            area.value = source["text"]
            areas.append(area)
            window.refreshCode(area)

    def make_input(area):
        def on_input(event):
            source["text"] = area.value
            for other in areas:
                if other is not area:
                    other.value = area.value
                    window.refreshCode(other)
        return on_input

    def make_keys(area):
        def on_key(event):
            if event.key == "Tab":          # indent instead of leaving the box
                event.preventDefault()
                start = area.selectionStart
                area.value = area.value[:start] + "    " + area.value[area.selectionEnd:]
                area.selectionStart = area.selectionEnd = start + 4
                source["text"] = area.value
                window.refreshCode(area)
        return on_key

    for i in range(boxes.length):
        box = boxes.item(i)
        area, toggle, reset = box.querySelector("textarea"), box.querySelector(".pyrun-toggle"), box.querySelector(".pyrun-reset")
        if area is None:
            continue
        editor = box.querySelector(".pyrun-editor")
        area.addEventListener("input", create_proxy(make_input(area)))
        area.addEventListener("keydown", create_proxy(make_keys(area)))

        def do_reset(e):
            source["text"] = ORIGINAL_SOURCE
            for other in areas:
                other.value = ORIGINAL_SOURCE
                window.refreshCode(other)
        reset.addEventListener("click", create_proxy(do_reset))


# -- the lab: sim.py's constants as sliders ---------------------------------------------------
def _html_table(header, rows, classes=None):
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>"]
    for i, row in enumerate(rows):
        cls = f' class="{classes[i]}"' if classes and classes[i] else ""
        out.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    return "".join(out) + "</table>"


def _lab_track(radius, second_turn):
    arc = radius * np.pi / 2
    if second_turn:
        return m.Track([(400.0, None), (arc, radius), (300.0, None), (arc, -radius), (400.0, None)])
    return m.Track([(400.0, None), (arc, radius), (400.0, None)])


def _turn_spans(track, v):
    spans, s = [], 0.0
    for length, radius in track.segments:
        if radius is not None:
            spans.append((s / v, (s + length) / v))
        s += length
    return spans


def lab_recompute(_=None):
    qe, qth, r = (float(el(i).value) for i in ("lab-qe", "lab-qth", "lab-r"))
    v, radius, e_max = (float(el(i).value) for i in ("lab-v", "lab-radius", "lab-emax"))
    horizon, qv = int(float(el("lab-n").value)), 10 ** float(el("lab-qv").value)
    second = el("lab-second").checked
    el("lab-qv-val").textContent = f"{qv:.3g}"
    dt = m.DT

    # 1. gains (section 1 of the report)
    ke, kth = m.lqr_gains_analytic(v, qe, qth, r)
    wn = float(np.sqrt(v * ke))
    zeta = kth / (2 * wn)
    kde, kdth = m.lqr_gains_discrete(float(v), dt, qe, qth, r)
    A, B = m.discrete_model(v, dt)
    pole = float(np.abs(np.linalg.eigvals(A - B @ np.array([[kde, kdth]])))[0])
    el("lab-gains").innerHTML = _html_table(["", "value"], [
        ["k_e (rad/s per mm)", f"{ke:.4f}"], ["k_theta (1/s)", f"{kth:.4f}"],
        ["natural frequency wn (rad/s)", f"{wn:.3f}"],
        ["damping ratio zeta", f"{zeta:.4f}" + ("  (= 1/sqrt(2) when Q_th = 0)" if qth == 0 else "")],
        ["settle 4/(zeta*wn)", f"{4 / (zeta * wn):.2f} s = {4 / (zeta * wn) * v:.0f} mm"],
        [f"discrete gains at dt = {dt:g} s", f"{kde:.4f}, {kdth:.4f}"],
        ["closed-loop pole radius", f"{pole:.3f}"]])

    # 3. e_ss and the speed limit against turn radius (section 3)
    rows, cls = [], []
    for rad in (None, 300.0, 150.0, 80.0, 40.0):
        kappa = 0.0 if rad is None else 1.0 / rad
        e_ss = -kappa * v / kde
        v_lim = float("inf") if kappa == 0 else e_max * ke / kappa
        blind = abs(e_ss) > e_max
        rows.append(["straight" if rad is None else f"{rad:.0f} mm", f"{kappa:.5f}", f"{e_ss:+.2f}",
                     "BLIND" if blind else "ok", "any" if v_lim == float("inf") else f"{v_lim:.0f} mm/s"])
        cls.append("bad" if blind else "")
    el("lab-radii").innerHTML = _html_table(
        ["turn radius", "kappa 1/mm", f"e_ss at {v:.0f}", "verdict", f"max v for |e|<{e_max:g}mm"], rows, cls)

    # 4. one MPC solve entering the first turn (section 4)
    track = _lab_track(radius, second)
    lead = track.segments[0][0]
    best, table = m.mpc_choose_speed(0.0, 0.0, lead - 10.0, track, m.CANDIDATES, horizon, dt, e_max,
                                     v, qe, qth, r, qv)
    rows, cls = [], []
    for cv, peak, cost in table:
        rows.append([f"{cv:.0f} mm/s", f"{peak:.2f} mm", f"{cost:.2f}",
                     "<-- chosen" if cv == best else ("violates window" if peak > e_max else "")])
        cls.append("pick" if cv == best else ("bad" if peak > e_max else ""))
    el("lab-solve").innerHTML = _html_table(["candidate v", "peak |e| over horizon", "cost J", ""], rows, cls)

    # 5. head to head (section 5) and the plots
    kde2, kdth2 = m.lqr_gains_discrete(float(v), dt, qe, qth, r)
    log_l = m.run_fixed_gain(track, v, kde2, kdth2, dt, e_max, 24.0)
    log_m = m.run_mpc(track, dt, e_max, 24.0, m.CANDIDATES, horizon, v, qe, qth, r, qv)
    rows = []
    for name, cls_, log in (("LQR, fixed v", "lqrd-c", log_l), ("MPC, v scheduled", "mpc-c", log_m)):
        pk, blind, lap, mv = m.stats(log, e_max)
        rows.append([f'<i class="swatch {cls_}"></i>{name}', f"{pk:.2f} mm", f"{blind:.1f}%", f"{lap:.2f} s", f"{mv:.0f} mm/s"])
    el("lab-stats").innerHTML = _html_table(["controller", "peak |e|", "blind", "lap", "mean v"], rows)
    spans = _turn_spans(track, v)
    el("lab-pe").innerHTML = chart(
        [dict(xs=log_m[:, 0], ys=log_m[:, 2], cls="mpc", label="MPC"),
         dict(xs=log_l[:, 0], ys=log_l[:, 2], cls="lqrd", label="LQR")],
        "t [s]", "e [mm]", hlines=[e_max, -e_max], vspan=spans, aspect=0.4)
    el("lab-pv").innerHTML = chart(
        [dict(xs=log_m[:, 0], ys=log_m[:, 4], cls="mpc", label="MPC"),
         dict(xs=log_l[:, 0], ys=log_l[:, 4], cls="lqrd", label="LQR")],
        "t [s]", "v [mm/s]", vspan=spans, ydomain=(0, 300), aspect=0.3)


def wire_lab():
    if el("lab-qth") is None:
        return
    defaults = [("lab-qe", 1.0, "{:g}"), ("lab-qth", 0.0, "{:g}"), ("lab-r", 25.0, "{:g}"),
                ("lab-v", 150.0, "{:.0f}"), ("lab-radius", 80.0, "{:.0f}"), ("lab-emax", 6.0, "{:g}"),
                ("lab-n", 15.0, "{:.0f}"), ("lab-qv", -2.0, "{:g}")]
    for id, initial, f in defaults:
        slider(id, lambda x, f=f: f.format(x), initial, lab_recompute)
    el("lab-second").addEventListener("change", create_proxy(lab_recompute))
    lab_recompute()
    el("lab-loading").style.display = "none"


# -- wiring to the boxes the reader built --------------------------------------------------------
def wire():
    boxes = document.querySelectorAll(".pyrun")
    for i in range(boxes.length):
        box = boxes.item(i)
        name = box.dataset.run
        arg_input = box.querySelector("input")
        button = box.querySelector("button")
        out = box.querySelector("pre.pyrun-out")

        def handler(event, name=name, arg_input=arg_input, out=out, box=box):
            args = arg_input.value if arg_input is not None else box.dataset.section
            toggle = box.querySelector(".pyrun-toggle")
            out.classList.remove("err")
            try:
                if toggle is not None and toggle.getAttribute("aria-pressed") == "true":      # run the edited sim.py
                    text, ok = run_source(source["text"])
                    sec = box.dataset.section
                    if ok and sec and sec != "all":
                        text = pick_section(text, int(sec))
                    out.textContent = text
                    if not ok:
                        out.classList.add("err")
                else:
                    out.textContent = RUNNERS[name](args)
            except Exception as exc:
                out.textContent = f"Could not run: {exc}"
            out.style.display = "block"

        button.addEventListener("click", create_proxy(handler))
        button.disabled = False
        button.textContent = "▶ Run"


wire()
wire_editors()
wire_lab()
