"""
AnnaCode — transfer_plot.py

Two plots of the network's response to one swept input (any other inputs are
held at their current values; a dot marks the live point):

  Basic -- one green line: first input vs first output, shown to the
              right of the network.
  Advanced -- below the run button: one panel per hidden layer plus one panel
              per output. Every neuron gets a solid line (after its activation)
              and a dashed line (before it).
"""
import asyncio
import json
import js
from pyscript import document
from pyscript.js_modules import Plotly

import state
import network_model

PLOT_ID = "transfer-plot"
ADV_ID = "adv-plot"
ADV_SELECT_ID = "adv-x-select"
LINE_COLOR = "#16a34a"   # green: the network's output line
POINTS = 41
PANEL_HEIGHT = 300
PANELS_PER_ROW = 3
OUTPUT_COLORS = [LINE_COLOR, "#0d9488", "#65a30d", "#0891b2"]   # greens, one per output
GUIDE_COLOR = "#dc2626"   # thin red lines from the live point to the axes
HIDDEN_COLORS = ["#2563eb", "#7c3aed", "#db2777", "#059669", "#dc2626", "#0891b2", "#9333ea", "#65a30d"]
CONFIG = {"displayModeBar": False, "staticPlot": True, "responsive": True}

_adv_panel_count = None
_adv_select_key = ""

def _js(obj):
    return js.JSON.parse(json.dumps(obj))

def _axis_label(kind: str, item_id: int, fallback: str) -> str:
    """What the student picked for this input/output, e.g. 'Reflection (Color Sensor)'."""
    dev_el = state.get_id(f"dev-{kind}-{item_id}")
    chan_el = state.get_id(f"chan-{kind}-{item_id}")
    dev = dev_el.value if dev_el else ""
    chan = chan_el.value if chan_el else ""
    if chan and dev:
        return f"{chan} ({dev})"
    return chan or dev or fallback

def _sweep_axis(base: list[float], idx: int):
    span = max(100.0, abs(base[idx]))
    return [-span + 2 * span * k / (POINTS - 1) for k in range(POINTS)]

# ── Basic ─────────────────────────────────────────────────────────────────

def redraw_beginner():
    if not state.inputs or not state.outputs or not state.layers:
        return
    if not state.get_id(PLOT_ID):
        return

    inp = state.inputs[0]
    base = [state.input_values.get(i["id"], 0.0) for i in state.inputs]
    xs = _sweep_axis(base, 0)
    x_now = base[0]

    # One line per output: the whole path from the input to that output (the neurons along it, summed up)
    def outputs_at(x):
        return network_model.evaluate([x] + base[1:])

    sweep = [outputs_at(x) for x in xs]
    now = outputs_at(x_now)
    outs = [(k, o) for k, o in enumerate(state.outputs) if k < len(now)]
    if not outs:
        return

    def short_label(out, k):
        el = state.get_id(f"chan-out-{out['id']}")
        return el.value if el and el.value else f"output {k + 1}"

    guide = {"color": GUIDE_COLOR, "width": 1}   # thin red lines from the live point to the axes
    traces = []
    for n, (k, out) in enumerate(outs):
        color = OUTPUT_COLORS[n % len(OUTPUT_COLORS)]
        traces.append({"x": xs, "y": [v[k] for v in sweep], "mode": "lines", "name": short_label(out, k),
                       "line": {"color": color, "width": 3}, "hoverinfo": "skip"})
    for k, out in outs:
        traces.append({"x": [x_now, x_now], "y": [0, now[k]], "mode": "lines", "line": guide,
                       "showlegend": False, "hoverinfo": "skip"})
        traces.append({"x": [0, x_now], "y": [now[k], now[k]], "mode": "lines", "line": guide,
                       "showlegend": False, "hoverinfo": "skip"})
    for k, out in outs:
        traces.append({"x": [x_now], "y": [now[k]], "mode": "markers", "showlegend": False,
                       "marker": {"color": "#1a1d2e", "size": 10}, "hoverinfo": "skip"})

    many = len(outs) > 1
    layout = {
        "margin": {"l": 55, "r": 20, "t": 10, "b": 90 if many else 45},
        "xaxis": {"title": {"text": _axis_label("in", inp["id"], inp["name"]), "font": {"size": 13}}, "zeroline": True},
        "yaxis": {"title": {"text": "outputs" if many else _axis_label("out", outs[0][1]["id"], "output"),
                            "font": {"size": 13}}, "zeroline": True},
        "showlegend": many,
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.42, "yanchor": "top",
                   "font": {"size": 11}},
        "paper_bgcolor": "rgba(0,0,0,0)",
    }
    box = state.get_id(PLOT_ID)   # the slot can change size (e.g. a wider network); Plotly won't notice by itself
    if box.clientWidth:
        layout["width"], layout["height"] = box.clientWidth, box.clientHeight
    Plotly.react(PLOT_ID, _js(traces), _js(layout), _js(CONFIG))

# ── Advanced ─────────────────────────────────────────────────────────────────

def _sync_x_select():
    """Keep the x-axis dropdown in step with the inputs; returns the chosen input index."""
    global _adv_select_key
    sel = state.get_id(ADV_SELECT_ID)
    names = [_axis_label("in", i["id"], i["name"]) for i in state.inputs]
    key = "|".join(names)
    if key != _adv_select_key:
        prev = int(sel.value) if sel.value else 0
        sel.innerHTML = "".join(f'<option value="{k}">{n}</option>' for k, n in enumerate(names))
        sel.value = str(min(prev, len(names) - 1))
        _adv_select_key = key
    return min(int(sel.value or 0), len(names) - 1)

def _y_range(entries, sweeps, live):
    """Size a panel to the solid (after-activation) lines. The dashed before-activation line only widens it when
    it is of similar size; otherwise it runs off the chart (a Sigmoid's 0-to-1 curve would be squashed flat by a
    dashed line spanning -100 to 100)."""
    post = [0.0] + [live[li][j][1] for _, _, li, j in entries]
    pre = []
    for _, _, li, j in entries:
        post += [s[li][j][1] for s in sweeps]
        pre += [s[li][j][0] for s in sweeps]
    lo, hi = min(post), max(post)
    if pre and (max(pre) - min(pre)) <= 4 * max(hi - lo, 1e-9):
        lo, hi = min(lo, min(pre)), max(hi, max(pre))
    pad = (hi - lo) * 0.08 or 1.0
    return [lo - pad, hi + pad]

def redraw_advanced():
    global _adv_panel_count
    el = state.get_id(ADV_ID)
    if not el or not state.inputs or not state.outputs or not state.layers:
        return

    xi = _sync_x_select()
    base = [state.input_values.get(i["id"], 0.0) for i in state.inputs]
    xs = _sweep_axis(base, xi)
    _update_collapse_note([_axis_label("in", i["id"], i["name"]) for i in state.inputs])

    def sweep(x):
        vals = list(base)
        vals[xi] = x
        return network_model.evaluate_layers(vals)

    sweeps = [sweep(x) for x in xs]
    live = sweep(base[xi])

    # panels: [(y-axis title, [(legend name, colour, layer idx, neuron idx)])]
    panels = []
    last = len(state.layers) - 1
    color_n = 0
    for li, layer in enumerate(state.layers[:last]):
        entries = []
        for j in range(len(layer["neurons"])):
            entries.append((f"L{li + 1} n{j + 1}", HIDDEN_COLORS[color_n % len(HIDDEN_COLORS)], li, j))
            color_n += 1
        panels.append((f"Layer {li + 1}", entries))
    # All the outputs share one panel, each in its own shade of green
    out_entries, out_labels = [], []
    for j, out in enumerate(state.outputs):
        if j < len(state.layers[last]["neurons"]):
            label = _axis_label("out", out["id"], f"output {j + 1}")
            out_entries.append((f"{label} (n{j + 1})", OUTPUT_COLORS[len(out_entries) % len(OUTPUT_COLORS)], last, j))
            out_labels.append(label)
    if out_entries:
        panels.append((out_labels[0] if len(out_labels) == 1 else "outputs", out_entries))

    # Panels sit side by side, wrapping onto another row after PANELS_PER_ROW
    cols = min(len(panels), PANELS_PER_ROW)
    rows = -(-len(panels) // PANELS_PER_ROW)
    area = rows * PANEL_HEIGHT   # plot-area height in px, so the legend can be placed just under the axes
    traces, layout = [], {
        "margin": {"l": 60, "r": 20, "t": 10, "b": 110},
        "grid": {"rows": rows, "columns": cols, "pattern": "independent",
                 "roworder": "top to bottom", "xgap": 0.12, "ygap": 0.4},
        "showlegend": True,
        "legend": {"orientation": "h", "x": 0, "y": -75 / area, "yanchor": "top", "font": {"size": 11}},
        "paper_bgcolor": "rgba(0,0,0,0)",
    }
    x_title = {"text": _axis_label("in", state.inputs[xi]["id"], state.inputs[xi]["name"]), "font": {"size": 12}}
    for k, (title, entries) in enumerate(panels, start=1):
        sfx = "" if k == 1 else str(k)
        layout[f"xaxis{sfx}"] = {"zeroline": True, "title": x_title}
        layout[f"yaxis{sfx}"] = {"title": {"text": title, "font": {"size": 12}}, "zeroline": True,
                                 "range": _y_range(entries, sweeps, live)}
        axes = {"xaxis": f"x{sfx}", "yaxis": f"y{sfx}", "hoverinfo": "skip"}
        for name, color, li, j in entries:
            group = f"{li}-{j}"
            common = {**axes, "x": xs, "mode": "lines", "legendgroup": group}
            traces.append({**common, "y": [s[li][j][0] for s in sweeps], "showlegend": False,
                           "line": {"color": color, "width": 1.5, "dash": "dash"}})
            traces.append({**common, "y": [s[li][j][1] for s in sweeps], "name": name,
                           "line": {"color": color, "width": 3}})
            # the live point: thin red lines to the axes, then a black dot (as in Basic)
            x_now, y_now = base[xi], live[li][j][1]
            guide = {**axes, "mode": "lines", "legendgroup": group, "showlegend": False,
                     "line": {"color": GUIDE_COLOR, "width": 1}}
            traces.append({**guide, "x": [x_now, x_now], "y": [0, y_now]})
            traces.append({**guide, "x": [0, x_now], "y": [y_now, y_now]})
            traces.append({**axes, "x": [x_now], "y": [y_now], "mode": "markers",
                           "legendgroup": group, "showlegend": False,
                           "marker": {"color": "#000", "size": 9, "line": {"color": "#fff", "width": 1.5}}})

    el.style.height = f"{area + 110}px"
    if (rows, cols, len(panels)) != _adv_panel_count:
        Plotly.purge(ADV_ID)   # drop stale axes when panels come or go
        _adv_panel_count = (rows, cols, len(panels))
    Plotly.react(ADV_ID, _js(traces), _js(layout), _js(CONFIG))

def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") or "0"

def _update_collapse_note(xi_names: list[str]):
    """With no activation anywhere, stacked layers collapse into one line per
    output -- show that line. With an activation they cannot be merged."""
    note = state.get_id("collapse-note")
    if len(state.layers) < 2:
        note.innerHTML = ""
        return
    if any(l["act_fn"] for l in state.layers):
        note.textContent = "Because of the activation, these layers can't be merged into a single line."
        return
    n = len(state.inputs)
    zero = network_model.evaluate([0.0] * n)
    lines = []
    for j, out in enumerate(state.outputs):
        if j >= len(zero):
            break
        terms = ""
        for i in range(n):
            e = [0.0] * n
            e[i] = 1.0
            m = network_model.evaluate(e)[j] - zero[j]
            if abs(m) > 1e-9:
                terms += f" {'-' if m < 0 else '+'} {_fmt(abs(m))}·{xi_names[i]}" if terms else f"{_fmt(m)}·{xi_names[i]}"
        b = zero[j]
        if abs(b) > 1e-9 or not terms:
            terms += f" {'-' if b < 0 else '+'} {_fmt(abs(b))}" if terms else _fmt(b)
        lines.append(f"<b>{_axis_label('out', out['id'], f'output {j + 1}')} = {terms}</b>")
    note.innerHTML = ("With no activation, your layers collapse into one line per output: "
                      + "; ".join(lines) + ". An activation lets the lines bend so they can't be merged.")

def _plot_hint_text() -> str:
    """What the greyed-out plots should say: connect two devices first, then pick the input/output, then Run."""
    if len(state.devices) < 2:   # a sensor and a motor
        return "Connect 2 devices" if not state.devices else "Connect 1 more device"
    missing = network_model.unchosen_io()
    if missing:
        ins = any(m.startswith("input") for m in missing)
        outs = any(m.startswith("output") for m in missing)
        return "Select input and output" if ins and outs else ("Select input" if ins else "Select output")
    return "Hit Run"

def update_plot_hints():
    text = _plot_hint_text()
    for el in document.querySelectorAll(".transfer-hint"):
        if el.textContent != text:
            el.textContent = text

async def run_loop():
    while True:
        try:
            update_plot_hints()
            if state.in_training_family():
                import training_plot   # imported late: it imports this module for its label helper
                training_plot.redraw()
            elif state.mode == "beginner":
                redraw_beginner()
            else:
                redraw_advanced()
        except Exception as e:
            print("transfer plot: " + str(e))
        await asyncio.sleep(0.3)
