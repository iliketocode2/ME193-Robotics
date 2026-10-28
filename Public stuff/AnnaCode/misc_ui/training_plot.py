"""
AnnaCode — training_plot.py

Training-mode plots: the recorded samples as dots (one panel per output, input on
the x axis) with the network's current response as a green line, plus the
loss curve. Click a dot to remove it.
"""
import json
import js
from pyscript import document
from pyscript.ffi import create_proxy
from pyscript.js_modules import Plotly

import state
import arrows
import network_model
import transfer_plot
import training

DATA_ID = "train-data-plot"
LOSS_ID = "train-loss-plot"
LINE_COLOR = transfer_plot.LINE_COLOR
DOT_COLOR = "#0891b2"
PANEL_HEIGHT = 260
CONFIG = {"displayModeBar": False, "responsive": True}
STATIC = {"displayModeBar": False, "staticPlot": True, "responsive": True}

_panel_count = 0
_last_sig = None
_last_ids = None
_click_attached = False
_last_loss_len = -1
_click_proxy = None
_select_proxy = None
ORDER_SCALE = [[0, "#a5f3fc"], [1, "#164e63"]]   # light = recorded first, dark = recorded last

def _js(obj):
    return js.JSON.parse(json.dumps(obj))

def _on_point_click(pid):
    if state.mode != "fitting":   # the example curve in Fitting is a fixed target
        training.remove_point(int(pid))

def _on_selected(ids_json):
    if state.mode != "fitting":
        training.remove_points(json.loads(ids_json))

def _redraw_data():
    global _panel_count, _click_attached, _click_proxy, _select_proxy, _last_sig, _last_ids
    el = state.get_id(DATA_ID)
    if not el or not state.inputs or not state.outputs or not state.layers:
        return
    last_neurons = state.layers[-1]["neurons"]
    outs = [(k, o) for k, o in enumerate(state.outputs) if k < len(last_neurons)]
    if not outs:
        return

    inp = state.inputs[0]
    data = state.training_data
    ids = [p["id"] for p in data]
    xs_data = [p["xs"].get(inp["id"], 0.0) for p in data]
    lo, hi = (min(xs_data), max(xs_data)) if xs_data else (-100.0, 100.0)
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    sweep = [lo + (hi - lo) * k / 40 for k in range(41)]

    # Other inputs are held at their average recorded value (or live value if nothing is recorded)
    others = []
    for i in state.inputs[1:]:
        vals = [p["xs"].get(i["id"], 0.0) for p in data]
        others.append(sum(vals) / len(vals) if vals else state.input_values.get(i["id"], 0.0))

    # Live reading (where the robot is right now), drawn as a ring
    toggle = state.get_id("live-dot-toggle")
    live_on = bool(toggle and toggle.checked) and state.mode != "fitting"   # Fitting has no sensors
    x_live = training._read("in", inp["id"]) if live_on else None
    ys_live = {o["id"]: training._read("out", o["id"]) for _, o in outs} if live_on else {}

    traces = []
    layout = {
        "margin": {"l": 70, "r": 20, "t": 10, "b": 45},
        "grid": {"rows": len(outs), "columns": 1, "pattern": "independent", "roworder": "top to bottom", "ygap": 0.18},
        "showlegend": False,
        "dragmode": False if state.mode == "fitting" else "select",
        "hovermode": "closest",
        "hoverdistance": 25,
        "paper_bgcolor": "rgba(0,0,0,0)",
    }
    hover = {"hoverinfo": "skip"} if state.mode == "fitting" else {"hovertemplate": "click to remove<extra></extra>"}
    for panel, (k, out) in enumerate(outs, start=1):
        sfx = "" if panel == 1 else str(panel)
        label = "y" if state.mode == "fitting" else transfer_plot._axis_label("out", out["id"], f"output {k + 1}")
        layout[f"xaxis{sfx}"] = {"zeroline": True}
        layout[f"yaxis{sfx}"] = {"title": {"text": label, "font": {"size": 12}}, "zeroline": True}
        axes = {"xaxis": f"x{sfx}", "yaxis": f"y{sfx}"}
        traces.append({**axes, "x": xs_data, "y": [p["ys"].get(out["id"], 0.0) for p in data],
                       "customdata": ids, "mode": "markers", **hover,
                       "marker": {"color": list(range(len(data))), "colorscale": ORDER_SCALE, "cmin": 0,
                                  "cmax": max(1, len(data) - 1), "size": 8, "opacity": 0.85}})
        if live_on and x_live is not None and ys_live.get(out["id"]) is not None:
            traces.append({**axes, "x": [x_live], "y": [ys_live[out["id"]]], "mode": "markers", "hoverinfo": "skip",
                           "marker": {"symbol": "circle-open", "size": 18, "color": "#1a1d2e", "line": {"width": 3}}})
        fit = [network_model.evaluate_layers([x] + others)[-1][k][1] for x in sweep]
        traces.append({**axes, "x": sweep, "y": fit, "mode": "lines", "hoverinfo": "skip",
                       "line": {"color": LINE_COLOR, "width": 3}})
    layout[f"xaxis{len(outs) if len(outs) > 1 else ''}"]["title"] = {
        "text": "x" if state.mode == "fitting" else transfer_plot._axis_label("in", inp["id"], inp["name"]),
        "font": {"size": 13}}

    # Redrawing wipes the hover tag, which made clicks land on nothing. Skip the redraw when nothing changed, and
    # while the pointer is on the plot unless the points themselves changed (e.g. one was just removed).
    traces_json, layout_json = json.dumps(traces), json.dumps(layout)
    sig = traces_json + layout_json
    if sig == _last_sig:
        return
    if Plotly.pointerInside(DATA_ID) and tuple(ids) == _last_ids and len(outs) == _panel_count:
        return
    _last_sig, _last_ids = sig, tuple(ids)

    el.style.height = f"{len(outs) * PANEL_HEIGHT + 60}px"
    if len(outs) != _panel_count:
        Plotly.purge(DATA_ID)   # drop stale axes (this also drops the click hook)
        _panel_count = len(outs)
        _click_attached = False
    Plotly.react(DATA_ID, js.JSON.parse(traces_json), js.JSON.parse(layout_json), _js(CONFIG))
    if not _click_attached:
        _click_proxy = create_proxy(_on_point_click)
        _select_proxy = create_proxy(_on_selected)
        Plotly.onPointClick(DATA_ID, _click_proxy)
        Plotly.onSelect(DATA_ID, _select_proxy)
        _click_attached = True

_mini_sig = None

def _redraw_mini():
    """Fitting: the network's own curve (y against x) drawn inside the output box."""
    global _mini_sig
    card = state.get_id("fit-mini-plot")
    if not card:
        host = document.querySelector(".output-node")
        if not host:
            return
        card = document.createElement("div")
        card.id = "fit-mini-plot"
        host.appendChild(card)
    if state.mode != "fitting" or not state.inputs or not state.layers:
        return
    xs = [p["xs"].get(state.inputs[0]["id"], 0.0) for p in state.training_data]
    lo, hi = (min(xs), max(xs)) if xs else (-100.0, 100.0)
    pad = (hi - lo) * 0.1 or 1.0
    sweep = [lo - pad + (hi - lo + 2 * pad) * k / 40 for k in range(41)]
    others = [0.0] * (len(state.inputs) - 1)
    ys = [network_model.evaluate([x] + others)[0] for x in sweep]
    traces = [{"x": sweep, "y": ys, "mode": "lines", "hoverinfo": "skip", "line": {"color": LINE_COLOR, "width": 3}}]
    layout = {"margin": {"l": 28, "r": 6, "t": 6, "b": 18}, "showlegend": False, "paper_bgcolor": "rgba(0,0,0,0)",
              "plot_bgcolor": "#ffffff", "font": {"size": 9},
              "xaxis": {"zeroline": True, "nticks": 4}, "yaxis": {"zeroline": True, "nticks": 4}}
    sig = json.dumps([traces, layout])
    if sig == _mini_sig:
        return
    _mini_sig = sig
    Plotly.react("fit-mini-plot", _js(traces), _js(layout), _js(STATIC))
    arrows.schedule_redraw()

def _redraw_loss():
    global _last_loss_len
    if not state.get_id(LOSS_ID) or len(state.loss_history) == _last_loss_len:
        return
    _last_loss_len = len(state.loss_history)
    traces = [{"y": list(state.loss_history), "mode": "lines", "line": {"color": "#d97706", "width": 3}}]
    layout = {
        "margin": {"l": 55, "r": 20, "t": 10, "b": 45},
        "xaxis": {"title": {"text": "epoch", "font": {"size": 12}}},
        "yaxis": {"title": {"text": "loss", "font": {"size": 12}}, "rangemode": "tozero"},
        "showlegend": False,
        "paper_bgcolor": "rgba(0,0,0,0)",
    }
    Plotly.react(LOSS_ID, _js(traces), _js(layout), _js(STATIC))

def _debug_preview():
    """Debug arrows show live values even when the network isn't Running (Run would drive the motors)."""
    if state.mode != "fitting" and state.debug_mode and not state.is_running and state.inputs and state.layers and state.outputs:
        network_model.preview_forward()
        arrows.redraw_arrows()

def _fitting_error_text() -> str:
    """How far the network's line is from the example dots: the typical miss, in the output's own units."""
    data = state.training_data
    if not data or not state.inputs or not state.outputs or not state.layers:
        return "0 points"
    inp, out = state.inputs[0], state.outputs[0]
    total = 0.0
    for p in data:
        vals = [p["xs"].get(i["id"], 0.0) for i in state.inputs]
        total += (network_model.evaluate(vals)[0] - p["ys"].get(out["id"], 0.0)) ** 2
    return f"{len(data)} points · average miss {(total / len(data)) ** 0.5:.1f}"

def _keep_output_linear():
    """Fitting hides the last activation block, so make sure nothing hidden is bending the output."""
    if state.mode == "fitting" and state.layers and state.layers[-1]["act_fn"]:
        last = state.layers[-1]
        last["act_fn"] = ""
        sel = state.get_id(f"act-select-{last['id']}")
        if sel:
            sel.value = ""

def redraw():
    _keep_output_linear()
    _debug_preview()
    count = state.get_id("train-count")
    if count:
        count.textContent = _fitting_error_text() if state.mode == "fitting" else f"{len(state.training_data)} points"
    undo_btn = state.get_id("undo-btn")
    if undo_btn:
        undo_btn.disabled = not state.undo_stack
    # Either Record or an example curve, not both: each greys out the other while there is data
    if not state.training_data:
        state.example_key = None
    sel = state.get_id("example-select")
    if sel:
        sel.disabled = state.mode != "fitting" and (bool(state.training_data) or state.is_recording)
        if state.example_key is None:
            sel.value = ""
    rec = state.get_id("record-btn")
    if rec:
        rec.disabled = state.example_key is not None and not state.is_recording
    hint = state.get_id("train-hint")
    if hint:
        shown = state.example_key is not None and not state.train_hint_closed
        hint.classList.toggle("hidden", not shown)
        if shown:
            state.get_id("train-hint-text").textContent = training.EXAMPLES[state.example_key]["hint"]
    _redraw_data()
    _redraw_mini()
    _redraw_loss()
