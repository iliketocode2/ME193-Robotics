"""
AnnaCode — training.py

Training mode: record (input, output) samples from the hardware while the
student pushes the robot around, clean the data up, then fit the network to it
with backprop. Activations are honoured (shown or hidden by the Training toggle), so students can see what
they change. Training happens in normalized space (stable gradients); the weights are folded
back into raw units after every epoch, so the displayed numbers, the fit line and Run all see
ordinary weights and biases. If the last layer has an activation, the targets are left in raw
units (a y-scale cannot be folded through a non-linear output).
"""
import asyncio
import math
import re
import time

import random

import js

import state
import Device
import sync
import activations
import network_model

SAMPLE_PERIOD = 0.1   # seconds between recorded samples
EPOCH_PAUSE = 0.03    # lets the plots repaint between epochs
MAX_GRAD_NORM = 5.0

def set_status(text: str):
    el = state.get_id("train-status")
    if el:
        el.textContent = text

# ── Recording ───────────────────────────────────────────────────────────────

def _read(kind: str, item_id: int):
    """Live value at the device/channel chosen for this input ('in') or output ('out'); None if unavailable."""
    dev_el = state.get_id(f"dev-{kind}-{item_id}")
    chan_el = state.get_id(f"chan-{kind}-{item_id}")
    dev = Device.device_by_name(dev_el.value if dev_el else "")
    channel = chan_el.value if chan_el else ""
    try:
        return float(dev.read(channel)) if dev and channel else None
    except (KeyError, TypeError, ValueError):
        return None

def add_point(xs: dict, ys: dict, t: float | None = None):
    """t is the recording time in seconds (any origin); points without one (examples) are timed by their order."""
    state.data_point_counter += 1
    p = {"id": state.data_point_counter, "xs": dict(xs), "ys": dict(ys)}
    if t is not None:
        p["t"] = t
    state.training_data.append(p)

def _times() -> list[float]:
    """Each point's time in seconds, starting at 0 for the earliest point."""
    raw = [p.get("t", k * SAMPLE_PERIOD) for k, p in enumerate(state.training_data)]
    t0 = min(raw, default=0.0)
    return [t - t0 for t in raw]

MAX_UNDO = 100

def _remove_where(should_remove) -> int:
    """Remove every point the predicate matches, remembering them as one undo step."""
    removed = [p for p in state.training_data if should_remove(p)]
    if removed:
        state.training_data[:] = [p for p in state.training_data if not should_remove(p)]
        state.undo_stack.append(removed)
        del state.undo_stack[:-MAX_UNDO]
    return len(removed)

def remove_point(pid: int):
    _remove_where(lambda p: p["id"] == pid)

def remove_points(pids):
    ids = set(pids)
    n = _remove_where(lambda p: p["id"] in ids)
    if n:
        set_status(f"Removed {n} point{'s' if n != 1 else ''}. Undo brings them back.")

def clear_data(evt=None):
    _remove_where(lambda p: True)
    state.loss_history.clear()

def remove_zero_points(evt=None):
    """Drop samples where every output reads zero -- the robot had paused."""
    oids = [o["id"] for o in state.outputs]
    n = _remove_where(lambda p: not any(abs(p["ys"].get(oid, 0.0)) > 1e-9 for oid in oids))
    set_status(f"Removed {n} zero-speed points.")

def undo(evt=None):
    if not state.undo_stack:
        return
    batch = state.undo_stack.pop()
    state.training_data.extend(batch)
    state.training_data.sort(key=lambda p: p["id"])   # ids grow with recording order
    set_status(f"Restored {len(batch)} point{'s' if len(batch) != 1 else ''}.")

# ── Example data ────────────────────────────────────────────────────────────

EXAMPLE_POINTS = 120
X_MIN, X_MAX = 20.0, 100.0   # like a reflection reading

def _frac(v: float) -> float:
    return v - math.floor(v)

# key -> label, curve f(x), noise, and the hint on how to shape the network
EXAMPLES = {
    "linear": {
        "label": "Linear",
        "f": lambda x: 2 * x - 100, "noise": 3.0,
        "hint": "A straight line needs no hidden layer and no activation: one neuron (weight about 2, bias about -100) is enough.",
    },
    "v": {
        "label": "V",
        "f": lambda x: 2 * abs(x - 60), "noise": 2.0,
        "hint": "A V is two ramps: add a hidden layer with at least 2 neurons and ReLU, and leave the output layer's activation as None. "
                "Then Randomize weights and press Train a few times.",
    },
    "parabola": {
        "label": "Parabola",
        "f": lambda x: 0.04 * (x - 60) ** 2, "noise": 2.0,
        "hint": "A smooth curve: try a hidden layer of 3-4 neurons with ReLU (it builds the curve from straight pieces), "
                "or Tanh or Softplus for a smoother fit. Keep the output layer's activation as None.",
    },
    "step": {
        "label": "Step (on/off)",
        "f": lambda x: 0.0 if x < 60 else 80.0, "noise": 3.0,
        "hint": "A switch: a hidden layer of 2 neurons with Sigmoid (or Tanh) makes an S-shaped jump. The jump needs big weights, "
                "so raise the learning rate to about 0.3 and train a few hundred epochs. Keep the output layer's activation as None.",
    },
    "bump": {
        "label": "Bump",
        "f": lambda x: 80 * math.exp(-((x - 60) / 12) ** 2), "noise": 2.0,
        "hint": "A single hill: a hidden layer of 3 neurons with Tanh (or Sigmoid) can build it. "
                "Keep the output layer's activation as None.",
    },
    "sine": {
        "label": "Sine wave",
        "f": lambda x: 50 * math.sin(2 * math.pi * (x - 20) / 80), "noise": 2.0,
        "hint": "A smooth wave with a hill and a valley: use a hidden layer of 4 or more neurons with Tanh. More neurons fit it better. "
                "Keep the output layer's activation as None and train for a few hundred epochs.",
    },
    "sawtooth": {
        "label": "Sawtooth",
        "f": lambda x: 80 * _frac((x - 20) / 40) - 40, "noise": 2.0,
        "hint": "Ramps with sudden drops: use a hidden layer of about 10 ReLU neurons and keep pressing Train 50 epochs (it needs 1000 or more). "
                "The drops will stay slightly slanted, because a smooth network never makes them perfectly vertical.",
    },
}

def populate_examples(sel):
    sel.innerHTML = '<option value="">Example data…</option>' + "".join(
        f'<option value="{k}">{e["label"]}</option>' for k, e in EXAMPLES.items())

def add_example(key: str):
    """Fill the plot with a noisy example curve. Only while there is no data (it is Record or an example, not both)."""
    ex = EXAMPLES.get(key)
    if not ex or state.training_data or not state.inputs or not state.outputs:
        return
    for _ in range(EXAMPLE_POINTS):
        x = random.uniform(X_MIN, X_MAX)
        xs = {i["id"]: (x if k == 0 else random.uniform(X_MIN, X_MAX)) for k, i in enumerate(state.inputs)}
        ys = {o["id"]: ex["f"](x) + random.gauss(0, ex["noise"]) for o in state.outputs}
        add_point(xs, ys)
    state.example_key = key
    state.train_hint_closed = False
    set_status(f"Added {EXAMPLE_POINTS} example points: {ex['label']}.")

# ── Save / load ─────────────────────────────────────────────────────────────

def _label(kind: str, item: dict, fallback: str) -> str:
    import transfer_plot
    return transfer_plot._axis_label(kind, item["id"], fallback)

def dataset_to_tsv() -> str:
    """Tab-separated, one row per point: time (s) starting at 0, then the input column(s), then the output column(s),
    with a header of their names. Columns are positional: after time, the first ones belong to the inputs in order."""
    in_ids = [i["id"] for i in state.inputs]
    out_ids = [o["id"] for o in state.outputs]
    header = (["Time (s)"] + [_label("in", i, i["name"]) for i in state.inputs]
              + [_label("out", o, f"output {k + 1}") for k, o in enumerate(state.outputs)])
    lines = ["\t".join(h.replace("\t", " ") for h in header)]
    for t, p in zip(_times(), state.training_data):
        vals = [t] + [p["xs"].get(i, 0.0) for i in in_ids] + [p["ys"].get(o, 0.0) for o in out_ids]
        lines.append("\t".join(f"{v:.6g}" for v in vals))
    return "\n".join(lines) + "\n"

def _parse_tsv(text: str):
    """(n_columns, rows of floats) from a TSV with a header line, or None if it isn't one."""
    lines = [l for l in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if l.strip()]
    if len(lines) < 2:
        return None
    n_cols = len(lines[0].split("\t"))
    rows = []
    for l in lines[1:]:
        cells = [c.strip().strip('"') for c in l.split("\t")]
        try:
            row = [float(c) for c in cells[:n_cols]]
        except ValueError:
            return None
        if len(row) != n_cols:
            return None
        rows.append(row)
    return n_cols, rows

def load_dataset(text: str) -> int:
    """Append the rows of a saved TSV to the current data. Returns how many were added (0 on a mismatch)."""
    parsed = _parse_tsv(text)
    if parsed is None:
        set_status("That doesn't look like a saved AnnaCode TSV (a header line, then rows of numbers).")
        return 0
    n_cols, rows = parsed
    n_in, n_out = len(state.inputs), len(state.outputs)
    has_time = n_cols == n_in + n_out + 1      # a leading Time (s) column; files without one still load
    if n_cols != n_in + n_out and not has_time:
        set_status(f"That file has {n_cols} column(s); your network has {n_in} input(s) and {n_out} output(s), "
                   f"so it needs {n_in + n_out} (plus an optional leading Time column).")
        return 0
    # Loaded times continue after any points already there, so Save keeps one increasing time axis
    existing = [p["t"] for p in state.training_data if "t" in p]
    base = (max(existing) + SAMPLE_PERIOD) if existing else 0.0
    in_ids = [i["id"] for i in state.inputs]
    out_ids = [o["id"] for o in state.outputs]
    for r in rows:
        t = base + r[0] if has_time else None
        if has_time:
            r = r[1:]
        add_point(dict(zip(in_ids, r[:n_in])), dict(zip(out_ids, r[n_in:])), t)
    set_status(f"Loaded {len(rows)} points.")
    return len(rows)

def save_dataset(evt=None):
    if not state.training_data:
        set_status("Nothing to save yet.")
        return
    blob = js.Blob.new([dataset_to_tsv()], js.JSON.parse('{"type": "text/tab-separated-values"}'))
    url = js.URL.createObjectURL(blob)
    a = js.document.createElement("a")
    a.href = url
    ex = EXAMPLES.get(state.example_key)
    slug = re.sub(r"[^a-z0-9]+", "-", ex["label"].lower()).strip("-") if ex else ""
    a.download = f"annacode-{slug}.tsv" if slug else "annacode-data.tsv"
    a.click()
    js.URL.revokeObjectURL(url)
    set_status(f"Saved {len(state.training_data)} points (a TSV you can open in Excel).")

async def _record_loop():
    while state.is_recording:
        xs = {i["id"]: _read("in", i["id"]) for i in state.inputs}
        ys = {o["id"]: _read("out", o["id"]) for o in state.outputs}
        if None not in xs.values() and None not in ys.values():
            add_point(xs, ys, time.monotonic())
        await asyncio.sleep(SAMPLE_PERIOD)

def start_recording():
    if not network_model.alert_unchosen("record"):
        set_status("Pick a device and a value for every input and output first.")
        return
    state.is_recording = True
    btn = state.get_id("record-btn")
    btn.textContent = "Stop recording"
    btn.classList.add("recording")
    set_status("Recording… push the robot through its paces.")
    asyncio.ensure_future(_record_loop())

def stop_recording():
    state.is_recording = False
    btn = state.get_id("record-btn")
    if btn:
        btn.textContent = "Record"
        btn.classList.remove("recording")

def toggle_recording(evt=None):
    if state.is_recording:
        stop_recording()
        set_status(f"{len(state.training_data)} points recorded.")
    else:
        start_recording()

# ── Normalization (same scheme as the Network Trainer page) ─────────────────

def _col_stats(vals: list[float]):
    if len(vals) < 2:
        return 0.0, 1.0
    mean = sum(vals) / len(vals)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) or 1.0

def _dataset_stats():
    x_stats = {i["id"]: _col_stats([p["xs"].get(i["id"], 0.0) for p in state.training_data]) for i in state.inputs}
    y_stats = {o["id"]: _col_stats([p["ys"].get(o["id"], 0.0) for p in state.training_data]) for o in state.outputs}
    return x_stats, y_stats

def _fold_input_scale(neurons, input_ids, x_stats, reverse: bool):
    for n in neurons:
        bias_adjust, new_weights = 0.0, []
        for i, w in enumerate(n["weights"]):
            mean, std = x_stats.get(input_ids[i], (0.0, 1.0)) if i < len(input_ids) else (0.0, 1.0)
            if reverse:
                new_weights.append(w * std)
                bias_adjust += w * mean
            else:
                new_weights.append(w / std)
                bias_adjust += w * mean / std
        n["bias"] = n["bias"] + bias_adjust if reverse else n["bias"] - bias_adjust
        n["weights"] = new_weights

def _to_normalized(x_stats, y_stats):
    out_ids = [o["id"] for o in state.outputs]
    for idx, n in enumerate(state.layers[-1]["neurons"]):
        if idx < len(out_ids):
            mean, std = y_stats.get(out_ids[idx], (0.0, 1.0))
            n["bias"] = (n["bias"] - mean) / std
            n["weights"] = [w / std for w in n["weights"]]
    _fold_input_scale(state.layers[0]["neurons"], [i["id"] for i in state.inputs], x_stats, reverse=True)

def _to_raw(x_stats, y_stats):
    _fold_input_scale(state.layers[0]["neurons"], [i["id"] for i in state.inputs], x_stats, reverse=False)
    out_ids = [o["id"] for o in state.outputs]
    for idx, n in enumerate(state.layers[-1]["neurons"]):
        if idx < len(out_ids):
            mean, std = y_stats.get(out_ids[idx], (0.0, 1.0))
            n["bias"] = mean + std * n["bias"]
            n["weights"] = [w * std for w in n["weights"]]

# ── Backprop ────────────────────────────────────────────────────────────────

def _clip(v: float) -> float:
    return max(-MAX_GRAD_NORM, min(MAX_GRAD_NORM, v))

def _forward(xs: list[float]):
    """Forward pass keeping what backprop needs: (outputs, sources per layer, pre- and post-activations per layer)."""
    sources, pres, posts, vals = [], [], [], list(xs)
    for layer in state.layers:
        sources.append(vals)
        pre = [sum(n["weights"][i] * vals[i] for i in range(min(len(n["weights"]), len(vals)))) + n["bias"]
               for n in layer["neurons"]]
        vals = [activations.apply_activation(p, layer["act_fn"], layer["custom_activation"]) for p in pre]
        pres.append(pre)
        posts.append(vals)
    return vals, sources, pres, posts

def _deriv(li: int, i: int, pres, posts) -> float:
    layer = state.layers[li]
    return activations.apply_activation_derivative(pres[li][i], posts[li][i], layer["act_fn"], layer["custom_activation"])

def _train_epoch(lr: float, x_stats, y_stats) -> float:
    """One pass over the data, one averaged update per weight. Returns the mean squared error
    (in normalized units). Weights must already be in normalized space."""
    in_ids = [i["id"] for i in state.inputs]
    out_ids = [o["id"] for o in state.outputs]
    w_grads = [[[0.0] * len(n["weights"]) for n in l["neurons"]] for l in state.layers]
    b_grads = [[0.0] * len(l["neurons"]) for l in state.layers]
    total, terms = 0.0, 0

    for p in state.training_data:
        xs = [(p["xs"].get(iid, 0.0) - x_stats[iid][0]) / x_stats[iid][1] for iid in in_ids]
        final, sources, pres, posts = _forward(xs)
        deltas = [0.0] * len(final)
        for k, oid in enumerate(out_ids):
            if k >= len(final):
                break
            err = final[k] - (p["ys"].get(oid, 0.0) - y_stats[oid][0]) / y_stats[oid][1]
            total += err ** 2
            terms += 1
            deltas[k] = 2.0 * err * _deriv(len(state.layers) - 1, k, pres, posts)
        for li in reversed(range(len(state.layers))):
            neurons = state.layers[li]["neurons"]
            src = sources[li]
            back = [0.0] * len(src)
            for i, n in enumerate(neurons):
                d = deltas[i] if i < len(deltas) else 0.0
                for j in range(min(len(n["weights"]), len(src))):
                    w_grads[li][i][j] += d * src[j]
                    back[j] += d * n["weights"][j]
                b_grads[li][i] += d
            if li > 0:
                deltas = [back[i] * _deriv(li - 1, i, pres, posts) for i in range(len(back))]

    count = len(state.training_data)
    for li, layer in enumerate(state.layers):
        for i, n in enumerate(layer["neurons"]):
            n["weights"] = [w - lr * _clip(g / count) for w, g in zip(n["weights"], w_grads[li][i])]
            n["bias"] -= lr * _clip(b_grads[li][i] / count)
    return total / terms if terms else 0.0

def _stats_for_training():
    """Dataset stats, with the y-scale switched off when the last layer has an activation."""
    x_stats, y_stats = _dataset_stats()
    if state.layers and state.layers[-1]["act_fn"] not in ("", None):
        y_stats = {oid: (0.0, 1.0) for oid in y_stats}
    return x_stats, y_stats

def _rebuild_equations():
    for idx in range(len(state.layers)):
        sync.rebuild_layer_eq_html(idx)

def _randomize(layer_indices):
    """Small random weights (and zero bias) for the given layers, leaving the others' behaviour unchanged."""
    x_stats, y_stats = _stats_for_training()
    _to_normalized(x_stats, y_stats)
    for li in layer_indices:
        for n in state.layers[li]["neurons"]:
            fan_in = max(1, len(n["weights"]))
            n["weights"] = [random.uniform(-1.0, 1.0) / math.sqrt(fan_in) for _ in n["weights"]]
            n["bias"] = 0.0
    _to_raw(x_stats, y_stats)
    _rebuild_equations()

def randomize_weights(evt=None):
    if state.is_training or not state.layers:
        return
    _randomize(range(len(state.layers)))
    state.loss_history.clear()
    set_status("Weights randomized. Press Train to start from here.")

def _hidden_layers_symmetric() -> bool:
    """Identical neurons in a hidden layer get identical gradients, so they would never learn different things."""
    for layer in state.layers[:-1]:
        seen = set()
        for n in layer["neurons"]:
            key = (tuple(round(w, 9) for w in n["weights"]), round(n["bias"], 9))
            if key in seen:
                return True
            seen.add(key)
    return False

def get_learning_rate() -> float:
    el = state.get_id("train-lr")
    try:
        return float(el.value)
    except (TypeError, ValueError):
        return 0.05

async def train(epochs: int):
    if state.is_training:
        return
    if not state.training_data or not state.layers or not state.outputs:
        set_status("Record some data first.")
        return
    state.is_training = True
    lr = get_learning_rate()
    if _hidden_layers_symmetric():
        _randomize(range(len(state.layers) - 1))
        set_status("Randomized the hidden layer(s) so their neurons can learn different things. Training…")
    else:
        set_status("Training…")
    x_stats, y_stats = _stats_for_training()
    for _ in range(epochs):
        _to_normalized(x_stats, y_stats)
        loss = _train_epoch(lr, x_stats, y_stats)
        _to_raw(x_stats, y_stats)
        state.loss_history.append(loss)
        _rebuild_equations()
        await asyncio.sleep(EPOCH_PAUSE)
    state.is_training = False
    set_status(f"Trained {epochs} epoch{'s' if epochs != 1 else ''}. Loss {state.loss_history[-1]:.3f}")
