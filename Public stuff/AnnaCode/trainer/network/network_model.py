"""Topology lookups, the data-mutation half of CRUD, forward propagation,
and dataset-based training. Imports state and activations only -- never
imports any rendering/DOM file.

Exception: compute_forward()/forward() read DOM elements directly (device/
channel select values) and forward() also writes DOM text and calls
Device.run_output()/plot updates. This is a pragmatic exception (see the
implementation plan, Phase 3 step 2) rather than an oversight -- splitting
them further would fragment one already-small function across two files
for no real gain.
"""
import math
import random

import state
from state import get_id
from activations import apply_activation, apply_activation_derivative
import Device

# ── Topology lookups ─────────────────────────────────────────────────────────

def input_by_id(iid: int) -> dict | None:
    return next((i for i in state.inputs if i["id"] == iid), None)

def output_by_id(oid: int) -> dict | None:
    return next((o for o in state.outputs if o["id"] == oid), None)

def layer_by_id(lid: int) -> dict | None:
    return next((l for l in state.layers if l["id"] == lid), None)

def layer_index_by_id(lid: int) -> int | None:
    for i, l in enumerate(state.layers):
        if l["id"] == lid:
            return i
    return None

def neuron_by_id(nid: int) -> dict | None:
    for l in state.layers:
        for n in l["neurons"]:
            if n["id"] == nid:
                return n
    return None

def get_source_count(layer_idx: int) -> int:
    """How many weight slots a neuron in this layer needs."""
    if layer_idx == 0:
        return len(state.inputs)
    return len(state.layers[layer_idx - 1]["neurons"])

def get_source_labels(layer_idx: int) -> list[str]:
    """Display labels for each weight term in this layer's equations."""
    if layer_idx == 0:
        return [inp["name"] for inp in state.inputs]
    return [f"n{idx + 1}" for idx in range(len(state.layers[layer_idx - 1]["neurons"]))]

# ── Forward propagation ──────────────────────────────────────────────────────

def propagate_layers(source_values: list[float], pre_store: dict, post_store: dict) -> tuple[list[float], list[list[float]]]:
    """Runs source_values through every layer in order, writing each neuron's
    pre/post-activation value into pre_store/post_store (caller passes the
    live global dicts for hardware mode, or fresh local dicts for a
    dataset-training forward pass so the two don't clobber each other).

    Returns (final_layer_outputs, layer_sources) where layer_sources[i] is
    the list of source values that fed layers[i]."""
    layer_sources: list[list[float]] = []
    for layer in state.layers:
        layer_sources.append(source_values)
        act_fn = layer["act_fn"]
        custom_activation = layer["custom_activation"]
        next_source_values = []
        for n in layer["neurons"]:
            nid = n["id"]
            weighted = sum(
                n["weights"][i] * source_values[i]
                for i in range(min(len(n["weights"]), len(source_values)))
            )
            pre_act = weighted + n["bias"]
            pre_store[nid] = pre_act

            post_act = apply_activation(pre_act, act_fn, custom_activation)
            post_store[nid] = post_act
            next_source_values.append(post_act)
        source_values = next_source_values
    return source_values, layer_sources

def compute_forward() -> list[list[float]]:
    """Read the current input device readings and forward-propagate them
    through every layer, populating input_values / neuron_pre_values /
    neuron_post_values / output_values as a side effect (used by the arrow
    debug display too). Does NOT touch the hardware outputs -- that's
    forward()'s job.

    Returns layer_sources: layer_sources[i] is the list of source values
    that fed layers[i] (raw inputs for layer 0, the previous layer's
    post-activations otherwise) -- backpropagate() needs this to compute
    weight gradients."""
    # 1. Collect input values
    for inp in state.inputs:
        iid = inp["id"]
        dev_el  = get_id(f"dev-input-{iid}")
        chan_el = get_id(f"chan-input-{iid}")
        dev_name = dev_el.value if dev_el else ""
        channel  = chan_el.value if chan_el else ""
        dev = Device.device_by_name(dev_name)
        try:
            val = float(dev.state[channel]) if dev and channel else 0.0
        except (KeyError, TypeError, ValueError):
            val = 0.0
        state.input_values[iid] = val

    # 2. Forward-propagate through every layer in order
    source_values = [state.input_values.get(inp["id"], 0.0) for inp in state.inputs]
    _, layer_sources = propagate_layers(source_values, state.neuron_pre_values, state.neuron_post_values)

    # 3. Route each output from the last layer's neuron at the same position
    last_layer_neurons = state.layers[-1]["neurons"] if state.layers else []
    for idx, out in enumerate(state.outputs):
        oid = out["id"]
        val = state.neuron_post_values.get(last_layer_neurons[idx]["id"], 0.0) if idx < len(last_layer_neurons) else 0.0
        state.output_values[oid] = val

    return layer_sources

def forward():
    compute_forward()
    state.run_tick += 1
    for idx, out in enumerate(state.outputs):
        oid = out["id"]
        y_val = state.output_values.get(oid, 0.0)
        result = int(y_val)
        dev_el  = get_id(f"dev-output-{oid}")
        chan_el = get_id(f"chan-output-{oid}")
        dev_name = dev_el.value if dev_el else ""
        channel  = chan_el.value if chan_el else ""
        Device.run_output(channel, dev_name, result)

    fit_plot_obj = state.all_plots.get("plot-fit")
    if fit_plot_obj and state.inputs and state.outputs:
        x_val = state.input_values.get(state.inputs[0]["id"], 0.0)
        for out in state.outputs:
            fit_plot_obj.add_run_point(out["id"], x_val, state.output_values.get(out["id"], 0.0))

def randomize_weights(evt=None):
    """Give every neuron in every layer a fresh small uniform(-1, 1) weight
    vector + bias, as if it were already fit in normalized training space,
    then fold that into raw storage space via _unnormalize_weights() (the
    same fold applied after every real training run) so a subsequent
    train() call's _renormalize_weights() exactly inverts it back to these
    small values.

    Without this fold, fresh random noise would sit directly in raw space:
    _renormalize_weights() would then multiply the first layer by the
    input's raw std (inflating the weight enough to immediately saturate
    tanh/sigmoid) and subtract the output's raw mean from the last layer's
    bias (producing a huge normalized-space offset whenever the output's
    mean is large relative to its std -- common for real-world data --
    which then takes hundreds of gradient-clipped steps to walk back)."""
    import sync
    import arrows
    for layer in state.layers:
        for n in layer["neurons"]:
            n["weights"] = [random.uniform(-1.0, 1.0) for _ in n["weights"]]
            n["bias"] = random.uniform(-1.0, 1.0)
    if state.normalize_enabled:
        x_stats, y_stats = _dataset_stats()
        _unnormalize_weights(x_stats, y_stats)
    for idx in range(len(state.layers)):
        sync.rebuild_layer_eq_html(idx)
    arrows.redraw_arrows()

    state.loss_history = []
    loss_plot_obj = state.all_plots.get("plot-loss")
    if loss_plot_obj:
        loss_plot_obj.reset()

# ── Dataset-based training (normalized internally; live mode untouched) ─────

def _col_stats(vals: list[float]) -> tuple[float, float]:
    """(mean, std) over a single dataset column, std falling back to 1.0
    whenever there's fewer than 2 points or zero spread."""
    if len(vals) < 2:
        return 0.0, 1.0
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) or 1.0
    return mean, std

def _dataset_stats():
    """(x_stats, y_stats): x_stats maps each CURRENT input id -> (mean, std)
    over that column's values across training_data (missing values in a
    point default to 0.0); y_stats does the same per output id."""
    x_stats = {inp["id"]: _col_stats([p["xs"].get(inp["id"], 0.0) for p in state.training_data])
               for inp in state.inputs}
    y_stats = {out["id"]: _col_stats([p["ys"].get(out["id"], 0.0) for p in state.training_data])
               for out in state.outputs}
    return x_stats, y_stats

def normalize_x(x, iid, x_stats):
    x_mean, x_std = x_stats.get(iid, (0.0, 1.0))
    return (x - x_mean) / x_std if state.normalize_enabled else x

def normalize_y(y, oid, y_stats):
    y_mean, y_std = y_stats.get(oid, (0.0, 1.0))
    return (y - y_mean) / y_std if state.normalize_enabled else y

MAX_GRAD_NORM = 5.0

def _clip_grad(value: float) -> float:
    if value > MAX_GRAD_NORM:
        return MAX_GRAD_NORM
    if value < -MAX_GRAD_NORM:
        return -MAX_GRAD_NORM
    return value

def _fold_input_scale(neurons: list[dict], input_ids: list[int], x_stats, reverse: bool):
    """Fold the dataset's per-input x-scale into a layer's weights/bias.
    reverse=False divides by std (raw -> normalized-space-trained weights
    becoming usable on raw inputs, i.e. the "unnormalize" direction);
    reverse=True multiplies by std (the inverse, "renormalize" direction).
    Exact for any activation function -- it only rescales the pre-activation
    input."""
    for n in neurons:
        old_weights = list(n["weights"])
        bias_adjust = 0.0
        new_weights = []
        for i, w in enumerate(old_weights):
            x_mean, x_std = x_stats.get(input_ids[i], (0.0, 1.0)) if i < len(input_ids) else (0.0, 1.0)
            if reverse:
                new_weights.append(w * x_std)
                bias_adjust += w * x_mean
            else:
                new_weights.append(w / x_std)
                bias_adjust += w * x_mean / x_std
        n["bias"] = n["bias"] + bias_adjust if reverse else n["bias"] - bias_adjust
        n["weights"] = new_weights

def _unnormalize_weights(x_stats, y_stats):
    """Fold the dataset's per-input/per-output normalization into the
    network's stored weights/biases, converting them from the normalized
    space used DURING training (see train()) into the raw space that
    live/hardware forward() reads directly and unmodified. Without this, a
    network trained on normalized data would predict wildly wrong raw
    values in Play mode -- the whole point of training in normalized
    space is gradient stability, not a permanent change of the model's
    units.

    Folding each input's x-scale into the first layer's matching weight is
    exact for any activation function (it only rescales the pre-activation
    input). Folding an output's y-scale into the matching last-layer
    neuron's weights is only exact when the last layer's activation is
    identity ("None") -- for a nonlinear final activation there is no
    weight-only equivalent, so that step is skipped for that neuron and
    live-mode output stays in normalized-ish units in that edge case.
    A last-layer neuron beyond the current number of outputs has no
    matching output stats and is left untouched (it isn't wired to
    anything yet, per update_neuron_usage_state())."""
    if not state.layers:
        return
    input_ids = [inp["id"] for inp in state.inputs]

    _fold_input_scale(state.layers[0]["neurons"], input_ids, x_stats, reverse=False)

    last_layer = state.layers[-1]
    if last_layer["act_fn"] in ("", None):
        output_ids = [out["id"] for out in state.outputs]
        for idx, n in enumerate(last_layer["neurons"]):
            if idx >= len(output_ids):
                continue
            y_mean, y_std = y_stats.get(output_ids[idx], (0.0, 1.0))
            n["bias"] = y_mean + y_std * n["bias"]
            n["weights"] = [w * y_std for w in n["weights"]]

def _renormalize_weights(x_stats, y_stats):
    """Inverse of _unnormalize_weights(): converts the canonical raw-space
    weights/biases into the normalized-space equivalents train_epoch()
    expects, immediately before a training run. Must undo the two folds
    in reverse order (last layer's y-fold, then the first layer's
    x-fold) to exactly invert _unnormalize_weights()."""
    if not state.layers:
        return

    last_layer = state.layers[-1]
    if last_layer["act_fn"] in ("", None):
        output_ids = [out["id"] for out in state.outputs]
        for idx, n in enumerate(last_layer["neurons"]):
            if idx >= len(output_ids):
                continue
            y_mean, y_std = y_stats.get(output_ids[idx], (0.0, 1.0))
            old_weights = list(n["weights"])
            n["bias"] = (n["bias"] - y_mean) / y_std
            n["weights"] = [w / y_std for w in old_weights]

    input_ids = [inp["id"] for inp in state.inputs]
    _fold_input_scale(state.layers[0]["neurons"], input_ids, x_stats, reverse=True)

def train_epoch(lr: float, x_stats, y_stats) -> float:
    """One pass over the whole dataset: accumulate gradients per point (and
    per output, positionally paired with the last layer's neurons -- same
    convention as compute_forward()), then apply one averaged update per
    weight/bias. Returns the mean per-point-per-output loss. Assumes the
    network's weights are already in normalized space (see train())."""
    if not state.training_data or not state.layers or not state.outputs:
        return 0.0

    input_ids = [inp["id"] for inp in state.inputs]
    output_ids = [out["id"] for out in state.outputs]
    act_fns = [layer["act_fn"] for layer in state.layers]
    customs = [layer["custom_activation"] for layer in state.layers]

    weight_grads = [[[0.0] * len(n["weights"]) for n in layer["neurons"]] for layer in state.layers]
    bias_grads = [[0.0 for _ in layer["neurons"]] for layer in state.layers]

    last_layer_neurons = state.layers[-1]["neurons"]
    total_loss = 0.0
    loss_terms = 0

    for point in state.training_data:
        xs_n = [normalize_x(point["xs"].get(iid, 0.0), iid, x_stats) for iid in input_ids]

        pre_store, post_store = {}, {}
        final_values, layer_sources = propagate_layers(xs_n, pre_store, post_store)

        deltas = [0.0] * len(last_layer_neurons)
        for oidx, oid in enumerate(output_ids):
            if oidx >= len(last_layer_neurons):
                break
            n = last_layer_neurons[oidx]
            pred_n = final_values[oidx] if oidx < len(final_values) else 0.0
            y_n = normalize_y(point["ys"].get(oid, 0.0), oid, y_stats)

            err = pred_n - y_n
            total_loss += err ** 2
            loss_terms += 1

            deriv = apply_activation_derivative(
                pre_store.get(n["id"], 0.0), pred_n, act_fns[-1], customs[-1])
            deltas[oidx] = 2.0 * err * deriv

        for layer_idx in reversed(range(len(state.layers))):
            layer = state.layers[layer_idx]
            neurons = layer["neurons"]
            sources = layer_sources[layer_idx] if layer_idx < len(layer_sources) else []
            next_deltas = [0.0] * len(sources)

            for i, n in enumerate(neurons):
                d = deltas[i] if i < len(deltas) else 0.0
                if d == 0.0:
                    continue
                for j in range(min(len(n["weights"]), len(sources))):
                    next_deltas[j] += d * n["weights"][j]
                    weight_grads[layer_idx][i][j] += d * sources[j]
                bias_grads[layer_idx][i] += d

            if layer_idx > 0:
                prev_layer = state.layers[layer_idx - 1]
                new_deltas = []
                for i, pn in enumerate(prev_layer["neurons"]):
                    pre = pre_store.get(pn["id"], 0.0)
                    post = post_store.get(pn["id"], 0.0)
                    deriv = apply_activation_derivative(pre, post, act_fns[layer_idx - 1], customs[layer_idx - 1])
                    raw = next_deltas[i] if i < len(next_deltas) else 0.0
                    new_deltas.append(raw * deriv)
                deltas = new_deltas

    n_points = len(state.training_data)
    for layer_idx, layer in enumerate(state.layers):
        for i, n in enumerate(layer["neurons"]):
            for j in range(len(n["weights"])):
                grad = _clip_grad(weight_grads[layer_idx][i][j] / n_points)
                n["weights"][j] -= lr * grad
            bgrad = _clip_grad(bias_grads[layer_idx][i] / n_points)
            n["bias"] -= lr * bgrad

    return total_loss / loss_terms if loss_terms else 0.0

def _append_loss(loss: float):
    state.loss_history.append(loss)
    loss_plot_obj = state.all_plots.get("plot-loss")
    if loss_plot_obj:
        update = loss_plot_obj.add_point(loss)
        loss_plot_obj.updatePlot(update)

def train(epochs: int, lr: float):
    import sync
    if not state.training_data:
        print("Add at least one data point before training.")
        return

    x_stats, y_stats = _dataset_stats()
    if state.normalize_enabled:
        _renormalize_weights(x_stats, y_stats)

    for _ in range(epochs):
        loss = train_epoch(lr, x_stats, y_stats)
        _append_loss(loss)

    if state.normalize_enabled:
        _unnormalize_weights(x_stats, y_stats)

    for idx in range(len(state.layers)):
        sync.rebuild_layer_eq_html(idx)
    print(f"Trained {epochs} epoch(s), lr={lr}. Loss: {state.loss_history[-1] if state.loss_history else 0.0:.4f}")

def train_step(lr: float):
    train(epochs=1, lr=lr)

def train_50_epochs(lr: float):
    train(epochs=50, lr=lr)

def add_data_point(xs: dict, ys: dict):
    state.data_point_counter += 1
    state.training_data.append({"id": state.data_point_counter, "xs": dict(xs), "ys": dict(ys)})

def remove_data_point(pid: int):
    idx = next((i for i, p in enumerate(state.training_data) if p["id"] == pid), None)
    if idx is None:
        return
    state.training_data.pop(idx)
