"""
AnnaCode — network_model.py

Topology lookups and the forward pass. Imports state and activations only --
never a rendering/DOM file (templates, arrows, sync, bindings, etc).
"""
import state
import activations
import Device

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
    return [n.get("name") or f"n{k + 1}" for k, n in enumerate(state.layers[layer_idx - 1]["neurons"])]

def evaluate_layers(input_vals: list[float]) -> list[list[tuple[float, float]]]:
    """Pure forward pass (no state/device side effects): for every layer, a
    (pre_activation, post_activation) pair per neuron."""
    source_values = list(input_vals)
    result = []
    for layer in state.layers:
        layer_vals = []
        for n in layer["neurons"]:
            weighted = sum(
                n["weights"][i] * source_values[i]
                for i in range(min(len(n["weights"]), len(source_values)))
            )
            pre_act = weighted + n["bias"]
            layer_vals.append((pre_act, activations.apply_activation(
                pre_act, layer["act_fn"], layer["custom_activation"])))
        result.append(layer_vals)
        source_values = [post for _, post in layer_vals]
    return result

def evaluate(input_vals: list[float]) -> list[float]:
    """The last layer's activated neuron outputs for the given input values."""
    layers = evaluate_layers(input_vals)
    return [post for _, post in layers[-1]] if layers else []

_last_result = {}   # last int result per output id, to log sudden jumps

def unchosen_io() -> list[str]:
    """The inputs/outputs that still need a device and a value picked (reads the dropdowns, like read_inputs)."""
    missing = []
    for kind, items in (("in", state.inputs), ("out", state.outputs)):
        for k, it in enumerate(items):
            dev_el = state.get_id(f"dev-{kind}-{it['id']}")
            chan_el = state.get_id(f"chan-{kind}-{it['id']}")
            if not (dev_el and dev_el.value and chan_el and chan_el.value):
                missing.append(f"input ({it['name']})" if kind == "in" else f"output {k + 1}")
    return missing

def alert_unchosen(action: str) -> bool:
    """Alert if any input or output has no device/value yet. True means go ahead."""
    missing = unchosen_io()
    if missing:
        from pyscript import window
        window.alert(f"Before you {action}, choose a device and a value for:\n\n" + "\n".join("  \u2022 " + m for m in missing))
    return not missing

def read_inputs():
    # Collect input values.
    # Pragmatic exception: reads the device/channel <select> elements directly
    # rather than going through a rendering module, and calls into Device I/O
    # below -- keeping the DOM read + device call co-located here avoids
    # splitting this small function further (per the split plan).
    for inp in state.inputs:
        iid = inp["id"]
        dev_el  = state.get_id(f"dev-in-{iid}")
        chan_el = state.get_id(f"chan-in-{iid}")
        dev_name = dev_el.value if dev_el else ""
        channel  = chan_el.value if chan_el else ""
        dev = Device.device_by_name(dev_name)
        try:
            val = float(dev.read(channel)) if dev and channel else 0.0
        except (KeyError, TypeError, ValueError):
            val = 0.0
        state.input_values[iid] = val

def propagate():
    """Forward-propagate state.input_values through every layer (no device output)."""
    source_values = [state.input_values.get(inp["id"], 0.0) for inp in state.inputs]
    for layer in state.layers:
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
            state.neuron_pre_values[nid] = pre_act

            post_act = activations.apply_activation(pre_act, act_fn, custom_activation)
            state.neuron_post_values[nid] = post_act
            next_source_values.append(post_act)
        source_values = next_source_values

def preview_forward():
    """Live values on the arrows without driving any output (Training mode, while the robot is pushed by hand)."""
    read_inputs()
    propagate()
    last = state.layers[-1]["neurons"] if state.layers else []
    for idx, out in enumerate(state.outputs):
        n = last[idx] if idx < len(last) else None
        state.output_values[out["id"]] = state.neuron_post_values.get(n["id"], 0.0) if n else 0.0

def forward():
    read_inputs()
    propagate()

    # 3. Route each output from the last layer's neuron at the same position
    last_layer_neurons = state.layers[-1]["neurons"] if state.layers else []
    for idx, out in enumerate(state.outputs):
        oid = out["id"]
        val = state.neuron_post_values.get(last_layer_neurons[idx]["id"], 0.0) if idx < len(last_layer_neurons) else 0.0
        state.output_values[oid] = val

        result = int(val)
        prev = _last_result.get(oid)
        if prev is not None and abs(result - prev) >= 40:
            print(f"Output {oid} jumped {prev} -> {result}; inputs={dict(state.input_values)} "
                  f"pre={[round(state.neuron_pre_values.get(n['id'], 0), 2) for n in last_layer_neurons]}")
        _last_result[oid] = result
        dev_el  = state.get_id(f"dev-out-{oid}")
        chan_el = state.get_id(f"chan-out-{oid}")
        dev_name = dev_el.value if dev_el else ""
        channel  = chan_el.value if chan_el else ""
        Device.run_output(channel, dev_name, result)
