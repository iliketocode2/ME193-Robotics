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
    return [f"n{idx + 1}" for idx in range(len(state.layers[layer_idx - 1]["neurons"]))]

def forward():
    # 1. Collect input values.
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
            val = float(dev.state[channel]) if dev and channel else 0.0
        except (KeyError, TypeError, ValueError):
            val = 0.0
        state.input_values[iid] = val

    # 2. Forward-propagate through every layer in order
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

    # 3. Route each output from the last layer's neuron at the same position
    last_layer_neurons = state.layers[-1]["neurons"] if state.layers else []
    for idx, out in enumerate(state.outputs):
        oid = out["id"]
        val = state.neuron_post_values.get(last_layer_neurons[idx]["id"], 0.0) if idx < len(last_layer_neurons) else 0.0
        state.output_values[oid] = val

        result = int(val)
        dev_el  = state.get_id(f"dev-out-{oid}")
        chan_el = state.get_id(f"chan-out-{oid}")
        dev_name = dev_el.value if dev_el else ""
        channel  = chan_el.value if chan_el else ""
        Device.run_output(channel, dev_name, result)
