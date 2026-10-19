"""Keeps the DOM in sync with state after topology/device changes: equation
rebuilds, delete-button visibility, device dropdown refreshes.

Exception: rebuild_layer_eq_html() needs network_model's pure topology
lookups (get_source_count/get_source_labels) to know how many weight terms
a layer's equations need -- there's no DOM-free way to keep those in sync
without them, so this file imports network_model for that alone (see the
implementation plan's dependency-direction note, which otherwise keeps
network_model out of the rendering files)."""
import state
from state import get_id
import network_model
import templates
import Device

def sync_var_labels():
    """Refresh the input-name labels shown inside layer 0's equations
    (deeper layers use auto-generated positional labels, unaffected)."""
    if not state.layers:
        return
    for i, inp in enumerate(state.inputs):
        for n in state.layers[0]["neurons"]:
            el = get_id(f"var-label-{n['id']}-{i}")
            if el:
                el.textContent = inp["name"]

def rebuild_layer_eq_html(layer_idx: int):
    """Rebuild every neuron's equation row in this layer to match its
    current source count/labels (input count for layer 0, previous
    layer's neuron count otherwise)."""
    import bindings
    if layer_idx < 0 or layer_idx >= len(state.layers):
        return
    layer = state.layers[layer_idx]
    source_count = network_model.get_source_count(layer_idx)
    labels = network_model.get_source_labels(layer_idx)
    for n in layer["neurons"]:
        nid = n["id"]
        container = get_id(f"eq-inline-{nid}")
        if not container:
            continue
        while len(n["weights"]) < source_count:
            n["weights"].append(1.0)
        if len(n["weights"]) > source_count:
            n["weights"] = n["weights"][:source_count]
        container.innerHTML = templates.make_neuron_eq_inner_html(n, labels)
        bindings.bind_neuron_eq_inputs(nid)

def sync_downstream_after_neuron_count_change(layer_idx: int, removed_pos: int | None = None):
    """Call after a neuron was added to or removed from layers[layer_idx].
    Keeps the NEXT layer's weight vectors in sync (append 1.0 on add,
    pop the matching index on remove) and rebuilds its equation display."""
    if layer_idx + 1 >= len(state.layers):
        return
    next_layer = state.layers[layer_idx + 1]
    if removed_pos is not None:
        for n in next_layer["neurons"]:
            if removed_pos < len(n["weights"]):
                n["weights"].pop(removed_pos)
    else:
        for n in next_layer["neurons"]:
            n["weights"].append(1.0)
    rebuild_layer_eq_html(layer_idx + 1)

def update_neuron_usage_state():
    """Neurons in the LAST layer beyond the number of outputs aren't wired
    to anything yet -- grey them out until a matching output is added."""
    for li, layer in enumerate(state.layers):
        is_last = (li == len(state.layers) - 1)
        for i, n in enumerate(layer["neurons"]):
            eq_el = get_id(f"eq-node-{n['id']}")
            if not eq_el:
                continue
            if is_last and i >= len(state.outputs):
                eq_el.classList.add("neuron-unused")
            else:
                eq_el.classList.remove("neuron-unused")

def update_delete_visibility(item_list: list, prefix: str):
    """Hide the delete button on every item in a column when only one remains,
    so a column can never be emptied."""
    can_delete = len(item_list) > 1
    for it in item_list:
        btn = get_id(f"del-{prefix}-{it['id']}")
        if btn:
            if can_delete:
                btn.classList.remove("item-delete-hidden")
            else:
                btn.classList.add("item-delete-hidden")

def update_neuron_delete_visibility(layer: dict):
    """The first neuron in a layer can be deleted (removing the whole layer)
    whenever another layer exists, even if it's that layer's only neuron.
    Any other neuron follows the normal 'can't empty this layer' rule."""
    neurons = layer["neurons"]
    layer_removable = len(state.layers) > 1
    for i, n in enumerate(neurons):
        btn = get_id(f"del-neuron-{n['id']}")
        if not btn:
            continue
        removable = (i == 0 and layer_removable) or len(neurons) > 1
        if removable:
            btn.classList.remove("item-delete-hidden")
        else:
            btn.classList.add("item-delete-hidden")

def refresh_all_neuron_delete_visibility():
    for layer in state.layers:
        update_neuron_delete_visibility(layer)

def refresh_device_dropdowns():
    dev_opts = Device.get_device_options_html()
    for inp in state.inputs:
        sel = get_id(f"dev-input-{inp['id']}")
        if sel:
            cur = sel.value
            sel.innerHTML = dev_opts
            sel.value = cur
    for out in state.outputs:
        sel = get_id(f"dev-output-{out['id']}")
        if sel:
            cur = sel.value
            sel.innerHTML = dev_opts
            sel.value = cur
