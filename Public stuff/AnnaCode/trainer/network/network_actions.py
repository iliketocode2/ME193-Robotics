"""CRUD for inputs/neurons/layers/outputs: the rendering/coordination half.
Each function calls into network_model for the data mutation, then
templates + bindings + sync + arrows to reflect it in the DOM. This is the
one file allowed to import from both the model side and the rendering side."""
from pyscript import window
from pyscript.ffi import create_proxy

import live_plot
import state
from state import get_id
import network_model
import templates
import bindings
import sync
import arrows
import dataset_ui

# ── CRUD: inputs ─────────────────────────────────────────────────────────────

def _next_input_name() -> str:
    n = len(state.inputs) + 1
    return chr((ord('x') - ord('a') + (n - 1)) % 26 + ord('a'))

def add_input(evt=None):
    state.input_counter += 1
    iid = state.input_counter
    inp = {"id": iid, "name": _next_input_name(), "prev_device": None, "prev_channel": None}
    state.inputs.append(inp)

    templates.append_html("inputs-container", templates.make_input_html(inp))
    bindings.bind_input_events(iid)

    if state.layers:
        for n in state.layers[0]["neurons"]:
            n["weights"].append(1.0)
        sync.rebuild_layer_eq_html(0)

    sync.update_delete_visibility(state.inputs, "input")

    for p in state.training_data:
        p["xs"].setdefault(iid, 0.0)
    dataset_ui.render_dataset_header()
    dataset_ui.render_dataset_table()
    dataset_ui.render_add_point_row()
    dataset_ui.refresh_dataset_plot_points()

    def make_plot(iid=iid):
        state.all_plots[f"plot-input-{iid}"] = live_plot.LivePlot(f"plot-input-{iid}")
    window.setTimeout(create_proxy(make_plot), 60)

    arrows.schedule_redraw(60)

def delete_input(iid: int):
    if len(state.inputs) <= 1:
        return
    idx = next((i for i, inp in enumerate(state.inputs) if inp["id"] == iid), None)
    if idx is None:
        return
    inp = state.inputs[idx]

    plot_id = f"plot-input-{iid}"
    bindings._detach_plot_from_device(plot_id, inp.get("prev_device"))
    state.all_plots.pop(plot_id, None)

    el = get_id(f"item-input-{iid}")
    if el:
        el.remove()

    state.inputs.pop(idx)
    if state.layers:
        for n in state.layers[0]["neurons"]:
            if idx < len(n["weights"]):
                n["weights"].pop(idx)
        sync.rebuild_layer_eq_html(0)

    for p in state.training_data:
        p["xs"].pop(iid, None)
    dataset_ui.render_dataset_header()
    dataset_ui.render_dataset_table()
    dataset_ui.render_add_point_row()
    dataset_ui.refresh_dataset_plot_points()

    sync.update_delete_visibility(state.inputs, "input")
    arrows.schedule_redraw(60)

# ── CRUD: neurons ────────────────────────────────────────────────────────────

def add_neuron(lid: int):
    layer = network_model.layer_by_id(lid)
    idx = network_model.layer_index_by_id(lid)
    if layer is None or idx is None:
        return

    state.neuron_counter += 1
    nid = state.neuron_counter
    n = {"id": nid, "weights": [1.0] * network_model.get_source_count(idx), "bias": 0.0}
    layer["neurons"].append(n)

    templates.append_html(f"neurons-container-{lid}", templates.make_neuron_html(n, network_model.get_source_labels(idx)))
    bindings.bind_neuron_events(lid, nid)

    sync.refresh_all_neuron_delete_visibility()
    sync.sync_downstream_after_neuron_count_change(idx)
    sync.update_neuron_usage_state()
    arrows.schedule_redraw(60)

def delete_neuron(lid: int, nid: int):
    layer = network_model.layer_by_id(lid)
    idx = network_model.layer_index_by_id(lid)
    if layer is None or idx is None:
        return
    pos = next((i for i, n in enumerate(layer["neurons"]) if n["id"] == nid), None)
    if pos is None:
        return

    # Deleting the first neuron in a layer removes the whole layer (and its
    # activation block), as long as another layer exists to take its place.
    if pos == 0 and len(state.layers) > 1:
        delete_layer(lid)
        return

    if len(layer["neurons"]) <= 1:
        return

    el = get_id(f"item-neuron-{nid}")
    if el:
        el.remove()

    layer["neurons"].pop(pos)
    state.neuron_pre_values.pop(nid, None)
    state.neuron_post_values.pop(nid, None)

    sync.refresh_all_neuron_delete_visibility()
    sync.sync_downstream_after_neuron_count_change(idx, removed_pos=pos)
    sync.update_neuron_usage_state()
    arrows.schedule_redraw(60)

# ── CRUD: layers ─────────────────────────────────────────────────────────────

def delete_layer(lid: int):
    """Remove an entire layer (all its neurons + its activation block).
    Never removes the last remaining layer. The layer that used to follow
    this one (if any) is reconnected to draw from this layer's predecessor."""
    idx = network_model.layer_index_by_id(lid)
    if idx is None or len(state.layers) <= 1:
        return
    layer = state.layers[idx]

    for n in layer["neurons"]:
        el = get_id(f"item-neuron-{n['id']}")
        if el:
            el.remove()
        state.neuron_pre_values.pop(n["id"], None)
        state.neuron_post_values.pop(n["id"], None)

    col_n = get_id(f"col-neurons-{lid}")
    if col_n:
        col_n.remove()
    col_a = get_id(f"act-col-{lid}")
    if col_a:
        col_a.remove()

    state.layers.pop(idx)

    # The layer now sitting at `idx` (previously idx+1, if any) needs its
    # weights/labels rebuilt against its new predecessor.
    if idx < len(state.layers):
        sync.rebuild_layer_eq_html(idx)

    sync.refresh_all_neuron_delete_visibility()
    sync.update_neuron_usage_state()
    arrows.schedule_redraw(60)

def add_layer(evt=None):
    import activation_editor
    state.layer_counter += 1
    lid = state.layer_counter
    layer = {"id": lid, "neurons": [], "act_fn": "", "custom_activation": {"expr": "x", "pieces": []}}
    state.layers.append(layer)

    templates.append_multi_html(
        "layers-container",
        templates.make_layer_neurons_col_html(layer) + templates.make_layer_activation_col_html(layer))

    activation_editor.populate_layer_act_select(lid)
    bindings.bind_layer_static_events(layer)

    add_neuron(lid)   # every layer starts with exactly one neuron

    sync.update_neuron_usage_state()
    arrows.schedule_redraw(80)

# ── CRUD: outputs ────────────────────────────────────────────────────────────

def add_output(evt=None):
    state.output_counter += 1
    oid = state.output_counter
    out = {"id": oid, "prev_device": None, "prev_channel": None}
    state.outputs.append(out)

    templates.append_html("outputs-container", templates.make_output_html(out))
    bindings.bind_output_events(oid)

    def make_plot(oid=oid):
        state.all_plots[f"plot-output-{oid}"] = live_plot.LivePlot(f"plot-output-{oid}")
    window.setTimeout(create_proxy(make_plot), 60)

    for p in state.training_data:
        p["ys"].setdefault(oid, 0.0)

    fit_plot_obj = state.all_plots.get("plot-fit")
    if fit_plot_obj:
        fit_plot_obj.add_output_trace(oid)

    dataset_ui.render_dataset_header()
    dataset_ui.render_dataset_table()
    dataset_ui.render_add_point_row()
    dataset_ui.refresh_dataset_plot_points()

    sync.update_delete_visibility(state.outputs, "output")
    sync.update_neuron_usage_state()
    arrows.schedule_redraw(60)

def delete_output(oid: int):
    if len(state.outputs) <= 1:
        return
    idx = next((i for i, o in enumerate(state.outputs) if o["id"] == oid), None)
    if idx is None:
        return
    state.all_plots.pop(f"plot-output-{oid}", None)

    fit_plot_obj = state.all_plots.get("plot-fit")
    if fit_plot_obj:
        fit_plot_obj.remove_output_trace(oid)

    el = get_id(f"item-output-{oid}")
    if el:
        el.remove()

    state.outputs.pop(idx)
    state.output_values.pop(oid, None)

    for p in state.training_data:
        p["ys"].pop(oid, None)
    dataset_ui.render_dataset_header()
    dataset_ui.render_dataset_table()
    dataset_ui.render_add_point_row()
    # Kept consistent with add_input/delete_input/add_output, which all
    # refresh the fit plot's data points after touching training_data --
    # this call was missing here originally (confirmed a bug, not
    # intentional, per the implementation plan's Phase 5.6 flag).
    dataset_ui.refresh_dataset_plot_points()

    sync.update_delete_visibility(state.outputs, "output")
    sync.update_neuron_usage_state()
    arrows.schedule_redraw(60)
