"""Event binding for network-diagram items: neuron equations, device/channel
selects, delete buttons, layer static controls."""
from pyscript.ffi import create_proxy

import state
from state import get_id
import network_model
import Device
import activation_editor

def bind_neuron_eq_inputs(nid: int):
    n = network_model.neuron_by_id(nid)
    if not n:
        return
    for i in range(len(n["weights"])):
        inp_el = get_id(f"coeff-{nid}-{i}")
        if inp_el:
            def make_ch(neuron, idx):
                def h(evt):
                    try:
                        neuron["weights"][idx] = float(evt.target.value)
                    except (ValueError, TypeError):
                        pass
                return create_proxy(h)
            inp_el.addEventListener("input", make_ch(n, i))
    bias_el = get_id(f"bias-{nid}")
    if bias_el:
        def make_bh(neuron):
            def h(evt):
                try:
                    neuron["bias"] = float(evt.target.value)
                except (ValueError, TypeError):
                    pass
            return create_proxy(h)
        bias_el.addEventListener("input", make_bh(n))

# ── Device/channel plumbing (shared shape for in/out) ───────────────────────

def _attach_plot_to_device(plot_id: str, dev_name: str, channel: str | None):
    plot_obj = state.all_plots.get(plot_id)
    dev = Device.device_by_name(dev_name)
    if dev and channel and plot_obj:
        dev.plots.append(plot_obj)
        dev.plot_vars.append(channel)

def _detach_plot_from_device(plot_id: str, dev_name: str | None):
    Device.show_reading(plot_id, None)
    if not dev_name:
        return
    plot_obj = state.all_plots.get(plot_id)
    dev = Device.device_by_name(dev_name)
    if dev and plot_obj and plot_obj in dev.plots:
        idx = dev.plots.index(plot_obj)
        dev.plots.pop(idx)
        if idx < len(dev.plot_vars):
            dev.plot_vars.pop(idx)

def _on_device_change(evt, prefix, item_lookup_fn, channels_html_fn, attach_plot):
    """Shared body for on_input_device_change/on_output_device_change --
    they differ only in id prefix, which channel-html function builds the
    channel dropdown, and whether the plot gets attached to the device (an
    output's live plot is fed exclusively by forward()'s per-tick push of
    the network's computed output, so it's deliberately never attached to
    device.plots/plot_vars)."""
    sel = evt.target
    item_id = int(sel.id[len(f"dev-{prefix}-"):])
    item = item_lookup_fn(item_id)
    if not item:
        return
    dev_name = sel.value
    chan_sel = get_id(f"chan-{prefix}-{item_id}")
    matched  = Device.device_by_name(dev_name)
    if chan_sel:
        chan_sel.innerHTML = channels_html_fn(matched)
    first_channel = None
    if chan_sel and chan_sel.options.length > 0:
        first_channel = chan_sel.options.item(0).value

    if attach_plot:
        plot_id = f"plot-{prefix}-{item_id}"
        _detach_plot_from_device(plot_id, item.get("prev_device"))
        _attach_plot_to_device(plot_id, dev_name, first_channel)

    item["prev_device"]  = dev_name
    item["prev_channel"] = first_channel

def _on_channel_change(evt, prefix, item_lookup_fn, attach_plot):
    """Shared body for on_input_channel_change/on_output_channel_change."""
    sel = evt.target
    item_id = int(sel.id[len(f"chan-{prefix}-"):])
    item = item_lookup_fn(item_id)
    if not item:
        return
    dev_sel = get_id(f"dev-{prefix}-{item_id}")
    dev_name = dev_sel.value if dev_sel else ""
    channel  = sel.value

    if attach_plot:
        plot_id = f"plot-{prefix}-{item_id}"
        _detach_plot_from_device(plot_id, item.get("prev_device"))
        _attach_plot_to_device(plot_id, dev_name, channel)

    item["prev_device"]  = dev_name
    item["prev_channel"] = channel

def on_input_device_change(evt):
    _on_device_change(evt, "input", network_model.input_by_id, Device.get_in_channels_html, attach_plot=True)

def on_input_channel_change(evt):
    _on_channel_change(evt, "input", network_model.input_by_id, attach_plot=True)

def on_output_device_change(evt):
    _on_device_change(evt, "output", network_model.output_by_id, Device.get_out_channels_html, attach_plot=True)

def on_output_channel_change(evt):
    _on_channel_change(evt, "output", network_model.output_by_id, attach_plot=True)

# ── Item binding ─────────────────────────────────────────────────────────────

def _bind_node_events(item_id: int, prefix: str, dev_change_handler, chan_change_handler, delete_fn):
    """Shared shape for bind_input_events/bind_output_events: device select,
    channel select, delete button."""
    dev_sel = get_id(f"dev-{prefix}-{item_id}")
    if dev_sel:
        dev_sel.addEventListener("change", create_proxy(dev_change_handler))

    chan_sel = get_id(f"chan-{prefix}-{item_id}")
    if chan_sel:
        chan_sel.addEventListener("change", create_proxy(chan_change_handler))

    del_btn = get_id(f"del-{prefix}-{item_id}")
    if del_btn:
        del_btn.addEventListener("click", create_proxy(lambda evt: delete_fn(item_id)))

def bind_input_events(iid: int):
    import network_actions
    import sync
    inp = network_model.input_by_id(iid)
    if not inp:
        return
    name_el = get_id(f"name-input-{iid}")
    if name_el:
        def h(evt):
            inp["name"] = evt.target.value.strip() or inp["name"]
            sync.sync_var_labels()
        name_el.addEventListener("input", create_proxy(h))

    _bind_node_events(iid, "input", on_input_device_change, on_input_channel_change,
                       network_actions.delete_input)

def bind_neuron_events(lid: int, nid: int):
    import network_actions
    n = network_model.neuron_by_id(nid)
    if not n:
        return
    bind_neuron_eq_inputs(nid)

    del_btn = get_id(f"del-neuron-{nid}")
    if del_btn:
        del_btn.addEventListener("click", create_proxy(lambda evt: network_actions.delete_neuron(lid, nid)))

def bind_output_events(oid: int):
    import network_actions
    out = network_model.output_by_id(oid)
    if not out:
        return
    _bind_node_events(oid, "output", on_output_device_change, on_output_channel_change,
                       network_actions.delete_output)

def bind_layer_static_events(layer: dict):
    import network_actions
    lid = layer["id"]

    sel = get_id(f"act-select-{lid}")
    if sel:
        def h(evt):
            activation_editor.on_layer_act_select_change(lid, evt)
        sel.addEventListener("change", create_proxy(h))

    help_btn = get_id(f"act-help-btn-{lid}")
    if help_btn:
        help_btn.addEventListener("click", create_proxy(activation_editor.open_act_help))

    add_neuron_btn = get_id(f"add-neuron-btn-{lid}")
    if add_neuron_btn:
        add_neuron_btn.addEventListener("click", create_proxy(lambda evt: network_actions.add_neuron(lid)))

    activation_editor.bind_custom_box_events(lid)
