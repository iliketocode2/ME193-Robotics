"""Activation-function selector, help popover, and the per-layer custom
piecewise-function editor."""
from pyscript.ffi import create_proxy

import state
from state import get_id
import network_model
import arrows

def open_act_help(evt=None):
    get_id("act-help-popover").classList.remove("hidden")

def close_act_help(evt=None):
    get_id("act-help-popover").classList.add("hidden")

def make_piece_html(p: dict) -> str:
    pid = p["id"]
    def op_opts(current):
        out = ""
        for val, sym in (("<", "&lt;"), ("<=", "&le;"), (">", "&gt;"), (">=", "&ge;")):
            sel = "selected" if val == current else ""
            out += f'<option value="{val}" {sel}>{sym}</option>'
        return out
    return f"""
<div class="custom-piece-row" data-piece="{pid}">
    <input type="text" class="custom-eq-input small" id="piece-expr-{pid}" value="{p['expr']}" />
    <span class="piece-if">if</span>
    <input type="text" class="piece-num-input" id="piece-lo-{pid}" placeholder="-∞" value="{p['lo']}" />
    <select class="piece-op-select" id="piece-lo-op-{pid}">{op_opts(p['lo_op'])}</select>
    <span class="piece-x">x</span>
    <select class="piece-op-select" id="piece-hi-op-{pid}">{op_opts(p['hi_op'])}</select>
    <input type="text" class="piece-num-input" id="piece-hi-{pid}" placeholder="∞" value="{p['hi']}" />
    <button class="btn-remove-piece" id="remove-piece-{pid}">×</button>
</div>
"""

def render_custom_pieces(lid: int):
    layer = network_model.layer_by_id(lid)
    if not layer:
        return
    list_el    = get_id(f"custom-pieces-list-{lid}")
    wrap_el    = get_id(f"custom-pieces-wrap-{lid}")
    default_el = get_id(f"custom-default-expr-{lid}")
    if not list_el:
        return
    pieces = layer["custom_activation"]["pieces"]
    if pieces:
        wrap_el.classList.remove("hidden")
        default_el.classList.add("hidden")
        list_el.innerHTML = "".join(make_piece_html(p) for p in pieces)
        for p in pieces:
            bind_piece_events(lid, p["id"])
    else:
        wrap_el.classList.add("hidden")
        default_el.classList.remove("hidden")
    arrows.schedule_redraw(60)

def find_piece(lid: int, pid: int):
    layer = network_model.layer_by_id(lid)
    if not layer:
        return None
    return next((p for p in layer["custom_activation"]["pieces"] if p["id"] == pid), None)

def bind_piece_events(lid: int, pid: int):
    def bind(elid, evname, key):
        el = get_id(elid)
        if not el:
            return
        def h(evt):
            p = find_piece(lid, pid)
            if p:
                p[key] = evt.target.value
        el.addEventListener(evname, create_proxy(h))

    bind(f"piece-expr-{pid}",  "input",  "expr")
    bind(f"piece-lo-{pid}",    "input",  "lo")
    bind(f"piece-hi-{pid}",    "input",  "hi")
    bind(f"piece-lo-op-{pid}", "change", "lo_op")
    bind(f"piece-hi-op-{pid}", "change", "hi_op")

    rm = get_id(f"remove-piece-{pid}")
    if rm:
        def h(evt):
            remove_piece(lid, pid)
        rm.addEventListener("click", create_proxy(h))

def add_piece(lid: int):
    layer = network_model.layer_by_id(lid)
    if not layer:
        return
    state.piece_counter += 1
    pid = state.piece_counter
    seed_expr = "x"
    if not layer["custom_activation"]["pieces"]:
        default_el = get_id(f"custom-default-expr-{lid}")
        seed_expr = default_el.value if default_el else layer["custom_activation"]["expr"]
    layer["custom_activation"]["pieces"].append({
        "id": pid, "expr": seed_expr,
        "lo": "", "lo_op": "<", "hi": "", "hi_op": "<",
    })
    render_custom_pieces(lid)

def remove_piece(lid: int, pid: int):
    layer = network_model.layer_by_id(lid)
    if not layer:
        return
    layer["custom_activation"]["pieces"] = [
        p for p in layer["custom_activation"]["pieces"] if p["id"] != pid
    ]
    render_custom_pieces(lid)

def bind_custom_box_events(lid: int):
    default_el = get_id(f"custom-default-expr-{lid}")
    if default_el:
        def h(evt):
            layer = network_model.layer_by_id(lid)
            if layer:
                layer["custom_activation"]["expr"] = evt.target.value
        default_el.addEventListener("input", create_proxy(h))

    add_btn = get_id(f"add-piece-btn-{lid}")
    if add_btn:
        add_btn.addEventListener("click", create_proxy(lambda evt: add_piece(lid)))

def on_layer_act_select_change(lid: int, evt):
    layer = network_model.layer_by_id(lid)
    if not layer:
        return
    val = evt.target.value
    layer["act_fn"] = val

    box = get_id(f"custom-act-box-{lid}")
    if box:
        is_custom = (val == "custom")
        box.classList.toggle("hidden", not is_custom)
        get_id(f"act-box-{lid}").classList.toggle("has-custom", is_custom)
        get_id(f"act-col-{lid}").classList.toggle("act-col-custom", is_custom)
    arrows.schedule_redraw(60)

def populate_layer_act_select(lid: int):
    sel = get_id(f"act-select-{lid}")
    if not sel:
        return
    html = ""
    for label, val in state.ACTIVATION_OPTIONS:
        html += f'<option value="{val}">{label}</option>'
    sel.innerHTML = html
