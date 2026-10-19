"""Pure HTML string-building for network-diagram items, plus the small
append-HTML-string-into-the-DOM helpers CRUD actions use to realize it."""
from pyscript import document
from state import get_id
import Device

def append_html(container_id: str, html: str):
    """Parse `html` (expected to be a single root element) and append it to
    the container. Returns the appended element."""
    container = get_id(container_id)
    wrapper = document.createElement("div")
    wrapper.innerHTML = html
    child = wrapper.firstElementChild
    container.appendChild(child)
    return child

def append_multi_html(container_id: str, html: str):
    """Like append_html, but appends every root element in `html` (used by
    add_layer, which builds a neurons column + an activation column in one
    string)."""
    container = get_id(container_id)
    wrapper = document.createElement("div")
    wrapper.innerHTML = html
    while wrapper.firstElementChild:
        container.appendChild(wrapper.firstElementChild)

def delete_x_svg() -> str:
    return ('<svg width="10" height="10" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="3" stroke-linecap="round">'
            '<path d="M18 6L6 18M6 6l12 12"/></svg>')

def make_input_html(inp: dict) -> str:
    iid  = inp["id"]
    name = inp["name"]
    dev_opts = Device.get_device_options_html()
    return f"""
<div class="network-item" id="item-input-{iid}" data-id="{iid}">
    <div class="cell-node" id="cell-input-{iid}">
        <div class="cell-label">
            <input type="text" class="name-input" id="name-input-{iid}"
                   value="{name}" maxlength="12" />
        </div>
        <div class="node-card input-node hover-target">
            <button class="item-delete-btn" id="del-input-{iid}" title="Remove input">{delete_x_svg()}</button>
            <div class="node-header">
                <select class="node-device-select" id="dev-input-{iid}">{dev_opts}</select>
                <span class="node-reading" id="reading-input-{iid}">—</span>
            </div>
            <div class="node-plot">
                <div class="plot-canvas" id="plot-input-{iid}" width="170" height="70"></div>
            </div>
            <div class="plot-footer">
                <select class="channel-select" id="chan-input-{iid}">
                    <option value="">— value —</option>
                </select>
            </div>
        </div>
    </div>
</div>
"""

def make_neuron_eq_inner_html(n: dict, labels: list[str]) -> str:
    nid = n["id"]
    parts = ""
    for i, label in enumerate(labels):
        coeff_val = n["weights"][i] if i < len(n["weights"]) else 1.0
        if i > 0:
            parts += '<span class="eq-op">+</span>'
        parts += (
            f'<input type="number" step="any" value="{coeff_val:.2f}"'
            f' class="eq-num-input" id="coeff-{nid}-{i}" data-neuron="{nid}" data-idx="{i}" />'
            f'<span class="eq-var" id="var-label-{nid}-{i}">{label}</span>'
        )
    parts += (
        f'<span class="eq-op">+</span>'
        f'<input type="number" step="any" value="{n["bias"]:.2f}"'
        f' class="eq-bias-input" id="bias-{nid}" data-neuron="{nid}" />'
    )
    return parts

def make_neuron_html(n: dict, labels: list[str]) -> str:
    nid = n["id"]
    return f"""
<div class="network-item" id="item-neuron-{nid}" data-id="{nid}">
    <div class="cell-node" id="cell-neuron-{nid}">
        <div class="eq-node hover-target" id="eq-node-{nid}">
            <button class="item-delete-btn" id="del-neuron-{nid}" title="Remove neuron">{delete_x_svg()}</button>
            <div class="eq-inline" id="eq-inline-{nid}">
                {make_neuron_eq_inner_html(n, labels)}
            </div>
        </div>
    </div>
</div>
"""

def make_output_html(out: dict) -> str:
    oid = out["id"]
    dev_opts = Device.get_device_options_html()
    return f"""
<div class="network-item" id="item-output-{oid}" data-id="{oid}">
    <div class="cell-node" id="cell-output-{oid}">
        <div class="node-card output-node hover-target">
            <button class="item-delete-btn" id="del-output-{oid}" title="Remove output">{delete_x_svg()}</button>
            <div class="node-header">
                <select class="node-device-select" id="dev-output-{oid}">{dev_opts}</select>
                <span class="node-reading" id="reading-output-{oid}">—</span>
            </div>
            <div class="node-plot">
                <div class="plot-canvas" id="plot-output-{oid}" width="170" height="70"></div>
            </div>
            <div class="plot-footer">
                <select class="channel-select" id="chan-output-{oid}">
                    <option value="">— value —</option>
                </select>
            </div>
        </div>
    </div>
</div>
"""

def make_layer_neurons_col_html(layer: dict) -> str:
    lid = layer["id"]
    return f"""
<div class="col col-neurons" id="col-neurons-{lid}" data-layer="{lid}">
    <div class="col-items" id="neurons-container-{lid}"></div>
    <button class="btn-add-col" id="add-neuron-btn-{lid}" title="Add neuron">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5">
            <path d="M12 5v14M5 12h14"/>
        </svg>
        Neuron
    </button>
</div>
"""

def make_layer_activation_col_html(layer: dict) -> str:
    lid = layer["id"]
    return f"""
<div class="col col-activation" id="act-col-{lid}" data-layer="{lid}">
    <div class="act-box" id="act-box-{lid}">
        <div class="act-box-label">Activation</div>
        <select class="act-select" id="act-select-{lid}"></select>
        <button class="act-help-btn" id="act-help-btn-{lid}" title="What is an activation function?">?</button>

        <div class="custom-act-box hidden" id="custom-act-box-{lid}">
            <div class="custom-main-row">
                <span class="custom-fx-label">f(x) =</span>
                <input type="text" class="custom-eq-input" id="custom-default-expr-{lid}" value="x" />
                <div class="custom-pieces-wrap hidden" id="custom-pieces-wrap-{lid}">
                    <div class="custom-brace">{{</div>
                    <div class="custom-pieces-list" id="custom-pieces-list-{lid}"></div>
                </div>
            </div>
            <button class="btn-add-piece" id="add-piece-btn-{lid}">+ Add piece</button>
        </div>
    </div>
</div>
"""
