"""
AnnaCode — presets.py

Worked examples from the "A Neuron (Anna style)" page, loadable with one click.
Loading keeps the existing inputs'/outputs' device selections where the counts
allow, and rebuilds the layers with the given weights and biases.
"""
import state
import network_actions
import sync
import arrows

# layers: list of layers; each layer is a list of {"w": [weights], "b": bias}
PRESETS = {
    "first": {
        "label": "A first neuron: y = 2x + 20",
        "caption": "This is one neuron that multiplies the sensor reading by 2 and then adds 20, assuming you have a Color Sensor as the input (set to <b>Reflection</b>) and a Double Motor as the output (set to <b>BothSpeed</b>). Change the numbers to see how the motor speed changes.",
        "n_in": 1, "n_out": 1,
        "layers": [[{"w": [2.0], "b": 20.0}]],
    },
    "wall": {
        "label": "Stop at the wall",
        "caption": "This stops at the wall: the output is 25 minus the sensor reading, so the car stops when the sensor reads exactly 25. It assumes you have a Color Sensor as the input (set to <b>Reflection</b>) and a Double Motor as the output (set to <b>BothSpeed</b>). Change the 25 if it does not stop where you want.",
        "n_in": 1, "n_out": 1,
        "layers": [[{"w": [-1.0], "b": 25.0}]],
    },
    "motors": {
        "label": "Each motor separately",
        "caption": "This controls each motor separately, assuming you have a Color Sensor as the input (set to <b>Reflection</b>) and a Double Motor as the outputs (one is <b>LeftSpeed</b> and the other is <b>RightSpeed</b>). Note the opposite signs on the two neurons, so the motors go the same direction.",
        "n_in": 1, "n_out": 2,
        "layers": [
            [{"w": [1.0], "b": -10.0}],
            [{"w": [1.0], "b": 0.0}, {"w": [-1.0], "b": 0.0}],
        ],
    },
    "line": {
        "label": "Follow a line",
        "caption": "This follows a line, assuming you have a Color Sensor as the input (set to <b>Reflection</b>) and a Double Motor as the outputs (one is <b>LeftSpeed</b> and the other is <b>RightSpeed</b>). Change the weights if it is not following well.",
        "n_in": 1, "n_out": 2,
        "layers": [
            [{"w": [1.0], "b": -75.0}],
            [{"w": [-0.2], "b": -10.0}, {"w": [-0.2], "b": 10.0}],
        ],
    },
}

def populate_select(sel):
    sel.innerHTML = '<option value="">Example networks…</option>' + "".join(
        f'<option value="{key}">{p["label"]}</option>' for key, p in PRESETS.items())

def load(key: str):
    p = PRESETS.get(key)
    if not p:
        return
    if state.is_running:
        network_actions.stop_network()

    while len(state.inputs) < p["n_in"]:
        network_actions.add_input()
    while len(state.inputs) > p["n_in"]:
        network_actions.delete_input(state.inputs[-1]["id"])
    while len(state.outputs) < p["n_out"]:
        network_actions.add_output()
    while len(state.outputs) > p["n_out"]:
        network_actions.delete_output(state.outputs[-1]["id"])

    # Build the new layers first (the last layer can never be deleted), then drop the old ones
    old_ids = [l["id"] for l in state.layers]
    for spec in p["layers"]:
        network_actions.add_layer()
        lid = state.layers[-1]["id"]
        for _ in range(len(spec) - 1):
            network_actions.add_neuron(lid)
    for lid in old_ids:
        network_actions.delete_layer(lid)

    for li, spec in enumerate(p["layers"]):
        for n, s in zip(state.layers[li]["neurons"], spec):
            n["weights"] = list(s["w"])
            n["bias"] = s["b"]
        sync.rebuild_layer_eq_html(li)
    sync.update_neuron_usage_state()
    arrows.schedule_redraw(80)
