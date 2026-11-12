"""
AnnaCode — main.py
PyScript 2026.3.1

Structure: independent columns, chained through any number of layers.
  Inputs  -- each with its own device/channel + plot
  Layers  -- each layer has its own column of neurons AND its own
             activation block (function + optional custom piecewise editor).
             layers[0]'s neurons take a weighted sum of the raw inputs;
             every later layer's neurons take a weighted sum of the
             *previous* layer's activated neuron outputs.
             "+ Layer" (between the last activation block and the outputs)
             appends a brand new neuron layer + activation block.
  Outputs -- each with its own device/channel + plot. Outputs pair up with
             the LAST layer's neurons positionally (1st output <-> 1st
             neuron of the last layer, etc). Any neuron in the last layer
             beyond the current number of outputs is shown greyed out
             until a matching output exists.
  A column (inputs / a layer's neurons / outputs) can never be emptied --
  the last remaining item's delete button is hidden.

This file only wires up the static @when(...) handlers and boots the app --
see state.py, activations.py, network_model.py, templates.py, arrows.py,
sync.py, bindings.py, activation_editor.py, network_actions.py, zoom.py,
and Device.py for the actual implementation.
"""
from pyscript import document, when

import state
import Device
import arrows
import network_actions
import activation_editor
import zoom
import transfer_plot
import help_system
import presets
import training
import asyncio

# ── Static wiring ──────────────────────────────────────────────────────────────

@when("click", "#add-device-btn")
async def _on_add_device(evt):
    await Device.create_new_device()

@when("click", "#add-input-btn")
def _on_add_input(evt):
    network_actions.add_input()

@when("click", "#add-layer-btn")
def _on_add_layer(evt):
    network_actions.add_layer()

@when("click", "#add-output-btn")
def _on_add_output(evt):
    network_actions.add_output()
    if len(state.outputs) == 2:
        help_system.show_hint("outputs")

@when("click", "#play-btn")
def _on_play(evt):
    network_actions.play_network()

@when("click", "#stop-btn")
def _on_stop(evt):
    network_actions.stop_network()

@when("click", "#close-act-help-btn")
def _on_close_act_help(evt):
    activation_editor.close_act_help()

@when("click", "#zoom-out-btn")
def _on_zoom_out(evt):
    zoom.zoom_out()

@when("click", "#zoom-in-btn")
def _on_zoom_in(evt):
    zoom.zoom_in()

@when("change", "#debug-toggle")
def _on_debug_toggle(evt):
    state.debug_mode = evt.target.checked
    arrows.redraw_arrows()

@when("change", "#preset-select")
def _on_preset(evt):
    key = evt.target.value
    caption = state.get_id("preset-caption")
    presets.load(key)
    state.get_id("preset-caption-text").innerHTML = presets.PRESETS[key]["caption"] if key else ""
    caption.classList.toggle("hidden", not key)   # loading an example always brings the hint back, even if it was closed

@when("click", "#preset-caption-close")
def _on_caption_close(evt):
    state.get_id("preset-caption").classList.add("hidden")

@when("click", "#train-hint-close")
def _on_train_hint_close(evt):
    state.train_hint_closed = True

@when("click", "#help-close-btn")
def _on_help_close(evt):
    help_system.close_help()

@when("click", "#hint-close-btn")
def _on_hint_close(evt):
    help_system.close_hint()

def update_activation_toggle():
    """Label for the Training-mode toggle; when hidden, still says which activations are active."""
    btn = state.get_id("show-activation-btn")
    names = dict((v, k) for k, v in state.ACTIVATION_OPTIONS)
    active = sorted({names.get(l["act_fn"], l["act_fn"]) for l in state.layers if l["act_fn"]})
    if state.show_activation:
        btn.textContent = "− Activation layers"
    else:
        btn.textContent = "+ Activation layers" + (f" (on: {', '.join(active)})" if active else "")

@when("click", "#show-activation-btn")
def _on_toggle_activation(evt):
    state.show_activation = not state.show_activation
    set_mode(state.mode)

@when("click", "#randomize-btn")
def _on_randomize(evt):
    training.randomize_weights()

FITTING_STATUS = "Pick an example curve, then change the weights, biases and activations until the green line follows the dots."
TRAINING_STATUS = "Pick a device and a value for each input and output, press Record, and push the robot through its paces."

def _place_example_select(in_input_box: bool):
    """Fitting has no sensor, so the example picker takes the sensor dropdown's place in the input box."""
    sel = state.get_id("example-select")
    sel.title = ("Pick an example curve to fit. Changing it replaces the dots." if in_input_box
                 else help_system.HELP["example"][0])
    if in_input_box:
        header = document.querySelector(".input-node .node-header")
        if header and sel.parentElement is not header:
            header.insertBefore(sel, header.firstChild)
            header.insertBefore(state.get_id("save-icon-btn"), sel.nextSibling)
    else:
        home = state.get_id("record-btn").parentElement
        if sel.parentElement is not home:
            home.insertBefore(sel, home.firstChild)
            home.insertBefore(state.get_id("save-icon-btn"), sel.nextSibling)

def set_mode(mode: str):
    previous = state.mode
    state.mode = mode
    state.beginner_mode = mode != "advanced"           # Beginner, Fitting and Training use the simplified layout
    state.use_activations = mode != "beginner"         # Beginner ignores activations; the others apply them
    # Fitting always shows the activation blocks (they are the point of it); Training hides them behind its toggle
    show_act = mode == "fitting" or (mode == "training" and state.show_activation)
    state.show_activation_blocks = mode == "advanced" or show_act
    wrap = state.get_id("page-wrap")
    wrap.classList.toggle("beginner", state.beginner_mode)
    wrap.classList.toggle("training", state.in_training_family())
    wrap.classList.toggle("fitting", mode == "fitting")
    wrap.classList.toggle("show-act", show_act)
    update_activation_toggle()
    for m in ("beginner", "advanced", "fitting", "training"):
        state.get_id(f"mode-{m}-btn").classList.toggle("active", m == mode)
    if previous != mode and "fitting" in (previous, mode):
        training.clear_data()                  # example curves don't carry over to or from recorded data
        state.undo_stack.clear()
        state.example_key = None
        training.set_status(FITTING_STATUS if mode == "fitting" else TRAINING_STATUS)
    _place_example_select(mode == "fitting")
    if previous != mode and mode == "fitting":
        state.zoom_level = 0.8                 # small networks fit on screen at once
        zoom.apply_zoom()
    if mode == "fitting":
        training.stop_recording()
        if state.is_running:
            network_actions.stop_network()
        for kind, items, delete in (("input", state.inputs, network_actions.delete_input),
                                    ("output", state.outputs, network_actions.delete_output)):
            while len(items) > 1:               # Fitting is one x in, one y out
                delete(items[-1]["id"])
        network_actions.match_last_layer_to_outputs()
    elif mode == "training":
        network_actions.match_last_layer_to_outputs()
    else:
        training.stop_recording()
        if not state.is_running:   # drop the live-preview numbers so they don't linger on the arrows
            for d in (state.input_values, state.neuron_pre_values, state.neuron_post_values, state.output_values):
                d.clear()
    # Layout changed (columns hidden/shown) -- re-measure the arrows
    arrows.schedule_redraw()

@when("click", "#mode-beginner-btn")
def _on_mode_beginner(evt):
    set_mode("beginner")

@when("click", "#mode-advanced-btn")
def _on_mode_advanced(evt):
    set_mode("advanced")

@when("click", "#mode-fitting-btn")
def _on_mode_fitting(evt):
    set_mode("fitting")

@when("click", "#mode-training-btn")
def _on_mode_training(evt):
    set_mode("training")

@when("click", "#record-btn")
def _on_record(evt):
    training.toggle_recording()

@when("click", "#remove-zero-btn")
def _on_remove_zero(evt):
    training.remove_zero_points()

@when("click", "#clear-data-btn")
def _on_clear_data(evt):
    training.clear_data()

@when("change", "#example-select")
def _on_example(evt):
    if state.mode == "fitting":      # Fitting can swap curves freely; Training keeps Record and examples apart
        training.clear_data()
        state.undo_stack.clear()
    training.add_example(evt.target.value)

@when("click", "#undo-btn")
def _on_undo(evt):
    training.undo()

@when("click", "#save-data-btn")
def _on_save(evt):
    training.save_dataset()

@when("click", "#save-icon-btn")
def _on_save_icon(evt):
    training.save_dataset()

@when("click", "#load-data-btn")
def _on_load_click(evt):
    state.get_id("load-file").click()

@when("change", "#load-file")
async def _on_load_file(evt):
    file = evt.target.files.item(0)
    if file:
        training.load_dataset(await file.text())
    evt.target.value = ""   # so the same file can be loaded again

@when("click", "#step-btn")
async def _on_step(evt):
    await training.train(1)

@when("click", "#train-btn")
async def _on_train(evt):
    await training.train(50)

# ── Boot ───────────────────────────────────────────────────────────────────────

def boot():
    state.get_id("loading-splash").style.display = "none"
    state.get_id("page-wrap").style.display = "flex"

    # one of each on load: 1 input, 1 layer (1 neuron + activation), 1 output
    network_actions.add_input()
    network_actions.add_layer()
    network_actions.add_output()

    presets.populate_select(state.get_id("preset-select"))
    training.populate_examples(state.get_id("example-select"))
    help_system.setup()
    set_mode(state.mode)
    asyncio.ensure_future(transfer_plot.run_loop())
    zoom.apply_zoom()
    zoom.setup_resize_observer()
    arrows.schedule_redraw(150)

boot()
