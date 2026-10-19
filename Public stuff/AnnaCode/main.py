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
from pyscript import when

import state
import Device
import arrows
import network_actions
import activation_editor
import zoom

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

# ── Boot ───────────────────────────────────────────────────────────────────────

def boot():
    state.get_id("loading-splash").style.display = "none"
    state.get_id("page-wrap").style.display = "flex"

    # one of each on load: 1 input, 1 layer (1 neuron + activation), 1 output
    network_actions.add_input()
    network_actions.add_layer()
    network_actions.add_output()

    zoom.apply_zoom()
    zoom.setup_resize_observer()
    arrows.schedule_redraw(150)

boot()
