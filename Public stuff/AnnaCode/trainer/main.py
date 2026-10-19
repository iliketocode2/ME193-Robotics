"""
AnnaCode Trainer — main.py
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
  Outputs -- each with its own device/channel + a scatter plot of output
             vs. input. Outputs pair up with the LAST layer's neurons
             positionally (1st output <-> 1st neuron of the last layer,
             etc). Any neuron in the last layer beyond the current number
             of outputs is shown greyed out until a matching output exists.
  A column (inputs / a layer's neurons / outputs) can never be emptied --
  the last remaining item's delete button is hidden.

Training loop (manual, no hardware involved):
  Randomize weights -- gives every neuron a fresh random weight vector + bias.
  Test              -- forward-propagates the CURRENT input reading (without
                        touching the hardware outputs) and drops a point on
                        each output's scatter plot at (first input's value,
                        predicted output).
  Backpropagate      -- compares each output's last Test prediction against
                        the target value typed into that output's "target y"
                        box, and nudges every weight/bias one gradient-descent
                        step to reduce that error.
  Run / Stop         -- unchanged: continuously forward-propagates live
                        sensor input straight into the hardware outputs.

This file only wires up the static @when(...) handlers and boots the app --
all logic lives in the modules it imports (state, activations,
network_model, templates, arrows, sync, bindings, network_actions,
dataset_ui, activation_editor, ui_chrome, Device, and the plot_* files).
"""
import asyncio
from pyscript import document, when, window

import state
from state import get_id
import network_model
import network_actions
import activation_editor
import ui_chrome
import dataset_ui
import arrows
import Device
import fit_plot
import loss_plot

# ── Play / Stop ──────────────────────────────────────────────────────────────

def play_network(evt=None):
    state.is_running = True
    state.run_tick = 0

    fit_plot_obj = state.all_plots.get("plot-fit")
    if fit_plot_obj:
        fit_plot_obj.reset_run()

    get_id("play-btn").setAttribute("disabled", "")
    get_id("stop-btn").removeAttribute("disabled")
    document.body.classList.add("running")

    debug_wrap = get_id("debug-toggle-wrap")
    if debug_wrap:
        debug_wrap.classList.remove("hidden")

    asyncio.ensure_future(loop_network())

def stop_network(evt=None):
    state.is_running = False
    get_id("stop-btn").setAttribute("disabled", "")
    get_id("play-btn").removeAttribute("disabled")
    document.body.classList.remove("running")
    Device.flush_pending_commands()
    for device in state.devices:
        try:
            device.stop()
        except Exception as e:
            print("Caught: " + str(e))

    debug_wrap = get_id("debug-toggle-wrap")
    if debug_wrap:
        debug_wrap.classList.add("hidden")

    debug_checkbox = get_id("debug-toggle")
    if debug_checkbox:
        debug_checkbox.checked = False
    state.debug_mode = False
    arrows.redraw_arrows()

async def loop_network():
    while state.is_running:
        network_model.forward()
        if state.debug_mode:
            arrows.redraw_arrows()
        await asyncio.sleep(0.05)

def get_learning_rate() -> float:
    el = get_id("learning-rate-input")
    try:
        val = float(el.value) if el else state.DEFAULT_LEARNING_RATE
        return val if val > 0 else state.DEFAULT_LEARNING_RATE
    except (ValueError, TypeError):
        return state.DEFAULT_LEARNING_RATE

# ── Browser support check ────────────────────────────────────────────────────
# Web Bluetooth (used by Device.py/ble.js to talk to the LEGO hardware) isn't
# implemented in Safari or Firefox -- everything else on this page still
# works without it, so this only surfaces a dismissible warning rather than
# blocking the app.

def check_browser_support():
    banner = get_id("ble-warning-banner")
    # Safari/Firefox don't define navigator.bluetooth at all (not just
    # undefined), so a plain attribute access raises AttributeError there --
    # hasattr() is the safe way to feature-detect it.
    if banner and not hasattr(window.navigator, "bluetooth"):
        banner.classList.remove("hidden")
        Device.enable_virtual_controller()

@when("click", "#close-ble-warning-btn")
def _on_close_ble_warning(evt):
    banner = get_id("ble-warning-banner")
    if banner:
        banner.classList.add("hidden")

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

@when("click", "#randomize-btn")
def _on_randomize(evt):
    network_model.randomize_weights()

@when("click", "#step-btn")
def _on_step(evt):
    network_model.train_step(get_learning_rate())

@when("click", "#train50-btn")
def _on_train30(evt):
    network_model.train_50_epochs(get_learning_rate())

@when("click", "#clear-fit-btn")
def _on_clear_fit(evt):
    fit_plot_obj = state.all_plots.get("plot-fit")
    if fit_plot_obj:
        fit_plot_obj.reset_run()

@when("click", "#play-btn")
def _on_play(evt):
    play_network()

@when("click", "#stop-btn")
def _on_stop(evt):
    stop_network()

@when("click", "#close-act-help-btn")
def _on_close_act_help(evt):
    activation_editor.close_act_help()

@when("click", "#zoom-out-btn")
def _on_zoom_out(evt):
    ui_chrome.zoom_out()

@when("click", "#zoom-in-btn")
def _on_zoom_in(evt):
    ui_chrome.zoom_in()

@when("click", "#show-activation-btn")
def _on_toggle_activation(evt):
    ui_chrome.toggle_activation_visibility()

@when("change", "#debug-toggle")
def _on_debug_toggle(evt):
    state.debug_mode = evt.target.checked
    arrows.redraw_arrows()

@when("click", "#generate-data-btn")
def on_generate_data_click(evt=None):
    if state.is_generating_data:
        dataset_ui._stop_generate_data()
    else:
        dataset_ui._start_generate_data()

@when("click", "#clear-data-btn")
def on_clear_click(evt=None):
    dataset_ui.clear_data()

@when("input", "#crop-top-slider")
def _on_crop_top_input(evt):
    dataset_ui.on_crop_top_input(evt)

@when("input", "#crop-bottom-slider")
def _on_crop_bottom_input(evt):
    dataset_ui.on_crop_bottom_input(evt)

# ── Boot ───────────────────────────────────────────────────────────────────────

def boot():
    get_id("loading-splash").style.display = "none"
    get_id("page-wrap").style.display = "flex"

    check_browser_support()

    state.all_plots["plot-fit"] = fit_plot.FitPlot("plot-fit")
    state.all_plots["plot-loss"] = loss_plot.LossPlot("plot-loss")

    # one of each on load: 1 input, 1 layer (1 neuron + activation), 1 output
    network_actions.add_input()
    network_actions.add_layer()
    network_actions.add_output()

    ui_chrome.apply_zoom()
    ui_chrome.apply_activation_visibility()
    ui_chrome.setup_resize_observer()
    arrows.schedule_redraw(150)

boot()
