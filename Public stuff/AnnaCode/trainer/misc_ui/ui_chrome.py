"""Zoom controls (network diagram spacing/box sizes only) and the resize
observer that keeps arrows aligned when the layout shifts."""
from pyscript import window

import state
from state import get_id
import arrows

def apply_zoom():
    wrap = get_id("network-wrap")
    if wrap:
        wrap.style.setProperty("--zoom", str(state.zoom_level))
        # Text stays full-size down to the threshold, then shrinks in step
        # with everything else below it.
        if state.zoom_level >= state.TEXT_ZOOM_THRESHOLD:
            text_zoom = 1.0
        else:
            text_zoom = state.zoom_level / state.TEXT_ZOOM_THRESHOLD
        wrap.style.setProperty("--text-zoom", str(round(text_zoom, 3)))

    label = get_id("zoom-level-label")
    if label:
        label.textContent = f"{round(state.zoom_level * 100)}%"

    out_btn = get_id("zoom-out-btn")
    if out_btn:
        if state.zoom_level <= state.ZOOM_MIN + 1e-9:
            out_btn.setAttribute("disabled", "")
        else:
            out_btn.removeAttribute("disabled")

    in_btn = get_id("zoom-in-btn")
    if in_btn:
        if state.zoom_level >= state.ZOOM_MAX - 1e-9:
            in_btn.setAttribute("disabled", "")
        else:
            in_btn.removeAttribute("disabled")

    # Layout changed -- re-measure anchor points once the browser has
    # applied the new spacing/sizes.
    arrows.schedule_redraw(60)

def zoom_out(evt=None):
    state.zoom_level = round(max(state.ZOOM_MIN, state.zoom_level - state.ZOOM_STEP), 2)
    apply_zoom()

def zoom_in(evt=None):
    state.zoom_level = round(min(state.ZOOM_MAX, state.zoom_level + state.ZOOM_STEP), 2)
    apply_zoom()

def apply_activation_visibility():
    wrap = get_id("network-wrap")
    if wrap:
        wrap.classList.toggle("show-activation", state.show_activation)

    btn = get_id("show-activation-btn")
    if btn:
        btn.textContent = "− Activation function" if state.show_activation else "+ Activation function"

    arrows.schedule_redraw(60)

def toggle_activation_visibility(evt=None):
    state.show_activation = not state.show_activation
    apply_activation_visibility()

def setup_resize_observer():
    from pyscript.ffi import create_proxy
    wrap = get_id("network-wrap")
    if not wrap:
        return
    def on_resize(entries, observer):
        arrows.redraw_arrows()
    observer = window.ResizeObserver.new(create_proxy(on_resize))
    observer.observe(wrap)
