"""
AnnaCode — zoom.py

Zoom in/out for the network diagram (spacing/box sizes only, plus text
scaling below TEXT_ZOOM_THRESHOLD) and the resize observer that keeps
arrows aligned when the layout changes.
"""
from pyscript import window
from pyscript.ffi import create_proxy

import state
import arrows

def apply_zoom():
    wrap = state.get_id("network-wrap")
    if wrap:
        wrap.style.setProperty("--zoom", str(state.zoom_level))
        # Text stays full-size down to the threshold, then shrinks in step
        # with everything else below it.
        if state.zoom_level >= state.TEXT_ZOOM_THRESHOLD:
            text_zoom = 1.0
        else:
            text_zoom = state.zoom_level / state.TEXT_ZOOM_THRESHOLD
        wrap.style.setProperty("--text-zoom", str(round(text_zoom, 3)))

    label = state.get_id("zoom-level-label")
    if label:
        label.textContent = f"{round(state.zoom_level * 100)}%"

    out_btn = state.get_id("zoom-out-btn")
    if out_btn:
        if state.zoom_level <= state.ZOOM_MIN + 1e-9:
            out_btn.setAttribute("disabled", "")
        else:
            out_btn.removeAttribute("disabled")

    in_btn = state.get_id("zoom-in-btn")
    if in_btn:
        if state.zoom_level >= state.ZOOM_MAX - 1e-9:
            in_btn.setAttribute("disabled", "")
        else:
            in_btn.removeAttribute("disabled")

    # Layout changed -- re-measure anchor points once the browser has
    # applied the new spacing/sizes.
    arrows.schedule_redraw()

def zoom_out(evt=None):
    state.zoom_level = round(max(state.ZOOM_MIN, state.zoom_level - state.ZOOM_STEP), 2)
    apply_zoom()

def zoom_in(evt=None):
    state.zoom_level = round(min(state.ZOOM_MAX, state.zoom_level + state.ZOOM_STEP), 2)
    apply_zoom()

def setup_resize_observer():
    wrap = state.get_id("network-wrap")
    if not wrap:
        return
    def on_resize(entries, observer):
        arrows.redraw_arrows()
    observer = window.ResizeObserver.new(create_proxy(on_resize))
    observer.observe(wrap)
