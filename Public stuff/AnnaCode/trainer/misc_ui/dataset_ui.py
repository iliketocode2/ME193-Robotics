"""Training-data table rendering/binding. Data mutation itself
(add_data_point/remove_data_point) lives in network_model.py -- this file
only renders/binds the table and re-renders after each mutation."""
import asyncio
from pyscript.ffi import create_proxy

import state
from state import get_id
import network_model
import templates
import Device

def render_dataset_header():
    container = get_id("dataset-header")
    if not container:
        return
    html = ""
    for inp in state.inputs:
        html += f'<span class="dataset-row-header-x">{inp["name"]}</span>'
    for idx, out in enumerate(state.outputs):
        html += f'<span class="dataset-row-header-y">y{idx + 1}</span>'
    html += "<span></span>"
    container.innerHTML = html

def _crop_bounds() -> tuple[int, int]:
    """(start, end): inclusive indices into state.training_data that stay
    visible on the fit plot; everything outside is cropped (kept in the
    table, highlighted red). Clamps state.dataset_crop_start/_end against
    the dataset's current size so cropping stays valid as points are added
    or removed."""
    n = len(state.training_data)
    if n == 0:
        return 0, -1
    start = max(0, min(state.dataset_crop_start, n - 1))
    end = (n - 1) if state.dataset_crop_end is None else max(0, min(state.dataset_crop_end, n - 1))
    if start > end:
        start = end
    return start, end

CROP_THUMB_LEN = 30   # must match the ::-webkit-slider-thumb / ::-moz-range-thumb width in styles.css

def _cap_dataset_card_height():
    """Cap the dataset card's height at the training card's (the fit/loss
    graphs) own rendered height -- never the other way around -- so a
    growing dataset scrolls internally (see .dataset-body's overflow-y)
    instead of pushing the card taller than the graphs next to it."""
    card = get_id("dataset-card")
    training_card = get_id("training-card")
    if card and training_card:
        card.style.maxHeight = f"{training_card.offsetHeight}px"

def render_crop_sliders():
    """Size the crop bar to match the dataset-rows column's actual rendered
    height (so one slider step lines up with one table row) and repaint the
    red/green segments to reflect the current crop window.

    Hidden entirely while the dataset is empty (nothing to crop yet)."""
    _cap_dataset_card_height()
    n = len(state.training_data)
    panel = get_id("dataset-crop-panel")
    if panel:
        if n > 0:
            panel.classList.remove("hidden")
        else:
            panel.classList.add("hidden")
    if n == 0:
        return

    start, end = _crop_bounds()
    max_idx = max(n - 1, 0)

    rows_el = get_id("dataset-rows")
    bar_h = rows_el.offsetHeight if rows_el else 0
    bar_h = max(bar_h, 24)

    bar = get_id("crop-bar")
    top = get_id("crop-top-slider")
    bottom = get_id("crop-bottom-slider")
    seg_top = get_id("crop-seg-top")
    seg_bottom = get_id("crop-seg-bottom")

    if bar:
        bar.style.height = f"{bar_h}px"

    # One index step should move exactly one row's worth of pixels. The bar
    # is n rows tall, but there are only max_idx steps between the n index
    # positions, so the per-step spacing that actually matches a row is
    # bar_h/n (NOT bar_h/max_idx, which overshoots -- badly so for a small
    # n, e.g. 1.5x for n=3). row_slot * max_idx is therefore the pixel span
    # the slider's value range should cover, short of the full bar by
    # exactly one row_slot.
    row_slot = bar_h / n
    travel = max_idx * row_slot

    # The native thumb's center never travels past half its own length from
    # either end of the input's box, so a slider exactly `travel` long would
    # leave a gap at each end where the handle can't reach. Padding the
    # slider's length by one thumb-length cancels that inset out.
    box_len = travel + CROP_THUMB_LEN

    # Centered on the bar (the CSS default), a box this long falls short of
    # the bar's actual top/bottom by row_slot/2 on each side (bar_h - travel
    # == row_slot). Nudging the top slider up and the bottom slider down by
    # that same row_slot/2 lands each one's resting end exactly on the
    # bar's edge, while both keep the identical per-step spacing (row_slot).
    half_slot = row_slot / 2
    for slider, top_css in ((top, f"calc(50% - {half_slot}px)"),
                             (bottom, f"calc(50% + {half_slot}px)")):
        if slider:
            slider.style.width = f"{box_len}px"
            slider.style.top = top_css

    if top:
        top.max = str(max_idx)
        top.value = str(start)
        top.disabled = n <= 1
    if bottom:
        bottom.max = str(max_idx)
        bottom.value = str(end)
        bottom.disabled = n <= 1

    if seg_top:
        seg_top.style.height = f"{start * row_slot}px"
    if seg_bottom:
        seg_bottom.style.height = f"{(max_idx - end) * row_slot}px"

def on_crop_top_input(evt):
    n = len(state.training_data)
    if n == 0:
        return
    _, end = _crop_bounds()
    try:
        val = int(evt.target.value)
    except (ValueError, TypeError):
        return
    state.dataset_crop_start = min(max(val, 0), end)
    render_dataset_table()
    refresh_dataset_plot_points()

def on_crop_bottom_input(evt):
    n = len(state.training_data)
    if n == 0:
        return
    start, _ = _crop_bounds()
    try:
        val = int(evt.target.value)
    except (ValueError, TypeError):
        return
    val = max(val, start)
    state.dataset_crop_end = None if val >= n - 1 else val
    render_dataset_table()
    refresh_dataset_plot_points()

def render_dataset_table():
    container = get_id("dataset-rows")
    if not container:
        return
    start, end = _crop_bounds()
    html = ""
    for idx, p in enumerate(state.training_data):
        pid = p["id"]
        cropped = idx < start or idx > end
        row_cls = "dataset-row dataset-row-cropped" if cropped else "dataset-row"
        html += f'<div class="{row_cls}" data-id="{pid}">'
        for inp in state.inputs:
            iid = inp["id"]
            val = p["xs"].get(iid, 0.0)
            html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-x" '
                      f'id="point-x-{pid}-{iid}" value="{val}" />')
        for out in state.outputs:
            oid = out["id"]
            val = p["ys"].get(oid, 0.0)
            html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-y" '
                      f'id="point-y-{pid}-{oid}" value="{val}" />')
        html += f'<button class="btn-remove-point" id="del-point-{pid}" title="Remove point">{templates.delete_x_svg()}</button>'
        html += "</div>"
    container.innerHTML = html
    for p in state.training_data:
        bind_dataset_row_events(p["id"])
    render_crop_sliders()

def bind_dataset_row_events(pid: int):
    for inp in state.inputs:
        iid = inp["id"]
        x_el = get_id(f"point-x-{pid}-{iid}")
        if x_el:
            def hx(evt, pid=pid, iid=iid):
                p = next((pt for pt in state.training_data if pt["id"] == pid), None)
                try:
                    if p:
                        p["xs"][iid] = float(evt.target.value)
                        refresh_dataset_plot_points()
                except (ValueError, TypeError):
                    pass
            x_el.addEventListener("input", create_proxy(hx))

    for out in state.outputs:
        oid = out["id"]
        y_el = get_id(f"point-y-{pid}-{oid}")
        if y_el:
            def hy(evt, pid=pid, oid=oid):
                p = next((pt for pt in state.training_data if pt["id"] == pid), None)
                try:
                    if p:
                        p["ys"][oid] = float(evt.target.value)
                        refresh_dataset_plot_points()
                except (ValueError, TypeError):
                    pass
            y_el.addEventListener("input", create_proxy(hy))

    del_btn = get_id(f"del-point-{pid}")
    if del_btn:
        del_btn.addEventListener("click", create_proxy(lambda evt, pid=pid: _on_remove_point(pid)))

def _on_remove_point(pid: int):
    network_model.remove_data_point(pid)
    render_dataset_table()
    refresh_dataset_plot_points()

def refresh_dataset_plot_points():
    """Push each output's Data trace: y = that output's column, x = the
    FIRST input's column (the fit graph's x-axis is fixed to it). Points
    outside the crop window (see _crop_bounds) are left out of the plot
    entirely even though they remain in state.training_data."""
    fit_plot_obj = state.all_plots.get("plot-fit")
    if not fit_plot_obj or not state.inputs:
        return
    start, end = _crop_bounds()
    visible = state.training_data[start:end + 1] if end >= start else []
    first_iid = state.inputs[0]["id"]
    xs = [p["xs"].get(first_iid, 0.0) for p in visible]
    for out in state.outputs:
        oid = out["id"]
        ys = [p["ys"].get(oid, 0.0) for p in visible]
        fit_plot_obj.update_data_points(oid, xs, ys)

def render_add_point_row():
    container = get_id("dataset-add-row")
    if not container:
        return
    html = ""
    for inp in state.inputs:
        iid = inp["id"]
        html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-x" '
                  f'id="new-point-x-{iid}" placeholder="{inp["name"]}" />')
    for idx, out in enumerate(state.outputs):
        oid = out["id"]
        html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-y" '
                  f'id="new-point-y-{oid}" placeholder="y{idx + 1}" />')
    html += ('<button class="btn-add-point" id="add-point-btn" title="Add this point to the dataset">'
             '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="2.5"><path d="M12 5v14M5 12h14"/></svg>'
             'Add point</button>')
    container.innerHTML = html
    btn = get_id("add-point-btn")
    if btn:
        btn.addEventListener("click", create_proxy(on_add_point_click))

def on_add_point_click(evt=None):
    xs = {}
    for inp in state.inputs:
        iid = inp["id"]
        el = get_id(f"new-point-x-{iid}")
        try:
            xs[iid] = float(el.value) if el and el.value.strip() != "" else None
        except ValueError:
            xs[iid] = None

    ys = {}
    for out in state.outputs:
        oid = out["id"]
        el = get_id(f"new-point-y-{oid}")
        try:
            ys[oid] = float(el.value) if el and el.value.strip() != "" else None
        except ValueError:
            ys[oid] = None

    if any(v is None for v in xs.values()) or any(v is None for v in ys.values()):
        print("Enter every x and y value before adding a point.")
        return

    network_model.add_data_point(xs, ys)
    render_dataset_table()
    refresh_dataset_plot_points()

    for inp in state.inputs:
        el = get_id(f"new-point-x-{inp['id']}")
        if el:
            el.value = ""
    for out in state.outputs:
        el = get_id(f"new-point-y-{out['id']}")
        if el:
            el.value = ""

def _read_device_value(dev_id: str, chan_id: str) -> float:
    """Read the live value currently sitting at a device/channel pair,
    same lookup compute_forward() uses for the raw inputs."""
    dev_el = get_id(dev_id)
    chan_el = get_id(chan_id)
    dev_name = dev_el.value if dev_el else ""
    channel = chan_el.value if chan_el else ""
    dev = Device.device_by_name(dev_name)
    try:
        return float(dev.state[channel]) if dev and channel else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0

def _start_generate_data():
    state.is_generating_data = True
    btn = get_id("generate-data-btn")
    if btn:
        btn.textContent = "Stop"
        btn.classList.add("is-active")
    asyncio.ensure_future(_generate_data_loop())

def _stop_generate_data():
    state.is_generating_data = False
    btn = get_id("generate-data-btn")
    if btn:
        btn.textContent = "Collect"
        btn.classList.remove("is-active")

def clear_data():
    for p in range(len(state.training_data)):
        state.training_data.pop(0)
    state.dataset_crop_start = 0
    state.dataset_crop_end = None
    render_dataset_table()
    refresh_dataset_plot_points()

async def _generate_data_loop():
    while state.is_generating_data:
        xs = {inp["id"]: _read_device_value(f"dev-input-{inp['id']}", f"chan-input-{inp['id']}") for inp in state.inputs}
        ys = {out["id"]: _read_device_value(f"dev-output-{out['id']}", f"chan-output-{out['id']}") for out in state.outputs}
        network_model.add_data_point(xs, ys)
        render_dataset_table()
        refresh_dataset_plot_points()
        await asyncio.sleep(0.1)
