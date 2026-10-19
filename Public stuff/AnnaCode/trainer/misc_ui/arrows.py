"""SVG arrow primitives + the main network-diagram redraw."""
import math
from pyscript import document, window
from pyscript.ffi import create_proxy

import state
from state import get_id

def svg_path(d: str, stroke_w: float = 2.0):
    p = document.createElementNS("http://www.w3.org/2000/svg", "path")
    p.setAttribute("d", d)
    p.setAttribute("fill", "none")
    p.setAttribute("stroke", state.ARROW_COLOR)
    p.setAttribute("stroke-width", str(stroke_w))
    p.setAttribute("stroke-linecap", "round")
    p.setAttribute("stroke-linejoin", "round")
    return p

def arrowhead(svg_el, x2: float, y2: float, dx: float, dy: float,
              stroke_w: float = 2.0, size: float = 9.0):
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-9:
        return
    ux, uy = dx / mag, dy / mag
    spread = math.radians(26)
    cs, sn = math.cos(spread), math.sin(spread)
    wx1, wy1 =  ux * cs - uy * sn,  ux * sn + uy * cs
    wx2, wy2 =  ux * cs + uy * sn, -ux * sn + uy * cs
    ax1, ay1 = x2 - size * wx1, y2 - size * wy1
    ax2, ay2 = x2 - size * wx2, y2 - size * wy2
    d = f"M{x2:.2f},{y2:.2f} L{ax1:.2f},{ay1:.2f} M{x2:.2f},{y2:.2f} L{ax2:.2f},{ay2:.2f}"
    svg_el.appendChild(svg_path(d, stroke_w))

def debug_label(svg_el, x: float, y: float, value: float):
    text = f"{value:.2f}"
    char_w, pad_x, h = 6.2, 6, 18
    w = len(text) * char_w + pad_x * 2

    rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
    rect.setAttribute("x", str(x - w / 2))
    rect.setAttribute("y", str(y - h / 2))
    rect.setAttribute("width", str(w))
    rect.setAttribute("height", str(h))
    rect.setAttribute("rx", "5")
    rect.setAttribute("fill", "#1a1d2e")
    rect.setAttribute("stroke", "#ffffff")
    rect.setAttribute("stroke-width", "1")

    txt = document.createElementNS("http://www.w3.org/2000/svg", "text")
    txt.setAttribute("x", str(x))
    txt.setAttribute("y", str(y + 3.5))
    txt.setAttribute("text-anchor", "middle")
    txt.setAttribute("font-family", "JetBrains Mono, monospace")
    txt.setAttribute("font-size", "10")
    txt.setAttribute("font-weight", "700")
    txt.setAttribute("fill", "#ffffff")
    txt.textContent = text

    svg_el.appendChild(rect)
    svg_el.appendChild(txt)

def straight_arrow(svg_el, x1, y1, x2, y2, stroke_w: float = 2.0, debug_value=None,
                    label_t: float = 0.5, label_side: float = 1.0):
    d = f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}"
    svg_el.appendChild(svg_path(d, stroke_w))
    arrowhead(svg_el, x2, y2, x2 - x1, y2 - y1, stroke_w)

    if state.debug_mode and debug_value is not None:
        mx, my = x1 + (x2 - x1) * label_t, y1 + (y2 - y1) * label_t
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy) or 1.0
        px, py = -dy / length, dx / length
        offset = 13 * label_side
        debug_label(svg_el, mx + px * offset, my + py * offset, debug_value)

def fan_arrows(svg_el, src_pts, tgt_pts, src_heights, tgt_heights, value_fn):
    """Bipartite fan of straight arrows from every src point to every tgt point,
    spread vertically at both ends so lines don't all overlap."""
    n_src, n_tgt = len(src_pts), len(tgt_pts)
    SPREAD_FRAC = 0.55
    MAX_STEP = 14

    for src_idx, (sx, sy, sid) in enumerate(src_pts):
        for tgt_idx, (tx, ty, tid) in enumerate(tgt_pts):
            if n_tgt > 1:
                src_h    = src_heights.get(sid, 120)
                spread_h = min(src_h * SPREAD_FRAC, (n_tgt - 1) * MAX_STEP)
                tail_y   = sy - spread_h / 2 + (tgt_idx / (n_tgt - 1)) * spread_h
            else:
                tail_y = sy

            if n_src > 1:
                tgt_h    = tgt_heights.get(tid, 80)
                spread_h = min(tgt_h * SPREAD_FRAC, (n_src - 1) * MAX_STEP)
                head_y   = ty - spread_h / 2 + (src_idx / (n_src - 1)) * spread_h
            else:
                head_y = ty

            # Fan lines cross each other in the middle of the gap. Stagger each
            # line's debug label along its own length (and alternate which side
            # it sits on) so labels on crossing lines don't land on top of
            # each other.
            t_step   = tgt_idx - (n_tgt - 1) / 2
            label_t  = 0.5 + (t_step / max(n_tgt, 1)) * 0.6
            label_t  = min(max(label_t, 0.15), 0.85)
            label_side = 1.0 if (src_idx + tgt_idx) % 2 == 0 else -1.0

            straight_arrow(svg_el, sx, tail_y, tx, head_y, stroke_w=1.8,
                           debug_value=value_fn(sid) if value_fn else None,
                           label_t=label_t, label_side=label_side)

def _anchor_points(id_prefix: str, item_list: list, id_key: str = "id"):
    """Returns (left_pts, right_pts, heights) for a list of items whose DOM
    cell has id f'{id_prefix}{item[id_key]}'."""
    wrap_el = get_id("network-wrap")
    wr = wrap_el.getBoundingClientRect()
    left_pts, right_pts, heights = [], [], {}
    for item in item_list:
        iid = item[id_key]
        el = get_id(f"{id_prefix}{iid}")
        if el:
            r = el.getBoundingClientRect()
            x = r.left - wr.left
            y = r.top - wr.top + r.height / 2
            left_pts.append((x, y, iid))
            right_pts.append((x + r.width, y, iid))
            heights[iid] = r.height
    return left_pts, right_pts, heights

def redraw_arrows():
    svg_el  = get_id("arrow-svg")
    wrap_el = get_id("network-wrap")
    if (not svg_el or not wrap_el or not state.inputs or not state.layers
            or not state.layers[0]["neurons"] or not state.outputs):
        if svg_el:
            svg_el.innerHTML = ""
        return

    svg_el.innerHTML = ""

    in_l_pts, in_r_pts, in_heights = _anchor_points("cell-input-", state.inputs)
    out_l_pts, _, _                = _anchor_points("cell-output-", state.outputs)

    if not in_r_pts or not out_l_pts:
        return

    def rel_box(el):
        wr = wrap_el.getBoundingClientRect()
        r = el.getBoundingClientRect()
        return {"x": r.left - wr.left, "y": r.top - wr.top, "w": r.width, "h": r.height}

    # 1. Input -> first layer's neurons (fan)
    layer0_l_pts, layer0_r_pts, layer0_heights = _anchor_points("cell-neuron-", state.layers[0]["neurons"])
    if not layer0_l_pts:
        return
    fan_arrows(svg_el, in_r_pts, layer0_l_pts, in_heights, layer0_heights,
               value_fn=lambda iid: state.input_values.get(iid))

    prev_r_pts, prev_heights = layer0_r_pts, layer0_heights

    for idx, layer in enumerate(state.layers):
        lid = layer["id"]
        neuron_l_pts, neuron_r_pts, neuron_heights = _anchor_points("cell-neuron-", layer["neurons"])
        if not neuron_l_pts:
            return

        if state.show_activation:
            act_el = get_id(f"act-box-{lid}")
            if not act_el:
                continue
            box = rel_box(act_el)

            # neurons_i -> act_i (converge, 1:1 at each neuron's own y)
            for (nx, ny, nid) in neuron_r_pts:
                straight_arrow(svg_el, nx, ny, box["x"], ny, stroke_w=2.0,
                               debug_value=state.neuron_pre_values.get(nid))

            act_rx = box["x"] + box["w"]
            src_pts = [(act_rx, ny, nid) for (nx, ny, nid) in neuron_r_pts]
        else:
            # Activation column hidden: skip straight from each neuron.
            src_pts = neuron_r_pts

        if idx + 1 < len(state.layers):
            # src -> neurons_{i+1} (fan: every neuron feeds every next neuron)
            next_l_pts, _, next_heights = _anchor_points("cell-neuron-", state.layers[idx + 1]["neurons"])
            fan_arrows(svg_el, src_pts, next_l_pts, neuron_heights, next_heights,
                       value_fn=lambda nid: state.neuron_post_values.get(nid))
        elif state.show_activation:
            # last layer -> outputs (1:1 diverge, matching positional pairing)
            for (ox, oy, oid) in out_l_pts:
                straight_arrow(svg_el, act_rx, oy, ox, oy, stroke_w=2.0,
                               debug_value=state.output_values.get(oid))
        else:
            # last layer -> outputs, no activation box to diverge from: fan
            # from the neuron column's shared right edge (mirrors the
            # activation-box-edge behavior above), so this still covers
            # every output even if outputs currently outnumber neurons.
            neuron_rx = max(nx for (nx, ny, nid) in src_pts)
            for (ox, oy, oid) in out_l_pts:
                straight_arrow(svg_el, neuron_rx, oy, ox, oy, stroke_w=2.0,
                               debug_value=state.output_values.get(oid))

def schedule_redraw(delay=60):
    window.setTimeout(create_proxy(lambda: redraw_arrows()), delay)
