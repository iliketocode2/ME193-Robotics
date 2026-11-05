"""Small helpers shared by the page: tables and SVG charts (no matplotlib)."""

import math

import numpy as np
from pyodide.ffi import create_proxy
from pyscript import document


def el(id):
    return document.getElementById(id)


def val(id):
    return float(el(id).value)


def fmt(x, d=2):
    return f"{float(x):.{d}f}"


def readout(rows):
    """rows: [(label, value_text, 'ok'|'bad'|'')] -> the original page's key/value grid."""
    return "".join(f'<div class="k">{k}</div><div class="v {c}">{v}</div>' for k, v, c in rows)


def on(id, event, fn):
    el(id).addEventListener(event, create_proxy(fn))


def slider(id, text, initial, on_change):
    """Set a range input's initial value and label, and call on_change(value) when it moves.

    text(value) gives the string shown next to the slider.
    """
    inp, out = el(id), el(id + "-val")
    inp.value = initial
    out.textContent = text(initial)

    def handler(*_):
        v = float(inp.value)
        out.textContent = text(v)
        on_change(v)

    inp.addEventListener("input", create_proxy(handler))


def _ticks(lo, hi, n=5):
    span = hi - lo or 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), default=mag)
    first = math.ceil(lo / step - 1e-9) * step
    out, x = [], first
    while x <= hi + 1e-9:
        out.append(round(x, 10))
        x += step
    return out


def chart(series, xlabel, ylabel, hlines=(), vspan=None, ydomain=None, aspect=0.42, width=640):
    """Line chart as SVG.

    series: [dict(xs, ys, cls, label)] where cls is sim / real / mpc / lqrd / scope.
    hlines: y values drawn as dashed red lines (the sensor window).
    vspan: (x0, x1) shaded region (the turn).
    """
    height = round(width * aspect)
    L, R, T, B = 58, 12, 8, 34
    xs_all = [np.asarray(s["xs"], float) for s in series if len(s["xs"])]
    ys_all = [np.asarray(s["ys"], float) for s in series if len(s["ys"])]
    xs = np.concatenate(xs_all) if xs_all else np.array([0.0, 1.0])
    ys = np.concatenate(ys_all + [np.array(list(hlines), float)]) if ys_all or hlines else np.array([-1.0, 1.0])
    x0, x1 = float(xs.min()), float(xs.max())
    spans = [] if not vspan else (list(vspan) if isinstance(vspan[0], (tuple, list)) else [vspan])
    for a, b in spans:
        x0, x1 = min(x0, a), max(x1, b)
    if ydomain:
        y0, y1 = ydomain
    else:
        y0, y1 = float(ys.min()), float(ys.max())
        pad = (y1 - y0) * 0.15 or 1.0
        y0, y1 = y0 - pad, y1 + pad
    if x1 - x0 < 1e-6:
        x1 = x0 + 1

    def X(x): return L + (x - x0) / (x1 - x0) * (width - L - R)
    def Y(y): return T + (1 - (y - y0) / (y1 - y0)) * (height - T - B)

    p = [f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="{ylabel} against {xlabel}">']
    for sa, sb in spans:
        a, b = X(max(sa, x0)), X(min(sb, x1))
        p.append(f'<rect class="vspan" x="{a:.1f}" y="{T}" width="{max(0, b - a):.1f}" height="{height - T - B}"/>')
    for t in _ticks(y0, y1):
        p.append(f'<line class="gridline" x1="{L}" x2="{width - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in _ticks(x0, x1):
        p.append(f'<text x="{X(t):.1f}" y="{height - B + 16}" text-anchor="middle">{t:g}</text>')
    for y in hlines:
        p.append(f'<line class="limit" x1="{L}" x2="{width - R}" y1="{Y(y):.1f}" y2="{Y(y):.1f}"/>')
    p.append(f'<line class="axis" x1="{L}" x2="{L}" y1="{T}" y2="{height - B}"/>'
             f'<line class="axis" x1="{L}" x2="{width - R}" y1="{height - B}" y2="{height - B}"/>')
    for s in series:
        if len(s["xs"]):
            d = "M" + " L".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(s["xs"], s["ys"]))
            p.append(f'<path class="s {s["cls"]}" d="{d}"/>')
    p.append(f'<text x="{(L + width - R) / 2}" y="{height - 4}" text-anchor="middle">{xlabel}</text>'
             f'<text transform="translate(12,{(T + height - B) / 2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
    p.append("</svg>")
    labels = [(s["cls"], s["label"]) for s in series if s.get("label") and len(s["xs"])]
    legend = ""
    if len(labels) > 1 or any(c == "real" for c, _ in labels):
        legend = '<div class="legend">' + "".join(
            f'<span><i class="swatch {c}-c"></i>{t}</span>' for c, t in labels) + "</div>"
    return legend + "".join(p)


def track_plot(track, log, e_max):
    """Plan view: grey centerline, black robot path (deviation exaggerated), red dots where blind."""
    s_line, tx, ty, psi = track.geometry()
    exag = max(3.0, min(40.0, track.length * 0.05 / max(e_max, 1e-6)))
    s, e = log[:, 1], log[:, 2]
    xl, yl, pl = (np.interp(s, s_line, a) for a in (tx, ty, psi))
    px, py = xl - e * exag * np.sin(pl), yl + e * exag * np.cos(pl)
    blind = np.abs(e) >= e_max * 0.999
    width, height, pad = 640, 352, 24
    allx, ally = np.concatenate([tx, px]), np.concatenate([ty, py, [0.0]])
    x0, x1, y0, y1 = allx.min(), allx.max(), ally.min(), ally.max()
    scale = min((width - 2 * pad) / max(x1 - x0, 1), (height - 2 * pad) / max(y1 - y0, 1))
    ox = pad + ((width - 2 * pad) - scale * (x1 - x0)) / 2
    oy = pad + ((height - 2 * pad) - scale * (y1 - y0)) / 2

    def P(x, y): return ox + (x - x0) * scale, height - oy - (y - y0) * scale

    def path(xs, ys, step=1):
        return "M" + " L".join("%.1f,%.1f" % P(a, b) for a, b in zip(xs[::step], ys[::step]))

    out = [f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="Track plan view">',
           f'<path class="s trk" d="{path(tx, ty, 2)}"/>',
           f'<path class="s sim" d="{path(px, py)}"/>']
    for a, b, bl in zip(px, py, blind):
        if bl:
            cx, cy = P(a, b)
            out.append(f'<circle class="blinddot" cx="{cx:.1f}" cy="{cy:.1f}" r="1.8"/>')
    sx, sy = P(0, 0)
    out.append(f'<circle class="startdot" cx="{sx:.1f}" cy="{sy:.1f}" r="3.5"/>'
               f'<text x="{sx + 6:.1f}" y="{sy - 6:.1f}">start</text>'
               f'<text class="exag" x="6" y="14">deviation exaggerated {exag:.0f}×</text></svg>')
    return "".join(out)
