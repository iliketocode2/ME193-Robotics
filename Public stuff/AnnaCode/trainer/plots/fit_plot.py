from collections import deque
from pyscript.js_modules import Plotly

from plot_utils import to_js

DATA_TRACE_COLORS = ["#17e262", '#0891b2', "#99c2ae", '#84cc16', '#14b8a6', '#65a30d']
LIVE_TRACE_COLORS = ['#f59e0b', '#ec4899', '#f97316', '#eab308', '#fb7185', '#fbbf24']

class FitPlot:
    """Fit graph for dataset-based training, x-axis fixed to the FIRST
    input. Each OUTPUT gets its own pair of traces:
      Data trace (markers) -- that output's raw (first-input x, y) dataset
                               points, set whenever the dataset table
                               changes.
      Live trace (markers) -- that output's ACTUAL live (input, output)
                               pairs, sampled once per forward() tick
                               while Play is running -- not a synthetic
                               sweep -- so what's drawn is a true
                               reflection of what the running network
                               produces, comparable against the Data trace.
    Trace pairs are appended/removed dynamically as outputs are added to
    or removed from the network (see add_output_trace/remove_output_trace)."""

    def __init__(self, id, max_run_points=500):
        self.id = id
        self.max_run_points = max_run_points
        self.output_order = []   # oid, in trace-append order; oid's Data trace is at index 2*pos, Live at 2*pos+1
        self.run_data = {}       # oid -> {"xs": deque, "ys": deque}
        layout = to_js({
            "margin": {"l": 36, "r": 14, "t": 10, "b": 30},
            "xaxis": {"tickfont": {"size": 10}, "zeroline": True},
            "yaxis": {"tickfont": {"size": 10}, "autorange": True, "zeroline": True},
            "showlegend": True,
            "legend": {"font": {"size": 10}, "x": 0, "y": 1}
        })
        Plotly.newPlot(self.id, to_js([]), layout)

    def _pos(self, oid):
        return self.output_order.index(oid)

    def add_output_trace(self, oid, label=None):
        if oid in self.output_order:
            return
        pos = len(self.output_order)
        color = DATA_TRACE_COLORS[pos % len(DATA_TRACE_COLORS)]
        live_color = LIVE_TRACE_COLORS[pos % len(LIVE_TRACE_COLORS)]
        name = label or f"y{pos + 1}"
        data_trace = {'x': [], 'y': [], 'mode': 'markers', 'type': 'scatter',
                      'name': f'{name} data', 'marker': {'color': color, 'size': 9}}
        live_trace = {'x': [], 'y': [], 'mode': 'markers', 'type': 'scatter',
                      'name': f'{name} live', 'marker': {'color': live_color, 'size': 6, 'opacity': 0.75}}
        Plotly.addTraces(self.id, to_js([data_trace, live_trace]))
        self.output_order.append(oid)
        self.run_data[oid] = {"xs": deque([], self.max_run_points), "ys": deque([], self.max_run_points)}

    def remove_output_trace(self, oid):
        if oid not in self.output_order:
            return
        pos = self._pos(oid)
        Plotly.deleteTraces(self.id, to_js([pos * 2, pos * 2 + 1]))
        self.output_order.pop(pos)
        self.run_data.pop(oid, None)

    def update_data_points(self, oid, xs, ys):
        if oid not in self.output_order:
            return
        Plotly.restyle(self.id, to_js({'x': [list(xs)], 'y': [list(ys)]}),
                        to_js([self._pos(oid) * 2]))

    def reset_run(self):
        for oid in self.output_order:
            rd = self.run_data[oid]
            rd["xs"].clear()
            rd["ys"].clear()
            Plotly.restyle(self.id, to_js({'x': [[]], 'y': [[]]}),
                            to_js([self._pos(oid) * 2 + 1]))

    def add_run_point(self, oid, x, y):
        rd = self.run_data.get(oid)
        if rd is None:
            return
        rd["xs"].append(x)
        rd["ys"].append(y)
        Plotly.restyle(self.id, to_js(
            {'x': [list(rd["xs"])], 'y': [list(rd["ys"])]}), to_js([self._pos(oid) * 2 + 1]))
