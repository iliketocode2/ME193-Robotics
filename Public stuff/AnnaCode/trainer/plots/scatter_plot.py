from collections import deque
from pyscript.js_modules import Plotly

from plot_utils import to_js

class ScatterPlot:
    """Persistent (x, y) marker plot -- unlike LivePlot (a scrolling
    line over an implicit time axis), points here accumulate and stay put
    so a set of test inputs/targets can be compared against the network's
    fitted curve."""

    def __init__(self, id, max_points=300):
        self.id = id
        self.max_points = max_points
        self.xs = deque([], max_points)
        self.ys = deque([], max_points)
        self.traces = to_js([
            {'x': [], 'y': [], 'mode': 'markers', 'type': 'scatter',
             'name': 'Test points', 'marker': {'color': "#476f7d", 'size': 8}}
        ])
        layout = to_js({
            "margin": {"l": 28, "r": 10, "t": 10, "b": 22},
            "xaxis": {"tickfont": {"size": 9}, "zeroline": True},
            "yaxis": {"tickfont": {"size": 9}, "autorange": True, "zeroline": True},
            "showlegend": False
        })
        Plotly.newPlot(self.id, self.traces, layout)

    def addPoint(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        return to_js({'x': [list(self.xs)], 'y': [list(self.ys)]})

    def addPoints(self, num, data):
        # Compatibility no-op: an output's device/channel hookup still routes
        # live hardware notifications here (see Device.py's notif_callback),
        # but this plot is meant to only move on explicit addPoint() calls
        # from Test(), so continuous notifications are ignored.
        return to_js({})

    def updatePlot(self, update):
        Plotly.update(self.id, update)

    def clear(self):
        self.xs.clear()
        self.ys.clear()
        self.updatePlot(to_js({'x': [[]], 'y': [[]]}))
