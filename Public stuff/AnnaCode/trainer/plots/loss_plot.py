from collections import deque
from pyscript.js_modules import Plotly

from plot_utils import to_js

class LossPlot:
    """Simple scrolling line plot for loss-vs-epoch during training."""

    def __init__(self, id, max_points=500):
        self.id = id
        self.max_points = max_points
        self.ys = deque([], max_points)
        self.traces = to_js([
            {'y': [], 'mode': 'lines', 'name': 'Loss', 'line': {'color': '#d97706'}}
        ])
        layout = to_js({
            "margin": {"l": 40, "r": 10, "t": 10, "b": 26},
            "xaxis": {"title": {"text": "epoch", "font": {"size": 10}}, "tickfont": {"size": 9}},
            "yaxis": {"tickfont": {"size": 9}, "autorange": True},
            "showlegend": False
        })
        Plotly.newPlot(self.id, self.traces, layout)

    def reset(self):
        self.ys.clear()
        self.updatePlot(to_js({'y': [[]]}))

    def add_point(self, loss):
        self.ys.append(loss)
        return to_js({'y': [list(self.ys)]})

    def updatePlot(self, update):
        Plotly.update(self.id, update)
