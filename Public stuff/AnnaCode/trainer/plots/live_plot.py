from collections import deque
from pyscript.js_modules import Plotly

from plot_utils import to_js

class LivePlot:
    def __init__(self, id):
        self.size = 100
        self.num = 12
        self.id = id
        self.buffer = []
        self.traces = to_js([
            {'y': 0, 'mode': 'lines', 'name': 'Element Data', 'line': {'color': '#82ca9d'}}
        ])
        for n in range(self.num):
            self.buffer.append(deque([], self.size))
        layout = to_js({
            "margin": {"l": 20, "r": 10, "t": 10, "b": 20},
            "xaxis": {
                "title": {"font": {"size": 10}},
                "tickfont": {"size": 9},
                "showticklabels": False,
                # Fixed 0..size-1 window instead of autorange -- otherwise
                # the line stretches/compresses as the buffer fills from
                # empty to full, making early readings look misleadingly
                # spiky regardless of the actual data.
                "range": [0, self.size - 1],
                "autorange": False
            },
            "yaxis": {
                "tickfont": {"size": 9},
                "autorange": True
            },
            "showlegend": False
        })
        Plotly.newPlot(self.id, self.traces, layout)

    def addPoints(self, num, data):
        lines = []
        num = (num - 1) * 2
        for i, d in enumerate(data):
            self.buffer[num + i].append(d)
            lines.append(list(self.buffer[num + i]))
        return to_js({'y': lines})

    def updatePlot(self, update):
        Plotly.update(self.id, update)
