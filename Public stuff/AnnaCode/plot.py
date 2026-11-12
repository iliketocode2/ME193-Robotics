from pyscript import document, window, when, Event
from pyscript.js_modules import Plotly
from collections import deque
import js
import json

class plot():
    def __init__(self, id):
        self.size = 100
        self.num = 12
        self.id = id
        self.buffer = []
        self.traces = js.JSON.parse(json.dumps([
            {'y': 0, 'mode': 'lines', 'name': 'Element Data', 'line': {'color': '#82ca9d'}}
        ]))
        for n in range(self.num):
            self.buffer.append(deque([], self.size))
        layout = js.JSON.parse(json.dumps({
            "margin": {"l": 20, "r": 10, "t": 10, "b": 20},
            "xaxis": {
                "title": {"font": {"size": 10}},
                "tickfont": {"size": 9},
                "showticklabels": False
            },
            "yaxis": {
                "tickfont": {"size": 9},
                "autorange": True
            },
            "showlegend": False
        }))
        Plotly.newPlot(self.id, self.traces, layout)

    def addPoints(self, num, data):
        lines = []
        num = (num - 1) * 2
        for i, d in enumerate(data):
            self.buffer[num + i].append(d)
            lines.append(list(self.buffer[num + i]))
        return js.JSON.parse(json.dumps({'y': lines}))

    def updatePlot(self, update):
        Plotly.update(self.id, update)