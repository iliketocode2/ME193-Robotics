# Q-Learning Visualizer (LEGO Education, browser/PyScript)

A browser-based app that teaches Q-learning by having students train a LEGO
Education robot (Double Motor hub + optional Color Sensor hub) over Web
Bluetooth. Everything runs client-side via [PyScript](https://pyscript.net)
(Python compiled to WASM via Pyodide) — there is no server or build step;
opening `index.html` in a Bluetooth-capable browser (Chrome/Edge) is enough.

The app has three tabs, each a self-contained Q-learning task driving the
same physical robot:

- **Silly Walk** — a tabular Q-learning demo. The robot picks LEFT/RIGHT
  (optionally + FORWARD) turns to try to walk in a straight line, using the
  Double Motor's IMU yaw reading as state.
- **Line Follower** — fixed 3-state/3-action Q-table. Uses the Color
  Sensor's reflection reading to stay on a line.
- **Maze Solver** — fixed 11-state/3-action Q-table combining Color Sensor
  color (tape color) and IMU yaw-drift into one state, to learn a path
  through a maze to a green goal tile.

All three share the same sidebar controls (Connect/Disconnect, Start
Training, Stop, Step Once, Run Learned Policy) and live Q-table
visualization, Bellman-update readout, and yaw/reflection monitor modals.

## Files

### `index.html`
The entire UI: layout, styling, the three tab pages, and a `<script>` block
of small DOM-manipulation helpers exposed on `window` (table rebuilding,
cell coloring, transition-arrow drawing, the yaw compass / reflection gauge
modals, tooltips, panel collapsing). These functions are called from Python
(`main.py`) to reflect Q-learning state into the page — the JS itself holds
no training logic. It also loads PyScript itself and points it at
`main.py`/`pyscript.toml`.

### `main.py`
The application logic, run in-browser as Python via PyScript. Responsibilities:

- **BLE device wiring** — `WebDevice`/`DoubleMotorDevice`/`ColorSensorDevice`
  wrap the `legoeducation` package's `DoubleMotor`/`ColorSensor` classes and
  bridge their internal BLE worker (normally a background thread + `bleak`)
  to the single-threaded Pyodide runtime, routing actual sends/notifications
  through `ble.js`'s Web Bluetooth wrapper instead.
- **UI event handlers** — every `py-click` target in `index.html` (connect/
  disconnect, apply settings, reset Q-table, start/stop/step training, run
  policy, yaw/reflection modals) lives here.
- **Three independent Q-learning loops** — one per tab (Silly Walk, Line
  Follower, Maze Solver), each with its own state discretization, reward
  table, training loop, and greedy-policy loop, all built on `QTable` from
  `qlearn.py`. Only one loop can run at a time.

### `qlearn.py`
The actual Q-learning algorithm, independent of hardware/UI: a `QTable`
class holding the Q-value and reward tables, `bellman_update()` (the
standard Q-learning update rule), and `choose_action()` (ε-greedy action
selection).

### `ble.js`
A thin `BLEDevice` wrapper around the browser's Web Bluetooth API
(`navigator.bluetooth`), imported into `main.py` as a PyScript JS module.
Handles device discovery/connect, GATT service/characteristic setup,
notification subscription, sending bytes, and disconnect — this is the only
place that talks to real Bluetooth hardware.

### `pyscript.toml`
PyScript project config: declares the Python packages to install
(`legoeducation`, `numpy`), the local Python file to make importable
(`qlearn.py`), and the JS module mapping (`ble.js` → `ble`, importable in
Python as `pyscript.js_modules.ble`).
