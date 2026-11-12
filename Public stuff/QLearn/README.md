# Q-Learning Visualizer (LEGO Education, browser/PyScript)

### 👉 [Click here to try it](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/QLearn/)

The link opens a page where you choose one of three versions:

| Version | Runs | What it is |
|---|---|---|
| **Q-Learning Visualizer** (`app.html`) | in the browser, nothing to install | the full web app described below, with three tasks |
| **Q-table dashboard** (`PascalQTable.py`) | Python on your computer | the Silly Walk Q-table as a desktop program |
| **Neural-network dashboard** (`PascalNN.py`) | Python on your computer | the same Silly Walk task with a small neural network (DQN) |

The two Python versions are described under [Python versions](#python-versions) below.

## The web app

A browser-based app that teaches Q-learning by having students train a LEGO
Education robot (Double Motor hub + optional Color Sensor hub) over Web
Bluetooth. Everything runs client-side via [PyScript](https://pyscript.net)
(Python compiled to WASM via Pyodide) — there is no server or build step;
opening `app.html` in a Bluetooth-capable browser (Chrome/Edge) is enough.

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
The landing page. It has one card for each of the three versions: open the web
app, or download one of the two Python programs. It also explains how to install
and run the Python versions. It's plain HTML and CSS, with no PyScript.

### `app.html`
The web app's entire UI: layout, styling, the three tab pages, and a `<script>` block
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
  route their sends and notifications through `ble.js`'s Web Bluetooth wrapper
  (via `webhub.py`) instead of the library's own Bluetooth thread.
- **Firmware check** — `webhub.py` (also used by the AnnaCode pages) checks each hub when
  it connects. Only the latest firmware is supported: a hub with older firmware gets an
  alert to update at https://code.legoeducation.com/ and is disconnected. This needs
  `legoeducation` 1.2.0 or later, which `pyscript.toml` pins.
- **UI event handlers** — every `py-click` target in `app.html` (connect/
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

## Python versions

`PascalQTable.py` and `PascalNN.py` run the Silly Walk task as desktop programs,
each with a live dashboard window. Each is a single file. Download it from the
landing page or from this folder, then:

```bash
python3 -m pip install "legoeducation>=1.2.0" numpy
python3 PascalQTable.py        # or PascalNN.py
```

On Windows, type `py` in place of `python3`. The window is drawn with tkinter,
which comes with Python from python.org. Turn on the Double Motor first. Its
firmware must be up to date (update it at https://code.legoeducation.com/). The
program connects to the first Double Motor it finds; set `SERIAL` near the top
of the file to pick one.

When the buttons light up, aim the robot and click:

- **Train** runs 10 episodes × 10 steps. It continues from what was learned before.
- **Run** takes 100 steps using what it has learned, with no exploring.
- **Clear & train** starts from scratch.
- **Pause** (space bar) stops the legs, for example when the robot walks into a
  door. Move it, aim it and Resume; its new direction counts as straight.
- **Stop** (Esc) stops and saves.

What it learns is saved next to the program (`pascal_qtable.npz` or
`pascal_nn.npz`).

### `PascalQTable.py`
The Silly Walk tab's Q-table, with the same settings: 3 states (Drifted Left
yaw > 10°, Straight, Drifted Right yaw < −10°), 2 actions (one motor turns 135°),
rewards −1 / +10 / −1, α = 0.1, γ = 0.9, and ε falling from 1 to 0.05 over
training. The dashboard shows:

- the Q-table, colored as it fills in
- every Bellman update written out with its actual numbers
- a yaw gauge
- the total reward per episode

### `PascalNN.py`
The same task with a small neural network in place of the table. It is written
in plain numpy so the forward pass, backpropagation and Adam optimizer are all
readable.

- **Input and output:** the network takes the yaw itself as input, not one of the
  three states, and outputs Q(LEFT) and Q(RIGHT).
- **Learning:** it learns as a DQN, with a replay buffer, a target network, and
  gradient descent on

  ```
  loss = ( Q(yaw, a) − [ r + γ·max Q_target(yaw′, a′) ] )²
  ```

- **Dashboard:** the main plot shows both Q-values at every yaw from −45° to +45°,
  plus the best action at each yaw. There is also a loss chart.

### Reward timing
The Python versions give the reward for the state the robot **ends up in** after
each step. This is standard Q-learning: a good action is credited straight away.
The web app gives it for the state the robot **started in**. Both lead to the same
best policy, but the web app's way learns more slowly, because both actions get
the same immediate reward. To make the Python versions match the web app, set
`REWARD_NEXT_STATE = False` near the top of the file.
