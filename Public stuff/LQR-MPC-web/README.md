# LQR and MPC for a line follower (runs in the browser)

### 👉 [Click here to try it](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/LQR-MPC-web/)

The same course as the original `LQR-MPC` project, rebuilt for the browser. The
simulator is Python running through [PyScript](https://pyscript.net), so there is
nothing to install. The **Your robot** tab connects a LEGO Education Double Motor
and Color Sensor over Bluetooth (Chrome or Edge), and every tab can run its
controller on the real robot.

## Same tabs, same order

The top bar matches the original: **Home** (the README), **00 · sim.py**,
**01 · LQR**, **02 · MPC**, **03 · LQR vs MPC**, **Interactive**, and
**🔌 Connect to robot**. The Interactive page has the same five tabs:
**Your robot**, **1. Bang-bang**, **2. P / PD**, **3. LQR** and **5. MPC**.

The text is the original's: the five Markdown files are in `notes/`, shown by
`reader.html`, and the Interactive page keeps its original wording. Changes:

- A short "Web edition" note at the top of the Home page, saying the course was written to run
  locally with the Quick start commands and now also runs on the web.
- Links to the hardware scripts (`code/…`) are plain text, because those scripts are
  not part of this site.
- "JS port in this page; Python original in…" now says the page runs the Python in
  `lqrmpc.py`, since that is now true.
- The Your robot tab has a "Connect to robot" panel, the calibration is saved in
  your browser instead of `calibration.json`.
- Each "Run on robot" panel has a Stop button.
- The notes put the web version first. The "Prerequisite" boxes are marked as local-only,
  every session sends you to the Interactive tab it uses, and the "Running the code" section
  says "Use the web" first, then "if you are not using the web version, do this".
- Where the notes say to run `sim.py` or a `run_*.py` script, there is a "Run it here" box
  that runs it in your browser and prints the same thing. The `run_*.py` boxes run on the
  simulator's model of the robot, since the scripts normally drive a real one.

## The files

- `index.html` — sends you to the Home page.
- `reader.html` — renders `notes/*.md` (Markdown and maths).
- `interactive.html` — the tabs and their text. The page loads `interactive_page.py`.
- `interactive_page.py` — the original page's logic, in Python: sliders, readouts,
  charts, calibration, the live oscilloscope and the robot runs.
- `lqrmpc.py` — the simulator (LQR, MPC, the track), plain numpy.
- `hardware.py` — connects hubs through the `legoeducation` library's own Web Bluetooth
  transport (library 1.2.0 or later), checks the firmware, and has the `LineFollower` and the
  bang-bang, PD, LQR and MPC controllers.
- `assets/robot.jpg` — the sample-build photo (scaled down, with its location data removed).
- `docs_runs.py` and `docs.toml` — the "Run it here" boxes in the notes: the report from `sim.py`
  and the `run_*.py` scripts, run in the browser. `reader.html` loads them only on pages that have a box.
- `ui.py` — tables and SVG charts.
- `sim.py` — the course's `sim.py` (the version the notes let you edit and run).
- `pyscript.toml` — loads numpy, `legoeducation` 1.2.0 and the modules above.

## Running it locally

PyScript loads files over HTTP, so double-clicking `interactive.html` will not work.
From this folder run `python3 -m http.server` and open http://localhost:8000/.

## On the robot

Mount the Color Sensor facing down over the edge of the tape. Connect both devices on
the Your robot tab, use "Redo calibration" (black, white, sensor window, then wheel
speed and track width), and press "Run on robot" on any tab. On a real robot there is
no map, so MPC previews with the turn it is holding now, as the MPC notes explain.
Until you calibrate, the simulations use the generic reference build.

Only the latest hub firmware is supported. If a hub is older, the page shows an alert asking you
to update at https://code.legoeducation.com/ and disconnects it.

The robot code follows the course's hardware scripts, but it has only been run against
simulated devices. On the first real run keep a hand near the robot and use Stop.
