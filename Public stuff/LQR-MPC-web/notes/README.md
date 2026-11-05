# Teaching LQR and MPC with a One-Sensor Line Follower

> **Web edition.** This course was originally written as a locally hosted
> version: you clone the repository and run the commands in the Quick start below.
> Now you can run it on the web instead, with nothing to install. The
> **Interactive** tab runs the simulator as Python (PyScript), and its
> **Your robot** tab connects a LEGO Education Color Sensor and Double Motor over
> Bluetooth (Chrome or Edge). The Quick start still works if you want the local
> version.

A hands-on course that gets students from "my robot wobbles" to "I derived my
own gains from a cost function" to "my robot slows down for hairpins because
the optimizer decided it should."

The vehicle is deliberately small: a **LEGO Education Double Motor** (differential
drive) plus a **single Color Sensor** reading reflected light, following a strip
of black tape. One sensor, two motors, two state variables. Small enough that
the Riccati equation can be solved **by hand on a whiteboard**, real enough that
every simplifying assumption eventually bites.

## Quick start

**Use the web.** Open the **[Interactive page](interactive.html)**. It is the whole
course in your browser, with nothing to install: the simulator with sliders for
every gain and cost weight, the calibration, and, if you have the LEGO hardware
and Chrome or Edge, buttons that connect over Bluetooth and run every controller
on the real robot. No robot on hand? Everything except the "real robot" buttons
works from simulation alone. Each session below links to the tab it uses, and the
**Run it here** boxes run what each script prints, right in the page.

**If you are not using the web version**, do this instead to run the course locally:

```bash
git clone https://github.com/chrisbuerginrogers/LQR-MPC.git
cd LQR-MPC
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt

.venv/bin/python3 code/calibrate_server.py
```

Then open **http://localhost:8000/** in a browser. That's the whole course:
rendered docs, an interactive sliders-and-plots page, and — if you have the
LEGO hardware connected — buttons to redo calibration and run every
controller on the real robot. No robot on hand? Everything except the
"real robot" buttons works from simulation alone.

## The documents

| | |
|---|---|
| **[docs/00-sim.md](00-sim.md)** | What `code/sim.py` is and how it's built: the model, the LQR and MPC solvers, and how the report at the bottom regenerates every number quoted in the other three documents. |
| **[docs/01-lqr.md](01-lqr.md)** | What LQR is, the math behind it, and a fully worked example over the three sensor states: on the line, on the edge, off the line. |
| **[docs/02-mpc.md](02-mpc.md)** | What MPC is, why you would reach for it, and a worked example of slowing down through a sharp turn. |
| **[docs/03-lqr-vs-mpc.md](03-lqr-vs-mpc.md)** | The difference in one page, including the numerical demonstration that MPC *becomes* LQR when you take the constraints away. |
| **[docs/interactive.html](interactive.html)** | The same math as an interactive page: sliders for every gain and cost weight, live e(t)/v(t) plots, using *your* calibrated numbers. Open it from the nav bar (the local version is described in "Running the code" below). |

## The teaching idea

Most control courses introduce LQR as "here is the Riccati equation." Students
can solve it and still have no idea why anyone would want to. This course
inverts that: **every piece of math is introduced as the answer to a failure the
students have already watched happen on the floor.**

```
bang-bang wobbles            ->  we need proportional action
P control oscillates         ->  we need to respond to how fast the error grows
PD works but how do I tune?  ->  write down what you care about   ->  LQR
LQR slides off sharp turns   ->  the sensor window is a constraint
constraints + preview        ->  MPC
```

The single sensor is not a limitation to apologize for. It is the pedagogy:

- It forces the distinction between **what you measure** (one number, saturating)
  and **the state** (two numbers, one of which you never see). That is the whole
  motivation for observers, and it lands hard the first time a student's robot
  loses the line and the controller keeps confidently reporting `e = 6.0 mm`.
- It gives a **hard, physical, measurable constraint** — the sensor's linear
  window is about ±6 mm wide — which is exactly the kind of thing LQR cannot
  express and MPC exists to handle.
- It makes the sharp-turn failure **predictable in closed form**, so students can
  compute in advance the speed at which their robot will fail, then go watch it
  fail at that speed.

## Where the equations come from

Every equation in this course falls out of two decisions, made in order:
**what to track**, then **how to score it**. Nothing is handed down — every
symbol traces back to either geometry or a cost weight a student picked.

**The state: why $(e, \theta)$ and nothing else.** A robot following a line
only needs to know two things about itself relative to the line: how far off
it is ($e$, lateral offset in mm) and which way it's pointed relative to the
line's own direction ($\theta$, heading error in rad). Two robots with the
same $(e, \theta)$ will do the identical thing next, no matter where on the
track they are or what shape the track took to get there — global position
and absolute heading are thrown away on purpose, because the controller never
needs them. That's also exactly why the sensor's single saturating number is
a genuine handicap and not just an annoyance: it can report $e$, sometimes,
but it never reports $\theta$ at all.

**The kinematics: two lines, both from projecting velocity.** Put the robot
at heading $\theta$ relative to the line, moving forward at speed $v$. The
component of that velocity *perpendicular* to the line is $v\sin\theta$ —
that is the rate $e$ grows or shrinks:

$$\dot e = v \sin\theta \approx v\,\theta$$

(the small-angle approximation holds to within about 2% out to 20°, which
covers everything a sane line follower should ever see — and it is also the
step that makes the model linear, which matters in a moment). $\theta$ itself
is *heading relative to the line*, so it changes for two independent reasons:
the robot turns at whatever yaw rate $\omega$ it is commanded, and the line
itself turns underneath it as the robot travels along a curve of curvature
$\kappa$ at speed $v$ (a curve of radius $R$ turns at $v/R = \kappa v$ rad per
second, by the definition of curvature):

$$\dot\theta = \omega - \kappa v$$

Together these two lines are the entire plant. Full derivation, with the
matrix form: [docs/01-lqr.md, "Step 2: the model"](01-lqr.md#step-2-the-model).

**The cost, and why the gains come out the way they do.** Linearizing
$\sin\theta \to \theta$ is what turns this into a linear system
$\dot{\mathbf x} = A\mathbf x + Bu$ — and a linear system is what lets you
pose control as *minimize a quadratic cost*, solve it exactly, and get a
state-feedback gain back in closed form instead of a number you tuned by
hand:

$$J = \int_0^\infty \big(q_e e^2 + q_\theta \theta^2 + r\,\omega^2\big)\,dt$$

Each weight is a direct trade you are choosing, in the units of the thing it
costs: $q_e$ is how much you mind being 1 mm off the line for a second,
$q_\theta$ is how much you mind pointing the wrong way, $r$ is how much you
mind spending yaw rate to fix either. Push the Riccati equation through for
*this specific* $A$, $B$ (only $v$ appears in $A$, and the whole thing is low
enough dimension to do by hand) and it collapses to

$$k_e = \sqrt{q_e / r}, \qquad k_\theta = \sqrt{q_\theta/r + 2vk_e}$$

worked from scratch, on a whiteboard, in
[docs/01-lqr.md, "Solving it by hand"](01-lqr.md#solving-it-by-hand).
MPC (Session 5) doesn't introduce new physics — it reuses this exact model
and this exact steering law, and only changes which $v$ gets plugged in each
control period. See [docs/02-mpc.md](02-mpc.md) for that derivation.

## Course arc

Seven sessions, 60–90 minutes each. Sessions 0–2 need no theory background;
3 onward assume comfort with matrices and derivatives.

### Session 0 — Build and calibrate
Build the robot: sensor mounted forward of the wheel axle, 5–8 mm above the
floor, pointed straight down. Lay a track from 19 mm black tape on white
poster board.

Then measure your own robot: reflection on black, reflection on white, the
width of the sensor's transition region, wheel speed at 100%, and track width.
On the web, do it on the **[Your robot tab](interactive.html#robot)** (Redo
calibration: connect the sensor, sample black and white, set the window, drive
the speed test, save). If you are running locally, use `python3 code/calibrate.py`
instead. **Nothing in the rest of the course uses a
number that a student did not measure.**

> Checkpoint: black/white contrast of at least 40 counts, and a sensor window
> the student can state in millimeters.

### Session 1 — Bang-bang, and why it wobbles
`if reflection > threshold: turn left, else: turn right`. It works. It also
oscillates visibly at a couple of Hz and cannot go fast.

On the web, open the **[1. Bang-bang tab](interactive.html#bangbang)**: the plot
shows the wobble, and *Run on robot* drives your own. You can also run the
script right here (in simulation) and watch the printed wobble frequency:

<div class="pyrun" data-run="bangbang" data-script="code/run_bangbang.py" data-args="20 100 2"></div>

If you are running locally, run `python3 code/run_bangbang.py` instead.

Ask: *what information are we throwing away?* Answer: the size of the error.
The sensor reports a number between 12 and 85 and we are using one bit of it.

> Checkpoint: students can state the control law of their own robot as an
> equation, and predict the wobble frequency will drop if they slow down.

### Session 2 — P, then PD, then the tuning wall
Proportional on the normalized error. Better. Push the gain up: it oscillates.
Push it down: it corners badly. Add a derivative term: much better, and now
there are two knobs and no principle for setting them.

On the web, use the **[2. P / PD tab](interactive.html#pd)**: sliders for $k_p$
and $k_d$, a plot of the turn, and *Run on robot*. Or run the script right here
(in simulation): the P-only pass first, then add a fourth number, $k_d$, once a
derivative term is added.

<div class="pyrun" data-run="pd" data-script="code/run_pd.py" data-args="20 150 0.15"></div>

If you are running locally: `python3 code/run_pd.py 20 150 <kp>` for the P-only
pass, then `python3 code/run_pd.py 20 150 <kp> <kd>`.

This is the session where students should get *frustrated* by hand-tuning. Let
them. Ten minutes of twiddling two numbers is worth an hour of motivation.

> Checkpoint: a table of (gain, observed behavior) pairs, and the honest
> admission that the "best" pair was found by guessing.

### Session 3 — LQR
Read **[docs/01-lqr.md](01-lqr.md)**.

Write down the model. Write down what you care about, as a cost. Turn the crank.
Out comes the gain pair the students were guessing at — and out comes a
principled reason it is that pair and not another. The payoff moment is
[the closed-form solution](01-lqr.md#solving-it-by-hand): for this robot the
whole Riccati equation collapses to

$$k_e = \sqrt{q_e/r}, \qquad k_\theta = \sqrt{q_\theta/r + 2 v k_e}$$

and with $q_\theta = 0$ the damping ratio comes out to exactly $1/\sqrt 2$ no
matter what else you choose. Students can check that on the whiteboard and then
watch it on the floor.

On the web, open the **[3. LQR tab](interactive.html#lqr)**, or run the script
right here (in simulation):

<div class="pyrun" data-run="lqr" data-script="code/run_lqr.py" data-args="20 150"></div>

If you are running locally, run `python3 code/run_lqr.py`.

> Checkpoint: a student can change $r$ from 25 to 100, predict which way the
> robot's behavior will move *before* running it, and be right.

### Session 4 — Break it on purpose
Give every group a track with a hairpin. Before driving, have them compute the
steady-state offset their controller will hold on that curve:

$$e_{ss} = -\frac{\kappa v}{k_e}$$

For $R = 80$ mm at 150 mm/s with $k_e = 0.2$, that is **9.4 mm** — larger than
the ±6 mm sensor window. Prediction: the robot will go blind mid-corner. Then
run it. It goes blind mid-corner.

This is the best session of the course. The theory made a falsifiable
prediction about a physical object and the object complied.

> Checkpoint: each group reports the speed at which *their* robot loses *their*
> track's tightest corner, and it matches $v = k_e e_{max} R$ within ~20%.

### Session 5 — MPC
Read **[docs/02-mpc.md](02-mpc.md)**.

The fix is not a bigger gain — students should try, and discover it makes the
robot unstable long before it fixes cornering. The fix is to stop treating
speed as a constant. But the moment speed becomes a decision, the model is no
longer linear, and there is a hard constraint ($|e| \le 6$ mm) that a quadratic
cost cannot express. That is precisely the pair of problems MPC was invented for.

On the web, open the **[5. MPC tab](interactive.html#mpc)**, or run the script
right here (in simulation) and watch the speed number drop before the corner:

<div class="pyrun" data-run="mpc" data-script="code/run_mpc.py" data-args="20"></div>

If you are running locally, run `python3 code/run_mpc.py`.

> Checkpoint: a student can point at the printed candidate table and explain
> why the optimizer rejected 150 mm/s.

### Session 6 — Same thing, different corner of the same idea
Read **[docs/03-lqr-vs-mpc.md](03-lqr-vs-mpc.md)**.

Run the last section of `code/sim.py` right here (or locally, with
`python3 code/sim.py`):

<div class="pyrun" data-run="sim" data-script="code/sim.py" data-section="6"></div>

With the constraints removed and the horizon stretched, the MPC's first move converges to $-Kx$ to twelve decimal
places. LQR is not a different theory from MPC. It is the closed-form answer to
the special case where the horizon is infinite and nothing is constrained.

> Checkpoint: students can say what you give up (optimality guarantees are now
> only over the horizon; you now solve an optimization every 50 ms) and what you
> buy (constraints, preview, a speed that varies).

### Session 7 — Capstone
Time trials on a track with one hairpin and one long straight. Score = lap time
+ a penalty per sample where the sensor was saturated. Teams may use anything
they can defend.

The scoring is chosen so that the naive answers lose: pure LQR at high speed
posts a fast lap and eats penalties; a timid constant speed posts no penalties
and a slow lap. Speed scheduling wins, which is the point.

## The robot

| Part | Role |
|---|---|
| LEGO Education Double Motor | Differential drive; also carries the IMU used in the observer lab |
| LEGO Education Color Sensor | Single downward light sensor, `reflection` 0–100 |
| 19 mm black vinyl tape on white board | The track |

Both devices connect over BLE (on the web, the Interactive page connects with
Web Bluetooth; locally, `pip install legoeducation`). Your hubs need the latest
firmware: if one is older, the page shows an alert asking you to update at
[code.legoeducation.com](https://code.legoeducation.com/) and will not connect. They are
**separate BLE devices** — the sensor is not plugged into the motor — so the
control loop runs on the laptop and the loop rate is limited by the BLE
notification period. Use 50 ms (20 Hz). This is slow enough to matter, which is
a feature: [docs/01-lqr.md](01-lqr.md#the-version-you-actually-run-discrete-time)
shows the discrete-time gains coming out ~18% softer than the continuous-time
ones because of it.

**Mount the sensor forward of the axle.** A sensor 40 mm ahead of the wheels at
150 mm/s is giving you 0.27 s of free preview, which matters in Session 5.

**Follow one edge of the tape, not its center.** With a single sensor, "centered
on the line" is a maximum of the reading — the sensor cannot tell left from
right there. The *edge* is where reflection changes monotonically with lateral
position, so that is where a single sensor carries a signed error. Everything in
these documents assumes edge following. One consequence worth warning students
about: an edge follower is **asymmetric**. Turns that curve toward the tape and
turns that curve away from it behave differently, and a track should include both.

## Running the code

**Use the web.** Open the **[Interactive page](interactive.html)**: there is
nothing to run or install. It is every doc's math turned into sliders: drag
Q_e/Q_θ/R and watch the damping ratio and e(t) response change together; drag
the turn radius until the verdict flips to BLIND; drag the MPC horizon and watch
its first move converge onto the LQR gain. No robot is required for any of that.
With a robot (Chrome or Edge, latest firmware), the **Your robot** tab connects
the sensor and motors over Bluetooth and walks through calibration, and every
controller tab has a **Run on robot** button that plots the real, measured e(t)
on top of the simulated curve. The **Run it here** boxes in these notes run what
`sim.py` and the `run_*.py` scripts print, in your browser.

**If you are not using the web version, do this:**

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt   # legoeducation, numpy, matplotlib, markdown

.venv/bin/python3 code/sim.py            # no robot needed; reproduces every number in the docs
.venv/bin/python3 code/calibrate.py      # measure your build, writes code/calibration.json
.venv/bin/python3 code/run_bangbang.py 20    # 20 s of bang-bang line following
.venv/bin/python3 code/run_pd.py 20 150 0.15 # 20 s of P (add a 4th arg for PD)
.venv/bin/python3 code/run_lqr.py 20     # 20 s of LQR line following
.venv/bin/python3 code/run_mpc.py 20     # 20 s of MPC with speed scheduling

.venv/bin/python3 code/calibrate_server.py   # from the repo root, then open:
                                              #   http://localhost:8000/
```

**`docs/interactive.html` is every doc's math turned into sliders.** It
reimplements `sim.py`'s LQR and MPC solvers directly in JavaScript,
cross-checked line-by-line against `sim.py`'s own printed output. Drag
Q_e/Q_θ/R and watch the damping ratio and e(t) response change together; drag
the turn radius until the verdict flips to BLIND; drag the MPC horizon and
watch its first move converge onto the LQR gain. No robot required for any of
that, and no dependency beyond a browser.

**`code/calibrate_server.py` is what turns the page from a simulator into a
control panel for the real robot.** Run it (from the venv, so it can `import
legoeducation`) instead of a bare `python3 -m http.server`, and three things
change:

- Every `README.md`/`docs/*.md` renders as a styled page instead of plain
  text, with a nav bar linking all of them plus `docs/interactive.html`.
- The "Your robot" tab gets a **Redo calibration** panel that walks through
  the same steps as `code/calibrate.py` — connect, sample black/white, set
  the sensor window, drive the speed test, save — except every step happens
  as a button click in the browser, with the live sensor reading updating as
  you slide the robot.
- Every controller tab (bang-bang, P/PD, LQR, MPC) gets a **Run on real
  robot** button: it drives your actual robot for a few seconds using that
  tab's exact gains, then plots the real, measured e(t) (and v(t) for MPC)
  in green right on top of the simulated curve — a genuine model-vs-reality
  comparison, not just a simulation.

Only one browser tab should drive the robot at a time (the connection state
lives in the server process), and each "Run on real robot" click confirms
before moving the robot, since it's a real physical action. If a hardware
call can't find the sensor/motor over BLE it can take up to ~30s to give up
and report a clear error rather than silently doing nothing — that's the
`legoeducation` library's own connection timeout, not something this project
controls.

**Every `run_*.py` script logs its run and pops up a live plot.** Each control
period is appended to a timestamped CSV under `code/runs/` (e.g.
`code/runs/lqr_20260806_143000.csv`), and a plot window opens showing $e(t)$
and $v(t)$ updating live, with your last couple of runs of that controller
faded in gray and the most recent run of every *other* controller faded in
color — the head-to-head view is built in, not something you have to
assemble by hand afterward. Pass `--no-plot` to skip the plot (e.g. over SSH
with no display); the CSV is still written either way. See
code/live_plot.py for how it works — it runs in its own
process specifically so a slow redraw can never cost the control loop a
50 ms period.

| File | What it is |
|---|---|
| [code/sim.py](sim.py) | Pure-numpy simulator, LQR solver, MPC solver, and the report that generates every table in the docs (walkthrough: [docs/00-sim.md](00-sim.md)) |
| [docs/interactive.html](interactive.html) | `sim.py`'s math in the browser: sliders for every gain/cost weight, live e(t)/v(t) plots, your calibration baked in |
| code/robot.py | The only file that knows about BLE. Exposes `read() -> (e, theta, saturated)` and `drive(v, omega)` |
| code/calibrate.py | Session 0 measurement script (terminal version) |
| code/calibrate_server.py | Local web server: renders the docs, and gives `interactive.html` a JSON API to redo calibration and run controllers on real hardware |
| code/hardware_runs.py | The bang-bang/PD/LQR/MPC control loops `calibrate_server.py` runs against the real robot, mirroring the `run_*.py` scripts |
| code/requirements.txt | `pip install -r` this into a `.venv` — legoeducation, numpy, matplotlib, markdown |
| code/telemetry.py | Per-run CSV logging (`code/runs/`) and lookup of previous runs, shared by every `run_*.py` script |
| code/live_plot.py | Live $e(t)$/$v(t)$ plot in its own process, with history and cross-controller comparison overlaid |
| code/run_bangbang.py | Session 1 |
| code/run_pd.py | Session 2 |
| code/run_lqr.py | Session 3 |
| code/run_mpc.py | Session 5 |

`sim.py` holds the control theory and `robot.py` holds the hardware, with no
dependency from the first on the second. Students can develop and debug a
controller on a laptop on the train and then run the identical code on the robot.

## Instructor notes

**Units.** Insist on millimeters, seconds, and radians everywhere, and make
students write units on the gains ($k_e$ is in rad/s per mm). Most wrong answers
in this material are unit errors wearing a disguise.

**"Three states" is an overloaded phrase.** Students will arrive thinking the
robot has three states because the sensor sees three things. It does not; it has
a two-dimensional state and a three-region *measurement map*.
[docs/01-lqr.md](01-lqr.md#the-three-sensor-states-are-not-the-state) settles
this explicitly, and it is worth ten minutes of class time — the confusion is
the single most common blocker in the LQR session.

**Don't skip the failure.** The temptation is to go from LQR straight to MPC
because MPC is the exciting one. Session 4 — computing the failure, then
watching it — is what makes MPC feel necessary rather than fashionable.

**Batteries.** A robot at 40% battery has a lower `wheel_max_mms` than the one
students calibrated, which shows up as everything being mysteriously sluggish
and the gains being mysteriously wrong. Recalibrate, or at least warn them.

**Lighting.** Overhead fluorescents at 100/120 Hz beating against a 20 Hz sample
rate produce a slow apparent drift in the reflection reading. If a group's
robot behaves differently in the morning and the afternoon, this is why.
