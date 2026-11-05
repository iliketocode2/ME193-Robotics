# LQR vs MPC: The Difference in One Page

> Read [01-lqr.md](01-lqr.md) and [02-mpc.md](02-mpc.md) first. This page exists
> to stop students filing LQR and MPC as two unrelated techniques, which is the
> most common way this material goes wrong. (That is the reading order for the
> course. On the web, nothing needs to be built or run first.)

> **Interactive:** the MPC tab of [docs/interactive.html](interactive.html) has
> a convergence panel that reproduces the table below live as you drag the
> horizon slider.

## The one-sentence answer

**MPC with an infinite horizon, no constraints, a linear model and a quadratic
cost *is* LQR** — LQR is the closed-form solution to that special case, worked
out in advance so you never have to run the optimizer.

Everything else on this page is a consequence of that sentence.

---

## Side by side

| | **LQR** | **MPC** |
|---|---|---|
| Horizon | Infinite | Finite ($N$ steps) |
| Constraints | None expressible | Hard limits on states and inputs |
| When the optimization runs | Once, offline, at your desk | Every control period, on the robot |
| What it produces | A gain matrix $K$ | A number: the first move of a plan |
| Runtime cost | Two multiplies | A QP, 20 times a second |
| The control law | $u = -K\mathbf{x}$, forever | No closed form; it's whatever the solver says |
| Handles preview | No | Yes, that's the main reason to use it |
| Speed as a decision | No — $v$ sits inside $A$ | Yes, natively |
| Stability | Guaranteed for the LTI model | Only with a terminal cost/constraint |
| Debuggable by inspection | Yes | Not really |
| Optimal? | Globally, over all controllers, for that model and cost | Over the horizon, for the previewed future |

---

## The demonstration

Don't take the one-sentence answer on faith — watch the convergence. Take the
reference build's discrete model at $v = 150$ mm/s, $T = 50$ ms, with the LQR
weights and **no constraints at all**. Put the robot 4 mm off the line at a
heading error of 0.1 rad, and ask an MPC with horizon $N$ what it would do.

Then lengthen the horizon. Run it here (section 6), or locally with
`python3 code/sim.py`:

<div class="pyrun" data-run="sim" data-script="code/sim.py" data-section="6"></div>

| Horizon $N$ | First move $\omega_0$ (rad/s) | Distance from LQR |
|---:|---:|---:|
| 1 | −0.035575 | 1.3 |
| 2 | −0.156996 | 1.2 |
| 5 | −0.929808 | 4.3 × 10⁻¹ |
| 10 | −1.338622 | 2.4 × 10⁻² |
| 20 | −1.360078 | 2.5 × 10⁻³ |
| 50 | −1.362575 | 3.3 × 10⁻⁸ |
| 200 | −1.362575 | 4.1 × 10⁻¹³ |
| **∞** | **−1.362575** | **0** — this is $-K\mathbf{x}$ |

Check the last row against the gains from [01-lqr.md](01-lqr.md#the-version-you-actually-run-discrete-time):

$$\omega = -k_e e - k_\theta \theta = -(0.164839)(4.0) - (7.032191)(0.1) = -1.362575$$

The two methods are the same method.

Notice the shape of the convergence. At $N = 1$ the MPC is nearly useless — it
can only see 50 ms ahead, where steering barely moves anything, so it does
almost nothing. By $N = 10$ (half a second) it agrees with LQR to 2%; by
$N = 50$ (2.5 s) it agrees to eight decimal places. **The horizon has to be
long compared to the closed-loop settling time**, which for this robot is
$4/(\zeta\omega_n) \approx 1.0$ s — and sure enough, convergence arrives right
around there. That is the same rule of thumb that set $N = 15$ in
[02-mpc.md](02-mpc.md#why-the-horizon-length-matters), now visible as a number.

### And it is the same code

Look at the Riccati iteration in [`sim.py`](sim.py):

```python
P = Q
for _ in range(N):
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)
    P = A.T @ P @ (A - B @ K) + Q
```

Run that loop $N$ times and $K$ is the optimal first move of an $N$-step MPC.
Run it until $P$ stops changing and $K$ is the LQR gain. It is one algorithm —
dynamic programming, backward in time — and "LQR" is just the name for its
fixed point.

---

## So when does the distinction actually matter?

Only when one of the special-case assumptions breaks. Precisely three things:

**1. A constraint binds.** If the optimal unconstrained solution never hits a
limit, MPC returns the LQR answer and you spent a QP to get it. MPC earns its
keep only when the constraint is *active* — for the line follower, when the
predicted error would leave the ±6 mm sensor window. On the straights of
[the head-to-head track](02-mpc.md#head-to-head), the MPC robot is running LQR;
it only diverges in the corner.

**2. You know something about the future.** LQR's infinite horizon sounds like
the ultimate look-ahead, but it assumes the future is *the model repeating
forever with no external input*. It cannot use "there is a hairpin in 10 cm."
Any time you have a forecast — a track map, a look-ahead sensor, a known
setpoint change — MPC can act on it and LQR structurally cannot.

**3. A decision changes the model.** This is the subtle one and the most
important for this robot. Forward speed $v$ appears *inside* $A$. Choosing $v$
is choosing a different plant, so no single $K$ can be optimal across speeds.
LQR's escape is gain scheduling — but scheduling tells you which gains to use
*after* something else picks the speed. MPC treats "which plant am I driving"
and "how should I steer it" as one optimization.

Absent all three, **use LQR.** It is not the training-wheels version. For an
unconstrained linear regulation problem it is the exact optimum, it runs in two
multiplies, it comes with unconditional stability and generous robustness
margins, and you can explain it to someone at a whiteboard. Reaching for MPC
there is strictly worse engineering.

---

## Misconceptions worth heading off

**"MPC is the modern one; LQR is what they used before computers were fast."**
No. They solve different problems and MPC contains LQR as a special case.
LQR is running right now in aircraft, disk drives, and spacecraft where its
guarantees matter more than constraint handling. New does not mean superseding.

**"MPC is optimal, so it must beat LQR."** MPC is optimal *over its horizon,
for its previewed future, subject to its model*. Give it a short horizon, a bad
forecast, or the wrong model and it will be confidently, optimally wrong.
On the unconstrained problem it is at best a slower LQR.

**"MPC plans a trajectory and follows it."** It plans a trajectory and then
throws almost all of it away, keeping the first move. If it ever *executed* a
full plan it would be open-loop control and would fall over. Re-solving is not
an implementation detail — it is the entire source of feedback.

**"LQR can't handle constraints so you just clip the output."** Clipping is what
everyone does and it works surprisingly often, but it is not the same thing.
A clipped LQR doesn't know it was clipped, keeps issuing commands as if it
weren't, and can wind up badly. And clipping addresses *input* limits only —
it can do nothing about a **state** constraint like the sensor window, which is
the one that actually breaks the line follower.

**"The three sensor states are the states."** See
[01-lqr.md](01-lqr.md#the-three-sensor-states-are-not-the-state). Three
measurement regions; two state variables; one sensor.

---

## The card students should leave with

> **LQR** answers: *given where I am, what is the best single move, assuming
> the world stays linear and nothing ever gets in my way?*
> It answers it once, in closed form, and hands you two numbers.
>
> **MPC** answers: *given where I am, what I expect the road to do, and the
> walls I must not hit, what is the best next 0.75 seconds?*
> It answers it twenty times a second, and hands you the first move.
>
> On an open straight road they hand you the same number.
> Corners and walls are what separate them.

---

**Back to:** [the course plan](README.md) · [LQR](01-lqr.md) · [MPC](02-mpc.md)
