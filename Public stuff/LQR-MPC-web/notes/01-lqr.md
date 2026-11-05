# LQR: Letting the Cost Function Choose Your Gains

> **Prerequisite (local version only):** you have a line follower running P or PD
> control, and you have hand-tuned it, and you are annoyed about it. If not, do
> [Session 2](README.md#session-2--p-then-pd-then-the-tuning-wall) first.
> The whole point of this page is to replace guessing, so you should have
> guessed at least once.
>
> **On the web there is nothing to build first.** Try P/PD on the
> [2. P / PD tab](interactive.html#pd): push the gain until it oscillates, pull it
> back until it corners badly, then come back here.

> **Interactive:** [docs/interactive.html](interactive.html) lets you drag
> $Q_e$, $Q_\theta$, and $R$ and watch the gains, damping ratio, and $e(t)$
> response change together, using your own calibrated numbers.

## Contents

- [The complaint LQR answers](#the-complaint-lqr-answers)
- [Step 1: what is the state?](#step-1-what-is-the-state)
  - [The three sensor states are not *the* state](#the-three-sensor-states-are-not-the-state)
- [Step 2: the model](#step-2-the-model)
- [Step 3: the cost](#step-3-the-cost)
- [Step 4: the Riccati equation](#step-4-the-riccati-equation)
- [Proof: P really is the infinite-horizon cost](#proof-p-really-is-the-infinite-horizon-cost)
- [Solving it by hand](#solving-it-by-hand)
- [Reading the answer](#reading-the-answer)
- [Worked example: the three sensor states](#worked-example-the-three-sensor-states)
- [The version you actually run: discrete time](#the-version-you-actually-run-discrete-time)
- [Where does θ come from?](#where-does-θ-come-from)
- [What LQR gives you, and what it assumes](#what-lqr-gives-you-and-what-it-assumes)
- [Labs](#labs)

---

## The complaint LQR answers

A PD line follower has two knobs. Turn them up, the robot oscillates. Turn them
down, it corners badly. Somewhere in between is "good," and the only way you
found it was by trying things.

The knobs are not the problem. The problem is that **the knobs are not the
language you think in.** You do not care about the numeric value of a
derivative gain. You care about things like:

> *"I want to be within a couple of millimeters of the line, and I'd rather not
> have the wheels jerking around to get there."*

LQR is the machine that converts that sentence into gains. You state what you
want as a **cost**, you state how the robot moves as a **model**, and it hands
back the *provably* best linear feedback law for that pair. There is no tuning
loop. If you don't like the result, you change what you asked for — which is a
statement about the robot's job, not about a number with no physical meaning.

**LQR** = **L**inear model, **Q**uadratic cost, **R**egulator (drive the state
to zero).

---

## Step 1: what is the state?

The state is *the smallest set of numbers such that, if you knew them, nothing
else about the past would help you predict the future.*

For a robot following a line, two numbers do it:

| Symbol | Meaning | Units |
|---|---|---|
| $e$ | how far the sensor is, sideways, from where it should be | mm |
| $\theta$ | how far the robot is pointed away from the line's direction | rad |

Two robots with the same $e$ and $\theta$ will do the same thing next. A robot
1 mm off the line and pointing straight is in a *different situation* from a
robot 1 mm off the line and heading further away at 10°, even though the sensor
reads the same in both cases. That is what it means for $\theta$ to be part of
the state.

$$\mathbf{x} = \begin{bmatrix} e \\ \theta \end{bmatrix}$$

The single input is the **yaw rate** $u = \omega$ (rad/s) — how fast we ask the
robot to rotate. The two wheel speeds come from it afterwards by inverse
kinematics, which is bookkeeping, not control:

$$v_L = v - \frac{\omega b}{2}, \qquad v_R = v + \frac{\omega b}{2}$$

with $b$ the track width (96 mm on the reference build). For now the forward
speed $v$ is a **constant you choose**. Remembering that this was a choice, and
not a law of nature, is what eventually leads to [MPC](02-mpc.md).

### The three sensor states are not *the* state

Students almost always arrive at this point with a different picture in mind:
the sensor sees three things — **on the line**, **on the edge**, **off the
line** — so surely the robot has three states?

Those three things are real and they matter enormously. But they are not the
state. They are the **measurement map**: how the one number the hardware gives
you relates to the state you actually care about.

Normalize the raw reflection so that 0 is fully over the tape and 1 is fully
over the background:

$$s = \frac{\text{reflection} - R_{\text{black}}}{R_{\text{white}} - R_{\text{black}}}$$

Then, on the reference build ($R_{\text{black}} = 12$, $R_{\text{white}} = 85$):

| Sensor state | $s$ | What it tells you about $e$ |
|---|---|---|
| **On the line** — spot fully over black | $s \approx 0$ | Only that $e \le -e_{max}$. Sign, no magnitude. |
| **On the edge** — spot straddling the boundary | $0 < s < 1$ | $e \approx 2 e_{max}(s - \tfrac12)$. **This is the useful region.** |
| **Off the line** — spot fully over white | $s \approx 1$ | Only that $e \ge +e_{max}$. Sign, no magnitude. |

with $e_{max} \approx 6$ mm for a typical LEGO sensor at 5–8 mm ride height.

Three facts follow, and all three drive the rest of the course:

1. **Edge following is not a stylistic choice.** Centered on the tape, the
   reading is at a maximum, and moving left or right both make it go down — the
   sensor cannot tell you which way you drifted. Only at the *edge* does the
   reading change monotonically with position, so only at the edge does one
   sensor produce a *signed* error.
2. **The measurement saturates, and lies while it does.** Off the line, the
   sensor happily reports a steady 85 forever. A controller that trusts it
   thinks the error is a constant 6 mm when it might be 40. This is exactly
   what happens in [Session 4](README.md#session-4--break-it-on-purpose).
3. **$\theta$ is never measured at all.** LQR assumes you know the whole state.
   You don't. [See below](#where-does-θ-come-from).

So: three sensor *regions*, two state *variables*, one measurement. Keep them
separate and the rest of this page is easy.

---

## Step 2: the model

Put the robot on a line of curvature $\kappa$ (so $\kappa = 0$ on a straight,
$\kappa = 1/R$ on a curve of radius $R$) and roll it forward at speed $v$.

**How does the lateral error change?** If the robot is pointed at an angle
$\theta$ to the line and moving at $v$, the sideways component of its velocity
is $v \sin\theta$. For the angles a line follower should ever see, $\sin\theta
\approx \theta$ to within about 2% out to 20°:

$$\dot e = v\sin\theta \;\approx\; v\,\theta$$

**How does the heading error change?** The robot's heading changes at the rate
we command, $\omega$. The *line's* direction changes at $\kappa v$ as we travel
along it. The error between them is the difference:

$$\dot\theta = \omega - \kappa v$$

On a straight line ($\kappa = 0$), in matrix form:

$$\dot{\mathbf{x}} = A\mathbf{x} + Bu, \qquad
A = \begin{bmatrix} 0 & v \\ 0 & 0 \end{bmatrix}, \qquad
B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

Recall $u = \omega$ — that's the whole input, not a scaled version of it. Expand
$A\mathbf{x} + Bu$ and check it against the two lines above term by term:

$$A\mathbf{x} + Bu = \begin{bmatrix} 0 & v \\ 0 & 0 \end{bmatrix}\begin{bmatrix} e \\ \theta \end{bmatrix}
+ \begin{bmatrix} 0 \\ 1 \end{bmatrix}u
= \begin{bmatrix} v\theta \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ u \end{bmatrix}
= \begin{bmatrix} v\theta \\ u \end{bmatrix}$$

Row 1 gives $\dot e = v\theta$ — that's where the $v$ in $A$'s top-right corner
comes from, it's literally "how much of $\theta$ leaks into $\dot e$." Row 2
gives $\dot\theta = u$: with $\kappa = 0$, $\dot\theta = \omega - \kappa v$
reduces to just $\omega$, i.e. just $u$, with no scaling and no dependence on
$e$ or $\theta$ themselves — which is exactly why row 2 of $A$ is $[0, 0]$
and $B$'s second entry is a bare $1$.

Three things to notice, because each one comes back later:

- **This is a double integrator.** Yaw rate integrates into heading, heading
  integrates into position. Double integrators are the reason naive
  proportional control oscillates: you are pushing on the *second* derivative
  of the thing you are trying to hold still.
- **$v$ is inside $A$.** The dynamics of a line follower depend on how fast it
  is going. Gains valid at 100 mm/s are not valid at 250 mm/s. LQR handles this
  by making you re-solve at each speed; MPC handles it by
  [treating $v$ as a decision](02-mpc.md#the-real-reason-lqr-cannot-do-this).
- **Curvature is a disturbance, and it is not in the model.** $\kappa v$ enters
  as an unmodeled input. LQR will reject it imperfectly, and *how* imperfectly
  turns out to be computable exactly — that is the whole of
  [Session 4](02-mpc.md#where-lqr-runs-out).

---

## Step 3: the cost

Now say what "good" means. LQR insists you say it as a quadratic integral:

$$J = \int_0^\infty \Big( \underbrace{\mathbf{x}^\top Q \mathbf{x}}_{\text{how bad is this situation}} + \underbrace{u^\top R\, u}_{\text{how much did it cost to fix}} \Big)\, dt$$

With $Q = \operatorname{diag}(q_e, q_\theta)$ and scalar $R = r$, that unpacks to

$$J = \int_0^\infty \big( q_e\,e^2 + q_\theta\,\theta^2 + r\,\omega^2 \big)\, dt$$

Read it as an **exchange rate**, which is the only way to pick the numbers
sensibly:

- $q_e = 1$ per mm² — "one second spent 1 mm off the line costs 1 unit."
- $q_\theta = 0$ — "I don't care about heading for its own sake, only insofar as
  it moves me off the line." (A perfectly reasonable position, and it makes the
  algebra pretty.)
- $r = 25$ per (rad/s)² — "one second at 1 rad/s of yaw rate costs 25 units,
  the same as sitting $\sqrt{25} = 5$ mm off the line."

That last line is the whole design decision, stated in units a person can
argue about: **5 mm of error is worth 1 rad/s of steering effort.** Halving the
tolerable error to 2.5 mm means $r = 6.25$.

Two properties worth stating out loud:

- **Only the ratio matters.** Scaling $Q$ and $R$ by the same factor scales $J$
  and leaves the optimal gains untouched. There is one knob here, not two —
  which is already an improvement over PD.
- **Quadratic is a choice, not a law.** Squaring means a 4 mm error is *sixteen*
  times worse than a 1 mm error. That is a strong opinion about your priorities.
  It is chosen mostly because it is the assumption under which the answer comes
  out in closed form, and it happens to be reasonable. Nothing physical requires
  it, and the hard limit that actually matters on this robot — "$|e|$ must stay
  under 6 mm or I go blind" — is not quadratic at all. Hold that thought.

---

## Step 4: the Riccati equation

Here is where it comes from, in about ten lines.

Define the **cost-to-go** $V(\mathbf x)$: the total future cost if you start at
$\mathbf x$ and play optimally forever. For a linear system with a quadratic
cost, guess that it is itself quadratic:

$$V(\mathbf x) = \mathbf x^\top P \mathbf x, \qquad P = P^\top \succ 0$$

Bellman's principle says: the best you can do from here is the cost you pay
right now, plus the best you can do from wherever that puts you. In continuous
time that is the **Hamilton–Jacobi–Bellman** equation:

$$0 = \min_u \Big[ \mathbf x^\top Q \mathbf x + u^\top R u + \nabla V^\top (A\mathbf x + Bu) \Big]$$

With $\nabla V = 2P\mathbf x$:

$$0 = \min_u \Big[ \mathbf x^\top Q \mathbf x + u^\top R u + 2\mathbf x^\top P (A\mathbf x + Bu) \Big]$$

The bracket is a convex quadratic in $u$, so set its derivative to zero:

$$2Ru + 2B^\top P \mathbf x = 0 \quad\Longrightarrow\quad \boxed{\,u = -R^{-1}B^\top P\, \mathbf x = -K\mathbf x\,}$$

**The optimal controller is linear state feedback.** Nobody assumed that; it
fell out. Substitute it back and use $2\mathbf x^\top PA\mathbf x = \mathbf
x^\top(PA + A^\top P)\mathbf x$, and the $\mathbf x$'s cancel off both sides:

$$\boxed{\,A^\top P + PA - PBR^{-1}B^\top P + Q = 0\,}$$

That is the **continuous-time algebraic Riccati equation (CARE)**. It is a
quadratic matrix equation in the unknown $P$. Solve it, and $K = R^{-1}B^\top P$
is your gain.

For a general system you call a library. For this robot you do it on a
whiteboard.

---

## Proof: P really is the infinite-horizon cost

The derivation above *assumes* $V(\mathbf x) = \mathbf x^\top P\mathbf x$ and
derives $P$ from that assumption. It's worth checking the assumption pays
off — that this $P$ actually equals the cost you get by running $u=-K\mathbf
x$ forever, and that no other control law beats it. Two short calculations
settle both.

**It equals the cost this controller collects.** Differentiate $\mathbf
x^\top P\mathbf x$ along the closed-loop trajectory $\dot{\mathbf x} =
(A-BK)\mathbf x$:

$$\frac{d}{dt}(\mathbf x^\top P\mathbf x)
= \mathbf x^\top\big[(A-BK)^\top P + P(A-BK)\big]\mathbf x$$

Substitute $K = R^{-1}B^\top P$, so $K^\top B^\top P = PBR^{-1}B^\top P$:

$$(A-BK)^\top P + P(A-BK) = A^\top P + PA - 2PBR^{-1}B^\top P$$

The CARE says $A^\top P + PA = PBR^{-1}B^\top P - Q$, so this becomes
$-PBR^{-1}B^\top P - Q = -(Q + K^\top RK)$ — and $\mathbf x^\top(Q+K^\top
RK)\mathbf x$ is exactly the running cost $\ell(\mathbf x,u) = \mathbf
x^\top Q\mathbf x + u^\top Ru$ evaluated at $u=-K\mathbf x$. So:

$$\frac{d}{dt}(\mathbf x^\top P\mathbf x) = -\ell(\mathbf x(t), u(t))$$

Integrate from $0$ to $T$ and use that $A-BK$ is Hurwitz, so $\mathbf
x(T)\to 0$ as $T\to\infty$:

$$\boxed{\,\mathbf x(0)^\top P\,\mathbf x(0) = \int_0^\infty \ell(\mathbf x(t), u(t))\,dt\,}$$

an exact identity, not an estimate: $P$ is literally the infinite-horizon
cost of running $u=-K\mathbf x$ from $\mathbf x(0)$.

**Nothing beats it.** Redo the same derivative for an *arbitrary* control
$u$ (not assuming $u=-K\mathbf x$):

$$\frac{d}{dt}(\mathbf x^\top P\mathbf x)
= -\mathbf x^\top Q\mathbf x + \mathbf x^\top PBR^{-1}B^\top P\mathbf x + 2u^\top B^\top P\mathbf x$$

Add $\ell(\mathbf x,u)$ to both sides — the $\mathbf x^\top Q\mathbf x$ terms
cancel — and use $B^\top P = RK$ to complete the square:

$$\ell(\mathbf x,u) + \frac{d}{dt}(\mathbf x^\top P\mathbf x)
= u^\top Ru + \mathbf x^\top K^\top RK\mathbf x + 2u^\top RK\mathbf x
= (u+K\mathbf x)^\top R\,(u+K\mathbf x) \;\geq\; 0$$

since $R\succ 0$, with equality iff $u=-K\mathbf x$. So $\ell(\mathbf x,u)
\geq -\tfrac{d}{dt}(\mathbf x^\top P\mathbf x)$ for every control.
Integrating to $T\to\infty$ (any finite-cost control must also drive
$\mathbf x\to 0$, killing the boundary term) gives:

$$\int_0^\infty \ell(\mathbf x(t),u(t))\,dt \;\geq\; \mathbf x(0)^\top P\,\mathbf x(0)$$

for *every* admissible control, with equality exactly when $u=-K\mathbf x$.

Together: $u=-K\mathbf x$ *attains* cost $\mathbf x_0^\top P\mathbf x_0$, and
nothing can do better. So that number is both achievable and a lower bound
— it must be the true optimal cost-to-go, and $u=-K\mathbf x$ is the true
optimal controller. This "complete the square, integrate, take $T\to\infty$"
argument is the standard verification-theorem proof for LQR (see Anderson
& Moore, *Optimal Control: Linear Quadratic Methods*); the discrete-time
Riccati recursion this project actually runs (below) follows the identical
pattern with sums replacing integrals.

---

## Solving it by hand

$$A = \begin{bmatrix} 0 & v \\ 0 & 0\end{bmatrix},\quad
B = \begin{bmatrix}0\\1\end{bmatrix},\quad
Q = \begin{bmatrix} q_e & 0 \\ 0 & q_\theta\end{bmatrix},\quad
R = r,\quad
P = \begin{bmatrix} p_1 & p_2 \\ p_2 & p_3 \end{bmatrix}$$

Build the three pieces of the CARE.

$$A^\top P + PA
= \begin{bmatrix}0&0\\v&0\end{bmatrix}\begin{bmatrix}p_1&p_2\\p_2&p_3\end{bmatrix}
+ \begin{bmatrix}p_1&p_2\\p_2&p_3\end{bmatrix}\begin{bmatrix}0&v\\0&0\end{bmatrix}
= \begin{bmatrix} 0 & v p_1 \\ v p_1 & 2 v p_2 \end{bmatrix}$$

$$PBR^{-1}B^\top P = \frac{1}{r}\begin{bmatrix}p_2\\p_3\end{bmatrix}\begin{bmatrix}p_2 & p_3\end{bmatrix}
= \frac{1}{r}\begin{bmatrix} p_2^2 & p_2 p_3 \\ p_2 p_3 & p_3^2\end{bmatrix}$$

Now read off the three independent entries of the CARE.

**(1,1):** $\;0 - \dfrac{p_2^2}{r} + q_e = 0 \;\Longrightarrow\; p_2 = \sqrt{r\,q_e}$

**(2,2):** $\;2vp_2 - \dfrac{p_3^2}{r} + q_\theta = 0 \;\Longrightarrow\; p_3 = \sqrt{r\,q_\theta + 2rv\,p_2}$

**(1,2):** $\;v p_1 - \dfrac{p_2 p_3}{r} = 0 \;\Longrightarrow\; p_1 = \dfrac{p_2 p_3}{rv}$

(Take the positive roots: $P$ must be positive definite for $V$ to be a
sensible cost-to-go.) Notice that the equations *decouple* — (1,1) gives $p_2$
outright, then (2,2) gives $p_3$, then (1,2) gives $p_1$. That is luck, and it
is why this particular robot is such a good teaching example.

The gain is $K = R^{-1}B^\top P = \frac{1}{r}\begin{bmatrix} p_2 & p_3\end{bmatrix}$, so:

$$\boxed{\;k_e = \sqrt{\frac{q_e}{r}}, \qquad
k_\theta = \sqrt{\frac{q_\theta}{r} + 2 v k_e}\;}$$

$$\omega = -k_e\, e - k_\theta\, \theta$$

With $q_e = 1$, $q_\theta = 0$, $r = 25$, $v = 150$ mm/s:

$$k_e = \sqrt{1/25} = 0.200 \ \text{rad/s per mm}, \qquad
k_\theta = \sqrt{2 \cdot 150 \cdot 0.2} = \sqrt{60} = 7.746 \ \text{s}^{-1}$$

The report in `sim.py` solves the same CARE numerically and agrees to four
decimals. Run it here (section 1), or locally with `python3 code/sim.py`:

<div class="pyrun" data-run="sim" data-script="code/sim.py" data-section="1"></div>

---

## Reading the answer

The formula is small enough to interrogate, which is the real reward for doing
it by hand.

**It's PD control.** $\omega = -k_e e - k_\theta \theta$, and since $\theta =
\dot e / v$, this is proportional-plus-derivative on $e$. LQR did not invent a
new structure. It told you *which* PD gains, and why.

**The gains have a ratio you can feel.** $k_\theta / k_e = 7.746/0.2 = 38.7$ mm
per radian, or **0.68 mm per degree**. One degree of heading error is worth the
same to this controller as 0.68 mm of position error. That is a sentence you can
sanity-check against your own intuition about the robot — and if you disagree
with it, you now know exactly which weight to change.

**The damping ratio is a constant.** Substitute $\theta = \dot e/v$ into the
control law and into $\ddot e = v\dot\theta = v\omega$:

$$\ddot e + k_\theta \dot e + v k_e\, e = 0$$

A standard second-order system with

$$\omega_n = \sqrt{v k_e}, \qquad \zeta = \frac{k_\theta}{2\omega_n} = \frac{\sqrt{q_\theta/r + 2vk_e}}{2\sqrt{vk_e}}$$

and when $q_\theta = 0$ the $vk_e$ cancels completely:

$$\zeta = \frac{\sqrt{2vk_e}}{2\sqrt{vk_e}} = \frac{1}{\sqrt2} \approx 0.707$$

**Whatever speed you run, whatever error weight you pick, LQR with no heading
penalty gives you a damping ratio of exactly $1/\sqrt2$.** That is the value
control engineers hand-tune toward — fastest response with essentially no
overshoot. It was not requested and not tuned. It is what minimizing
$\int (e^2/r' + \omega^2)$ *means* for a double integrator.

This is the moment to run the robot. On the web, open the
[3. LQR tab](interactive.html#lqr) and press *Run on robot* (the simulated curve is
there to compare against). Or run the script right here, in simulation:

<div class="pyrun" data-run="lqr" data-script="code/run_lqr.py" data-args="20 150"></div>

If you are running locally, run `python3 code/run_lqr.py`.

For the reference numbers, $\omega_n = \sqrt{150 \times 0.2} = 5.48$ rad/s, so
disturbances settle in $4/(\zeta\omega_n) = 1.03$ s — about 155 mm of travel.
Bump the robot sideways and it should be back on the line within two hand-widths.

---

## Worked example: the three sensor states

Everything above, evaluated at the three things the sensor can see. Reference
build: $R_{\text{black}} = 12$, $R_{\text{white}} = 85$, $e_{max} = 6$ mm,
$b = 96$ mm, wheels reach 400 mm/s at 100%, $v = 150$ mm/s, $k_e = 0.2$,
$k_\theta = 7.746$.

The chain for every row is the same four steps:

$$\text{reflection} \;\xrightarrow{\;s=\frac{R-12}{73}\;}\; s
\;\xrightarrow{\;e = 12(s-\tfrac12)\;}\; e
\;\xrightarrow{\;\omega = -k_e e - k_\theta\theta\;}\; \omega
\;\xrightarrow{\;v \mp \omega b/2\;}\; v_L, v_R$$

Take $\theta = 0$ for now so the arithmetic stays visible.

| Sensor state | refl | $s$ | $e$ (mm) | $\omega$ (rad/s) | $v_L$ | $v_R$ |
|---|---:|---:|---:|---:|---:|---:|
| **On the line** (fully black) | 12.0 | 0.000 | −6.00 | **+1.200** | 23% | 52% |
| Edge, drifted onto the line | 40.0 | 0.384 | −1.40 | +0.279 | 34% | 41% |
| **On the edge**, on target | 48.5 | 0.500 | 0.00 | 0.000 | 37.5% | 37.5% |
| Edge, drifted off the line | 60.0 | 0.658 | +1.89 | −0.378 | 42% | 33% |
| **Off the line** (fully white) | 85.0 | 1.000 | +6.00 | **−1.200** | 52% | 23% |

Sign convention: $e > 0$ means the sensor has drifted onto the background, away
from the tape; a negative $\omega$ steers back toward it.

Work one row by hand to check the machinery. **Row 4**, reflection 60:

$$s = \frac{60 - 12}{73} = 0.658, \qquad e = 12(0.658 - 0.5) = +1.89\ \text{mm}$$
$$\omega = -0.2 \times 1.89 = -0.378\ \text{rad/s}$$
$$v_L = 150 - (-0.378)(48) = 168.1\ \text{mm/s} \to 42\%$$
$$v_R = 150 + (-0.378)(48) = 131.9\ \text{mm/s} \to 33\%$$

### What the saturated rows really mean

Rows 1 and 5 are the interesting ones, and they are where a student's mental
model usually needs correcting.

At full black, the controller commands $\omega = +1.2$ rad/s. At $v = 150$ mm/s
that is a turn of radius $v/\omega = 125$ mm — a firm, sensible correction. But
it is the *same* command whether the robot is 6 mm onto the tape or 60 mm onto
it, because the sensor cannot tell the difference. **The controller has stopped
being a feedback controller and become a constant.** It will still recover, but
only by luck and geometry, and it has no idea how long that will take.

This is the honest statement of the ±6 mm window: outside it, the LQR you just
derived is not running. Its model of the measurement is wrong, its gain is
effectively zero, and the guarantees on the previous page evaporate. Keeping
$|e| \le 6$ mm is not a nice-to-have — it is the condition under which any of
this is true.

Nothing in the LQR formulation lets you say that. $Q$ and $R$ can express "I
would prefer smaller errors." They cannot express "errors above 6 mm are
categorically different." That gap is the subject of [the next page](02-mpc.md).

### Adding heading error

Take row 4 again, but now the robot is also drifting further off at
$\theta = +0.05$ rad (2.9°):

$$\omega = -0.2(1.89) - 7.746(0.05) = -0.378 - 0.387 = -0.765\ \text{rad/s}$$

Twice the correction. A 2.9° heading error contributed as much as the 1.9 mm
position error did — which is the "0.68 mm per degree" exchange rate from
above, made concrete. This is precisely the information a bang-bang or pure-P
controller throws away, and precisely why it wobbles.

---

## The version you actually run: discrete time

The CARE assumes you control continuously. You don't. The Color Sensor pushes a
BLE notification every 50 ms and the motors get a new command in between, so the
loop is a 20 Hz **sampled-data** system. Between samples, the yaw rate is held
constant — a zero-order hold.

Discretize exactly over one period $T$ (for this $A$, $A^2 = 0$, so the matrix
exponential series terminates after two terms and there is nothing to
approximate):

$$A_d = e^{AT} = I + AT = \begin{bmatrix} 1 & vT \\ 0 & 1\end{bmatrix}, \qquad
B_d = \int_0^T e^{A\tau}B\,d\tau = \begin{bmatrix} vT^2/2 \\ T \end{bmatrix}$$

At $v = 150$, $T = 0.05$: $A_d = \begin{bmatrix}1 & 7.5\\0&1\end{bmatrix}$,
$B_d = \begin{bmatrix}0.1875\\0.05\end{bmatrix}$. In words: over one control
period the robot travels 7.5 mm, so a heading error of 1 rad would move it 7.5
mm sideways.

The cost becomes a sum, and the CARE becomes the **discrete algebraic Riccati
equation**:

$$P = A_d^\top P A_d - A_d^\top P B_d\left(R_d + B_d^\top P B_d\right)^{-1}B_d^\top P A_d + Q_d$$

$$K = \left(R_d + B_d^\top P B_d\right)^{-1} B_d^\top P A_d$$

with $Q_d = QT$ and $R_d = RT$ so the weights keep meaning the same thing.

You do not need a library for this. Start at $P = Q_d$ and apply the right-hand
side until it stops changing — it converges in a few dozen iterations, and it is
[eleven lines of numpy](sim.py):

```python
def dlqr(A, B, Q, R, iters=100_000, tol=1e-12):
    P = np.array(Q, dtype=float)
    for _ in range(iters):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P_next = A.T @ P @ (A - B @ K) + Q
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    S = R + B.T @ P @ B
    return np.linalg.solve(S, B.T @ P @ A), P
```

That loop is worth staring at, because it is [the same loop MPC
runs](03-lqr-vs-mpc.md#the-demonstration) — just stopped early.

Sampling costs you gain:

| Control period | $k_e$ | $k_\theta$ | closed-loop pole magnitude |
|---|---:|---:|---:|
| continuous | 0.2000 | 7.746 | — |
| 20 ms (50 Hz) | 0.1851 | 7.452 | 0.925 |
| **50 ms (20 Hz)** | **0.1648** | **7.032** | **0.824** |
| 100 ms (10 Hz) | 0.1361 | 6.390 | 0.681 |
| 200 ms (5 Hz) | 0.0939 | 5.307 | 0.469 |

The slower you sample, the *softer* the optimal gains. That is not a bug — the
discrete design knows it is flying blind between samples and correctly refuses
to push as hard. A student who computes continuous gains and runs them at 5 Hz
will get a robot that oscillates, and will have learned something real about
digital control. (Have them try it. It's a good demo and nothing breaks.)

**Use the discrete gains.** Everything else on this page — the $1/\sqrt2$
damping, the exchange rates, the intuition — is continuous-time and stays true
enough to reason with.

---

## Where does θ come from?

LQR assumes **full state feedback**: $u = -K\mathbf x$ needs all of $\mathbf x$.
You measure one saturating number. You never measure $\theta$ at all.

This gap has a name — the estimation problem — and LQR's usual partner is a
Kalman filter (the pair is called **LQG**). The *separation principle* says you
may design the controller and the estimator independently and combine them,
which is a genuinely surprising theorem and worth stating in class even if you
don't prove it.

For this robot, three options in increasing order of effort:

**1. Finite difference.** Since $\dot e = v\theta$:

$$\hat\theta_k = \frac{e_k - e_{k-1}}{T\,v}$$

One line. Also very noisy: 7.5 mm of travel per sample and maybe ±0.3 mm of
sensor noise gives ±0.04 rad of estimate noise, which $k_\theta = 7$ amplifies
into ±0.28 rad/s of command jitter. Try it, hear it, then improve it.

**2. The IMU.** The Double Motor has one. `motors.imu_device.gyroscopeZ` gives
yaw rate directly, and integrating it gives heading. It is low-noise and fast,
but it drifts, and it measures heading in the *world*, not relative to the line
— so on a curve it reports a turn that is entirely correct and entirely not an
error. (The scale factor is firmware-defined; spin the robot through a known
number of turns and divide. That is a nice ten-minute lab in itself.)

**3. Predict and correct** — what `robot.py` does, and the
right answer here. Predict $\theta$ forward using the yaw rate you *asked for*
(you know it exactly; you commanded it), then nudge it toward what the drift
rate of $e$ implies:

```python
theta_hat += dt * omega_commanded            # predict:  theta_dot = omega
theta_meas = (e - e_prev) / (dt * v)         # correct:  e_dot = v * theta
theta_hat = 0.7 * theta_hat + 0.3 * theta_meas
```

This is a hand-rolled complementary filter — a Kalman filter with the gain
chosen by hand instead of by covariance. The prediction step is smooth and
trusts the model; the correction step is noisy and anchors it to reality; the
0.3 decides who wins.

**One detail matters more than the filter design.** When the sensor is
saturated, $e$ is pinned at ±6 mm, so $e_k - e_{k-1} = 0$, so the correction
step confidently reports $\hat\theta = 0$ — "we're perfectly aligned" — at the
exact moment the robot is sliding off the track. Freeze the correction whenever
the reading is saturated and run on prediction alone:

```python
if not saturated and v > 1.0:
    theta_hat = 0.7 * theta_hat + 0.3 * theta_meas
```

Students will hit this bug. It is a good bug: it teaches that a sensor model
needs to include *when the sensor is lying*, which is the same lesson the
constraint in [MPC](02-mpc.md) encodes.

---

## What LQR gives you, and what it assumes

**What you get:**

- The optimal linear feedback for that model and that cost. Not "good" — optimal,
  provably, over all controllers.
- **Guaranteed closed-loop stability**, provided $(A,B)$ is stabilizable and
  $(A,\sqrt Q)$ is detectable. Both hold here.
- **Robustness for free.** Continuous-time LQR state feedback has an infinite
  gain margin, a −6 dB downward gain margin, and at least 60° of phase margin.
  You get that without asking, which is a large part of why LQR is the default
  in aerospace. *Caveat worth saying out loud:* those margins are for the loop
  broken at the plant input with true state feedback. Add the estimator from the
  previous section and they are no longer guaranteed. (This is the famous
  Doyle 1978 result, "Guaranteed Margins for LQG Regulators: none.")
- Gains computed **offline, once**. Runtime cost is two multiplies.

**What you assumed to get it:**

| Assumption | Where it breaks on this robot |
|---|---|
| Linear model | $\sin\theta \approx \theta$ starts to bite past ~25° (3% error); the robot is well off the line by then anyway |
| Quadratic cost | The real requirement is a hard wall at $\vert e\vert = 6$ mm, not a smooth penalty |
| No constraints | Motors saturate at 100%; the *sensor* saturates at ±6 mm |
| Constant $v$ | $v$ sits inside $A$; one $K$ is correct at exactly one speed |
| Infinite horizon | The controller has no idea a hairpin is coming |
| No preview | Curvature enters as an unmodeled disturbance |
| Full state feedback | You measure one saturating number out of two states |

The bottom four rows are what [MPC](02-mpc.md) fixes. The last one is what a
Kalman filter fixes. Only the first two are things you simply live with.

---

## Labs

On the web, do these on the [3. LQR tab](interactive.html#lqr): the sliders change
$Q$ and $R$ and the plot shows the result, and *Run on robot* tests your own.

1. **Predict, then measure.** Change $r$ from 25 to 100. Before running:
   which way does $k_e$ move, and by what factor? Does the damping ratio change?
   Then run it and watch. *(Answers: $k_e$ halves to 0.1; $\zeta$ does not
   change at all, because $q_\theta = 0$.)*
2. **Find the $1/\sqrt2$.** Run at $r = 25$, bump the robot sideways, and film
   the recovery at 60 fps. Measure the overshoot. For $\zeta = 0.707$ theory
   says 4.3%. How close are you, and what explains the gap?
3. **Break the damping.** Set $q_\theta = 5$. Now $\zeta = \sqrt{5/25 + 60}\,/\,(2\sqrt{30}) = 0.708$
   — barely moved. Set $q_\theta = 500$ instead. Predict $\zeta$, then run it.
   Why is the heading weight so hard to feel?
4. **Sampling.** Compute the continuous gains, run them at a 200 ms notification
   period, and record what happens. Then use the discrete gains at 200 ms.
   Explain the difference in terms of the pole magnitudes in the table above.
5. **The estimator.** Replace the complementary filter with the raw finite
   difference. Record the yaw-rate command over 10 s in both cases and compare
   the standard deviation. Then remove the saturation freeze and drive the robot
   off the line on purpose. Describe what $\hat\theta$ does and why it is wrong.
6. **Speed sweep.** For $v \in \{75, 150, 300\}$, compute $k_e$, $k_\theta$, and
   $\omega_n$. Which gain changes with speed and which doesn't? Explain from the
   formula, then verify that gains from one speed misbehave at another.

---

**Next:** [MPC — planning ahead, and slowing down for the turn](02-mpc.md)
