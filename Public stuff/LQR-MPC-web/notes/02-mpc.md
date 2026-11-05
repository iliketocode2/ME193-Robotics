# MPC: Planning Ahead, and Slowing Down for the Turn

> **Prerequisite (local version only):** [docs/01-lqr.md](01-lqr.md). This page
> assumes you have an LQR line follower running, that you know its gains came from
> $Q$ and $R$, and that you have watched it slide off a sharp corner.
>
> **On the web there is nothing to build first.** Open the
> [3. LQR tab](interactive.html#lqr), drag the turn radius until the verdict flips to
> BLIND, and you have watched it slide off a sharp corner.

> **Interactive:** [docs/interactive.html](interactive.html) has an MPC tab
> with sliders for the horizon and the speed-deviation weight $q_v$, plotting
> $e(t)$ and $v(t)$ head-to-head against fixed-speed LQR on the same turn.

## Contents

- [Where LQR runs out](#where-lqr-runs-out)
- ["Just turn the gain up"](#just-turn-the-gain-up)
- [The real reason LQR cannot do this](#the-real-reason-lqr-cannot-do-this)
- [What MPC is](#what-mpc-is)
- [The problem, written out](#the-problem-written-out)
- [Worked example: one MPC solve at the corner](#worked-example-one-mpc-solve-at-the-corner)
- [Solving it properly](#solving-it-properly)
- [Where does the preview come from?](#where-does-the-preview-come-from)
- [Head to head](#head-to-head)
- [What MPC costs you](#what-mpc-costs-you)
- [Labs](#labs)

---

## Where LQR runs out

Put the LQR robot on a constant-radius curve and let it settle. Steady state
means nothing is changing, so set both derivatives to zero:

$$\dot e = v\theta = 0 \;\;\Longrightarrow\;\; \theta_{ss} = 0$$

$$\dot\theta = \omega - \kappa v = 0 \;\;\Longrightarrow\;\; \omega_{ss} = \kappa v$$

The robot must be *continuously turning* at $\kappa v$ just to stay on the
curve. But the only thing that produces yaw rate is the control law, and with
$\theta_{ss} = 0$ the heading term contributes nothing:

$$\omega_{ss} = -k_e\,e_{ss} - k_\theta \cdot 0 = \kappa v$$

$$\boxed{\;e_{ss} = -\frac{\kappa v}{k_e} = -\frac{v}{k_e R}\;}$$

**The controller has to sit off the line in order to steer.** The error is not
a failure of tuning — it is the *input* the controller needs in order to
generate the turn. This is the same phenomenon as the steady-state droop of a
proportional controller against a constant load, and no amount of care in
choosing $Q$ and $R$ removes it, because the structure $\omega = -Kx$ has no
way to produce a non-zero output from a zero state.

Now put in the numbers from the reference build ($k_e = 0.2$, $v = 150$ mm/s,
sensor window $\pm 6$ mm):

| Turn radius | $\kappa$ (1/mm) | $e_{ss}$ at 150 mm/s | | Max $v$ for $\vert e\vert \le 6$ mm |
|---:|---:|---:|:--|---:|
| straight | 0.00000 | 0.00 mm | ok | any |
| 300 mm | 0.00333 | −2.50 mm | ok | 360 mm/s |
| 150 mm | 0.00667 | −5.00 mm | ok (barely) | 180 mm/s |
| **80 mm** | 0.01250 | **−9.38 mm** | **blind** | **96 mm/s** |
| **40 mm** | 0.02500 | **−18.75 mm** | **blind** | **48 mm/s** |

The last column is the whole story of this page, and it comes straight from
setting $|e_{ss}| \le e_{max}$:

$$\boxed{\;v_{\max} = k_e\,e_{max}\,R\;}$$

**The tighter the turn, the slower you must go — and the relationship is
linear in the radius.** For this robot, $v_{\max} \approx 1.2 \times R$: a
100 mm-radius corner may be taken at 120 mm/s, and no faster, or the sensor
goes off the end of its usable window and the controller stops being a
controller.

This is a falsifiable prediction about a physical object. Go make it fail.

---

## "Just turn the gain up"

$e_{ss} = \kappa v / k_e$, so doubling $k_e$ halves the offset. Every student
proposes this, and they are right to — it is genuinely the correct first move,
and it works for a while. It is worth doing properly rather than waving away,
because *where* it runs out is instructive.

Two independent ceilings, both computable for the reference build.

**Ceiling 1 — the wheels saturate.** At $|e| = e_{max}$ the commanded yaw rate
is $k_e \cdot 6$ rad/s, and the outer wheel has to run at $v + \omega b/2$.
With $v = 150$ mm/s, $b = 96$ mm, and wheels topping out at 400 mm/s:

$$\omega_{\max} = \frac{2(400 - 150)}{96} = 5.21 \ \text{rad/s}
\quad\Longrightarrow\quad k_e \le \frac{5.21}{6} = 0.87$$

**Ceiling 2 — the delay.** The model in [01-lqr.md](01-lqr.md#step-2-the-model)
pretends the yaw rate you ask for appears instantly. Over BLE, with motors that
have their own dynamics, the command lands roughly one control period late. Add
a single period of input delay to the model and re-check the closed-loop poles
(with $k_\theta = \sqrt{2vk_e}$ tracking $k_e$ as LQR requires):

| $k_e$ | max pole magnitude with 1 period of delay | |
|---:|---:|:--|
| 0.2 | 0.770 | comfortable |
| 0.4 | 0.880 | fine |
| 0.6 | 0.978 | on the edge |
| 0.8 | 1.059 | **unstable** |

So the practical ceiling is around $k_e \approx 0.6$ — a 3× improvement over
the baseline, bought at the price of a barely-damped loop that amplifies sensor
noise by the same 3×.

Now spend it. At $k_e = 0.6$:

- $R = 80$ mm at 150 mm/s: $e_{ss} = 3.1$ mm. **Fixed.** The student was right.
- $R = 40$ mm at 150 mm/s: $e_{ss} = 6.25$ mm. **Still fails.**

And there is the point. Raising the gain **moved** the speed limit; it did not
**remove** it. The limit is still

$$v_{\max} = k_e e_{max} R$$

still proportional to $R$, and $k_e$ is now spent — there is no second 3× to
find. Whatever gain you settle on, there is a corner tight enough to beat it,
and on a track with both a straight and a hairpin you now have to pick one
constant speed that is either too slow for the straight or too fast for the
hairpin.

The variable that always works is $v$. It appears in the numerator. Stop
treating it as a constant.

---

## The real reason LQR cannot do this

Making $v$ a decision variable breaks LQR in two separate ways, and it is worth
being precise about both because they map onto the two things MPC adds.

**1. The model stops being linear.** Recall

$$A = \begin{bmatrix} 0 & v \\ 0 & 0\end{bmatrix}$$

The speed lives *inside the state matrix*. If $v$ is an input, then
$\dot e = v\theta$ is a **product of two decision quantities** — bilinear, not
linear. The "L" in LQR is gone. You can patch around this with gain scheduling
(solve the Riccati equation at ten speeds, interpolate), and people do, but
scheduling only tells you what gains to use *once someone has chosen the speed*.
It does not choose the speed.

**2. The binding requirement is not a cost, it is a wall.** "Stay under 6 mm"
is not "prefer smaller errors." You can approximate it by cranking $q_e$ up,
but a quadratic penalty always trades: given a large enough reward elsewhere, it
will happily accept 8 mm. And 8 mm is not 33% worse than 6 mm on this robot —
it is *categorically different*, because past 6 mm the measurement saturates and
the feedback loop opens. There is no finite $Q$ that encodes a cliff.

So we need a formulation that can (a) treat speed as something to be chosen,
(b) express hard constraints, and (c) look ahead far enough to slow down
*before* the corner rather than after it.

---

## What MPC is

**Model Predictive Control**, in three sentences:

1. From where you are right now, use your model to **simulate forward** over a
   short horizon — say the next 0.75 seconds — for many possible sequences of
   actions.
2. Pick the sequence that **minimizes your cost while satisfying every
   constraint** over that whole horizon.
3. **Apply only the first action.** Throw the rest away. Next control period,
   measure again and re-solve from scratch.

Step 3 is the one that surprises people, and it is the one that makes MPC work.
You did all that planning and then used 5% of it. But the plan was never the
deliverable — the *first move* was. Re-planning every period is what makes the
scheme a feedback controller instead of an open-loop trajectory: model error,
disturbances, and a wheel slipping on dust all get absorbed, because next period
you re-plan from where you actually are rather than where you predicted you'd be.

This is called the **receding horizon**: the planning window slides forward with
you, always looking the same distance ahead, never arriving anywhere.

```
now                                    horizon
 |--------------------------------------|
 [ solve ] -> apply first move only
     |
     +-- 50 ms later ---------------------------|
         |------------------------------------- |
         [ solve again from the new state ] -> apply first move only
```

An MPC "controller" is therefore not a formula. It is an **optimization problem
that gets re-solved twenty times a second.** That is the deepest difference
from LQR, where the optimization happened once, at your desk, months before the
robot moved.

---

## The problem, written out

For the line follower, at each control period, given the current state
$(e_0, \theta_0)$ and a preview of the curvature ahead $\kappa_0 \ldots
\kappa_{N-1}$:

$$\min_{\;v,\;\omega_0 \ldots \omega_{N-1}\;}\quad
\underbrace{\sum_{k=0}^{N-1}\big(q_e e_k^2 + q_\theta \theta_k^2 + r\,\omega_k^2\big)T}_{\text{track the line, don't thrash the wheels}}
\;+\; \underbrace{q_v\,(v - v_{\text{ref}})^2\,NT}_{\text{...but go fast}}$$

subject to, for every $k$ in the horizon:

$$\begin{aligned}
e_{k+1} &= e_k + T\,v\,\theta_k && \text{(dynamics)}\\
\theta_{k+1} &= \theta_k + T(\omega_k - \kappa_k v) && \text{(dynamics)}\\[4pt]
|e_k| &\le e_{max} && \textbf{(stay inside the sensor window)}\\
\left|v \pm \tfrac{\omega_k b}{2}\right| &\le v_{\text{wheel,max}} && \text{(motors are finite)}\\
v_{\min} \le\;& v \;\le v_{\max} && \text{(speed is a decision now)}
\end{aligned}$$

Everything new lives in the constraints. The cost is the same quadratic you
already wrote for LQR, plus one term.

**The new cost term.** $q_v (v - v_{\text{ref}})^2$ is the "go fast" objective:
deviating from your desired cruise speed is itself a cost. Setting
$q_v = 0.01$ per (mm/s)² says *giving up 60 mm/s hurts as much as sitting 6 mm
off the line* — again, an exchange rate you can argue about in physical units.

**The new constraint.** $|e_k| \le e_{max}$ is a **state** constraint, and it is
the thing LQR structurally cannot express. Note it applies at every step of the
horizon, not just now: the optimizer is forbidden from choosing a speed that
will put it outside the window *in half a second*, even though right now
everything is fine. That is what makes it brake before the corner.

**And here is the payoff.** Nowhere in this problem is there a rule that says
"slow down in turns." There is no lookup table of speed versus radius, no
tuning parameter for corner aggressiveness. There is a model, a preference for
speed, and a hard wall. The slowdown is *derived* — it is what the optimizer
discovers when it simulates 150 mm/s through the hairpin, watches the predicted
error walk out past 6 mm, and rejects that branch. Change the tape width and
the behavior changes correctly with no code edit.

---

## Worked example: one MPC solve at the corner

Reference build. The robot is on the straight at $s = 390$ mm, perfectly on
line ($e_0 = 0$, $\theta_0 = 0$), doing 150 mm/s. A 90°, $R = 80$ mm turn
begins at $s = 400$ mm — **10 mm ahead**. The horizon is $N = 15$ steps of
50 ms, which is 0.75 s, which at these speeds is 30–160 mm of look-ahead:
enough to see the corner.

To keep this hand-checkable, restrict the decision to a **single constant speed
over the horizon**, chosen from eight candidates, with the steering $\omega_k$
supplied by the LQR law re-solved at that speed. (This turns the optimization
into eight simulations, which is something you can single-step in a debugger.
[Below](#solving-it-properly) is the version without training wheels.) For each
candidate, roll the model forward 15 steps through the previewed curvature and
score it.

Run it here (section 4 of the report), or locally with `python3 code/sim.py`:

<div class="pyrun" data-run="sim" data-script="code/sim.py" data-section="4"></div>

| Candidate $v$ | Peak $\vert e\vert$ over the horizon | Cost $J$ | |
|---:|---:|---:|:--|
| 40 mm/s | 1.29 mm | 90.91 | feasible, but slow — pays for it in $q_v$ |
| 60 mm/s | 2.86 mm | 61.76 | feasible |
| **80 mm/s** | **4.86 mm** | **40.46** | **← chosen** |
| 100 mm/s | 7.08 mm | 332.09 | violates the window |
| 120 mm/s | 9.39 mm | 8 484.93 | violates the window |
| 150 mm/s | 13.95 mm | 67 383.33 | violates the window |
| 180 mm/s | 19.67 mm | 229 918.71 | violates the window |
| 210 mm/s | 27.09 mm | 754 911.67 | violates the window |

Read the cost column from both ends and you can see the two forces:

- Going **down** from 80 mm/s, tracking gets easier but the $q_v$ term punishes
  the lost speed: 90.91 at 40 mm/s versus 40.46 at 80.
- Going **up** from 80 mm/s, the predicted error crosses 6 mm and the
  constraint penalty explodes — three orders of magnitude between 80 and 150.

80 mm/s is where those two curves cross. **The optimizer chose to slow to 53% of
cruise speed while the corner was still 10 mm ahead and the robot was perfectly
on the line, because it simulated the alternative and didn't like it.**

In the full run it starts backing off even earlier — the first reduction below
150 mm/s comes at $s = 345$ mm, **55 mm and 0.37 s before the turn begins**,
which is the moment the corner first enters the horizon.

Now cross-check against the closed form from the top of this page:

$$v_{\max} = k_e e_{max} R = 0.2 \times 6 \times 80 = 96 \ \text{mm/s}$$

The MPC picked 80 — the largest candidate below 96. **The numerical optimizer
and the pencil-and-paper analysis agree**, which is the single best thing that
can happen in a control course: two completely different methods, one answer.
(Give it a finer candidate grid and it lands nearer 96. Lab 2.)

### Why the horizon length matters

At 150 mm/s the robot covers 7.5 mm per 50 ms period. With $N = 15$ that is
112 mm of preview — the corner at 10 mm is comfortably inside it. Shrink the
horizon to $N = 2$ (15 mm of preview) and the corner is barely visible; the
optimizer sees almost no penalty for staying fast, commits to 150 mm/s, and
then has to brake mid-corner from a position it should never have been in.

**Too short a horizon makes MPC myopic — it degenerates into a reactive
controller with extra steps.** Too long, and you spend compute predicting a
future your model is too crude to get right, and you slow down for corners that
turn out not to be there. The horizon should be roughly the time it takes the
closed loop to settle: from [01-lqr.md](01-lqr.md#reading-the-answer),
$4/(\zeta\omega_n) \approx 1.0$ s, so $N \approx 15$–20 at 50 ms. That is not a
coincidence, and it is the standard rule of thumb.

---

## Solving it properly

Enumerating eight speeds is honest MPC — when the decision is one-dimensional,
enumeration *is* a solver, and it has the enormous classroom advantage that
students can print the table above. But you should know what the grown-up
version looks like.

**Fix $v$, and the problem becomes a QP.** With $v$ held constant, the dynamics
$e_{k+1} = e_k + Tv\theta_k$ are linear again. Stack the states as a linear
function of the input sequence $\mathbf{u} = [\omega_0 \ldots \omega_{N-1}]^\top$:

$$\mathbf{x} = \mathcal{A}\,\mathbf{x}_0 + \mathcal{B}\,\mathbf{u} + \mathcal{K}\boldsymbol{\kappa}$$

and the whole problem collapses to a **quadratic program** — quadratic
objective, linear constraints:

$$\min_{\mathbf u} \; \tfrac12 \mathbf u^\top H \mathbf u + f^\top \mathbf u
\quad \text{s.t.} \quad G\mathbf u \le h$$

which is convex, has a unique solution, and is solved by well-understood
software (OSQP, qpOASES, `cvxpy`) in well under a millisecond at this size.
The outer loop over $v$ stays an enumeration or a line search, because that is
where the bilinearity lives. This structure — *nonconvex in one variable,
convex given it* — is extremely common in practice, and "enumerate the nasty
one, solve the nice one exactly" is a standard, respectable answer.

**Three practical points that matter more than the solver choice:**

*Soft constraints.* A hard $|e_k| \le e_{max}$ can make the problem
**infeasible** — if the robot is already at 7 mm, no input satisfies it, and the
solver returns nothing at all. A controller that returns nothing is worse than a
bad controller. The standard fix, and what [`sim.py`](sim.py) does, is
to soften state constraints into steep penalties:

$$\text{penalty} = \rho \sum_k \max(0,\ |e_k| - e_{max})^2, \qquad \rho = 10^4$$

Always feasible, always returns something, and when it does violate you can see
by how much. Keep *actuator* limits hard — the motors will enforce those
whether you model them or not.

*Terminal cost.* A finite-horizon MPC optimizes the next 0.75 s and is
indifferent to what happens at 0.76 s. That can produce a controller that
cheerfully drives itself into a corner just past the horizon — finite-horizon
MPC is **not** automatically stabilizing. The standard fix is a terminal cost
$\mathbf x_N^\top P \mathbf x_N$ that stands in for "and behave well forever
after," and the canonical choice of $P$ is **the solution of the Riccati
equation you already computed for LQR**. The two methods are not rivals; LQR
ends up inside MPC, holding up the far end of the horizon.

*Warm starting.* Last period's solution, shifted forward one step, is an
excellent initial guess for this period's. It typically cuts solve time several
fold, and it is free.

---

## Where does the preview come from?

MPC needs to know $\kappa_0 \ldots \kappa_{N-1}$ — the curvature ahead. A robot
with one downward-facing light sensor cannot see the future. This is the
weakest joint in the whole scheme and students should be told so plainly.

Four answers, in increasing order of honesty about cost:

**1. A track map.** If the track is known and you have odometry (the Double
Motor reports wheel positions), you know your arclength $s$ and can look up
$\kappa(s)$. This is what [`sim.py`](sim.py) does, and it is legitimate
— warehouse AGVs work exactly this way. It is also the right classroom default:
hand out the map, keep the lesson about MPC rather than about perception.

**2. Mount the sensor forward.** A sensor 40 mm ahead of the wheel axle at
150 mm/s is reporting where the robot *will be* in 0.27 s. That is real preview,
it costs one LEGO beam, and it is why line-follower geometry is not arbitrary.
It buys about a third of the horizon.

**3. Estimate the current curvature and assume it persists.** The controller
knows what yaw rate it is holding, so

$$\hat\kappa = \frac{\omega}{v} \quad \text{(low-pass filtered)}$$

Assume the road ahead keeps the curvature the road behind had. This is what
`run_mpc.py` does, because it needs no map and no extra
hardware.

Strictly this is not preview at all — it is *constraint awareness with a
persistence assumption*, and it cannot brake before a corner it has not entered.
But it works better than it deserves to, for a reason worth pointing out in
class: **the failure mode we are guarding against is a steady-state effect.**
$e_{ss} = \kappa v / k_e$ depends only on the curvature *now*. So a controller
that knows the current curvature already knows most of what it needs, and the
penalty for lacking true preview is a brief error excursion at corner entry
rather than a qualitative failure.

**4. Look at the track.** A camera, or a second sensor out on an outrigger, gives
genuine preview. This is what real autonomous vehicles do, and it is a fine
capstone extension — at which point you have rebuilt, in LEGO, the actual
architecture of a self-driving car's lateral controller.

---

## Head to head

Same track for both: 400 mm straight → 90° turn at $R = 80$ mm → 400 mm
straight. Same steering law, same $Q$ and $R$. The only difference is that one
holds $v = 150$ mm/s and the other re-decides $v$ every 50 ms. Run it here (section 5), or locally with `python3 code/sim.py`:

<div class="pyrun" data-run="sim" data-script="code/sim.py" data-section="5"></div>

| | Peak $\vert e \vert$ | Fraction of run blind | Lap time | Mean speed |
|---|---:|---:|---:|---:|
| **LQR**, fixed $v = 150$ | 18.95 mm | 22.6% | 6.20 s | 150 mm/s |
| **MPC**, $v$ scheduled | 6.19 mm | 4.9% | 7.10 s | 131 mm/s |

MPC's speed ranged from 60 mm/s in the corner to 150 mm/s on the straights.

Read this honestly, because the honest reading is the interesting one.

**MPC's lap is 0.9 s slower — 15% worse.** If the only metric is lap time on a
track you always complete, fixed-speed LQR wins and MPC is an expensive way to
lose. That is a real result and students should not be shielded from it.

**But the LQR robot spent 22.6% of the run with its sensor pinned**, peaking at
19 mm off the line — three sensor windows out. For that fifth of the run it was
not being controlled at all; it was executing a fixed turn and hoping. In the
simulator, geometry rescues it. On a real track with 19 mm tape, a robot 19 mm
off the line has the tape entirely outside the sensor's field of view and is
*gone*. Its "faster lap" is partly fiction.

So the comparison is not "MPC is better." It is:

> **LQR bought speed by spending safety margin it never knew it had. MPC spent
> 15% of the speed to buy the margin back, deliberately and by a stated
> exchange rate you chose.**

Which is the better trade depends entirely on $q_v$, and $q_v$ is yours to set.
That is the point: MPC does not decide the tradeoff. It *exposes* it, in units,
where you can argue about it. This is why the capstone scores lap time *plus a
penalty per blind sample* — under that metric the tradeoff is explicit and MPC
wins on the merits rather than by assertion.

**Two more honest details**, both visible if you print the speed each period:

*The peak error is 6.19 mm — slightly over the 6 mm limit it was told to
respect.* That is the soft constraint doing its job: the penalty is steep, not
infinite, and at corner entry the optimizer accepted 0.19 mm of violation rather
than crawl. Ask students whether they consider that a bug. It isn't, but knowing
why is the difference between using a solver and understanding one.

*The speed chatters.* Mid-corner the choice flips 80 → 60 → 80 → 60 → 80 over
about a fifth of a second. This is an artifact of the coarse candidate grid: the
true optimum sits between two candidates, and tiny state changes tip the
comparison back and forth. Real MPC implementations suppress it by penalizing
*change* in speed — add $q_{\Delta v}(v_k - v_{k-1})^2$ to the cost — which is
both the standard fix and a nice illustration that "smooth actuation" is
something you ask for in the objective rather than bolt on afterwards. Lab 8.

---

## What MPC costs you

Nothing here is free, and a course that presents MPC as a strict upgrade is
teaching a marketing slide.

| | LQR | MPC |
|---|---|---|
| **Runtime work** | 2 multiplies | 8 rollouts × 15 steps here; a real QP in production |
| **Things to tune** | $Q$, $R$ | $Q$, $R$, plus $N$, $q_v$, terminal cost, constraint softening |
| **Can it fail to produce output?** | No | Yes — infeasibility is a real failure mode needing a fallback |
| **Stability guarantee** | Yes, unconditionally, for the LTI model | Only with a proper terminal cost/constraint |
| **Needs a forecast?** | No | Yes, and it is only as good as your preview |
| **Debuggable by inspection?** | Yes — two numbers | Not really; you debug a solver's output |

That last row is not a small thing in a teaching context. When an LQR robot
misbehaves, a student can read two gains and reason about them. When an MPC
robot misbehaves, the answer is somewhere inside an optimization that ran 20
times a second, and the diagnostic skill required is different and harder.
Printing the candidate table every period, as `run_mpc.py`
does, is the minimum viable observability.

**Use MPC when** you have hard constraints that genuinely bind, useful preview,
or a decision (like speed) that changes the model itself.

**Use LQR when** you don't. It is not the beginner's version of MPC — for an
unconstrained linear regulation problem it is the *exact* answer, and MPC will
at best reproduce it after doing far more work. Which is exactly what
[the next page](03-lqr-vs-mpc.md) demonstrates numerically.

---

## Labs

On the web, use the [5. MPC tab](interactive.html#mpc): sliders for the horizon and
$q_v$ plot MPC against fixed-speed LQR, and *Run on robot* tests your own.

1. **Shrink the horizon.** Run the MPC with $N = 15$, then 5, then 2. At what
   horizon does the robot stop braking early and start braking late? Relate
   your answer to the settling time $4/(\zeta\omega_n)$ from
   [01-lqr.md](01-lqr.md#reading-the-answer).
2. **Refine the grid.** The example uses eight candidate speeds and picks 80
   mm/s where theory says 96. Change the candidates to steps of 10 mm/s and
   confirm it moves to 90. What is the cost of a finer grid, and where would
   you rather spend that compute?
3. **Change $q_v$.** Set it to 0.001, then 0.1. Predict the effect on the corner
   speed *before* running. Then find the $q_v$ that reproduces plain LQR
   behavior, and explain what that value means.
4. **Widen the tape.** Rebuild the track with 25 mm tape instead of 19 mm,
   recalibrate $e_{max}$, and change *nothing else*. Predict the new corner
   speed from $v_{\max} = k_e e_{max} R$, then confirm the MPC finds it on its
   own. This is the demonstration that the slowdown was derived, not coded.
5. **Break the preview.** Run the map-based MPC with the map deliberately
   shifted 200 mm out of phase with reality. What does it do, and what does that
   tell you about how much of MPC's advantage is the optimizer versus the
   forecast?
6. **Hard versus soft.** Make the state constraint hard and drive the robot to
   an infeasible state by hand (park it 8 mm off the line before starting).
   Watch the solver fail. Then implement a fallback — "if infeasible, use the
   slowest candidate" — and argue about whether that is engineering or cheating.
7. **Terminal cost.** Add $\mathbf x_N^\top P\mathbf x_N$ using the $P$ from
   your DARE solution. Does it change behavior at $N = 15$? At $N = 3$?
   Explain why the effect is larger at short horizons.
8. **Kill the chatter.** Reproduce the 80/60/80/60 flip-flop mid-corner, then
   add a rate penalty $q_{\Delta v}(v_k - v_{k-1})^2$. Find the smallest
   $q_{\Delta v}$ that stops it. Compare against the cheaper hack — requiring a
   candidate to beat the current speed by some margin before switching — and
   say which one you would ship and why.

---

**Next:** [The difference in one page](03-lqr-vs-mpc.md)
