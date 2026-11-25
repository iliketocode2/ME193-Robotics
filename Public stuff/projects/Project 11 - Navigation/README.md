# Project 11 – Behavior Cloning Navigation

A shoe-mounted UNO Q that talks you to Nolop without GPS or wifi positioning. The board counts steps and
tracks heading from its IMU, and a policy learned by behavior cloning from our own walks decides when to
say "left turn coming up" and "turn left". The phone is only the screen and the speaker; all sensing,
recording, training and inference run on the board.

Status: running on the board. Julian's first real walks (Nov 23) led to three sensing fixes and
combined cues for maneuvers that come close together. They're described under
[Changes from the first board tests](#changes-from-the-first-board-tests).

**After pulling these changes**, retrain the policy on the Review page (the list of cue classes changed,
so an older saved policy is ignored) and press **Reprocess walks** (the sensing version changed).

## Contents

1. [How it works](#how-it-works)
2. [Sensing: steps, heading, stairs](#sensing-steps-heading-stairs)
3. [Changes from the first board tests](#changes-from-the-first-board-tests)
4. [Hardware: the shoe case](#hardware-the-shoe-case)
5. [Running it](#running-it)
6. [Field workflow for the shared day](#field-workflow-for-the-shared-day)
7. [The policy](#the-policy)
8. [Files and data formats](#files-and-data-formats)
9. [Known limits](#known-limits)

## How it works

```txt
MPU-6050 ─I²C 100 Hz─▶ sketch.ino ─Bridge─▶ main.py ─▶ NavCore (nav_core.py)
                                                     ├─ sensing.py   raw IMU → steps, heading, stairs
                                                     │    ├─ foot_ins.py     foot tracking, stance phases
                                                     │    ├─ stepnet_np.py   the retrained step network
                                                     │    └─ stairs.py       stairs classifier
                                                     ├─ walks.py     saves every walk to data/walks/
                                                     ├─ features.py  steps → the state the policy sees
                                                     └─ policy.py    state → cue (k-NN), spoken on the phone
phone browser ◀──── web_ui brick, port 7000 ────▶ assets/ (index.html: walk, review.html: label + train)
```

The phone page has four walking modes, plus sensor recordings on the Calibrate tab:

- **Calibrate**: walk a known distance so step counts can be turned into meters. Both partners and the
  test person calibrate once, which lets everyone's walks line up even though our strides differ.
- **Record**: a demonstration. You walk the route and tap Left / Right / Stairs / Arrived as you start
  each maneuver.
- **Navigate**: the policy speaks the cues. "Save as a test walk" keeps a record for the video and
  the trial with someone who doesn't know the way. Test walks are never trained on.
- **DAgger**: the policy speaks cues while you walk the correct route anyway. You tap what you actually
  do, plus **Wrong cue** whenever it says something wrong. These walks cover states the demos never
  reached, which is the point of DAgger.
- **Sensor data** (on the Calibrate tab): a recording of one activity (walking, stairs, turns, standing)
  for retraining the step counter and stairs classifier. Never counted against the 10 walks.

The **Review** page (works best on a laptop browser pointed at the board) shows each walk as heading
against time, with the taps, the detected turns and, for DAgger walks, the cues the policy gave. Taps
mid-walk are always late, so **Snap turns to heading** moves each Left/Right to where the heading
actually started to bend. Lines can be dragged, added and deleted. Then **Train policy**.

## Sensing: steps, heading, stairs

`sensing.py` turns 100 Hz IMU samples into steps for the policy. Each step carries a heading and a stairs
flag. **One step is one stride of the foot wearing the board** (two footsteps); calibration converts it
to meters, so consistency matters more than the convention. The navigator needs, in order: turns of
about 90° to read as about 90°, a consistent step count, and only then slow heading drift (the policy
mostly uses distance since the last turn and the number of turns).

### Foot tracking (`foot_ins.py`)

A shoe stops completely on every stride. For a few tenths of a second the foot is flat on the ground
(the stance phase). The tracker follows the board's full 3D orientation from the gyro, and at every
stance it uses that stillness three ways:

- velocity must be zero, which wipes out the drift built up during the swing
- the accelerometer reads pure gravity, which levels the orientation estimate
- the gyro should read zero, so whatever it reads is its offset, and that gets re-measured

The foot counts as still when it rotates slower than 65°/s and its acceleration is within 0.18 g of
gravity, for 6 samples in a row. (These started at 40°/s and 0.12 g, which real stair descents never
met. See [the board-test changes](#changes-from-the-first-board-tests).)

A stance counts as a new stride only if the foot swung (rotated faster than 150°/s) since the last one.
The jolt when the *other* foot lands briefly breaks a stance but doesn't swing this foot, so it isn't
counted.

At each stride it reports heading (+ = left, measured from the start of the walk, since there's no
compass), horizontal length and height change. None of this depends on how the board is mounted. It
also produces four mount-independent signals for the step network: vertical and horizontal
acceleration, and vertical and horizontal rotation rate.

On six held-out simulated walks with random mountings, it found every stride, held heading within 3°,
measured distance within 1.5% and stair heights within 0.07 m.

### Step network (`stepnet_np.py`, trained by `tools/train_stepnet.py`)

This is Project 8's CNN architecture, retrained. Project 8's labels mixed two conventions: some sessions
were marked every footstep, others every stride. Run on its own recordings, the old network counted
0.56 s apart on some walks and 1.36 s apart on others. Now every recording is labeled the same way by
the foot tracker (one label at the start of each stance after a real swing). The input is the tracker's
four mount-independent channels instead of raw axes, so a new housing doesn't invalidate the model. It
trains in PyTorch on a laptop, batch norm is folded into the weights, and the board runs a numpy copy
that matches PyTorch to about 1e-7. Project 8's debouncer turns its output into steps, with a 0.6 s
minimum gap.

Results (`results/stepnet.md`, `tools/test_sensing.py`):

| Recording | Network | Tracker alone | Truth |
| --- | --- | --- | --- |
| 4 held-out simulated walks (stairs, turns, 4 stride lengths) | exact on all 4 | exact | known |
| Project 8 50-step trial (real, held out) | 29–30 strides | 26 | ~25 (50 footsteps) |
| Project 8 turning session (real, trained on) | 24 | 24 | ~23 (46 footsteps) |

On the only real held-out recording the tracker alone (+4%) beats the network (+16–20%). That recording
is from Project 8's mount, which wobbled (see below), so retrain on the new mount before choosing.
`STEP_SOURCE` at the top of `sensing.py` switches between `"net"` and `"tracker"`, and **Reprocess
walks** reruns either one over every recorded walk, so the choice can wait.

### Stairs (`stairs.py`, trained by `tools/train_stairs.py`)

On stairs, the instrumented foot rises or falls about two stairs (~0.34 m) per stride and moves less far
forward. A logistic regression on height change, stride length, stride time and swing time decides each
stride. The current model is trained on simulated strides plus Julian's real board sessions: 98.9% on
held-out strides, and no strides on flat walks flagged as stairs. If the tracker's median stride is shorter than 0.4 m, the board
isn't tracking the foot properly, so the flag reports −1 (unknown) instead of guessing.

### Mounting the board (this matters for the housing)

Everything above assumes the board is **rigid on the foot**: on top of the shoe over the laces or
midfoot, strapped so it can't rock or slide. Project 8's recordings show what happens otherwise. During
stance that sensor kept turning at 20–60°/s and its swing barely registered, so the tracker measured
strides of 3 cm. Step counting still worked, but stride length and stairs did not. So: not on the ankle,
and not loose on a strap.

### Retraining on the new mount (before or on the shared day)

1. Record **Sensor data** sessions on the board: 2–3 × walking (60 s), 2 × stairs up and down a full
   flight, 1 × turns, 1 × standing and fidgeting.
2. Review → **Export with raw IMU**, and save the JSON next to this README.
3. On a laptop (the step network needs PyTorch: `tools/requirements-train.txt`, already in `my_env_cpu`):

   ```
   my_env_cpu/Scripts/python tools/train_stepnet.py --data p8 sim export.json --plots
   python tools/train_stairs.py --data sim export.json
   ```

   Check `results/label_checks/*.png`: the green lines (labels) should sit at the start of each still
   period after a swing.
4. Copy `unoq_app/` to the board again, then Review → **Reprocess walks** if any walks were already
   recorded.

`python tools/test_sensing.py` checks the tracker, network and stairs against simulated truth and
Project 8's counts. `python tools/test_pipeline.py` checks that the navigator still works on top.

## Changes from the first board tests

Julian's first runs on the UNO Q, with the MPU-6050 on a shoe, turned up three problems. Each fix was
checked on the board recordings and on the simulator (commit `13cbdd6`, merged in PR #5).

| Problem | Fix | Before → after |
| --- | --- | --- |
| Going down real stairs the foot never got still enough to count as a stance, so strides went missing | `foot_ins.py`: stance thresholds 40°/s, 0.12 g → 65°/s, 0.18 g | strides found on two descents: 7/13 and 5/12 → 13/13 and 11/12 |
| The stairs classifier, trained only on simulated feet, flagged flat strides | `models/stairs.json` retrained on simulated + real board sessions | flat strides flagged: 5 → 0; 98.9% on held-out strides |
| Taking stairs down quickly, neither the network nor the tracker saw about 7 strides per flight | `sensing.py`: swing fill-in (below) | the same route measured 63 / 53 / 49 m → 63 / 63 / 61 m; other recordings unchanged |
| A left turn 1–2 m before the stairs was never announced (the stairs cue swallowed it) | combined cues (see [The policy](#the-policy)) | leave-one-walk-out on 3 real demos: 71% → 92% of maneuvers cued, every one with a heads-up, no false alarms |

**Swing fill-in.** Every stride has one sharp rotation spike while the foot swings, even when the foot
never comes to rest. A spike of at least 200°/s that is the fastest rotation within 0.25 s either side,
and came 0.5–1.6 s after the previous spike (a walking rhythm), is checked 1.5 s later. If the network
reported no step from 0.3 s before it to 0.9 s after, a step is added 0.45 s after the spike (the
typical time from spike to stride end on the real walks), as long as it isn't within 0.5 s of another
step. `FILL_IN = False` at the top of `sensing.py` turns this off. When it's on, the sensing version ends
in `-fill`, so walks recorded before it show as needing reprocessing.

The known cost: `tools/test_sensing.py` now has one failure. On simulated walk seed 703, a slow descent
with long strides, the step count is still exact (85/85), but only 7 of the 14 stair strides are flagged
as stairs. We accepted that in favor of the real data.

## Hardware: the shoe case

Two printed parts: the existing lid (`case_top_embossed.stl`, unchanged) and a bottom tray
(`hardware/case_tray.stl`) built by `hardware/make_case.py` (`pip install manifold3d`). Every dimension
is a named constant at the top of the script. Running it rewrites the STL, `preview_top.png`,
`preview_3d.png` and `fit_report.txt`, which checks that the lid, board and IMU don't collide.

![The tray, exploded and assembled](hardware/preview_3d.png)

From the shoe up: a 2.4 mm floor with a pocket for the MPU-6050 breakout sunk into it, standoffs at
the four Uno mounting holes, the UNO Q 5 mm above the floor, and the lid. The lid's rim surrounds the
board and rests on a ledge inside the tray, and the tray walls come up around it with 0.8 mm left to
grab. The USB-C side has an opening sized for a plug body. The other short side has a channel, at the
lid's second gap, that carries the IMU wires from under the board up to its top surface. A groove in
the floor leads there from the IMU pocket. Overall 100 × 61 mm including the loops, 14 mm tall.

**Strap loops**: both short ends, at shoe level. Each one flares out of the wall with curved fillets and
arcs down to a 3 mm bar, with a 26 × 4 mm slot whose edges are softened. It fits a 20–24 mm watch band or
1-inch masking tape. Thread one band down through a slot, under the sole, up through the other slot and
back over the lid: that holds the lid on too.

**Printing**: tray floor-down, no supports (the slots are vertical). The lid prints plate-down as it
comes.

**Assembly**:

1. Glue the IMU breakout into its pocket (hot glue or double-sided foam tape). It must not move.
   The pocket is sized for a GY-521 (21.2 × 16.4 mm) with no header pins; solder wires directly or
   change `IMU_L`/`IMU_W` for another board.
2. Run its wires along the groove and up the channel.
3. Screw the UNO Q onto the standoffs with four M3 self-tapping screws, 6–8 mm long (2.6 mm pilot
   holes).
4. Press the lid on.

**Check before printing**: the standoff positions are the standard Uno R3 holes. Hold the board over
`preview_top.png` (or measure with calipers) to confirm the UNO Q matches. Also make sure nothing on the
board's underside is taller than about 3.5 mm where the IMU sits.

## Running it

**On the board.** Copy `unoq_app/` into Arduino App Lab and run it, as in Project 7. Put the phone on
the same network as the board (the phone's hotspot works outdoors), then open `http://<board name or
IP>:7000`. Before walking:

- Set the phone's auto-lock to Never. The page can't keep the screen awake over plain http.
- Turn the volume up and the silent switch off. Cues are spoken.
- The board records on its own. If the page drops mid-walk, reconnect; taps queued on the phone are
  resent and duplicates are ignored.

**On a laptop, without the board.**

```
pip install -r tools/requirements-dev.txt
python tools/dev_board.py --speed 4          # fake walks of a 5-maneuver route, 4x speed
python tools/dev_board.py --replay <walk>/imu.csv   # replay a real recorded walk instead
python tools/test_pipeline.py                # end-to-end test: calibrate, demos, review, train, navigate, DAgger
python tools/test_sensing.py                 # foot tracker, step network, stairs vs simulated truth + Project 8
```

`dev_board.py` serves the same pages at `http://localhost:7000` and runs the real NavCore. When you
press Start, a fake walk plays and notices say when each true maneuver happens, so you can practice
tapping. Fake walks come from `tools/foot_sim.py`, which moves a simulated foot through space (stance,
swing, pitch, turns, stair rises, board at a random angle) and derives what the IMU would read. Its data
goes to `dev_data/` (git-ignored).

## Field workflow for the shared day

The course allows **at most 10 walks** for training, demos and DAgger together. The page counts them
per route ("Training walks used: n/10"). Excluded walks and test walks don't count.

0. Before the day, on the final housing: record the Sensor data sessions and retrain (see above).
1. Both partners calibrate (40 m or more between two landmarks).
2. **3 demos each** (6 walks). After each one, review it: Snap turns, check every maneuver is there
   once, Save.
3. Train. The report shows the leave-one-walk-out score; aim for every maneuver cued.
4. **1–2 DAgger walks** wherever the report shows misses. Review, retrain.
5. Keep the remaining walks in reserve. Then do test walks (Navigate, "Save as a test walk"): each of
   us, then someone who doesn't know the route, who first calibrates their own stride.
6. Review → **Export JSON** downloads everything (walks, labels, policy report) for the Notion plots.

## The policy

**State** (computed per step by `features.StateTracker`, identically in training and live):
meters walked, meters since the last detected turn, number of turns so far, heading since the start,
direction of the last turn, and the stairs flag.

**Actions**: `straight`, or `<maneuver>_soon` / `<maneuver>_now` for left, right, stairs and arrive.
Labels come from the reviewed maneuver times. Steps from 3 m before a maneuver to 2 m after are
`_now`, and the 8 m before that `_soon`. That window gives the heads-up, and it lets "now" fire as soon
as the gyro feels a turn begin when distance alone isn't precise enough.

**Combined maneuvers**: a "now" zone spans 5 m (3 m before to 2 m after), so when two maneuvers are
closer than `COMBINE_M` (6 m) the later one's labels used to overwrite the earlier one's, and the
earlier one was never cued. Close pairs are now one maneuver, named like `left+stairs`, labeled at the
first one's place and spoken together ("Left turn, then stairs coming up", then "Turn left, then take
the stairs"). Which pairs to combine is decided once per route (`features.route_plan`): if every
training walk has the same maneuver sequence, a pair is combined when its median gap over those walks
is under 6 m, so every walk gets labeled the same way. A walk whose sequence doesn't match falls back to
its own gaps. Pairs are taken in order, so three close maneuvers become a pair plus a single. The phone
shows a combined cue with the first maneuver's arrow and the name "Left + Stairs", and the Review page's
report names it the same way.

**Model**: distance-weighted k-nearest-neighbors (k = 3) on standardized, weighted features, in plain
numpy. With 10 walks there are only a few hundred steps per maneuver class, too few to fit a network,
and k-NN can't invent behavior it never saw. The turn count gets the largest weight because it says
which leg of the route you're on.

**Cues**: a new cue has to be predicted for 2 steps in a row before it is spoken (one step for "now"
right after "soon"), and each maneuver is announced once as soon and once as now.

On simulated walks (3 demos per walker, two different strides, a test walker with a third stride, all
through the real sensing), the held-out score was 30/30 maneuvers cued with a mean timing error of
2.5 m. Every maneuver got a heads-up and nothing fired falsely. On Julian's 3 real demos, with combined
cues and the sensing fixes, it was 92% cued, all with a heads-up, and no false alarms.

## Files and data formats

```txt
unoq_app/
  app.yaml                 web_ui brick, port 7000
  sketch/                  MPU-6050 at 100 Hz, batches of 10 (unchanged from Projects 7/8)
  python/main.py           Bridge + WebUI glue; everything else is in NavCore
  python/nav_core.py       modes, recording, page messages, live policy
  python/sensing.py        raw IMU → steps, heading, stairs (STEP_SOURCE switch at the top)
  python/foot_ins.py       foot tracking: orientation, stance phases, stride length/height/heading
  python/stepnet_np.py     the step network in numpy + Project 8's debouncer
  python/stairs.py         stairs classifier on stride features
  python/models/           stepnet.npz, stairs.json (from the training scripts)
  python/features.py       state per step, labels from maneuvers, combined maneuvers, turn detection and snapping
  python/policy.py         k-NN, cue filter, training, leave-one-walk-out report
  python/walks.py          walk storage, walkers, reprocessing
  python/config.py         every tunable number
  assets/                  index.html + app.js (phone), review.html + review.js, common.js, style.css
  data/                    created on the board: walks/, walkers.json, policy/<route>/
tools/
  dev_board.py             laptop stand-in for the board
  foot_sim.py              physically consistent foot-IMU simulator with known truth
  fake_walk.py             foot_sim walks split into batches, with late taps, for the tests and dev board
  sensor_data.py           loads Project 8 sessions, board walks, exports and simulations; tracker labels
  train_stepnet.py         retrains the step network (PyTorch), exports and checks the numpy copy
  train_stairs.py          trains the stairs classifier
  test_pipeline.py         end-to-end navigator test
  test_sensing.py          sensing tests
training_data/p8/          Project 8's five recorded sessions
results/                   stepnet.md, stepnet_training.png (label_checks/ is git-ignored)
```

Each walk folder `data/walks/<time>_<walker>_<kind>/` holds:

| File | Contents |
| --- | --- |
| `imu.csv` | `t,ax,ay,az,gx,gy,gz`, every raw sample in g and deg/s |
| `steps.csv` | `t,n,heading_deg,stairs`, one row per step |
| `events.jsonl` | taps (`label`, `undo`, `wrong_cue`) and the policy's spoken `cue`s, with board times |
| `meta.json` | walker, kind, route, meters per step, sensing version, reviewed/included, and the cleaned `maneuvers` |

## Known limits

- The stairs classifier now includes real board data, but the step network has still only seen
  Project 8's wobbly mount and simulated feet. Retrain it on Sensor data from the new housing if step
  counts look off.
- The swing fill-in was tuned on a handful of real walks. On simulated seed 703 it costs half the
  stairs flags on one slow descent (see above).
- Combining needs every training walk to share the same maneuver sequence. A demo with a missing or
  extra tap makes the route plan fall back to per-walk gaps, which can label walks inconsistently, so
  review taps carefully.
- Sensing costs about 25–80 ms of laptop CPU per second of walking. The board's ARM cores are slower;
  check that the page stays responsive during a walk.
- The route is learned as "after this many meters and turns, do this". If a walker misses a turn, the
  turn count is off for the rest of the walk. DAgger walks that include a recovery are the fix.
- Leg lengths vary a few percent between walks, so on long legs "now" often fires just as the turn
  begins rather than before it. The "soon" cue covers that.
