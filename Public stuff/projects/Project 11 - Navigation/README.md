# Project 11 – Behavior Cloning Navigation

A shoe-mounted UNO Q that talks you to Nolop without GPS or wifi positioning. The board counts steps and
tracks heading from its IMU, and a policy learned by behavior cloning from our own walks decides when to
say "left turn coming up" and "turn left". The phone is only the screen and the speaker; all sensing,
recording, training and inference run on the board.

Status: the UI, recording, label cleanup, training and live policy are done and tested on synthetic walks.
`sensing.py` is a simple placeholder for the real step counter and heading (Partner A's part).

## Contents

1. [How it works](#how-it-works)
2. [Partner A: plugging in sensing](#partner-a-plugging-in-sensing)
3. [Running it](#running-it)
4. [Field workflow for the shared day](#field-workflow-for-the-shared-day)
5. [The policy](#the-policy)
6. [Files and data formats](#files-and-data-formats)
7. [Known limits](#known-limits)

## How it works

```txt
MPU-6050 ─I²C 100 Hz─▶ sketch.ino ─Bridge─▶ main.py ─▶ NavCore (nav_core.py)
                                                     ├─ sensing.py   raw IMU → steps + heading   (Partner A)
                                                     ├─ walks.py     saves every walk to data/walks/
                                                     ├─ features.py  steps → the state the policy sees
                                                     └─ policy.py    state → cue (k-NN), spoken on the phone
phone browser ◀──── web_ui brick, port 7000 ────▶ assets/ (index.html: walk, review.html: label + train)
```

The phone page has four modes:

- **Calibrate**: walk a known distance so step counts can be turned into meters. Both partners and the
  test person calibrate once, which lets everyone's walks line up even though our strides differ.
- **Record**: a demonstration. You walk the route and tap Left / Right / Stairs / Arrived as you start
  each maneuver.
- **Navigate**: the policy speaks the cues. "Save as a test walk" keeps a record for the video and
  the trial with someone who doesn't know the way. Test walks are never trained on.
- **DAgger**: the policy speaks cues while you walk the correct route anyway. You tap what you actually
  do, plus **Wrong cue** whenever it says something wrong. These walks cover states the demos never
  reached, which is the point of DAgger.

The **Review** page (works best on a laptop browser pointed at the board) shows each walk as heading
against time, with the taps, the detected turns and, for DAgger walks, the cues the policy gave. Taps
mid-walk are always late, so **Snap turns to heading** moves each Left/Right to where the heading
actually started to bend. Lines can be dragged, added and deleted. Then **Train policy**.

## Partner A: plugging in sensing

Replace `unoq_app/python/sensing.py`, keeping this interface:

```python
class Sensing:
    VERSION = "your-version-1"     # change it whenever the output changes
    def reset(self) -> None        # called at the start of every walk
    def feed(self, t, rows) -> list[Step]
        # t     board time.time() of the LAST row in the batch
        # rows  [[ax, ay, az, gx, gy, gz], ...] in g and deg/s, 100 Hz, usually 10 rows
        # return the steps detected in this batch (usually none, sometimes one)

@dataclass
class Step:
    t: float            # time of the step, board seconds
    n: int              # 1, 2, 3... since reset()
    heading_deg: float  # heading since reset(), unwrapped (can pass 360); + = turned LEFT
    stairs: int         # 1 on stairs, 0 not, -1 if you don't detect stairs
```

What the rest of the system needs from it, in order of importance:

1. **Turns of about 90° showing up as about 90°**, within a few steps. Turn detection is "heading moved
   60° within 4 m of walking" (`TURN_DEG`, `TURN_SPAN_M` in `config.py`).
2. **A step count that is consistent**, more than one that is exact. Calibration absorbs a constant
   bias (counting every footfall vs. every stride doesn't matter), but random misses don't cancel out.
3. Slow heading drift matters less. The policy mostly uses distance since the last turn and how many
   turns have happened so far.

The baseline is a hysteresis peak detector on |a| for steps. For heading, it takes the gyro along the
gravity direction (so mounting orientation doesn't matter), re-measures the gyro offset whenever the
foot is still during stance, and integrates. The PyTorch CNN from Project 8 can't run on the board as
is: no PyTorch there. Port its forward pass to numpy the way Project 9's `npnet.py` did for FingerNet.

Every walk saves its raw IMU (`imu.csv`), so recorded walks don't have to be redone. Once the new
`sensing.py` is in, press **Reprocess walks** on the Review page to rerun it over all of them, check the
labels still sit on the turns, and retrain.

Test it without the board: `python tools/test_pipeline.py` must still pass, and `python
tools/fake_walk.py` prints the step count and heading your sensing gets on a synthetic walk.

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
```

`dev_board.py` serves the same pages at `http://localhost:7000` and runs the real NavCore. When you
press Start, a fake walk plays and notices say when each true maneuver happens, so you can practice
tapping. Its data goes to `dev_data/` (git-ignored).

## Field workflow for the shared day

The course allows **at most 10 walks** for training, demos and DAgger together. The page counts them
per route ("Training walks used: n/10"). Excluded walks and test walks don't count.

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

**Model**: distance-weighted k-nearest-neighbors (k = 3) on standardized, weighted features, in plain
numpy. With 10 walks there are only a few hundred steps per maneuver class, too few to fit a network,
and k-NN can't invent behavior it never saw. The turn count gets the largest weight because it says
which leg of the route you're on.

**Cues**: a new cue has to be predicted for 2 steps in a row before it is spoken (one step for "now"
right after "soon"), and each maneuver is announced once as soon and once as now.

On synthetic walks (3 demos per walker, two different strides, a test walker with a third stride), the
held-out score was 29/30 maneuvers cued with a mean timing error of 2.8 m. Every maneuver got a heads-up
and nothing fired falsely. Real walks will be worse; the report on the Review page measures that.

## Files and data formats

```txt
unoq_app/
  app.yaml                 web_ui brick, port 7000
  sketch/                  MPU-6050 at 100 Hz, batches of 10 (unchanged from Projects 7/8)
  python/main.py           Bridge + WebUI glue; everything else is in NavCore
  python/nav_core.py       modes, recording, page messages, live policy
  python/sensing.py        raw IMU → steps + heading (Partner A replaces)
  python/features.py       state per step, labels from maneuvers, turn detection and snapping
  python/policy.py         k-NN, cue filter, training, leave-one-walk-out report
  python/walks.py          walk storage, walkers, reprocessing
  python/config.py         every tunable number
  assets/                  index.html + app.js (phone), review.html + review.js, common.js, style.css
  data/                    created on the board: walks/, walkers.json, policy/<route>/
tools/
  dev_board.py             laptop stand-in for the board
  fake_walk.py             synthetic shoe-IMU walks
  test_pipeline.py         end-to-end test
```

Each walk folder `data/walks/<time>_<walker>_<kind>/` holds:

| File | Contents |
| --- | --- |
| `imu.csv` | `t,ax,ay,az,gx,gy,gz`, every raw sample in g and deg/s |
| `steps.csv` | `t,n,heading_deg,stairs`, one row per step |
| `events.jsonl` | taps (`label`, `undo`, `wrong_cue`) and the policy's spoken `cue`s, with board times |
| `meta.json` | walker, kind, route, meters per step, sensing version, reviewed/included, and the cleaned `maneuvers` |

## Known limits

- Everything has been tested on synthetic IMU data and in a browser against the laptop stand-in. Nothing
  has run on the board or on a real walk yet.
- The route is learned as "after this many meters and turns, do this". If a walker misses a turn, the
  turn count is off for the rest of the walk. DAgger walks that include a recovery are the fix.
- Leg lengths vary a few percent between walks, so on long legs "now" often fires just as the turn
  begins rather than before it. The "soon" cue covers that.
- `stairs` is only learned from position until sensing reports a stairs flag.
