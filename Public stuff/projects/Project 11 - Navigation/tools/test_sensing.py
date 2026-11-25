"""Tests for Partner A's half: foot tracker, step network, stairs classifier, sensing.

Simulated walks have exact truth. Project 8's real sessions only have human
footstep counts (both feet), so there the check is "about half as many strides".

    python tools/test_sensing.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "unoq_app" / "python")]

import fake_walk as fw          # noqa: E402
import foot_ins as fi           # noqa: E402
import foot_sim as fs           # noqa: E402
import sensor_data as sd        # noqa: E402
from sensing import Sensing     # noqa: E402

ROUTE = [("walk", 30), ("left",), ("walk", 20), ("stairs", 14, 1), ("walk", 8), ("right",), ("walk", 25),
         ("stairs", 14, -1), ("walk", 10), ("left",), ("walk", 15), ("arrive",)]
failures = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        failures.append(what)


def run_sensing(sim, **kw):
    s = Sensing(**kw)
    t = sim["t"]
    steps = []
    for k in range(0, len(t), 10):
        steps += s.feed(float(t[min(k + 9, len(t) - 1)]), sim["imu"][k:k + 10].tolist())
    return s, steps


def main():
    print("foot tracker vs simulated truth (6 walks, random mounts, 3 stride lengths, held-out seeds)")
    for seed in range(500, 506):
        sim = fs.simulate(ROUTE, stride_m=[1.2, 1.4, 1.6][seed % 3], seed=seed)
        truth = sim["truth"]["strides"]
        strides, _, _, _ = fi.run(sim["t"], sim["imu"])
        n = min(len(strides), len(truth))
        herr = max(abs(strides[i].heading_deg - truth[i]["heading_deg"]) for i in range(n))
        dist = sum(s.length for s in strides) / sum(s["length"] for s in truth)
        st = [i for i in range(n) if truth[i]["stairs"]]
        dz_err = max(abs(strides[i].dz - truth[i]["dz"]) for i in range(n))
        check(len(strides) == len(truth) and herr < 5 and abs(dist - 1) < 0.05 and dz_err < 0.08 and st,
              f"seed {seed}: {len(strides)}/{len(truth)} strides, heading error <= {herr:.1f} deg, "
              f"distance x{dist:.3f}, height error <= {dz_err:.2f} m")

    print("same walk, three different board mountings")
    counts, headings = [], []
    for k in range(3):
        sim = fs.simulate(ROUTE, seed=600, mount=fs.random_rotation(np.random.default_rng(k)))
        s, steps = run_sensing(sim)
        counts.append(len(steps))
        headings.append(round(steps[-1].heading_deg))
    check(len(set(counts)) == 1 and max(headings) - min(headings) <= 5,
          f"step counts {counts}, final headings {headings} deg")

    print("sensing (step network + tracker + stairs) on held-out simulated walks")
    for seed in range(700, 704):
        stride = [1.15, 1.35, 1.55, 1.7][seed % 4]
        sim = fs.simulate(ROUTE, stride_m=stride, seed=seed, stride_s=0.95 + 0.1 * (seed % 4))
        truth = sim["truth"]["strides"]
        s, steps = run_sensing(sim)
        true_stairs = sum(x["stairs"] for x in truth)
        flagged = sum(st.stairs == 1 for st in steps)
        tt = np.array([x["t_strike"] for x in truth])
        wrong = sum(1 for st in steps
                    if st.stairs == 1 and not truth[int(np.argmin(np.abs(tt - st.t)))]["stairs"])
        check(abs(len(steps) - len(truth)) <= 1 and abs(flagged - true_stairs) <= 2 and wrong <= 1,
              f"{s.VERSION} seed {seed}: {len(steps)} steps / {len(truth)} true, stairs {flagged} flagged / "
              f"{true_stairs} true, {wrong} flagged on flat ground")

    print("Project 8 real recordings (human counted both feet; one shoe sees half)")
    for rec in sd.load_p8():
        meta = json.loads((sd.P8_DIR / f"{rec.name}.json").read_text())
        human = meta.get("true_steps") or len(meta.get("steps", []))
        if not human or rec.name == "walk_20261102-222548":
            continue    # that session's labels are incomplete: 10 marked per stride, then 25 per footstep
        s, steps = run_sensing({"t": rec.t, "imu": rec.imu})
        tracker = len(rec.tracked()[0])
        flags = {st.stairs for st in steps}
        check(abs(tracker - human / 2) <= 2,
              f"{rec.name}: tracker {tracker}, network {len(steps)}, human {human} footsteps "
              f"(~{human / 2:.0f} strides); stairs flags {sorted(flags)}")

    print("stairs flag is 'unknown' when strides are implausibly short (board not rigid on the foot)")
    rec = next(r for r in sd.load_p8() if r.name.startswith("walk_20261102-222548"))
    s, steps = run_sensing({"t": rec.t, "imu": rec.imu})
    check(steps and all(st.stairs == -1 for st in steps[5:]), f"flags after 5 strides: {sorted({st.stairs for st in steps[5:]})}")

    print("works without the trained network (tracker steps)")
    sim = fs.simulate(ROUTE, seed=800)
    s, steps = run_sensing(sim, step_source="tracker")
    check(s.VERSION.startswith("tracker") and len(steps) == len(sim["truth"]["strides"]),
          f"{s.VERSION}: {len(steps)} / {len(sim['truth']['strides'])}")

    print("speed")
    walk = fw.simulate()
    s = Sensing()
    t0 = time.perf_counter()
    for t, rows in walk["batches"]:
        s.feed(t, rows)
    ms = (time.perf_counter() - t0) / (len(walk["batches"]) / 10) * 1000
    check(ms < 100, f"{ms:.0f} ms of laptop CPU per second of walking (the board is slower; budget 1000)")

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
