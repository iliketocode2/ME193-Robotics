"""End-to-end test of the navigator on synthetic walks, no board needed.

Runs everything through NavCore exactly as the web page would drive it:
calibrate two walkers with different strides -> record 3 demos each with late
taps -> review (snap taps to detected turns) -> train -> a third walker
navigates a new walk and must get a heads-up and a "now" cue for every
maneuver -> a DAgger walk round-trips. Data goes to a temp folder.

    python tools/test_pipeline.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "unoq_app" / "python")]

import config as cfg            # noqa: E402
import fake_walk as fw          # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="nav_test_"))
cfg.DATA_DIR = TMP
cfg.WALKS_DIR = TMP / "walks"
cfg.WALKERS_PATH = TMP / "walkers.json"
cfg.POLICY_DIR = TMP / "policy"

from nav_core import NavCore    # noqa: E402

failures = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        failures.append(what)


class Board:
    """NavCore with a clock that follows the simulated walk, and a log of what it sent."""

    def __init__(self):
        self.t = 900.0
        self.sent = []
        self.core = NavCore(send=lambda e, d: self.sent.append((e, d)), clock=lambda: self.t)

    def msg(self, event, data=None):
        self.core.handle(event, data or {})

    def last(self, event):
        return next((d for e, d in reversed(self.sent) if e == event), None)

    def notices(self):
        return [d["text"] for e, d in self.sent if e == "notice"]

    def walk(self, sim, taps=(), extra_events=()):
        """Feed a simulated walk; send each tap when the clock passes its time."""
        pending = sorted([{"type": "label", **tp} for tp in taps] + list(extra_events), key=lambda e: e["t"])
        for i, ev in enumerate(pending):
            ev.setdefault("id", f"tap-{sim['batches'][0][0]}-{i}")
        for t, rows in sim["batches"]:
            self.t = t
            self.core.on_imu(rows, t=t)
            while pending and pending[0]["t"] <= t:
                self.msg("tap", pending.pop(0))


def sim_walk(board, walker, mode, seed, mps, tap=True, **kw):
    board.msg("set_walker", {"name": walker})
    sim = fw.simulate(meters_per_step=mps, seed=seed, start_t=board.t + 2, **kw)
    board.msg("start", {"mode": mode, "test": True})
    board.walk(sim, sim["taps"] if tap else ())
    board.msg("stop", {})
    return sim


def main():
    b = Board()
    walkers = {"alex": 1.2, "sam": 1.6}

    print("calibration")
    for name, mps in walkers.items():
        b.msg("set_walker", {"name": name})
        sim = fw.simulate(route=[("walk", 40)], meters_per_step=mps, seed=99, start_t=b.t + 2, length_jitter=0)
        b.msg("start", {"mode": "calibrate"})
        b.walk(sim)
        b.msg("stop", {"distance_m": 40})
        got = b.core.walkers.get(name, {}).get("meters_per_step")
        check(got is not None and abs(got - mps) / mps < 0.1, f"{name}: {got} m/step (true {mps})")

    print("demos")
    seed = 0
    demo_sims = []
    for name, mps in walkers.items():
        for _ in range(3):
            demo_sims.append(sim_walk(b, name, "record", seed, mps))
            seed += 1
    b.msg("list_walks")
    listing = b.last("walks")
    demos = [w for w in listing["walks"] if w["kind"] == "demo"]
    check(len(demos) == 6, f"6 demo walks saved (got {len(demos)})")
    check(listing["budget"]["recorded"] == 6, f"budget counts 6 training walks: {listing['budget']}")

    print("duplicate taps")
    sim = fw.simulate(meters_per_step=1.2, seed=50, start_t=b.t + 2)
    b.msg("set_walker", {"name": "alex"})
    b.msg("start", {"mode": "record"})
    b.walk(sim, sim["taps"])
    for tp in sim["taps"][:2]:          # the phone resends after a reconnect
        b.msg("tap", {"id": f"tap-{sim['batches'][0][0]}-0", "type": "label", **tp})
    b.msg("stop", {})
    b.msg("list_walks")
    dup = b.last("walks")["walks"][-1]
    check(dup["maneuvers"] == len(sim["taps"]), f"resent tap ignored ({dup['maneuvers']} maneuvers)")
    b.msg("get_walk", {"id": dup["id"]})
    w = b.last("walk")
    b.msg("save_review", {"id": dup["id"], "maneuvers": w["maneuvers"], "include": False})

    print("review")
    snap_err_before, snap_err_after = [], []
    for d, sim in zip(demos, demo_sims):
        b.msg("get_walk", {"id": d["id"]})
        w = b.last("walk")
        truth = sim["maneuvers"]
        fixed = []
        for m, snap in zip(w["maneuvers"], w["snaps"]):
            t = snap if snap is not None else m["t"]
            fixed.append({"t": t, "action": m["action"]})
        for m, f, tr in zip(w["maneuvers"], fixed, truth):
            if m["action"] in ("left", "right"):
                snap_err_before.append(abs(m["t"] - tr["t"]))
                snap_err_after.append(abs(f["t"] - tr["t"]))
        b.msg("save_review", {"id": d["id"], "maneuvers": fixed, "include": True})
    before = sum(snap_err_before) / len(snap_err_before)
    after = sum(snap_err_after) / len(snap_err_after)
    check(after < before, f"snapping moves turn taps closer to the true turn ({before:.2f} s -> {after:.2f} s)")

    print("training")
    b.msg("train")
    result = b.last("train_result")
    check(result is not None, "trained")
    rep = result["report"]
    print(f"        leave-one-walk-out: {rep['hits']}/{rep['maneuvers']} hit, mean |error| "
          f"{rep['mean_abs_error_m']} m, heads-up {rep['heads_up_rate']}, false alarms {rep['false_alarms']}")
    check(rep["hit_rate"] >= 0.9, f"hit rate {rep['hit_rate']} >= 0.9")
    check(result["budget"]["used"] == 6, f"budget used 6 (excluded walk not counted): {result['budget']}")

    print("navigate (new walker, new walk)")
    b.msg("set_walker", {"name": "tester"})
    sim = fw.simulate(route=[("walk", 40)], meters_per_step=1.4, seed=98, start_t=b.t + 2, length_jitter=0)
    b.msg("start", {"mode": "calibrate"})
    b.walk(sim)
    b.msg("stop", {"distance_m": 40})
    n_before = len(b.sent)
    sim = sim_walk(b, "tester", "navigate", 77, 1.4, tap=False)
    cues = [d for e, d in b.sent[n_before:] if e == "cue"]
    print("        cues: " + ", ".join(f"{c['action']}@{c['t'] - sim['batches'][0][0]:.0f}s" for c in cues))
    for m in sim["maneuvers"]:
        soon = [c for c in cues if c["action"] == f"{m['action']}_soon" and m["t"] - 15 < c["t"] < m["t"]]
        now = [c for c in cues if c["action"] == f"{m['action']}_now" and abs(c["t"] - m["t"]) < 4]
        when = f"{now[0]['t'] - m['t']:+.1f} s" if now else "missing"
        check(bool(soon) and bool(now), f"{m['action']}: heads-up {'yes' if soon else 'no'}, now cue {when}")
    stray = [c for c in cues if c["action"].endswith("_now")
             and not any(abs(c["t"] - m["t"]) < 4 for m in sim["maneuvers"])]
    check(not stray, f"no stray 'now' cues ({len(stray)})")

    print("dagger")
    sim = fw.simulate(meters_per_step=1.6, seed=60, start_t=b.t + 2)
    b.msg("set_walker", {"name": "sam"})
    b.msg("start", {"mode": "dagger"})
    wrong = {"t": sim["batches"][100][0], "type": "wrong_cue"}
    b.walk(sim, sim["taps"], extra_events=[wrong])
    b.msg("stop", {})
    b.msg("list_walks")
    dg = b.last("walks")["walks"][-1]
    b.msg("get_walk", {"id": dg["id"]})
    w = b.last("walk")
    kinds = {e["type"] for e in w["events"]}
    check(dg["kind"] == "dagger" and {"cue", "wrong_cue"} <= kinds and len(w["maneuvers"]) == len(sim["taps"]),
          f"dagger walk saved with cues, the wrong-cue tap and {len(w['maneuvers'])} labels")

    print("export / reprocess")
    b.msg("export")
    exp = b.last("export")
    check(exp is not None and len(exp["walks"]) >= 9, f"export has {len(exp['walks'])} walks")
    b.msg("reprocess")
    check(any("Re-ran sensing" in n for n in b.notices()), "reprocess ran")
    errors = [n for e, n in b.sent if e == "notice" and n["error"]]
    check(not errors, f"no error notices ({[n['text'] for n in errors]})")

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
