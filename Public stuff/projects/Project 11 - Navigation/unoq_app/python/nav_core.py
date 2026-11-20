"""Everything the board does, with no Arduino imports, so it also runs on a laptop.

main.py (on the board) and tools/dev_board.py (on a laptop) both create one
NavCore, pass IMU batches to `on_imu`, page messages to `handle`, and call
`tick` about every 0.1 s. NavCore talks back to the web page through the
`send(event, data)` function it was given.

Modes
  idle        sensing runs so the page shows live steps and heading; nothing saved
  calibrate   walk a known distance; meters per step = distance / steps
  record      a demonstration walk: the walker taps each maneuver as they do it
  navigate    the policy gives cues; with test=True the walk is saved as a test
  dagger      navigate + the walker taps what they actually do and "wrong cue"
"""

import threading
import time

import numpy as np

import config as cfg
import features as feat
import policy as pol
import walks as wk
from sensing import Sensing

MODE_KIND = {"calibrate": "calibrate", "record": "demo", "navigate": "test", "dagger": "dagger"}


class NavCore:
    def __init__(self, send, clock=time.time):
        self.send = send
        self.clock = clock
        self.lock = threading.RLock()
        self.sensing = Sensing()
        self.policy = pol.Policy.load(cfg.POLICY_DIR / cfg.DEFAULT_ROUTE)
        self.walkers = wk.load_walkers()
        self.walker = next(iter(self.walkers), None)
        self.route = cfg.DEFAULT_ROUTE
        self.mode = "idle"
        self.recorder = None
        self.tracker = None
        self.cues = None
        self.last_cue = None
        self.last_pred = None
        self.steps = 0
        self.heading = 0.0
        self.imu_status = "waiting"
        self.last_sample = 0.0
        self.handlers = {
            "get_status": self.on_get_status, "set_walker": self.on_set_walker,
            "set_route": self.on_set_route, "start": self.on_start, "stop": self.on_stop,
            "tap": self.on_tap, "list_walks": self.on_list_walks, "get_walk": self.on_get_walk,
            "save_review": self.on_save_review, "train": self.on_train,
            "reprocess": self.on_reprocess, "export": self.on_export,
        }

    # ---------- inputs ----------

    def handle(self, event, data):
        fn = self.handlers.get(event)
        if fn is None:
            return
        with self.lock:
            try:
                fn(data or {})
            except Exception as e:      # never let one bad message kill the app
                self.notice(f"{event} failed: {e}", error=True)

    def on_imu(self, rows, t=None):
        """rows: [[ax, ay, az, gx, gy, gz], ...] in g and deg/s; t = time of the last row."""
        t = self.clock() if t is None else t
        with self.lock:
            self.imu_status = "ok"
            self.last_sample = t
            if self.recorder is not None:
                self.recorder.add_imu(t, rows)
            for step in self.sensing.feed(t, rows):
                self._on_step(step)

    def on_imu_status(self, status):
        with self.lock:
            self.imu_status = str(status)

    def tick(self):
        """Call regularly; sends status once a second."""
        now = self.clock()
        if now - getattr(self, "_last_status", 0.0) >= 1.0:
            self._last_status = now
            with self.lock:
                self.send("status", self.status())

    # ---------- steps ----------

    def _mps(self):
        if self.walker and self.walker in self.walkers:
            return self.walkers[self.walker]["meters_per_step"]
        return cfg.DEFAULT_METERS_PER_STEP

    def _on_step(self, step):
        self.steps = step.n
        self.heading = step.heading_deg
        out = {"n": step.n, "heading": round(step.heading_deg, 1), "meters": round(step.n * self._mps(), 1)}
        if self.recorder is not None:
            self.recorder.add_step(step)
        if self.tracker is not None:
            x = self.tracker.update(step)
            out.update(meters_since_turn=round(float(x[1]), 1), turns=int(x[2]))
            if self.mode in ("navigate", "dagger") and self.policy.trained:
                cls, conf = self.policy.predict(x)
                self.last_pred = cls
                out.update(prediction=cls, confidence=round(conf, 2))
                spoken = self.cues.feed(cls)
                if spoken:
                    self.last_cue = spoken
                    if self.recorder is not None:
                        self.recorder.add_event({"t": step.t, "type": "cue", "action": spoken})
                    self.send("cue", {"action": spoken, "text": cfg.SPOKEN.get(spoken, spoken),
                                      "t": step.t, "step": step.n})
                out["cue"] = self.cues.current
        self.send("step", out)

    # ---------- page messages ----------

    def on_get_status(self, data):
        self.send("status", self.status())

    def on_set_walker(self, data):
        name = str(data.get("name", "")).strip()
        if not name:
            return
        if self.recorder is not None:
            return self.notice("Stop the current walk before changing walker.", error=True)
        self.walker = name
        if name not in self.walkers:
            self.notice(f"New walker {name}: do a calibration walk so distances are right.")
        self.send("status", self.status())

    def on_set_route(self, data):
        name = wk.safe_name(str(data.get("name", "")).lower())
        if name and self.recorder is None:
            self.route = name
            self.policy = pol.Policy.load(cfg.POLICY_DIR / name)
            self.send("status", self.status())

    def on_start(self, data):
        mode = data.get("mode")
        if mode not in MODE_KIND:
            return
        if not self.walker:
            return self.notice("Pick or add a walker first.", error=True)
        if mode in ("navigate", "dagger") and not self.policy.trained:
            return self.notice("No trained policy yet. Record and review some demos, then Train.", error=True)
        if self.recorder is not None:
            self._finish()
        kind = MODE_KIND[mode]
        if mode == "navigate" and not data.get("test", True):
            kind = None         # navigate without saving
        self.sensing.reset()
        self.steps, self.heading = 0, 0.0
        self.mode = mode
        self.recorder = wk.WalkRecorder(self.walker, kind, self.route, self._mps(), Sensing.VERSION,
                                        now=self.clock()) if kind else None
        self.tracker = feat.StateTracker(self._mps()) if mode != "calibrate" else None
        self.cues = pol.CueFilter()
        self.last_cue = self.last_pred = None
        self.notice({"calibrate": "Calibrating: walk to your end marker, then press Stop.",
                     "record": "Recording a demo. Tap each maneuver as you start it.",
                     "navigate": "Navigating. Follow the cues.",
                     "dagger": "DAgger walk. Follow the cues, tap what you actually do."}[mode])
        self.send("status", self.status())

    def on_stop(self, data):
        if self.mode == "calibrate":
            try:
                distance = float(data.get("distance_m", 0))
            except (TypeError, ValueError):
                distance = 0.0
            if distance <= 0 or self.steps < 5:
                self.notice(f"Calibration not saved: need a distance and at least 5 steps (got {self.steps}).",
                            error=True)
            else:
                mps = distance / self.steps
                self.walkers = wk.save_walker(self.walker, mps)
                self.notice(f"{self.walker}: {mps:.2f} m per step ({self.steps} steps over {distance:g} m).")
        walk_id = self._finish()
        if walk_id and self.mode != "calibrate":
            self.notice(f"Saved walk {walk_id}. Review it before training.")
        self.mode = "idle"
        self.tracker = self.cues = None
        self.send("status", self.status())

    def _finish(self):
        if self.recorder is None:
            return None
        walk_id = self.recorder.close(now=self.clock())
        self.recorder = None
        return walk_id

    def on_tap(self, data):
        if self.recorder is None or self.mode not in ("record", "dagger"):
            return
        kind = data.get("type")
        if kind not in ("label", "undo", "wrong_cue"):
            return
        if kind == "label" and data.get("action") not in cfg.MANEUVERS:
            return
        if data.get("walk") and data["walk"] != self.recorder.id:
            self.send("tap_ok", {"id": data.get("id"), "stale": True})    # queued during an older walk
            return
        now = self.clock()
        try:
            t = float(data.get("t"))
        except (TypeError, ValueError):
            t = now
        t = min(max(t, self.recorder.meta["start"]), now)     # phone clock estimate can be a bit off
        event = {"id": str(data.get("id", "")) or None, "t": t, "type": kind}
        if kind == "label":
            event["action"] = data["action"]
        if self.recorder.add_event(event):
            self.send("tap_ok", {"id": event["id"]})
        else:
            self.send("tap_ok", {"id": event["id"], "duplicate": True})

    def on_list_walks(self, data):
        self.send("walks", {"walks": wk.list_walks(), "budget": wk.budget(self.route), "route": self.route})

    def on_get_walk(self, data):
        w = wk.load_walk(str(data.get("id", "")))
        mps = w["meters_per_step"]
        steps = w["steps"]
        meters = [round(s["n"] * mps, 2) for s in steps]
        heading = [s["heading_deg"] for s in steps]
        turns = feat.detected_turn_starts(np.array(meters), np.array(heading)) if steps else []
        maneuvers = w["maneuvers"] if w["reviewed"] else wk.raw_maneuvers(w["events"])
        self.send("walk", {
            "id": w["id"], "walker": w["walker"], "kind": w["kind"], "route": w["route"],
            "meters_per_step": mps, "reviewed": w["reviewed"], "include": w["include"],
            "sensing": w["sensing"], "start": w["start"], "end": w["end"],
            "steps": [{"t": s["t"], "m": m, "h": round(s["heading_deg"], 1), "s": s["stairs"]}
                      for s, m in zip(steps, meters)],
            "taps": wk.raw_maneuvers(w["events"]),
            "events": [e for e in w["events"] if e["type"] in ("cue", "wrong_cue")],
            "maneuvers": maneuvers,
            "detected_turns": [{"t": steps[i]["t"], "sign": sign} for i, sign in turns],
            "snaps": feat.snap_suggestions(steps, maneuvers, mps),
            "walker_mps": self.walkers.get(w["walker"], {}).get("meters_per_step"),
            "snap_window_m": cfg.SNAP_WINDOW_M,
        })

    def on_save_review(self, data):
        meta = wk.save_review(str(data.get("id", "")), data.get("maneuvers", []), data.get("include", True),
                              data.get("meters_per_step"))
        self.notice(f"Saved review: {len(meta['maneuvers'])} maneuvers, "
                    f"{'included' if meta['include'] else 'excluded'}.")
        self.on_list_walks({})

    def on_train(self, data):
        walks = wk.training_walks(self.route)
        if not walks:
            return self.notice("No reviewed, included demo or DAgger walks for this route yet.", error=True)
        b = wk.budget(self.route)
        policy, report = pol.train(walks)
        policy.save(cfg.POLICY_DIR / self.route)
        self.policy = policy
        self.send("train_result", {"info": {k: v for k, v in policy.info.items() if k != "report"},
                                   "report": report, "budget": b})
        warn = f" Over the {cfg.WALK_BUDGET}-walk limit!" if b["used"] > cfg.WALK_BUDGET else ""
        self.notice(f"Trained on {len(walks)} walks ({policy.info['steps']} steps).{warn}",
                    error=bool(warn))
        self.send("status", self.status())

    def on_reprocess(self, data):
        if self.recorder is not None:
            return self.notice("Stop the current walk first.", error=True)
        count = 0
        for s in wk.list_walks():
            wk.reprocess(s["id"], Sensing())
            count += 1
        self.sensing.reset()
        self.notice(f"Re-ran sensing ({Sensing.VERSION}) on {count} walks. Check reviews, then retrain.")
        self.on_list_walks({})

    def on_export(self, data):
        walks = [wk.load_walk(s["id"]) for s in wk.list_walks()]
        self.send("export", {"exported": self.clock(), "walkers": self.walkers, "walks": walks,
                             "policy": self.policy.info, "config": {
                                 "maneuvers": cfg.MANEUVERS, "classes": cfg.CLASSES, "features": cfg.FEATURES,
                                 "now_before_m": cfg.NOW_BEFORE_M, "now_after_m": cfg.NOW_AFTER_M, "lookahead_m": cfg.LOOKAHEAD_M, "k": cfg.K}})

    # ---------- outputs ----------

    def notice(self, text, error=False):
        self.send("notice", {"text": text, "error": error})

    def status(self):
        now = self.clock()
        imu = self.imu_status
        if imu == "ok" and now - self.last_sample > 1.0:
            imu = "stalled"
        rec = self.recorder
        info = self.policy.info if self.policy.trained else None
        return {
            "board_time": now, "imu": imu, "mode": self.mode, "walker": self.walker,
            "walkers": self.walkers, "route": self.route, "sensing": Sensing.VERSION,
            "steps": self.steps, "heading": round(self.heading, 1), "meters": round(self.steps * self._mps(), 1),
            "meters_per_step": self._mps(),
            "walk": None if rec is None else {"id": rec.id, "kind": rec.meta["kind"],
                                              "seconds": round(now - rec.meta["start"], 1)},
            "cue": self.last_cue, "budget": wk.budget(self.route),
            "policy": None if info is None else {"walks": len(info.get("walks", [])), "steps": info.get("steps"),
                                                 "report": _report_summary(info.get("report"))},
        }


def _report_summary(report):
    if not report:
        return None
    return {k: report[k] for k in ("hit_rate", "mean_abs_error_m", "false_alarms", "maneuvers")}
