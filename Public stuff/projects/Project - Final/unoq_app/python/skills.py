"""Runs a validated plan, one skill at a time.

The executor doesn't know about MQTT, the Bridge or the model. main.py
hands it four things:

  display   object with show(scene) / clear(); scenes come from matrix.py
  imu       imu.Imu (or anything with snapshot() / reset_yaw())
  publish   publish(topic, text) -> None
  status_fn () -> str, the same text /status sends

so the whole thing can be exercised on a laptop with fakes (see
tests/test_offline.py).

Live values in text/messages ({yaw}, {time}, ...) are filled in at the
moment the step runs, so "wait 5 seconds then tell me the yaw" reports the
yaw after the wait, not when the command arrived.
"""

import re
import threading
import time

import matrix

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
MAX_SHOWN_TEXT = 80      # ~10 s of scrolling at most
MAX_SENT_MESSAGE = 1000


class Cancelled(Exception):
    pass


class Executor:
    def __init__(self, display, imu, publish, status_fn, default_topic):
        self.display = display
        self.imu = imu
        self.publish = publish
        self.status_fn = status_fn
        self.default_topic = default_topic
        self._cancel = threading.Event()
        self.current = "idle"          # human-readable, read by /status

    # --- live values -------------------------------------------------------
    def _values(self, names):
        """Returns (values, missing): missing maps a name to why it can't be read."""
        values, missing = {}, {}
        if names & {"time", "date"}:
            now = time.localtime()
            values["time"] = time.strftime("%H:%M:%S", now)
            values["date"] = time.strftime("%Y-%m-%d", now)
        if "status" in names:
            values["status"] = self.status_fn()
        imu_names = names & {"yaw", "pitch", "roll", "temperature", "accel", "gyro"}
        if imu_names:
            snap = self.imu.snapshot()
            for n in sorted(imu_names):
                if not snap:
                    missing[n] = "no IMU is connected"
                elif n == "yaw" and not snap.get("calibrated", True):
                    missing[n] = "the gyro is still calibrating (keep me still for 2 seconds)"
                else:
                    values[n] = snap[n]
        for n in names - set(values) - set(missing):
            missing[n] = "I have no such sensor"
        return values, missing

    def fill(self, text, limit, short=False):
        """Fill live values, then cap the length: {status} alone can be
        hundreds of characters, and the caps in planner.validate only saw
        the template.

        If a value can't be read, the whole text becomes one honest
        sentence ("I can't read yaw: no IMU is connected") rather than
        "My yaw is unknown degrees". short=True gives "no yaw" for the
        matrix."""
        names = set(_PLACEHOLDER.findall(text))
        if names:
            values, missing = self._values(names)
            if missing:
                if short:
                    return "no " + " ".join(sorted(missing))
                why = "; ".join(sorted(set(missing.values())))
                return f"I can't read {' and '.join(sorted(missing))}: {why}."[:limit]
            text = _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)
        return text if len(text) <= limit else text[:limit - 3] + "..."

    # --- waiting that /stop can interrupt -----------------------------------
    def _sleep(self, seconds):
        if self._cancel.wait(seconds):
            raise Cancelled()

    # --- the skills ----------------------------------------------------------
    def run(self, steps, cancel=None):
        """Run every step. Returns a list of result strings (one per step).

        cancel: a threading.Event that /stop sets. The caller owns it and
        never clears it (main.py swaps in a fresh one for the next command),
        so a /stop can't be erased by a command that is just starting."""
        self._cancel = cancel or threading.Event()
        results = []
        try:
            for i, step in enumerate(steps, 1):
                if self._cancel.is_set():
                    raise Cancelled()
                self.current = f"step {i}/{len(steps)}: {describe(step)}"
                results.append(self._run_step(step))
        finally:
            self.current = "idle"
        return results

    def _run_step(self, step):
        skill = step["skill"]
        if skill == "show_text":
            text = self.fill(step["text"], MAX_SHOWN_TEXT, short=True)
            scene = matrix.TextScene(text)
            self.display.show(scene)
            # Don't move on (e.g. to "I am done") until the whole text has
            # been on screen once.
            self._sleep(scene.duration_once())
            return f"showed {text!r}"
        if skill == "show_icon":
            self.display.show(matrix.icon_frame(step["icon"]))
            return f"showed icon {step['icon']}"
        if skill == "clear_display":
            self.display.clear()
            return "cleared the display"
        if skill == "reset_yaw":
            self.imu.reset_yaw()
            return "yaw reset to 0"
        if skill == "wait":
            self._sleep(step["seconds"])
            return f"waited {step['seconds']:g}s"
        if skill == "send_message":
            topic = step["topic"] or self.default_topic
            message = self.fill(step["message"], MAX_SENT_MESSAGE)
            self.publish(topic, message)
            return f"sent {message!r} to {topic}"
        if skill == "refuse":
            reason = step["reason"]
            self.publish(self.default_topic, f"Sorry, I can't do that: {reason}")
            return f"refused: {reason}"
        raise ValueError(f"unknown skill {skill}")


def describe(step):
    args = ", ".join(f"{k}={v!r}" for k, v in step.items() if k != "skill")
    return f"{step['skill']}({args})"
