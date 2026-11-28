"""Laptop-side checks for the parts of jim that don't need the board or a model.

    my_env/Scripts/python "Public stuff/projects/Project - Final/tests/test_offline.py"

Covers the rule planner (milestone 1), plan validation, and the executor
with a fake display / IMU / MQTT.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unoq_app" / "python"))

import matrix  # noqa: E402
import planner  # noqa: E402
from skills import Executor  # noqa: E402

FAILS = 0


def check(name, got, want):
    global FAILS
    ok = got == want
    FAILS += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        print(f"     got:  {got}\n     want: {want}")


# --- rule planner ---------------------------------------------------------------
rp = planner.RulePlanner()
CASES = [
    ('Type "Hi" on the LED matrix and then send back on MQTT on topic fred message "I am done"',
     [{"skill": "show_text", "text": "Hi"},
      {"skill": "send_message", "topic": "fred", "message": "I am done"}]),
    ("write Hi on the LED matrix and then tell me over MQTT that it is done",
     [{"skill": "show_text", "text": "Hi"},
      {"skill": "send_message", "topic": "", "message": "I am done"}]),
    ("Type “Hello world” on the LED matrix",
     [{"skill": "show_text", "text": "Hello world"}]),
    ('send to topic ME193/q/fred/reply message "ready"',
     [{"skill": "send_message", "topic": "ME193/q/fred/reply", "message": "ready"}]),
    ("clear the matrix and let me know when you're done",
     [{"skill": "clear_display"},
      {"skill": "send_message", "topic": "", "message": "I am done"}]),
]
for cmd, want in CASES:
    plan, _ = rp.plan(cmd)
    check(f"rules: {cmd[:60]}", planner.validate(plan, {"ME193/q/will/cmd"}), want)

plan, _ = rp.plan("make me a sandwich")
check("rules: unknown command refuses", plan["steps"][0]["skill"], "refuse")

# Review finding: rules used to do the part they understood and drop the rest.
for cmd in ['Type "Hi" on the LED matrix and then spin the motor',
            'write "OK" on the screen and then tell me the weather on topic fred']:
    plan, _ = rp.plan(cmd)
    check(f"rules: half-understood command refuses: {cmd[:45]}", plan["steps"][0]["skill"], "refuse")

# --- validation -------------------------------------------------------------------
for bad, why in [
    ({"steps": [{"skill": "launch_rocket"}]}, "unknown skill"),
    ({"steps": [{"skill": "send_message", "topic": "ME193/#", "message": "x"}]}, "wildcard topic"),
    ({"steps": [{"skill": "send_message", "topic": "ME193/q/will/cmd", "message": "x"}]}, "own cmd topic"),
    ({"steps": [{"skill": "send_message", "topic": "ME193/q/fred/cmd", "message": "x"}]}, "another bot's cmd topic"),
    ({"steps": [{"skill": "send_message", "topic": "ME193/q/will/online", "message": "offline"}]}, "own system topic"),
    ({"steps": [{"skill": "wait", "seconds": 999}]}, "long wait"),
    ({"steps": []}, "empty plan"),
    ({"steps": [{"skill": "show_icon", "icon": "dragon"}]}, "unknown icon"),
]:
    try:
        planner.validate(bad, {"ME193/q/will/cmd", "ME193/q/will/online"})
        check(f"validate rejects {why}", "accepted", "PlanError")
    except planner.PlanError:
        check(f"validate rejects {why}", "PlanError", "PlanError")

check("validate: refuse wins over other steps",
      planner.validate({"steps": [{"skill": "show_text", "text": "Hi"},
                                  {"skill": "refuse", "reason": "no motors"}]}),
      [{"skill": "refuse", "reason": "no motors"}])

# --- schema is valid JSON schema-ish and lists every skill -------------------------
check("schema skills", planner.SKILLS,
      ["show_text", "show_icon", "clear_display", "reset_yaw", "wait", "send_message", "refuse"])


# --- placeholder repair (v3) ---------------------------------------------------------
check("repair: bare 'temperature' -> live value",
      planner.repair_placeholders({"steps": [{"skill": "send_message", "topic": "fred", "message": "temperature"}]},
                                  "tell me the temperature on topic fred")["steps"][0]["message"], "{temperature}")
check("repair: quoted word stays literal",
      planner.repair_placeholders({"steps": [{"skill": "show_text", "text": "yaw"}]},
                                  'show the word "yaw" on the matrix')["steps"][0]["text"], "yaw")
check("repair: normal text untouched",
      planner.repair_placeholders({"steps": [{"skill": "show_text", "text": "Hi"}]}, "type Hi")["steps"][0]["text"], "Hi")
check("schema v3 lets the model name an icon we don't have (validate refuses it)",
      "enum" in planner.PLAN_SCHEMAS["v3"]["properties"]["steps"]["items"]["anyOf"][1]["properties"]["icon"], False)

# --- executor with fakes -------------------------------------------------------------
class FakeDisplay:
    def __init__(self):
        self.shown = []

    def show(self, scene):
        self.shown.append(scene.text if isinstance(scene, matrix.TextScene) else "icon")

    def clear(self):
        self.shown.append("clear")


class FakeImu:
    def __init__(self, snap):
        self.snap = snap

    def snapshot(self):
        return self.snap

    def reset_yaw(self):
        pass


sent = []
disp = FakeDisplay()
ex = Executor(disp, FakeImu({"yaw": 12.5, "pitch": 1.0, "roll": -2.0, "temperature": 24.1,
                             "accel": "a", "gyro": "g", "calibrated": True}),
              lambda t, m: sent.append((t, m)), lambda: "fine", "ME193/q/will/reply")
ex.run([{"skill": "show_text", "text": "Hi"},
        {"skill": "send_message", "topic": "fred", "message": "I am done"}])
check("executor: shows Hi then replies on fred", (disp.shown, sent),
      (["Hi"], [("fred", "I am done")]))

sent.clear()
ex.run([{"skill": "send_message", "topic": "", "message": "yaw {yaw}, temp {temperature}, {bogus}"}])
check("executor: unknown value -> one honest sentence", sent,
      [("ME193/q/will/reply", "I can't read bogus: I have no such sensor.")])
sent.clear()
ex.run([{"skill": "send_message", "topic": "", "message": "yaw {yaw}, temp {temperature}"}])
check("executor: fills live values, default topic", sent,
      [("ME193/q/will/reply", "yaw 12.5, temp 24.1")])

sent.clear()
ex_no_imu = Executor(disp, FakeImu(None), lambda t, m: sent.append((t, m)), lambda: "fine", "r")
ex_no_imu.run([{"skill": "send_message", "topic": "", "message": "yaw is {yaw}"}])
check("executor: no IMU -> says so instead of a made-up number", sent,
      [("r", "I can't read yaw: no IMU is connected.")])

sent.clear()
ex_cal = Executor(disp, FakeImu({"yaw": 0.0, "calibrated": False}), lambda t, m: sent.append((t, m)),
                  lambda: "fine", "r")
ex_cal.run([{"skill": "send_message", "topic": "", "message": "yaw {yaw}"}])
check("executor: yaw before calibration is refused, not 0.0", "calibrating" in sent[0][1], True)

# Review finding: {status} expanded after the length check -> minutes of scrolling.
disp.shown.clear()
ex_long = Executor(disp, FakeImu(None), lambda t, m: None, lambda: "x" * 900, "r")
import skills  # noqa: E402
skills.matrix.TextScene.SCROLL_COLS_PER_S = 1e6   # don't actually wait in the test
ex_long.run([{"skill": "show_text", "text": "{status}"}])
check("executor: filled text is capped", len(disp.shown[-1]) <= skills.MAX_SHOWN_TEXT, True)

# /stop: an already-set cancel event stops the plan before the first step.
import threading  # noqa: E402
ev = threading.Event()
ev.set()
disp.shown.clear()
try:
    ex.run([{"skill": "show_text", "text": "never"}], ev)
    check("executor: pre-set cancel stops the plan", "ran", "Cancelled")
except skills.Cancelled:
    check("executor: pre-set cancel stops the plan", disp.shown, [])

# --- matrix ---------------------------------------------------------------------------
check("matrix: 'Hi' fits without scrolling", matrix.TextScene("Hi").scrolls, False)
check("matrix: 'Hello' scrolls", matrix.TextScene("Hello").scrolls, True)

print("\nall passed" if not FAILS else f"\n{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
