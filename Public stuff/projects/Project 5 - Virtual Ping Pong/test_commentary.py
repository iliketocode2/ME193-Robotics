"""
Self-check for commentary.py (the announcer) -- no hardware or browser needed.

Run:
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_commentary.py"
"""

import random

import game_logic as gl
from commentary import Announcer, score_call
from game_logic import Game, Paddle, Swing

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def texts(calls):
    return " | ".join(c["text"] for c in calls)


# --- score calls ------------------------------------------------------------
check(score_call(3, 3, "Pip") == "Three all.", "3-3 -> 'Three all.'")
check(score_call(0, 2, "Pip") == "Two, love, to Pip.", "0-2 -> 'Two, love, to Pip.'")
check(score_call(7, 4, "Pip") == "Seven, four, to you.", "7-4 -> 'Seven, four, to you.'")
check(score_call(10, 6, "Pip") == "Game point, you!", "10-6 -> game point")
check(score_call(10, 10, "Pip") == "Deuce!", "10-10 -> deuce")
check(score_call(12, 11, "Pip") == "Advantage, you!", "12-11 -> advantage")

# --- full simulated match: every event turns into sensible calls ------------
g = Game(rng=random.Random(2))
a = Announcer(rng=random.Random(2))
g.set_highlight("rita")
g.confirm()
DT, t = 1 / 60, 0.0
calls, all_events = [], []
while g.state != gl.GAME_OVER and t < 600:
    t += DT
    b = g.ball
    swing = (g.state == gl.SERVE and g.server == gl.PLAYER) or \
            (g.state == gl.RALLY and b.hitter == gl.OPP and 1.3 < b.z < 1.55 and random.random() < 0.3)
    ev = g.update(t, DT, Paddle(b.x, b.y, 0, True), [Swing(t, 1.3)] if swing else [])
    all_events += ev
    calls += a.update(t, ev, g)
for _ in range(int(3 / DT)):              # let the game-over event fire
    t += DT
    ev = g.update(t, DT, Paddle(), [])
    calls += a.update(t, ev, g)

check(g.state == gl.GAME_OVER, f"simulated match finished ({g.score})")
check(calls and "Welcome" in calls[0]["text"] and "Coach Rita" in calls[0]["text"], "match opens with a welcome naming the opponent")
points = [e for e in all_events if e["type"] == "point"]
point_calls = [c for c in calls if c["interrupt"]][1:]   # minus the welcome
check(len(point_calls) == len(points), f"every point gets a call ({len(point_calls)} calls / {len(points)} points)")
check(any(w in calls[-1]["text"] for w in ("Game", "game")), f"last call announces the game: '{calls[-1]['text']}'")
check(all(0 <= c["excite"] <= 1 for c in calls), "excitement stays in 0..1")
check(all("{" not in c["text"] for c in calls), "no unfilled {placeholders}")
print("  sample:", texts(calls[:6]))

# --- throttling: optional chatter is dropped while busy ----------------------
a = Announcer(rng=random.Random(0))
first = a._call(0.0, "A fairly long sentence that takes a couple of seconds to say.", 0.5)
dropped = a._call(0.5, "Smash!", 0.7, optional=True)
later = a._call(10.0, "Smash!", 0.7, optional=True)
check(first is not None and dropped is None and later is not None, "optional lines are skipped while the announcer is talking")
check(a._call(0.6, "Point!", 0.5, interrupt=True)["interrupt"], "point calls interrupt")

# --- miss reason makes it into the point call ------------------------------
a = Announcer(rng=random.Random(0))
g = Game(); g.opp = gl.OPPONENTS["pip"]
c = a.update(1.0, [{"type": "miss", "reason": "no_swing"},
                   {"type": "point", "winner": gl.OPP, "reason": "miss", "score": {gl.PLAYER: 0, gl.OPP: 1}}], g)
check(any(s in c[0]["text"] for s in ("swing", "flat-footed", "trigger")), f"no-swing miss is called out: '{c[0]['text']}'")

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED")
raise SystemExit(1 if failures else 0)
