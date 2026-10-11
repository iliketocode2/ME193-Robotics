"""
Self-check for game_logic.py -- no camera, BLE, or MQTT needed. Simulates
rallies with a scripted paddle (stand-in for pose) and scripted swings
(stand-in for the IMU) and checks the rules.

Run:
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_game_logic.py"
"""

import random

import game_logic as gl
from game_logic import Game, Paddle, Swing

DT = 1 / 60
failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def new_game(opp="pip", seed=1):
    g = Game(rng=random.Random(seed))
    g.update(0.0, DT, Paddle())
    g.set_highlight(opp)
    g.confirm()
    g.end_show()                    # (the pre-match show has its own tests below)
    return g


def run(g, t0, seconds, paddle_fn, swing_fn=lambda g, t: False, until=None):
    """Step the game. paddle_fn(g) -> Paddle; swing_fn(g, t) -> bool."""
    t = t0
    events = []
    while t < t0 + seconds:
        t += DT
        swings = [Swing(t, 1.2)] if swing_fn(g, t) else []
        events += g.update(t, DT, paddle_fn(g), swings)
        if until and until(g, events):
            break
    return t, events


def tracking(g):
    """A perfect player: hand always right at the ball."""
    return Paddle(x=g.ball.x, y=g.ball.y, visible=True)


def swing_in_zone(g, t):
    b = g.ball
    return g.state == gl.RALLY and b.hitter == gl.OPP and gl.HIT_ZONE[0] <= b.z <= gl.PLAYER_HIT_Z


def types(events):
    return [e["type"] for e in events]


# --- solve_launch: lands where aimed and clears the net --------------------
for speed in (3.0, 3.9, 4.9):
    vx, vy, vz = gl.solve_launch(0.2, 0.25, -1.55, -0.4, 0.8, speed, ax=1.6)
    x, y, z, t, net_y = 0.2, 0.25, -1.55, 0.0, None
    while y > 0 or t == 0:
        h = 0.0005
        vx += 1.6 * h
        vy -= gl.GRAVITY * h
        pz = z
        x, y, z, t = x + vx * h, y + vy * h, z + vz * h, t + h
        if pz < 0 <= z:
            net_y = y
    check(abs(x + 0.4) < 0.03 and abs(z - 0.8) < 0.03,
          f"launch @ {speed} m/s bounces at target (got x={x:.2f}, z={z:.2f})")
    check(net_y is not None and net_y > gl.NET_H, f"launch @ {speed} m/s clears the net ({net_y:.2f} m)")

# --- select screen ----------------------------------------------------------
g = Game(rng=random.Random(0))
g.confirm()
check(g.state == gl.SELECT, "confirm with no tag held does nothing")
g.set_highlight("viktor")
g.confirm()
check(g.state == gl.SHOW and g.opp["key"] == "viktor", "tag + confirm starts the pre-match show vs. chosen opponent")
ev = g.update(0.1, DT, Paddle())
check({"start", "say"} <= set(types(ev)), "events raised by confirm() reach the next update() (start + intro line)")

# --- the pre-match show -----------------------------------------------------
check(not g.ball.visible and g.snapshot()["show"]["len"] == gl.SHOW_MAX_S, "show: no ball, full length")
x0 = g.opp_x
g.update(1.0, DT, Paddle())
check(g.state == gl.SHOW and g.opp_x == x0, "show: nothing moves (no physics)")
g = Game(); g.set_highlight("pip"); g.confirm(); g.update(0.0, DT, Paddle())
g.update(0.3, DT, Paddle()); g.confirm()
check(g.state == gl.SHOW, "show: a second ENTER right away doesn't skip it (habitual double-press)")
g.update(gl.SHOW_SKIP_LOCK_S + 0.1, DT, Paddle()); g.confirm()
ev = g.update(gl.SHOW_SKIP_LOCK_S + 0.2, DT, Paddle())
check(g.state == gl.COUNTDOWN and {"type": "show_end", "skipped": True} in ev, "show: ENTER skips it -> countdown")
g = Game(); g.set_highlight("pip"); g.confirm(); g.update(0.0, DT, Paddle())
ev = g.update(gl.SHOW_MAX_S + 0.05, DT, Paddle())
check(g.state == gl.COUNTDOWN and {"type": "show_end", "skipped": False} in ev, "show: ends by itself at its length")
g = Game(); g.set_highlight("pip"); g.confirm(); g.update(0.0, DT, Paddle())
g.set_show_len(99)
check(g.show_len == gl.SHOW_MAX_S, "show: set_show_len never lengthens it")
g.set_show_len(5); g.update(5.1, DT, Paddle())
check(g.state == gl.COUNTDOWN, "show: set_show_len shortens it")
g = Game(); g.set_highlight("pip"); g.confirm(); g.update(0.0, DT, Paddle()); g.end_show()
check(g.state == gl.COUNTDOWN, "show: end_show() (script done) -> countdown")
g = Game(); g.set_highlight("pip"); g.confirm(); g.update(0.0, DT, Paddle()); g.go_home()
check(g.state == gl.SELECT, "show: home quits it")
g = Game()
g.set_highlight("nobody")
check(g.highlight is None, "unknown opponent key is ignored")

# --- opponent serve, perfect return ----------------------------------------
g = new_game(seed=3)
g.server = gl.OPP
t, ev = run(g, 0.0, gl.COUNTDOWN_S + 0.1, tracking)
check(g.state == gl.SERVE, "countdown -> serve")
g.server = gl.OPP  # (countdown chose player; force an opponent serve for this test)
g._start_serve()
t, ev = run(g, t, 5.0, tracking, swing_in_zone, until=lambda g, e: "hit" in types(e))
check("opp_hit" in types(ev) and "hit" in types(ev), "pose on ball + IMU swing in zone = hit")
check(g.streak == 1 and g.record == 1.0 and isinstance(g.record, float), "hit -> streak 1, record 1.0 (float)")
check("record" in types(ev), "new record emits a record event (-> MQTT publish)")

# --- no swing: miss because of IMU -----------------------------------------
g = new_game(seed=4)
g.server = gl.OPP
g._start_serve()
t, ev = run(g, 0.0, 6.0, tracking, until=lambda g, e: "point" in types(e))
miss = [e for e in ev if e["type"] == "miss"]
check(miss and miss[0]["reason"] == "no_swing", "in place but no swing -> miss (reason no_swing)")
check(g.score["opp"] == 1, "your miss scores for the opponent")

# --- swing but paddle in the wrong place: miss because of pose -------------
g = new_game(seed=5)
g.server = gl.OPP
g._start_serve()
far = lambda g: Paddle(x=-g.ball.x + (1.0 if g.ball.x < 0 else -1.0), y=g.ball.y, visible=True)
t, ev = run(g, 0.0, 6.0, far, swing_in_zone, until=lambda g, e: "point" in types(e))
miss = [e for e in ev if e["type"] == "miss"]
check(miss and miss[0]["reason"] == "wrong_place", "swing with paddle off the ball -> miss (reason wrong_place)")

# --- swing way too early ----------------------------------------------------
g = new_game(seed=6)
g.server = gl.OPP
g._start_serve()
early = lambda g, t: g.state == gl.RALLY and g.ball.hitter == gl.OPP and -0.2 < g.ball.z < 0.0
t, ev = run(g, 0.0, 6.0, tracking, early, until=lambda g, e: "point" in types(e))
check("hit" not in types(ev) and "miss" in types(ev), "swing while ball is still over the net -> no hit")

# --- player serve -----------------------------------------------------------
g = new_game(seed=7)
t, _ = run(g, 0.0, gl.COUNTDOWN_S + 0.1, tracking)
check(g.state == gl.SERVE and g.server == gl.PLAYER, "you serve first")
t, ev = run(g, t, 0.5, tracking)
check(g.state == gl.SERVE and not g.ball.live, "ball waits on your paddle until you swing")
t, ev = run(g, t, 0.1, tracking, lambda g, t: True)
check(g.state == gl.RALLY and g.ball.hitter == gl.PLAYER and g.streak == 1, "swing serves the ball")
t, ev = run(g, t, 3.0, tracking, until=lambda g, e: "bounce" in types(e))
check(any(e["type"] == "bounce" and e["side"] == gl.OPP for e in ev), "your serve lands on the opponent's side")

# --- long rally: record only goes up ---------------------------------------
g = new_game(opp="viktor", seed=11)
t, ev = run(g, 0.0, 60.0, tracking, lambda g, t: swing_in_zone(g, t) or g.state == gl.SERVE)
records = [e["record"] for e in ev if e["type"] == "record"]
check(len(records) >= 3 and records == sorted(records), f"record is monotonic over a rally ({records[:6]}...)")
best = g.record
g.server = gl.OPP
g._start_serve()
t, ev = run(g, t, 6.0, tracking, until=lambda g, e: "point" in types(e))
check(g.streak == 0 and g.record == best, "missing resets the streak but keeps the record")

# --- opponent misses keep your streak alive --------------------------------
g = new_game(seed=12)
g.streak = 4
g._point(gl.PLAYER, "opp_miss")
check(g.streak == 4, "opponent error doesn't break your streak")

# --- scoring: to 11, win by 2 ----------------------------------------------
g = new_game(seed=13)
g.score = {gl.PLAYER: 10, gl.OPP: 10}
g._point(gl.PLAYER, "test")
g.now += gl.POINT_PAUSE_S
g._after_point()
check(g.state == gl.SERVE, "11-10 is not game over (win by 2)")
check(g.server == gl.OPP, "at deuce the serve alternates every point")
g._point(gl.PLAYER, "test")
g._after_point()
check(g.state == gl.GAME_OVER and g.winner == gl.PLAYER, "12-10 wins the game")
g.confirm()
check(g.state == gl.GAME_OVER, "confirm right at game over is ignored (no accidental restart)")
g.now += gl.GAME_OVER_LOCK_S
g.confirm()
check(g.state == gl.SELECT and g.highlight is None, "confirm after game over -> back to select")

# --- rematch / home buttons -------------------------------------------------
g = new_game(opp="rita", seed=16)
g.streak = 3
g.score = {gl.PLAYER: 3, gl.OPP: 10}
g._point(gl.OPP, "t"); g._after_point()
g.rematch()
check(g.state == gl.GAME_OVER, "rematch is locked right at game over too")
g.now += gl.GAME_OVER_LOCK_S
g.rematch()
check(g.state == gl.SHOW and g.show_rematch and g.show_len == gl.SHOW_REMATCH_S and g.opp["key"] == "rita"
      and g.score == {gl.PLAYER: 0, gl.OPP: 0}, "rematch -> same opponent, score reset, a shorter show")
check(g.history == [{"opp": "rita", "winner": gl.OPP, "you": 3, "them": 11, "best_streak": 0}],
      f"the session history records the finished match ({g.history})")
check(g.streak == 0, "a new match resets the streak (the record stays)")
g._start_serve()
g.go_home()
check(g.state == gl.SELECT and not g.ball.visible, "home mid-match quits straight to select")
g.go_home()
check(g.state == gl.SELECT, "home on the select screen does nothing")

servers = []
g2 = new_game(seed=15)
for i in range(6):
    g2._point(gl.PLAYER if i % 2 else gl.OPP, "t"); g2._after_point(); servers.append(g2.server)
check(servers == ["player", "opp", "opp", "player", "player", "opp"], f"serve alternates every 2 points ({servers})")

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED")
raise SystemExit(1 if failures else 0)
