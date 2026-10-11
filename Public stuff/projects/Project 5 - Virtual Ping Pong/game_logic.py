"""
Virtual ping-pong rules + ball physics. Pure Python -- no camera, no BLE,
no MQTT -- so it can be unit tested (test_game_logic.py) and every sensor
is just an input to Game.update():

    paddle  -- where your hand is (from MediaPipe pose, vision.py)
    swings  -- when you swung (from the Double Motor IMU, paddle_imu.py)

Coordinates are meters, origin at the center of the table top:
    x = sideways (+ = your right), y = height above the table top,
    z = depth (+ = toward you). Your end of the table is z = +1.37,
    the opponent's end is z = -1.37, the net is at z = 0.

A hit needs BOTH signals:
    1. pose  -- paddle within HIT_RADIUS of the ball (right place), and
    2. IMU   -- a swing while the ball is in your hit zone (right time).
"""

import math
import random
from dataclasses import dataclass, field

from opponents import OPPONENTS

# --- Table (regulation size) ---------------------------------------------
TABLE_HALF_W = 0.7625
TABLE_HALF_L = 1.37
NET_H = 0.1525
FLOOR_Y = -0.76            # floor, relative to table top
GRAVITY = 9.81
RESTITUTION = 0.88         # table bounce
FLOOR_RESTITUTION = 0.5

# --- Hitting ---------------------------------------------------------------
PLAYER_HIT_Z = 1.55        # where you "should" meet the ball
HIT_ZONE = (1.20, 2.00)    # z-range where a swing can connect (generous on the
                           # late side: camera + BLE latency make swings arrive late)
SWING_EARLY_S = 0.25       # a swing up to this long before the ball enters the zone still counts
HIT_RADIUS_X = 0.38        # paddle-to-ball sideways tolerance (pose check)
HIT_RADIUS_Y = 0.40        # paddle-to-ball height tolerance (pose check)
OPP_HIT_Z = -1.55
PLAYER_SPEED_SCALE = 1.05  # your returns vs. the opponent's shot speed

# --- Match -----------------------------------------------------------------
GAME_POINTS = 11
COUNTDOWN_S = 3.0
POINT_PAUSE_S = 2.8        # also the main window for Sonia (the AI) to think: ball is dead
OPP_SERVE_DELAY_S = 1.6
GAME_OVER_LOCK_S = 2.5     # ignore confirm right after game over (no accidental restart)
STREAK_LINE_EVERY = 5      # opponent comments every N hits in a row
# Pre-match show: the commentators introduce the match, the players and the
# weather (commentary.ShowDirector), which also gives Sonia (the AI) time to
# warm up. Skippable; ends early once the script is done.
SHOW_MAX_S = 15.0
SHOW_REMATCH_S = 9.0
SHOW_SKIP_LOCK_S = 0.8     # a habitual double-ENTER mustn't skip the show instantly

# States
SELECT, SHOW, COUNTDOWN, SERVE, RALLY, POINT, GAME_OVER = (
    "select", "show", "countdown", "serve", "rally", "point", "game_over")
PLAYER, OPP = "player", "opp"


@dataclass
class Paddle:
    """Your paddle, from pose. vx = sideways velocity (m/s)."""
    x: float = 0.0
    y: float = 0.3
    vx: float = 0.0
    visible: bool = False


@dataclass
class Swing:
    """One swing detected by the IMU. strength ~1.0 = just past threshold."""
    t: float
    strength: float = 1.0


@dataclass
class Ball:
    x: float = 0.0
    y: float = 0.3
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0            # sideways spin acceleration
    live: bool = False         # moving under physics
    visible: bool = False
    hitter: str = ""           # who touched it last
    bounced_on: str = ""       # side it last bounced on since the last hit ("" = none yet)
    in_net: bool = False


def side_of(z):
    return PLAYER if z > 0 else OPP


def solve_launch(x0, y0, z0, xb, zb, speed, ax=0.0, net_clear=0.06):
    """Velocity that makes the ball first bounce at (xb, zb) on the table.

    Flight time comes from horizontal distance / speed; vertical velocity is
    whatever parabola lands at y=0 at that time. If that parabola would clip
    the net, the flight is slowed (higher arc) until it clears.
    Returns (vx, vy, vz)."""
    dist = math.hypot(xb - x0, zb - z0)
    T = max(dist / speed, 0.15)
    for _ in range(40):
        vz = (zb - z0) / T
        vx = (xb - x0 - 0.5 * ax * T * T) / T
        vy = (-y0 + 0.5 * GRAVITY * T * T) / T
        crosses_net = (z0 > 0) != (zb > 0)
        if not crosses_net or net_clear is None:
            break
        t_net = -z0 / vz
        y_net = y0 + vy * t_net - 0.5 * GRAVITY * t_net * t_net
        if y_net >= NET_H + net_clear:
            break
        T *= 1.06
    return vx, vy, vz


class Game:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.state = SELECT
        self.state_t = 0.0          # time the current state started
        self.now = 0.0
        self.highlight = None       # opponent key under the AprilTag
        self.opp = None             # chosen opponent dict
        self.score = {PLAYER: 0, OPP: 0}
        self.server = PLAYER
        self.ball = Ball()
        self.opp_x = 0.0
        self.opp_target_x = 0.0
        self.opp_will_miss = False
        self.opp_hit_pending = False
        self.streak = 0
        self.record = 0.0
        self.winner = None
        self.point_winner = None
        self.point_reason = ""
        self.show_len = SHOW_MAX_S
        self.show_rematch = False
        self.match_best = 0         # best streak this match
        self.history = []           # finished matches this session: {opp, winner, you, them, best_streak}
        self._swings = []           # recent unconsumed swings
        self._events = []
        # what happened while the ball was in your hit zone (for miss feedback)
        self._zone_enter_t = 0.0
        self._zone_saw_pose = self._zone_saw_on_ball = self._zone_saw_swing = False

    # ---------------------------------------------------------------- input
    def set_highlight(self, opp_key):
        """AprilTag currently held up (or None). Only matters on the select screen."""
        if self.state == SELECT:
            self.highlight = opp_key if opp_key in OPPONENTS else None

    def confirm(self):
        """Manual confirm from the computer (Enter / Start button). On the
        select screen it starts a match (with the pre-match show), during the
        show it skips to the countdown, after a game it goes home."""
        if self.state == SELECT and self.highlight:
            self._start_match(self.highlight)
        elif self.state == SHOW and self.now - self.state_t >= SHOW_SKIP_LOCK_S:
            self._end_show(skipped=True)
        elif self.state == GAME_OVER:
            self.go_home()

    def rematch(self):
        """Play the same opponent again (game-over screen only), after a short show."""
        if self.state == GAME_OVER and self._game_over_unlocked():
            self._start_match(self.opp["key"], rematch=True)

    def end_show(self):
        """The show's script is done: go to the countdown now."""
        if self.state == SHOW:
            self._end_show(skipped=False)

    def set_show_len(self, seconds):
        """Shorten the show (e.g. nothing to say with commentary muted). Never lengthens it."""
        self.show_len = max(0.0, min(self.show_len, seconds))

    def go_home(self):
        """Back to the opponent-select screen. From the game-over screen this
        waits out GAME_OVER_LOCK_S; mid-match it quits right away."""
        if self.state == SELECT or (self.state == GAME_OVER and not self._game_over_unlocked()):
            return
        self.ball = Ball()
        self.highlight = None
        self._enter(SELECT)

    def _game_over_unlocked(self):
        return self.now - self.state_t >= GAME_OVER_LOCK_S

    def _start_match(self, opp_key, rematch=False):
        self.opp = OPPONENTS[opp_key]
        self.score = {PLAYER: 0, OPP: 0}
        self.winner = None
        self.server = PLAYER
        self.opp_x = 0.0
        self.streak = 0                 # streaks are per match; the record is per session
        self.match_best = 0
        self.ball = Ball()
        self.show_len = SHOW_REMATCH_S if rematch else SHOW_MAX_S
        self.show_rematch = rematch
        self._emit("start", opp=self.opp["key"], rematch=rematch)
        self._say("intro")
        self._enter(SHOW)

    def _end_show(self, skipped):
        self._emit("show_end", skipped=skipped)
        self._enter(COUNTDOWN)

    # --------------------------------------------------------------- update
    def update(self, now, dt, paddle, swings=()):
        """Advance the game to time `now`. Returns a list of event dicts
        (hit, miss, bounce, point, record, say, ...) that happened."""
        self.now = now
        self._swings.extend(swings)
        # forget swings too old to matter
        self._swings = [s for s in self._swings if now - s.t <= SWING_EARLY_S + 0.5]

        if self.state == SHOW:
            if now - self.state_t >= self.show_len:
                self._end_show(skipped=False)
        elif self.state == COUNTDOWN:
            if now - self.state_t >= COUNTDOWN_S:
                self._start_serve()
        elif self.state == SERVE:
            self._update_serve(now, paddle)
        elif self.state == POINT:
            if self.ball.live:
                self._physics(dt)
            if now - self.state_t >= POINT_PAUSE_S:
                self._after_point()

        if self.state in (RALLY, SERVE, POINT):
            self._move_opponent(dt)
            if self.state == RALLY:
                self._physics(dt)
                self._judge(paddle)
        # includes events raised between ticks (confirm(), set_highlight())
        events, self._events = self._events, []
        return events

    # ----------------------------------------------------------- internals
    def _enter(self, state):
        self.state = state
        self.state_t = self.now
        self._emit("state", state=state)

    def _emit(self, kind, **data):
        self._events.append({"type": kind, **data})

    def _say(self, mood, **fmt):
        if self.opp:
            line = self.rng.choice(self.opp["lines"][mood]).format(**fmt)
            self._emit("say", text=line, mood=mood)

    def _start_serve(self):
        self.ball = Ball(visible=True)
        self._swings.clear()
        self._enter(SERVE)
        if self.server == OPP:
            self._say("serve")

    def _update_serve(self, now, paddle):
        b = self.ball
        if self.server == PLAYER:
            # ball floats in front of your paddle until you swing
            b.x, b.y, b.z = paddle.x, min(max(paddle.y, 0.05) + 0.12, 0.9), PLAYER_HIT_Z - 0.1
            fresh = [s for s in self._swings if s.t >= self.state_t + 0.3]
            if fresh:
                self._player_hit(fresh[-1], paddle, serve=True)
        else:
            b.x, b.y, b.z = self.opp_x, 0.25, OPP_HIT_Z
            if now - self.state_t >= OPP_SERVE_DELAY_S:
                self._opp_shoot(serve=True)

    def _physics(self, dt, substeps=4):
        b = self.ball
        if not b.live:
            return
        h = dt / substeps
        for _ in range(substeps):
            prev_z = b.z
            b.vx += b.ax * h
            b.vy -= GRAVITY * h
            b.x += b.vx * h
            b.y += b.vy * h
            b.z += b.vz * h
            # net
            if not b.in_net and (prev_z > 0) != (b.z > 0) and b.y < NET_H \
                    and abs(b.x) <= TABLE_HALF_W + 0.15:
                b.in_net = True
                b.z = 0.02 if prev_z > 0 else -0.02
                b.vz *= -0.15
                b.vx *= 0.3
                self._emit("net")
            # table
            on_table = abs(b.x) <= TABLE_HALF_W and abs(b.z) <= TABLE_HALF_L
            if b.y <= 0 and b.vy < 0 and on_table and b.y > -0.05:
                b.y = 0.0
                b.vy = -b.vy * RESTITUTION
                b.ax = 0.0 if b.ax == 0 else b.ax * 0.5
                b.bounced_on = side_of(b.z)
                self._emit("bounce", side=b.bounced_on)
            # floor
            if b.y <= FLOOR_Y and b.vy < 0:
                b.y = FLOOR_Y
                b.vy = -b.vy * FLOOR_RESTITUTION
                b.vx *= 0.7
                b.vz *= 0.7

    def _judge(self, paddle):
        """Decide whether the rally just ended, or whether someone hits."""
        b = self.ball
        own_side = b.hitter
        other = OPP if own_side == PLAYER else PLAYER

        if b.in_net:
            return self._point(other, "net")
        if b.bounced_on == own_side:
            return self._point(other, "fault")
        if b.y <= FLOOR_Y + 0.01 and not b.bounced_on:
            return self._point(other, "out")

        if b.hitter == OPP and b.vz > 0:
            # incoming to you
            if b.z < HIT_ZONE[0]:
                self._zone_enter_t = self.now   # keeps updating until the ball enters
            elif b.z <= HIT_ZONE[1] and b.bounced_on == PLAYER:
                swings = [s for s in self._swings if s.t >= self._zone_enter_t - SWING_EARLY_S]
                on_ball = self._paddle_on_ball(paddle)
                self._zone_saw_pose |= paddle.visible
                self._zone_saw_on_ball |= on_ball
                self._zone_saw_swing |= bool(swings)
                if swings and on_ball:
                    return self._player_hit(swings[-1], paddle)
            elif b.bounced_on == PLAYER:
                self._emit("miss", reason=self._miss_reason())
                self._say("you_miss")
                return self._point(OPP, "miss")

        elif b.hitter == PLAYER and b.vz < 0:
            if b.bounced_on == OPP and b.z <= OPP_HIT_Z:
                if self.opp_will_miss:
                    if b.z < -2.4 or b.y <= FLOOR_Y + 0.01:
                        self._say("opp_miss")
                        return self._point(PLAYER, "opp_miss")
                else:
                    return self._opp_shoot()

    def _paddle_on_ball(self, paddle):
        b = self.ball
        return (paddle.visible
                and abs(paddle.x - b.x) <= HIT_RADIUS_X
                and abs(paddle.y - b.y) <= HIT_RADIUS_Y)

    def _miss_reason(self):
        """Tell the player WHICH signal failed while the ball was in the zone:
        not seen by the camera, paddle in the wrong place (pose), or no
        swing (IMU)."""
        if not self._zone_saw_pose:
            return "no_pose"
        if not self._zone_saw_on_ball:
            return "wrong_place"
        if not self._zone_saw_swing:
            return "no_swing"
        return "bad_timing"   # in place and swung, just never both at once

    def _player_hit(self, swing, paddle, serve=False):
        self._swings = [s for s in self._swings if s.t > swing.t]
        b = self.ball
        strength = max(0.6, min(swing.strength, 2.0))
        # aim: where the ball sits on your paddle steers it (like a real paddle
        # angle), and a sideways swing pulls it the same way
        offset = (b.x - paddle.x) / HIT_RADIUS_X
        xb = b.x * 0.3 + offset * 0.5 + paddle.vx * 0.15
        xb = max(-TABLE_HALF_W + 0.1, min(TABLE_HALF_W - 0.1, xb))
        zb = -self.rng.uniform(0.75, 1.15)
        speed = self.opp["speed"] * PLAYER_SPEED_SCALE * (0.85 + 0.2 * strength)
        y0 = max(b.y, 0.08)
        b.vx, b.vy, b.vz = solve_launch(b.x, y0, b.z, xb, zb, speed)
        b.y, b.ax = y0, 0.0
        b.live, b.hitter, b.bounced_on, b.in_net = True, PLAYER, "", False

        # opponent decides now whether this one beats them
        wide = abs(xb) / TABLE_HALF_W
        p_miss = self.opp["miss_pct"] * (1 + 0.8 * wide + 0.5 * (strength - 1))
        self.opp_will_miss = self.rng.random() < p_miss
        t_arrive = (OPP_HIT_Z - b.z) / b.vz
        pred_x = b.x + b.vx * t_arrive
        self.opp_target_x = pred_x + (self.rng.choice([-1, 1]) * 0.55 if self.opp_will_miss else 0)

        self.streak += 1
        self.match_best = max(self.match_best, self.streak)
        self._emit("hit", streak=self.streak, strength=round(strength, 2), serve=serve)
        if self.streak > self.record:
            self.record = float(self.streak)
            self._emit("record", record=self.record)
        if self.streak % STREAK_LINE_EVERY == 0:
            self._say("streak", n=self.streak)
        elif self.rng.random() < 0.25:
            self._say("you_hit")
        if serve:
            self._enter(RALLY)

    def _opp_shoot(self, serve=False):
        b = self.ball
        o = self.opp
        b.x = self.opp_x if serve else b.x
        b.z = OPP_HIT_Z
        y0 = 0.25 if serve else max(b.y, 0.1)
        xb = self.rng.uniform(-o["aim_x"], o["aim_x"])
        zb = self.rng.uniform(0.55, 1.05)
        spin = o["spin"] * self.rng.choice([-1, 1])
        b.vx, b.vy, b.vz = solve_launch(b.x, y0, b.z, xb, zb, o["speed"], ax=spin)
        b.y, b.ax = y0, spin
        b.live, b.visible, b.hitter, b.bounced_on, b.in_net = True, True, OPP, "", False
        self._zone_enter_t = self.now
        self._zone_saw_pose = self._zone_saw_on_ball = self._zone_saw_swing = False
        self._emit("opp_hit", serve=serve)
        if serve:
            self._enter(RALLY)

    def _move_opponent(self, dt):
        if self.state == SERVE and self.server == OPP:
            self.opp_target_x = 0.0
        elif self.ball.hitter == OPP or not self.ball.live:
            self.opp_target_x = 0.0  # drift back to the middle
        step = self.opp["reach"] * dt
        d = self.opp_target_x - self.opp_x
        self.opp_x += max(-step, min(step, d))

    def _point(self, winner, reason):
        self.point_winner, self.point_reason = winner, reason
        self.score[winner] += 1
        if winner == OPP:
            self.streak = 0
        self._emit("point", winner=winner, reason=reason, score=dict(self.score))
        self._enter(POINT)

    def _after_point(self):
        p, o = self.score[PLAYER], self.score[OPP]
        if max(p, o) >= GAME_POINTS and abs(p - o) >= 2:
            self.winner = PLAYER if p > o else OPP
            self.history.append({"opp": self.opp["key"], "winner": self.winner, "you": p, "them": o,
                                 "best_streak": self.match_best})
            self.ball = Ball()
            self._emit("game_over", winner=self.winner, score=dict(self.score))
            self._say("lose" if self.winner == PLAYER else "win")
            self._enter(GAME_OVER)
            return
        total = p + o
        deuce = p >= GAME_POINTS - 1 and o >= GAME_POINTS - 1
        turn = total if deuce else total // 2
        self.server = PLAYER if turn % 2 == 0 else OPP
        self._start_serve()

    # ------------------------------------------------------------- output
    def snapshot(self):
        b = self.ball
        return {
            "state": self.state,
            "state_age": round(self.now - self.state_t, 3),
            "highlight": self.highlight,
            "opp": self.opp["key"] if self.opp else None,
            "opp_name": self.opp["name"] if self.opp else None,
            "score": {"you": self.score[PLAYER], "opp": self.score[OPP]},
            "server": self.server,
            "streak": self.streak,
            "record": self.record,
            "winner": self.winner,
            "point_winner": self.point_winner,
            "point_reason": self.point_reason,
            "countdown": max(0.0, COUNTDOWN_S - (self.now - self.state_t))
            if self.state == COUNTDOWN else 0,
            "show": {"len": self.show_len, "rematch": self.show_rematch, "skip_lock": SHOW_SKIP_LOCK_S}
            if self.state == SHOW else None,
            # velocity lets the browser extrapolate smoothly between server updates
            "ball": {"x": round(b.x, 4), "y": round(b.y, 4), "z": round(b.z, 4),
                     "vx": round(b.vx, 3), "vy": round(b.vy, 3), "vz": round(b.vz, 3), "ax": round(b.ax, 3),
                     "live": b.live and self.state in (RALLY, POINT), "visible": b.visible},
            "opp_x": round(self.opp_x, 4),
        }
