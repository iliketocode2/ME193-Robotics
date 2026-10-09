"""
The match announcer. Turns game events (from game_logic.Game.update) into
spoken lines. Pure Python like game_logic.py; the browser does the actual
speaking (Web Speech API) -- see web/audio.js.

Each line is a "call" event sent to the browser alongside the game events:

    {"type": "call", "text": "...", "excite": 0..1, "interrupt": bool}

  excite     -- how hyped the delivery is (the browser raises pitch + rate)
  interrupt  -- cut off whatever the announcer is saying (used for point
                results, which must be heard while they're still true)

To keep it from talking over itself, the announcer estimates how long each
line takes to say and drops low-priority chatter while it's still "busy".
"""

import random

import game_logic as gl

SECONDS_PER_CHAR = 0.062       # rough speaking speed, used to estimate when a line ends
RALLY_CALLS = {                # rally length (shots by both players) -> (lines, excitement)
    6: (["Nice rally building here.", "Good exchange!", "Back and forth they go."], 0.5),
    10: (["What a rally!", "Ten shots and counting!", "This is fantastic table tennis!"], 0.75),
    14: (["Unbelievable rally!", "The crowd is on its feet!", "Neither player will give an inch!"], 0.9),
    20: (["Twenty shots! This is incredible!", "I have never seen a rally like this!"], 1.0),
}
STREAK_EVERY = 5
BIG_SWING = 1.6                # IMU swing strength that counts as a smash

WORDS = ("love one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three "
         "twenty-four twenty-five").split()

INTROS = {
    "pip": "Pip the Penguin is the crowd favourite, but don't let the waddle fool you.",
    "rita": "Coach Rita has trained champions. Expect her to attack the corners.",
    "viktor": "Viktor the Wall. Fast, silent, and every shot curves.",
}

POINT_LINES = {
    # (winner, reason) -> lines
    (gl.PLAYER, "opp_miss"): ["{opp} can't reach it!", "Too good for {opp}!", "A clean winner!", "Point to you!"],
    (gl.PLAYER, "net"): ["{opp} finds the net!", "Into the net from {opp}!"],
    (gl.PLAYER, "fault"): ["{opp} misses the table!", "An error from {opp}!"],
    (gl.PLAYER, "out"): ["Long from {opp}!", "{opp} sends it out!"],
    (gl.OPP, "no_swing"): ["No swing there!", "Caught flat-footed!", "Didn't pull the trigger!"],
    (gl.OPP, "wrong_place"): ["Out of position!", "Couldn't get across in time!", "Wrong side of the table!"],
    (gl.OPP, "no_pose"): ["Where did our player go?", "Stepped out of the picture!"],
    (gl.OPP, "bad_timing"): ["Ooh, just mistimed!", "So close!", "A fraction too late!"],
    (gl.OPP, "net"): ["Into the net!"],
    (gl.OPP, "fault"): ["That one misses the table."],
}
SPIN_LINES = {"viktor": ["Look at the curve on that!", "Wicked spin from Viktor!"],
              "rita": ["Right into the corner!", "Rita goes wide!"]}


def say_num(n):
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def score_call(you, opp, opp_name):
    """'Three all', 'Deuce!', 'Game point, you!', 'Seven, four to you.'"""
    if you >= gl.GAME_POINTS - 1 and opp >= gl.GAME_POINTS - 1:
        if you == opp:
            return "Deuce!"
        return "Advantage, you!" if you > opp else f"Advantage, {opp_name}."
    if you == opp:
        return f"{say_num(you).capitalize()} all."
    lead, trail = max(you, opp), min(you, opp)
    leader = "you" if you > opp else opp_name
    if lead == gl.GAME_POINTS - 1:
        return f"Game point, {leader}!"
    return f"{say_num(lead).capitalize()}, {say_num(trail)}, to {leader}."


class Announcer:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.busy_until = 0.0
        self.rally = 0              # shots by both players since the serve
        self.max_deficit = 0        # your worst deficit this game (for comeback calls)
        self.comeback_called = False
        self.record_at_rally_start = 0.0
        self.last_miss_reason = None

    def update(self, now, events, game):
        """Game events from this tick -> list of call events."""
        calls = []
        opp = game.opp["name"] if game.opp else "your opponent"
        for ev in events:
            kind = ev["type"]
            if kind == "start":
                self.__init__(self.rng)
                key = ev["opp"]
                text = self.rng.choice([
                    f"Welcome to the Rogers Cup! Today's match: you, versus {opp}.",
                    f"Good evening, and welcome to centre court! It's you against {opp}.",
                ]) + " " + INTROS.get(key, "")
                calls.append(self._call(now, text, 0.55, interrupt=True))
            elif kind == "hit":
                self.rally += 1
                calls += self._rally_calls(now, ev)
            elif kind == "opp_hit":
                if ev.get("serve"):
                    self.rally = 0
                    self.record_at_rally_start = game.record
                self.rally += 1
                lines = SPIN_LINES.get(game.opp["key"] if game.opp else "")
                if lines and not ev.get("serve") and self.rng.random() < 0.15:
                    calls.append(self._call(now, self.rng.choice(lines), 0.5, optional=True))
            elif kind == "state" and ev["state"] == gl.SERVE:
                self.rally = 0
                self.record_at_rally_start = game.record
            elif kind == "miss":
                self.last_miss_reason = ev["reason"]
            elif kind == "point":
                calls.append(self._point_call(now, ev, game, opp))
            elif kind == "game_over":
                calls.append(self._game_over_call(now, ev, game, opp))
        return [c for c in calls if c]

    # ------------------------------------------------------------ internals
    def _call(self, now, text, excite, interrupt=False, optional=False):
        """optional lines are dropped if the announcer is still talking."""
        if optional and now < self.busy_until:
            return None
        start = now if interrupt else max(now, self.busy_until)
        self.busy_until = start + len(text) * SECONDS_PER_CHAR / (1 + 0.25 * excite)
        return {"type": "call", "text": text, "excite": round(min(1.0, excite), 2), "interrupt": interrupt}

    def _rally_calls(self, now, ev):
        if self.rally in RALLY_CALLS:
            lines, excite = RALLY_CALLS[self.rally]
            return [self._call(now, self.rng.choice(lines), excite, optional=True)]
        n = ev.get("streak", 0)
        if n and n % STREAK_EVERY == 0:
            return [self._call(now, self.rng.choice(
                [f"That's {say_num(n)} in a row!", f"{say_num(n).capitalize()} straight hits!",
                 f"{say_num(n).capitalize()} consecutive returns. Superb!"]), 0.6, optional=True)]
        if ev.get("strength", 0) >= BIG_SWING:
            return [self._call(now, self.rng.choice(["Big forehand!", "Smash!", "What power!"]), 0.7,
                               optional=True)]
        return []

    def _point_call(self, now, ev, game, opp):
        winner, reason = ev["winner"], ev["reason"]
        you, them = ev["score"][gl.PLAYER], ev["score"][gl.OPP]
        if reason == "miss":
            reason = self.last_miss_reason or "wrong_place"
        self.last_miss_reason = None

        parts = [self.rng.choice(POINT_LINES.get((winner, reason), ["Point."])).format(opp=opp)]
        if self.rally >= 8:
            parts.append(f"That rally lasted {say_num(self.rally)} shots!")
        if game.record > self.record_at_rally_start and game.record >= 3:
            parts.append(f"And a new personal best: {say_num(int(game.record))} in a row!")

        self.max_deficit = max(self.max_deficit, them - you)
        game_ends = max(you, them) >= gl.GAME_POINTS and abs(you - them) >= 2
        if (not self.comeback_called and self.max_deficit >= 4 and you >= them and not game_ends):
            parts.append("What a comeback!")
            self.comeback_called = True
        if not game_ends:                      # the game-over call announces the final score
            parts.append(score_call(you, them, opp))

        excite = 0.45 + min(self.rally, 20) / 40
        if max(you, them) >= gl.GAME_POINTS - 1:
            excite += 0.25
        if winner == gl.OPP:
            excite -= 0.1
        return self._call(now, " ".join(parts), excite, interrupt=True)

    def _game_over_call(self, now, ev, game, opp):
        you, them = ev["score"][gl.PLAYER], ev["score"][gl.OPP]
        hi, lo = say_num(max(you, them)), say_num(min(you, them))
        if ev["winner"] == gl.PLAYER:
            text = self.rng.choice([f"Game! You win it, {hi} to {lo}! What a performance!",
                                    f"And that's the game! You beat {opp}, {hi} to {lo}!"])
            excite = 1.0
        else:
            text = self.rng.choice([f"And that's the game. {opp} takes it, {hi} to {lo}.",
                                    f"Game to {opp}, {hi} to {lo}. A valiant effort!"])
            excite = 0.55
        # queued (not interrupting) so the final point's call finishes first
        return self._call(now, text, excite)
