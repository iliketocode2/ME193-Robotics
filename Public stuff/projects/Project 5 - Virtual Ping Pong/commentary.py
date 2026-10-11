"""
The commentary booth. Pure Python like game_logic.py (no model, no audio);
the browser does the speaking (Web Speech API, web/audio.js) and the AI
model runs in its own process (ai_commentator.py).

Two commentators, like a TV broadcast:

  Ray   (Announcer)  play-by-play. Scripted, instant: point results, score
                     calls, rally/streak calls. Never waits on anything.
  Sonia (Booth)      colour analyst. Her lines are written by a local AI
                     model from a running MatchStory, between points and at
                     the start/end of the match. If a line is late it's
                     dropped -- the game never waits for the AI.

Every line is a "call" event sent to the browser with the game events:

    {"type": "call", "speaker": "ray"|"sonia", "text": "...", "excite": 0..1,
     "interrupt": bool}

  excite     -- how hyped the delivery is (the browser raises pitch + rate)
  interrupt  -- cut off whatever is being said (point results must be heard
                while they're still true)

To keep Ray from talking over himself, he estimates how long each line takes
to say and drops low-priority chatter while he's still "busy".
"""

import difflib
import json
import random
import re
from collections import Counter

import game_logic as gl
from weather import PLACE

SECONDS_PER_CHAR = 0.062       # rough speaking speed, used to estimate when a line ends
CLIP_SECONDS_PER_CHAR = 0.071  # the natural-voice clips, measured (they include short lead-in/out silences)
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
    (gl.OPP, "net"): ["Into the net!", "Caught the tape!"],
    (gl.OPP, "fault"): ["That one misses the table.", "Just off the edge!"],
    (gl.OPP, "out"): ["Long! Too much on that one.", "That sails past the end!"],
}
# Point and score calls use the short name: "Four, two, to Pip." is a second
# shorter than "...to Pip the Penguin." -- airtime Sonia gets instead.
SHORT_NAMES = {"pip": "Pip", "rita": "Rita", "viktor": "Viktor"}
SPIN_LINES = {"viktor":["Look at the curve on that!", "Wicked spin from Viktor!"],
              "rita": ["Right into the corner!", "Rita goes wide!"]}
WELCOMES = ["Welcome to the Rogers Cup! It's you versus {opp}.",
            "Good evening, and welcome to centre court! It's you against {opp}."]
HANDOFFS = ["Sonia, what do we make of this one?", "Sonia, your thoughts?"]
STREAK_LINES = ["That's {n} in a row!", "{N} straight hits!", "{N} consecutive returns. Superb!"]
SMASH_LINES = ["Big forehand!", "Smash!", "What power!"]
RALLY_LENGTH = "That rally lasted {n} shots!"
RECORD_LINES = {gl.PLAYER: "And a new personal best: {n} in a row!",
                gl.OPP: "Still, that's a new personal best: {n} in a row!"}
COMEBACK = "What a comeback!"
GAME_OVER_LINES = {gl.PLAYER: ["Game! You win it, {hi} to {lo}! What a performance!",
                               "And that's the game! You beat {opp}, {hi} to {lo}!"],
                   gl.OPP: ["And that's the game. {opp} takes it, {hi} to {lo}.",
                            "Game to {opp}, {hi} to {lo}. A valiant effort!"]}
NO_OPP = "your opponent"

# --- the pre-match show (ShowDirector) ------------------------------------
# Every sentence is pre-rendered in Ray's natural voice (ray_phrases()), so
# the show costs nothing to voice -- only Sonia's lines are synthesized.
SHOW_LATE = "Sonia is still finding her seat in the booth."
SHOW_HANDOFF = "Sonia?"                     # short: the show has 15 s
CLOSERS = ["Let's play!", "Players, ready!"]
REMATCH_LINES = ["Rematch! You and {opp} go again.", "It's a rematch with {opp}!"]      # short names
PLAYER_FIRST = "It's your first match of the day."
PLAYER_NTH = "This is match number {n} for our player today."                          # 2..25
PLAYER_BEST = "Your best streak so far: {n} in a row."                                 # 3..100
LAST_MEETING = {gl.PLAYER: "You won the last meeting.", gl.OPP: "{opp} won the last meeting."}   # full names
# The weather, from weather.bucket() (sky, time of day, temperature, wind)
WEATHER_ADJ = {"clear": "clear", "partly": "partly cloudy", "cloudy": "grey", "fog": "foggy", "drizzle": "drizzly",
               "rain": "rainy", "heavy_rain": "rain-soaked", "showers": "showery", "sleet": "sleety",
               "snow": "snowy", "heavy_snow": "wintry", "storm": "stormy"}
TIMES_OF_DAY = ("morning", "afternoon", "evening", "night")
OPENER = "Live from {place}, on a {adj} {tod}."
TEMP_LINE = "{n} degrees."                                                              # -20..110
TEMP_RANGE = (-20, 110)
FEELS_LINES = {"colder": "But it feels colder than that.", "warmer": "But it feels warmer than that."}
WIND_LINES = {0: "Barely a breath of wind.", 1: "A light breeze from the {dir}.", 2: "A stiff breeze from the {dir}.",
              3: "Strong winds from the {dir}.", 4: "It's blowing a gale out there!"}
COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
# Between points, when the weather changes (weather.WeatherWatch keys)
WEATHER_CHANGE_LINES = {
    "storm": "Thunder in the distance!", "rain_start": "And here comes the rain.", "rain_stop": "The rain has stopped.",
    "snow_start": "The snow has started to fall!", "snow_stop": "The snow has stopped.",
    "rain_heavier": "The rain is getting heavier!", "rain_easing": "The rain is easing off.",
    "night": "The floodlights are on as night falls.", "day": "The sun is coming up over Tufts.",
    "fog_in": "The fog is rolling in.", "fog_out": "The fog is lifting.", "wind_up": "The wind is really picking up.",
    "sun_out": "The sun has broken through.",
}
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n):
    """-20 -> 'minus twenty', 48 -> 'forty-eight', 105 -> 'one hundred and five'."""
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    rest = n % 100
    return number_words(n // 100) + " hundred" + ("" if rest == 0 else " and " + number_words(rest))


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

    def update(self, now, events, game, sonia_cued=False):
        """Game events from this tick -> list of call events (the match
        welcome is the ShowDirector's). sonia_cued=True when she's been asked
        about this point: Ray just calls the score and leaves the colour to
        her, so her line isn't stuck waiting behind his."""
        calls = []
        opp = game.opp["name"] if game.opp else NO_OPP
        short = SHORT_NAMES.get(game.opp["key"], opp) if game.opp else opp
        for ev in events:
            kind = ev["type"]
            if kind == "start":
                self.__init__(self.rng)          # the welcome is the pre-match show's (ShowDirector)
            elif kind == "weather_change" and ev.get("key") in WEATHER_CHANGE_LINES:
                call = self._call(now, WEATHER_CHANGE_LINES[ev["key"]], 0.6)   # queued after the point call
                call["kind"] = "weather"
                calls.append(call)
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
                calls.append(self._point_call(now, ev, game, short, brief=sonia_cued))
            elif kind == "game_over":
                calls.append(self._game_over_call(now, ev, game, opp))
        return [c for c in calls if c]

    # ------------------------------------------------------------ internals
    def _call(self, now, parts, excite, interrupt=False, optional=False):
        """parts: the call's sentences (one clip each in the natural voice; see
        tts.py). optional lines are dropped if the announcer is still talking."""
        if optional and now < self.busy_until:
            return None
        parts = [p for p in ([parts] if isinstance(parts, str) else parts) if p]
        text = " ".join(parts)
        start = now if interrupt else max(now, self.busy_until)
        self.busy_until = start + len(text) * SECONDS_PER_CHAR / (1 + 0.25 * excite)
        return {"type": "call", "speaker": "ray", "text": text, "parts": parts,
                "excite": round(min(1.0, excite), 2), "interrupt": interrupt}

    def _rally_calls(self, now, ev):
        if self.rally in RALLY_CALLS:
            lines, excite = RALLY_CALLS[self.rally]
            return [self._call(now, self.rng.choice(lines), excite, optional=True)]
        n = ev.get("streak", 0)
        if n and n % STREAK_EVERY == 0:
            return [self._call(now, streak_line(self.rng.choice(STREAK_LINES), n), 0.6, optional=True)]
        if ev.get("strength", 0) >= BIG_SWING:
            return [self._call(now, self.rng.choice(SMASH_LINES), 0.7, optional=True)]
        return []

    def _point_call(self, now, ev, game, opp, brief=False):
        winner, reason = ev["winner"], ev["reason"]
        you, them = ev["score"][gl.PLAYER], ev["score"][gl.OPP]
        if reason == "miss":
            reason = self.last_miss_reason or "wrong_place"
        self.last_miss_reason = None
        game_ends = max(you, them) >= gl.GAME_POINTS and abs(you - them) >= 2

        parts = []
        if not brief or game_ends:             # brief: Sonia is about to describe the point
            parts.append(self.rng.choice(POINT_LINES.get((winner, reason), ["Point."])).format(opp=opp))
        if self.rally >= 8:
            parts.append(RALLY_LENGTH.format(n=say_num(self.rally)))
        if game.record > self.record_at_rally_start and game.record >= 3:
            parts.append(RECORD_LINES[winner].format(n=say_num(int(game.record))))

        self.max_deficit = max(self.max_deficit, them - you)
        if (not self.comeback_called and self.max_deficit >= 4 and you >= them and not game_ends):
            parts.append(COMEBACK)
            self.comeback_called = True
        if not game_ends:                      # the game-over call announces the final score
            parts.append(score_call(you, them, opp))

        excite = 0.45 + min(self.rally, 20) / 40
        if max(you, them) >= gl.GAME_POINTS - 1:
            excite += 0.25
        if winner == gl.OPP:
            excite -= 0.1
        return self._call(now, parts, excite, interrupt=True)

    def _game_over_call(self, now, ev, game, opp):
        you, them = ev["score"][gl.PLAYER], ev["score"][gl.OPP]
        winner = ev["winner"]
        text = game_over_line(self.rng.choice(GAME_OVER_LINES[winner]), you, them, opp)
        # queued (not interrupting) so the final point's call finishes first
        return self._call(now, text, 1.0 if winner == gl.PLAYER else 0.55)


def streak_line(template, n):
    return template.format(n=say_num(n), N=say_num(n).capitalize())


def game_over_line(template, you, them, opp):
    return template.format(hi=say_num(max(you, them)), lo=say_num(min(you, them)), opp=opp)


def weather_calls(b):
    """weather.bucket() -> Ray's weather report: [[opener, temperature, (strong
    wind)]] -- one call, kept short: the whole show is at most 15 s."""
    adj = "sunny" if b["sky"] == "clear" and b["tod"] != "night" else WEATHER_ADJ[b["sky"]]
    temp = max(TEMP_RANGE[0], min(TEMP_RANGE[1], b["temp"]))
    parts = [OPENER.format(place=PLACE, adj=adj, tod=b["tod"]), TEMP_LINE.format(n=number_words(temp)).capitalize()]
    if b["wind_level"] >= 3:                     # only wind you can see in the flags and the rain
        parts.append(WIND_LINES[b["wind_level"]].format(dir=b["wind_from"]))
    return [parts]


def player_parts(game):
    """Ray on the human player, one sentence: the last meeting with this
    opponent, else a good streak today, else which match of the day it is."""
    n = len(game.history) + 1
    last = next((m for m in reversed(game.history) if game.opp and m["opp"] == game.opp["key"]), None)
    if last:
        return [LAST_MEETING[last["winner"]].format(opp=game.opp["name"])]
    if game.record >= 5:
        return [PLAYER_BEST.format(n=number_words(int(game.record)))]
    return [PLAYER_FIRST if n == 1 else PLAYER_NTH.format(n=number_words(n))]


def ray_phrases(max_count=100, max_score=25):
    """Every sentence Ray can say (each call is made of these), for
    pre-rendering his voice (tts.py --setup). Counts above max_count and
    scores above max_score are rare enough to fall back to the browser voice."""
    shorts = list(SHORT_NAMES.values())                 # (Ray only talks in a match: there's always an opponent)
    fulls = [o["name"] for o in gl.OPPONENTS.values()]
    out = {"Point.", COMEBACK, *SMASH_LINES, *HANDOFFS, *INTROS.values()}
    # the pre-match show
    out.update([SHOW_LATE, SHOW_HANDOFF, PLAYER_FIRST, *CLOSERS, *FEELS_LINES.values(), *WEATHER_CHANGE_LINES.values()])
    out.update(t.format(opp=name) for t in REMATCH_LINES for name in shorts)
    out.update(PLAYER_NTH.format(n=number_words(n)) for n in range(2, 26))
    out.update(PLAYER_BEST.format(n=number_words(n)) for n in range(3, max_count + 1))
    out.update(t.format(opp=name) for t in LAST_MEETING.values() for name in fulls)
    for sky in WEATHER_ADJ:
        for tod in TIMES_OF_DAY:
            out.update(weather_calls({"sky": sky, "tod": tod, "temp": 50, "wind_level": 0, "wind_from": "north"})[0])
    out.update(TEMP_LINE.format(n=number_words(n)).capitalize() for n in range(TEMP_RANGE[0], TEMP_RANGE[1] + 1))
    out.update(line.format(dir=d) for line in WIND_LINES.values() for d in COMPASS)
    for lines in POINT_LINES.values():
        out.update(line.format(opp=name) for line in lines for name in shorts)
    for lines, _ in RALLY_CALLS.values():
        out.update(lines)
    for lines in SPIN_LINES.values():
        out.update(lines)
    out.update(w.format(opp=name) for w in WELCOMES for name in fulls)
    for n in range(STREAK_EVERY, max_count + 1, STREAK_EVERY):
        out.update(streak_line(t, n) for t in STREAK_LINES)
    out.update(RALLY_LENGTH.format(n=say_num(n)) for n in range(8, max_count + 1))
    out.update(t.format(n=say_num(n)) for t in RECORD_LINES.values() for n in range(3, max_count + 1))
    g = gl.GAME_POINTS
    for you in range(max_score + 1):
        for them in range(max_score + 1):
            hi, lo = max(you, them), min(you, them)
            if hi >= g and hi - lo >= 2:                       # a finished game
                if hi == g or hi - lo == 2:                    # ...that can actually happen
                    winner = gl.PLAYER if you > them else gl.OPP
                    out.update(game_over_line(t, you, them, name)
                               for t in GAME_OVER_LINES[winner] for name in fulls)
            elif hi <= g or hi - lo <= 1:                      # a score mid-game
                out.update(score_call(you, them, name) for name in shorts)
    return out


# =========================================================================
# The pre-match show
# =========================================================================

class ShowDirector:
    """Runs the pre-match show (game state SHOW, at most game.show_len):

        Ray    welcome                              ("Welcome to the Rogers Cup! It's you versus Pip the Penguin.")
        Ray    the weather [+ handoff to Sonia]      ("Live from Tufts, on a rainy evening. Forty-eight degrees.")
        Sonia  the opponent, given the conditions   (written ahead; else Ray's opponent intro)
        Sonia  what our player must do              (written ahead; else Ray: "This is match number three...")
        Ray    "Let's play!"                         (if it still fits)

    Sonia's lines are pre-written (Booth.prepare_show) while you pick an
    opponent; a slot waits at most SONIA_WAIT_S for hers, then moves on.
    Each line is released LEAD_S before the previous one should end (the
    browser queues it), and a line that wouldn't finish inside the show is
    left out. When the script is done the show ends early (game.end_show()).

    durations(call) -> seconds a call takes to say (clip lengths if known)."""

    LEAD_S = 0.3
    SONIA_WAIT_S = 1.0
    SONIA_LATEST_S = {"opp": 9.5, "player": 12.5, "rematch": 5.0}   # a slot can't start later than this
    MUTED_S = 6.0               # commentary muted: captions only, short show
    TAIL_S = 0.4                # pause after the last line before the countdown
    OVERRUN_S = 1.5             # the last line may run this far into the countdown (no speech there)

    def __init__(self, rng=None, durations=None):
        self.rng = rng or random.Random()
        self.durations = durations or (lambda call: call.get("secs") or len(call["text"]) * CLIP_SECONDS_PER_CHAR)
        self.active = False
        self.beats = []
        self.last_weather = None   # (sky, tod, temp, wind) of the last show, so rematches skip a repeat

    def update(self, now, events, game, booth, wx_bucket, sonia, muted):
        """Call every tick. sonia: "ready" | "loading" | "off" (no AI). Returns
        this tick's call events."""
        for ev in events:
            if ev["type"] == "start":
                self._start(now, game, booth, wx_bucket, sonia, muted, ev.get("rematch", False))
            elif ev["type"] == "show_end" or (ev["type"] == "state" and ev["state"] == gl.SELECT):
                self.active = False
        if not self.active or game.state != gl.SHOW:
            return []
        calls = []
        while self.beats and now >= self.free_at - self.LEAD_S:
            beat = self.beats[0]
            nxt = self.beats[1] if len(self.beats) > 1 else None
            call = self._resolve(beat, nxt, now, booth)
            if call == "wait":
                break
            self.beats.pop(0)
            if not call:
                continue
            dur = self.durations(call)
            start = max(now, self.free_at)
            if start + dur > self.t0 + game.show_len + self.OVERRUN_S and calls + self.emitted:
                continue                         # wouldn't finish in time: leave it out
            self.free_at = start + dur
            self.emitted.append(call)
            calls.append(call)
        if not self.beats and now >= self.free_at + self.TAIL_S:
            self.active = False
            game.end_show()
        return calls

    # ------------------------------------------------------------ internals
    def _start(self, now, game, booth, b, sonia_state, muted, rematch):
        self.active, self.t0, self.free_at, self.emitted = True, now, now, []
        opp = game.opp
        key, full, short = opp["key"], opp["name"], SHORT_NAMES.get(opp["key"], opp["name"])
        wx = weather_calls(b) if b else []
        wx_key = (b["sky"], b["tod"], b["temp"], b["wind_level"]) if b else None
        if muted:
            game.set_show_len(self.MUTED_S)
        sonia = sonia_state == "ready" and not muted
        beats = []
        if rematch:
            beats.append(("ray", [self.rng.choice(REMATCH_LINES).format(opp=short)], True))
            if sonia:
                beats.append(("sonia", "rematch", None))
            if wx and wx_key != self.last_weather:
                beats += [("ray", parts, False) for parts in wx]
        else:
            beats.append(("ray", [self.rng.choice(WELCOMES).format(opp=full)], True))
            if sonia_state == "loading" and not muted:
                beats.append(("ray", [SHOW_LATE], False))
            beats += [("ray", parts, False) for parts in wx]
            beats.append(("sonia", "opp", INTROS.get(key)) if sonia else ("ray", [INTROS.get(key, "")], False))
            player = player_parts(game)[0]
            beats.append(("sonia", "player", player) if sonia else ("ray", [player], False))
        beats.append(("ray", [self.rng.choice(CLOSERS)], False))
        self.beats = [{"who": w, "what": what, "extra": extra, "tried": None} for w, what, extra in beats]
        self.key = key
        self.last_weather = wx_key

    def _resolve(self, beat, nxt, now, booth):
        """A beat -> a call event, None (skip it) or "wait" (Sonia's line may still come)."""
        if beat["who"] == "ray":
            parts = beat["what"]
            if nxt and nxt["who"] == "sonia" and booth.has_show(self.key, nxt["what"]):
                parts = parts + [SHOW_HANDOFF]                       # her line is ready: hand over
            return self._ray(parts, interrupt=beat["extra"] is True)
        slot = beat["what"]
        call = booth.take_show(self.key, slot)
        if call:
            return dict(call, kind="show")
        if beat["tried"] is None:
            beat["tried"] = now
        latest = min(self.t0 + self.SONIA_LATEST_S[slot], beat["tried"] + self.SONIA_WAIT_S)
        if now < latest:
            return "wait"
        return self._ray([beat["extra"]], interrupt=False) if beat["extra"] else None   # Ray covers for her

    @staticmethod
    def _ray(parts, interrupt=False):
        parts = [p for p in parts if p]
        if not parts:
            return None
        return {"type": "call", "speaker": "ray", "text": " ".join(parts), "parts": parts,
                "excite": 0.55, "interrupt": interrupt, "kind": "show"}


# =========================================================================
# Sonia -- the AI colour analyst
# =========================================================================

PERSONAS = {
    "pip": "Pip the Penguin: goofy, cheerful, plays slow loopy lobs and misses a lot",
    "rita": "Coach Rita: a competitive coach who attacks the corners and gives tips",
    "viktor": "Viktor the Wall: a silent robot whose fast shots all curve; almost never misses",
}

# How each point ended, in words the model can riff on.
HOW = {
    (gl.PLAYER, "opp_miss"): "the opponent couldn't reach our player's return",
    (gl.PLAYER, "net"): "the opponent hit the net",
    (gl.PLAYER, "fault"): "the opponent missed the table",
    (gl.PLAYER, "out"): "the opponent hit it long",
    (gl.OPP, "no_swing"): "our player never swung",
    (gl.OPP, "wrong_place"): "our player was out of position",
    (gl.OPP, "no_pose"): "our player stepped out of view",
    (gl.OPP, "bad_timing"): "our player was in place but mistimed the swing",
    (gl.OPP, "net"): "our player hit the net",
    (gl.OPP, "fault"): "our player missed the table",
    (gl.OPP, "out"): "our player hit it long",
}
PATTERN_WORDS = {"wrong_place": "caught out of position", "no_swing": "frozen without swinging",
                 "bad_timing": "mistiming the swing", "no_pose": "drifting out of view"}

# The system prompt and examples never change, so the model runtime can reuse
# its work on them between requests (only the final situation is new).
ANALYST_SYSTEM = (
    "You are Sonia, the colour commentator on a live TV broadcast of a cartoon table tennis video game. "
    "The human player is 'our player'. Your play-by-play partner Ray has just called the action and the "
    "score, so never state the score or any numbers from it. React to the match story in ONE short spoken "
    "sentence of at most 12 words, in first person, present tense, talking to the viewers. Build the line "
    "around the 'focus' field. Use only facts given in the situation, and get right who is winning. "
    "Don't start with the opponent's name, never start the way any line in 'avoid_repeating' starts, "
    "and don't reuse its images or phrases. Be warm, witty and British. "
    "No emojis, quotes, hashtags or stage directions."
)

# What Sonia's line should be about -- picked at random (never twice running)
# so she doesn't fall into one groove ("Pip's lobs...", "Pip's lobs...").
FOCUS = {
    "intro": ["the opponent's style", "what our player must do to win", "the atmosphere in the arena",
              "a playful joke about the opponent"],
    # pre-match show slots (ShowDirector): the opponent, then our player
    "show_opp": ["how the conditions suit the opponent", "the opponent's style", "a playful joke about the opponent",
                 "the atmosphere in the arena"],
    "show_player": ["what our player must do to win", "our player's form today", "how the conditions affect our player"],
    "show_rematch": ["what changes in the rematch", "revenge or repeat", "the opponent's mood after last time"],
    "point": ["how the point was won or lost", "our player's footwork and positioning", "the opponent's reaction",
              "the crowd's mood", "a quick tactical tip for our player", "the momentum of the match",
              "a playful joke about the opponent", "the length and quality of the rally"],
    "game_over": ["the turning point of the match", "our player's performance", "the opponent's reaction",
                  "the crowd's send-off"],
}
FEW_SHOT = [
    ({"moment": "point", "opponent": PERSONAS["rita"], "point_to": "opponent",
      "how": HOW[(gl.OPP, "wrong_place")], "rally_shots": 9, "pattern": "our player caught out of position 3 times",
      "momentum": "Coach Rita has won 3 in a row", "score_situation": "Coach Rita leads by 3",
      "on_top": "Coach Rita is in control", "focus": "a quick tactical tip for our player"},
     "Our player has to stop drifting wide, because Rita is feasting on that corner."),
    ({"moment": "point", "opponent": PERSONAS["pip"], "point_to": "player", "how": HOW[(gl.PLAYER, "opp_miss")],
      "rally_shots": 12, "player_streak": 8, "score_situation": "our player leads by 2",
      "focus": "the length and quality of the rally"},
     "Twelve shots of pure nerve, and it's Pip who blinks first!"),
    ({"moment": "point", "opponent": PERSONAS["viktor"], "point_to": "opponent", "how": HOW[(gl.OPP, "no_swing")],
      "rally_shots": 3, "score_situation": "Viktor the Wall leads by 2", "focus": "the crowd's mood"},
     "You could hear a pin drop as that one curled past a frozen racket."),
    ({"moment": "show", "opponent": PERSONAS["viktor"], "conditions": "clear skies, mild, afternoon",
      "our_player": "first match today", "focus": "what our player must do to win"},
     "Quick feet will be everything today, because Viktor bends every ball like a banana."),
    ({"moment": "show", "opponent": PERSONAS["pip"], "conditions": "light rain, chilly, breezy, evening",
      "our_player": "won 1 of 2 matches today", "focus": "how the conditions suit the opponent"},
     "A penguin in the rain? Pip will feel right at home out there tonight."),
    ({"moment": "game_over", "opponent": PERSONAS["rita"], "winner": "player",
      "comeback": "our player came back from well behind", "longest_rally": 15, "best_streak": 10,
      "focus": "the opponent's reaction"},
     "What a fightback! Even Coach Rita has to tip her cap to that."),
]


def build_messages(situation):
    """Chat messages for the analyst model: fixed system prompt + examples,
    then this moment's situation as compact JSON."""
    msgs = [{"role": "system", "content": ANALYST_SYSTEM}]
    for sit, line in FEW_SHOT:
        msgs.append({"role": "user", "content": json.dumps(sit, separators=(",", ":"))})
        msgs.append({"role": "assistant", "content": line})
    msgs.append({"role": "user", "content": json.dumps(situation, separators=(",", ":"))})
    return msgs


_SCORE_RE = re.compile(r"\b\d{1,2}\s*(-|–|to)\s*\d{1,2}\b|\b\d{1,2}\s+all\b", re.I)
_NON_SPEECH = re.compile(r"[*_#`~<>\[\]{}|\\]")
MAX_WORDS = 20
NAMES = ("Pip", "Rita", "Viktor")


def sanitize_line(text, names=NAMES):
    """Model output -> one clean speakable sentence, or None to drop it."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)       # safety net for thinking models
    text = text.replace("’", "'").replace("‘", "'").replace("“", "").replace("”", "")
    text = re.sub(r"\s*[—–]\s*", ", ", text)                     # dashes -> a spoken pause
    text = re.sub(r"\s*,(\s*,)+", ",", text)                               # ", ," -> ","
    text = text.encode("ascii", "ignore").decode()                        # drops emojis and odd symbols
    text = _NON_SPEECH.sub("", text).strip().strip('"').strip()
    text = re.sub(r"^(Sonia|Ray)\s*:\s*", "", text, flags=re.I).replace('"', "")
    text = re.sub(r"\s+([,.!?])", r"\1", " ".join(text.split()))
    m = re.match(r"(.+?[.!?])(\s|$)", text)                              # first sentence only...
    if m and len(m.group(1).split()) >= 4:                                # ...unless it's a tiny one
        text = m.group(1)
    words = text.split()
    if len(words) < 4 or len(words) > MAX_WORDS + 6:
        return None
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS]).rstrip(",;:") + "."
    if _SCORE_RE.search(text):                                            # Ray already called the score
        return None
    text = _fix_names(text, names)
    if text[-1] not in ".!?":
        text += "."
    return text


def _fix_names(text, names):
    """Small models sometimes misspell names ("Vikor"): snap near-misses back."""
    def fix(m):
        word = m.group(0)
        close = difflib.get_close_matches(word, names, n=1, cutoff=0.75)
        return close[0] if close and word not in names else word
    return re.sub(r"\b[A-Z][a-z]{2,}\b", fix, text)


def _content_words(text):
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3}


def _opening(text):
    return " ".join(re.findall(r"[a-z']+", text.lower())[:2])


def too_similar(text, recent, threshold=0.5):
    """Near-duplicate of a recent line, or opens the same way ("Pip's lobs...")."""
    words, opening = _content_words(text), _opening(text)
    if recent and opening.split()[:1] == _opening(recent[-1]).split()[:1]:
        return True                                # same first word as her last line ("Pip's... Pip's...")
    for r in recent:
        if opening and opening == _opening(r):
            return True
        other = _content_words(r)
        if words and other and len(words & other) / len(words | other) >= threshold:
            return True
    return False


class MatchStory:
    """Everything a colour analyst would have noticed this match."""

    def __init__(self):
        self.reset(None)

    def reset(self, opp_key):
        self.opp_key = opp_key
        self.points = []               # [(winner, reason, rally_shots)]
        self.rally = 0
        self.longest_rally = 0
        self.max_deficit = 0
        self.comeback = False
        self.miss_counts = Counter()
        self.best_streak = 0
        self.last_miss_reason = None

    def update(self, events, game):
        for ev in events:
            kind = ev["type"]
            if kind == "start":
                self.reset(ev["opp"])
            elif kind in ("hit", "opp_hit"):
                if ev.get("serve"):
                    self.rally = 0
                self.rally += 1
                self.longest_rally = max(self.longest_rally, self.rally)
                if kind == "hit":
                    self.best_streak = max(self.best_streak, ev.get("streak", 0))
            elif kind == "miss":
                self.last_miss_reason = ev["reason"]
            elif kind == "point":
                reason = ev["reason"]
                if reason == "miss":
                    reason = self.last_miss_reason or "wrong_place"
                    self.miss_counts[reason] += 1
                self.last_miss_reason = None
                you, them = ev["score"][gl.PLAYER], ev["score"][gl.OPP]
                self.max_deficit = max(self.max_deficit, them - you)
                if self.max_deficit >= 4 and you >= them:
                    self.comeback = True
                self.points.append((ev["winner"], reason, self.rally))

    def run(self):
        """(winner, length) of the current run of consecutive points."""
        if not self.points:
            return None, 0
        who, n = self.points[-1][0], 0
        for w, _, _ in reversed(self.points):
            if w != who:
                break
            n += 1
        return who, n

    def situation(self, moment, game):
        """Compact JSON-able summary of the match story for the analyst model."""
        opp_name = game.opp["name"] if game.opp else "the opponent"
        sit = {"moment": moment, "opponent": PERSONAS.get(self.opp_key, opp_name)}
        if moment == "intro":
            return sit
        you, them = game.score[gl.PLAYER], game.score[gl.OPP]
        if moment == "game_over":
            sit["winner"] = "player" if game.winner == gl.PLAYER else "opponent"
            if self.comeback:
                sit["comeback"] = "our player came back from well behind"
            sit["longest_rally"] = self.longest_rally
            sit["best_streak"] = self.best_streak
            sit["margin"] = "close" if abs(you - them) <= 2 else "comfortable"
            return sit
        winner, reason, rally = self.points[-1]
        sit["point_to"] = "player" if winner == gl.PLAYER else "opponent"
        sit["how"] = HOW.get((winner, reason), "the rally ended")
        sit["rally_shots"] = rally
        who, n = self.run()
        if n >= 3:
            sit["momentum"] = f"{'our player' if who == gl.PLAYER else opp_name} has won {n} in a row"
        common, count = (self.miss_counts.most_common(1) or [(None, 0)])[0]
        if count >= 2 and common in PATTERN_WORDS:
            sit["pattern"] = f"our player {PATTERN_WORDS[common]} {count} times"
        if game.streak >= 5:
            sit["player_streak"] = game.streak
        diff = you - them
        if you >= gl.GAME_POINTS - 1 and them >= gl.GAME_POINTS - 1 and diff == 0:
            sit["score_situation"] = "deuce"
        elif max(you, them) >= gl.GAME_POINTS - 1 and diff != 0:
            sit["score_situation"] = f"game point to {'our player' if diff > 0 else opp_name}"
        elif diff == 0:
            sit["score_situation"] = "level"
        else:
            sit["score_situation"] = f"{'our player' if diff > 0 else opp_name} leads by {abs(diff)}"
        if abs(diff) >= 3:
            sit["on_top"] = f"{'our player is' if diff > 0 else opp_name + ' is'} in control"
        if self.comeback:
            sit["comeback"] = "our player has fought back from well behind"
        return sit


def player_profile(history, record):
    """The human player's session so far, in words, for Sonia."""
    if not history:
        return "first match today"
    wins = sum(m["winner"] == gl.PLAYER for m in history)
    text = f"won {wins} of {len(history)} matches today"
    return text + (f", best streak {int(record)}" if record >= 3 else "")


class Booth:
    """Decides when Sonia speaks, and keeps her lines fresh.

    update() returns analyst requests ({"id", "messages", "seed", ...}) for
    the AI worker; accept() turns a finished line into a call event -- or
    drops it if it's stale (the match has moved on) or repetitive.

    Pre-match show lines are written AHEAD (prepare_show(), while an opponent
    is picked on the menu) into show_cache; the ShowDirector take_show()s them
    when their slot comes up."""

    COOLDOWN_S = 7.0           # minimum gap between Sonia's point lines
    EVERY_NTH_POINT = 3        # quiet points still get a line about this often
    POINT_TTL_S = 30.0         # backstop only: a point line is fresh until the NEXT point ends
    SHOW_TTL_S = 600.0         # pre-written show lines keep (the weather is in them)
    SHOW_REASK_S = 30.0        # a show request with no answer this long was dropped: ask again

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.story = MatchStory()
        self.next_id = 0
        self.pending = {}          # id -> request
        self.held = {}             # id -> (request, call): accepted lines waiting for their voice clip
        self.show_cache = {}       # (opp, slot) -> call, ready to air
        self.recent = []           # Sonia's last lines (anti-repetition)
        self.last_spoke = -1e9
        self.last_asked = -1e9
        self.points_since_line = 0
        self.last_focus = None
        self.conditions = None     # weather.describe(): "light rain, chilly, breezy, evening"
        self.weather_notable = False

    def update(self, now, events, game, ready):
        """Game events -> analyst requests (none unless the AI worker is ready)."""
        self.story.update(events, game)
        if not ready:
            return []
        out = []
        for ev in events:
            kind = ev["type"]
            if kind == "start":                 # (show lines were written before the match started)
                self.pending = {i: r for i, r in self.pending.items() if r["kind"] == "show"}
                self.held = {i: h for i, h in self.held.items() if h[0]["kind"] == "show"}
                self.points_since_line = 0
            elif kind == "point" and not self._game_ends(ev):
                self.points_since_line += 1
                if self._worth_a_line(now, game):
                    out.append(self._request(now, "point", game, ttl=self.POINT_TTL_S))
            elif kind == "game_over":
                out.append(self._request(now, "game_over", game, ttl=15.0))
        return out

    def accept(self, result, now, game, hold=False):
        """Worker result {"id", "text", "latency"} -> Sonia call event, or None if
        it's late, stale, malformed or repetitive. hold=True: the line is kept
        back while her voice clip is rendered -- release() it afterwards.
        A show line goes to show_cache (returns None) unless held."""
        req = self.pending.pop(result["id"], None)
        if req is None or self._stale(req, now, game):
            return None
        text = sanitize_line(result.get("text", ""))
        if not text or too_similar(text, self.recent):
            return None
        self.recent = (self.recent + [text])[-6:]
        if req["kind"] != "show":
            self.last_spoke = now
            self.points_since_line = 0
        call = {"type": "call", "speaker": "sonia", "text": text, "excite": req["excite"],
                "interrupt": False, "kind": req["kind"], "latency": round(result.get("latency", 0.0), 2)}
        if hold:
            self.held[result["id"]] = (req, call)
            return call
        if req["kind"] == "show":
            self.show_cache[(req["opp"], req["slot"])] = call
            return None
        return call

    def release(self, req_id, now, game, clips=None, secs=None):
        """A held line whose voice clip is done (clips=[url], secs = its length)
        or failed (None: the browser voice says it) -> the call event, or None
        if the match has moved on meanwhile (or it's a show line: cached)."""
        req, call = self.held.pop(req_id, (None, None))
        if req is None or self._stale(req, now, game):
            return None
        call = dict(call)
        if clips:
            call["clips"] = clips
        if secs:
            call["secs"] = secs
        if req["kind"] == "show":
            self.show_cache[(req["opp"], req["slot"])] = call
            return None
        return call

    def _stale(self, req, now, game):
        if now > req["expires"]:
            return True
        if req["kind"] == "point":                  # talking over the next rally is fine (TV analysts do);
            return len(self.story.points) != req["points"]   # the NEXT point ending is not
        if req["kind"] == "show":                   # written for a different opponent's match
            return game.state != gl.SELECT and game.opp is not None and game.opp["key"] != req["opp"]
        return False

    # ------------------------------------------------------- pre-match show
    def prepare_show(self, now, opp_key, game, rematch=False):
        """Requests for the show's Sonia slots (written ahead, speculative =
        prio 2), for whichever are neither cached nor already asked."""
        slots = ("rematch",) if rematch else ("opp", "player")
        for i in [i for i, r in self.pending.items()           # asked long ago, no answer: the worker
                  if r["kind"] == "show" and now - r["asked"] > self.SHOW_REASK_S]:   # dropped it -- ask again
            del self.pending[i]
        asked = {(r["opp"], r["slot"]) for r in self.pending.values() if r["kind"] == "show"}
        asked |= {(h[0]["opp"], h[0]["slot"]) for h in self.held.values() if h[0]["kind"] == "show"}
        out = []
        for slot in slots:
            key = (opp_key, slot)
            if key not in self.show_cache and key not in asked:
                out.append(self._show_request(now, opp_key, slot, game))
        return out

    def commit_show(self, opp_key):
        """The match vs opp_key is starting: ids of show requests for anyone
        else, to drop from the worker so the right lines come first."""
        drop = [i for i, r in self.pending.items() if r["kind"] == "show" and r["opp"] != opp_key]
        for i in drop:
            del self.pending[i]
        return drop

    def has_show(self, opp_key, slot):
        return (opp_key, slot) in self.show_cache

    def take_show(self, opp_key, slot):
        """The cached show line (a call event) -- used up once aired."""
        return self.show_cache.pop((opp_key, slot), None)

    def _show_request(self, now, opp_key, slot, game):
        self.next_id += 1
        sit = {"moment": "rematch" if slot == "rematch" else "show",
               "opponent": PERSONAS.get(opp_key, opp_key)}
        if self.conditions:
            sit["conditions"] = self.conditions
        sit["our_player"] = player_profile(game.history, game.record)
        if slot == "rematch" and game.history:
            last = game.history[-1]
            sit["last_match"] = "our player won it" if last["winner"] == gl.PLAYER else "the opponent won it"
        focus = FOCUS["show_" + slot]
        sit["focus"] = self.rng.choice([f for f in focus if f != self.last_focus])
        self.last_focus = sit["focus"]
        if self.recent:
            sit["avoid_repeating"] = self.recent[-3:]
        req = {"id": self.next_id, "kind": "show", "opp": opp_key, "slot": slot, "prio": 2, "asked": now,
               "messages": build_messages(sit), "expires": now + self.SHOW_TTL_S,
               "points": 0, "excite": 0.5, "seed": self.rng.randrange(1, 2 ** 31)}
        self.pending[req["id"]] = req
        return req

    # ------------------------------------------------------------ internals
    def _request(self, now, kind, game, ttl):
        self.next_id += 1
        sit = self.story.situation(kind, game)
        focus = FOCUS[kind] + (["the weather conditions"] if self.weather_notable and self.conditions else [])
        sit["focus"] = self.rng.choice([f for f in focus if f != self.last_focus])
        self.last_focus = sit["focus"]
        if sit["focus"] == "the weather conditions":
            sit["conditions"] = self.conditions
        if self.recent:
            sit["avoid_repeating"] = self.recent[-3:]
        req = {"id": self.next_id, "kind": kind, "messages": build_messages(sit),
               "expires": now + ttl, "points": len(self.story.points), "prio": 1,
               "excite": self._excite(kind, sit), "seed": self.rng.randrange(1, 2 ** 31)}
        self.pending[req["id"]] = req
        if kind == "point":
            self.last_asked = now
        return req

    def _worth_a_line(self, now, game):
        if now - max(self.last_spoke, self.last_asked) < self.COOLDOWN_S:
            return False
        s = self.story
        _, run = s.run()
        last_rally = s.points[-1][2] if s.points else 0
        you, them = game.score[gl.PLAYER], game.score[gl.OPP]
        big = (last_rally >= 6 or run >= 3 or s.comeback
               or max(you, them) >= gl.GAME_POINTS - 1
               or max(s.miss_counts.values(), default=0) in (3, 5))
        return big or self.points_since_line >= self.EVERY_NTH_POINT

    @staticmethod
    def _game_ends(ev):
        you, them = ev["score"][gl.PLAYER], ev["score"][gl.OPP]
        return max(you, them) >= gl.GAME_POINTS and abs(you - them) >= 2

    @staticmethod
    def _excite(kind, sit):
        if kind == "game_over":
            return 0.8 if sit.get("winner") == "player" else 0.45
        if kind in ("intro", "show"):
            return 0.5
        e = 0.35 + min(sit.get("rally_shots", 0), 16) / 40
        if "momentum" in sit or "comeback" in sit or "game point" in sit.get("score_situation", ""):
            e += 0.2
        return round(min(e, 0.9), 2)
