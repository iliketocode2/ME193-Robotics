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
    (gl.OPP, "net"): ["Into the net!", "Caught the tape!"],
    (gl.OPP, "fault"): ["That one misses the table.", "Just off the edge!"],
    (gl.OPP, "out"): ["Long! Too much on that one.", "That sails past the end!"],
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

    def update(self, now, events, game, handoff=False):
        """Game events from this tick -> list of call events. handoff=True
        when Sonia (the AI analyst) is ready: Ray keeps the welcome short and
        hands over to her for the opponent introduction."""
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
                ])
                text += " " + (self.rng.choice(["Sonia, what do we make of this one?",
                                                "Sonia, your thoughts?"]) if handoff else INTROS.get(key, ""))
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
        return {"type": "call", "speaker": "ray", "text": text, "excite": round(min(1.0, excite), 2),
                "interrupt": interrupt}

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
            lead = "And a new personal best" if winner == gl.PLAYER else "Still, that's a new personal best"
            parts.append(f"{lead}: {say_num(int(game.record))} in a row!")

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
    ({"moment": "intro", "opponent": PERSONAS["viktor"], "focus": "what our player must do to win"},
     "Quick feet will be everything today, because Viktor bends every ball like a banana."),
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


class Booth:
    """Decides when Sonia speaks, and keeps her lines fresh.

    update() returns analyst requests ({"id", "messages", "seed", ...}) for
    the AI worker; accept() turns a finished line into a call event -- or
    drops it if it's stale (the match has moved on) or repetitive."""

    COOLDOWN_S = 7.0           # minimum gap between Sonia's point lines
    EVERY_NTH_POINT = 3        # quiet points still get a line about this often
    STALE_RALLY_SHOTS = 4      # a point line is dropped once the next rally is this far along
                               # (talking over the next serve is fine -- TV analysts do)

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.story = MatchStory()
        self.next_id = 0
        self.pending = {}          # id -> request
        self.recent = []           # Sonia's last lines (anti-repetition)
        self.last_spoke = -1e9
        self.last_asked = -1e9
        self.points_since_line = 0
        self.last_focus = None

    def update(self, now, events, game, ready):
        """Game events -> analyst requests (none unless the AI worker is ready)."""
        self.story.update(events, game)
        if not ready:
            return []
        out = []
        for ev in events:
            kind = ev["type"]
            if kind == "start":
                self.pending.clear()
                self.points_since_line = 0
                out.append(self._request(now, "intro", game, ttl=12.0))
            elif kind == "point" and not self._game_ends(ev):
                self.points_since_line += 1
                if self._worth_a_line(now, game):
                    out.append(self._request(now, "point", game, ttl=9.0))
            elif kind == "game_over":
                out.append(self._request(now, "game_over", game, ttl=15.0))
        return out

    def accept(self, result, now, game):
        """Worker result {"id", "text", "latency"} -> Sonia call event, or None if
        it's late, stale, malformed or repetitive."""
        req = self.pending.pop(result["id"], None)
        if req is None or now > req["expires"]:
            return None
        moved_on = self.story.rally >= self.STALE_RALLY_SHOTS and game.state == gl.RALLY
        if req["kind"] == "point" and (len(self.story.points) != req["points"] or moved_on):
            return None
        if req["kind"] == "intro" and moved_on:
            return None
        text = sanitize_line(result.get("text", ""))
        if not text or too_similar(text, self.recent):
            return None
        self.recent = (self.recent + [text])[-6:]
        self.last_spoke = now
        self.points_since_line = 0
        return {"type": "call", "speaker": "sonia", "text": text, "excite": req["excite"],
                "interrupt": False, "latency": round(result.get("latency", 0.0), 2)}

    # ------------------------------------------------------------ internals
    def _request(self, now, kind, game, ttl):
        self.next_id += 1
        sit = self.story.situation(kind, game)
        sit["focus"] = self.rng.choice([f for f in FOCUS[kind] if f != self.last_focus])
        self.last_focus = sit["focus"]
        if self.recent:
            sit["avoid_repeating"] = self.recent[-3:]
        req = {"id": self.next_id, "kind": kind, "messages": build_messages(sit),
               "expires": now + ttl, "points": len(self.story.points),
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
        if kind == "intro":
            return 0.5
        e = 0.35 + min(sit.get("rally_shots", 0), 16) / 40
        if "momentum" in sit or "comeback" in sit or "game point" in sit.get("score_situation", ""):
            e += 0.2
        return round(min(e, 0.9), 2)
