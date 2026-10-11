"""
Self-check for commentary.py (the announcer) -- no hardware or browser needed.

Run:
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_commentary.py"
"""

import random

import game_logic as gl
import commentary
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
check(not any("Welcome" in c["text"] for c in calls), "the Announcer leaves the welcome to the pre-match show")
points = [e for e in all_events if e["type"] == "point"]
point_calls = [c for c in calls if c["interrupt"]]
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

check(all(c["speaker"] == "ray" for c in calls), "all play-by-play calls are tagged speaker=ray")
a = Announcer(rng=random.Random(0))
g = Game(); g.opp = gl.OPPONENTS["viktor"]
wc = a.update(0.0, [{"type": "point", "winner": gl.OPP, "reason": "net", "score": {gl.PLAYER: 1, gl.OPP: 2}},
                    {"type": "weather_change", "key": "rain_start"}], g)
check(len(wc) == 2 and wc[0]["interrupt"] and wc[1]["text"] == "And here comes the rain." and not wc[1]["interrupt"],
      f"a weather change is called after the point call, queued: {texts(wc)}")

# --- when Sonia is cued for a point, Ray just calls the score (her line isn't stuck behind his) ---
a = Announcer(rng=random.Random(0))
g = Game(); g.opp = gl.OPPONENTS["pip"]
pt = [{"type": "miss", "reason": "no_swing"},
      {"type": "point", "winner": gl.OPP, "reason": "miss", "score": {gl.PLAYER: 2, gl.OPP: 4}}]
full = a.update(1.0, pt, g)[0]["text"]
brief = Announcer(rng=random.Random(0)).update(1.0, pt, g, sonia_cued=True)[0]["text"]
check(brief == "Four, two, to Pip.", f"Sonia cued: Ray calls only the score: '{brief}'")
check(full.endswith("Four, two, to Pip.") and len(full) > len(brief), f"score calls use the short name: '{full}'")

# --- every sentence Ray says has a pre-rendered natural-voice clip (tts.py --setup) ---
from commentary import ray_phrases  # noqa: E402
phrases = ray_phrases()
said = [p for c in calls for p in c["parts"]]
for seed in range(6):                      # more matches: every opponent, AI booth on and off
    g2, a2 = Game(rng=random.Random(seed)), Announcer(rng=random.Random(seed))
    g2.set_highlight(["pip", "rita", "viktor"][seed % 3]); g2.confirm()
    t2 = 0.0
    while g2.state != gl.GAME_OVER and t2 < 600:
        t2 += DT
        b = g2.ball
        swing = (g2.state == gl.SERVE and g2.server == gl.PLAYER) or \
                (g2.state == gl.RALLY and b.hitter == gl.OPP and 1.3 < b.z < 1.55 and random.random() < 0.5)
        ev = g2.update(t2, DT, Paddle(b.x, b.y, 0, True), [Swing(t2, 1.3 + seed / 5)] if swing else [])
        said += [p for c in a2.update(t2, ev, g2, sonia_cued=seed % 3 == 0) for p in c["parts"]]
    for _ in range(int(3 / DT)):
        t2 += DT
        said += [p for c in a2.update(t2, g2.update(t2, DT, Paddle(), []), g2) for p in c["parts"]]
missing = sorted({p for p in said if p not in phrases})
check(not missing, f"all {len(said)} sentences Ray said in 7 matches have a clip ({len(phrases)} pre-rendered); missing: {missing[:5]}")
check(all(c["text"] == " ".join(c["parts"]) for c in calls), "a call's text is its parts, in order")

end = {"type": "point", "winner": gl.PLAYER, "reason": "opp_miss", "score": {gl.PLAYER: 11, gl.OPP: 5}}
check(Announcer(rng=random.Random(0)).update(1.0, [end], g, sonia_cued=True)[0]["text"],
      "a game-ending point is never left blank")

# =================================================================== the pre-match show
import weather  # noqa: E402
from commentary import Booth, ShowDirector, number_words, weather_calls  # noqa: E402

check([number_words(n) for n in (-20, 0, 7, 48, 100, 105, 110)] ==
      ["minus twenty", "zero", "seven", "forty-eight", "one hundred", "one hundred and five", "one hundred and ten"],
      "temperatures read as words")

# every weather sentence Ray could say has a clip: all WMO codes x times of day x temps x winds
wsaid = set()
for code in weather.WMO:
    for hour in (3, 9, 15, 19, 23):
        for temp in range(-25, 116, 7):
            for mph, deg in ((0, 0), (8, 45), (15, 200), (25, 290), (40, 135)):
                p, now_w = weather.demo_payload("clear")
                p = dict(p, code=code, temp_f=temp, feels_f=temp - (8 if mph > 10 else 0), wind_mph=mph,
                         wind_dir=deg, obs=p["obs"][:11] + f"{hour:02d}:00", is_day=int(7 <= hour < 18))
                b = weather.bucket(p, weather._epoch(p["obs"], p["utc_offset_s"]))
                for parts in weather_calls(b):
                    wsaid.update(parts)
missing = sorted(x for x in wsaid if x not in phrases)
check(not missing and len(wsaid) > 50, f"all {len(wsaid)} weather sentences have a clip; missing: {missing[:4]}")


class FakeBooth:
    """Booth stand-in for the director: show lines that are (or become) ready."""
    def __init__(self, ready=()):
        self.ready = {k: {"type": "call", "speaker": "sonia", "text": f"Sonia on {k[1]}.", "secs": 2.0,
                          "kind": "show", "interrupt": False, "excite": 0.5} for k in ready}

    def has_show(self, opp, slot):
        return (opp, slot) in self.ready

    def take_show(self, opp, slot):
        return self.ready.pop((opp, slot), None)


def run_show(booth, sonia_ready=True, muted=False, bucket=None, skip_at=None, rematch=False, add_at=None, game=None):
    """Drive game + director through a show; returns (calls with their times, end time, show_end event)."""
    g = game or Game(rng=random.Random(5))
    if rematch:
        g.state, g.state_t, g.now = gl.GAME_OVER, -20.0, 0.0
        g.rematch()
    else:
        g.set_highlight("rita"); g.confirm()
    d = ShowDirector(rng=random.Random(5))
    out, t, end = [], 0.0, None
    while t < 20:
        if add_at and t >= add_at[0]:
            booth.ready[add_at[1]] = FakeBooth([add_at[1]]).ready[add_at[1]]; add_at = None
        if skip_at is not None and t >= skip_at:
            g.confirm(); skip_at = None
        ev = g.update(t, DT, Paddle())
        sonia = sonia_ready if isinstance(sonia_ready, str) else "ready" if sonia_ready else "loading"
        out += [(round(t, 2), c) for c in d.update(t, ev, g, booth, bucket, sonia, muted)]
        end = next((e for e in ev if e["type"] == "show_end"), end)
        if g.state != gl.SHOW:
            break
        t += DT
    ev = g.update(t + DT, DT, Paddle())            # events raised by the director's end_show() arrive next tick
    end = next((e for e in ev if e["type"] == "show_end"), end)
    return out, t, end, g


wb = {"sky": "rain", "tod": "evening", "temp": 52, "feels": "colder", "wind_level": 2, "wind_from": "west"}
calls_t, t_end, end, _ = run_show(FakeBooth([("rita", "opp"), ("rita", "player")]), bucket=wb)
seq = [(c["speaker"], c["text"]) for _, c in calls_t]
check(seq[0][0] == "ray" and "welcome" in seq[0][1].lower() and "Coach Rita" in seq[0][1] and calls_t[0][1]["interrupt"],
      "show opens with Ray's welcome (interrupting anything before it)")
check(seq[1][1].startswith("Live from Tufts, on a rainy evening. Fifty-two degrees."), f"Ray reports the weather: '{seq[1][1]}'")
check([s for s, _ in seq][:4] == ["ray", "ray", "sonia", "sonia"],
      f"script: welcome, weather, Sonia on the opponent, Sonia on our player ({[s for s, _ in seq]})")
check(seq[1][1].endswith("Sonia?"), f"Ray hands over to Sonia when her line is ready: '{seq[1][1]}'")
check(t_end <= gl.SHOW_MAX_S and end == {"type": "show_end", "skipped": False}, f"show ends by itself when the script is done ({t_end:.1f} s)")
check(all(c["kind"] == "show" for _, c in calls_t), "every show call is kind 'show'")
times = [t for t, _ in calls_t]
check(all(b > a for a, b in zip(times, times[1:])), "lines are released one by one, not all at once")

calls_t, _, _, _ = run_show(FakeBooth(), bucket=wb)
seq = [(c["speaker"], c["text"]) for _, c in calls_t]
check(not any(s == "sonia" for s, _ in seq) and any(c == commentary.INTROS["rita"] for _, c in seq)
      and not any("Sonia?" in c for _, c in seq), "Sonia's line not ready in time: Ray covers the opponent, no handoff")

calls_t, _, _, _ = run_show(FakeBooth(), bucket=wb, add_at=(7.0, ("rita", "opp")))
check(any(c["speaker"] == "sonia" for _, c in calls_t), "Sonia's line that lands just in time still airs")

calls_t, _, _, _ = run_show(FakeBooth(), sonia_ready=False, bucket=wb)
check(any(c["text"] == commentary.SHOW_LATE for _, c in calls_t), "Sonia still warming up: Ray says so")
calls_t, _, _, _ = run_show(FakeBooth(), sonia_ready="off", bucket=wb)
check(not any(c["text"] == commentary.SHOW_LATE for _, c in calls_t) and not any(c["speaker"] == "sonia" for _, c in calls_t),
      "no AI at all (--no-ai): Ray runs the show alone without mentioning her")

calls_t, t_end, _, g_m = run_show(FakeBooth([("rita", "opp")]), muted=True, bucket=wb)
check(t_end <= 6.1 and not any(c["speaker"] == "sonia" for _, c in calls_t), "muted: a short show, no Sonia")

calls_t, t_end, end, _ = run_show(FakeBooth([("rita", "opp")]), bucket=wb, skip_at=2.0)
check(end == {"type": "show_end", "skipped": True} and t_end < 2.1 and all(t < 2.1 for t, _ in calls_t),
      "skip: the show stops at once, nothing more is said")

calls_t, t_end, _, _ = run_show(FakeBooth(), bucket=None)
check(not any("Live from" in c["text"] for _, c in calls_t) and len(calls_t) >= 3 and t_end <= gl.SHOW_MAX_S + DT,
      f"no weather: the show goes on without it ({len(calls_t)} lines, {t_end:.1f} s)")

gr = Game(rng=random.Random(5)); gr.opp = gl.OPPONENTS["rita"]; gr.history = [{"opp": "rita", "winner": gl.OPP, "you": 4, "them": 11, "best_streak": 2}]
calls_t, t_end, _, _ = run_show(FakeBooth([("rita", "rematch")]), bucket=wb, rematch=True, game=gr)
seq = [(c["speaker"], c["text"]) for _, c in calls_t]
check(seq[0][1].startswith(("Rematch!", "It's a rematch")) and any(s == "sonia" for s, _ in seq)
      and t_end <= gl.SHOW_REMATCH_S + DT,
      f"rematch: a short show with its own opener and Sonia ({t_end:.1f} s)")

all_show = []
for kw in ({"bucket": wb}, {"bucket": None, "sonia_ready": False}, {"bucket": wb, "muted": True}):
    all_show += [p for _, c in run_show(FakeBooth(), **kw)[0] if c["speaker"] == "ray" for p in c["parts"]]
gp = Game(rng=random.Random(5)); gp.set_highlight("pip"); gp.confirm()
gp.record = 12.0; gp.history = [{"opp": "pip", "winner": gl.PLAYER, "you": 11, "them": 3, "best_streak": 12}] * 4
all_show += commentary.player_parts(gp)
missing = sorted({p for p in all_show if p not in phrases})
check(not missing, f"every show sentence Ray says has a clip; missing: {missing[:4]}")
check(commentary.player_parts(gp) == ["You won the last meeting."], f"player segment: the last meeting ({commentary.player_parts(gp)})")
gp.history = [{"opp": "rita", "winner": gl.PLAYER, "you": 11, "them": 3, "best_streak": 2}] * 2; gp.record = 2
check(commentary.player_parts(gp) == ["This is match number three for our player today."],
      f"player segment: else which match of the day it is ({commentary.player_parts(gp)})")

# =================================================================== Sonia
from commentary import Booth, MatchStory, build_messages, sanitize_line, too_similar  # noqa: E402

# --- sanitizer ------------------------------------------------------------------
check(sanitize_line("<think>\n</think>\nRita’s got that edge — always pushing wide.") ==
      "Rita's got that edge, always pushing wide.", "sanitizer: strips <think>, fixes quotes and dashes")
check(sanitize_line('Sonia: "What a rally that was, simply superb!" And more.') ==
      "What a rally that was, simply superb!", "sanitizer: drops speaker label, quotes, extra sentences")
check(sanitize_line("Our player trails 4-7 now, oh dear.") is None, "sanitizer: drops lines that call the score")
check(sanitize_line("Too short.") is None, "sanitizer: drops fragments")
check(sanitize_line("A fightback \U0001F525 for the ages from our player.") == "A fightback for the ages from our player.",
      "sanitizer: strips emojis")
long = sanitize_line(" ".join(["word"] * 24))
check(long is not None and len(long.split()) == 20, "sanitizer: trims slightly-long lines to 20 words")
check(sanitize_line("Vikor has our player rooted to the spot again.") == "Viktor has our player rooted to the spot again.",
      "sanitizer: fixes misspelled opponent names")
check(too_similar("Rita keeps pulling our player wide again", ["Rita keeps pulling our player wide"]),
      "near-repeats are detected")
check(too_similar("Pip's lobs are a lullaby tonight.", ["Pip's lobs still loop like a broken record."]),
      "lines that open the same way count as repeats")
check(not too_similar("The crowd is loving every second of this.", ["Pip's lobs still loop like a broken record."]),
      "genuinely different lines pass")
check(too_similar("Pip's face lights up at that one.", ["The crowd is up.", "Pip's daffy lob hits the net."]),
      "two lines in a row can't open with the same word")

# --- story ----------------------------------------------------------------------
def point(winner, you, them, reason="miss", miss=None):
    evs = [{"type": "miss", "reason": miss}] if miss else []
    return evs + [{"type": "point", "winner": winner, "reason": reason, "score": {gl.PLAYER: you, gl.OPP: them}}]

g = Game(); g.set_highlight("rita"); g.confirm()
st = MatchStory()
st.update([{"type": "start", "opp": "rita"}], g)
for i in range(3):
    g.score = {gl.PLAYER: 0, gl.OPP: i + 1}
    st.update([{"type": "opp_hit", "serve": True}, {"type": "hit", "streak": 1}] + point(gl.OPP, 0, i + 1, miss="wrong_place"), g)
sit = st.situation("point", g)
check(sit["momentum"].endswith("won 3 in a row") and "out of position 3 times" in sit["pattern"],
      f"story notices runs and repeated mistakes: {sit.get('momentum')} / {sit.get('pattern')}")
check(sit["how"] == "our player was out of position" and sit["rally_shots"] == 2, "story describes how the point ended")
check(not any(ch.isdigit() for ch in sit["score_situation"].replace("leads by 3", "")), "score is given as a situation, not numbers")
msgs = build_messages(sit)
check(msgs[0]["role"] == "system" and msgs[-1]["role"] == "user" and '"moment":"point"' in msgs[-1]["content"],
      "prompt = fixed system + examples, situation last")
check(build_messages({"moment": "intro"})[:-1] == build_messages({"moment": "x"})[:-1],
      "prompt prefix is identical every time (lets the runtime reuse work)")

# --- booth timing ---------------------------------------------------------------
b = Booth(rng=random.Random(1))
g = Game(); g.set_highlight("pip"); g.confirm()
check(b.update(0.0, [{"type": "start", "opp": "pip"}], g, ready=False) == [], "no AI requests while the worker isn't ready")
b = Booth(rng=random.Random(1))
check(b.update(0.0, [{"type": "start", "opp": "pip"}], g, ready=True) == [],
      "match start asks for nothing (the show's lines were written ahead)")
b.last_spoke = 0.0

g.score = {gl.PLAYER: 1, gl.OPP: 0}
r1 = b.update(3.0, point(gl.PLAYER, 1, 0, reason="opp_miss"), g, ready=True)
check(r1 == [], "quiet point right after Sonia spoke -> no request (cooldown)")
reqs = []
for i, t in enumerate((12.0, 13.0, 14.0)):
    g.score = {gl.PLAYER: 2 + i, gl.OPP: 0}
    reqs += b.update(t, point(gl.PLAYER, 2 + i, 0, reason="opp_miss"), g, ready=True)
check(len(reqs) == 1 and reqs[0]["kind"] == "point" and reqs[0]["prio"] == 1, "a run of points earns exactly one (urgent) request")
req = b._request(14.5, "point", g, ttl=b.POINT_TTL_S)            # asked about the latest point
line = {"id": req["id"], "text": "Pip simply cannot live with our player at the moment.", "latency": 3}
# the next rally is well under way when her line is ready: TV analysts talk over play
b.story.update([{"type": "opp_hit", "serve": True}] + [{"type": "hit", "streak": 1}, {"type": "opp_hit"}] * 4, g)
g.state = gl.RALLY
call = b.accept(dict(line), 18.0, g, hold=True)
check(call is not None, "a point line is still fresh 8 shots into the next rally")
check(b.release(line["id"], 21.0, g, ["tts/bf_emma/x.wav"], 3.1) == dict(call, clips=["tts/bf_emma/x.wav"], secs=3.1),
      "...and its voice clip (with its length) is attached on release")
reqs2 = []
for i, t in enumerate((40.0, 41.0, 42.0)):
    g.score = {gl.PLAYER: 5 + i, gl.OPP: 0}
    reqs2 += b.update(t, point(gl.PLAYER, 5 + i, 0, reason="opp_miss"), g, ready=True)
b.accept({"id": reqs2[0]["id"], "text": "Our player is absolutely flying through this one now.", "latency": 2}, 42.5, g, hold=True)
g.score = {gl.PLAYER: 9, gl.OPP: 0}
b.update(44.0, point(gl.PLAYER, 9, 0, reason="opp_miss"), g, ready=True)
check(b.release(reqs2[0]["id"], 44.5, g) is None, "...but once the NEXT point has ended, it's dropped")

b2 = Booth(rng=random.Random(2)); g2 = Game(); g2.set_highlight("rita"); g2.confirm()
b2.story.update(point(gl.OPP, 0, 1, reason="net"), g2)
req = b2._request(0.0, "point", g2, ttl=b2.POINT_TTL_S)
check(b2.accept({"id": req["id"], "text": "Rita looks absolutely ready for business today.", "latency": 40}, 40.0, g2) is None,
      "a line that arrives after its deadline is dropped")

# --- pre-match show lines, written ahead -------------------------------------------
b4 = Booth(rng=random.Random(4)); b4.conditions = "light rain, chilly, breezy, evening"
g4 = Game(); g4.set_highlight("pip")
r = b4.prepare_show(0.0, "pip", g4)
check([(x["opp"], x["slot"], x["prio"]) for x in r] == [("pip", "opp", 2), ("pip", "player", 2)],
      "picking an opponent asks for both show lines (speculative)")
check('"conditions":"light rain, chilly, breezy, evening"' in r[0]["messages"][-1]["content"]
      and '"our_player":"first match today"' in r[0]["messages"][-1]["content"], "show prompts carry the weather and the player")
check(b4.prepare_show(1.0, "pip", g4) == [], "...asked once, not every tick")
r_rita = b4.prepare_show(2.0, "rita", g4)
check(len(r_rita) == 2 and len(b4.pending) == 4, "switching tags asks for the new opponent too (Pip's are kept)")
check(sorted(b4.commit_show("pip")) == sorted(x["id"] for x in r_rita), "ENTER vs Pip drops Rita's pending lines")
check(b4.accept({"id": r[0]["id"], "text": "A penguin in the rain will feel right at home out there.", "latency": 2},
                5.0, g4) is None and b4.has_show("pip", "opp"), "a show line goes into the show cache, not on air")
check(b4.accept({"id": r[1]["id"], "text": "Our player needs quick feet and a calm head today.", "latency": 2},
                6.0, g4, hold=True) is not None and not b4.has_show("pip", "player"), "a show line held for its voice...")
check(b4.release(r[1]["id"], 9.0, g4, ["tts/bf_emma/y.wav"], 2.5) is None and b4.has_show("pip", "player"),
      "...is cached once voiced")
got = b4.take_show("pip", "player")
check(got["clips"] == ["tts/bf_emma/y.wav"] and got["secs"] == 2.5 and not b4.has_show("pip", "player"),
      "take_show hands it over once (used up)")
g4.confirm(); g4.update(0.0, DT, Paddle())
b4.update(10.0, [{"type": "start", "opp": "pip"}], g4, ready=True)
check(b4.has_show("pip", "opp"), "the match starting keeps the cached show lines")
b5 = Booth(rng=random.Random(5)); g5 = Game(); g5.set_highlight("viktor")
r5 = b5.prepare_show(0.0, "viktor", g5)
g5.confirm(); g5.update(0.0, DT, Paddle())
g5.opp = gl.OPPONENTS["pip"]
check(b5.accept({"id": r5[0]["id"], "text": "Viktor bends every ball like a banana, so stay sharp.", "latency": 2}, 3.0, g5) is None
      and not b5.has_show("viktor", "opp"), "a show line for a different opponent's match is dropped")
check(len(b5.prepare_show(100.0, "viktor", Game())) >= 1, "a show request that never got an answer is asked again later")
b6 = Booth(rng=random.Random(6)); b6.conditions = "heavy rain, cold, windy, night"; b6.weather_notable = True
g6 = Game(); g6.set_highlight("rita"); g6.confirm()
b6.story.update(point(gl.OPP, 0, 1, reason="net"), g6)
foci = []
for i in range(30):
    req = b6._request(float(i), "point", g6, ttl=5)
    foci.append('"focus":"the weather conditions"' in req["messages"][-1]["content"])
    if foci[-1]:
        check('"conditions":"heavy rain, cold, windy, night"' in req["messages"][-1]["content"],
              "a weather-focused point line gets the conditions")
        break
check(any(foci), "in notable weather, Sonia sometimes talks about it mid-match")
b6.weather_notable = False
check(not any('"conditions"' in b6._request(50.0 + i, "point", g6, ttl=5)["messages"][-1]["content"] for i in range(30)),
      "in ordinary weather, she doesn't")

g3 = Game(); g3.set_highlight("viktor"); g3.confirm()
b3 = Booth(rng=random.Random(3)); b3.update(0, [{"type": "start", "opp": "viktor"}], g3, ready=True)
b3.last_spoke = -1e9
g3.score = {gl.PLAYER: 11, gl.OPP: 9}
check(b3.update(50.0, point(gl.PLAYER, 11, 9, reason="opp_miss"), g3, ready=True) == [],
      "no point line for the game-winning point (the outro covers it)")
g3.winner = gl.PLAYER
out = b3.update(52.0, [{"type": "game_over", "winner": gl.PLAYER, "score": g3.score}], g3, ready=True)
check(len(out) == 1 and out[0]["kind"] == "game_over" and '"winner":"player"' in out[0]["messages"][-1]["content"],
      "game over -> outro request with the result")
first = b3.accept({"id": out[0]["id"], "text": "Viktor has finally met his match, and what a match it was.", "latency": 2}, 53.0, g3)
again = b3.update(54.0, [{"type": "game_over", "winner": gl.PLAYER, "score": g3.score}], g3, ready=True)[0]
dup = b3.accept({"id": again["id"], "text": "Viktor has finally met his match, and what a match that was!", "latency": 2}, 55.0, g3)
check(first is not None and dup is None, "Sonia never repeats herself")
check(len({r["seed"] for r in reqs + out + [again]}) == len(reqs + out + [again]), "every request gets its own random seed")
foci = [b3._request(60.0 + i, "point", g3, ttl=5)["messages"][-1]["content"] for i in range(6)]
foci = [f.split('"focus":"')[1].split('"')[0] for f in foci]
check(all(a != b for a, b in zip(foci, foci[1:])) and len(set(foci)) >= 3,
      f"each request gets a fresh focus, never the same twice running ({foci[:3]}...)")

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED")
raise SystemExit(1 if failures else 0)
