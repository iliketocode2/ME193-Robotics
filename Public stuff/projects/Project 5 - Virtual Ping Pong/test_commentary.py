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

check(all(c["speaker"] == "ray" for c in calls), "all play-by-play calls are tagged speaker=ray")
a = Announcer(rng=random.Random(0))
g = Game(); g.opp = gl.OPPONENTS["viktor"]
welcome = a.update(0.0, [{"type": "start", "opp": "viktor"}], g, handoff=True)[0]["text"]
check("Sonia" in welcome and "banana" not in welcome, f"with the AI booth ready, Ray hands over to Sonia: '{welcome}'")

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
        said += [p for c in a2.update(t2, ev, g2, handoff=seed % 2 == 0, sonia_cued=seed % 3 == 0) for p in c["parts"]]
    for _ in range(int(3 / DT)):
        t2 += DT
        said += [p for c in a2.update(t2, g2.update(t2, DT, Paddle(), []), g2) for p in c["parts"]]
missing = sorted({p for p in said if p not in phrases})
check(not missing, f"all {len(said)} sentences Ray said in 7 matches have a clip ({len(phrases)} pre-rendered); missing: {missing[:5]}")
check(all(c["text"] == " ".join(c["parts"]) for c in calls), "a call's text is its parts, in order")

end = {"type": "point", "winner": gl.PLAYER, "reason": "opp_miss", "score": {gl.PLAYER: 11, gl.OPP: 5}}
check(Announcer(rng=random.Random(0)).update(1.0, [end], g, sonia_cued=True)[0]["text"],
      "a game-ending point is never left blank")

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
reqs = b.update(0.0, [{"type": "start", "opp": "pip"}], g, ready=True)
check(len(reqs) == 1 and reqs[0]["kind"] == "intro", "match start -> intro request")
intro = b.accept({"id": reqs[0]["id"], "text": "Pip waddles out to a huge roar from the crowd tonight.", "latency": 1.8}, 2.0, g)
check(intro and intro["speaker"] == "sonia" and not intro["interrupt"], "a timely line becomes a Sonia call (no interrupt)")

g.score = {gl.PLAYER: 1, gl.OPP: 0}
r1 = b.update(3.0, point(gl.PLAYER, 1, 0, reason="opp_miss"), g, ready=True)
check(r1 == [], "quiet point right after Sonia spoke -> no request (cooldown)")
reqs = []
for i, t in enumerate((12.0, 13.0, 14.0)):
    g.score = {gl.PLAYER: 2 + i, gl.OPP: 0}
    reqs += b.update(t, point(gl.PLAYER, 2 + i, 0, reason="opp_miss"), g, ready=True)
check(len(reqs) == 1 and reqs[0]["kind"] == "point", "a run of points earns exactly one request")
g.score = {gl.PLAYER: 5, gl.OPP: 0}
b.update(15.0, point(gl.PLAYER, 5, 0, reason="opp_miss"), g, ready=True)
check(b.accept({"id": reqs[0]["id"], "text": "Pip simply cannot live with our player at the moment.", "latency": 2}, 15.5, g) is None,
      "a line about an older point is dropped once another point has happened")

b2 = Booth(rng=random.Random(2)); g2 = Game(); g2.set_highlight("rita"); g2.confirm()
req = b2.update(0.0, [{"type": "start", "opp": "rita"}], g2, ready=True)[0]
check(b2.accept({"id": req["id"], "text": "Rita looks absolutely ready for business today.", "latency": 20}, 20.0, g2) is None,
      "a line that arrives after its deadline is dropped")
req = b2.update(30.0, [{"type": "start", "opp": "rita"}], g2, ready=True)[0]
b2.story.update([{"type": "opp_hit", "serve": True}, {"type": "hit", "streak": 1},
                 {"type": "opp_hit"}, {"type": "hit", "streak": 2}], g2); g2.state = gl.RALLY
check(b2.accept({"id": req["id"], "text": "Rita looks absolutely ready for business today.", "latency": 1}, 31.0, g2) is None,
      "the intro is dropped once the first rally is under way")

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
