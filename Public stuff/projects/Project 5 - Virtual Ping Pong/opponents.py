"""
The three opponents you can pick with an AprilTag.

Gameplay numbers live here (Python is the authoritative game). The 3D look
of each character lives in web/game.js under the same `key`.

  tag_id     -- tag36h11 ID that selects this opponent (print tags/ folder)
  speed      -- horizontal ball speed (m/s) of THEIR shots; your returns are
                scaled to match, so a harder opponent means a faster rally
  reach      -- how fast (m/s) they can slide sideways to meet the ball
  miss_pct   -- base chance they fail to return a shot (raised by wide /
                fast shots from you)
  aim_x      -- how far from center line (m) they aim; table half-width is 0.76
  spin       -- sideways acceleration (m/s^2) on their shots -- curveballs
  lines      -- what they say on each game event (picked at random)
"""

OPPONENTS = {
    "pip": {
        "key": "pip",
        "tag_id": 0,
        "name": "Pip the Penguin",
        "level": "EASY",
        "stars": 1,
        "tagline": "Goofy, cheerful, loves a loopy lob.",
        "speed": 3.0,
        "reach": 1.4,
        "miss_pct": 0.22,
        "aim_x": 0.25,
        "spin": 0.0,
        "lines": {
            "intro": ["Waddle waddle! Let's play!", "Ooh, a new friend!"],
            "serve": ["Here it comes~", "Wheee!"],
            "you_hit": ["Nice one!", "Wow, you're good!", "Hehe, fun!"],
            "you_miss": ["Oopsie! You'll get the next one!", "Don't worry!"],
            "opp_miss": ["Oh no, my flippers!", "Whoops!", "Too fast for me!"],
            "streak": ["{n} in a row?! Amazing!", "Keep it going, {n}!"],
            "win": ["I won?! Hooray! Good game!"],
            "lose": ["You won! Fish snacks on me!"],
        },
    },
    "rita": {
        "key": "rita",
        "tag_id": 1,
        "name": "Coach Rita",
        "level": "MEDIUM",
        "stars": 2,
        "tagline": "Competitive coach. Aims for the corners.",
        "speed": 3.9,
        "reach": 2.0,
        "miss_pct": 0.10,
        "aim_x": 0.55,
        "spin": 0.0,
        "lines": {
            "intro": ["Warm up's over. Let's go!", "Show me what you've got."],
            "serve": ["Ready position!", "Eyes on the ball!"],
            "you_hit": ["Good footwork!", "Follow through!", "That's it!"],
            "you_miss": ["Watch the corners!", "Faster feet!", "Swing earlier!"],
            "opp_miss": ["Hah, nice shot.", "Okay, you earned that."],
            "streak": ["{n} straight! Now we're training!"],
            "win": ["Good effort. Back to drills!"],
            "lose": ["Well played. I'm proud of you."],
        },
    },
    "viktor": {
        "key": "viktor",
        "tag_id": 2,
        "name": "Viktor the Wall",
        "level": "HARD",
        "stars": 3,
        "tagline": "Silent. Fast. Every shot curves.",
        "speed": 4.9,
        "reach": 2.8,
        "miss_pct": 0.03,
        "aim_x": 0.62,
        "spin": 1.6,
        "lines": {
            "intro": ["...", "Begin."],
            "serve": ["...", "Hm."],
            "you_hit": ["Hm.", "...Acceptable."],
            "you_miss": ["Predictable.", "Too slow.", "..."],
            "opp_miss": ["...Impossible.", "Error logged."],
            "streak": ["{n}. Interesting."],
            "win": ["As calculated."],
            "lose": ["...Recalibrating. Well played, human."],
        },
    },
}

TAG_TO_OPPONENT = {o["tag_id"]: k for k, o in OPPONENTS.items()}


def public_info():
    """The subset of opponent data the browser needs for the select screen."""
    return [
        {k: o[k] for k in ("key", "tag_id", "name", "level", "stars", "tagline")}
        for o in OPPONENTS.values()
    ]
