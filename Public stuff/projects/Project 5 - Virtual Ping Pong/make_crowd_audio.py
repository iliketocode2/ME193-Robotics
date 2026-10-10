"""
Builds the crowd sound bank in web/audio/crowd/ from freely licensed real
recordings (Wikimedia Commons, OpenGameArt). Dev-time tool: run once; the
small .ogg outputs, manifest.json and web/audio/CREDITS.md are committed.

    my_env/Scripts/pip install imageio-ffmpeg        # bundles ffmpeg
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/make_crowd_audio.py"

Downloads go to <repo>/assets_cache/crowd/ (outside "Public stuff/", so not
tracked). Each clip is trimmed, mixed to mono, loudness-matched, faded (or
crossfaded into a seamless loop for the ambience beds) and encoded as Ogg
Vorbis. Trim points were chosen from per-second loudness and rhythm analysis
of each recording (the chant sections are the strongly periodic ones).
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import zipfile

import imageio_ffmpeg
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "..", "..", "assets_cache", "crowd")
OUT = os.path.join(HERE, "web", "audio", "crowd")
CREDITS = os.path.join(HERE, "web", "audio", "CREDITS.md")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SR = 44100
UA = "RogersCupPingPong/1.0 (Tufts ME193 class project)"   # Wikimedia asks for a descriptive User-Agent

QUENDEL_ZIP = "https://opengameart.org/sites/default/files/gregor_quendel_-_free_crowd_cheering_sounds_-_mp3.zip"
QUENDEL_DIR = "Gregor Quendel - Free Crowd Cheering Sounds - MP3"
QUENDEL = {
    "author": "Gregor Quendel", "license": "CC BY 4.0",
    "page": "https://opengameart.org/content/free-crowd-cheering-sounds",
}


def wiki(file_url, page, author, license_):
    return {"url": file_url, "page": page, "author": author, "license": license_}


# source key -> where it comes from (+ licence for CREDITS.md)
SOURCES = {
    "wws_stadium": wiki("https://upload.wikimedia.org/wikipedia/commons/e/ec/WWS_FootballAustriavs.Sweden.ogg",
                        "https://commons.wikimedia.org/wiki/File:WWS_FootballAustriavs.Sweden.ogg",
                        "Work With Sounds / Torsten Nilsson (Ernst Happel Stadium, Vienna)", "CC BY 4.0"),
    "walla": wiki("https://upload.wikimedia.org/wikipedia/commons/a/a9/360703_eguobyte_large-crowd-medium-distance-stereo.wav",
                  "https://commons.wikimedia.org/wiki/File:360703_eguobyte_large-crowd-medium-distance-stereo.wav",
                  "eguobyte", "CC0"),
    "millwall": wiki("https://upload.wikimedia.org/wikipedia/commons/0/07/Noonelikesus.ogg",
                     "https://commons.wikimedia.org/wiki/File:Noonelikesus.ogg", "NoOneLikesUs", "Public domain"),
    "strasbourg": wiki("https://upload.wikimedia.org/wikipedia/commons/c/ce/Score_Strasbourg-PSG_%28RCSA-Paris_Saint_Germain%29_Racing_2_-_Paris_0_MERCI_-_DE_RIEN.ogg",
                       "https://commons.wikimedia.org/wiki/File:Score_Strasbourg-PSG_(RCSA-Paris_Saint_Germain)_Racing_2_-_Paris_0_MERCI_-_DE_RIEN.ogg",
                       "X22c23a", "CC BY-SA 4.0"),
    "clapping_hurray": wiki("https://upload.wikimedia.org/wikipedia/commons/a/a8/Clapping_hurray.ogg",
                            "https://commons.wikimedia.org/wiki/File:Clapping_hurray.ogg", "starlite", "Public domain"),
    "hurray": wiki("https://upload.wikimedia.org/wikipedia/commons/e/e8/Hurray.ogg",
                   "https://commons.wikimedia.org/wiki/File:Hurray.ogg", "starlite", "Public domain"),
    "ohhh_ahhh": wiki("https://upload.wikimedia.org/wikipedia/commons/0/0f/Ohhh_ahhh.ogg",
                      "https://commons.wikimedia.org/wiki/File:Ohhh_ahhh.ogg", "starlite", "Public domain"),
    "applause_i": wiki("https://upload.wikimedia.org/wikipedia/commons/5/5b/Applause_i.ogg",
                       "https://commons.wikimedia.org/wiki/File:Applause_i.ogg", "thore", "Public domain"),
    "applause_ii": wiki("https://upload.wikimedia.org/wikipedia/commons/0/09/Applause_ii.ogg",
                        "https://commons.wikimedia.org/wiki/File:Applause_ii.ogg", "thore", "Public domain"),
    "applause_concert": wiki("https://upload.wikimedia.org/wikipedia/commons/3/32/Sound_Effects_-_Applause_after_a_concert.ogg",
                             "https://commons.wikimedia.org/wiki/File:Sound_Effects_-_Applause_after_a_concert.ogg",
                             "Amada44", "CC0"),
    "oooo": {"url": "https://opengameart.org/sites/default/files/oooooooooo.ogg",
             "page": "https://opengameart.org/content/oooooooooooooo", "author": "Nocturnal_Vanguard", "license": "CC0"},
    "q_strong_rhythmic": {**QUENDEL, "zip": f"{QUENDEL_DIR}/Gregor Quendel - Crowd Cheering Sounds - 01 - Strong cheering and strong rhythmic cheering.mp3"},
    "q_strong_1": {**QUENDEL, "zip": f"{QUENDEL_DIR}/Gregor Quendel - Crowd Cheering Sounds - 03 - Strong cheering - I.mp3"},
    "q_strong_short": {**QUENDEL, "zip": f"{QUENDEL_DIR}/Gregor Quendel - Crowd Cheering Sounds - 04 - Strong cheering - II - Short.mp3"},
    "q_soft_1": {**QUENDEL, "zip": f"{QUENDEL_DIR}/Gregor Quendel - Crowd Cheering Sounds - 05 - Soft cheering - I.mp3"},
    "q_soft_2": {**QUENDEL, "zip": f"{QUENDEL_DIR}/Gregor Quendel - Crowd Cheering Sounds - 06 - Soft cheering - II.mp3"},
    "q_rhythmic": {**QUENDEL, "zip": f"{QUENDEL_DIR}/Gregor Quendel - Crowd Cheering Sounds - 08 - Rhythmic cheering.mp3"},
    "q_ambience": {**QUENDEL, "zip": f"{QUENDEL_DIR}/Gregor Quendel - Crowd Cheering Sounds - 10 - Ambience.mp3"},
}

# output clip -> (source, start s, duration s, role, target loudness dBFS RMS, loop?)
# Roles map to how audio.js uses them: bed (looping ambience), chant, roar,
# cheer, cheer_small, applause_s/m/l, ooh.
CLIPS = {
    "bed_stadium":   ("wws_stadium",       10.0, 34.0, "bed",         -24, True),   # steady stadium murmur
    "bed_walla":     ("walla",              2.0, 30.0, "bed",         -26, True),   # big crowd chatter
    "bed_buzz":      ("q_ambience",         8.0, 22.0, "bed_buzz",    -22, True),   # louder expectant crowd
    "chant_vienna":  ("wws_stadium",       46.0, 18.0, "chant",       -18, False),  # rhythmic stadium chant/clap
    "chant_clap":    ("q_rhythmic",         3.0, 13.0, "chant",       -18, False),  # rhythmic cheering
    "chant_terrace": ("millwall",          12.0, 23.0, "chant",       -19, False),  # two verses of terrace singing
    "chant_merci":   ("strasbourg",         0.0, 20.0, "chant",       -19, False),  # call-and-response goal chant
    "roar_1":        ("q_strong_short",     0.5, 12.5, "roar",        -15, False),
    "roar_2":        ("q_strong_1",         2.0, 16.0, "roar",        -15, False),
    "roar_rhythmic": ("q_strong_rhythmic",  3.0, 20.0, "roar",        -15, False),
    "cheer_1":       ("clapping_hurray",    0.0, 11.0, "cheer",       -17, False),
    "cheer_2":       ("q_soft_2",           1.0, 12.0, "cheer",       -17, False),
    "cheer_3":       ("q_soft_1",           6.0, 12.0, "cheer",       -17, False),
    "cheer_small":   ("hurray",             0.0, 3.6,  "cheer_small", -19, False),
    "applause_s":    ("applause_ii",        0.8, 8.5,  "applause_s",  -21, False),
    "applause_m":    ("applause_i",         0.8, 9.0,  "applause_m",  -20, False),
    "applause_l":    ("applause_concert",   0.0, 14.0, "applause_l",  -19, False),
    "ooh_1":         ("ohhh_ahhh",          0.0, 4.5,  "ooh",         -19, False),
    "ooh_2":         ("oooo",               0.0, 2.4,  "ooh",         -19, False),
}

FADE_S = 0.03
TAIL_FADE_S = 1.2        # one-shots fade out over their last stretch so cuts never sound chopped
LOOP_XFADE_S = 2.0
PEAK_LIMIT = 0.9


def fetch(src):
    os.makedirs(CACHE, exist_ok=True)
    if "zip" in src:
        zpath = os.path.join(CACHE, "quendel_cheering.zip")
        if not os.path.exists(zpath):
            download(QUENDEL_ZIP, zpath)
        out = os.path.join(CACHE, "quendel", src["zip"])
        if not os.path.exists(out):
            with zipfile.ZipFile(zpath) as z:
                z.extract(src["zip"], os.path.join(CACHE, "quendel"))
        return out
    path = os.path.join(CACHE, os.path.basename(src["url"]).split("?")[0])
    if not os.path.exists(path):
        download(src["url"], path)
    return path


def download(url, path, tries=5):
    print("  downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:     # 429 = rate limited: wait and retry
                raise
            wait = int(e.headers.get("Retry-After") or 10 * (attempt + 1))
            print(f"    rate limited, retrying in {wait}s")
            time.sleep(wait)
    with open(path, "wb") as f:
        f.write(data)


def decode(path, start, dur):
    cmd = [FFMPEG, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", path,
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).copy()


def encode(x, path):
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
           "-c:a", "libvorbis", "-q:a", "3", path]
    subprocess.run(cmd, input=x.astype(np.float32).tobytes(), check=True)


def match_loudness(x, target_db):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    x = x * (10 ** (target_db / 20) / rms)
    peak = np.abs(x).max()
    return x * (PEAK_LIMIT / peak) if peak > PEAK_LIMIT else x


def ramp(n):
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def make_loop(x):
    """Crossfade the tail into the head so the clip loops with no click or gap."""
    n = int(LOOP_XFADE_S * SR)
    body, tail = x[:-n].copy(), x[-n:]
    w = ramp(n)
    body[:n] = body[:n] * np.sqrt(w) + tail * np.sqrt(1 - w)   # equal-power
    return body


def one_shot(x):
    n_in, n_out = int(FADE_S * SR), min(int(TAIL_FADE_S * SR), len(x) // 3)
    x = x.copy()
    x[:n_in] *= ramp(n_in)
    x[-n_out:] *= ramp(n_out)[::-1]
    return x


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    used = {}
    for name, (src_key, start, dur, role, target_db, loop) in CLIPS.items():
        src = SOURCES[src_key]
        x = decode(fetch(src), start, dur + (LOOP_XFADE_S if loop else 0))
        x = match_loudness(x, target_db)
        x = make_loop(x) if loop else one_shot(x)
        path = os.path.join(OUT, f"{name}.ogg")
        encode(x, path)
        manifest.setdefault(role, []).append(f"{name}.ogg")
        used.setdefault(src_key, []).append(name)
        print(f"  {name:15s} {len(x) / SR:5.1f}s  {os.path.getsize(path) / 1024:6.0f} KB  <- {src_key}")

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    lines = ["# Audio credits", "",
             "Crowd sounds in `crowd/` are trimmed, mixed to mono, loudness-matched and re-encoded",
             "excerpts of the recordings below (built by `make_crowd_audio.py`). Licences apply to",
             "these audio files only, not to the game's code. Files from CC BY-SA sources are",
             "shared under CC BY-SA 4.0.", "",
             "| Files | Source | Author | Licence |", "|---|---|---|---|"]
    for key, names in used.items():
        s = SOURCES[key]
        lines.append(f"| {', '.join(f'`{n}.ogg`' for n in names)} | [{key}]({s['page']}) | {s['author']} | {s['license']} |")
    with open(CREDITS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"wrote {len(CLIPS)} clips, {total / 1e6:.1f} MB total, manifest.json and CREDITS.md")


if __name__ == "__main__":
    main()
