"""
Natural commentator voices: Kokoro-82M, a small open-source neural
text-to-speech model, run on the CPU through OpenVINO.

Browser voices (Web Speech) sound robotic, so the booth speaks with Kokoro
instead -- without costing the game any speed:

  Ray    every sentence he can say is rendered ONCE, at setup, into a clip
         cache (models/tts/). In game his calls are just clips played back
         to back by the browser: zero synthesis while you play.
  Sonia  her lines are written live by the AI, so they're synthesized live,
         in the AI worker process (ai_commentator.py), on 2 CPU threads,
         and only while the ball is dead -- same gate as her text.

Any line without a clip falls back to the browser's voice, so a missing
model or cache never silences the booth.

Measured on this laptop (Core Ultra 9 288V, on battery): ~0.85 s of CPU per
second of speech on 2 threads -- the same as on 8, so 2 leaves the camera's
cores alone. The Arc GPU can't run this model (OpenVINO's GPU plugin lacks
one of its ops), and ONNX Runtime was 2-3x slower than OpenVINO on the CPU.

One-time setup (~350 MB of model files into the gitignored repo-root
models/, then renders Ray's lines -- a few minutes):
    my_env/Scripts/pip install kokoro-onnx
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/tts.py" --setup
"""

import argparse
import hashlib
import os
import re
import sys
import time
import types
import urllib.request
import wave

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
KOKORO_DIR = os.path.join(MODELS_DIR, "kokoro")
MODEL_FILE = os.path.join(KOKORO_DIR, "kokoro-v1.0.onnx")
VOICES_FILE = os.path.join(KOKORO_DIR, "voices-v1.0.bin")
CLIP_DIR = os.path.join(MODELS_DIR, "tts")                 # served to the browser at /tts/
DOWNLOAD = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

# speaker -> (Kokoro voice, speed). British voices; "bm_" = male, "bf_" = female.
# Others to try: bm_fable, bm_lewis, bm_daniel / bf_isabella, bf_alice, bf_lily.
VOICES = {"ray": ("bm_george", 1.1), "sonia": ("bf_emma", 1.0)}
THREADS = 2                       # per synthesizer: no faster with more (measured), and spares the camera's cores
PRERENDER_PROCS = 4               # parallel synthesizers during --setup (nothing else is running then)


def clip_name(speaker, text):
    """Cache file for a line, relative to CLIP_DIR: '<voice>/<hash>.wav'.
    The hash covers voice, speed and text, so changing any re-renders it."""
    voice, speed = VOICES[speaker]
    digest = hashlib.sha1(f"{voice}|{speed}|{text}".encode()).hexdigest()[:16]
    return f"{voice}/{digest}.wav"


def clip_path(speaker, text):
    return os.path.join(CLIP_DIR, clip_name(speaker, text))


def clip_url(speaker, text):
    return "tts/" + clip_name(speaker, text)


SAFE_CLIP = re.compile(r"^/tts/([a-z]{2}_[a-z]+/[0-9a-f]{16}\.wav)$")


def resolve_url(url_path):
    """HTTP path -> clip file, or None (only exact cache names: no traversal)."""
    m = SAFE_CLIP.match(url_path)
    return os.path.join(CLIP_DIR, *m.group(1).split("/")) if m else None


class ClipIndex:
    """Which clips exist, read once at startup (no disk checks per call)."""

    def __init__(self):
        self.names = set()
        if os.path.isdir(CLIP_DIR):
            for voice in os.listdir(CLIP_DIR):
                folder = os.path.join(CLIP_DIR, voice)
                if os.path.isdir(folder):
                    self.names.update(f"{voice}/{f}" for f in os.listdir(folder) if f.endswith(".wav"))

    def __len__(self):
        return len(self.names)

    def urls(self, speaker, parts):
        """Clip URLs for every part of a call, or None if any part is missing
        (then the whole call uses the browser voice: never two voices in one call)."""
        names = [clip_name(speaker, p) for p in parts]
        return ["tts/" + n for n in names] if names and all(n in self.names for n in names) else None


# ------------------------------------------------------------------ synthesis
class _OVSession:
    """The bit of onnxruntime.InferenceSession kokoro-onnx uses, on OpenVINO."""

    def __init__(self, path, threads):
        import openvino as ov
        self._model_path = path                  # kokoro-onnx reads this
        core = ov.Core()
        core.set_property({"CACHE_DIR": os.path.join(MODELS_DIR, "ov_cache")})
        self.compiled = core.compile_model(core.read_model(path), "CPU",
                                           {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads})
        self.req = self.compiled.create_infer_request()
        self.names = [p.get_any_name() for p in self.compiled.inputs]

    def get_inputs(self):
        return [types.SimpleNamespace(name=n) for n in self.names]

    def run(self, _outputs, inputs):
        import numpy as np
        res = self.req.infer({k: np.asarray(v) for k, v in inputs.items()})
        return [res[self.compiled.outputs[0]]]


class Synth:
    """Kokoro on the CPU. say(speaker, text, path) writes a WAV clip."""

    def __init__(self, threads=THREADS):
        from kokoro_onnx import Kokoro               # imported here: only synthesizing processes need it
        if not (os.path.isfile(MODEL_FILE) and os.path.isfile(VOICES_FILE)):
            raise RuntimeError("voice model not downloaded -- run: tts.py --setup")
        self.kokoro = Kokoro.from_session(_OVSession(MODEL_FILE, threads), VOICES_FILE)

    def say(self, speaker, text, path):
        """Render text in speaker's voice to path (atomically: the web server
        never sees half a file). Returns seconds of audio."""
        import numpy as np
        voice, speed = VOICES[speaker]
        audio, rate = self.kokoro.create(text, voice=voice, speed=speed, lang="en-gb")
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        os.replace(tmp, path)
        return len(pcm) / rate


# ---------------------------------------------------------------------- setup
def download():
    os.makedirs(KOKORO_DIR, exist_ok=True)
    for path in (MODEL_FILE, VOICES_FILE):
        if os.path.isfile(path):
            continue
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(DOWNLOAD + os.path.basename(path), path + ".part")
        os.replace(path + ".part", path)


_synth = None


def _init_proc():
    global _synth
    _synth = Synth()


def _render(text):
    path = clip_path("ray", text)
    if not os.path.isfile(path):
        _synth.say("ray", text, path)
    return text


def prerender():
    """Render every line Ray can say (commentary.ray_phrases) that isn't cached yet."""
    from multiprocessing import Pool

    from commentary import ray_phrases
    todo = [t for t in sorted(ray_phrases()) if not os.path.isfile(clip_path("ray", t))]
    total = len(ray_phrases())
    print(f"Ray has {total} lines; {total - len(todo)} already rendered, {len(todo)} to go.")
    if not todo:
        return
    t = time.time()
    with Pool(PRERENDER_PROCS, initializer=_init_proc) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render, todo, chunksize=4), 1):
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - t)
                print(f"  {i}/{len(todo)}  (~{(len(todo) - i) / rate / 60:.1f} min left)")
    print(f"Done in {(time.time() - t) / 60:.1f} min -> {CLIP_DIR}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", action="store_true", help="download the voice model and render Ray's lines")
    ap.add_argument("--say", nargs=2, metavar=("SPEAKER", "TEXT"), help="render one line (ray|sonia) and time it")
    args = ap.parse_args()
    if args.setup:
        download()
        prerender()
    elif args.say:
        speaker, text = args.say
        s = Synth()
        t = time.perf_counter()
        secs = s.say(speaker, text, clip_path(speaker, text))
        print(f"{time.perf_counter() - t:.2f} s for {secs:.1f} s of audio -> {clip_path(speaker, text)}")
    else:
        ap.print_help()
