"""
Sonia's brain: a local, open-source language model (Qwen3-4B, int4) running
on the laptop's Intel Arc GPU through OpenVINO GenAI.

It runs as its OWN PROCESS (this file with --worker, talking JSON lines over
stdin/stdout) so that loading the model (tens of seconds the first time) and
generating lines (~1.5-2 s each) can never stall the camera, the 60 Hz game
tick, or Bluetooth -- and if it crashes or runs out of memory the game
carries on with Ray (the scripted play-by-play) alone. If the game exits,
the worker sees its stdin close and exits too.

DEAD BALL ONLY. The model shares the laptop's GPU, memory bandwidth and power
budget with the camera (MediaPipe) and the 3D graphics. Measured: while it
generated, camera fps fell from ~30 to ~22, and during its warm-up to ~17.
So the game holds a "gate": it is open only when the ball is dead (menus,
countdown, serve, between points, game over). When a rally starts the gate
closes, and any line being written is cancelled within one token.
commentary.Booth decides when to ask for a line and what to say it about;
this file only turns chat messages into text.

One-time setup (downloads ~2.2 GB into the gitignored repo-root models/):
    my_env/Scripts/pip install openvino-genai huggingface_hub
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/ai_commentator.py" --setup

Check speed + sample lines on this machine:
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/ai_commentator.py" --bench
"""

import argparse
import collections
import json
import os
import queue
import subprocess
import sys
import threading
import time

# Qwen3-4B writes the best lines; OpenVINO/Qwen3-1.7B-int4-ov is ~2x faster but blander.
MODEL_ID = os.environ.get("PINGPONG_AI_MODEL", "OpenVINO/Qwen3-4B-int4-ov")
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_ID.split("/")[1])
CACHE_DIR = os.path.join(MODELS_DIR, "ov_cache")      # compiled-model cache: later startups are faster
DEVICES = ("GPU", "CPU")                               # Arc iGPU first; CPU only as a fallback
TEMPERATURE = 0.8                                      # lively but still coherent
MAX_NEW_TOKENS = 26                                    # one <=12-word sentence; tokens are most of the time
PROTO = "@@AI "                                       # marks protocol lines (anything else on stdout is ignored)
MAX_QUEUED_REQUESTS = 2  # requests waiting to be sent/run; beyond this new ones are dropped, never waited on


class Analyst:
    """Handle the game uses (main process). Non-blocking throughout: requests
    go out through a writer thread (a pipe write can block if the worker is
    busy or stuck, and the 60 Hz tick must never wait on it), and answers come
    back through a reader thread."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.status = "loading" if enabled else "off"
        self.detail = ""
        self.device = None
        self.last_latency = None
        self.generating = False
        self.cancelled = 0             # lines cut off because a rally started
        self.gate_open = True
        self._queued = 0                 # requests not yet written to the worker (guarded by _qlock)
        self._qlock = threading.Lock()
        self._proc = None

    @property
    def ready(self):
        return self.status == "ready"

    def start(self):
        if not self.enabled:
            return
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__), "--worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            self.status, self.detail = "offline", f"could not start AI process: {e}"
            return
        self._results = queue.Queue()
        self._outbox = queue.Queue()   # unbounded: gate messages must never be dropped (requests are capped below)
        threading.Thread(target=self._read, name="ai-reader", daemon=True).start()
        threading.Thread(target=self._write, name="ai-writer", daemon=True).start()

    def _read(self):
        """Background thread: worker stdout -> queue (so poll() never blocks)."""
        try:
            for line in self._proc.stdout:
                if line.startswith(PROTO):
                    try:
                        self._results.put(json.loads(line[len(PROTO):]))
                    except ValueError:
                        pass
        except Exception as e:                       # never die silently: report it
            self._results.put({"type": "status", "status": "offline", "detail": f"AI reader stopped: {e}"})

    def _write(self):
        """Background thread: queued requests -> worker stdin. Only this thread
        can block on the pipe."""
        while True:
            msg = self._outbox.get()
            if msg is None:
                return
            try:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError):            # worker gone / pipe closed
                self._results.put({"type": "status", "status": "offline", "detail": "AI process stopped"})
                return
            finally:
                if msg.get("type") == "req":
                    with self._qlock:
                        self._queued -= 1

    def request(self, req):
        """Ask for a line. req: {"id", "messages", "seed", "expires"(game clock)}.
        Never blocks: if the worker is backed up, the request is simply dropped
        (the Booth treats a line that never arrives like a late one)."""
        if not self.ready:
            return
        with self._qlock:
            if self._queued >= MAX_QUEUED_REQUESTS:
                return
            self._queued += 1
        ttl = req["expires"] - time.monotonic()
        self._outbox.put({"type": "req", "id": req["id"], "messages": req["messages"], "seed": req["seed"],
                          "expires_wall": time.time() + ttl})

    def set_gate(self, open_):
        """Open = the ball is dead and Sonia may use the GPU. Closing it cancels
        a line in progress. Only changes are sent."""
        if self._proc is None or open_ == self.gate_open:
            return
        self.gate_open = open_
        self._outbox.put({"type": "gate", "open": open_})

    def poll(self):
        """Finished lines since the last call: [{"id", "text", "latency"}]. Also
        picks up status changes (loading -> ready / offline)."""
        if self._proc is None:
            return []
        out = []
        while True:
            try:
                msg = self._results.get_nowait()
            except queue.Empty:
                break
            if msg["type"] == "status":
                self.status, self.detail, self.device = msg["status"], msg.get("detail", ""), msg.get("device")
                print(f"[AI booth] {self.status} {self.device or ''} {self.detail}".rstrip())
            elif msg["type"] == "busy":
                self.generating = True
            elif msg["type"] == "line":
                self.generating = False
                if msg.get("cancelled"):
                    self.cancelled += 1
                else:
                    self.last_latency = msg["latency"]
                    out.append(msg)
        if self.status in ("loading", "ready") and self._proc.poll() is not None:
            self.status, self.detail = "offline", "AI process stopped"
        return out

    def info(self):
        return {"status": self.status, "detail": self.detail, "device": self.device,
                "latency": self.last_latency, "gate": self.gate_open, "generating": self.generating,
                "cancelled": self.cancelled}

    def close(self):
        """Stop the worker: closing its stdin asks it to exit; kill it if it
        doesn't. Never raises (shutdown cleanup after this must still run) and
        waits at most ~5 s."""
        if self._proc is None:
            return
        try:
            self._outbox.put(None)                   # stop the writer thread
            # Close stdin from a side thread: if the writer is stuck in a write
            # (pipe full because the worker never got past loading), close()
            # would wait on the same lock forever. The kill below unsticks it.
            def close_stdin():
                try:
                    self._proc.stdin.close()
                except (OSError, ValueError):
                    pass
            threading.Thread(target=close_stdin, name="ai-close", daemon=True).start()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            if self._proc.poll() is None:
                self._proc.kill()
                self._proc.wait(timeout=2)
        except BaseException as e:                   # incl. a second Ctrl-C
            print(f"AI worker shutdown: {e!r}")


# --------------------------------------------------------------------- worker
def load_pipeline(log=print):
    """(pipeline, device). Raises with a readable reason if it can't load."""
    import openvino_genai as ov_genai            # imported here: only the worker process needs it
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError("model not downloaded -- run: ai_commentator.py --setup")
    errors = []
    for device in DEVICES:
        try:
            t = time.time()
            pipe = ov_genai.LLMPipeline(MODEL_DIR, device, CACHE_DIR=CACHE_DIR)
            log(f"loaded {MODEL_ID} on {device} in {time.time() - t:.1f}s")
            return pipe, device
        except Exception as e:                   # e.g. no GPU driver -> try the CPU
            errors.append(f"{device}: {e}")
    raise RuntimeError("; ".join(errors))


def make_generator(pipe):
    """Returns generate(messages, seed, should_stop=None) -> (text, seconds, cancelled).
    should_stop() is checked after every token; True cancels the line."""
    import openvino_genai as ov_genai
    tok = pipe.get_tokenizer()

    def generate(messages, seed, should_stop=None):
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = MAX_NEW_TOKENS
        cfg.do_sample = True
        cfg.temperature = TEMPERATURE
        cfg.top_p = 0.95
        cfg.repetition_penalty = 1.1
        cfg.rng_seed = seed                      # without this every repeat gives the same line
        cfg.stop_strings = {"\n"}
        cfg.apply_chat_template = False          # applied below, with Qwen3's "thinking" switched off
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                         extra_context={"enable_thinking": False})
        cancelled = [False]

        def streamer(_chunk):
            if should_stop and should_stop():
                cancelled[0] = True
                return ov_genai.StreamingStatus.CANCEL
            return ov_genai.StreamingStatus.RUNNING

        t = time.perf_counter()
        text = str(pipe.generate(prompt, cfg, streamer)).strip()
        return ("" if cancelled[0] else text), time.perf_counter() - t, cancelled[0]

    return generate


def warm_up_prompts():
    """The GPU compiles kernels the first time it sees each prompt shape (that
    first call can take 10+ s). Real booth-sized prompts, run once at startup
    -- through the same dead-ball gate as everything else."""
    from commentary import PERSONAS, build_messages
    return [build_messages({"moment": "intro", "opponent": PERSONAS["pip"]}),
            build_messages({"moment": "point", "opponent": PERSONAS["rita"], "point_to": "player",
                            "how": "the opponent hit the net", "rally_shots": 4, "score_situation": "level"})]


def set_process_mode(background):
    """Windows scheduling hints for THIS process. background=False (used by the
    game for its camera + tick threads): above-normal priority, no power
    throttling (Windows prefers the performance cores -- measured: worst tick
    gap 91 -> 19 ms) and always honour the 1 ms timer request, even while the
    window is hidden. background=True: below-normal priority (unused now).
    (EcoQoS for the AI worker was measured and rejected: it doubled line time,
    1.3 s -> 2.9 s, for no benefit once the gate keeps it out of rallies.)
    Best effort: returns False if the OS refuses."""
    if sys.platform != "win32":
        try:
            os.nice(5 if background else 0)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    me = k32.GetCurrentProcess()
    BELOW_NORMAL, ABOVE_NORMAL, PROCESS_POWER_THROTTLING = 0x4000, 0x8000, 4
    EXECUTION_SPEED, IGNORE_TIMER_RESOLUTION = 0x1, 0x4
    ok = bool(k32.SetPriorityClass(me, BELOW_NORMAL if background else ABOVE_NORMAL))
    if not background:
        state = PowerThrottlingState(1, EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, 0)   # opt out of both
        ok &= bool(k32.SetProcessInformation(me, PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)))
    return ok


def _send(msg):
    print(PROTO + json.dumps(msg), flush=True)


class WorkerLoop:
    """The worker's scheduling, separate from the model so it can be tested
    with a fake generate(). feed() is called from the stdin-reader thread;
    run() processes requests only while the dead-ball gate is open."""

    MAX_PENDING = 3

    def __init__(self, generate, send):
        self.generate = generate             # (messages, seed, should_stop) -> (text, secs, cancelled)
        self.send = send
        self.pending = collections.deque()
        self.gate_open = True                # menus first: open
        self.closed = False
        self.cond = threading.Condition()

    def feed(self, msg):
        with self.cond:
            if msg.get("type") == "gate":
                self.gate_open = bool(msg["open"])
            elif msg.get("type") == "req":
                self.pending.append(msg)
                while len(self.pending) > self.MAX_PENDING:
                    self.pending.popleft()   # oldest first: it's the stalest
            self.cond.notify_all()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()

    def should_stop(self):
        return self.closed or not self.gate_open

    def wait_for_gate(self):
        """Block until the ball is dead. False if the game has gone away."""
        with self.cond:
            self.cond.wait_for(lambda: self.closed or self.gate_open)
            return not self.closed

    def warm_up(self, prompts):
        for i, messages in enumerate(prompts):
            while self.wait_for_gate():
                _, _, cancelled = self.generate(messages, i + 1, self.should_stop)
                if not cancelled:
                    break

    def run(self):
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.closed or (self.gate_open and self.pending))
                if self.closed:
                    return
                req = self.pending.popleft()
            if time.time() > req["expires_wall"]:
                continue                     # already too late to be useful -- skip it
            self.send({"type": "busy", "id": req["id"]})
            try:
                text, secs, cancelled = self.generate(req["messages"], req["seed"], self.should_stop)
            except Exception as e:
                text, secs, cancelled = "", 0.0, False
                self.send({"type": "status", "status": "ready", "detail": f"last line failed: {e}"})
            self.send({"type": "line", "id": req["id"], "text": text, "latency": secs, "cancelled": cancelled})


def worker():
    """The AI process: requests and gate changes arrive as JSON lines on stdin,
    answers go out on stdout. Exits when stdin closes (the game quit or crashed)."""
    try:
        # Load EVERYTHING before starting the stdin reader. On Windows, a thread
        # blocked reading stdin deadlocks native-library imports (numpy/OpenVINO
        # touch the standard handles while loading) -- the worker hung forever.
        # Messages sent meanwhile just wait in the pipe.
        pipe, device = load_pipeline(log=lambda m: None)   # normal priority: finishes during camera pick
        loop = WorkerLoop(make_generator(pipe), _send)
        prompts = warm_up_prompts()
    except Exception as e:
        _send({"type": "status", "status": "offline", "detail": str(e)[:300]})
        return

    def read_stdin():
        for line in sys.stdin:
            try:
                loop.feed(json.loads(line))
            except ValueError:
                continue
        loop.close()                         # game gone

    threading.Thread(target=read_stdin, name="stdin", daemon=True).start()
    loop.warm_up(prompts)
    if loop.closed:
        return
    # No priority games needed: the gate keeps the model off the hardware during
    # rallies, and at normal priority lines come out faster (measured A/B:
    # median ~3.0 s vs ~3.7 s at below-normal under full load; rallies 30 fps either way).
    _send({"type": "status", "status": "ready", "device": device})
    loop.run()


# ----------------------------------------------------------------------- CLI
def setup():
    from huggingface_hub import snapshot_download
    print(f"Downloading {MODEL_ID} (~2.2 GB) to {MODEL_DIR} ...")
    snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    print("Done. Try: ai_commentator.py --bench")


def bench():
    import random

    import game_logic as gl
    from commentary import Booth, sanitize_line
    pipe, device = load_pipeline()
    generate = make_generator(pipe)
    for i, messages in enumerate(warm_up_prompts()):
        generate(messages, i + 1)

    # a few realistic moments, built the same way the game builds them
    g = gl.Game(rng=random.Random(4))
    g.set_highlight("rita"); g.confirm()
    booth = Booth(rng=random.Random(4))
    moments = [booth.update(0.0, [{"type": "start", "opp": "rita"}], g, ready=True)[0]]
    g.score = {gl.PLAYER: 3, gl.OPP: 6}
    for i, (w, r) in enumerate([(gl.OPP, "wrong_place"), (gl.OPP, "wrong_place"), (gl.OPP, "wrong_place")]):
        evs = [{"type": "miss", "reason": r}, {"type": "point", "winner": w, "reason": "miss",
                                                 "score": {gl.PLAYER: 3, gl.OPP: 4 + i}}]
        booth.last_spoke = -1e9
        moments += booth.update(10.0 + i, evs, g, ready=True)
    g.winner, g.score = gl.PLAYER, {gl.PLAYER: 11, gl.OPP: 9}
    moments += booth.update(30.0, [{"type": "game_over", "winner": gl.PLAYER, "score": g.score}], g, ready=True)

    print(f"\n{MODEL_ID} on {device}")
    for req in moments:
        for _ in range(2):
            text, secs, _ = generate(req["messages"], random.randrange(1, 2 ** 31))
            print(f"  {req['kind']:9s} {secs * 1000:5.0f} ms  {sanitize_line(text) or '[dropped] ' + text}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", action="store_true", help="download the model")
    ap.add_argument("--bench", action="store_true", help="time a few sample lines on this machine")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)   # started by the game
    args = ap.parse_args()
    if args.worker:
        worker()
    elif args.setup:
        setup()
    elif args.bench:
        bench()
    else:
        ap.print_help()
