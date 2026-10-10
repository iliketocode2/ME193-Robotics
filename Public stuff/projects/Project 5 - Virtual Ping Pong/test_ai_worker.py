"""
Self-check for the AI worker's scheduling (ai_commentator.WorkerLoop) -- no
model, GPU or game needed: a fake generate() stands in for the LLM.

The rule under test: Sonia only uses the GPU while the ball is dead. When
the game closes the gate (a rally starts) any line in progress is cancelled
and nothing new starts until it opens again.

Run:
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_ai_worker.py"
"""

import threading
import time

from ai_commentator import WorkerLoop

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


class FakeModel:
    """generate() that takes `tokens` steps of `step` seconds, honouring should_stop."""

    def __init__(self, tokens=10, step=0.02):
        self.tokens, self.step = tokens, step
        self.calls = []

    def __call__(self, messages, seed, should_stop=None):
        self.calls.append(messages)
        t = time.perf_counter()
        for i in range(self.tokens):
            time.sleep(self.step)
            if should_stop and should_stop():
                return "", time.perf_counter() - t, True
        return f"line {len(self.calls)}", time.perf_counter() - t, False


def start(model):
    sent = []
    loop = WorkerLoop(model, sent.append)
    th = threading.Thread(target=loop.run, daemon=True)
    th.start()
    return loop, sent, th


def req(i, ttl=10.0):
    return {"type": "req", "id": i, "messages": [{"role": "user", "content": str(i)}], "seed": i,
            "expires_wall": time.time() + ttl}


def wait_for(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


lines = lambda sent: [m for m in sent if m["type"] == "line"]

# --- gate open: requests are answered --------------------------------------------
m = FakeModel()
loop, sent, th = start(m)
loop.feed(req(1))
check(wait_for(lambda: lines(sent)), "gate open: a request is answered")
check(lines(sent)[0]["text"] == "line 1" and not lines(sent)[0]["cancelled"], "answer carries the text, not cancelled")
check(sent[0] == {"type": "busy", "id": 1}, "worker announces when it starts generating")

# --- gate closed: nothing runs; it runs once the gate opens ------------------------
loop.feed({"type": "gate", "open": False})
loop.feed(req(2))
time.sleep(0.3)
check(len(m.calls) == 1, "gate closed (rally on): queued request does NOT start")
loop.feed({"type": "gate", "open": True})
check(wait_for(lambda: len(lines(sent)) == 2), "gate opens (ball dead): the queued request runs")

# --- closing the gate cancels a line in progress, quickly --------------------------
slow = FakeModel(tokens=100, step=0.02)          # a 2-second line
loop2, sent2, th2 = start(slow)
loop2.feed(req(1))
wait_for(lambda: slow.calls)
time.sleep(0.1)
t = time.perf_counter()
loop2.feed({"type": "gate", "open": False})
check(wait_for(lambda: lines(sent2)), "closing the gate ends the line in progress")
took = time.perf_counter() - t
check(lines(sent2)[0]["cancelled"] and lines(sent2)[0]["text"] == "", "...reported as cancelled, with no text")
check(took < 0.1, f"...within one token of the rally starting ({took * 1000:.0f} ms)")

# --- stale and excess requests ------------------------------------------------------
m3 = FakeModel(tokens=1)
loop3, sent3, th3 = start(m3)
loop3.feed({"type": "gate", "open": False})
loop3.feed(req(1, ttl=-1))                      # already expired
for i in range(2, 7):
    loop3.feed(req(i))
check(len(loop3.pending) == WorkerLoop.MAX_PENDING, f"backlog is capped at {WorkerLoop.MAX_PENDING} (oldest dropped)")
loop3.feed({"type": "gate", "open": True})
wait_for(lambda: len(lines(sent3)) >= 3)
check([l["id"] for l in lines(sent3)] == [4, 5, 6], "the newest requests are the ones answered")

# --- warm-up waits for a dead ball, and retries a cancelled warm-up --------------------
m4 = FakeModel(tokens=20, step=0.01)
loop4 = WorkerLoop(m4, lambda msg: None)
loop4.feed({"type": "gate", "open": False})
done = threading.Event()
threading.Thread(target=lambda: (loop4.warm_up([["a"], ["b"]]), done.set()), daemon=True).start()
time.sleep(0.2)
check(not m4.calls, "warm-up does not touch the GPU while a rally is on")
loop4.feed({"type": "gate", "open": True})
check(done.wait(2) and len(m4.calls) == 2, "warm-up runs both prompts once the ball is dead")

# --- shutdown -----------------------------------------------------------------------
loop.close(); loop2.close(); loop3.close()
check(wait_for(lambda: not th.is_alive() and not th2.is_alive() and not th3.is_alive()),
      "the loop exits when the game closes the pipe")

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED")
raise SystemExit(1 if failures else 0)
