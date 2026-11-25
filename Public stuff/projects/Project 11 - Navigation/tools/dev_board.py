"""Run the navigator's web app on a laptop, with fake or replayed IMU data.

Serves unoq_app/assets/ and the same socket messages as the board, driving the
real NavCore, so the pages can be tried on a laptop or phone (same wifi)
before the board is ready. Data goes to dev_data/, not the board's data folder.

    pip install -r tools/requirements-dev.txt
    python tools/dev_board.py                 fake walks, real time
    python tools/dev_board.py --speed 5       fake walks, 5x faster
    python tools/dev_board.py --replay path/to/imu.csv     replay a recorded walk's raw IMU

Open http://localhost:7000 (or http://<laptop IP>:7000 on a phone).

When you press Start, a fake walk of the default route begins (calibration: a
straight 40 m; sensor sessions: the chosen activity). Notices tell you when each true maneuver happens, so you can
practice tapping. Between walks the fake shoe stands still.
"""

import argparse
import asyncio
import hashlib
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "unoq_app" / "python")]

import numpy as np                      # noqa: E402
import socketio                         # noqa: E402
from aiohttp import web                 # noqa: E402

import config as cfg                    # noqa: E402
import fake_walk as fw                  # noqa: E402

ASSETS = ROOT / "unoq_app" / "assets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--speed", type=float, default=1.0, help="play walks this many times faster")
    ap.add_argument("--replay", type=Path, help="imu.csv to play on every walk instead of a fake one")
    ap.add_argument("--data", type=Path, default=ROOT / "dev_data", help="where walks are saved")
    args = ap.parse_args()

    cfg.DATA_DIR = args.data
    cfg.WALKS_DIR = args.data / "walks"
    cfg.WALKERS_PATH = args.data / "walkers.json"
    cfg.POLICY_DIR = args.data / "policy"
    from nav_core import NavCore       # after the paths are set

    sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
    app = web.Application()
    sio.attach(app)
    loop = asyncio.new_event_loop()

    # simulated board clock: runs `speed` times faster than real time
    real0, sim0 = time.time(), time.time()

    def clock():
        return sim0 + (time.time() - real0) * args.speed

    def send(event, data):
        asyncio.run_coroutine_threadsafe(sio.emit(event, data), loop)

    core = NavCore(send=send, clock=clock)

    for event in core.handlers:
        def make(ev):
            async def handler(sid, data=None):
                await asyncio.to_thread(core.handle, ev, data)
            return handler
        sio.on(event, make(event))

    @sio.event
    async def connect(sid, environ):
        await asyncio.to_thread(core.handle, "get_status", {})

    async def index(request):
        return web.FileResponse(ASSETS / "index.html")

    app.router.add_get("/", index)
    app.router.add_static("/", ASSETS)

    threading.Thread(target=feeder, args=(core, clock, args), daemon=True).start()

    async def ticker():
        while True:
            await asyncio.to_thread(core.tick)
            await asyncio.sleep(0.1)

    async def start():
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", args.port).start()
        print(f"dev board on http://localhost:{args.port}  (data in {args.data})")
        await ticker()

    asyncio.set_event_loop(loop)
    loop.run_until_complete(start())


SENSOR_ROUTES = {"walk": [("walk", 50)], "stairs": [("walk", 3), ("stairs", 16, 1), ("walk", 3)],
                 "turn": [("walk", 10), ("left",), ("walk", 10), ("right",), ("walk", 10), ("left",), ("walk", 5)],
                 "still": [("still", 30)]}


def walker_stride(name):
    """A stable fake stride per walker name, 1.2-1.7 m per step."""
    h = int(hashlib.md5((name or "").encode()).hexdigest(), 16)
    return 1.2 + (h % 50) / 100


def feeder(core, clock, args):
    """Sends 10-sample batches every 0.1 simulated seconds, like the sketch."""
    still = np.tile([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (10, 1))
    walk = None                 # (batches, maneuvers, index) while a fake walk plays
    mode_seen = "idle"
    seed = 0
    next_t = clock()
    if args.replay:
        data = np.loadtxt(args.replay, delimiter=",", skiprows=1, ndmin=2)[:, 1:]
        replay_batches = [data[i:i + 10].tolist() for i in range(0, len(data), 10)]
    while True:
        with core.lock:                 # mode and recorder change together in on_start
            mode = core.mode
            activity = core.recorder.meta.get("activity") if core.recorder else None
        if mode != mode_seen:
            mode_seen = mode
            if mode == "idle":
                walk = None
            else:
                seed += 1
                if args.replay:
                    walk = {"batches": replay_batches, "maneuvers": [], "i": 0}
                else:
                    route = ([("walk", 40)] if mode == "calibrate" else
                             SENSOR_ROUTES.get(activity) if mode == "sensor" else None)
                    sim = fw.simulate(route=route, meters_per_step=walker_stride(core.walker), seed=seed,
                                      start_t=0.0, length_jitter=0 if route else 0.05)
                    walk = {"batches": [rows for _, rows in sim["batches"]], "i": 0,
                            "maneuvers": sorted(sim["maneuvers"], key=lambda m: m["t"])}
                walk["t0"] = clock()
        if walk is not None and walk["i"] < len(walk["batches"]):
            rows = walk["batches"][walk["i"]]
            walk["i"] += 1
            elapsed = walk["i"] / 10
            while walk["maneuvers"] and walk["maneuvers"][0]["t"] <= elapsed:
                m = walk["maneuvers"].pop(0)
                core.notice(f"(fake walk) {cfg.SPOKEN.get(m['action'] + '_now', m['action'])} happening now")
            if walk["i"] == len(walk["batches"]):
                core.notice("(fake walk) route finished: press Stop")
        else:
            rows = (still + np.random.default_rng().normal(0, 0.01, still.shape)).tolist()
        # one batch per 0.1 simulated seconds, catching up if sleep() overslept
        # (Windows sleeps in ~15 ms ticks, which matters at --speed > 1)
        next_t += 0.1
        core.on_imu(rows, t=next_t)
        wait = (next_t - clock()) / args.speed
        if wait > 0:
            time.sleep(wait)
        elif wait < -1.0:
            next_t = clock()        # fell far behind (laptop asleep?): skip ahead


if __name__ == "__main__":
    main()
