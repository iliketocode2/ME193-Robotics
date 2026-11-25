"""UNO Q app (Linux side): connects the sketch and the web page to NavCore.

The sketch (unchanged from Projects 7/8) sends 10 raw MPU-6050 samples every
100 ms. They are converted to g and deg/s here and handed to NavCore, which
does everything else (see nav_core.py). The page in assets/ is served by the
web_ui brick on port 7000: open http://<board>:7000 on the phone.
"""

import time

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI

import config as cfg
from nav_core import NavCore

ui = WebUI()
core = NavCore(send=ui.send_message)


def on_imu_batch(payload):
    rows = []
    for chunk in str(payload).split(";"):
        parts = chunk.split(",")
        if len(parts) != 6:
            continue
        try:
            v = [int(p) for p in parts]
        except ValueError:
            continue
        rows.append([v[0] / cfg.ACC_COUNTS_PER_G, v[1] / cfg.ACC_COUNTS_PER_G, v[2] / cfg.ACC_COUNTS_PER_G,
                     v[3] / cfg.GYRO_COUNTS_PER_DPS, v[4] / cfg.GYRO_COUNTS_PER_DPS, v[5] / cfg.GYRO_COUNTS_PER_DPS])
    if rows:
        core.on_imu(rows)


Bridge.provide("imu_batch", on_imu_batch)
Bridge.provide("imu_status", core.on_imu_status)

for event in core.handlers:
    ui.on_message(event, lambda sid, data, event=event: core.handle(event, data))
ui.on_connect(lambda sid: core.handle("get_status", {}))


def loop():
    core.tick()
    time.sleep(0.1)


App.run(user_loop=loop)
