"""
UNO Q app (Linux side) for the activity wearable.

The sketch streams MPU-6050 samples (100 Hz, batches of 10) through the
Bridge. This file:
  - keeps the newest 2 s and, every 0.5 s, classifies them with both models
    (activity.py): Model 1 = peak acceleration on one axis, Model 2 = all IMU
    channels;
  - records labelled training data when asked from the web page
    (saved as CSV in ../data/), and trains both models from it
    (saved to ../models.json, so they survive a restart);
  - pushes the live samples, predictions and status to the web page
    (assets/, served by the web_ui brick on port 7000).
"""

import json
import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI

import activity as act

ACC_COUNTS_PER_G = 2048.0       # sketch sets +-16 g
GYRO_COUNTS_PER_DPS = 16.4      # sketch sets +-2000 deg/s
TRIM_S = 2.0                    # dropped from both ends of a recording (pressing the button, getting going)
SMOOTH = 3                      # predictions shown = most common of the last 3 (1.5 s)
UI_EVERY_NTH_SAMPLE = 2         # live chart gets 50 Hz, plenty to look at
PEAK_AXIS = None                # None = pick the best axis when training; or 0/1/2 to force ax/ay/az

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
MODEL_PATH = APP_DIR / "models.json"

ui = WebUI()
lock = threading.Lock()
live = deque(maxlen=act.WINDOW)     # newest window, for prediction
to_ui = []                          # samples not yet sent to the page
sample_count = 0
imu_status = "waiting"              # "waiting" (nothing from the sketch yet), "ok", "missing"
last_sample_time = 0.0
recording = None                    # {"label", "samples", "start"} while recording
models = {"peak": None, "knn": None, "report": None}
recent = {"m1": deque(maxlen=SMOOTH), "m2": deque(maxlen=SMOOTH)}


def log(msg):
    print(f"[wearable] {msg}", flush=True)


# ---------- from the sketch ----------

def on_imu_batch(payload):
    global sample_count, last_sample_time, imu_status
    rows = []
    for chunk in str(payload).split(";"):
        parts = chunk.split(",")
        if len(parts) != 6:
            continue
        try:
            v = [int(p) for p in parts]
        except ValueError:
            continue
        rows.append([v[0] / ACC_COUNTS_PER_G, v[1] / ACC_COUNTS_PER_G, v[2] / ACC_COUNTS_PER_G,
                     v[3] / GYRO_COUNTS_PER_DPS, v[4] / GYRO_COUNTS_PER_DPS, v[5] / GYRO_COUNTS_PER_DPS])
    with lock:
        for row in rows:
            live.append(row)
            if sample_count % UI_EVERY_NTH_SAMPLE == 0:
                to_ui.append(row)
            sample_count += 1
        if recording is not None:
            recording["samples"].extend(rows)
        last_sample_time = time.time()
        imu_status = "ok"


def on_imu_status(status):
    global imu_status
    status = str(status)
    if status != imu_status:
        log(f"IMU {status}")
    imu_status = status


Bridge.provide("imu_batch", on_imu_batch)
Bridge.provide("imu_status", on_imu_status)


# ---------- data + models on disk ----------

def load_recordings():
    recs = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        label = path.stem.split("_")[0]
        if label in act.LABELS:
            recs.append((label, np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)))
    return recs


def seconds_per_label():
    secs = {lab: 0.0 for lab in act.LABELS}
    for path in DATA_DIR.glob("*.csv"):
        label = path.stem.split("_")[0]
        if label in secs:
            with open(path) as f:
                secs[label] += (sum(1 for _ in f) - 1) / act.SAMPLE_HZ
    return {k: round(v, 1) for k, v in secs.items()}


def load_models():
    if not MODEL_PATH.exists():
        return
    try:
        d = json.loads(MODEL_PATH.read_text())
        models["peak"] = act.PeakModel.from_dict(d["peak"])
        models["knn"] = act.KnnModel.from_dict(d["knn"])
        models["report"] = d["report"]
        log(f"loaded models (peak axis {models['report']['peak_axis']})")
    except Exception as e:
        log(f"couldn't load {MODEL_PATH.name}: {e}")


def save_recording(rec):
    trim = int(TRIM_S * act.SAMPLE_HZ)
    samples = rec["samples"][trim:len(rec["samples"]) - trim]
    if len(samples) < act.WINDOW:
        min_s = 2 * TRIM_S + act.WINDOW / act.SAMPLE_HZ
        return f"Too short: record at least {min_s:.0f} s (the first and last {TRIM_S:.0f} s are dropped)."
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{rec['label']}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    n = 1
    while path.exists():
        n += 1
        path = path.with_name(f"{rec['label']}_{time.strftime('%Y%m%d-%H%M%S')}-{n}.csv")
    np.savetxt(path, np.array(samples), delimiter=",", fmt="%.4f",
               header="ax,ay,az,gx,gy,gz", comments="")
    log(f"saved {len(samples) / act.SAMPLE_HZ:.1f} s of {rec['label']} to {path.name}")
    return f"Saved {len(samples) / act.SAMPLE_HZ:.1f} s of {rec['label']}."


# ---------- from the web page ----------

def on_record(sid, data):
    global recording
    label = (data or {}).get("label")
    if label not in act.LABELS:
        return
    with lock:
        previous, recording = recording, {"label": label, "samples": [], "start": time.time()}
    if previous is not None:
        notice(save_recording(previous))
    log(f"recording {label}")
    send_status()


def on_stop(sid, data):
    global recording
    with lock:
        rec, recording = recording, None
    if rec is not None:
        notice(save_recording(rec))
    send_status()


def on_train(sid, data):
    try:
        recs = load_recordings()
        peak, knn, report = act.train(recs, peak_axis=PEAK_AXIS)
    except Exception as e:
        notice(f"Training failed: {e}")
        return
    models.update(peak=peak, knn=knn, report=report)
    MODEL_PATH.write_text(json.dumps({"peak": peak.to_dict(), "knn": knn.to_dict(), "report": report}))
    for d in recent.values():
        d.clear()
    log(f"trained on {len(recs)} recordings, peak axis {report['peak_axis']}")
    notice("Trained both models.")
    send_status()


def on_delete(sid, data):
    label = (data or {}).get("label")
    if label not in act.LABELS:
        return
    removed = 0
    for path in DATA_DIR.glob(f"{label}_*.csv"):
        path.unlink()
        removed += 1
    notice(f"Deleted {removed} {label} recording(s). Retrain to update the models.")
    send_status()


def notice(text):
    ui.send_message("notice", {"text": text})


def model_info():
    peak, report = models["peak"], models["report"]
    if peak is None:
        return None
    return {"peak_axis": act.AXES[peak.axis],
            "peak_stats": {lab: [round(m, 3), round(s, 3)] for lab, (m, s) in peak.stats.items()},
            "report": report}


def send_status():
    with lock:
        rec = recording
        status = {
            "imu": imu_status if time.time() - last_sample_time < 1.0 or imu_status != "ok" else "stalled",
            "recording": None if rec is None else {"label": rec["label"],
                                                   "seconds": round(time.time() - rec["start"], 1)},
        }
    status["seconds"] = seconds_per_label()
    status["models"] = model_info()
    ui.send_message("status", status)


ui.on_message("record", on_record)
ui.on_message("stop", on_stop)
ui.on_message("train", on_train)
ui.on_message("delete", on_delete)
ui.on_connect(lambda sid: send_status())


# ---------- main loop ----------

def smoothed(key, label):
    recent[key].append(label)
    return Counter(recent[key]).most_common(1)[0][0]


def predict(window):
    peaks = act.peak_features(window)
    out = {"peaks": [round(float(p), 3) for p in peaks], "m1": None, "m2": None}
    if models["peak"] is not None:
        label, conf = models["peak"].predict(peaks)
        out["m1"] = {"label": smoothed("m1", label), "raw": label, "conf": round(conf, 2),
                     "value": round(float(peaks[models["peak"].axis]), 3)}
    if models["knn"] is not None:
        feats = act.full_features(window)
        label, conf = models["knn"].predict(feats)
        out["m2"] = {"label": smoothed("m2", label), "raw": label, "conf": round(conf, 2),
                     "cadence": round(float(feats[-2]), 2)}
    return out


next_predict = 0.0
next_status = 0.0


def loop():
    global next_predict, next_status
    now = time.time()
    with lock:
        new = to_ui[:]
        to_ui.clear()
        window = np.array(live) if len(live) == act.WINDOW else None
    if new:
        ui.send_message("samples", {"s": np.round(new, 3).tolist()})
    if now >= next_predict and window is not None:
        next_predict = now + act.HOP / act.SAMPLE_HZ
        ui.send_message("prediction", predict(window))
    if now >= next_status:
        next_status = now + 1.0
        send_status()
    time.sleep(0.05)


load_models()
App.run(user_loop=loop)
