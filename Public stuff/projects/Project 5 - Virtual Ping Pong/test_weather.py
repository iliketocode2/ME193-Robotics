"""
Self-check for weather.py -- the live local weather behind the arena's sky
and the commentators' weather talk. No network, no MQTT, no UNO Q needed.

Run:
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/test_weather.py"
"""

import importlib.util
import json
import os
import threading
import time

import weather as w

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# An Open-Meteo reading for Tufts (shape copied from a real response)
OM = {"utc_offset_seconds": -14400,
      "current": {"time": "2026-10-10T22:15", "interval": 900, "temperature_2m": 47.9, "apparent_temperature": 43.9,
                  "relative_humidity_2m": 74, "is_day": 0, "precipitation": 0.0, "rain": 0.0, "showers": 0.0,
                  "snowfall": 0.0, "weather_code": 3, "cloud_cover": 100, "wind_speed_10m": 3.8,
                  "wind_direction_10m": 203, "wind_gusts_10m": 6.5, "visibility": 21800.0},
      "daily": {"time": ["2026-10-10"], "sunrise": ["2026-10-10T06:51"], "sunset": ["2026-10-10T18:10"]}}
NOW = w._epoch("2026-10-10T22:20", -14400)            # five minutes after that reading

# --- the payload: the PC and the UNO Q build exactly the same thing ----------------
p = w.normalize(OM, "pc", sent=1)
check(p["code"] == 3 and p["temp_f"] == 47.9 and p["sunset"] == "2026-10-10T18:10" and p["place"] == "Tufts",
      "normalize: Open-Meteo -> payload")
check(len(json.dumps(p)) < 600, f"payload is small ({len(json.dumps(p))} bytes)")
board = os.path.join(os.path.expanduser("~"), "GitHub Projects", "ArduinoApps", "ip-weather", "python", "weather_payload.py")
if os.path.isfile(board):
    spec = importlib.util.spec_from_file_location("weather_payload", board)
    wp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wp)
    check(wp.build_payload(OM, "pc", sent=1) == p and wp.TOPIC == w.TOPIC and wp.URL == w.URL,
          "the UNO Q's build_payload() == the PC's normalize(), same topic and URL")
    check(wp.short_label(p) == "48F CLOUD" and wp.short_label(None) == "--F", "the board's LED label: '48F CLOUD'")
else:
    print("SKIP  (ArduinoApps/ip-weather not found: board payload not compared)")

# --- validation: the broker is public ----------------------------------------------
check(w.validate(json.dumps(p)) == p, "a good payload passes unchanged")
check(w.validate(json.dumps(p).encode()) == p, "...also as bytes")
for bad, why in [("not json", "garbage"), ("[1,2]", "not an object"), (json.dumps({**p, "v": 2}), "wrong version"),
                 (json.dumps({**p, "code": 42}), "unknown weather code"), (json.dumps({**p, "obs": "yesterday"}), "bad time"),
                 (json.dumps({k: v for k, v in p.items() if k != "temp_f"}), "missing temperature"),
                 ("x" * 5000, "too big"), (json.dumps({**p, "temp_f": float("nan")}).replace("NaN", "NaN"), "NaN")]:
    check(w.validate(bad) is None, f"rejected: {why}")
q = w.validate(json.dumps({**p, "wind_mph": 9999, "cloud": -5, "src": "<script>"}))
check(q["wind_mph"] == 150 and q["cloud"] == 0 and q["src"] == "unoq", "out-of-range numbers are clamped, odd strings replaced")

# --- scene parameters --------------------------------------------------------------
ok = True
for code in w.WMO:
    for hour in range(0, 24, 3):
        sp = w.scene_params({**p, "code": code}, w._epoch(f"2026-10-10T{hour:02d}:00", -14400))
        for k in ("cloud", "cloud_dark", "precip", "snow_mix", "fog", "wind", "gust", "daylight", "stars", "floods",
                  "wet", "snow_cover", "moon_phase"):
            ok &= 0 <= sp[k] <= 1
        ok &= sp["fog_near"] >= 6.5 and sp["fog_far"] > sp["fog_near"] and sp["rain_alpha"] <= 0.35 and sp["key_min"] >= 0.6
check(ok, "every WMO code at every hour: params in 0..1, fog never closer than 6.5 m, rain <= 35% opaque")
day = lambda hhmm: w.daylight(p, w._epoch(f"2026-10-10T{hhmm}", -14400))   # noqa: E731
check(day("12:00") == 1 and day("23:00") == 0 and day("03:00") == 0, "daylight: 1 at noon, 0 at night")
dusk = [day(f"{17 + m // 60}:{m % 60:02d}") for m in range(30, 120, 5)]
check(all(b <= a for a, b in zip(dusk, dusk[1:])) and 0 < day("18:10") < 1, "dusk fades smoothly through sunset")
noon, night = w.scene_params(p, w._epoch("2026-10-10T12:30", -14400)), w.scene_params(p, NOW)
check(noon["sun_el"] > 30 and noon["sun_dir"][2] > 0.5 and night["sun_el"] < 0, "the sun is high in the south at noon, down at night")
check(night["floods"] >= 0.9 and noon["floods"] == 0, "floodlights on at night, off at noon")
clear_night = w.scene_params({**p, "code": 0, "cloud": 0}, NOW)
check(clear_night["stars"] > 0.9 and night["stars"] == 0, "stars on a clear night, none under cloud")
west = w.scene_params({**p, "wind_dir": 270, "wind_mph": 15}, NOW)
check(west["wind_x"] > 0.99 and abs(west["wind_z"]) < 0.01, "a west wind blows toward +x (east)")
storm = w.scene_params({**p, "code": 95, "precip_mm": 8}, NOW)
snow = w.scene_params({**p, "code": 73, "snow_cm": 1.0}, NOW)
check(storm["lightning"] > 0 and storm["precip"] > 0.9 and snow["snow_mix"] == 1 and snow["snow_cover"] > 0.3,
      "storms have lightning and heavy rain; snow falls as snow and settles")

# --- words ---------------------------------------------------------------------------
check(w.describe(p, NOW) == "grey and overcast, chilly, night", f"describe: '{w.describe(p, NOW)}'")
b = w.bucket({**p, "code": 63, "wind_mph": 22, "wind_dir": 250}, w._epoch("2026-10-10T19:00", -14400))
check((b["sky"], b["tod"], b["precip_kind"], b["wind_level"], b["wind_from"]) == ("rain", "evening", "rain", 3, "west"),
      f"bucket: rain, evening, strong west wind ({b})")
check(w.notable({**p, "code": 61}, NOW) and not w.notable({**p, "code": 1, "temp_f": 65}, w._epoch("2026-10-10T13:00", -14400)),
      "notable: rain is, a mild clear afternoon isn't")

# --- change calls: hysteresis --------------------------------------------------------
wt = w.WeatherWatch()
cloudy, partly = w.bucket(p, NOW), w.bucket({**p, "code": 2}, NOW)
rain = w.bucket({**p, "code": 63}, NOW)
check(wt.observe(cloudy, 0) is None, "first reading sets the baseline (no call)")
flaps = [wt.observe(partly if i % 2 else cloudy, 30.0 * i) for i in range(1, 20)]
check(not any(flaps), "flapping between partly and mostly cloudy never triggers a call")
calls = [wt.observe(rain, t) for t in (600, 630, 661, 700)]
check(calls == [None, None, "rain_start", None], f"rain starting is called once, after it has held for a minute ({calls})")
check(wt.observe(cloudy, 720) is None and wt.observe(cloudy, 790) is None, "...and the next change waits out the 3-minute gap")
check(wt.observe(cloudy, 900) == "rain_stop", "...then 'the rain has stopped'")

# --- the feed: UNO Q first, PC fallback ----------------------------------------------
class Clock:
    t = 0.0

    def __call__(self):
        return self.t


clk, fetches, resubs = Clock(), [], []
wall = lambda: NOW + clk.t   # noqa: E731
feed = w.WeatherFeed(fetch=lambda: fetches.append(clk.t) or w.normalize(OM, "pc"), clock=clk, wall=wall,
                     resubscribe=lambda: resubs.append(clk.t))
feed.step(0.0)
check(feed.current() is None and not fetches, "start: waits for the UNO Q first")
clk.t = 10.5; feed.step(clk.t)
check(feed.current()["src"] == "pc" and fetches == [10.5], "no UNO Q after 10 s: the PC fetches it")
clk.t = 20; feed.on_mqtt(w.TOPIC, json.dumps(w.normalize(OM, "unoq")))
check(feed.current()["src"] == "unoq", "a UNO Q reading takes over")
for t in range(22, 180, 2):
    clk.t = t; feed.step(t)
check(fetches == [10.5], "while the UNO Q is live, the PC doesn't fetch")
clk.t = 201; feed.step(clk.t)
check(resubs == [201] and fetches[-1] == 201, "UNO Q silent for 3 min: re-subscribe and fetch on the PC")
clk.t = 202; feed.on_mqtt(w.TOPIC, "{not json")
feed.on_mqtt(w.TOPIC, b"\xff\xfe")
check(feed.errors == 0 and feed.current()["src"] == "pc", "junk on the topic is ignored without errors")
v0 = feed.current()["version"]
clk.t = 240; feed.step(clk.t)
check(feed.current()["version"] > v0, "the scene is refreshed every 30 s even without new data (dusk moves)")
boom = w.WeatherFeed(fetch=lambda: (_ for _ in ()).throw(OSError("offline")), clock=clk, wall=wall)
clk.t = 1000; boom._start = 0; boom.step(clk.t)
check(boom.current() is None and boom.errors == 1 and boom._next_pc == 1060, "a failed fetch retries in a minute")
stale = w.WeatherFeed(fetch=lambda: w.normalize(OM, "pc"), clock=clk, wall=lambda: NOW + 4000)
stale.on_mqtt(w.TOPIC, json.dumps(w.normalize(OM, "unoq")))
stale.step(clk.t)
check(stale.current()["src"] == "pc", "a UNO Q reading over 30 min old: the PC fetches a fresh one")
demo = w.WeatherFeed(demo="storm")
check(demo.current()["src"] == "demo" and demo.current()["params"]["lightning"] > 0, "--weather-demo storm")

# thread safety: the MQTT thread publishes while the game tick reads
hammer = w.WeatherFeed(fetch=lambda: w.normalize(OM, "pc"))
msg = json.dumps(w.normalize(OM, "unoq"))
errors, versions = [], []


def writer():
    for _ in range(300):
        hammer.on_mqtt(w.TOPIC, msg)


threads = [threading.Thread(target=writer) for _ in range(4)]
for th in threads:
    th.start()
t_end = time.time() + 5
while any(th.is_alive() for th in threads) and time.time() < t_end:
    try:
        snap = hammer.current()
        if snap:
            versions.append(snap["version"])
            _ = snap["params"]["cloud"], snap["desc"]
    except Exception as e:
        errors.append(e)
for th in threads:
    th.join()
check(not errors and hammer.errors == 0 and versions == sorted(versions) and hammer.current()["version"] == 1200,
      f"4 MQTT threads x 300 readings while the tick reads: no errors, versions only go up ({len(versions)} reads)")

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED")
raise SystemExit(1 if failures else 0)
