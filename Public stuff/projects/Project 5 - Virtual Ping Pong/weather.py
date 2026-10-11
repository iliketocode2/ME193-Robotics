"""
Live local weather for the arena.

Source: the Arduino UNO Q running the ip-weather app (ArduinoApps/ip-weather)
fetches Open-Meteo for Tufts and publishes it over MQTT (retained) on TOPIC.
If the board is silent, the PC fetches the same data itself, so the scene
and the commentators always have real weather.

  normalize()      Open-Meteo JSON -> the payload (same as the board's build_payload)
  validate()       payload from the (public!) broker -> clamped payload, or None
  scene_params()   payload -> 0..1 parameters the browser's weather renderer uses
  describe()       payload -> "light rain, chilly, breezy, evening" (for Sonia)
  bucket()         payload -> coarse categories (Ray's weather lines, change calls)
  WeatherWatch     notable changes worth a between-points call ("here comes the rain")
  WeatherFeed      MQTT + PC fallback; the 60 Hz game tick reads current() without blocking

Pure apart from fetch_open_meteo() (urllib, on the feed's own thread).
"""

import calendar
import json
import math
import threading
import time
import urllib.request
from datetime import datetime

PLACE = "Tufts"
LAT, LON = 42.4075, -71.1190
TOPIC = "ME193/WilliamGoldman/weather"
URL = ("https://api.open-meteo.com/v1/forecast"
       f"?latitude={LAT}&longitude={LON}"
       "&current=temperature_2m,apparent_temperature,relative_humidity_2m,is_day,precipitation,rain,showers,"
       "snowfall,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility"
       "&daily=sunrise,sunset&temperature_unit=fahrenheit&wind_speed_unit=mph"
       "&timezone=America%2FNew_York&forecast_days=1")
MAX_PAYLOAD = 2048

# WMO weather code -> (cloud, precip, snow_mix, fog, lightning, hail, cloud_dark)
WMO = {
    0: (0.05, 0, 0, 0, 0, 0, 0), 1: (0.25, 0, 0, 0, 0, 0, 0), 2: (0.55, 0, 0, 0, 0, 0, 0.05),
    3: (0.95, 0, 0, 0, 0, 0, 0.3),
    45: (0.9, 0, 0, 0.85, 0, 0, 0.2), 48: (0.9, 0, 0, 0.95, 0, 0, 0.2),
    51: (0.85, 0.15, 0, 0.1, 0, 0, 0.3), 53: (0.85, 0.3, 0, 0.15, 0, 0, 0.35), 55: (0.85, 0.45, 0, 0.2, 0, 0, 0.4),
    56: (0.9, 0.25, 0.2, 0.15, 0, 0, 0.4), 57: (0.9, 0.45, 0.2, 0.2, 0, 0, 0.45),
    61: (0.9, 0.35, 0, 0.1, 0, 0, 0.4), 63: (0.9, 0.6, 0, 0.15, 0, 0, 0.5), 65: (0.9, 0.9, 0, 0.25, 0, 0, 0.6),
    66: (0.9, 0.4, 0.3, 0.15, 0, 0, 0.45), 67: (0.9, 0.75, 0.3, 0.2, 0, 0, 0.55),
    71: (0.9, 0.3, 1, 0.1, 0, 0, 0.3), 73: (0.9, 0.6, 1, 0.2, 0, 0, 0.35), 75: (0.9, 0.9, 1, 0.35, 0, 0, 0.4),
    77: (0.9, 0.25, 1, 0.1, 0, 0, 0.3),
    80: (0.7, 0.45, 0, 0.05, 0, 0, 0.4), 81: (0.7, 0.7, 0, 0.1, 0, 0, 0.5), 82: (0.7, 1.0, 0, 0.2, 0, 0, 0.6),
    85: (0.8, 0.5, 1, 0.15, 0, 0, 0.35), 86: (0.8, 0.9, 1, 0.3, 0, 0, 0.45),
    95: (1.0, 0.8, 0, 0.15, 0.5, 0, 0.8), 96: (1.0, 0.85, 0, 0.15, 0.8, 1, 0.8), 99: (1.0, 0.9, 0, 0.2, 0.8, 1, 0.85),
}
SKY = {0: "clear", 1: "clear", 2: "partly", 3: "cloudy", 45: "fog", 48: "fog",
       51: "drizzle", 53: "drizzle", 55: "drizzle", 56: "sleet", 57: "sleet",
       61: "rain", 63: "rain", 65: "heavy_rain", 66: "sleet", 67: "sleet",
       71: "snow", 73: "snow", 75: "heavy_snow", 77: "snow", 80: "showers", 81: "showers", 82: "heavy_rain",
       85: "snow", 86: "heavy_snow", 95: "storm", 96: "storm", 99: "storm"}
PRECIP_KIND = {"drizzle": "rain", "rain": "rain", "heavy_rain": "rain", "showers": "rain", "sleet": "sleet",
               "snow": "snow", "heavy_snow": "snow", "storm": "storm"}
COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
TWILIGHT_S = 2400            # +/- 40 min of dusk/dawn light around sunrise/sunset


# ---------------------------------------------------------------- the payload
def normalize(om, src, sent=None):
    """Open-Meteo forecast JSON -> payload (schema v1). Keep identical to
    ArduinoApps/ip-weather/python/weather_payload.py build_payload()."""
    c, d = om["current"], om.get("daily") or {}
    first = lambda key: (d.get(key) or [None])[0]   # noqa: E731
    return {
        "v": 1, "src": src, "place": PLACE, "lat": LAT, "lon": LON,
        "obs": c["time"], "utc_offset_s": int(om.get("utc_offset_seconds", 0)),
        "sent": int(time.time() if sent is None else sent),
        "temp_f": round(float(c["temperature_2m"]), 1), "feels_f": round(float(c["apparent_temperature"]), 1),
        "rh": int(c.get("relative_humidity_2m", 50)), "code": int(c["weather_code"]),
        "precip_mm": round(float(c.get("precipitation", 0)), 2), "rain_mm": round(float(c.get("rain", 0)), 2),
        "showers_mm": round(float(c.get("showers", 0)), 2), "snow_cm": round(float(c.get("snowfall", 0)), 2),
        "cloud": int(c.get("cloud_cover", 0)), "wind_mph": round(float(c.get("wind_speed_10m", 0)), 1),
        "gust_mph": round(float(c.get("wind_gusts_10m", 0)), 1), "wind_dir": int(c.get("wind_direction_10m", 0)),
        "is_day": int(c.get("is_day", 1)), "vis_m": int(c.get("visibility", 20000)),
        "sunrise": first("sunrise"), "sunset": first("sunset"),
    }


_RANGES = {"temp_f": (-60, 130), "feels_f": (-80, 150), "rh": (0, 100), "precip_mm": (0, 200), "rain_mm": (0, 200),
           "showers_mm": (0, 200), "snow_cm": (0, 50), "cloud": (0, 100), "wind_mph": (0, 150),
           "gust_mph": (0, 200), "wind_dir": (0, 360), "is_day": (0, 1), "vis_m": (0, 100000),
           "utc_offset_s": (-14 * 3600, 14 * 3600)}


def _iso(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M") if isinstance(s, str) else None
    except ValueError:
        return None


def validate(payload):
    """Anyone can publish to a public broker: accept only a well-formed v1
    payload, clamp every number into a physical range. Returns a clean dict
    or None. Never raises."""
    try:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", "replace")
        if isinstance(payload, str):
            if len(payload) > MAX_PAYLOAD:
                return None
            payload = json.loads(payload)
        if not isinstance(payload, dict) or payload.get("v") != 1:
            return None
        code = int(payload["code"])
        if code not in WMO or _iso(payload.get("obs")) is None:
            return None
        out = {"v": 1, "src": payload.get("src") if payload.get("src") in ("unoq", "pc", "demo") else "unoq",
               "place": PLACE, "lat": LAT, "lon": LON, "obs": payload["obs"], "code": code,
               "sent": int(payload.get("sent", 0)),
               "sunrise": payload.get("sunrise") if _iso(payload.get("sunrise")) else None,
               "sunset": payload.get("sunset") if _iso(payload.get("sunset")) else None}
        for key, (lo, hi) in _RANGES.items():
            v = float(payload[key]) if key in ("temp_f", "feels_f") else float(payload.get(key, lo if key != "vis_m" else 20000))
            if not math.isfinite(v):
                return None
            out[key] = max(lo, min(hi, v))
        for key in ("rh", "cloud", "wind_dir", "is_day", "vis_m", "utc_offset_s"):
            out[key] = int(out[key])
        return out
    except (KeyError, TypeError, ValueError):
        return None


def _epoch(local_iso, utc_offset_s):
    """Open-Meteo local time 'YYYY-MM-DDTHH:MM' -> unix time (UTC)."""
    dt = _iso(local_iso)
    return None if dt is None else calendar.timegm(dt.timetuple()) - utc_offset_s


def obs_age_s(p, now_wall):
    t = _epoch(p["obs"], p["utc_offset_s"])
    return 0.0 if t is None else max(0.0, now_wall - t)


# ------------------------------------------------------------- scene mapping
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _smooth(e0, e1, x):
    t = _clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def _lerp(a, b, t):
    return a + (b - a) * t


def daylight(p, now_wall):
    """0 = night .. 1 = day, smooth through dusk and dawn."""
    rise, sset = _epoch(p.get("sunrise"), p["utc_offset_s"]), _epoch(p.get("sunset"), p["utc_offset_s"])
    if rise is None or sset is None:
        return float(p["is_day"])
    shift = 86400 * round((now_wall - rise) / 86400)               # the sunrise nearest "now" (games past midnight)
    rise, sset = rise + shift, sset + shift
    return _smooth(-TWILIGHT_S / 2, TWILIGHT_S / 2, min(now_wall - rise, sset - now_wall))


def _sun(p, now_wall):
    """Unit vector toward the sun in scene coordinates (x = east, y = up,
    z = south: the camera looks north) and its elevation in degrees."""
    rise, sset = _epoch(p.get("sunrise"), p["utc_offset_s"]), _epoch(p.get("sunset"), p["utc_offset_s"])
    doy = time.gmtime(now_wall + p["utc_offset_s"]).tm_yday
    max_el = 90 - LAT + 23.44 * math.sin(2 * math.pi * (doy - 81) / 365)
    if rise is not None and sset is not None and sset > rise:
        span = sset - rise
        frac = ((now_wall - rise) % 86400) / span
    else:
        frac = 0.5 if p["is_day"] else 1.5
    if 0 <= frac <= 1:
        el = max_el * math.sin(math.pi * frac)
        az = math.pi * frac                                    # 0 = east, pi/2 = south, pi = west
    else:
        el, az = -12.0, math.pi / 2                            # below the horizon
    e = math.radians(el)
    x, y, z = math.cos(e) * math.cos(az), math.sin(e), math.cos(e) * math.sin(az)
    return (round(x, 3), round(y, 3), round(z, 3)), round(el, 1)


def moon_phase(now_wall):
    """0 = new, 0.5 = full."""
    synodic = 29.530588 * 86400
    return round(((now_wall - 947182440) % synodic) / synodic, 3)     # 2000-01-06 18:14 UTC new moon


def scene_params(p, now_wall=None):
    """Payload -> what the browser's weather renderer needs (0..1 unless noted).
    Gameplay clamps live here so they're tested: fog never closer than 6.5 m,
    rain at most 35% opaque, the table's key light never below 60%."""
    now_wall = time.time() if now_wall is None else now_wall
    cloud, precip, snow_mix, fog, lightning, hail, dark = WMO[p["code"]]
    cloud = max(cloud, p["cloud"] / 100)
    precip = _clamp(precip + min(0.3, p["precip_mm"] / 5)) if precip else _clamp(p["precip_mm"] / 3)
    if p["snow_cm"] > 0:
        snow_mix = max(snow_mix, _clamp(p["snow_cm"] / (p["snow_cm"] + p["rain_mm"] / 10 + 1e-6)))
    fog = max(fog, _clamp((3000 - p["vis_m"]) / 2800), 0.25 * precip)
    light = daylight(p, now_wall)
    sun_dir, sun_el = _sun(p, now_wall)
    wind = _clamp(p["wind_mph"] / 30)
    gust = _clamp((p["gust_mph"] - p["wind_mph"]) / 20)
    toward = math.radians(p["wind_dir"])                          # meteorological: where it blows FROM
    return {
        "code": p["code"], "temp_f": p["temp_f"],
        "cloud": round(cloud, 3), "cloud_dark": round(dark, 3), "precip": round(precip, 3),
        "snow_mix": round(snow_mix, 3), "hail": hail, "lightning": lightning, "fog": round(fog, 3),
        "wind": round(wind, 3), "gust": round(gust, 3),
        "wind_x": round(-math.sin(toward), 3), "wind_z": round(math.cos(toward), 3),   # direction it blows TO
        "daylight": round(light, 3), "sun_dir": sun_dir, "sun_el": sun_el, "moon_phase": moon_phase(now_wall),
        "stars": round((1 - light) * (1 - cloud), 3),
        "floods": round(_clamp((0.45 - light) * 3 + 0.4 * dark + 0.3 * fog), 3),
        "wet": round(_clamp(precip * 1.4) * (1 - snow_mix), 3),
        "snow_cover": round(_clamp(snow_mix * (0.3 + precip * 0.6) + min(0.4, p["snow_cm"] / 5)), 3),
        # gameplay clamps
        "fog_near": round(_lerp(14, 6.5, fog), 2), "fog_far": round(_lerp(34, 16, fog), 2),
        "rain_alpha": round(min(0.35, 0.16 + 0.19 * precip), 3), "key_min": 0.6,
    }


# --------------------------------------------------------------------- words
def time_of_day(p, now_wall):
    hour = time.gmtime(now_wall + p["utc_offset_s"]).tm_hour
    if daylight(p, now_wall) < 0.15 and not 17 <= hour < 21:
        return "night"
    return "morning" if 4 <= hour < 12 else "afternoon" if hour < 17 else "evening" if hour < 21 else "night"


def wind_level(mph):
    return 0 if mph < 4 else 1 if mph < 12 else 2 if mph < 20 else 3 if mph < 32 else 4


def bucket(p, now_wall=None):
    """Coarse categories: Ray's pre-rendered weather lines are keyed by these."""
    now_wall = time.time() if now_wall is None else now_wall
    sky = SKY[p["code"]]
    params = WMO[p["code"]]
    temp = round(p["temp_f"])
    feels = round(p["feels_f"])
    level = wind_level(p["wind_mph"])
    tod = time_of_day(p, now_wall)
    return {"sky": sky, "tod": tod, "night": tod == "night" or daylight(p, now_wall) < 0.3,
            "temp": temp, "feels": "colder" if feels <= temp - 6 else "warmer" if feels >= temp + 6 else None,
            "wind_level": level, "wind_from": COMPASS[round(p["wind_dir"] / 45) % 8],
            "precip_kind": PRECIP_KIND.get(sky), "precip_heavy": sky in ("heavy_rain", "heavy_snow") or params[1] >= 0.75,
            "fog": sky == "fog" or p["vis_m"] < 1500}


def describe(p, now_wall=None):
    """'light rain, chilly, breezy, evening' -- words, not numbers, for Sonia."""
    b = bucket(p, now_wall)
    sky = {"clear": "clear skies" if not b["night"] else "a clear night sky", "partly": "partly cloudy",
           "cloudy": "grey and overcast", "fog": "foggy", "drizzle": "drizzle", "rain": "rain",
           "heavy_rain": "heavy rain", "showers": "rain showers", "sleet": "freezing rain", "snow": "snow",
           "heavy_snow": "heavy snow", "storm": "a thunderstorm"}[b["sky"]]
    t = b["temp"]
    temp = ("freezing" if t < 32 else "cold" if t < 45 else "chilly" if t < 58 else "mild" if t < 72
            else "warm" if t < 85 else "hot")
    wind = {0: None, 1: None, 2: "breezy", 3: "windy", 4: "blowing a gale"}[b["wind_level"]]
    return ", ".join(x for x in (sky, temp, wind, b["tod"]) if x)


def notable(p, now_wall=None):
    """Worth a weather angle from Sonia mid-match?"""
    b = bucket(p, now_wall)
    return bool(b["precip_kind"] or b["fog"] or b["wind_level"] >= 2 or b["night"] or not 40 <= b["temp"] <= 85)


class WeatherWatch:
    """Notable changes for Ray's between-points call ("And here comes the
    rain."): a change must hold MIN_HOLD_S (no flapping between partly/mostly
    cloudy), and calls are at least MIN_GAP_S apart."""

    MIN_HOLD_S = 60.0
    MIN_GAP_S = 180.0

    def __init__(self):
        self.base = None            # last announced (or first seen) state
        self.cand, self.cand_since = None, 0.0
        self.last_call = -1e9

    @staticmethod
    def _state(b):
        return (b["precip_kind"], b["precip_heavy"], b["night"], b["fog"], b["wind_level"], b["sky"])

    def observe(self, b, now):
        """b = bucket(). Returns a change key (see commentary.WEATHER_CHANGE_LINES) or None."""
        st = self._state(b)
        if self.base is None:
            self.base = st
            return None
        if st == self.base:
            self.cand = None
            return None
        if st != self.cand:
            self.cand, self.cand_since = st, now
            return None
        if now - self.cand_since < self.MIN_HOLD_S or now - self.last_call < self.MIN_GAP_S:
            return None
        key = self._key(self.base, st)
        self.base, self.cand = st, None
        if key:
            self.last_call = now
        return key

    @staticmethod
    def _key(old, new):
        (pk0, heavy0, night0, fog0, wind0, sky0), (pk1, heavy1, night1, fog1, wind1, sky1) = old, new
        if pk1 == "storm" and pk0 != "storm":
            return "storm"
        if pk1 != pk0:
            if pk1 == "snow":
                return "snow_start"
            if pk1 in ("rain", "sleet"):
                return "rain_start"
            if pk0 == "snow":
                return "snow_stop"
            if pk0 in ("rain", "sleet", "storm"):
                return "rain_stop"
        elif pk1 and heavy1 != heavy0:
            return "rain_heavier" if heavy1 else "rain_easing"
        if night1 != night0:
            return "night" if night1 else "day"
        if fog1 != fog0:
            return "fog_in" if fog1 else "fog_out"
        if wind1 >= wind0 + 2:
            return "wind_up"
        if not night1 and sky1 in ("clear", "partly") and sky0 in ("cloudy", "fog", "drizzle", "rain", "showers"):
            return "sun_out"
        return None


# ---------------------------------------------------------------------- feed
def fetch_open_meteo(timeout=6):
    """The PC's own fetch (fallback when the UNO Q is silent). Blocking: only
    ever called on WeatherFeed's thread."""
    with urllib.request.urlopen(URL, timeout=timeout) as r:
        return normalize(json.loads(r.read().decode()), "pc")


DEMOS = {   # --weather-demo: fixed conditions for checking the visuals and performance
    "clear": dict(code=0, cloud=5, is_day=1, hour=14), "rain": dict(code=63, cloud=95, precip=2.5, wind=12, hour=15),
    "storm": dict(code=95, cloud=100, precip=6, wind=22, gust=40, hour=17),
    "snow": dict(code=73, cloud=95, snow=1.2, temp=28, wind=8, hour=11),
    "fog": dict(code=45, cloud=90, vis=400, hour=8), "night": dict(code=0, cloud=0, is_day=0, hour=23),
    "windy": dict(code=2, cloud=50, wind=28, gust=45, hour=13),
}


def demo_payload(name, now_wall=None):
    """A fake Open-Meteo reading (local time pinned to the demo's hour)."""
    d = DEMOS[name]
    now_wall = time.time() if now_wall is None else now_wall
    off = -14400
    today = time.strftime("%Y-%m-%d", time.gmtime(now_wall + off))
    om = {"utc_offset_seconds": off,
          "current": {"time": f"{today}T{d['hour']:02d}:00", "temperature_2m": d.get("temp", 55),
                      "apparent_temperature": d.get("temp", 55) - 3, "relative_humidity_2m": 70,
                      "is_day": d.get("is_day", int(7 <= d["hour"] < 18)), "precipitation": d.get("precip", d.get("snow", 0)),
                      "rain": d.get("precip", 0), "showers": 0, "snowfall": d.get("snow", 0), "weather_code": d["code"],
                      "cloud_cover": d["cloud"], "wind_speed_10m": d.get("wind", 5), "wind_direction_10m": 250,
                      "wind_gusts_10m": d.get("gust", d.get("wind", 5) * 1.6), "visibility": d.get("vis", 20000)},
          "daily": {"sunrise": [f"{today}T06:50"], "sunset": [f"{today}T18:10"]}}
    p = normalize(om, "demo")
    return p, _epoch(p["obs"], off) + 120          # "now" = the demo's local hour


class WeatherFeed:
    """Current weather for the game, from the UNO Q (MQTT) or the PC fallback.

    on_mqtt() runs on the MQTT thread, step() on the feed's own thread; the
    game tick only calls current(), which returns an immutable snapshot (one
    atomic attribute read: no lock, never blocks).

    Policy (step): no UNO Q reading within uno_wait_s of starting -> the PC
    fetches; the UNO Q's reading is too old (stale_s) or it has gone silent
    (silent_s) -> re-subscribe and fetch on the PC; PC readings refresh every
    pc_refetch_s, failures retry after retry_s. UNO Q readings always win."""

    def __init__(self, fetch=fetch_open_meteo, clock=time.monotonic, wall=time.time, resubscribe=None,
                 uno_wait_s=5.0, pc_refetch_s=600.0, stale_s=1800.0, silent_s=180.0, retry_s=60.0,
                 params_every_s=30.0, demo=None):
        self.fetch, self.clock, self.wall, self.resubscribe = fetch, clock, wall, resubscribe
        self.uno_wait_s, self.pc_refetch_s, self.stale_s = uno_wait_s, pc_refetch_s, stale_s
        self.silent_s, self.retry_s, self.params_every_s = silent_s, retry_s, params_every_s
        self._lock = threading.Lock()
        self._snap = None
        self._version = 0
        self._start = clock()
        self._uno_last = None           # clock time of the last valid UNO Q reading
        self._uno_data = None
        self._next_pc = self._start + uno_wait_s
        self._last_resub = self._start
        self._last_params = -1e9
        self._demo = demo
        self.errors = 0
        self._stop = threading.Event()
        if demo:
            p, self._demo_wall = demo_payload(demo)
            self._publish(p, "demo")

    # --- inputs
    def on_mqtt(self, topic, payload):
        """paho callback thread. Must never raise."""
        try:
            p = validate(payload)
            if p is None or self._demo:
                return
            with self._lock:
                self._uno_last = self.clock()
                self._uno_data = p
            self._publish(p, "unoq")
        except Exception:
            self.errors += 1

    def step(self, now=None):
        """The fallback/staleness policy; called every couple of seconds."""
        now = self.clock() if now is None else now
        if self._demo:
            return
        with self._lock:
            uno_last, uno_data = self._uno_last, self._uno_data
        uno_live = uno_last is not None and now - uno_last < self.silent_s \
            and obs_age_s(uno_data, self.wall()) < self.stale_s
        if uno_last is not None and now - uno_last >= self.silent_s and now - self._last_resub >= self.silent_s:
            self._last_resub = now
            if self.resubscribe:
                try:
                    self.resubscribe()
                except Exception:
                    self.errors += 1
        snap = self._snap
        showing_dead_uno = snap is not None and snap["src"] == "unoq"     # the board went silent/stale: fetch now
        if not uno_live and (now >= self._next_pc or showing_dead_uno) \
                and (uno_last is not None or now - self._start >= self.uno_wait_s):
            try:
                self._publish(self.fetch(), "pc")
                self._next_pc = now + self.pc_refetch_s
            except Exception:
                self.errors += 1
                self._next_pc = now + self.retry_s
        snap = self._snap
        if snap and now - self._last_params >= self.params_every_s:
            self._publish(snap["data"], snap["src"])             # the sun moves: refresh the scene

    def _publish(self, p, src):
        with self._lock:
            wall = self._demo_wall if self._demo else self.wall()
            self._version += 1
            self._last_params = self.clock()
            self._snap = {"version": self._version, "data": p, "src": src, "params": scene_params(p, wall),
                          "desc": describe(p, wall), "bucket": bucket(p, wall), "notable": notable(p, wall),
                          "age_min": round(obs_age_s(p, wall) / 60, 1),
                          "stale": obs_age_s(p, wall) >= self.stale_s}

    # --- output
    def current(self):
        """Latest snapshot dict (never mutated after publishing), or None."""
        return self._snap

    def start(self):
        threading.Thread(target=self._run, name="weather", daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                self.errors += 1
            self._stop.wait(2.0)
