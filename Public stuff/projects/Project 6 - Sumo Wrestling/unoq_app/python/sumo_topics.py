"""
sumo_topics.py -- the ONE place the MQTT schema and robot hardware IDs live.

Shared by every side of the system so they can't drift apart:
  - the UNO Q app (python/main.py, python/relay.py),
  - the laptop brain (../../sumo_brain.py imports this through config.py),
  - the laptop BLE fallback (../../ble_relay_laptop.py).

MQTT schema (broker test.mosquitto.org, qos 0, never retained):

  T/cmd     laptop -> robot   {"seq", "l": -100..100, "r": -100..100, "mode"}
                              sent at ~15 Hz; robot stops if none arrive for CMD_TIMEOUT_S
  T/song    laptop -> robot   {"song": "win" | "lose"}  -> robot stops, beeps the song
  T/sensor  robot  -> laptop  {"color", "name", "reflection", "hue", "sat", "val", "rgb", "ts"}
  T/status  robot  -> laptop  {"motor": bool, "color": bool, "uptime", "host", "cmd_fresh"}  1 Hz heartbeat
  T/unoq    UNO Q  -> laptop  {"fan_rpm", "relay_ble", "ts"}  1 Hz, fan speed from the tach wire
  ME193/Rogers                referee: plain "start" (optionally "[ME193/...] start")

where T = TOPIC_ROOT below.
"""
import legoeducation as le

# ---- fill these in (placeholders, NOT real values) -------------------------
TEAM = "Will-Codrin"            # TODO: e.g. "Will-<partner>" -- makes our topics unique

# Numbers printed on each device's Connection Card. None = connect to the first
# advertising device of that type (fine alone at home, ambiguous in class).
MOTOR_CARD_COLOR = le.LEGO_COLOR_RED        # TODO: e.g. le.LEGO_COLOR_RED (import legoeducation to use the constant)
MOTOR_CARD_SERIAL = 3664         # TODO: number printed on the double motor's card
COLOR_CARD_COLOR = le.LEGO_COLOR_RED          # TODO
COLOR_CARD_SERIAL = 3664         # TODO: number printed on the color sensor's card
# ---------------------------------------------------------------------------

BROKER_HOST = "test.mosquitto.org"
BROKER_PORT = 1883

TOPIC_ROOT = f"ME193/{TEAM}/sumo"
TOPIC_CMD = f"{TOPIC_ROOT}/cmd"
TOPIC_SONG = f"{TOPIC_ROOT}/song"
TOPIC_SENSOR = f"{TOPIC_ROOT}/sensor"
TOPIC_STATUS = f"{TOPIC_ROOT}/status"
TOPIC_UNOQ = f"{TOPIC_ROOT}/unoq"
TOPIC_REFEREE = "ME193/Rogers"

CMD_TIMEOUT_S = 0.6              # robot watchdog: no fresh cmd for this long -> stop motors
                                 # (public-broker jitter; lower it if you use a local broker)

# Same notes as Project 3 - World Cup/songs.py (copied, because the UNO Q
# can't import files from the laptop's repo). Laptop plays the matching mp3s.
SUCCESS_SONG = [523, 659, 784, 1047, 1319]
DEATH_SONG = [523, 466, 415, 349, 311, 233]
