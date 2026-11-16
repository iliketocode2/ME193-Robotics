"""Pure logic for saving the Color Sensor's battery (no PyScript or DOM here, so it is
unit-tested with plain Python; see AnnaCode/tests/test_power_logic.py).

The sensor's LED is on in its normal mode and is a real drain (a sensor went 80% -> 30% ->
dead in one session). The library documents three modes: color detection (LED on), ambient
light (LED off) and off (sensor and LED off). With "Save power" ticked, the page keeps a
sensor OFF except while data is actually being collected or the network is running.

The page asks with the library's own set_color_sensor_mode(mode, blocking=False) and reads the
sensor's mode from the colorSensorMode field of each color reading (legoeducation 1.2.0).
misc_ui/power.py applies these rules to the page.
"""

MODE_OFF = 0       # sensor and LED off
MODE_COLOR = 1     # color detection, LED on: the normal mode
MODE_AMBIENT = 2   # ambient light, LED off

SETTLE_S = 0.6         # when the sensor can't report its mode, assume a change took effect after this
RESEND_AFTER_S = 3.0   # a request the sensor has not acted on by now is sent again
MAX_RETRIES = 3        # ... up to this many times, then give up (and say so in the console)


def wanted_mode(save_power: bool, needed: bool, used_as_input: bool) -> int:
    """The light is on unless Save power is on and nothing needs this sensor right now.
    "Needed" means data is being collected or the network is running, and only a sensor
    that an Input node actually reads counts."""
    return MODE_COLOR if (not save_power or (needed and used_as_input)) else MODE_OFF


def effective_mode(requested, observed):
    """What to believe the sensor is doing: what it reports if it reports, else what we asked."""
    return observed if observed is not None else requested


def ready(wanted: int, observed, since_request_s: float) -> bool:
    """Is the sensor in the wanted mode (so its readings can be trusted)?
    If it reports its mode, wait for that; if not, give it SETTLE_S after the request."""
    if observed is not None:
        return observed == wanted
    return since_request_s >= SETTLE_S


def resend_due(requested, observed, since_request_s: float, retries: int) -> bool:
    """Should a request be sent again? Only when the sensor reports its mode, still reports
    the wrong one after RESEND_AFTER_S, and we have not already tried MAX_RETRIES times."""
    if requested is None or observed is None or observed == requested:
        return False
    return since_request_s >= RESEND_AFTER_S and retries < MAX_RETRIES


def note_text(mode) -> str:
    """Short label on the sensor's chip."""
    return "light off" if mode == MODE_OFF else ""
