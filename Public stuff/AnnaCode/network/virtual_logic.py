"""Pure logic for the Trainer's virtual devices (no PyScript or DOM here, so it
can be unit-tested with plain Python; see AnnaCode/tests/test_virtual_logic.py).

virtual_devices.py is the glue that applies these to the page.
"""

# Stick travel when a key is held, in stick percent per second. Springing back
# is faster than pushing out, like a real spring-loaded joystick.
KEY_RATE_UP = 400.0
KEY_RATE_DOWN = 800.0

# Keys that drive a virtual controller (letters compared lower-case).
#   Arrow keys drive like a car:  up/down = both sticks forward/back, left/right = turn.
#   W/S = left stick, I/K = right stick, for single-stick (tank) control.
STICK_KEYS = frozenset({"ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "w", "s", "i", "k"})

DEGREES_PER_SPEED_SECOND = 6.0   # a motor "speed" of 1 is about 1 RPM = 6 degrees per second


def normalize_key(key: str) -> str:
    """Single letters compare lower-case (Shift/CapsLock must not change the control)."""
    return key.lower() if isinstance(key, str) and len(key) == 1 else key


def key_targets(keys) -> tuple[float, float]:
    """(left, right) stick targets in percent (-100..100) for the set of held keys."""
    held = {normalize_key(k) for k in keys}
    fwd = (1 if "ArrowUp" in held else 0) - (1 if "ArrowDown" in held else 0)
    turn = (1 if "ArrowRight" in held else 0) - (1 if "ArrowLeft" in held else 0)
    left = fwd + turn + (1 if "w" in held else 0) - (1 if "s" in held else 0)
    right = fwd - turn + (1 if "i" in held else 0) - (1 if "k" in held else 0)
    clamp = lambda v: max(-1, min(1, v)) * 100.0  # noqa: E731
    return clamp(left), clamp(right)


def has_control_key(keys) -> bool:
    return any(normalize_key(k) in STICK_KEYS for k in keys)


def ramp(value: float, target: float, dt: float,
         rate_up: float = KEY_RATE_UP, rate_down: float = KEY_RATE_DOWN) -> float:
    """Move value toward target by at most rate*dt. Moving away from zero uses
    rate_up; moving toward zero (letting go) uses rate_down."""
    rate = rate_up if abs(target) > abs(value) else rate_down
    step = rate * max(dt, 0.0)
    if target > value:
        return min(value + step, target)
    return max(value - step, target)


def clamp_speed(speed) -> int:
    """Motor commands are integers in -100..100."""
    try:
        return int(max(-100, min(100, round(float(speed)))))
    except (TypeError, ValueError):
        return 0


def advance_position(position: float, speed: float, dt: float) -> float:
    """Motor position in degrees after dt seconds at `speed`."""
    return position + speed * DEGREES_PER_SPEED_SECOND * max(dt, 0.0)


def wrap_angle(position: float) -> float:
    """Absolute angle in -180..180 from a running position."""
    return (position + 180.0) % 360.0 - 180.0


def spin_seconds(speed: float):
    """Seconds per revolution for the on-screen blade, or None when stopped."""
    return None if not speed else 60.0 / abs(speed)
