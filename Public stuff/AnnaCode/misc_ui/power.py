"""Save power for the Color Sensor (PyScript glue).

The sensor's light is a real drain, and the page used to leave it on for as long as the sensor
was connected. With "Save power" ticked (the default) a sensor stays OFF except while data is
being recorded or the network is running, and only if an Input node actually reads it.

The rules are in power_logic.py (unit-tested). This applies them:
  * a once-a-second loop keeps every connected Color Sensor in the mode it should be in, so it
    follows the checkbox, an Input node being pointed at it, and Record/Run starting and stopping;
  * ensure_ready() runs as Record or Run starts: it switches the sensor on and waits until the
    sensor really is reading again, so the first rows and the first motor commands are not based
    on a stale or zero reading from the sleeping sensor;
  * the sensor's chip shows "light off" while it is off.

Cost: with the light off the live reading is blank. Untick Save power to watch the sensor live.
"""
import asyncio
import time

import state
from state import get_id
import Device
import power_logic as pw

TICK_S = 1.0
ENSURE_READY_TIMEOUT_S = 2.0

_loop_running = False


def enabled() -> bool:
    el = get_id("save-power")
    return bool(el.checked) if el else False


def needed() -> bool:
    """Is data being recorded, or the network running?"""
    return bool(state.is_running or state.is_recording)


def _input_device_names() -> set:
    names = set()
    for inp in state.inputs:
        el = get_id(f"dev-in-{inp['id']}")
        if el and el.value:
            names.add(el.value)
    return names


def sensors() -> list:
    """Connected real Color Sensors we can talk to."""
    return [d for d in state.devices
            if isinstance(d, Device.Element) and d.is_color_sensor
            and d.hub is not None and d.link is not None and not Device.link_dropped(d)]


def wanted(dev, used=None) -> int:
    used = _input_device_names() if used is None else used
    return pw.wanted_mode(enabled(), needed(), dev.name in used)


def _show_note(dev):
    chip = get_id(f"chip-{dev.name}")
    note = chip.querySelector(".device-note") if chip else None
    if note:
        text = pw.note_text(pw.effective_mode(dev.mode_requested, dev.sensor_mode))
        if note.textContent != text:
            note.textContent = text


async def reconcile():
    """Bring every connected Color Sensor into the mode it should be in."""
    used = _input_device_names()
    for dev in sensors():
        want = wanted(dev, used)
        now = time.monotonic()
        if dev.mode_requested is None:
            if want == pw.MODE_COLOR:
                # A sensor that has just connected is already in color mode: nothing to send.
                dev.mode_requested = pw.MODE_COLOR
                dev.mode_requested_at = now
            else:
                await dev.set_sensor_mode(want)
        elif dev.mode_requested != want:
            await dev.set_sensor_mode(want)
        elif pw.resend_due(dev.mode_requested, dev.sensor_mode, now - dev.mode_requested_at, dev.mode_retries):
            await dev.set_sensor_mode(want)
        elif (dev.sensor_mode is not None and dev.sensor_mode != dev.mode_requested
              and dev.mode_retries >= pw.MAX_RETRIES and not getattr(dev, "_power_warned", False)):
            dev._power_warned = True
            print(f"{dev.name}: asked {pw.MAX_RETRIES + 1} times but the sensor still reports mode "
                  f"{dev.sensor_mode}; leaving it. (Untick Save power to stop asking.)")
        _show_note(dev)


async def ensure_ready(timeout: float = ENSURE_READY_TIMEOUT_S) -> bool:
    """Call as Record or Run starts. Switches on the sensors that need to be on and waits until
    they are really reading (or the timeout passes, in which case it carries on and says so)."""
    await reconcile()
    used = _input_device_names()
    deadline = time.monotonic() + timeout
    while True:
        now = time.monotonic()
        pending = [d for d in sensors()
                   if wanted(d, used) == pw.MODE_COLOR
                   and not pw.ready(pw.MODE_COLOR, d.sensor_mode, now - d.mode_requested_at)]
        if not pending:
            return True
        if now >= deadline:
            print("The sensor light did not confirm in time for: " + ", ".join(d.name for d in pending)
                  + ". Carrying on; the first readings may be stale.")
            return False
        await asyncio.sleep(0.05)


async def _loop():
    global _loop_running
    try:
        while True:
            try:
                await reconcile()
            except Exception as e:
                print("Save power check failed: " + str(e))
            await asyncio.sleep(TICK_S)
    finally:
        _loop_running = False


def start():
    global _loop_running
    if not _loop_running:
        _loop_running = True
        asyncio.ensure_future(_loop())
