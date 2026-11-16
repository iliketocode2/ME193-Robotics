"""Keeps the page's device dropdowns in step with the connected devices.

Device.py calls refresh_device_dropdowns() whenever a device connects or goes away
(the main page's version of this module refreshes its network nodes instead).

Each hub on the robot is its own Bluetooth device: three Color Sensors and one Double
Motor, plus an optional Controller. Sensors that haven't been assigned yet are filled in
the order they were connected (first = Left, second = Front, third = Right), and the
dropdowns let you swap them.
"""
import state
from state import get_id

# ids are dev-in-<id> because the shared Save power code looks sensors up that way (see state.py)
SENSOR_SELECTS = ("dev-in-left", "dev-in-front", "dev-in-right")
MOTOR_SELECT = "motor-select"


def _options(devs, selected):
    html = '<option value="">- none -</option>'
    for d in devs:
        mark = " selected" if d.name == selected else ""
        html += f'<option value="{d.name}"{mark}>{d.name}</option>'
    return html


def refresh_device_dropdowns():
    sensors = [d for d in state.devices if "Color Sensor" in (d.name or "")]
    motors = [d for d in state.devices if "Double Motor" in (d.name or "")]
    names = {d.name for d in state.devices}

    chosen = {}
    for sel_id in SENSOR_SELECTS:
        sel = get_id(sel_id)
        cur = sel.value if sel else ""
        chosen[sel_id] = cur if cur in names else ""
    used = {v for v in chosen.values() if v}
    for sel_id in SENSOR_SELECTS:          # fill empty slots in connection order
        if chosen[sel_id] == "":
            free = next((d for d in sensors if d.name not in used), None)
            if free:
                chosen[sel_id] = free.name
                used.add(free.name)
    for sel_id in SENSOR_SELECTS:
        sel = get_id(sel_id)
        if sel:
            sel.innerHTML = _options(sensors, chosen[sel_id])
            sel.value = chosen[sel_id]

    sel = get_id(MOTOR_SELECT)
    if sel:
        cur = sel.value if sel.value in names else (motors[0].name if motors else "")
        sel.innerHTML = _options(motors, cur)
        sel.value = cur
