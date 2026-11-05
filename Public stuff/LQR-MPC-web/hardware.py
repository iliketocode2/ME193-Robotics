"""
LEGO Education Double Motor + Color Sensor, driven from the browser.

The `legoeducation` library (1.2.0 and later) has its own Web Bluetooth transport that
switches on inside a browser. Its synchronous API needs a browser feature (stack
switching) that this page's Python does not have, so the page uses the library's
async internals directly: pick a hub, connect, check the firmware, then send
commands without waiting. Only the latest firmware is supported.

On top of that sits `LineFollower`, the one-sensor line follower from the course:
read() gives (e, theta, saturated), drive() takes a speed and a yaw rate.
"""

import asyncio
import math

import legoeducation as le
import legoeducation.basic_device as bd
from legoeducation import ColorSensor as _CS
from legoeducation import DoubleMotor as _DM
from legoeducation.rpc_message import (RPC_VERSION, DeviceNotificationRequest, InfoRequest,
                                       InfoResponse)
from pyscript import window

import lqrmpc as m

# The page sets these: hooks["log"](text) and hooks["changed"]() on (dis)connect.
hooks = {"log": print, "changed": lambda: None}

NOTIFY_MS = 50   # the library default is 100 ms (10 Hz); the control loop wants 20 Hz
UPDATE_URL = "https://code.legoeducation.com/"


def _fire_and_forget(coro):
    """Stand-in for the library's _run_sync, which needs stack switching.

    Every command this page sends is non-blocking, so scheduling it is enough.
    """
    task = asyncio.ensure_future(coro)

    def done(t):
        if not t.cancelled() and t.exception() is not None:
            hooks["log"](f"Bluetooth command failed: {t.exception()}")

    task.add_done_callback(done)
    return None


bd._run_sync = _fire_and_forget


def firmware_status(hub_version, library_version=RPC_VERSION):
    """'ok' for the latest firmware, 'old' if the hub needs updating, 'new' if this page is out of date.

    As in the library: from RPC major 2 on, major.minor must match (the build number is ignored);
    before that, all three parts must match.
    """
    hub_version, library_version = tuple(hub_version), tuple(library_version)
    n = 2 if library_version[0] >= 2 else 3
    if hub_version[:n] == library_version[:n]:
        return "ok"
    return "old" if hub_version < library_version else "new"


def _versions_text(hub, lib):
    return f"hub RPC {'.'.join(map(str, hub))}, this page needs {'.'.join(map(str, lib))}"


class WebDevice:
    """Mixin: connect a library device through the browser's Bluetooth picker."""

    def _init_web(self, label):
        self.label = label
        self.version_rpc = None

    async def connect_web(self):
        """Show the picker, connect, check the firmware, start notifications. True if ready to use."""
        try:
            found = await self.my_worker.scan(0, self._compose_filters())
        except Exception as exc:
            hooks["log"](f"{self.label}: could not open the Bluetooth picker ({exc}).")
            return False
        if not found:
            hooks["log"](f"{self.label}: no device chosen.")
            return False
        device = found[0]
        self.device = device
        try:
            ok = await self._async_connect(device, self._device_callback, self._disconnect_callback)
        except Exception as exc:
            hooks["log"](f"{self.label}: could not connect ({exc}).")
            return False
        if not ok:
            hooks["log"](f"{self.label}: could not connect.")
            return False
        self.connected = True

        # Only the latest firmware is supported.
        try:
            payload = await asyncio.wait_for(
                self._async_send_commands(InfoRequest().Serialize(), blocking=True), timeout=4.0)
            info = InfoResponse.Deserialize(payload)
            hub = (info.rpcMajor, info.rpcMinor, info.rpcBuild)
        except Exception as exc:
            hooks["log"](f"{self.label}: could not read the firmware version ({exc}).")
            await self._drop()
            return False
        self.version_rpc = hub
        status = firmware_status(hub)
        if status != "ok":
            if status == "old":
                text = (f"The {self.label.lower()} has older firmware than this page supports "
                        f"({_versions_text(hub, RPC_VERSION)}).\n\n"
                        f"Please update your hardware at {UPDATE_URL} and connect again.")
            else:
                text = (f"The {self.label.lower()} has newer firmware than this page supports "
                        f"({_versions_text(hub, RPC_VERSION)}).\n\n"
                        f"This page needs an update. Please tell your instructor.")
            hooks["log"](text.replace("\n\n", " "))
            window.alert(text)
            await self._drop()
            return False

        try:
            await self._async_send_commands(DeviceNotificationRequest(NOTIFY_MS).Serialize(), blocking=False)
            await asyncio.wait_for(self._first_notification_ready.wait(), timeout=3.0)
        except Exception:
            hooks["log"](f"{self.label}: no sensor readings yet; they may take a moment to start.")
        hooks["log"](f"{self.label} connected.")
        hooks["changed"]()
        return True

    async def _drop(self):
        self.connected = False
        try:
            await self.my_worker.disconnect(self.device)
        except Exception:
            pass
        hooks["changed"]()

    def disconnect_web(self):
        asyncio.ensure_future(self._drop())

    def _disconnect_callback(self):
        super()._disconnect_callback()
        hooks["log"](f"{self.label} disconnected.")
        hooks["changed"]()


class DoubleMotorDevice(WebDevice, _DM):
    def __init__(self):
        _DM.__init__(self)
        self._init_web("Motors")


class ColorSensorDevice(WebDevice, _CS):
    def __init__(self):
        _CS.__init__(self)
        self._init_web("Sensor")


# -- the line follower ----------------------------------------------------------
DEFAULT_CAL = {
    "reflect_black": 12.0,   # reading fully over the tape
    "reflect_white": 85.0,   # reading fully over the background
    "e_max_mm": 6.0,         # half-width of the sensor's linear region
    "track_width_mm": 96.0,  # distance between wheel contact patches
    "wheel_max_mms": 400.0,  # wheel speed at motor speed 100%
    "dt": 0.05,              # control period
}


def curvature_estimate(omega, v, floor=20.0):
    """kappa ~= omega / v: the stand-in for track preview when there is no map."""
    return omega / max(v, floor)


class LineFollower:
    """Differential-drive robot with one downward-facing light sensor."""

    def __init__(self, motors, sensor, cal):
        self.motors, self.sensor, self.cal = motors, sensor, cal
        self.dt = cal["dt"]
        self._theta = 0.0
        self._e_prev = None
        self._v_prev = 0.0

    def start(self):
        self.motors.movement_set_acceleration(100, 100, blocking=False)
        self.motors.movement_set_end_state(le.MOTOR_END_STATE_BRAKE, blocking=False)

    def raw_reflection(self):
        r = self.sensor.sensor.reflection
        if r is None or (isinstance(r, float) and math.isnan(r)):
            return None
        return float(r)

    def normalized(self):
        """Raw reflection mapped to s in [0, 1]: 0 = fully on the tape, 1 = fully off."""
        raw = self.raw_reflection()
        if raw is None:
            return 0.5
        black, white = self.cal["reflect_black"], self.cal["reflect_white"]
        return min(1.0, max(0.0, (raw - black) / (white - black)))

    def read(self):
        """(e_mm, theta_rad, saturated). e > 0 means drifted off the line, onto the background."""
        e_max = self.cal["e_max_mm"]
        s = self.normalized()
        e = max(-e_max, min(e_max, 2.0 * e_max * (s - 0.5)))
        saturated = s <= 0.02 or s >= 0.98
        # Complementary estimate of theta: predict with the yaw rate we asked for,
        # correct with the observed drift rate e_dot = v * theta. The correction is
        # frozen while the sensor is pinned, because a pinned reading has zero slope.
        if self._e_prev is not None and not saturated and self._v_prev > 1.0:
            theta_meas = (e - self._e_prev) / (self.dt * self._v_prev)
            self._theta = 0.7 * self._theta + 0.3 * theta_meas
        self._e_prev = e
        return e, self._theta, saturated

    def drive(self, v_mms, omega_rads):
        left_mms, right_mms = m.wheel_speeds(v_mms, omega_rads, self.cal["track_width_mm"])
        left = int(round(m.to_percent(left_mms, self.cal["wheel_max_mms"])))
        right = int(round(m.to_percent(right_mms, self.cal["wheel_max_mms"])))
        self.motors.movement_move_tank(left, right, blocking=False)
        self._theta += self.dt * omega_rads
        self._v_prev = v_mms
        return left, right

    def stop(self):
        try:
            self.motors.movement_stop(blocking=False)
        except Exception as exc:
            hooks["log"](f"stop failed: {exc}")

    async def loop(self, duration_s, should_stop=lambda: False):
        """Yield (k, t) once per control period, holding the period as well as BLE allows."""
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        k = 0
        while loop.time() - t0 <= duration_s and not should_stop():
            yield k, loop.time() - t0
            k += 1
            await asyncio.sleep(max(0.0, t0 + k * self.dt - loop.time()))


# -- controllers: (e, theta) -> (v, omega) --------------------------------------
def _mpc_speed(e, theta, kappa_hat, candidates, horizon, dt, e_max, v_ref, qe, qth, r, qv, pen=1e4):
    """The MPC's enumeration, with the curvature held at its current estimate.

    A real robot has no map, so the preview is "whatever turn I am holding now",
    not a true look-ahead. See the MPC notes, "Where does the preview come from?".
    """
    best_cost, best_v = float("inf"), candidates[0]
    for v in candidates:
        k_e, k_th = m.lqr_gains_discrete(float(v), dt, qe, qth, r)
        ep, thp, cost = e, theta, 0.0
        for _ in range(horizon):
            omega = -k_e * m.coerce(ep, e_max) - k_th * thp
            cost += (qe * ep ** 2 + qth * thp ** 2) * dt
            cost += pen * max(0.0, abs(ep) - e_max) ** 2 * dt
            ep, thp = m.step(ep, thp, v, omega, kappa_hat, dt)
        cost += qv * (v - v_ref) ** 2 * horizon * dt
        if cost < best_cost:
            best_cost, best_v = cost, v
    return best_v


def make_controller(name, p, cal):
    """Return ctl(e, theta) -> (v, omega) for 'bangbang', 'pd', 'lqr' or 'mpc'."""
    dt, e_max, v = cal["dt"], cal["e_max_mm"], p["v"]
    if name == "bangbang":
        return lambda e, th: (v, -p["omega_mag"] * (1.0 if e > 0 else -1.0))
    if name == "pd":
        return lambda e, th: (v, -p["kp"] * e - p["kd"] * th)
    if name == "lqr":
        k_e, k_th = m.lqr_gains_discrete(float(v), dt, p["qe"], p["qth"], p["r"])
        return lambda e, th: (v, -k_e * e - k_th * th)
    if name == "mpc":
        cands = tuple(c for c in m.CANDIDATES if c <= cal["wheel_max_mms"] * 0.95) or (v,)
        state = {"kappa": 0.0}

        def ctl(e, th):
            vv = _mpc_speed(e, th, state["kappa"], cands, int(p["n"]), dt, e_max, v,
                            p["qe"], p["qth"], p["r"], p["qv"])
            k_e, k_th = m.lqr_gains_discrete(float(vv), dt, p["qe"], p["qth"], p["r"])
            omega = -k_e * e - k_th * th
            state["kappa"] = 0.8 * state["kappa"] + 0.2 * curvature_estimate(omega, vv)
            return vv, omega
        return ctl
    raise ValueError(name)
