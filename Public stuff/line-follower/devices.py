"""One LEGO hub over Web Bluetooth: a Double Motor or a Color Sensor (used as a light sensor).

Device("Color Sensor") is only what the page *expects*; after connect() `kind` is what was
actually picked.

    dev = Device("Color Sensor")          # or "Double Motor"
    ok = await dev.connect()              # opens Chrome's picker, sets everything up
    dev.reflection                        # latest reflected light, 0-100 (Color Sensor)
    dev.left, dev.right                   # latest motor speeds, forward = + (Double Motor)
    dev.drive(40, 40)                     # left, right speed in percent (Double Motor)
    dev.stop()

`dev.battery` is the hub's battery percent (None until it reports).
`dev.on_data` is called after every notification, `dev.on_drop` if the link is lost.
"""

import asyncio
import struct
import time
from collections import deque

import legoeducation as le
from legoeducation import rpc_message as rm
from pyscript import window
from pyscript.ffi import create_proxy
from pyscript.js_modules import ble

import webhub

SERVICE_UUID = "0000fd02-0000-1000-8000-00805f9b34fb"
WRITE_UUID = "0000fd02-0001-1000-8000-00805f9b34fb"
NOTIFY_UUID = "0000fd02-0002-1000-8000-00805f9b34fb"

NOTIFY_MS = 20          # the library default is 100 ms; 15 is the minimum
SENSOR_LIGHT_ON = 1     # color-sensor mode 1 = LED on (reflection mode)

_CLASSES = {"Double Motor": le.DoubleMotor, "Color Sensor": le.ColorSensor}


def _classify(data: bytes):
    """Device type name from one raw BLE message, or None."""
    if len(data) < 3 or data[0] != rm.DEVICE_NOTIFICATION:
        return None
    payload = data[1:]
    (n,) = struct.unpack("<H", payload[:2])
    note = rm.DeviceNotification(deviceDataByteLength=n, deviceData=payload[2:2 + n])
    items = rm.device_notification_parser(note)
    if any(isinstance(i, rm.ColorSensorNotification) for i in items):
        return "Color Sensor"
    motors = {i.motorBitMask for i in items if isinstance(i, rm.MotorNotification)}
    if len(motors) >= 2:
        return "Double Motor"
    return None


class Device:
    def __init__(self, kind: str):
        self.kind = kind
        self.myble = ble.BLEDevice.new()
        self.hub = None
        self.name = ""
        self.connected = False
        self.on_data = None
        self.on_drop = None
        self._info_default = None
        self.notify_ms = NOTIFY_MS     # how often the hub is asked to report
        self._arrivals = deque()       # recent notification times, for the measured rate
        self.battery = None          # percent, from the hub's info notification
        self.reflection = None
        self.gyro_z = None             # yaw rate from the Double Motor's IMU, degrees per second
        self.left = None
        self.right = None
        self._mode_asked = 0.0
        self._moving = False

    # -- connecting -----------------------------------------------------------------------
    async def _probe(self, timeout=4.0):
        """Chrome sometimes hides the device name: connect and tell from its data instead.
        Leaves the GATT link up and returns the type name, or None."""
        found = []

        def on_raw(data):
            try:
                kind = _classify(bytes(data.to_py()))
            except Exception:
                return
            if kind and not found:
                found.append(kind)

        proxy = create_proxy(on_raw)
        try:
            self.myble.callback = proxy
            if not await self.myble.connect(SERVICE_UUID, WRITE_UUID, NOTIFY_UUID):
                return None
            request = list(rm.DeviceNotificationRequest(100).Serialize())
            for tick in range(int(timeout / 0.05)):
                if found:
                    break
                if tick % 20 == 0:
                    await self.myble.send(request)
                await asyncio.sleep(0.05)
            return found[0] if found else None
        except Exception as exc:
            print(f"Probe failed: {exc}")
            return None
        finally:
            self.myble.callback = None
            proxy.destroy()

    async def connect(self) -> bool:
        """Pick and set up the hub. True once it is streaming data."""
        try:
            await self.myble.scan()
        except Exception as exc:
            print(f"Scan cancelled or failed: {exc}")
            return False

        raw_name = str(self.myble.name).strip()
        probed = False
        if raw_name:
            found = next((k for k in _CLASSES if k in raw_name), None)
        else:
            found = await self._probe()
            probed = found is not None
            raw_name = found or ""
        if found is None:
            window.alert("Could not tell what kind of device that is. Please pick a Double Motor or a Color Sensor.")
            self.myble.disconnect()
            return False
        # Whatever was picked is what this object becomes: the page puts it in the matching slot,
        # so picking the "wrong" device from a Connect button is fine.
        self.kind = found
        self.name = raw_name

        try:
            self.hub = _CLASSES[self.kind]()
            hub = self.hub
            self._notify_proxy = create_proxy(
                lambda data: asyncio.ensure_future(hub._device_callback(NOTIFY_UUID, bytes(data.to_py()))))
            self.myble.callback = self._notify_proxy
            if not probed and not await self.myble.connect(SERVICE_UUID, WRITE_UUID, NOTIFY_UUID):
                print("BLE connect() returned false")
                self.hub = None
                return False
            self._drop_proxy = create_proxy(self._on_ble_disconnect)
            self.myble.disconnectCallback = self._drop_proxy

            self._info_default = hub.info_device   # placeholder until the hub reports
            hub.device = hub
            webhub.registry[id(hub)] = self.myble
            hub._first_notification_ready = asyncio.Event()
            hub.connected = True
            await asyncio.sleep(0)

            if not await webhub.check_firmware(hub, self.kind):
                await self.disconnect()
                return False

            # the hub sometimes ignores the first request: ask up to 3 times
            for _ in range(3):
                hub.device_notification_request(self.notify_ms, blocking=False)
                waited = 0.0
                while not hub._first_notification_ready.is_set() and waited < 2.0:
                    await asyncio.sleep(0.05)
                    waited += 0.05
                if hub._first_notification_ready.is_set():
                    break
            if not hub._first_notification_ready.is_set():
                print("Warning: no data from the hub yet")

            hub.set_notification_callback(self._notification)
            self._sync_battery()     # the report sent at connect arrived before our callback
            if self.kind == "Color Sensor":
                self._ensure_light()
            self.connected = True
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            await self.disconnect()
            return False

    def _on_ble_disconnect(self, evt=None):
        was = self.connected
        self.connected = False
        if was and self.on_drop:
            self.on_drop(self)

    async def disconnect(self):
        if self.myble is not None:
            self.myble.disconnectCallback = None
        if self.hub is not None:
            if self.kind == "Double Motor" and self.connected:
                self.stop()
                await asyncio.sleep(0.2)         # let the stop command out before the link closes
            try:
                self.hub.disconnect()
            except Exception:
                pass
            webhub.registry.pop(id(self.hub), None)
        self.connected = False
        self.hub = None
        self.myble = ble.BLEDevice.new()

    # -- data -----------------------------------------------------------------------------
    def set_rate(self, milliseconds):
        """Ask the hub to report every `milliseconds` (15 is the library's minimum)."""
        self.notify_ms = int(milliseconds)
        if self.hub is not None and self.connected:
            self.hub.device_notification_request(self.notify_ms, blocking=False)

    def measured_hz(self):
        """Notifications per second actually arriving over the last two seconds, or None if
        there are too few to say (nothing is arriving, or it has only just connected)."""
        now = time.monotonic()
        while self._arrivals and now - self._arrivals[0] > 2.0:
            self._arrivals.popleft()
        if len(self._arrivals) < 3 or now - self._arrivals[-1] > 1.0:
            return None
        return (len(self._arrivals) - 1) / (self._arrivals[-1] - self._arrivals[0])

    def _ensure_light(self):
        """The sensor's LED must be on for reflection readings; ask again if it isn't."""
        import time
        if time.monotonic() - self._mode_asked > 1.0:
            self._mode_asked = time.monotonic()
            self.hub.set_color_sensor_mode(SENSOR_LIGHT_ON, blocking=False)

    def _sync_battery(self):
        """The library keeps the hub's latest battery report on hub.info_device (and handles it
        before our callback runs, so the callback never sees it). Until the hub has reported,
        that attribute still holds the library's placeholder."""
        info = getattr(self.hub, "info_device", None)
        if info is not None and info is not self._info_default:
            self.battery = info.batteryLevel

    def _notification(self, data):
        self._arrivals.append(time.monotonic())
        self._sync_battery()
        for item in le.device_notification_parser(data):
            if isinstance(item, le.InfoDeviceNotification):
                self.battery = item.batteryLevel
            elif isinstance(item, le.ImuDeviceNotification):
                self.gyro_z = item.gyroscopeZ
            elif isinstance(item, le.ColorSensorNotification):
                self.reflection = item.reflection
                if getattr(item, "colorSensorMode", SENSOR_LIGHT_ON) != SENSOR_LIGHT_ON:
                    self._ensure_light()
            elif isinstance(item, le.MotorNotification) and self.kind == "Double Motor":
                # the two motors face opposite ways: the library reads forward as + on the
                # right and - on the left, so flip the left to make forward + on both
                if item.motorBitMask == le.MOTOR_BITS_LEFT:
                    self.left = -item.speed
                else:
                    self.right = item.speed
        if self.on_data:
            self.on_data(self)

    # -- motors ---------------------------------------------------------------------------
    def drive(self, left, right):
        """Left and right speed in percent, -100..100, forward = +."""
        if self.kind == "Double Motor" and self.hub is not None and self.connected:
            self._moving = True
            self.hub.movement_move_tank(int(left), int(right), blocking=False)

    def stop(self):
        """Stop both motors. Drive commands are 'movement' commands, so they are stopped with
        movement_stop (motor_stop alone may not end them); motor_stop is sent too, and both are
        sent again 0.15 s and 0.4 s later in case a Bluetooth write was lost."""
        if self.kind != "Double Motor" or self.hub is None:
            return
        self._moving = False
        self._send_stop()
        asyncio.ensure_future(self._repeat_stop())

    def _send_stop(self):
        webhub.flush_pending_commands()          # drop drive commands still waiting to be sent
        self.hub.movement_stop(blocking=False)
        self.hub.motor_stop(motor=le.MOTOR_BOTH, blocking=False)

    async def _repeat_stop(self):
        for delay in (0.15, 0.4):
            await asyncio.sleep(delay)
            if self._moving or self.hub is None or not self.connected:
                return                           # driving again, or gone
            self._send_stop()
