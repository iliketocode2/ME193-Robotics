"""
Browser Bluetooth for the `legoeducation` library, 1.2.0 or later, using ble.js.

The library's synchronous API needs a browser feature (stack switching) that these
pages' Python does not have, and it sends through a "transport". This module:

  * registers a transport that writes through the page's ble.js `BLEDevice` objects,
  * makes the library's `_run_sync` schedule commands instead of waiting for them,
  * checks a hub's firmware once it is connected. Only the latest firmware is supported:
    a hub with older firmware gets an alert to update at code.legoeducation.com and is
    disconnected.

A page registers each hub's ble.js object with `registry[id(hub)] = js_ble`, then calls
`await check_firmware(hub, label)` after the connection is up and the hub's notification
callback is attached.
"""

import asyncio

import legoeducation.basic_device as bd
from legoeducation.ble_transport import BLETransport, register_transport
from legoeducation.rpc_message import RPC_VERSION, InfoRequest, InfoResponse
from pyscript import window

UPDATE_URL = "https://code.legoeducation.com/"
log = print            # a page may replace this to show messages in its own log

registry = {}          # id(hub) -> ble.js BLEDevice


# -- sending -------------------------------------------------------------------------------
class JsBleTransport(BLETransport):
    """Writes one command at a time to the hub's ble.js object.

    Commands wait their turn. `flush()` drops the ones still waiting, which a page uses
    when a slow connection has let stale speed commands pile up.
    """

    instance = None

    def __init__(self, shutdown_callback=None):
        super().__init__(shutdown_callback)
        self._lock = asyncio.Lock()
        self._generation = 0
        self.waiting = 0
        JsBleTransport.instance = self

    async def scan_devices(self, timeout, filters=None):   # the pages pick devices with ble.js
        return None

    async def connect(self, device, notification_callback, disconnect_callback):
        return True

    async def send(self, device, message):
        generation = self._generation
        self.waiting += 1
        try:
            async with self._lock:
                if generation != self._generation:        # flushed while waiting
                    return
                js_ble = registry.get(id(device))
                if js_ble is not None and message:
                    try:
                        await js_ble.send(list(message))
                    except Exception as exc:
                        log(f"BLE send error: {exc}")
        finally:
            self.waiting -= 1

    async def device_disconnect(self, device):
        js_ble = registry.pop(id(device), None)
        if js_ble is not None:
            js_ble.disconnect()

    def shutdown_all(self):
        for key in list(registry):
            registry.pop(key).disconnect()

    def flush(self):
        self._generation += 1


register_transport(JsBleTransport)


def pending_commands():
    t = JsBleTransport.instance
    return t.waiting if t is not None else 0


def flush_pending_commands():
    """Drop commands still waiting to be written (call right before a stop command)."""
    if JsBleTransport.instance is not None:
        JsBleTransport.instance.flush()


def drop_stale_commands(max_pending=10):
    """Flush only when the queue has backed up."""
    if pending_commands() > max_pending:
        flush_pending_commands()


def _fire_and_forget(coro):
    """Stand-in for the library's _run_sync: schedule the command and carry on."""
    task = asyncio.ensure_future(coro)

    def done(t):
        if not t.cancelled() and t.exception() is not None:
            log(f"Bluetooth command failed: {t.exception()}")

    task.add_done_callback(done)
    return None


bd._run_sync = _fire_and_forget


# -- firmware --------------------------------------------------------------------------------
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


def _dotted(version):
    return ".".join(map(str, version))


async def check_firmware(hub, label="The device", timeout=4.0):
    """True if the hub runs the latest firmware. Otherwise alert the user and return False.

    Call after the connection is up and `hub._device_callback` is receiving data. The caller
    should then disconnect the hub.
    """
    try:
        payload = await asyncio.wait_for(
            hub._async_send_commands(InfoRequest().Serialize(), blocking=True), timeout=timeout)
        info = InfoResponse.Deserialize(payload)
        version = (info.rpcMajor, info.rpcMinor, info.rpcBuild)
    except Exception as exc:
        log(f"{label}: could not read the firmware version ({exc}).")
        window.alert(f"Could not check the firmware on the {label.lower()}.\n\n"
                     f"Please make sure it is up to date at {UPDATE_URL} and connect again.")
        return False
    hub.version_rpc = version
    status = firmware_status(version)
    if status == "ok":
        return True
    if status == "old":
        text = (f"The {label.lower()} has older firmware than this page supports "
                f"(hub RPC {_dotted(version)}, this page needs {_dotted(RPC_VERSION)}).\n\n"
                f"Please update your hardware at {UPDATE_URL} and connect again.")
    else:
        text = (f"The {label.lower()} has newer firmware than this page supports "
                f"(hub RPC {_dotted(version)}, this page needs {_dotted(RPC_VERSION)}).\n\n"
                f"This page needs an update. Please tell your instructor.")
    log(text.replace("\n\n", " "))
    window.alert(text)
    return False
