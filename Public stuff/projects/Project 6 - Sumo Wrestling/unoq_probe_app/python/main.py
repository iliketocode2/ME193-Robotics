"""
BLE probe as an App Lab app: can the UNO Q's App Lab container reach the LEGO
devices over Bluetooth? Upload this whole folder as its own app in App Lab,
start it, and open the app's logs (Python console). Every step prints
PASS/FAIL, and the summary repeats every 10 s so it's easy to find.

It does NOT need command-line arguments (App Lab doesn't pass any).

    DRIVE_TEST = True -> after a successful scan it connects the double motor
    and runs both wheels at 30% for 1 s. Put the robot on blocks first!
"""

import asyncio
import glob
import os
import sys
import time
import traceback

from arduino.app_utils import App

DRIVE_TEST = True
COLOR_TEST = True
SCAN_SECONDS = 8.0
MOTOR_CARD_SERIAL = None        # None = first double motor found; or the number on its card, e.g. 3664
COLOR_CARD_SERIAL = None        # None = first color sensor found

LEGO_COMPANY_ID = 0x0397
SERVICE_UUID = "0000fd02-0000-1000-8000-00805f9b34fb"
PRODUCTS = {512: "single motor", 513: "double motor", 514: "color sensor", 515: "controller"}

results = []


def log(msg):
    print(msg, flush=True)      # flush: container stdout is buffered, prints can vanish otherwise


def step(name, ok, detail=""):
    results.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}".rstrip())
    log(f"[probe] {results[-1]}")
    return ok


def check_environment():
    step("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    try:
        import legoeducation as le
        import bleak
        step("packages", True, f"legoeducation {getattr(le, '__version__', '?')}, bleak {getattr(bleak, '__version__', '?')}")
    except ImportError as e:
        return step("packages", False, f"{e} -> add it to python/requirements.txt")
    # BlueZ talks over the system D-Bus socket; the adapter shows up in sysfs
    dbus = os.path.exists("/run/dbus/system_bus_socket") or os.path.exists("/var/run/dbus/system_bus_socket")
    step("D-Bus system socket visible in container", dbus,
         "" if dbus else "-> container can't talk to BlueZ (Bluetooth daemon)")
    adapters = [os.path.basename(p) for p in glob.glob("/sys/class/bluetooth/hci*")]
    step("Bluetooth adapter visible", bool(adapters), ", ".join(adapters) or "no /sys/class/bluetooth/hci*")
    return True


async def _scan():
    from bleak import BleakScanner
    found = {}

    def on_adv(device, adv):
        data = adv.manufacturer_data.get(LEGO_COMPANY_ID)
        if not data or len(data) < 5:
            return
        product = (data[0] << 8) | data[1]
        found[device.address] = (PRODUCTS.get(product, f"product {product}"), data[2],
                                 data[3] | (data[4] << 8), adv.rssi)

    async with BleakScanner(detection_callback=on_adv, service_uuids=[SERVICE_UUID]):
        await asyncio.sleep(SCAN_SECONDS)
    return found


def check_scan():
    log(f"[probe] scanning {SCAN_SECONDS:.0f} s for LEGO devices (turn them on, light blinking)...")
    try:
        found = asyncio.run(_scan())
    except Exception as e:
        step("BLE scan runs", False, f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        return None
    step("BLE scan runs", True)
    if not step("LEGO devices seen", bool(found), f"{len(found)} found"):
        return None
    for addr, (kind, fw_color, serial, rssi) in found.items():
        log(f"[probe]     {kind:13s} {addr}  card_serial={serial}  fw_color={fw_color}  rssi={rssi}")
    return found


def check_motor():
    from lelib import doubleMotor
    motor = doubleMotor()
    try:
        log("[probe] connecting double motor (lelib, up to 5 tries)...")
        motor.connect(card_serial=MOTOR_CARD_SERIAL)
        step("double motor connects", motor.connected)
        log("[probe] driving both wheels 30% for 1 s")
        motor.movement_move_tank(30, 30)
        time.sleep(1.0)
        motor.movement_stop()
        step("double motor drives", True, "did the wheels turn?")
    except Exception as e:
        step("double motor", False, f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
    finally:
        if motor.connected:
            motor.movement_stop()
            motor.disconnect()


def check_color():
    from lelib import colorSensor
    sensor = colorSensor()
    try:
        log("[probe] connecting color sensor...")
        sensor.connect(COLOR_CARD_SERIAL)
        step("color sensor connects", sensor.connected)
        for _ in range(5):
            time.sleep(0.4)
            log(f"[probe]     color={sensor.detect_color()}  reflection={sensor.reflection()}")
    except Exception as e:
        step("color sensor", False, f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
    finally:
        if sensor.connected:
            sensor.disconnect()


def run_probe():
    log("[probe] ===== Sumo BLE probe starting =====")
    if not check_environment():
        return
    found = check_scan()
    if not found:
        return
    kinds = {kind for kind, *_ in found.values()}
    if DRIVE_TEST and "double motor" in kinds:
        check_motor()
    if COLOR_TEST and "color sensor" in kinds:
        check_color()


try:
    run_probe()
except Exception as e:
    step("probe crashed", False, f"{type(e).__name__}: {e}")
    log(traceback.format_exc())


def loop():
    log("[probe] ===== SUMMARY =====")
    for line in results:
        log(f"[probe] {line}")
    time.sleep(10)


App.run(user_loop=loop)
