"""
ble_probe_unoq.py -- STEP 1: can this machine reach the LEGO devices over BLE?

Run it in each place the relay might live, in this order, and stop at the
first one that passes:
  a) inside the App Lab container: VS Code "Python in UNO Q container"
     (while the Sumo Robot app is running), then  exec(open("ble_probe_unoq.py").read())
  b) on the UNO Q's Linux host over SSH:
        cd ~/ArduinoApps/<sumo app>/python
        python3 -m pip install --user legoeducation==1.1.1 paho-mqtt   (once)
        python3 ble_probe_unoq.py --drive
  c) on the laptop (fallback):  my_env\\Scripts\\python unoq_app\\python\\ble_probe_unoq.py --drive

Checks, in order:
  1. Python >= 3.11 (legoeducation's requirement)
  2. bleak imports and a scan works (needs a BLE adapter + BlueZ/D-Bus on Linux)
  3. LEGO Education devices are seen (manufacturer id 0x0397) and their card
     color/serial are printed -- copy those into sumo_topics.py
  4. --drive: connect the double motor with lelib, run both wheels at 30% for
     1 s, stop, disconnect. Put the robot on its back/blocks first!
  5. --color: connect the color sensor and print readings for 10 s
     (both flags together test both:  ble_probe_unoq.py --drive --color)
"""

import asyncio
import sys

LEGO_COMPANY_ID = 0x0397
SERVICE_UUID = "0000fd02-0000-1000-8000-00805f9b34fb"
PRODUCTS = {512: "single motor", 513: "double motor", 514: "color sensor", 515: "controller"}


def check_python():
    ok = sys.version_info >= (3, 11)
    print(f"[1] Python {sys.version.split()[0]} -> {'OK' if ok else 'TOO OLD (need 3.11+)'}")
    return ok


async def scan(seconds=8.0):
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
        await asyncio.sleep(seconds)
    return found


def check_scan():
    try:
        import bleak
    except ImportError:
        print("[2] bleak not installed -> pip install legoeducation==1.1.1 (pulls bleak in)")
        return False
    print(f"[2] bleak {getattr(bleak, '__version__', '?')} imported; scanning 8 s...")
    try:
        found = asyncio.run(scan())
    except Exception as e:
        print(f"[2] SCAN FAILED: {type(e).__name__}: {e}")
        print("    In the App Lab container this usually means no access to the host's")
        print("    Bluetooth/D-Bus -> use option (b) or (c) from the docstring.")
        return False
    if not found:
        print("[3] No LEGO devices seen. Are they on (light blinking) and not connected elsewhere?")
        return False
    print("[3] LEGO devices seen (raw firmware card color; serial is the number on the card):")
    for addr, (kind, fw_color, serial, rssi) in found.items():
        print(f"    {kind:13s} {addr}  card_serial={serial}  fw_color={fw_color}  rssi={rssi}")
    return True


def drive_test():
    import time
    from lelib import doubleMotor
    from sumo_topics import MOTOR_CARD_COLOR, MOTOR_CARD_SERIAL
    motor = doubleMotor()
    try:
        motor.connect(card_serial=MOTOR_CARD_SERIAL, card_color=MOTOR_CARD_COLOR)
        print("[4] double motor connected -> 30% forward for 1 s")
        motor.movement_move_tank(30, 30)
        time.sleep(1.0)
    finally:
        if motor.connected:
            motor.movement_stop()
            motor.disconnect()
    print("[4] drive test done")


def color_test(seconds=10):
    """Connect the color sensor with lelib and print readings for `seconds`.
    Hold it over the floor, then over the blue tape, and watch 'color' change."""
    import time
    from lelib import colorSensor
    from sumo_topics import COLOR_CARD_COLOR, COLOR_CARD_SERIAL
    sensor = colorSensor()
    try:
        sensor.connect(COLOR_CARD_SERIAL, card_color=COLOR_CARD_COLOR)
        print(f"[5] color sensor connected -> printing readings for {seconds} s")
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            print(f"    color={sensor.detect_color():8s} reflection={sensor.reflection()}  rgb={sensor.raw_rgb()}")
            time.sleep(0.5)
    finally:
        if sensor.connected:
            sensor.disconnect()
    print("[5] color test done")


if __name__ == "__main__":
    if check_python() and check_scan():
        if "--drive" in sys.argv:
            drive_test()
        if "--color" in sys.argv:
            color_test()
