"""
Drive a Single Motor using a Controller's left joystick.

Setup:
    1. python -m venv my_env
    2. my_env\\Scripts\\activate.bat        (Windows)
       source my_env/bin/activate           (macOS/Linux)
    3. pip install --upgrade pip
    4. pip install legoeducation
    5. Copy lelib.py into this project's folder (same directory as this file).
    6. Fill in the MOTOR_* and CONTROLLER_* card color/serial constants below
       with the values printed on your two LEGO Connection Cards.

Behavior:
    Push the left stick up/down to spin the motor forward/backward at a
    speed proportional to how far the stick is pushed. Release the stick
    (center position) to stop the motor. Ctrl+C to exit cleanly.
"""

import time

import legoeducation as le
from lelib import singleMotor, controller

# --- Bluetooth card info for your hardware -------------------------------
# Fill these in with the color/serial printed on your LEGO Connection Cards.
# Valid colors: le.LEGO_COLOR_RED, _YELLOW, _BLUE, _GREEN, _PURPLE,
# _MAGENTA, _AZURE, _ORANGE.
MOTOR_CARD_COLOR = le.LEGO_COLOR_RED
MOTOR_CARD_SERIAL = 3664  # <-- fill in with your motor card's serial number

CONTROLLER_CARD_COLOR = le.LEGO_COLOR_RED
CONTROLLER_CARD_SERIAL = 3664  # <-- fill in with your controller card's serial number

POLL_DELAY_S = 0.05  # seconds between joystick reads


def main():
    motor = singleMotor()
    ctl = controller()

    motor.connect(card_serial=MOTOR_CARD_SERIAL, card_color=MOTOR_CARD_COLOR)
    ctl.connect(card_serial=CONTROLLER_CARD_SERIAL, card_color=CONTROLLER_CARD_COLOR)

    try:
        while motor.connected and ctl.connected:
            if ctl.left_released():
                motor.stop()
            else:
                motor.run(ctl.left_position())  # -100..100, sign sets direction
            time.sleep(POLL_DELAY_S)
    except KeyboardInterrupt:
        pass
    finally:
        motor.stop()
        motor.disconnect()
        ctl.disconnect()


if __name__ == "__main__":
    main()
