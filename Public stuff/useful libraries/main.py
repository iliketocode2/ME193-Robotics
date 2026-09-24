"""
Install first:
    pip install legoeducation
Then copy lelib.py from the SimpleLE repo into this project's folder.

"""

import time

import legoeducation as le
from lelib import colorSensor, controller, doubleMotor, singleMotor

# --- Bluetooth card info for your hardware -------------------------------
# Fill these in with the color/serial printed on your LEGO connection card.
# Valid values: le.LEGO_COLOR_RED, _YELLOW, _BLUE, _GREEN, _PURPLE,
# _MAGENTA, _AZURE, _ORANGE.
COLOR_SENSOR_CARD_COLOR = le.LEGO_COLOR_GREEN
COLOR_SENSOR_CARD_SERIAL = 1126

CONTROLLER_CARD_COLOR = le.LEGO_COLOR_GREEN
CONTROLLER_CARD_SERIAL = 1126

POLL_DELAY_S = 0.1  # seconds between reads
BLUE_OSCILLATION_PERIOD_S = 1.0
STICK_SPEED = 50  # wheel speed when a joystick is pushed up/down

motor = singleMotor()
drive = doubleMotor()
blue_direction = 50
blue_last_switch = time.monotonic()
left_active = False   # is the left stick currently driving the left wheel?
right_active = False  # is the right stick currently driving the right wheel?
last_color = None     # color seen on the previous loop



def beep():
    """Beep the single motor's built-in speaker."""
    motor.beep()



# --- Empty handler functions ----------------------------------------------
# Fill these in with whatever behavior you want.

def DoRed():
    print("red")
    # Beep once when red first appears, not on every 0.1 s poll.
    if last_color != "Red":
        beep()
    motor.run()



def DoYellow():
    print("yellow")
    motor.run(-50)



def DoBlue():
    global blue_direction, blue_last_switch

    print("blue")
    now = time.monotonic()
    if now - blue_last_switch >= BLUE_OSCILLATION_PERIOD_S:
        blue_direction *= -1
        blue_last_switch = now
    motor.run(blue_direction)



def DoTeal():
    print("teal")
    drive.run_left(45)



def DoGreen():
    print("green")
    drive.run_right(45)



def DoPurple():
    print("purple")
    drive.turn_left(90)



def DoWhite():
    print("white")
    motor.stop()
    drive.motor_stop(motor=le.MOTOR_BOTH)



def DoMagenta():
    print("magenta")
    drive.run(50)



def DoOrange():
    print("orange")
    motor.spin(1)



def DoAzure():
    print("azure")
    drive.turn_right(90)



def DoNoColor():
    # Nothing under the sensor -- keep doing whatever the last color started.
    pass



def DoUnknownColor():
    print("unknown color")



def DoLeftUp():
    global left_active

    left_active = True
    drive.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_LEFT, speed=STICK_SPEED)



def DoLeftDown():
    global left_active

    left_active = True
    drive.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_LEFT, speed=-STICK_SPEED)



def DoLeftReleased():
    global left_active

    # Only stop on the up/down -> released transition, so the color
    # handlers can still drive the left wheel while the stick is idle.
    if left_active:
        left_active = False
        drive.motor_stop(motor=le.MOTOR_LEFT)



def DoRightUp():
    global right_active

    right_active = True
    drive.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_RIGHT, speed=STICK_SPEED)



def DoRightDown():
    global right_active

    right_active = True
    drive.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_RIGHT, speed=-STICK_SPEED)



def DoRightReleased():
    global right_active

    if right_active:
        right_active = False
        drive.motor_stop(motor=le.MOTOR_RIGHT)



# --- Dispatch helpers -------------------------------------------------

def handle_color(color_name):
    """Big switch statement on the color sensor's detected color."""
    global last_color

    match color_name:
        case "Red":
            DoRed()
        case "Yellow":
            DoYellow()
        case "Blue":
            DoBlue()
        case "Teal":
            DoTeal()
        case "Green":
            DoGreen()
        case "Purple":
            DoPurple()
        case "White":
            DoWhite()
        case "Magenta":
            DoMagenta()
        case "Orange":
            DoOrange()
        case "Azure":
            DoAzure()
        case "No color":
            DoNoColor()
        case _:
            DoUnknownColor()

    last_color = color_name



def handle_controller(ctl):
    """Big switch statement on the controller's joystick state."""
    if ctl.left_up():
        left_state = "up"
    elif ctl.left_down():
        left_state = "down"
    else:
        left_state = "released"

    if ctl.right_up():
        right_state = "up"
    elif ctl.right_down():
        right_state = "down"
    else:
        right_state = "released"

    match left_state:
        case "up":
            DoLeftUp()
        case "down":
            DoLeftDown()
        case "released":
            DoLeftReleased()

    match right_state:
        case "up":
            DoRightUp()
        case "down":
            DoRightDown()
        case "released":
            DoRightReleased()



# --- Main loop -------------------------------------------------------------

def main():
    sensor = colorSensor()
    sensor.connect(card_serial=COLOR_SENSOR_CARD_SERIAL, card_color=COLOR_SENSOR_CARD_COLOR)

    ctl = controller()
    ctl.connect(card_serial=CONTROLLER_CARD_SERIAL, card_color=CONTROLLER_CARD_COLOR)

    motor.connect(card_serial=CONTROLLER_CARD_SERIAL, card_color=CONTROLLER_CARD_COLOR)
    drive.connect(card_serial=CONTROLLER_CARD_SERIAL, card_color=CONTROLLER_CARD_COLOR)

    try:
        while True:
            handle_color(sensor.detect_color())
            handle_controller(ctl)
            time.sleep(POLL_DELAY_S)
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    main()
