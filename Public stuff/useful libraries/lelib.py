'''
Shared SimpleLE wrapper for all ME193 examples.

Usage:
    from lelib import singleMotor, doubleMotor, colorSensor, controller
'''

import time
import legoeducation as le


class _CardReader:
    """Mixin that adds card-tap reading to any LEGO Education device."""
    _last_card_serial = None

    def _connect_with_retry(self, device_label, card_serial, card_color=None):
        """Retry connect() on a transient BLE failure.

        legoeducation's own connect() doesn't raise on a failed BLE link --
        it logs an error and returns with self.connected still False (see
        basic_device.py). So retrying only on a raised "not ready"
        exception (the original approach here) never actually retries that
        failure mode; this also retries whenever self.connected is still
        False after a clean return, which is what a flaky/stale BLE
        connection (e.g. a Windows GATT-cache race after an earlier
        ungraceful disconnect) looks like in practice.
        """
        for attempt in range(5):
            try:
                super(_CardReader, self).connect(card_color=card_color, card_serial=card_serial)
            except Exception as e:
                if "not ready" in str(e).lower() and attempt < 4:
                    time.sleep(1)
                    continue
                raise
            if self.connected:
                return
            if attempt < 4:
                time.sleep(1)
        if not self.connected:
            raise ConnectionError(f'Error connecting to {device_label}.')

    def card_serial(self):
        """Return the serial number of the card currently on the sensor (0 = no card)."""
        return self.scanned_card.serial

    def card_tapped(self):
        """Return serial number when a new card is placed, else None.
        Tracks state internally so consecutive calls only fire once per tap."""
        current = self.scanned_card.serial
        if current != self._last_card_serial:
            self._last_card_serial = current
            if current != 0:
                return current
        return None


class singleMotor(_CardReader, le.SingleMotor):
    def __init__(self):
        super().__init__()

    def connect(self, card_serial, card_color=None):
        self._connect_with_retry('Single Motor', card_serial, card_color)

    def spin(self, rotations=1):
        self.motor_run_for_degrees(rotations * 360)

    def stop(self):
        self.motor_stop()

    def set_speed(self, speed):
        self.motor_set_speed(speed)

    def run(self, speed=50):
        if speed >= 0:
            self.motor_set_speed(speed)
            self.motor_run()
        else:
            self.motor_set_speed(-speed)
            self.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)


class doubleMotor(_CardReader, le.DoubleMotor):

    def connect(self, card_serial=None, card_color=None):
        """Connect to a Double Motor. With no arguments, connects to the
        first advertising Double Motor found (no Connection Card needed)."""
        self._connect_with_retry('Double Motor', card_serial, card_color)

    def move_steps(self, step=1):
        self.movement_move_for_degrees(-180 * step)

    def run(self, speed=50):
        if speed >= 0:
            self.movement_set_speed(speed)
            self.movement_move(direction=le.MOVEMENT_MOVE_DIRECTION_FORWARD)
        else:
            self.movement_set_speed(-speed)
            self.movement_move(direction=le.MOVEMENT_MOVE_DIRECTION_BACKWARD)

    def run_time(self, time=2000):
        self.movement_move_for_time(time)

    def run_left(self, degrees=None):
        if degrees is None:
            self.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_LEFT)
        else:
            self.motor_run_for_degrees(degrees=degrees, direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_LEFT)

    def run_right(self, degrees=None):
        if degrees is None:
            self.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_RIGHT)
        else:
            self.motor_run_for_degrees(degrees=degrees, direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_RIGHT)

    def turn_left(self, degrees=90):
        self.movement_turn_for_degrees(degrees, direction=le.MOVEMENT_TURN_DIRECTION_LEFT)

    def turn_right(self, degrees=90):
        self.movement_turn_for_degrees(degrees, direction=le.MOVEMENT_TURN_DIRECTION_RIGHT)

    def set_speed(self, speed):
        self.motor_set_speed(speed, motor=le.MOTOR_LEFT)
        self.motor_set_speed(speed, motor=le.MOTOR_RIGHT)
        self.movement_set_speed(speed)

    def set_speed_left(self, speed):
        self.motor_set_speed(speed, motor=le.MOTOR_LEFT)

    def set_speed_right(self, speed):
        self.motor_set_speed(speed, motor=le.MOTOR_RIGHT)

    def stop(self):
        self.motor_stop()

    # ── IMU convenience methods ──────────────────────────────────────────────

    def reset_heading(self):
        """Zero the yaw at the current orientation (call once before driving)."""
        self.imu_reset_yaw_axis(0)

    def yaw(self):
        """Yaw in degrees since the last reset_heading() call.
        Positive = clockwise (drifted right), negative = counter-clockwise."""
        return float(self.imu_device.yaw)

    def gyro_z(self):
        """Z-axis angular velocity (raw int16). Positive = rotating clockwise."""
        return float(self.imu_device.gyroscopeZ)


class controller(_CardReader, le.Controller):

    def connect(self, card_serial, card_color=None):
        self._connect_with_retry('Controller', card_serial, card_color)

    def left_up(self):        return self.sensor.leftPercent > 0
    def left_down(self):      return self.sensor.leftPercent < 0
    def left_released(self):  return self.sensor.leftPercent == 0
    def right_up(self):       return self.sensor.rightPercent > 0
    def right_down(self):     return self.sensor.rightPercent < 0
    def right_released(self): return self.sensor.rightPercent == 0
    def left_position(self):  return self.sensor.leftPercent
    def right_position(self): return self.sensor.rightPercent

    def drive(self, dm, t=100):
        for i in range(t):
            dm.movement_move_tank(self.left_position(), self.right_position())
            time.sleep(0.1)


class colorSensor(_CardReader, le.ColorSensor):
    def __init__(self):
        super().__init__()

    def connect(self, card_serial, card_color=None):
        self._connect_with_retry('Color Sensor', card_serial, card_color)

    def reflection(self):
        return self.sensor.reflection

    def detect_color(self):
        color_mapping = {
            0: 'No color', 1: 'Red',     2: 'Yellow', 3: 'Blue',
            4: 'Teal',     5: 'Green',   6: 'Purple', 7: 'White',
            8: 'Magenta',  9: 'Orange', 10: 'Azure'
        }
        return color_mapping.get(self.sensor.color, 'Unknown')

    def raw_rgb(self):
        """Raw red, green, blue as 16-bit integers (0–65535)."""
        return (self.sensor.rawRed, self.sensor.rawGreen, self.sensor.rawBlue)

    def raw_reading(self):
        """Full sensor snapshot for classification.
        Returns a dict with all channels normalized to 0–1."""
        MAX16 = 65535.0
        MAX8  =   255.0
        return {
            "rawRed":     self.sensor.rawRed     / MAX16,
            "rawGreen":   self.sensor.rawGreen   / MAX16,
            "rawBlue":    self.sensor.rawBlue    / MAX16,
            "reflection": self.sensor.reflection / MAX8,
            "hue":        self.sensor.hue        / MAX16,
            "saturation": self.sensor.saturation / MAX8,
            "value":      self.sensor.value      / MAX8,
        }
