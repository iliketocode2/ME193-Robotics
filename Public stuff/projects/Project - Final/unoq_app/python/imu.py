"""MPU-6050 readings from the sketch -> yaw, pitch, roll, temperature.

The sketch sends Bridge.notify("imu_batch", "ax,ay,az,gx,gy,gz,t;...") ten
times a second, ten 100 Hz samples per batch, all raw sensor counts
(accel +-16 g = 2048 counts/g, gyro +-2000 deg/s = 16.4 counts/(deg/s),
temperature = counts / 340 + 36.53 C), and Bridge.notify("imu_status", ...)
whenever it finds or loses the IMU.

The MPU-6050 has no magnetometer, so yaw is the integrated gyro Z rate:
zero when the app starts (or on reset_yaw), + = counterclockwise seen from
above when the board lies flat, and it drifts slowly (a few degrees a
minute even after bias removal). Pitch and roll come from the gravity
direction (accelerometer), smoothed with a complementary filter, so they
don't drift but only make sense while the board isn't accelerating.
"""

import math
import threading
import time

ACCEL_PER_G = 2048.0
GYRO_PER_DPS = 16.4
DT = 0.01                 # 100 Hz
BIAS_SAMPLES = 200        # 2 s of gyro averaged at start-up for the zero-rate bias
STILL_DPS = 3.0           # a bias window counts as "still" if every axis's std dev is below this
MAX_BIAS_DPS = 20.0       # MPU-6050 zero-rate offset is within about +-20 deg/s
ALPHA = 0.98             # complementary filter: gyro weight for pitch/roll
STALE_S = 1.0             # no batch for this long = IMU considered disconnected


class Imu:
    def __init__(self):
        self._lock = threading.Lock()
        self.status = "waiting for the sketch"
        self._last_batch = 0.0
        self._bias = [0.0, 0.0, 0.0]
        self._bias_acc = [0.0, 0.0, 0.0]
        self._bias_sq = [0.0, 0.0, 0.0]
        self._bias_n = 0
        self.calibrated = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.temperature = float("nan")
        self.accel_g = (0.0, 0.0, 0.0)
        self.gyro_dps = (0.0, 0.0, 0.0)

    # --- Bridge callbacks (called from the Bridge thread) -----------------
    def on_status(self, text):
        with self._lock:
            self.status = str(text)
        print(f"[imu] {text}", flush=True)

    def on_batch(self, text):
        rows = []
        for chunk in str(text).split(";"):
            parts = chunk.split(",")
            if len(parts) == 7:
                try:
                    rows.append([int(p) for p in parts])
                except ValueError:
                    pass
        if not rows:
            return
        with self._lock:
            for ax, ay, az, gx, gy, gz, t in rows:
                self._step(ax / ACCEL_PER_G, ay / ACCEL_PER_G, az / ACCEL_PER_G,
                           gx / GYRO_PER_DPS, gy / GYRO_PER_DPS, gz / GYRO_PER_DPS,
                           t / 340.0 + 36.53)
            self._last_batch = time.monotonic()
            self.status = "ok" if self.calibrated else "calibrating (keep the board still)"

    def _step(self, ax, ay, az, gx, gy, gz, temp_c):
        self.accel_g = (ax, ay, az)
        self.temperature = temp_c
        if not self.calibrated:
            # Average a 2 s window; keep it only if the board was still
            # (small spread on every axis), otherwise start a new window.
            g = (gx, gy, gz)
            self._bias_acc = [s + v for s, v in zip(self._bias_acc, g)]
            self._bias_sq = [s + v * v for s, v in zip(self._bias_sq, g)]
            self._bias_n += 1
            if self._bias_n >= BIAS_SAMPLES:
                n = self._bias_n
                mean = [s / n for s in self._bias_acc]
                std = [math.sqrt(max(0.0, q / n - m * m)) for q, m in zip(self._bias_sq, mean)]
                # Small spread = not shaking; small mean = not turning steadily
                # (a smooth turn would otherwise be baked in as "bias").
                if max(std) < STILL_DPS and max(map(abs, mean)) < MAX_BIAS_DPS:
                    self._bias = mean
                    self.calibrated = True
                    self.yaw = 0.0
                self._bias_acc, self._bias_sq, self._bias_n = [0.0] * 3, [0.0] * 3, 0
            # Until calibrated, pitch/roll still track gravity; yaw stays 0.
        gx, gy, gz = gx - self._bias[0], gy - self._bias[1], gz - self._bias[2]
        self.gyro_dps = (gx, gy, gz)
        if self.calibrated:
            self.yaw += gz * DT
        acc_roll = math.degrees(math.atan2(ay, az))
        acc_pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
        self.roll = ALPHA * (self.roll + gx * DT) + (1 - ALPHA) * acc_roll
        self.pitch = ALPHA * (self.pitch + gy * DT) + (1 - ALPHA) * acc_pitch

    # --- used by the skills ---------------------------------------------
    def connected(self):
        return time.monotonic() - self._last_batch < STALE_S

    def reset_yaw(self):
        with self._lock:
            self.yaw = 0.0

    def snapshot(self):
        """Values for message placeholders, or None if there's no IMU."""
        if not self.connected():
            return None
        with self._lock:
            yaw = (self.yaw + 180.0) % 360.0 - 180.0   # wrap to -180..180
            return {
                "yaw": round(yaw, 1),
                "pitch": round(self.pitch, 1),
                "roll": round(self.roll, 1),
                "temperature": round(self.temperature, 1),
                "accel": "x={:.2f} y={:.2f} z={:.2f} g".format(*self.accel_g),
                "gyro": "x={:.1f} y={:.1f} z={:.1f} deg/s".format(*self.gyro_dps),
                "calibrated": self.calibrated,
            }

    def describe(self):
        s = self.snapshot()          # one snapshot: the IMU can drop out between two calls
        if s is None:
            return f"IMU not connected ({self.status})"
        cal = "" if s["calibrated"] else " (still calibrating)"
        return f"IMU ok{cal}: yaw {s['yaw']} pitch {s['pitch']} roll {s['roll']} deg"
