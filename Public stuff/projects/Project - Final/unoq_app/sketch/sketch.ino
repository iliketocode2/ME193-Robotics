// jim -- MCU side.
//
// 1. LED matrix: python/main.py sends whole 8x13 frames with
//    Bridge.call("draw", bytes); loop() keeps redrawing the latest one.
// 2. IMU: samples an MPU-6050 at 100 Hz and sends ten samples at a time with
//    Bridge.notify("imu_batch", "ax,ay,az,gx,gy,gz,t;..."), raw counts
//    (accel +-16 g, gyro +-2000 deg/s, temperature register). Python
//    (imu.py) does the unit conversion and yaw integration.
//    Wiring: Qwiic connector (Wire1) or the SDA/SCL header pins (Wire); the
//    sketch looks for the IMU at 0x68 / 0x69 on both and keeps looking once a
//    second if it's missing, sending Bridge.notify("imu_status", ...).
//    No IMU is fine: everything else works and yaw reads "unknown".
//
// IMU code adapted from Project 11 (Campus Navigator).

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>
#include <vector>

// --- LED matrix -------------------------------------------------------------
Arduino_LED_Matrix matrix;
const uint8_t FRAME_SIZE = 8 * 13;
uint8_t frame[FRAME_SIZE] = {0};

void draw(std::vector<uint8_t> newFrame) {
    size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
    memcpy(frame, newFrame.data(), len);
}

// --- MPU-6050 -----------------------------------------------------------------
const uint8_t MPU_ADDRS[] = {0x68, 0x69};
const unsigned long SAMPLE_US = 10000;      // 100 Hz
const int BATCH_SIZE = 10;
const int MAX_READ_FAILS = 5;

const uint8_t REG_SMPLRT_DIV = 0x19;
const uint8_t REG_CONFIG = 0x1A;
const uint8_t REG_GYRO_CONFIG = 0x1B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_ACCEL_XOUT_H = 0x3B;
const uint8_t REG_PWR_MGMT_1 = 0x6B;

TwoWire *bus = nullptr;
uint8_t addr = 0;
int readFails = 0;
String batch;
int batchCount = 0;
unsigned long nextSampleUs = 0;
unsigned long lastSearchMs = 0;
unsigned long lastMissingMs = 0;

bool writeReg(uint8_t reg, uint8_t value) {
    bus->beginTransmission(addr);
    bus->write(reg);
    bus->write(value);
    return bus->endTransmission() == 0;
}

bool setupImu() {
    return writeReg(REG_PWR_MGMT_1, 0x01)       // wake up, clock = gyro X PLL
        && writeReg(REG_SMPLRT_DIV, 9)          // 1 kHz / (1 + 9) = 100 Hz
        && writeReg(REG_CONFIG, 0x03)           // 44 Hz low-pass filter
        && writeReg(REG_GYRO_CONFIG, 0x18)      // +-2000 deg/s
        && writeReg(REG_ACCEL_CONFIG, 0x18);    // +-16 g
}

bool findImu() {
    TwoWire *buses[] = {&Wire1, &Wire};         // Qwiic first, then header pins
    for (TwoWire *b : buses) {
        b->begin();
        b->setClock(400000);
        for (uint8_t a : MPU_ADDRS) {
            b->beginTransmission(a);
            if (b->endTransmission() == 0) {
                bus = b;
                addr = a;
                delay(100);
                if (setupImu()) {
                    readFails = 0;
                    return true;
                }
            }
        }
    }
    bus = nullptr;
    return false;
}

bool readImu(int16_t out[7]) {
    bus->beginTransmission(addr);
    bus->write(REG_ACCEL_XOUT_H);
    if (bus->endTransmission(false) != 0) return false;
    if (bus->requestFrom(addr, (uint8_t)14) != 14) return false;
    uint8_t raw[14];
    for (int i = 0; i < 14; i++) raw[i] = bus->read();
    // registers: accel xyz (0-5), temperature (6-7), gyro xyz (8-13), big-endian
    const int offsets[7] = {0, 2, 4, 8, 10, 12, 6};
    for (int i = 0; i < 7; i++) out[i] = (int16_t)((raw[offsets[i]] << 8) | raw[offsets[i] + 1]);
    return true;
}

void serviceImu() {
    if (bus == nullptr) {
        if (millis() - lastSearchMs >= 1000) {
            lastSearchMs = millis();
            if (findImu()) {
                Bridge.notify("imu_status", String("found MPU-6050"));
                lastMissingMs = 0;
                nextSampleUs = micros();
            } else if (millis() - lastMissingMs >= 10000) {
                // Repeated every 10 s: Python may not have been listening yet
                // the first time (the sketch starts before the app does).
                Bridge.notify("imu_status", String("no MPU-6050 on Qwiic or SDA/SCL"));
                lastMissingMs = millis();
            }
        }
        return;
    }
    if ((long)(micros() - nextSampleUs) < 0) return;
    nextSampleUs += SAMPLE_US;

    int16_t v[7];
    if (!readImu(v)) {
        if (++readFails >= MAX_READ_FAILS) {
            bus = nullptr;
            batch = "";
            batchCount = 0;
            Bridge.notify("imu_status", String("lost the MPU-6050"));
        }
        return;
    }
    readFails = 0;
    for (int i = 0; i < 7; i++) {
        batch += v[i];
        batch += (i < 6) ? ',' : ';';
    }
    if (++batchCount >= BATCH_SIZE) {
        Bridge.notify("imu_batch", batch);
        batch = "";
        batchCount = 0;
    }
}

// --- main ---------------------------------------------------------------------
unsigned long lastDrawMs = 0;

void setup() {
    matrix.begin();
    matrix.setGrayscaleBits(3);
    matrix.clear();
    Bridge.begin();
    // provide_safe runs draw() between loop() iterations, so a frame is never
    // overwritten halfway through matrix.draw() (same as Project 9's sketch).
    Bridge.provide_safe("draw", draw);
    batch.reserve(BATCH_SIZE * 48);
    nextSampleUs = micros();
}

void loop() {
    serviceImu();
    // Redraw at ~100 Hz without blocking, so IMU sampling stays on time.
    if (millis() - lastDrawMs >= 10) {
        lastDrawMs = millis();
        matrix.draw(frame);
    }
}
