// Activity wearable -- MCU side: samples the MPU-6050 IMU at 100 Hz.
//
// Every 10 samples (100 ms) it sends one batch to python/main.py with
// Bridge.notify("imu_batch", "ax,ay,az,gx,gy,gz;ax,ay,...;"), all raw sensor
// counts. Python converts them: accel is +-16 g (2048 counts per g) because a
// shoe hitting the ground while running easily passes 8 g; gyro is +-2000 deg/s
// (16.4 counts per deg/s).
//
// Wiring: the Qwiic connector (Wire1) or the SDA/SCL header pins (Wire) both
// work -- the sketch looks for the IMU at 0x68 / 0x69 on each bus. If it's not
// found (or the cable is knocked loose mid-run) it sends
// Bridge.notify("imu_status", "missing") once a second and keeps looking.

#include <Arduino_RouterBridge.h>
#include <Wire.h>

const uint8_t MPU_ADDRS[] = {0x68, 0x69};   // 0x69 if the board's AD0 pin is pulled high
const unsigned long SAMPLE_US = 10000;      // 100 Hz
const int BATCH_SIZE = 10;
const int MAX_READ_FAILS = 5;               // consecutive failed reads before re-searching

// MPU-6050 registers
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
                delay(100);                     // let it finish power-on before configuring
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

bool readImu(int16_t out[6]) {
    bus->beginTransmission(addr);
    bus->write(REG_ACCEL_XOUT_H);
    if (bus->endTransmission(false) != 0) return false;
    if (bus->requestFrom(addr, (uint8_t)14) != 14) return false;
    uint8_t raw[14];
    for (int i = 0; i < 14; i++) raw[i] = bus->read();
    // 0-5: accel xyz, 6-7: temperature (skipped), 8-13: gyro xyz -- all big-endian
    const int offsets[6] = {0, 2, 4, 8, 10, 12};
    for (int i = 0; i < 6; i++) out[i] = (int16_t)((raw[offsets[i]] << 8) | raw[offsets[i] + 1]);
    return true;
}

void setup() {
    Bridge.begin();
    batch.reserve(BATCH_SIZE * 40);
    nextSampleUs = micros();
}

void loop() {
    if (bus == nullptr) {
        if (millis() - lastSearchMs >= 1000) {
            lastSearchMs = millis();
            if (findImu()) {
                Bridge.notify("imu_status", String("ok"));
                nextSampleUs = micros();
            } else {
                Bridge.notify("imu_status", String("missing"));
            }
        }
        return;
    }

    if ((long)(micros() - nextSampleUs) < 0) return;
    nextSampleUs += SAMPLE_US;

    int16_t v[6];
    if (!readImu(v)) {
        if (++readFails >= MAX_READ_FAILS) {
            bus = nullptr;
            batch = "";
            batchCount = 0;
        }
        return;
    }
    readFails = 0;

    for (int i = 0; i < 6; i++) {
        batch += v[i];
        batch += (i < 5) ? ',' : ';';
    }
    if (++batchCount >= BATCH_SIZE) {
        Bridge.notify("imu_batch", batch);
        batch = "";
        batchCount = 0;
    }
}
