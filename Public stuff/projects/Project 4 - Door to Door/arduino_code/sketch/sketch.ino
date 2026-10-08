#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <ArduinoJson.h>
#include <CytronMotorDriver.h>

Arduino_LED_Matrix matrix;

const uint8_t ROWS = 8;
const uint8_t COLS = 13;
const uint8_t BRIGHTNESS = 7;
uint8_t frame[ROWS * COLS] = {0};

CytronMD motorA(PWM_DIR, 5, 6);
CytronMD motorB(PWM_DIR, 10, 11);

const unsigned long WATCHDOG_TIMEOUT_MS = 500;

const int DRIVE_PWM = 70;
const int DEADBAND = 40;
const unsigned long PULSE_MS = 150;
const unsigned long PAUSE_MS = 250;

struct Channel {
    int col = -1;
    int row = -1;
    bool seen = false;
    unsigned long lastSeenMillis = 0;
};

Channel green;
Channel blue;

int latestErr = 0;
bool driving = false;
unsigned long phaseStart = 0;
int appliedPwm = 0;

void setMotors(int pwm) {
    if (pwm == appliedPwm) return;
    appliedPwm = pwm;
    motorA.setSpeed(pwm);
    motorB.setSpeed(pwm);
}

void stopMotors(unsigned long now) {
    setMotors(0);
    if (driving) {
        driving = false;
        phaseStart = now;
    }
}

void renderMatrix() {
    memset(frame, 0, sizeof(frame));
    Channel &active = green.seen ? green : blue;
    if (active.seen && active.col >= 0 && active.col < COLS && active.row >= 0 && active.row < ROWS) {
        frame[active.row * COLS + active.col] = BRIGHTNESS;
    }
}

void handleGreen(String payload) {
    JsonDocument doc;
    if (deserializeJson(doc, payload) != DeserializationError::Ok) return;

    green.seen = (doc["seen"] | 0) != 0;
    green.col = doc["col"] | -1;
    green.row = doc["row"] | -1;
    green.lastSeenMillis = millis();
    latestErr = doc["err"] | 0;

    renderMatrix();
}

void handleBlue(String payload) {
    JsonDocument doc;
    if (deserializeJson(doc, payload) != DeserializationError::Ok) return;

    blue.seen = (doc["seen"] | 0) != 0;
    blue.col = doc["col"] | -1;
    blue.row = doc["row"] | -1;
    blue.lastSeenMillis = millis();

    renderMatrix();
}

void updateMotors(unsigned long now) {
    if (!green.seen || abs(latestErr) < DEADBAND) {
        stopMotors(now);
        return;
    }

    if (driving) {
        if (now - phaseStart >= PULSE_MS) stopMotors(now);
    } else if (now - phaseStart >= PAUSE_MS) {
        setMotors(latestErr > 0 ? -DRIVE_PWM : DRIVE_PWM);
        driving = true;
        phaseStart = now;
    }
}

void setup() {
    matrix.begin();
    matrix.setGrayscaleBits(3);
    matrix.clear();

    Bridge.begin();
    Bridge.provide("green", handleGreen);
    Bridge.provide("blue", handleBlue);
}

void loop() {
    unsigned long now = millis();
    bool changed = false;

    if (green.seen && now - green.lastSeenMillis > WATCHDOG_TIMEOUT_MS) {
        green.seen = false;
        changed = true;
    }
    if (blue.seen && now - blue.lastSeenMillis > WATCHDOG_TIMEOUT_MS) {
        blue.seen = false;
        changed = true;
    }

    if (changed) renderMatrix();

    updateMotors(now);
    matrix.draw(frame);
    delay(10);
}