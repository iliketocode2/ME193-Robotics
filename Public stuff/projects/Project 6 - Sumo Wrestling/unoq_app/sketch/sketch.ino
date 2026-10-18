// Sumo robot -- MCU side: measures the downforce fan's speed.
//
// The fan (Arctic P12 PRO, 4-pin PC fan) is powered straight from the UNO Q's
// 5V and GND, with its PWM wire left unconnected (= full speed), so there is
// nothing to switch here. Its TACH wire is open-collector: it pulls low twice
// per revolution and needs a pull-up, which INPUT_PULLUP provides. Counting
// those pulses gives RPM; python/main.py reads it with Bridge.call("fan_rpm")
// and publishes it over MQTT (T/unoq), so the laptop can see the fan really is
// spinning.

#include <Arduino_RouterBridge.h>

const int TACH_PIN = 2;                  // fan pin 3 (sense / tach). TODO: change if wired elsewhere
const int PULSES_PER_REV = 2;            // standard for PC fans
const unsigned long WINDOW_MS = 1000;

volatile unsigned long tachPulses = 0;
int lastRpm = 0;
unsigned long windowStart = 0;

void onTach() {
    tachPulses++;
}

int fanRpm() {
    return lastRpm;
}

void setup() {
    pinMode(TACH_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(TACH_PIN), onTach, FALLING);
    Bridge.begin();
    Bridge.provide("fan_rpm", fanRpm);
    windowStart = millis();
}

void loop() {
    unsigned long now = millis();
    if (now - windowStart >= WINDOW_MS) {
        noInterrupts();
        unsigned long pulses = tachPulses;
        tachPulses = 0;
        interrupts();
        lastRpm = (int)(pulses * 60000UL / PULSES_PER_REV / (now - windowStart));
        windowStart = now;
    }
    delay(10);
}
