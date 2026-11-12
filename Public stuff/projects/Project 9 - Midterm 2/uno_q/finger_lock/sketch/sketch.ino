// Finger-count lock: the microcontroller side of the UNO Q.
//
// The Linux side (python/main.py) runs the camera, the network and lock_policy.py. It tells this sketch which
// LEDs to light and whether the latch should be open. This sketch does the parts that need exact timing:
// it reads the button, drives the motor until a limit switch clicks, and stops the motor if a switch never does.
//
// Messages over the Bridge:
//   Linux -> here   set_leds(mask)   bit 0-2 digit LEDs, bit 3 green, bit 4 red
//                   set_latch(open)  1 to open, 0 to lock
//                   get_status()     returns the inputs, read by Python every frame:
//                                    bit 0 button down, bit 1 locked switch, bit 2 open switch,
//                                    bits 3-4 latch (0 locked, 1 open, 2 switch did not click, 3 moving),
//                                    bits 8-15 how many times the button has been pressed (counts up, wraps at 256)
//   here -> Linux   latch(state)     sent when the latch stops; Python also gets it from get_status
//
// The button goes through get_status rather than a message from here, so a press is never lost if the Linux
// side is busy or not listening yet, and the live view can show the raw button and switch states.
//
// Checks you can run before the Python side works:
//   - At power-up every LED lights in turn.
//   - Until the Linux side starts talking, the yellow LEDs chase back and forth.
//   - Hold the button while the board powers up to run the latch once: open, wait, lock.

#include <Arduino_RouterBridge.h>

// Pins, matching hardware/wiring.py
const int PIN_ENA = 9;          // PWM, HW-095 ENA (jumper removed)
const int PIN_IN1 = 7;
const int PIN_IN2 = 8;
const int PIN_SW_LOCKED = 2;    // limit switch the arm's tail presses when locked
const int PIN_SW_OPEN = 4;      // limit switch the arm presses when open
const int PIN_BUTTON = 12;
const int LED_PINS[5] = {3, 5, 6, 10, 11};   // digit 1, digit 2, digit 3, green, red

// Settings to tune on the real latch
const int MOTOR_SPEED = 200;                 // 0-255; lower it if the arm slams into the switches
const bool OPEN_MEANS_IN1_HIGH = true;       // flip this if the arm turns the wrong way
const unsigned long MOVE_TIMEOUT_MS = 1500;  // stop the motor if no switch clicks in this time
const unsigned long DEBOUNCE_MS = 30;
const unsigned long LINUX_TIMEOUT_MS = 3000; // no LED update for this long means the Linux side stopped

enum LatchState { LATCH_LOCKED = 0, LATCH_OPEN = 1, LATCH_FAULT = 2, LATCH_MOVING = 3 };

// Written by the Bridge callbacks, read in loop()
volatile int ledMask = 0;
volatile int wantOpen = 0;
volatile unsigned long lastHeardMs = 0;
volatile bool heardLinux = false;

int latchState = LATCH_FAULT;   // unknown until the first move finishes
int movingTo = 0;               // 1 opening, 0 locking
int lastWant = -1;
unsigned long moveStartMs = 0;

bool buttonDown = false;
int pressCount = 0;
unsigned long buttonChangeMs = 0;

bool pressed(int pin);

// Bridge callbacks: only store values here, loop() does the work.
void setLeds(int mask) {
  ledMask = mask;
  lastHeardMs = millis();
  heardLinux = true;
}

void setLatch(int open) {
  wantOpen = open ? 1 : 0;
  lastHeardMs = millis();
  heardLinux = true;
}

int getStatus() {
  lastHeardMs = millis();
  heardLinux = true;
  return (pressed(PIN_BUTTON) ? 1 : 0) | (pressed(PIN_SW_LOCKED) ? 2 : 0) | (pressed(PIN_SW_OPEN) ? 4 : 0)
         | ((latchState & 3) << 3) | (pressCount << 8);
}

bool pressed(int pin) {
  return digitalRead(pin) == LOW;   // switch between the pin and GND, internal pull-up
}

// +1 open, -1 lock, 0 stop (brake: both inputs low with the driver enabled)
void motor(int dir) {
  if (dir == 0) {
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, LOW);
    analogWrite(PIN_ENA, 255);
    return;
  }
  bool in1 = (dir > 0) == OPEN_MEANS_IN1_HIGH;
  digitalWrite(PIN_IN1, in1 ? HIGH : LOW);
  digitalWrite(PIN_IN2, in1 ? LOW : HIGH);
  analogWrite(PIN_ENA, MOTOR_SPEED);
}

void startMove(int open) {
  int targetSwitch = open ? PIN_SW_OPEN : PIN_SW_LOCKED;
  if (pressed(targetSwitch)) {          // already there
    motor(0);
    latchState = open ? LATCH_OPEN : LATCH_LOCKED;
    Bridge.notify("latch", latchState);
    return;
  }
  movingTo = open;
  moveStartMs = millis();
  latchState = LATCH_MOVING;
  motor(open ? 1 : -1);
}

void updateMove() {
  if (latchState != LATCH_MOVING) return;
  int targetSwitch = movingTo ? PIN_SW_OPEN : PIN_SW_LOCKED;
  if (pressed(targetSwitch)) {
    motor(0);
    latchState = movingTo ? LATCH_OPEN : LATCH_LOCKED;
    Bridge.notify("latch", latchState);
  } else if (millis() - moveStartMs > MOVE_TIMEOUT_MS) {
    motor(0);
    latchState = LATCH_FAULT;
    Bridge.notify("latch", latchState);
  }
}

// Blocking move, only for the power-up latch test
void moveAndWait(int open) {
  startMove(open);
  while (latchState == LATCH_MOVING) {
    updateMove();
    delay(1);
  }
}

void showLeds(int mask) {
  for (int i = 0; i < 5; i++) {
    digitalWrite(LED_PINS[i], (mask >> i) & 1 ? HIGH : LOW);
  }
}

void updateButton() {
  bool down = pressed(PIN_BUTTON);
  unsigned long now = millis();
  if (down != buttonDown && now - buttonChangeMs > DEBOUNCE_MS) {
    buttonDown = down;
    buttonChangeMs = now;
    if (down) {
      pressCount = (pressCount + 1) & 0xFF;
    }
  }
}

void setup() {
  Bridge.begin();
  Bridge.provide_safe("set_leds", setLeds);
  Bridge.provide_safe("set_latch", setLatch);
  Bridge.provide_safe("get_status", getStatus);

  pinMode(PIN_ENA, OUTPUT);
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  motor(0);
  pinMode(PIN_SW_LOCKED, INPUT_PULLUP);
  pinMode(PIN_SW_OPEN, INPUT_PULLUP);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  for (int i = 0; i < 5; i++) {
    pinMode(LED_PINS[i], OUTPUT);
  }

  // LED check: each one in turn
  for (int i = 0; i < 5; i++) {
    showLeds(1 << i);
    delay(200);
  }
  showLeds(0);

  if (pressed(PIN_BUTTON)) {   // latch test
    showLeds(0b01000);
    moveAndWait(1);
    delay(1500);
    showLeds(0b10000);
    moveAndWait(0);
    showLeds(0);
    while (pressed(PIN_BUTTON)) delay(10);   // wait for the button to be let go
  }

  wantOpen = 0;   // start locked
}

void loop() {
  updateButton();

  int want = wantOpen;
  if (want != lastWant) {
    lastWant = want;
    startMove(want);
  }
  updateMove();

  unsigned long now = millis();
  bool linuxAlive = heardLinux && now - lastHeardMs < LINUX_TIMEOUT_MS;
  int mask;
  if (linuxAlive) {
    mask = ledMask;
  } else {
    // waiting for the Linux side: yellow LEDs chase 1, 2, 3, 2, 1 ...
    const int chase[4] = {0, 1, 2, 1};
    mask = 1 << chase[(now / 250) % 4];
  }
  if (latchState == LATCH_FAULT && (now / 150) % 2 == 0) {
    mask |= 0b10000;   // fast red flicker: a limit switch did not click
  }
  showLeds(mask);
}
