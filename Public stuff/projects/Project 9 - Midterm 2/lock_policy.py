"""The rules of the finger-count lock, with no camera, network or hardware code.

lock.py feeds it from the webcam on the computer. On the UNO Q the same file runs on the Linux side, and the
LED and latch outputs go to the microcontroller instead of the screen.

How an attempt works:
  1. LOCKED. Press the button to start.
  2. CLEAR. Take your hand out of view (the network must read 0) so every digit starts from an empty view.
  3. READ. Show a count from 1 to 5 and hold it. It is accepted once the network gives that same count, at least
     MIN_CONF sure, for HOLD_S seconds without a break. One digit LED lights. Back to CLEAR for the next digit,
     which is what lets a code repeat a digit (3, 3, 5).
  4. After the third digit the whole code is checked at once, so a wrong code does not say which digit was wrong.
     Right: OPEN for OPEN_S seconds, then it locks itself again. Wrong: a short red signal, then LOCKED.
  5. MAX_FAILS wrong codes in a row start a LOCKOUT of LOCKOUT_S seconds that ignores the button.
  Any step with no progress for DIGIT_TIMEOUT_S cancels the attempt. Pressing the button mid-attempt also cancels
  it, and pressing it while open locks the door right away.
"""

import config

LOCKED, CLEAR, READ, WRONG, OPEN, LOCKOUT = "LOCKED", "CLEAR", "READ", "WRONG", "OPEN", "LOCKOUT"


class LockPolicy:
    def __init__(self, code=config.LOCK_CODE, min_conf=config.MIN_CONF, hold_s=config.HOLD_S,
                 clear_s=config.CLEAR_S, digit_timeout_s=config.DIGIT_TIMEOUT_S, open_s=config.OPEN_S,
                 max_fails=config.MAX_FAILS, lockout_s=config.LOCKOUT_S, result_s=config.RESULT_S):
        if len(code) != 3 or not all(1 <= d <= 5 for d in code):
            raise ValueError("the code must be three counts from 1 to 5")
        self.code = tuple(code)
        self.min_conf, self.hold_s, self.clear_s = min_conf, hold_s, clear_s
        self.digit_timeout_s, self.open_s = digit_timeout_s, open_s
        self.max_fails, self.lockout_s, self.result_s = max_fails, lockout_s, result_s

        self.state = LOCKED
        self.entered = []           # digits accepted so far in this attempt
        self.fails = 0              # wrong codes in a row
        self.state_since = 0.0      # when the current state (or step) started
        self.candidate = None       # the count currently being held
        self.candidate_since = 0.0
        self.message = "press the button"

    # inputs ---------------------------------------------------------------------------------------------------

    def press(self, now):
        """The button. Returns a list of events, like update()."""
        if self.state == LOCKED:
            self.entered = []
            self._go(CLEAR, now, "take your hand away")
            return ["start"]
        if self.state in (CLEAR, READ):
            self._go(LOCKED, now, "cancelled, press the button")
            return ["cancel"]
        if self.state == OPEN:
            self._go(LOCKED, now, "locked, press the button")
            return ["relock"]
        return []                   # WRONG and LOCKOUT ignore the button

    def update(self, count, conf, now):
        """One network reading: the most likely count, how sure the network is, and the time in seconds.

        Returns a list of events ("digit", "open", "wrong", "lockout", "timeout", "relock") for logging or sounds.
        """
        events = []
        elapsed = now - self.state_since

        if self.state == OPEN and elapsed >= self.open_s:
            self._go(LOCKED, now, "locked, press the button")
            events.append("relock")
        elif self.state == WRONG and elapsed >= self.result_s:
            self._go(LOCKED, now, f"wrong code ({self.fails}/{self.max_fails}), press the button")
        elif self.state == LOCKOUT and elapsed >= self.lockout_s:
            self.fails = 0
            self._go(LOCKED, now, "press the button")
        elif self.state in (CLEAR, READ):
            if elapsed >= self.digit_timeout_s:
                self._go(LOCKED, now, "timed out, press the button")
                events.append("timeout")
            else:
                events += self._step(count, conf, now)
        return events

    # outputs --------------------------------------------------------------------------------------------------

    def progress(self, now):
        """0 to 1: how far through holding the current count (READ) or the empty view (CLEAR)."""
        if self.candidate is None:
            return 0.0
        need = self.clear_s if self.state == CLEAR else self.hold_s
        return min((now - self.candidate_since) / need, 1.0)

    def time_left(self, now):
        """Seconds left in a timed state, or None."""
        limit = {OPEN: self.open_s, WRONG: self.result_s, LOCKOUT: self.lockout_s,
                 CLEAR: self.digit_timeout_s, READ: self.digit_timeout_s}.get(self.state)
        return None if limit is None else max(limit - (now - self.state_since), 0.0)

    def leds(self, now):
        """What the lights should show.

        These match the LEDs on the front of the box: three yellow digit LEDs, one green and one red.
        digits: three booleans. While listening, the accepted digits are lit and the next one blinks.
        status: "off", "green" (open) or "red" (wrong code; it blinks during the lockout).
        """
        blink_on = int((now - self.state_since) * 4) % 2 == 0     # 2 blinks per second
        if self.state == OPEN:
            return {"digits": [True] * 3, "status": "green"}
        if self.state == WRONG:
            return {"digits": [True] * 3, "status": "red"}
        if self.state == LOCKOUT:
            return {"digits": [False] * 3, "status": "red" if blink_on else "off"}
        if self.state in (CLEAR, READ):
            n = len(self.entered)
            return {"digits": [i < n or (i == n and blink_on) for i in range(3)], "status": "off"}
        return {"digits": [False] * 3, "status": "off"}

    @property
    def unlocked(self):
        """True while the latch should be open."""
        return self.state == OPEN

    # internals ------------------------------------------------------------------------------------------------

    def _go(self, state, now, message):
        self.state, self.state_since, self.message = state, now, message
        self.candidate = None

    def _held(self, value, count, conf, now, need):
        """True once `value` has been read, sure enough, for `need` seconds in a row."""
        if count != value or conf < self.min_conf:
            self.candidate = None
            return False
        if self.candidate is None:
            self.candidate, self.candidate_since = value, now
        return now - self.candidate_since >= need

    def _step(self, count, conf, now):
        if self.state == CLEAR:
            if self._held(0, count, conf, now, self.clear_s):
                self._go(READ, now, f"show digit {len(self.entered) + 1} of 3")
            return []

        # READ: follow whichever count from 1 to 5 the network sees; a change restarts the hold.
        if count != self.candidate:
            self.candidate = None
        if count == 0 or not self._held(count, count, conf, now, self.hold_s):
            return []

        self.entered.append(count)
        if len(self.entered) < 3:
            self._go(CLEAR, now, "got it, take your hand away")
            return ["digit"]

        if tuple(self.entered) == self.code:
            self.fails = 0
            self._go(OPEN, now, "unlocked")
            return ["digit", "open"]

        self.fails += 1
        if self.fails >= self.max_fails:
            self._go(LOCKOUT, now, f"{self.max_fails} wrong codes, locked out")
            return ["digit", "wrong", "lockout"]
        self._go(WRONG, now, "wrong code")
        return ["digit", "wrong"]
