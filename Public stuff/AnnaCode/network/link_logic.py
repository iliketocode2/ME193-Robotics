"""Pure logic for how healthy a Bluetooth device link is (no PyScript or DOM here,
so it is unit-tested with plain Python; see AnnaCode/tests/test_link_logic.py).

Device.py keeps one LinkTracker per connected hub and a once-a-second watchdog
turns its state into the chip you see in the Devices card.

Two kinds of "lost":
  disconnected  the browser told us the link dropped (laptop slept, sensor ran out of
                power, walked out of range). Definitive. The page stops writing to the
                device and ignores a controller that has gone.
  no_signal     no data for a while but no drop event. Shown on the chip only, and it
                clears by itself if data returns, so a hiccup never changes behavior.
"""

# The hub sends a notification about every 100 ms, so this much silence means trouble.
NO_SIGNAL_AFTER_S = 5.0


class LinkTracker:
    def __init__(self, now: float):
        self.last_data = now           # last notification of any kind
        self.last_battery = None       # when the hub last reported a battery level
        self.battery_reports = 0       # how many battery reports so far
        self.disconnected = False      # set when the browser says the link dropped

    def data(self, now: float):
        self.last_data = now

    def battery_report(self, now: float):
        self.last_battery = now
        self.battery_reports += 1

    def mark_disconnected(self):
        self.disconnected = True

    def silent_for(self, now: float) -> float:
        return max(0.0, now - self.last_data)

    def battery_age(self, now: float):
        """Seconds since the last battery report, or None if there has not been one."""
        return None if self.last_battery is None else max(0.0, now - self.last_battery)

    def state(self, now: float) -> str:
        if self.disconnected:
            return "disconnected"
        if self.silent_for(now) > NO_SIGNAL_AFTER_S:
            return "no_signal"
        return "ok"

    def lost(self, now: float) -> bool:
        return self.state(now) != "ok"


def status_text(link_state: str) -> str:
    return {"disconnected": "Disconnected", "no_signal": "No signal"}.get(link_state, "")


def age_text(seconds) -> str:
    if seconds is None:
        return "never"
    seconds = max(0, int(seconds))
    if seconds < 5:
        return "just now"
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h ago"


def age_label(seconds) -> str:
    """Text shown on the chip beside the battery: how long since the last report.
    Empty until a report has arrived."""
    return "" if seconds is None else age_text(seconds)


def battery_title(level, age, reports: int, link_state: str) -> str:
    """Tooltip for the battery icon."""
    if level is None:
        return "Waiting for a battery report from the device"
    text = f"Battery {level}%"
    if age is not None:
        text += f", last reported {age_text(age)} ({reports} report{'s' if reports != 1 else ''} so far)"
    if link_state != "ok":
        text += ". Connection lost, so this may be out of date"
    return text
