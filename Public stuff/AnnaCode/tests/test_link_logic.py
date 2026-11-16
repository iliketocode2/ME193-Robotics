"""Run from the AnnaCode folder:  python3 -m unittest discover tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "network"))
import link_logic as ll  # noqa: E402


class Tracker(unittest.TestCase):
    def test_fresh_link_is_ok(self):
        self.assertEqual(ll.LinkTracker(100.0).state(100.0), "ok")

    def test_steady_data_stays_ok(self):
        t = ll.LinkTracker(0.0)
        for now in range(1, 60):
            t.data(float(now))
            self.assertEqual(t.state(float(now)), "ok")

    def test_silence_shows_no_signal(self):
        t = ll.LinkTracker(0.0)
        self.assertEqual(t.state(ll.NO_SIGNAL_AFTER_S - 0.1), "ok")
        self.assertEqual(t.state(ll.NO_SIGNAL_AFTER_S + 0.1), "no_signal")

    def test_no_signal_clears_itself_when_data_returns(self):
        t = ll.LinkTracker(0.0)
        self.assertEqual(t.state(30.0), "no_signal")
        t.data(30.0)
        self.assertEqual(t.state(30.5), "ok")

    def test_drop_event_is_definitive(self):
        t = ll.LinkTracker(0.0)
        t.mark_disconnected()
        self.assertEqual(t.state(0.1), "disconnected")
        self.assertTrue(t.lost(0.1))

    def test_disconnected_wins_over_no_signal_and_stays(self):
        t = ll.LinkTracker(0.0)
        t.mark_disconnected()
        self.assertEqual(t.state(500.0), "disconnected")
        t.data(500.0)      # stray data does not undo a real drop
        self.assertEqual(t.state(500.1), "disconnected")

    def test_lost_is_false_when_ok(self):
        self.assertFalse(ll.LinkTracker(0.0).lost(1.0))

    def test_clock_going_backwards_does_not_crash(self):
        t = ll.LinkTracker(100.0)
        self.assertEqual(t.state(50.0), "ok")
        self.assertEqual(t.silent_for(50.0), 0.0)


class Battery(unittest.TestCase):
    def test_no_report_yet(self):
        t = ll.LinkTracker(0.0)
        self.assertIsNone(t.battery_age(10.0))
        self.assertEqual(t.battery_reports, 0)

    def test_reports_are_counted_and_aged(self):
        t = ll.LinkTracker(0.0)
        t.battery_report(10.0)
        t.battery_report(70.0)
        self.assertEqual(t.battery_reports, 2)
        self.assertEqual(t.battery_age(100.0), 30.0)

    def test_data_alone_does_not_count_as_a_battery_report(self):
        t = ll.LinkTracker(0.0)
        t.data(5.0)
        self.assertEqual(t.battery_reports, 0)
        self.assertIsNone(t.battery_age(6.0))


class Text(unittest.TestCase):
    def test_age_text(self):
        self.assertEqual(ll.age_text(None), "never")
        self.assertEqual(ll.age_text(0), "just now")
        self.assertEqual(ll.age_text(4.9), "just now")
        self.assertEqual(ll.age_text(30), "30 s ago")
        self.assertEqual(ll.age_text(59), "59 s ago")
        self.assertEqual(ll.age_text(60), "1 min ago")
        self.assertEqual(ll.age_text(150), "2 min ago")
        self.assertEqual(ll.age_text(3600), "1 h ago")
        self.assertEqual(ll.age_text(7300), "2 h ago")
        self.assertEqual(ll.age_text(-5), "just now")

    def test_age_label_is_empty_until_a_report_arrives(self):
        self.assertEqual(ll.age_label(None), "")
        self.assertEqual(ll.age_label(2), "just now")
        self.assertEqual(ll.age_label(150), "2 min ago")

    def test_status_text(self):
        self.assertEqual(ll.status_text("ok"), "")
        self.assertEqual(ll.status_text("disconnected"), "Disconnected")
        self.assertEqual(ll.status_text("no_signal"), "No signal")

    def test_battery_title_without_a_level(self):
        self.assertIn("Waiting", ll.battery_title(None, None, 0, "ok"))

    def test_battery_title_shows_age_and_report_count(self):
        title = ll.battery_title(80, 120, 3, "ok")
        self.assertIn("80%", title)
        self.assertIn("2 min ago", title)
        self.assertIn("3 reports", title)
        self.assertNotIn("lost", title.lower())

    def test_battery_title_singular_report(self):
        self.assertIn("1 report so far", ll.battery_title(80, 10, 1, "ok"))

    def test_battery_title_warns_when_connection_is_lost(self):
        self.assertIn("Connection lost", ll.battery_title(80, 10, 2, "disconnected"))
        self.assertIn("Connection lost", ll.battery_title(80, 10, 2, "no_signal"))

    def test_battery_title_with_level_but_no_report_time(self):
        title = ll.battery_title(55, None, 0, "ok")
        self.assertEqual(title, "Battery 55%")


if __name__ == "__main__":
    unittest.main()
