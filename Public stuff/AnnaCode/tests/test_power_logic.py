"""Run from the AnnaCode folder:  python3 -m unittest discover tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "network"))
import power_logic as pw  # noqa: E402


class Modes(unittest.TestCase):
    def test_constants_match_the_library(self):
        # legoeducation 1.2.0: COLOR_SENSOR_MODE_OFF / COLOR_DETECTION / AMBIENT_LIGHT = 0 / 1 / 2
        self.assertEqual((pw.MODE_OFF, pw.MODE_COLOR, pw.MODE_AMBIENT), (0, 1, 2))


class Wanted(unittest.TestCase):
    def test_save_power_off_keeps_the_light_on_always(self):
        for needed in (False, True):
            for used in (False, True):
                self.assertEqual(pw.wanted_mode(False, needed, used), pw.MODE_COLOR)

    def test_idle_with_save_power_is_off(self):
        self.assertEqual(pw.wanted_mode(True, False, True), pw.MODE_OFF)
        self.assertEqual(pw.wanted_mode(True, False, False), pw.MODE_OFF)

    def test_needed_and_read_by_the_network_is_on(self):
        self.assertEqual(pw.wanted_mode(True, True, True), pw.MODE_COLOR)

    def test_a_sensor_nothing_reads_stays_off_even_when_needed(self):
        self.assertEqual(pw.wanted_mode(True, True, False), pw.MODE_OFF)


class Believing(unittest.TestCase):
    def test_reported_mode_wins(self):
        self.assertEqual(pw.effective_mode(pw.MODE_OFF, pw.MODE_COLOR), pw.MODE_COLOR)

    def test_falls_back_to_what_was_asked(self):
        self.assertEqual(pw.effective_mode(pw.MODE_OFF, None), pw.MODE_OFF)
        self.assertIsNone(pw.effective_mode(None, None))


class Ready(unittest.TestCase):
    def test_waits_for_the_reported_mode(self):
        self.assertFalse(pw.ready(pw.MODE_COLOR, pw.MODE_OFF, 10.0))   # still reports off, however long
        self.assertTrue(pw.ready(pw.MODE_COLOR, pw.MODE_COLOR, 0.0))

    def test_without_a_reported_mode_it_waits_a_settling_time(self):
        self.assertFalse(pw.ready(pw.MODE_COLOR, None, pw.SETTLE_S - 0.01))
        self.assertTrue(pw.ready(pw.MODE_COLOR, None, pw.SETTLE_S))


class Resend(unittest.TestCase):
    def test_no_resend_when_it_matches(self):
        self.assertFalse(pw.resend_due(pw.MODE_OFF, pw.MODE_OFF, 99, 0))

    def test_no_resend_before_the_wait(self):
        self.assertFalse(pw.resend_due(pw.MODE_OFF, pw.MODE_COLOR, pw.RESEND_AFTER_S - 0.1, 0))

    def test_resend_when_it_still_reports_the_wrong_mode(self):
        self.assertTrue(pw.resend_due(pw.MODE_OFF, pw.MODE_COLOR, pw.RESEND_AFTER_S, 0))

    def test_gives_up_after_the_retry_limit(self):
        self.assertFalse(pw.resend_due(pw.MODE_OFF, pw.MODE_COLOR, 99, pw.MAX_RETRIES))
        self.assertTrue(pw.resend_due(pw.MODE_OFF, pw.MODE_COLOR, 99, pw.MAX_RETRIES - 1))

    def test_never_resends_when_the_sensor_cannot_report_its_mode(self):
        self.assertFalse(pw.resend_due(pw.MODE_OFF, None, 99, 0))

    def test_nothing_requested_nothing_to_resend(self):
        self.assertFalse(pw.resend_due(None, pw.MODE_COLOR, 99, 0))


class Note(unittest.TestCase):
    def test_only_off_gets_a_label(self):
        self.assertEqual(pw.note_text(pw.MODE_OFF), "light off")
        self.assertEqual(pw.note_text(pw.MODE_COLOR), "")
        self.assertEqual(pw.note_text(pw.MODE_AMBIENT), "")
        self.assertEqual(pw.note_text(None), "")


if __name__ == "__main__":
    unittest.main()
