"""Run from the AnnaCode folder:  python3 -m unittest discover tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "network"))
import virtual_logic as vl  # noqa: E402


class KeyTargets(unittest.TestCase):
    def test_no_keys_is_centred(self):
        self.assertEqual(vl.key_targets(set()), (0.0, 0.0))

    def test_up_drives_both_sticks_forward(self):
        self.assertEqual(vl.key_targets({"ArrowUp"}), (100.0, 100.0))

    def test_down_drives_both_back(self):
        self.assertEqual(vl.key_targets({"ArrowDown"}), (-100.0, -100.0))

    def test_left_and_right_spin_in_place(self):
        # turning right: left wheel forward, right wheel back
        self.assertEqual(vl.key_targets({"ArrowRight"}), (100.0, -100.0))
        self.assertEqual(vl.key_targets({"ArrowLeft"}), (-100.0, 100.0))

    def test_forward_plus_turn_is_an_arc(self):
        self.assertEqual(vl.key_targets({"ArrowUp", "ArrowRight"}), (100.0, 0.0))
        self.assertEqual(vl.key_targets({"ArrowUp", "ArrowLeft"}), (0.0, 100.0))

    def test_opposite_keys_cancel(self):
        self.assertEqual(vl.key_targets({"ArrowUp", "ArrowDown"}), (0.0, 0.0))
        self.assertEqual(vl.key_targets({"ArrowLeft", "ArrowRight"}), (0.0, 0.0))

    def test_tank_keys_drive_one_stick_each(self):
        self.assertEqual(vl.key_targets({"w"}), (100.0, 0.0))
        self.assertEqual(vl.key_targets({"s"}), (-100.0, 0.0))
        self.assertEqual(vl.key_targets({"i"}), (0.0, 100.0))
        self.assertEqual(vl.key_targets({"k"}), (0.0, -100.0))

    def test_caps_lock_and_shift_do_not_matter(self):
        self.assertEqual(vl.key_targets({"W", "K"}), (100.0, -100.0))

    def test_combined_keys_are_clamped(self):
        self.assertEqual(vl.key_targets({"ArrowUp", "w"}), (100.0, 100.0))

    def test_unrelated_keys_are_ignored(self):
        self.assertEqual(vl.key_targets({"a", "Enter", " "}), (0.0, 0.0))
        self.assertFalse(vl.has_control_key({"a", "Enter"}))
        self.assertTrue(vl.has_control_key({"W"}))
        self.assertTrue(vl.has_control_key({"ArrowLeft"}))


class Ramp(unittest.TestCase):
    def test_moves_toward_target_at_the_up_rate(self):
        self.assertAlmostEqual(vl.ramp(0, 100, 0.05), 20.0)

    def test_does_not_overshoot(self):
        self.assertEqual(vl.ramp(95, 100, 0.05), 100)
        self.assertEqual(vl.ramp(-95, -100, 0.05), -100)

    def test_springs_back_faster_than_it_pushes(self):
        self.assertAlmostEqual(vl.ramp(100, 0, 0.05), 60.0)
        self.assertEqual(vl.ramp(30, 0, 0.05), 0)

    def test_negative_direction(self):
        self.assertAlmostEqual(vl.ramp(0, -100, 0.05), -20.0)
        self.assertAlmostEqual(vl.ramp(-100, 0, 0.05), -60.0)

    def test_full_travel_time_is_about_a_quarter_second(self):
        v, t = 0.0, 0.0
        while v < 100 and t < 2:
            v, t = vl.ramp(v, 100, 0.05), t + 0.05
        self.assertLessEqual(t, 0.30)

    def test_negative_dt_does_not_move_backwards(self):
        self.assertEqual(vl.ramp(10, 100, -1.0), 10)


class Motor(unittest.TestCase):
    def test_clamp_speed(self):
        self.assertEqual(vl.clamp_speed(150), 100)
        self.assertEqual(vl.clamp_speed(-150), -100)
        self.assertEqual(vl.clamp_speed(33.6), 34)
        self.assertEqual(vl.clamp_speed(None), 0)
        self.assertIsInstance(vl.clamp_speed(12.2), int)

    def test_position_advances_with_speed(self):
        self.assertAlmostEqual(vl.advance_position(0, 10, 1.0), 60.0)
        self.assertAlmostEqual(vl.advance_position(100, -10, 1.0), 40.0)
        self.assertEqual(vl.advance_position(5, 0, 1.0), 5)

    def test_wrap_angle(self):
        self.assertEqual(vl.wrap_angle(0), 0)
        self.assertEqual(vl.wrap_angle(190), -170)
        self.assertEqual(vl.wrap_angle(-190), 170)
        self.assertEqual(vl.wrap_angle(360), 0)

    def test_spin_seconds(self):
        self.assertIsNone(vl.spin_seconds(0))
        self.assertEqual(vl.spin_seconds(30), 2.0)
        self.assertEqual(vl.spin_seconds(-60), 1.0)


if __name__ == "__main__":
    unittest.main()
