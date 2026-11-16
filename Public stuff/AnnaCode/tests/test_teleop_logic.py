"""Run from the AnnaCode folder:  python3 -m unittest discover tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "network"))
import teleop_logic as tl  # noqa: E402


class StickValues(unittest.TestCase):
    def test_reads_both_sticks(self):
        self.assertEqual(tl.stick_values({"LeftPercent": 40, "RightPercent": -25}), (40.0, -25.0))

    def test_none_before_first_notification_is_zero(self):
        self.assertEqual(tl.stick_values({"LeftPercent": None, "RightPercent": None}), (0.0, 0.0))

    def test_missing_keys_are_zero(self):
        self.assertEqual(tl.stick_values({}), (0.0, 0.0))


class Deflection(unittest.TestCase):
    def test_centred_sticks_are_not_deflected(self):
        self.assertFalse(tl.is_deflected(0, 0))
        self.assertFalse(tl.is_deflected(tl.DEADBAND, -tl.DEADBAND))  # exactly on the edge

    def test_either_stick_counts(self):
        self.assertTrue(tl.is_deflected(30, 0))
        self.assertTrue(tl.is_deflected(0, -30))

    def test_none_is_not_deflected(self):
        self.assertFalse(tl.is_deflected(None, None))


class Commands(unittest.TestCase):
    def test_inside_deadband_is_zero(self):
        self.assertEqual(tl.command_for("LeftSpeed", 5, 0), 0)

    def test_full_stick_gives_max_speed(self):
        self.assertEqual(tl.command_for("LeftSpeed", 100, 0, max_speed=50), 50)
        self.assertEqual(tl.command_for("LeftSpeed", -100, 0, max_speed=50), -50)

    def test_ramps_from_zero_just_outside_deadband(self):
        self.assertLessEqual(abs(tl.command_for("LeftSpeed", tl.DEADBAND + 1, 0, max_speed=100)), 2)

    def test_right_wheel_is_mirrored(self):
        # Both sticks forward: the example networks use opposite signs per wheel to go straight.
        self.assertEqual(tl.command_for("LeftSpeed", 100, 100, 100), 100)
        self.assertEqual(tl.command_for("RightSpeed", 100, 100, 100), -100)

    def test_each_wheel_follows_its_own_stick(self):
        self.assertEqual(tl.command_for("LeftSpeed", 0, 100, 100), 0)
        self.assertEqual(tl.command_for("RightSpeed", 100, 0, 100), 0)

    def test_reverse_flips_every_sign(self):
        self.assertEqual(tl.command_for("LeftSpeed", 100, 0, 100, reverse=True), -100)
        self.assertEqual(tl.command_for("RightSpeed", 0, 100, 100, reverse=True), 100)

    def test_single_output_follows_the_harder_stick(self):
        self.assertEqual(tl.command_for("Speed", 30, 90, 100), tl.command_for("Speed", 0, 90, 100))
        self.assertGreater(tl.command_for("BothSpeed", 90, 30, 100), 0)
        self.assertLess(tl.command_for("BothSpeed", -90, 30, 100), 0)

    def test_light_and_beep_channels_are_not_driven(self):
        for ch in ("LightIntensity", "BeepFrequency", "Hue", ""):
            self.assertIsNone(tl.command_for(ch, 100, 100))

    def test_clamped_to_motor_range(self):
        self.assertEqual(tl.command_for("LeftSpeed", 500, 0, max_speed=500), 100)
        self.assertEqual(tl.command_for("LeftSpeed", 100, 0, max_speed=-20), 0)

    def test_result_is_an_int(self):
        self.assertIsInstance(tl.command_for("LeftSpeed", 63.7, 0, 47), int)
        # shaped: (63.7 - 8) / 92 * 100 = 60.54;  x 0.47 = 28.45  ->  28
        self.assertEqual(tl.command_for("LeftSpeed", 63.7, 0, 47), 28)

    def test_garbage_stick_values_are_zero(self):
        self.assertEqual(tl.command_for("LeftSpeed", None, "x"), 0)

    def test_negative_zero_is_plain_zero(self):
        self.assertEqual(str(tl.command_for("RightSpeed", 0, 0)), "0")


class CommandsForOutputs(unittest.TestCase):
    def test_two_wheel_network(self):
        got = tl.commands_for_outputs({1: "LeftSpeed", 2: "RightSpeed"}, 100, 100, 50)
        self.assertEqual(got, {1: 50, 2: -50})

    def test_non_drive_outputs_are_left_out(self):
        got = tl.commands_for_outputs({1: "LeftSpeed", 2: "LightIntensity"}, 100, 0, 50)
        self.assertEqual(got, {1: 50})

    def test_no_outputs(self):
        self.assertEqual(tl.commands_for_outputs({}, 100, 100), {})



class DriveHint(unittest.TestCase):
    def test_nothing_chosen_asks_for_the_output_box(self):
        for channels in ([], [""], ["", ""]):
            self.assertIn("Set the Output box", tl.drive_hint(channels))

    def test_only_a_light_or_beep_has_nothing_to_drive(self):
        self.assertIn("nothing to drive", tl.drive_hint(["Light"]))
        self.assertIn("nothing to drive", tl.drive_hint(["", "Beep"]))

    def test_one_combined_speed_cannot_turn(self):
        for channels in (["BothSpeed"], ["Speed"], ["LeftSpeed"], ["RightSpeed"]):
            self.assertIn("can't turn", tl.drive_hint(channels), channels)

    def test_both_wheels_is_fine(self):
        self.assertEqual(tl.drive_hint(["LeftSpeed", "RightSpeed"]), "")
        self.assertEqual(tl.drive_hint(["RightSpeed", "LeftSpeed", "Light"]), "")

    def test_a_wheel_plus_a_combined_speed_still_cannot_turn(self):
        self.assertIn("can't turn", tl.drive_hint(["LeftSpeed", "BothSpeed"]))



class Quantize(unittest.TestCase):
    def test_the_five_levels_of_max_speed_50(self):
        got = {v: tl.quantize(v, 50) for v in (-50, -40, -25, -10, 0, 10, 25, 40, 50)}
        self.assertEqual(got, {-50: -50.0, -40: -50.0, -25: -25.0, -10: 0.0, 0: 0.0, 10: 0.0, 25: 25.0, 40: 50.0, 50: 50.0})

    def test_boundaries_round_half_up_in_size(self):
        self.assertEqual(tl.quantize(12.4, 50), 0.0)      # just under a quarter
        self.assertEqual(tl.quantize(12.5, 50), 25.0)     # a quarter goes to half
        self.assertEqual(tl.quantize(37.4, 50), 25.0)     # just under three quarters
        self.assertEqual(tl.quantize(37.5, 50), 50.0)     # three quarters goes to full
        self.assertEqual(tl.quantize(-12.5, 50), -25.0)   # symmetric for reverse

    def test_beyond_max_speed_is_full(self):
        self.assertEqual(tl.quantize(100, 50), 50.0)
        self.assertEqual(tl.quantize(-999, 50), -50.0)

    def test_only_five_distinct_results_ever(self):
        results = {tl.quantize(v / 4.0, 50) for v in range(-800, 801)}
        self.assertEqual(results, {-50.0, -25.0, 0.0, 25.0, 50.0})
        self.assertEqual(sorted(r / 50.0 for r in results), list(tl.LEVEL_STEPS))

    def test_scales_with_max_speed(self):
        self.assertEqual(tl.quantize(60, 100), 50.0)
        self.assertEqual(tl.quantize(60, 20), 20.0)

    def test_zero_is_plain_zero_not_negative_zero(self):
        self.assertEqual(str(tl.quantize(-3, 50)), "0.0")

    def test_bad_input_is_zero(self):
        self.assertEqual(tl.quantize(None, 50), 0.0)
        self.assertEqual(tl.quantize(30, 0), 0.0)
        self.assertEqual(tl.quantize(30, None), 0.0)
        self.assertEqual(tl.quantize(float("nan"), 50), 0.0)


class ReverseBox(unittest.TestCase):
    """The main page's Reverse checkbox, checked against what a real car did (virtual controller
    arrow keys, Max speed 50, the Double Motor's light as the car's front). Sticks are what the
    virtual controller's arrow keys give."""
    KEYS = {"Up": (100, 100), "Down": (-100, -100), "Left": (-100, 100), "Right": (100, -100)}

    def wheels(self, key, reverse_checked):
        l, r = self.KEYS[key]
        reverse, swap = tl.chassis_for_reverse_box(reverse_checked)
        return (tl.command_for("LeftSpeed", l, r, 50, reverse, swap_sides=swap),
                tl.command_for("RightSpeed", l, r, 50, reverse, swap_sides=swap))

    def test_unticked_matches_the_car(self):
        self.assertEqual(self.wheels("Up", False), (-50, 50))      # away from the light
        self.assertEqual(self.wheels("Down", False), (50, -50))    # toward the light
        self.assertEqual(self.wheels("Left", False), (50, 50))     # counter-clockwise
        self.assertEqual(self.wheels("Right", False), (-50, -50))  # clockwise

    def test_ticked_matches_the_car(self):
        self.assertEqual(self.wheels("Up", True), (50, -50))       # toward the light
        self.assertEqual(self.wheels("Down", True), (-50, 50))     # away from the light
        self.assertEqual(self.wheels("Left", True), (50, 50))      # still counter-clockwise
        self.assertEqual(self.wheels("Right", True), (-50, -50))   # still clockwise

    def test_ticking_swaps_forward_and_back_but_never_a_turn(self):
        for key in ("Up", "Down"):
            a, b = self.wheels(key, False), self.wheels(key, True)
            self.assertEqual((a[0], a[1]), (-b[0], -b[1]))
        for key in ("Left", "Right"):
            self.assertEqual(self.wheels(key, False), self.wheels(key, True))

    def test_a_single_combined_speed_still_drives_forward_and_back_in_step_with_the_wheels(self):
        # BothSpeed's motor does LeftSpeed=v, RightSpeed=-v, so v matches the left wheel.
        for reverse_checked in (False, True):
            for key in ("Up", "Down"):
                l, r = self.KEYS[key]
                rev, swap = tl.chassis_for_reverse_box(reverse_checked)
                both = tl.command_for("BothSpeed", l, r, 50, rev, swap_sides=swap)
                self.assertEqual(both, self.wheels(key, reverse_checked)[0])

    def test_swap_sides_defaults_off_so_other_callers_are_unchanged(self):
        self.assertEqual(tl.command_for("LeftSpeed", -100, 100, 50), -50)           # left stick drives LeftSpeed
        self.assertEqual(tl.command_for("RightSpeed", -100, 100, 50), -50)          # right stick (mirrored)
        self.assertEqual(tl.command_for("LeftSpeed", -100, 100, 50, swap_sides=True), 50)

    def test_commands_for_outputs_passes_swap_sides_through(self):
        out = tl.commands_for_outputs({1: "LeftSpeed", 2: "RightSpeed"}, -100, 100, 50, False, swap_sides=True)
        self.assertEqual(out, {1: 50, 2: 50})


if __name__ == "__main__":
    unittest.main()
