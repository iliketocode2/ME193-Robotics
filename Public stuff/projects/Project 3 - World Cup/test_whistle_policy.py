"""
Synthetic-signal self-check for whistle_policy.py -- no microphone, no BLE
hardware needed. Feeds WhistlePolicy hand-built numpy blocks (clean tones,
chirps, white noise, pulse trains) and checks the Command it produces
matches what the policy is supposed to do.

This exists because the repo otherwise has no automated test suite (real
hardware tests need physical BLE devices in range) -- but the DSP/decision
logic here doesn't touch hardware at all, so it's exactly the kind of thing
that CAN be verified before ever touching the robot.

Run:
    my_env_audio/Scripts/python "Public stuff/projects/Project 3 - World Cup/test_whistle_policy.py"
"""

import numpy as np

from whistle_policy import DEFAULT_CONFIG, WhistlePolicy

SR = DEFAULT_CONFIG["sample_rate"]
BLOCK = DEFAULT_CONFIG["block_size"]
DT = BLOCK / SR


def tone(freq, n=BLOCK, sr=SR, amplitude=0.5):
    t = np.arange(n) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(n=BLOCK):
    return np.zeros(n, dtype=np.float32)


def noise(n=BLOCK, amplitude=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


def check(name, condition):
    print(f"[{'OK' if condition else 'FAIL'}] {name}")
    assert condition, name


def test_stop_band():
    policy = WhistlePolicy()
    lo, hi = DEFAULT_CONFIG["stop_band"]
    mid, now, cmd = (lo + hi) / 2, 0.0, None
    for _ in range(5):
        cmd = policy.update(tone(mid), now=now)
        now += DT
    check("stop band -> zero forward speed", cmd.forward_speed == 0.0)
    check("stop band -> zero turn", cmd.turn_bias == 0.0)
    check("stop band -> state reports STOP", cmd.state == "STOP")


def test_forward_band_scales_with_pitch():
    # +/-100Hz margin from the band edges: the FFT bin width here is
    # SR/block_size =~ 21.5Hz, so a test tone placed within one bin of a
    # band boundary can snap to either side of it after EMA smoothing --
    # not a policy bug, just not a fair edge case for a discrete FFT.
    lo, hi = DEFAULT_CONFIG["forward_band"]

    policy_low, now, cmd_low = WhistlePolicy(), 0.0, None
    for _ in range(5):
        cmd_low = policy_low.update(tone(lo + 100), now=now)
        now += DT

    policy_high, now, cmd_high = WhistlePolicy(), 0.0, None
    for _ in range(5):
        cmd_high = policy_high.update(tone(hi - 100), now=now)
        now += DT

    check("higher pitch in forward band -> faster forward speed",
          cmd_high.forward_speed > cmd_low.forward_speed)
    check("forward speed never exceeds the configured max",
          cmd_high.forward_speed <= DEFAULT_CONFIG["max_speed"])


def test_turn_is_a_gradient_of_pitch_position():
    # +/-100Hz margin from the band edges, same FFT-bin-quantization reasoning
    # as test_forward_band_scales_with_pitch.
    lo, hi = DEFAULT_CONFIG["turn_band"]
    mid = (lo + hi) / 2

    policy_low, now, cmd_low = WhistlePolicy(), 0.0, None
    for _ in range(5):
        cmd_low = policy_low.update(tone(lo + 100), now=now)
        now += DT
    check("low end of turn band -> turns LEFT (negative)", cmd_low.turn_bias < 0)

    policy_high, now, cmd_high = WhistlePolicy(), 0.0, None
    for _ in range(5):
        cmd_high = policy_high.update(tone(hi - 100), now=now)
        now += DT
    check("high end of turn band -> turns RIGHT (positive)", cmd_high.turn_bias > 0)

    policy_mid, now, cmd_mid = WhistlePolicy(), 0.0, None
    for _ in range(5):
        cmd_mid = policy_mid.update(tone(mid), now=now)
        now += DT
    check("center of turn band -> ~straight (near zero turn)", abs(cmd_mid.turn_bias) < 5)

    policy_quarter, now, cmd_quarter = WhistlePolicy(), 0.0, None
    for _ in range(5):
        cmd_quarter = policy_quarter.update(tone(lo + (hi - lo) * 0.25), now=now)
        now += DT
    check("a steady pitch a quarter into the band turns less sharply than the low edge",
          cmd_quarter.turn_bias < 0 and abs(cmd_quarter.turn_bias) < abs(cmd_low.turn_bias))


def test_broadband_noise_is_masked():
    policy, now, cmd = WhistlePolicy(), 0.0, None
    for _ in range(8):
        cmd = policy.update(noise(seed=1), now=now)
        now += DT
    check("broadband noise never registers as a whistle", cmd.frequency is None)
    check("noise -> no drive command produced", cmd.forward_speed == 0.0 and cmd.turn_bias == 0.0)


def test_no_whistle_ramps_down_instead_of_cutting_instantly():
    lo, hi = DEFAULT_CONFIG["forward_band"]
    policy, now, cmd = WhistlePolicy(), 0.0, None
    for _ in range(5):
        cmd = policy.update(tone(hi - 100), now=now)
        now += DT
    check("driving fast right before going silent", cmd.forward_speed > 50)

    cmd_right_after = policy.update(silence(), now=now)
    now += DT
    check("no instant cut the moment silence starts (grace period)",
          cmd_right_after.forward_speed > 0)

    elapsed = 0.0
    horizon = DEFAULT_CONFIG["silence_timeout_s"] + DEFAULT_CONFIG["ramp_time_s"] + 0.2
    while elapsed < horizon:
        cmd = policy.update(silence(), now=now)
        now += DT
        elapsed += DT
    check("eventually ramps all the way down to a full stop", cmd.forward_speed == 0.0)


def test_goal_gesture_three_high_pulses():
    _, hi = DEFAULT_CONFIG["forward_band"][0], DEFAULT_CONFIG["forward_band"][1]
    policy, now, cmd = WhistlePolicy(), 0.0, None
    for _ in range(3):
        cmd = policy.update(tone(hi - 100), now=now); now += DT
        cmd = policy.update(tone(hi - 100), now=now); now += DT
        cmd = policy.update(silence(), now=now); now += 0.15
    check("3 short high-band pulses -> goal_detected fires", cmd.goal_detected)


if __name__ == "__main__":
    test_stop_band()
    test_forward_band_scales_with_pitch()
    test_turn_is_a_gradient_of_pitch_position()
    test_broadband_noise_is_masked()
    test_no_whistle_ramps_down_instead_of_cutting_instantly()
    test_goal_gesture_three_high_pulses()
    print("\nAll whistle_policy self-checks passed.")
