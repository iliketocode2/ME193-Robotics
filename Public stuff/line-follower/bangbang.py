"""Bang-bang (two-state) line following: the pure control rule, no browser needed.

The Bang bang tab shows `decide` as its code card, so keep it short and readable."""


def clamp(speed):
    return max(-100, min(100, speed))


def target(on_line, off_line):
    return (on_line + off_line) / 2


def decide(reflection, on_line, off_line, follow_right, base, turn):
    # the reflection halfway between "on the line" and "off the line"
    goal = target(on_line, off_line)

    # is the sensor on the line's side of that goal?
    if on_line < off_line:
        on_the_line = reflection < goal
    else:
        on_the_line = reflection > goal

    # bang-bang: a full turn one way or the other, nothing in between
    steer = turn if on_the_line else -turn
    if not follow_right:
        steer = -steer

    left = base + steer
    right = base - steer
    return clamp(left), clamp(right)
