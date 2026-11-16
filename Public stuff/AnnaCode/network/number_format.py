"""How a weight or bias is written in its box (pure Python, unit-tested; see tests/test_number_format.py).

Two decimals hide small weights: a perfectly good weight of 0.004 shows as "0.00", which looks like a
zero (dead) weight, and 0.0174 would show as "0.02". So values of 0.1 and up keep two decimals, and
smaller ones get three significant digits. Only a true zero is written "0".
"""
import math


def fmt_coeff(value) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "0"
    if x == 0.0 or math.isnan(x) or math.isinf(x):
        return "0" if x == 0.0 or math.isnan(x) else ("inf" if x > 0 else "-inf")
    if abs(x) >= 0.1:
        return f"{x:.2f}"
    return f"{x:.3g}"
