"""
AnnaCode — activations.py

Activation math. Every function here takes all its inputs as arguments and
returns a value -- no DOM access, no mutation of network topology state.
The one exception is reading state.CUSTOM_ACT_NAMES, a fixed constant table
(not app state), for the custom-expression eval namespace.
"""
import math
import re
import traceback
import numpy as np

import state

_CALLABLE_NAMES = {k for k, v in state.CUSTOM_ACT_NAMES.items() if callable(v)}
_KNOWN_NAMES = set(state.CUSTOM_ACT_NAMES) | {"x"}
_TOKEN_RE = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|\d+\.\d*|\.\d+|\d+|\*\*|.')
_TRAILING_DIGITS_RE = re.compile(r'^([A-Za-z_]+)(\d+)$')

def sigmoid_numpy(x):
    return 1.0 / (1.0 + np.exp(-x))

def apply_activation(x: float, fn: str, custom_activation: dict | None = None) -> float:
    if fn == "relu":
        return max(0.0, x)
    elif fn == "sigmoid":
        return sigmoid_numpy(x)
    elif fn == "tanh":
        return math.tanh(x)
    elif fn == "softplus":
        return math.log(1.0 + math.exp(x))
    elif fn == "custom":
        return apply_custom_activation(x, custom_activation or {"expr": "x", "pieces": []})
    else:
        return x

def _tokenize(s: str):
    """Split into (kind, text) pairs, keeping known names like `log2` intact
    instead of splitting them at the digit boundary."""
    tokens = []
    for tok in _TOKEN_RE.findall(s):
        c = tok[0]
        if c.isalpha() or c == "_":
            if tok not in _KNOWN_NAMES:
                m = _TRAILING_DIGITS_RE.match(tok)
                if m and m.group(1) in _KNOWN_NAMES:
                    tokens.append(("NAME", m.group(1)))
                    tokens.append(("NUM", m.group(2)))
                    continue
            tokens.append(("NAME", tok))
        elif c.isdigit() or c == ".":
            tokens.append(("NUM", tok))
        elif tok == "(":
            tokens.append(("LPAREN", tok))
        elif tok == ")":
            tokens.append(("RPAREN", tok))
        else:
            tokens.append(("OP", tok))
    return tokens

def normalize_expr(expr: str) -> str:
    """Insert implicit-multiplication `*` (e.g. `3x`, `2(x+1)`) while leaving
    calls to known functions like `abs(x)` or `log2(x)` untouched."""
    cleaned = expr.replace("^", "**")
    tokens = _tokenize(cleaned)
    out = []
    prev_kind = prev_val = None
    for kind, val in tokens:
        if prev_kind is not None:
            need_mult = (
                (prev_kind == "NUM" and kind in ("NAME", "LPAREN")) or
                (prev_kind in ("NAME", "RPAREN") and kind == "NUM") or
                (prev_kind == "RPAREN" and kind == "LPAREN") or
                (prev_kind == "NAME" and kind == "LPAREN" and prev_val not in _CALLABLE_NAMES)
            )
            if need_mult:
                out.append("*")
        out.append(val)
        prev_kind, prev_val = kind, val
    return "".join(out)

def safe_eval_expr(expr: str, x: float) -> float:
    if not expr or not expr.strip():
        return x
    cleaned = normalize_expr(expr)
    ns = dict(state.CUSTOM_ACT_NAMES)
    ns["x"] = x
    try:
        return float(eval(cleaned, {"__builtins__": {}}, ns))
    except Exception:
        print("Custom activation eval error:\n" + traceback.format_exc())
        return x

def parse_bound(raw: str):
    if raw is None:
        return None
    s = raw.strip().lower().replace("∞", "inf")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def apply_custom_activation(x: float, custom_activation: dict) -> float:
    pieces = custom_activation["pieces"]
    if not pieces:
        return safe_eval_expr(custom_activation["expr"], x)
    for p in pieces:
        lo, hi = parse_bound(p["lo"]), parse_bound(p["hi"])

        if lo is None:
            lo_ok = True
        elif p["lo_op"] == "<":
            lo_ok = lo < x
        elif p["lo_op"] == "<=":
            lo_ok = lo <= x
        elif p["lo_op"] == ">":
            lo_ok = lo > x
        elif p["lo_op"] == ">=":
            lo_ok = lo >= x
        else:
            lo_ok = True

        if hi is None:
            hi_ok = True
        elif p["hi_op"] == "<":
            hi_ok = x < hi
        elif p["hi_op"] == "<=":
            hi_ok = x <= hi
        elif p["hi_op"] == ">":
            hi_ok = x > hi
        elif p["hi_op"] == ">=":
            hi_ok = x >= hi
        else:
            hi_ok = True

        if lo_ok and hi_ok:
            return safe_eval_expr(p["expr"], x)
    return 0.0
