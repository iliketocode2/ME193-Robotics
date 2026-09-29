#!/usr/bin/env python3
"""PostToolUse hook (Write|Edit): syntax-check any .py file Claude just wrote.

This repo has no automated test suite (hardware tests need physical BLE
devices), so a fast syntax check is the closest thing to a "verification
loop" available for most edits. Exit 2 feeds stderr back to Claude as a
blocking error; exit 0 is silent success.
"""
import json
import os
import subprocess
import sys


def find_python():
    for candidate in ("my_env/Scripts/python.exe", "my_env/bin/python", "my_env/bin/python3"):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    tool_input = data.get("tool_input") or {}
    tool_response = data.get("tool_response") or {}
    file_path = tool_input.get("file_path") or tool_response.get("filePath")

    if not file_path or not str(file_path).endswith(".py"):
        return 0
    if not os.path.isfile(file_path):
        return 0

    py = find_python()
    if py is None:
        return 0  # no venv found here -- nothing to check against

    result = subprocess.run(
        [py, "-m", "py_compile", file_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"Syntax check failed for {file_path}:\n"
            f"{result.stderr or result.stdout}"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
