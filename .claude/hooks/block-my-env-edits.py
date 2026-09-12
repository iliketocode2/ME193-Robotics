#!/usr/bin/env python3
"""PreToolUse hook (Write|Edit): refuse edits to anything under my_env/.

my_env/ is a vendored virtualenv -- installed third-party package source
(legoeducation, bleak) lives there and should be treated as read-only
reference material, never hand-edited.
"""
import json
import sys


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    file_path = (data.get("tool_input") or {}).get("file_path") or ""
    normalized = str(file_path).replace("\\", "/")

    if normalized.startswith("my_env/") or "/my_env/" in normalized:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "my_env/ is the vendored virtualenv (installed legoeducation/bleak "
                    "source) -- read-only reference material, never hand-edited. "
                    "Edit the actual project file instead."
                ),
            }
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
