#!/usr/bin/env python3
"""PreToolUse hook (Bash): refuse a `git add`/`git commit` that would stage
a file over 5MB.

This repo has no Git LFS set up, and the CNN motor-control work will soon
produce camera-frame datasets and possibly model checkpoints -- an
accidental large-binary commit is a real risk once that starts.
"""
import json
import os
import re
import subprocess
import sys

MAX_BYTES = 5 * 1024 * 1024


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    cmd = (data.get("tool_input") or {}).get("command") or ""
    if not re.search(r"\bgit\s+(add|commit)\b", cmd):
        return 0

    offenders = []

    add_match = re.search(r"\bgit\s+add\b(.*)", cmd)
    if add_match:
        for tok in add_match.group(1).split():
            if tok.startswith("-"):
                continue
            if os.path.isfile(tok):
                size = os.path.getsize(tok)
                if size > MAX_BYTES:
                    offenders.append(f"{tok} ({size // 1024}KB, about to be added)")

    if re.search(r"\bgit\s+commit\b", cmd):
        try:
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True, text=True, check=False,
            ).stdout.splitlines()
        except FileNotFoundError:
            staged = []
        for path in staged:
            if path and os.path.isfile(path):
                size = os.path.getsize(path)
                if size > MAX_BYTES:
                    offenders.append(f"{path} ({size // 1024}KB, already staged)")

    if offenders:
        reason = (
            "Blocked: this git command would commit file(s) over 5MB with no Git LFS "
            "set up in this repo: " + "; ".join(offenders) + ". This repo only tracks "
            "'Public stuff/', '.gitignore', 'CLAUDE.md' and '.claude/' -- large "
            "datasets/model checkpoints from the CNN work likely don't belong in git "
            "at all. Set up Git LFS first, or get explicit user confirmation before "
            "proceeding."
        )
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
