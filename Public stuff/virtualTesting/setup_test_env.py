#!/usr/bin/env python3
"""Spin up a throwaway virtual environment for trying out `legoeducation`.

Creates ./test/, builds a venv inside it, installs `legoeducation`, and
opens a Terminal window already `cd`'d into that folder with the venv
activated — just type `python3` there to start a REPL with `legoeducation`
ready to import.

macOS only (uses Terminal.app via AppleScript). On other platforms this
still creates and provisions the venv, it just can't auto-open a terminal
for you.
"""

import platform
import shlex
import subprocess
import venv
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent / "test"
VENV_DIR = TEST_DIR / ".venv"


def create_venv():
    TEST_DIR.mkdir(exist_ok=True)
    if VENV_DIR.exists():
        print(f"Virtual environment already exists at {VENV_DIR}, reusing it.")
    else:
        print(f"Creating virtual environment at {VENV_DIR} ...")
        venv.create(VENV_DIR, with_pip=True)


def install_legoeducation():
    venv_python = VENV_DIR / "bin" / "python3"
    print("Installing legoeducation ...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "legoeducation"],
        check=True,
    )


def open_terminal():
    if platform.system() != "Darwin":
        print(
            "\nNot on macOS, so I can't auto-open a terminal for you. "
            f"Open one yourself, cd into {TEST_DIR}, run "
            "`source .venv/bin/activate`, then `python3`."
        )
        return

    activate = VENV_DIR / "bin" / "activate"
    shell_cmd = (
        f"cd {shlex.quote(str(TEST_DIR))} && "
        f"source {shlex.quote(str(activate))} && clear"
    )
    # Escape for embedding inside an AppleScript string literal.
    as_escaped = shell_cmd.replace("\\", "\\\\").replace('"', '\\"')
    script = f'tell application "Terminal" to do script "{as_escaped}"'
    subprocess.run(["osascript", "-e", script], check=True)
    subprocess.run(
        ["osascript", "-e", 'tell application "Terminal" to activate'],
        check=True,
    )


def main():
    create_venv()
    install_legoeducation()
    open_terminal()
    print(
        "\nDone. In the new Terminal window the virtual environment is "
        "already active — just run `python3` to start a REPL with "
        "legoeducation installed."
    )


if __name__ == "__main__":
    main()
