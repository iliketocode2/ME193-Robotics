"""Copies the shared lock files into the UNO Q app, and optionally sends the app to the board.

    python uno_q/sync_app.py                          # only refresh the copies in uno_q/finger_lock/python
    python uno_q/sync_app.py --host 192.168.1.42      # also copy the app to the board over SSH and start it

The board needs Wi-Fi and SSH turned on (App Lab's first-time setup does both). The user is `arduino` and the
password is the one you picked in that setup. Run this again after retraining so the board gets the new weights.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
APP = Path(__file__).resolve().parent / "finger_lock"
SHARED = ["config.py", "camera.py", "npnet.py", "lock_policy.py", "models/finger_cnn.npz"]
REMOTE_DIR = "ArduinoApps/finger_lock"


def sync():
    for rel in SHARED:
        dst = APP / "python" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / rel, dst)
        print("copied", rel)
    shutil.rmtree(APP / "python" / "__pycache__", ignore_errors=True)


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def deploy(host):
    target = f"arduino@{host}"
    run(["ssh", target, f"mkdir -p ~/{REMOTE_DIR}"])
    run(["scp", "-r", str(APP / "app.yaml"), str(APP / "python"), str(APP / "sketch"), f"{target}:~/{REMOTE_DIR}/"])
    run(["ssh", target, f"arduino-app-cli app stop ~/{REMOTE_DIR}; arduino-app-cli app start ~/{REMOTE_DIR}"])
    print(f"\nStarted. Watch it with:  ssh {target} arduino-app-cli app logs ~/{REMOTE_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="the board's IP address; leave it out to only refresh the copies")
    args = ap.parse_args()
    sync()
    if args.host:
        deploy(args.host)


if __name__ == "__main__":
    main()
