#!/usr/bin/env python3
"""Copy the Jim app to the UNO Q and (re)start it.

    python tools/deploy.py              deploy unoq_app/ as ~/ArduinoApps/jim, start it, show logs
    python tools/deploy.py --logs       just follow the running app's logs
    python tools/deploy.py --stop       stop the app
    python tools/deploy.py --cmd "..."  run a command on the board

The board address comes from the JIM_HOST environment variable, else
tools/board_host.txt (one line, e.g. 192.168.1.251; gitignored), else
jim.local. Needs passwordless ssh (see DevelopingUNOQAppsInVSCode.md, step 2).
"""

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent / "unoq_app"
REMOTE_APP = "/home/arduino/ArduinoApps/jim"
SSH_OPTS = ["-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]


def host():
    if os.environ.get("JIM_HOST"):
        return os.environ["JIM_HOST"]
    f = HERE / "board_host.txt"
    if f.exists() and f.read_text().strip():
        return f.read_text().strip()
    return "jim.local"


def ssh(command, stdin=None, check=True, stream=True):
    cmd = ["ssh", *SSH_OPTS, f"arduino@{host()}", command]
    print("$", command, flush=True)
    r = subprocess.run(cmd, input=stdin, check=False,
                       stdout=None if stream else subprocess.PIPE)
    if check and r.returncode != 0:
        sys.exit(f"command failed on the board (exit {r.returncode})")
    return r


def app_tarball():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(APP_DIR.rglob("*")):
            rel = path.relative_to(APP_DIR)
            if "__pycache__" in rel.parts or path.suffix == ".pyc":
                continue
            tar.add(path, arcname=str(rel).replace("\\", "/"), recursive=False)
    return buf.getvalue()


def main():
    args = sys.argv[1:]
    if args[:1] == ["--cmd"]:
        sys.exit(ssh(" ".join(args[1:]), check=False).returncode)
    if args[:1] == ["--logs"]:
        ssh(f"arduino-app-cli app logs {REMOTE_APP} --follow", check=False)
        return
    if args[:1] == ["--stop"]:
        ssh(f"arduino-app-cli app stop {REMOTE_APP}", check=False)
        return

    ssh(f"arduino-app-cli app stop {REMOTE_APP} >/dev/null 2>&1; true", check=False)
    data = app_tarball()
    print(f"copying {len(data) // 1024} KB to {REMOTE_APP}", flush=True)
    # Replace python/ and sketch/ wholesale so deleted files don't linger on
    # the board; keep .cache (App Lab's installed Python packages).
    ssh(f"mkdir -p {REMOTE_APP} && rm -rf {REMOTE_APP}/python {REMOTE_APP}/sketch "
        f"&& tar xzf - -C {REMOTE_APP}", stdin=data)
    ssh(f"arduino-app-cli app start {REMOTE_APP}")
    print("\nfollowing logs (Ctrl+C to stop watching; the app keeps running)\n", flush=True)
    try:
        ssh(f"timeout 60 arduino-app-cli app logs {REMOTE_APP} --follow", check=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
