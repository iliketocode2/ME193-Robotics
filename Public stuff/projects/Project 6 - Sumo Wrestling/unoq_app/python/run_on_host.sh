#!/bin/bash
# Run the BLE relay (or the probe) on the UNO Q's Linux HOST, outside App Lab.
#
# Why: App Lab runs python/main.py in a container with no access to the
# board's Bluetooth service (ble_probe showed "D-Bus system socket: FAIL"),
# so bleak can't scan from there. The host has it. The App Lab app keeps
# running alongside (RELAY_BLE = False) and only reports the fan RPM.
#
# Usage, from a terminal ON THE BOARD (App Lab terminal or SSH):
#   bash run_on_host.sh ble_probe_unoq.py --drive   # test first (robot on blocks!)
#   bash run_on_host.sh                             # run the relay (Ctrl-C to stop)
#
# First run creates ~/sumo-venv and installs requirements.txt into it
# (Debian blocks plain "pip install --user" system-wide, hence the venv).
# No sudo needed: the board lacks Debian's python3-venv package (no ensurepip),
# so if a normal venv fails we make one without pip and bootstrap pip from
# the official get-pip.py instead.
set -e
cd "$(dirname "$0")"
VENV="$HOME/sumo-venv"
if [ ! -x "$VENV/bin/pip" ]; then
    rm -rf "$VENV"                                   # clear any half-made venv from a failed run
    echo "[run_on_host] creating $VENV ..."
    if ! python3 -m venv "$VENV" 2>/dev/null; then
        echo "[run_on_host] no ensurepip on this board -> venv without pip + get-pip.py"
        rm -rf "$VENV"
        python3 -m venv --without-pip "$VENV"
        "$VENV/bin/python" -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')"
        "$VENV/bin/python" /tmp/get-pip.py
    fi
    "$VENV/bin/pip" install -r requirements.txt
fi
SCRIPT="${1:-relay.py}"
shift || true
exec "$VENV/bin/python" -u "$SCRIPT" "$@"
