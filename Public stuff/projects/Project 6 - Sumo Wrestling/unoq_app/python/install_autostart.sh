#!/bin/bash
# Make the BLE relay start by itself whenever the UNO Q boots (battery pack, no laptop).
#
# Why: the App Lab "default app" only starts python/main.py inside App Lab's
# container, which can't use Bluetooth. The relay (relay.py, via run_on_host.sh)
# must run on the board's own Linux, so this installs it as a systemd service:
# starts at boot, waits for Wi-Fi + Bluetooth, restarts itself if it ever crashes.
#
# Run ONCE, from a terminal on the board, inside this python/ folder:
#     bash install_autostart.sh            # install + start now (asks for the arduino password)
#     bash install_autostart.sh --remove   # uninstall
#
# Afterwards:
#     sudo systemctl status sumo-relay     # is it running?
#     journalctl -u sumo-relay -f          # live log ([relay] double motor connected ...)
#     sudo systemctl stop sumo-relay       # stop it (e.g. before running run_on_host.sh by hand --
#                                          #  two relays would fight over the LEGO devices)
#     sudo systemctl start sumo-relay      # start it again
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"
SERVICE=/etc/systemd/system/sumo-relay.service

if [ "$1" = "--remove" ]; then
    sudo systemctl disable --now sumo-relay.service 2>/dev/null || true
    sudo rm -f "$SERVICE"
    sudo systemctl daemon-reload
    echo "[autostart] removed"
    exit 0
fi

# the venv must exist before boot-time runs (creating it needs internet once)
if [ ! -x "$HOME/sumo-venv/bin/pip" ]; then
    echo "[autostart] setting up ~/sumo-venv first (one time, needs internet)..."
    bash run_on_host.sh -c "print('venv ready')" 2>/dev/null || true
    if [ ! -x "$HOME/sumo-venv/bin/pip" ]; then
        echo "[autostart] venv setup failed -- run 'bash run_on_host.sh ble_probe_unoq.py' once and retry"
        exit 1
    fi
fi

echo "[autostart] installing $SERVICE (relay in $HERE)"
sudo tee "$SERVICE" >/dev/null <<EOF
[Unit]
Description=Sumo robot MQTT <-> BLE relay (LEGO double motor + color sensor)
Wants=network-online.target bluetooth.target
After=network-online.target bluetooth.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$HERE
ExecStart=/bin/bash $HERE/run_on_host.sh
Restart=always
RestartSec=5
# SIGTERM -> relay.py stops the motors and disconnects before exiting
KillSignal=SIGTERM
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now sumo-relay.service
sleep 3
sudo systemctl --no-pager --lines=8 status sumo-relay.service || true
echo
echo "[autostart] done: the relay now starts on every boot."
echo "[autostart] watch it with:  journalctl -u sumo-relay -f"
