"""
UNO Q app (Linux side): streams the sketch's IMU batches to laptops over TCP.

The sketch (copied unchanged from Project 7) reads the MPU-6050 at 100 Hz and
sends 10 raw samples every 100 ms. A laptop connects to TCP port 5005 (App Lab
only forwards TCP) and receives one line per message:

    "<seq>|ax,ay,az,gx,gy,gz;ax,ay,...;"     raw counts, seq += 1 per batch
    "status|ok"  or  "status|missing | <I2C scan>"
"""

import socket
import threading

from arduino.app_utils import App, Bridge

PORT = 5005

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(4)
clients = []
lock = threading.Lock()
seq = 0
imu_status = "waiting"


def send(line):
    data = (line + "\n").encode()
    with lock:
        for c in list(clients):
            try:
                c.sendall(data)
            except OSError:
                clients.remove(c)
                c.close()


def on_imu_batch(payload):
    global seq
    seq += 1
    send(f"{seq}|{payload}")


def on_imu_status(message):
    global imu_status
    imu_status = str(message)
    send(f"status|{imu_status}")


def accept():
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(1.0)
        try:
            conn.sendall(f"status|{imu_status}\n".encode())
        except OSError:
            conn.close()
            continue
        with lock:
            clients.append(conn)
        print(f"[stream] streaming to {addr[0]}", flush=True)


Bridge.provide("imu_batch", on_imu_batch)
Bridge.provide("imu_status", on_imu_status)
threading.Thread(target=accept, daemon=True).start()
App.run()
