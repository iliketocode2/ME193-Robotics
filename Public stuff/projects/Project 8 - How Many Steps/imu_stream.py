"""Receives the UNO Q's IMU stream over TCP (see unoq_app/python/main.py)."""

import socket
import time

import numpy as np

import config

RETRY_S = 1.0


class ImuStream:
    """Iterate batches() to get lists of (t, sample): t in seconds from the
    board's sample clock (10 ms per sample, gaps where batches were lost),
    sample = 6 floats [ax, ay, az] in g and [gx, gy, gz] in deg/s.
    Reconnects on its own; yields [] about 5 times a second while nothing arrives."""

    def __init__(self, host=config.BOARD_HOST, port=config.BOARD_PORT):
        self.addr = (host, port)
        self.sock = None
        self.status = "connecting"
        self.lost_batches = 0
        self._buf = b""
        self._next_try = 0.0
        self._first_seq = None
        self._last_seq = None

    def _connect(self):
        if time.time() < self._next_try:
            time.sleep(0.2)
            return False
        self._next_try = time.time() + RETRY_S
        try:
            self.sock = socket.create_connection(self.addr, timeout=2.0)
        except OSError:
            self.status = f"can't reach {self.addr[0]}:{self.addr[1]}"
            self.sock = None
            return False
        self.sock.settimeout(0.2)
        self._buf = b""
        return True

    def _parse(self, line):
        head, _, body = line.partition("|")
        if head == "status":
            self.status = body
            return []
        try:
            seq = int(head)
        except ValueError:
            return []
        self.status = "ok"
        if self._first_seq is None:
            self._first_seq = seq
        elif seq > self._last_seq + 1:
            self.lost_batches += seq - self._last_seq - 1
        self._last_seq = seq
        rows = [r.split(",") for r in body.split(";") if r.count(",") == 5]
        out = []
        for i, r in enumerate(rows):
            v = np.array(r, dtype=float)
            v[:3] /= config.ACC_COUNTS_PER_G
            v[3:] /= config.GYRO_COUNTS_PER_DPS
            out.append((((seq - self._first_seq) * len(rows) + i) / config.SAMPLE_HZ, v))
        return out

    def batches(self):
        while True:
            if self.sock is None and not self._connect():
                yield []
                continue
            try:
                chunk = self.sock.recv(8192)
            except socket.timeout:
                yield []
                continue
            except OSError:
                chunk = b""
            if not chunk:
                self.sock.close()
                self.sock = None
                self.status = "connection lost, reconnecting"
                yield []
                continue
            self._buf += chunk
            *lines, self._buf = self._buf.split(b"\n")
            out = []
            for line in lines:
                out += self._parse(line.decode(errors="replace"))
            yield out

    def close(self):
        if self.sock:
            self.sock.close()
