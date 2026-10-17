"""
controller/alerts.py
Append-only JSON-lines alert log shared by the controller (anomaly alerts)
and controller/dpi.py (confirmed alerts). Each alert is one write() of one
line on an O_APPEND descriptor, so lines from the two processes never
interleave.
"""

import json
import os
import threading


class AlertLog:

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self._lock = threading.Lock()
        self.n_written = 0

    def write(self, alert: dict) -> None:
        line = (json.dumps(alert, separators=(",", ":"), default=str) + "\n").encode()
        with self._lock:
            os.write(self._fd, line)
            self.n_written += 1

    def close(self) -> None:
        os.close(self._fd)
