"""
controller/alerts.py
Append-only JSON-lines alert log shared by the controller (anomaly /
pair_escalation / forced_drop alerts) and controller/dpi.py (confirmed
alerts). Each alert is one write() of one line on an O_APPEND descriptor,
so lines from the two processes never interleave.

AlertTailer lets the controller read DPI's confirmed alerts back out of the
same file without a second IPC channel: it is a plain read-only follower
that remembers how many bytes of the file it has already consumed.
"""

import json
import os
import threading
from typing import List


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


class AlertTailer:
    """
    Returns new complete JSON lines appended to `path` since the last call.
    Only consumes bytes up to the last newline in the file, so a line the
    writer is still in the middle of appending is left for the next call
    rather than parsed as a truncated fragment.
    """

    def __init__(self, path: str):
        self.path = path
        self._pos = 0

    def read_new(self) -> List[dict]:
        try:
            with open(self.path, "rb") as f:
                f.seek(self._pos)
                data = f.read()
        except OSError:
            return []
        nl = data.rfind(b"\n")
        if nl == -1:
            return []
        complete = data[:nl + 1]
        self._pos += len(complete)
        out = []
        for line in complete.splitlines():
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out
