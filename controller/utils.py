"""
controller/utils.py
Shared utility functions for AdaptMon controller.
"""

import socket
import struct
import logging
import os
import json
import time
import subprocess
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


def ip_int_to_str(ip_int: int) -> str:
    """Convert a 32-bit integer to dotted-decimal IP string."""
    try:
        return socket.inet_ntoa(struct.pack(">I", ip_int & 0xFFFFFFFF))
    except Exception:
        return "0.0.0.0"


def ip_str_to_int(ip_str: str) -> int:
    """Convert dotted-decimal IP string to 32-bit integer."""
    try:
        return struct.unpack(">I", socket.inet_aton(ip_str))[0]
    except Exception:
        return 0


def flow_key(src_ip: str, dst_ip: str, protocol: int) -> str:
    """Create a canonical string key for a flow."""
    return f"{src_ip}:{dst_ip}:{protocol}"


def parse_flow_key(key: str) -> Dict[str, Any]:
    """Parse flow key back into components."""
    parts = key.split(":")
    if len(parts) == 3:
        return {
            "src_ip"  : parts[0],
            "dst_ip"  : parts[1],
            "protocol": int(parts[2]),
        }
    return {}


def setup_logging(
    log_dir  : str,
    log_file : str  = "controller.log",
    level    : int  = logging.INFO
) -> None:
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Only add handlers if none exist yet (prevents duplicate log lines)
    if not root.handlers:
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        root.addHandler(ch)
    else:
        # Just add the new file handler if not already logging to this file
        existing_files = [
            h.baseFilename for h in root.handlers
            if isinstance(h, logging.FileHandler)
        ]
        if log_path not in existing_files:
            fh = logging.FileHandler(log_path)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            root.addHandler(fh)

    logger.info(f"Logging to {log_path}")


def wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    """Wait until a TCP port is open."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(1)
    return False


def run_command(
    cmd     : List[str],
    timeout : int  = 60,
    check   : bool = False
) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    logger.debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check
    )


def save_json(data: Any, path: str) -> None:
    """Save data as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Optional[Any]:
    """Load JSON data."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def human_readable_bytes(n: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def get_timestamp_str() -> str:
    """Return current timestamp as string."""
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dirs(paths: List[str]) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


class RollingStats:
    """Compute rolling statistics over a window."""

    def __init__(self, window: int = 60):
        self.window  = window
        self._values : List[float] = []

    def update(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self.window:
            self._values.pop(0)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def std(self) -> float:
        import math
        if len(self._values) < 2:
            return 0.0
        m = self.mean
        variance = sum((x - m) ** 2 for x in self._values) / len(self._values)
        return math.sqrt(variance)

    @property
    def max(self) -> float:
        return max(self._values) if self._values else 0.0

    @property
    def min(self) -> float:
        return min(self._values) if self._values else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std" : self.std,
            "max" : self.max,
            "min" : self.min,
            "n"   : len(self._values),
        }


class ExperimentTimer:
    """Simple context manager for timing experiments."""

    def __init__(self, name: str):
        self.name  = name
        self.start = 0.0
        self.end   = 0.0

    def __enter__(self):
        self.start = time.time()
        logger.info(f"[TIMER] {self.name} started")
        return self

    def __exit__(self, *args):
        self.end = time.time()
        logger.info(
            f"[TIMER] {self.name} finished in "
            f"{self.elapsed:.2f}s"
        )

    @property
    def elapsed(self) -> float:
        if self.end > 0:
            return self.end - self.start
        return time.time() - self.start
