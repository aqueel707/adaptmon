"""
traffic/normal.py
Normal background traffic generator using iperf3 and ping.
Generates realistic TCP/UDP flows between hosts.
"""

import subprocess
import threading
import time
import random
import logging
import os
import signal
import socket
import struct
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Default host pairs for traffic generation
DEFAULT_HOST_PAIRS = [
    ("10.0.1.1", "10.0.2.1"),
    ("10.0.1.2", "10.0.2.2"),
    ("10.0.1.1", "10.0.2.2"),
    ("10.0.1.2", "10.0.2.1"),
]

def _detect_iperf() -> str:
    """Auto-detect available iperf binary."""
    for candidate in ["iperf3", "iperf", "iperf3.exe"]:
        try:
            r = subprocess.run([candidate, "--version"],
                               capture_output=True, timeout=3)
            if r.returncode == 0 or b"iperf" in r.stderr.lower() or b"iperf" in r.stdout.lower():
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None

IPERF3_BIN = os.environ.get("IPERF3_BIN") or _detect_iperf()
IPERF_BIN  = IPERF3_BIN  # alias


class IperfServer:
    """Manages an iperf3 server process on a given host."""

    def __init__(
        self,
        host      : str,
        port      : int  = 5201,
        bind_ip   : str  = "0.0.0.0",
        namespace : str  = None,
    ):
        self.host      = host
        self.port      = port
        self.bind_ip   = bind_ip
        self.namespace = namespace
        self._proc     : Optional[subprocess.Popen] = None
        self._log_path = f"/tmp/iperf3_server_{host}_{port}.log"

    def start(self) -> bool:
        cmd = [IPERF3_BIN, "-s", "-p", str(self.port),
               "-B", self.bind_ip, "--one-off"]

        if self.namespace:
            cmd = ["ip", "netns", "exec", self.namespace] + cmd

        logger.debug(f"Starting iperf3 server: {' '.join(cmd)}")
        try:
            with open(self._log_path, "w") as lf:
                self._proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=lf,
                    preexec_fn=os.setsid
                )
            time.sleep(0.5)
            if self._proc.poll() is None:
                logger.info(
                    f"iperf3 server started at {self.host}:{self.port}"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to start iperf3 server: {e}")
        return False

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except Exception:
                pass
        self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class IperfFlow:
    """Represents a single iperf3 client flow."""

    def __init__(
        self,
        src_ip      : str,
        dst_ip      : str,
        dst_port    : int   = 5201,
        duration    : int   = 30,
        bandwidth   : str   = "10M",
        protocol    : str   = "tcp",   # tcp or udp
        parallel    : int   = 1,
        namespace   : str   = None,
    ):
        self.src_ip   = src_ip
        self.dst_ip   = dst_ip
        self.dst_port = dst_port
        self.duration = duration
        self.bandwidth= bandwidth
        self.protocol = protocol
        self.parallel = parallel
        self.namespace= namespace
        self._proc    : Optional[subprocess.Popen] = None
        self._thread  : Optional[threading.Thread] = None
        self._log     = f"/tmp/iperf3_{src_ip}_{dst_ip}_{dst_port}.log"

    def start(self) -> bool:
        cmd = [
            IPERF3_BIN,
            "-c", self.dst_ip,
            "-p", str(self.dst_port),
            "-t", str(self.duration),
            "-b", self.bandwidth,
            "-P", str(self.parallel),
        ]
        if self.protocol == "udp":
            cmd.append("-u")

        if self.namespace:
            cmd = ["ip", "netns", "exec", self.namespace] + cmd

        try:
            with open(self._log, "w") as lf:
                self._proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=lf,
                    preexec_fn=os.setsid
                )
            logger.debug(
                f"iperf3 flow {self.src_ip} -> "
                f"{self.dst_ip}:{self.dst_port} started"
            )
            return True
        except Exception as e:
            logger.warning(f"iperf3 flow failed: {e}")
            return False

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except Exception:
                pass

    def wait(self) -> int:
        if self._proc:
            return self._proc.wait()
        return -1

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class NormalTrafficGenerator:
    """
    Generates realistic background traffic using iperf3.
    Creates multiple concurrent flows with randomized parameters.
    """

    def __init__(
        self,
        host_pairs  : List[Tuple[str, str]] = None,
        base_port   : int                   = 5201,
        log_dir     : str                   = "/tmp",
    ):
        self.host_pairs  = host_pairs or DEFAULT_HOST_PAIRS
        self.base_port   = base_port
        self.log_dir     = log_dir
        self._servers    : List[IperfServer]  = []
        self._flows      : List[IperfFlow]    = []
        self._thread     : Optional[threading.Thread] = None
        self._running    = False
        os.makedirs(log_dir, exist_ok=True)

    def start_servers(self) -> None:
        """Start iperf3 servers on destination hosts."""
        dst_hosts = list(set(dst for _, dst in self.host_pairs))
        port = self.base_port
        for dst in dst_hosts:
            server = IperfServer(host=dst, port=port)
            ok = server.start()
            if ok:
                self._servers.append(server)
            port += 1

    def _flow_worker(
        self,
        src_ip   : str,
        dst_ip   : str,
        port     : int,
        duration : int,
        bw       : str,
        proto    : str,
    ) -> None:
        flow = IperfFlow(
            src_ip   = src_ip,
            dst_ip   = dst_ip,
            dst_port = port,
            duration = duration,
            bandwidth= bw,
            protocol = proto,
        )
        flow.start()
        flow.wait()

    def start(
        self,
        flows_per_pair  : int   = 2,
        duration        : int   = 120,
        min_bw          : str   = "1M",
        max_bw          : str   = "50M",
        mix_protocols   : bool  = True,
    ) -> None:
        """
        Start generating normal traffic.
        Creates multiple parallel flows between host pairs.
        """
        self._running = True

        def _generate():
            threads = []
            port = self.base_port
            for pair_idx, (src, dst) in enumerate(self.host_pairs):
                for flow_idx in range(flows_per_pair):
                    bw_mbps = random.randint(1, 50)
                    bw_str  = f"{bw_mbps}M"
                    proto   = "tcp"
                    if mix_protocols and random.random() < 0.3:
                        proto = "udp"
                    t = threading.Thread(
                        target=self._flow_worker,
                        args=(
                            src, dst,
                            port + pair_idx * flows_per_pair + flow_idx,
                            duration,
                            bw_str,
                            proto,
                        ),
                        daemon=True
                    )
                    t.start()
                    threads.append(t)
                    time.sleep(random.uniform(0.1, 0.5))

            for t in threads:
                t.join(timeout=duration + 10)

        self._thread = threading.Thread(
            target=_generate, daemon=True
        )
        self._thread.start()
        logger.info(
            f"Normal traffic generator started: "
            f"{len(self.host_pairs)} pairs, "
            f"{flows_per_pair} flows each, "
            f"{duration}s duration"
        )

    def generate_continuous(
        self,
        total_duration   : int   = 300,
        flow_duration    : int   = 30,
        flows_per_pair   : int   = 2,
    ) -> None:
        """
        Continuously regenerate flows for total_duration seconds.
        """
        self._running = True
        end_time = time.time() + total_duration

        def _worker():
            while self._running and time.time() < end_time:
                self.start(
                    flows_per_pair = flows_per_pair,
                    duration       = flow_duration,
                )
                time.sleep(flow_duration * 0.8)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        logger.info(
            f"Continuous normal traffic for {total_duration}s"
        )

    def stop(self) -> None:
        self._running = False
        for flow in self._flows:
            flow.stop()
        for server in self._servers:
            server.stop()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Normal traffic generator stopped")

    def wait(self) -> None:
        if self._thread:
            self._thread.join()


class PingFloodGenerator:
    """
    Simple ICMP ping traffic generator for connectivity testing.
    """

    def __init__(
        self,
        src_ip : str,
        dst_ip : str,
        count  : int = 1000,
        interval: float = 0.01,
    ):
        self.src_ip   = src_ip
        self.dst_ip   = dst_ip
        self.count    = count
        self.interval = interval
        self._proc    : Optional[subprocess.Popen] = None

    def start(self) -> None:
        cmd = [
            "ping",
            "-c", str(self.count),
            "-i", str(self.interval),
            self.dst_ip,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning(f"ping failed: {e}")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def wait(self) -> None:
        if self._proc:
            self._proc.wait()


def run_normal_traffic(
    host_pairs    : List[Tuple[str, str]],
    duration      : int  = 120,
    flows_per_pair: int  = 2,
    log_dir       : str  = "/tmp/adaptmon_traffic",
) -> NormalTrafficGenerator:
    """
    Convenience function: start and return a normal traffic generator.
    """
    gen = NormalTrafficGenerator(host_pairs=host_pairs, log_dir=log_dir)
    gen.generate_continuous(
        total_duration  = duration,
        flow_duration   = min(30, duration // 4),
        flows_per_pair  = flows_per_pair,
    )
    return gen
