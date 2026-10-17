"""
traffic/paper_traffic.py
Stage-4 traffic generators for the paper evaluation runs, all run inside
Linux network namespaces (scripts/sandbox.sh). Every generator fixes its
flows' source port before sending (iperf3 --cport, nmap -g/--source-port,
or an explicit socket bind) and records each one into a shared
traffic.ground_truth.GroundTruthRecorder the moment it starts -- ground
truth is what was actually sent, never inferred from captures afterwards.

Generators:
  Iperf3Benign    real iperf3 TCP (+ optional UDP) flow between two hosts
  HttpBenign      real HTTP GET/POST against a plain http.server target,
                  well-formed bodies, no SQL-like content
  HttpSqliAttack  real HTTP POST, same server, bodies that match
                  controller/dpi.py's SQLi regexes (identical strings to
                  tests/test_dpi.py's own validated cases)
  NmapSynScan     real `nmap -sS` (SYN scan) at Decision D's capped rate
  NmapNullScan    real `nmap -sN` (NULL scan), same rate cap
  RateSpikeAttack paced UDP flood (a volumetric/DoS-shaped anomaly that is
                  not any DPI rule's target -- Decision A's limit case)

Rates (Decision D): every generator here defaults to a rate that keeps the
*offered* load of a single instance under a few hundred pkt/s; the caller
(experiments/paper_run.py) is responsible for not running so many at once
that the combined originals+clones total crosses bmv2's measured capacity
(tools/bmv2_capacity.py: reliable to ~4000 pkt/s) or the ~2500 pkt/s budget
from Decision D -- in practice that means running the attack generators one
at a time, not simultaneously (see experiments/paper_run.py's schedule).

NmapScan passes --send-ip for cross-switch targets; see tests/smoke_nmap.py
for why: nmap's own ARP probing (not the kernel's, which this topology's
static neighbor entries already satisfy) never gets a reply across the
inter-switch link, because the ARP-flood logic in p4/adaptmon.p4's ingress
only floods within one switch.
"""

import logging
import os
import random
import signal
import socket
import subprocess
import threading
import time
from typing import Dict, List, Optional
from urllib.parse import unquote_plus

from .ground_truth import GroundTruthRecorder

logger = logging.getLogger(__name__)

HTTP_PORT = 8000
IPERF_PORT = 5201

# Exactly the bodies tests/test_dpi.py validates against controller/dpi.py's
# SQLI_PATTERNS, one per pattern, so a Stage-4 run exercises every pattern
# controller/dpi.py's SQLi rule actually checks for
SQLI_BODIES = [
    ("user=admin%27+OR+1%3D1&pw=x", "or_1_eq_1"),
    ("id=1+UNION+SELECT+username%2Cpassword+FROM+users", "union_select"),
    ("user=admin%27--&pw=x", "sql_comment_dashes"),
    ("user=admin%27%2F%2A&pw=x", "sql_comment_block"),
]
BENIGN_HTTP_BODIES = [
    "name=Jane+Doe&comment=great+product%2C+fast+shipping&rating=5",
    "query=wireless+mouse&category=electronics",
    "username=jdoe&remember_me=1",
]


def _run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def _ns_popen(ns: str, argv: List[str], **kw) -> subprocess.Popen:
    return subprocess.Popen(["ip", "netns", "exec", ns] + argv,
                            preexec_fn=os.setsid, **kw)


def _kill(proc: Optional[subprocess.Popen]) -> None:
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Minimal real HTTP target: accepts GET/POST, drains the body, replies 200.
# Not an application under test -- a wire-level target so DPI inspects real
# HTTP framing, not a hand-rolled approximation of it.
# --------------------------------------------------------------------------

_HTTPD_SCRIPT = """
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class H(BaseHTTPRequestHandler):
    def _ok(self):
        body = b"ok"
        if "Content-Length" in self.headers:
            self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self): self._ok()
    def do_POST(self): self._ok()
    def log_message(self, *a): pass

ThreadingHTTPServer.allow_reuse_address = True
s = ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), H)
s.serve_forever()
"""


class HttpServer:
    """A real http.server instance inside one namespace."""

    def __init__(self, ns: str, port: int = HTTP_PORT):
        self.ns, self.port = ns, port
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self._proc = _ns_popen(self.ns, ["python3", "-c", _HTTPD_SCRIPT, str(self.port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.3)

    def stop(self) -> None:
        _kill(self._proc)


def _http_request(ns: str, src_ip: str, sport: int, dst_ip: str, dport: int,
                  method: str, path: str, body: str) -> None:
    """One HTTP request over a socket bound to `sport`, run inside `ns`."""
    script = f"""
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(({src_ip!r}, {sport}))
s.settimeout(3)
s.connect(({dst_ip!r}, {dport}))
body = {body!r}.encode()
req = ({method!r} + " " + {path!r} + " HTTP/1.1\\r\\n"
       "Host: " + {dst_ip!r} + "\\r\\n"
       "Content-Type: application/x-www-form-urlencoded\\r\\n"
       "Content-Length: " + str(len(body)) + "\\r\\n\\r\\n")
s.sendall(req.encode() + body)
try:
    s.recv(4096)
except Exception:
    pass
s.close()
"""
    r = subprocess.run(["ip", "netns", "exec", ns, "python3", "-c", script],
                       capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        logger.warning(f"HTTP request {src_ip}:{sport} -> {dst_ip}:{dport} failed: {r.stderr}")


class HttpBenign:
    """Real HTTP GET/POST requests, well-formed bodies, at a modest rate."""

    def __init__(self, gt: GroundTruthRecorder, src_ns: str, src_ip: str,
                dst_ip: str, port: int = HTTP_PORT, duration: int = 60,
                interval: float = 2.0, base_sport: int = 51000,
                rng: Optional[random.Random] = None, interval_jitter: float = 0.0):
        """rng (seeded by the caller): pick each body at random and jitter
        each gap by +/- interval_jitter (a fraction); without it, bodies
        rotate and the gap is exactly `interval`."""
        self.gt, self.src_ns, self.src_ip = gt, src_ns, src_ip
        self.dst_ip, self.port, self.duration = dst_ip, port, duration
        self.interval, self.base_sport = interval, base_sport
        self.rng, self.interval_jitter = rng, interval_jitter
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        def _loop():
            end = time.time() + self.duration
            i = 0
            while time.time() < end and not self._stop.is_set():
                sport = self.base_sport + i
                i += 1
                body = (self.rng.choice(BENIGN_HTTP_BODIES) if self.rng
                        else BENIGN_HTTP_BODIES[i % len(BENIGN_HTTP_BODIES)])
                rec = self.gt.add(self.src_ip, self.dst_ip, 6, sport, self.port,
                                  "benign", "http")
                _http_request(self.src_ns, self.src_ip, sport, self.dst_ip, self.port,
                              "POST", "/submit", body)
                self.gt.end(rec)
                j = self.interval_jitter
                self._stop.wait(self.interval * (self.rng.uniform(1 - j, 1 + j)
                                                 if self.rng and j else 1.0))
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


class HttpSqliAttack:
    """
    Real HTTP POST requests whose bodies match every pattern
    controller/dpi.py's SQLi rule checks for, one request per pattern.
    """

    def __init__(self, gt: GroundTruthRecorder, src_ns: str, src_ip: str,
                dst_ip: str, port: int = HTTP_PORT, base_sport: int = 52000,
                gap: float = 1.0, bodies: Optional[List] = None):
        """bodies: (body, pattern) pairs in send order (default SQLI_BODIES)."""
        self.gt, self.src_ns, self.src_ip = gt, src_ns, src_ip
        self.dst_ip, self.port, self.base_sport, self.gap = dst_ip, port, base_sport, gap
        self.bodies = list(bodies) if bodies is not None else SQLI_BODIES
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        def _loop():
            for i, (body, pattern) in enumerate(self.bodies):
                sport = self.base_sport + i
                rec = self.gt.add(self.src_ip, self.dst_ip, 6, sport, self.port,
                                  "attack", "sqli_post", attack_class="sqli")
                _http_request(self.src_ns, self.src_ip, sport, self.dst_ip, self.port,
                              "POST", "/login", body)
                self.gt.end(rec)
                logger.info(f"SQLi POST sent: pattern={pattern} sport={sport}")
                time.sleep(self.gap)
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)


class Iperf3Benign:
    """Real iperf3 client/server flow (TCP by default)."""

    def __init__(self, gt: GroundTruthRecorder, src_ns: str, src_ip: str,
                dst_ns: str, dst_ip: str, port: int, duration: int = 60,
                bitrate: str = "2M", udp: bool = False, cport: int = 53000):
        self.gt = gt
        self.src_ns, self.src_ip, self.dst_ns, self.dst_ip = src_ns, src_ip, dst_ns, dst_ip
        self.port, self.duration, self.bitrate, self.udp, self.cport = \
            port, duration, bitrate, udp, cport
        self._server: Optional[subprocess.Popen] = None
        self._client: Optional[subprocess.Popen] = None
        self._rec = None

    def start(self) -> None:
        self._server = _ns_popen(self.dst_ns,
                                 ["iperf3", "-s", "-p", str(self.port), "-1"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        cmd = ["iperf3", "-c", self.dst_ip, "-p", str(self.port),
              "-t", str(self.duration), "--cport", str(self.cport),
              "-B", self.src_ip]
        if self.udp:
            cmd += ["-u", "-b", self.bitrate]
        else:
            cmd += ["-b", self.bitrate]
        self._rec = self.gt.add(self.src_ip, self.dst_ip, 17 if self.udp else 6,
                                self.cport, self.port, "benign",
                                "iperf3_udp" if self.udp else "iperf3_tcp")
        self._client = _ns_popen(self.src_ns, cmd,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._client:
            try:
                self._client.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        if self._rec:
            self.gt.end(self._rec)

    def stop(self) -> None:
        _kill(self._client)
        _kill(self._server)
        if self._rec and self._rec.end_ts is None:
            self.gt.end(self._rec)


class NmapScan:
    """
    Real nmap -sS (SYN) or -sN (NULL) scan at a capped rate (Decision D).
    --send-ip: see module docstring. -g fixes the source port so ground
    truth is exact; every probed port is its own ground-truth flow record
    sharing that source port.
    """

    def __init__(self, gt: GroundTruthRecorder, mode: str, src_ns: str, src_ip: str,
                dst_ip: str, ports: List[int], max_rate: int = 500,
                base_sport: int = 54000):
        assert mode in ("syn", "null")
        self.gt, self.mode, self.src_ns, self.src_ip = gt, mode, src_ns, src_ip
        self.dst_ip, self.ports, self.max_rate, self.sport = \
            dst_ip, ports, max_rate, base_sport
        self._proc: Optional[subprocess.Popen] = None
        self._recs = []

    def run(self, timeout: float = 30) -> subprocess.CompletedProcess:
        flag = "-sS" if self.mode == "syn" else "-sN"
        cls = "syn_scan" if self.mode == "syn" else "null_scan"
        gen = "nmap_syn" if self.mode == "syn" else "nmap_null"
        for p in self.ports:
            self._recs.append(self.gt.add(self.src_ip, self.dst_ip, 6, self.sport, p,
                                          "attack", gen, attack_class=cls))
        cmd = ["ip", "netns", "exec", self.src_ns, "nmap", flag, "-Pn", "-n",
              "--send-ip", "-g", str(self.sport), "--max-rate", str(self.max_rate),
              "-p", ",".join(str(p) for p in self.ports), self.dst_ip]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        for rec in self._recs:
            self.gt.end(rec)
        if r.returncode != 0:
            logger.warning(f"nmap {self.mode} scan failed: {r.stderr}")
        return r


class RateSpikeAttack:
    """
    Paced UDP flood -- a volumetric anomaly, not matched by any DPI rule
    (it is neither a TCP scan nor HTTP), so it can only ever be caught by
    the Isolation Forest score (and, per Decision A, never escalate past
    CLONE on its own).
    """

    def __init__(self, gt: GroundTruthRecorder, src_ns: str, src_ip: str,
                dst_ip: str, dport: int, duration: int = 15, rate: int = 1500,
                sport: int = 45000, payload_size: int = 64):
        self.gt = gt
        self.src_ns, self.src_ip, self.dst_ip, self.dport = src_ns, src_ip, dst_ip, dport
        self.duration, self.rate, self.sport, self.payload_size = \
            duration, rate, sport, payload_size
        self._proc: Optional[subprocess.Popen] = None
        self._rec = None

    def start(self) -> None:
        script = f"""
import socket, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(({self.src_ip!r}, {self.sport}))
payload = b"x" * {self.payload_size}
end = time.time() + {self.duration}
n, t0 = 0, time.time()
while time.time() < end:
    s.sendto(payload, ({self.dst_ip!r}, {self.dport}))
    n += 1
    lag = t0 + n / {self.rate} - time.time()
    if lag > 0:
        time.sleep(lag)
"""
        self._rec = self.gt.add(self.src_ip, self.dst_ip, 17, self.sport, self.dport,
                                "attack", "udp_rate_spike", attack_class="rate_spike")
        self._proc = _ns_popen(self.src_ns, ["python3", "-c", script],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._proc:
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        if self._rec:
            self.gt.end(self._rec)

    def stop(self) -> None:
        _kill(self._proc)
        if self._rec and self._rec.end_ts is None:
            self.gt.end(self._rec)
