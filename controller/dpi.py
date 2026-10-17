"""
controller/dpi.py
Deep packet inspection of the clones the switches send to their CPU port.

    python3 -m controller.dpi --ifaces 1:am_s1dpi,2:am_s2dpi \
        --alerts results/<run>/alerts.jsonl --out-dir results/<run>

A Scapy sniffer on each CPU-port interface receives clones, i.e.
cpu_hdr (14 B: flow_index u32, flow_level u8, pkt_count u32, byte_count u32,
reason u8) followed by the original Ethernet frame. The cpu_hdr is stripped,
the frame is parsed with Scapy, and these rules run:

  syn_scan  per (src IP, dst IP) aggregate of cloned TCP packets: share of
            SYN-only packets (flags == SYN) > 0.9 AND >= 5 distinct dst
            ports. A single 5-tuple has exactly one dst port, so the
            ">= 5 distinct dst ports within the flow" condition is evaluated
            on the host-pair aggregate; each SYN-only 5-tuple flow of a
            matching aggregate gets the confirmed alert.
  null_scan TCP packet with no flags set.
  sqli      in HTTP payloads only (TCP payload of a flow from its first
            cloned HTTP request line on), URL-decoded, compiled
            case-insensitive regexes: OR 1=1, UNION SELECT, "--", "/*".

Each (switch, 5-tuple, class) is confirmed at most once; the confirmed alert
is a JSON line {"type": "confirmed", ...} appended to alerts.jsonl.
Payload byte entropy (bits/byte, over all cloned payload bytes of the flow)
is reported per flow in dpi_flows.csv and in the alerts; it is not used by
the Isolation Forest (see controller/features.py).
"""

import argparse
import csv
import json
import math
import os
import re
import signal
import socket
import struct
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote_plus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from controller.alerts import AlertLog   # noqa: E402

CPU_HDR = struct.Struct("!IBIIB")
CPU_HDR_LEN = CPU_HDR.size     # 14

SYN, RST, ACK, FIN = 0x02, 0x04, 0x10, 0x01
SYN_ONLY_RATIO = 0.9
SYN_MIN_PORTS = 5

SQLI_PATTERNS = {
    "or_1_eq_1"   : re.compile(r"\bor\s+'?1'?\s*=\s*'?1\b", re.IGNORECASE),
    "union_select": re.compile(r"\bunion\s+(?:all\s+)?select\b", re.IGNORECASE),
    "sql_comment_dashes": re.compile(r"--"),
    "sql_comment_block" : re.compile(r"/\*"),
}
HTTP_REQUEST_LINE = re.compile(
    rb"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) \S+ HTTP/1\.[01]\r?\n")

FiveTuple = Tuple[str, str, int, int, int]


def byte_entropy(hist: List[int]) -> float:
    total = sum(hist)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in hist if c)


class FlowRecord:
    __slots__ = ("switch", "tuple", "flow_idx", "n", "payload_bytes", "hist",
                 "is_http", "classes", "first_ts", "last_ts", "max_level")

    def __init__(self, switch, tup, flow_idx, ts):
        self.switch, self.tuple, self.flow_idx = switch, tup, flow_idx
        self.n = 0
        self.payload_bytes = 0
        self.hist = [0] * 256
        self.is_http: Optional[bool] = None
        self.classes = set()
        self.first_ts = self.last_ts = ts
        self.max_level = 0


class DPIEngine:
    """Rule engine; process() takes one raw clone (cpu_hdr + frame)."""

    def __init__(self):
        from scapy.all import Ether, IP, TCP, UDP   # lazy: heavy import
        self._Ether, self._IP, self._TCP, self._UDP = Ether, IP, TCP, UDP
        self.flows: Dict[Tuple[int, FiveTuple], FlowRecord] = {}
        # (switch, src, dst) -> {"syn_only": n, "tcp": n, "ports": set, "flows": set}
        self.pairs: Dict[tuple, dict] = defaultdict(
            lambda: {"syn_only": 0, "tcp": 0, "ports": set(), "syn_flows": set()})
        self.stats = {"clones": 0, "short": 0, "non_ip": 0, "tcp": 0, "udp": 0,
                      "other_l4": 0, "http_payload_pkts": 0, "confirmed": 0}

    def _confirm(self, rec: FlowRecord, cls: str, ts: float, cpu: dict,
                 evidence: dict) -> Optional[dict]:
        if cls in rec.classes:
            return None
        rec.classes.add(cls)
        self.stats["confirmed"] += 1
        return {
            "type": "confirmed", "ts": ts, "switch": rec.switch,
            "flow_idx": cpu["flow_index"],
            "five_tuple": dict(zip(("src", "dst", "proto", "sport", "dport"), rec.tuple)),
            "attack_class": cls, "level": cpu["level"],
            "evidence": evidence,
            "payload_entropy": round(byte_entropy(rec.hist), 4),
            "clones_seen": rec.n,
        }

    def process(self, raw: bytes, switch: int, ts: float) -> List[dict]:
        self.stats["clones"] += 1
        if len(raw) < CPU_HDR_LEN + 14:
            self.stats["short"] += 1
            return []
        idx, level, pkt_count, byte_count, reason = CPU_HDR.unpack_from(raw)
        cpu = {"flow_index": idx, "level": level, "pkt_count": pkt_count,
               "byte_count": byte_count, "reason": reason}
        eth = self._Ether(raw[CPU_HDR_LEN:])
        if not eth.haslayer(self._IP):
            self.stats["non_ip"] += 1
            return []
        ip = eth[self._IP]
        alerts = []
        if ip.haslayer(self._TCP):
            l4 = ip[self._TCP]
            self.stats["tcp"] += 1
        elif ip.haslayer(self._UDP):
            l4 = ip[self._UDP]
            self.stats["udp"] += 1
        else:
            self.stats["other_l4"] += 1
            l4 = None
        tup = (ip.src, ip.dst, int(ip.proto),
               int(l4.sport) if l4 is not None else 0,
               int(l4.dport) if l4 is not None else 0)
        key = (switch, tup)
        rec = self.flows.get(key)
        if rec is None:
            rec = self.flows[key] = FlowRecord(switch, tup, idx, ts)
        rec.n += 1
        rec.last_ts = ts
        rec.max_level = max(rec.max_level, level)
        payload = bytes(l4.payload) if l4 is not None else bytes(ip.payload)
        rec.payload_bytes += len(payload)
        for b in payload:
            rec.hist[b] += 1

        if l4 is not None and ip.haslayer(self._TCP):
            flags = int(l4.flags)
            # NULL scan
            if flags == 0:
                a = self._confirm(rec, "null_scan", ts, cpu, {"tcp_flags": 0})
                if a:
                    alerts.append(a)
            # SYN scan (host-pair aggregate)
            pair = self.pairs[(switch, ip.src, ip.dst)]
            pair["tcp"] += 1
            pair["ports"].add(int(l4.dport))
            if flags == SYN:
                pair["syn_only"] += 1
                pair["syn_flows"].add(tup)
            ratio = pair["syn_only"] / pair["tcp"]
            if ratio > SYN_ONLY_RATIO and len(pair["ports"]) >= SYN_MIN_PORTS:
                ev = {"syn_only_ratio": round(ratio, 4),
                      "distinct_dst_ports": len(pair["ports"]),
                      "pair_tcp_pkts": pair["tcp"]}
                for t in pair["syn_flows"]:
                    r = self.flows.get((switch, t))
                    if r is not None:
                        a = self._confirm(r, "syn_scan", ts, cpu if r is rec else
                                          {"flow_index": r.flow_idx, "level": r.max_level}, ev)
                        if a:
                            alerts.append(a)
            # SQLi in HTTP payloads
            if payload:
                # A flow escalated mid-connection may first be cloned mid-body:
                # it counts as HTTP from its first cloned request line on
                if not rec.is_http:
                    rec.is_http = bool(HTTP_REQUEST_LINE.match(payload))
                if rec.is_http:
                    self.stats["http_payload_pkts"] += 1
                    text = unquote_plus(payload.decode("latin-1"))
                    hits = [n for n, rx in SQLI_PATTERNS.items() if rx.search(text)]
                    if hits:
                        a = self._confirm(rec, "sqli", ts, cpu, {"patterns": hits})
                        if a:
                            alerts.append(a)
        return alerts

    def write_flows_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["switch", "src", "dst", "proto", "sport", "dport", "flow_idx",
                        "clones", "payload_bytes", "payload_entropy", "max_level",
                        "first_ts", "last_ts", "classes"])
            for rec in self.flows.values():
                w.writerow([rec.switch, *rec.tuple, rec.flow_idx, rec.n,
                            rec.payload_bytes, f"{byte_entropy(rec.hist):.4f}",
                            rec.max_level, f"{rec.first_ts:.6f}", f"{rec.last_ts:.6f}",
                            "|".join(sorted(rec.classes))])


def sniff_loop(ifaces: Dict[str, int], engine: DPIEngine, alerts: AlertLog,
               stop: threading.Event) -> Dict[str, dict]:
    """Scapy sniffer on the CPU-port interfaces until stop is set."""
    from scapy.all import AsyncSniffer, conf

    socks = {}
    for name in ifaces:
        s = conf.L2listen(iface=name)
        try:
            s.ins.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        except OSError:
            pass
        socks[s] = name
    lock = threading.Lock()

    def on_pkt(pkt):
        raw = getattr(pkt, "original", None) or bytes(pkt)
        with lock:
            for a in engine.process(raw, ifaces[pkt.sniffed_on], float(pkt.time)):
                alerts.write(a)

    sniffer = AsyncSniffer(opened_socket=socks, prn=on_pkt, store=False)
    sniffer.start()
    stop.wait()
    time.sleep(0.5)
    sniffer.stop()
    kstats = {}
    for s, name in socks.items():
        try:
            tp_packets, tp_drops = struct.unpack("II", s.ins.getsockopt(263, 6, 8))
            kstats[name] = {"kernel_received": tp_packets, "kernel_drops": tp_drops}
        except OSError:
            kstats[name] = {}
        s.close()
    return kstats


def main():
    ap = argparse.ArgumentParser(description="AdaptMon DPI on CPU-port clones")
    ap.add_argument("--ifaces", required=True, help="1:am_s1dpi,2:am_s2dpi")
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    ifaces = {}
    for item in args.ifaces.split(","):
        sid, name = item.split(":")
        ifaces[name] = int(sid)

    engine = DPIEngine()
    alerts = AlertLog(args.alerts)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    print(f"DPI sniffing on {ifaces}", flush=True)
    t0 = time.time()
    kstats = sniff_loop(ifaces, engine, alerts, stop)
    engine.write_flows_csv(os.path.join(args.out_dir, "dpi_flows.csv"))
    with open(os.path.join(args.out_dir, "dpi_stats.json"), "w") as f:
        json.dump({"engine": engine.stats, "sockets": kstats,
                   "runtime_s": time.time() - t0}, f, indent=2)
    print(f"DPI stopped: {engine.stats}", flush=True)


if __name__ == "__main__":
    main()
