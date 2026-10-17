"""
traffic/ground_truth.py
Per-run ground truth: every flow 5-tuple a traffic generator actually
created, labelled benign/attack with its attack class, generator and
active time window.

Every generator in traffic/paper_traffic.py records into the SAME
GroundTruthRecorder the moment it creates a flow (fixing the source port
first, so the record IS the 5-tuple that goes on the wire, not a guess
made after the fact). This is the source of truth analysis/evaluate.py
scores detections against in Stage 6 -- never inferred from what the
detector itself flagged.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class FlowRecord:
    src         : str
    dst         : str
    proto       : int             # 6 = TCP, 17 = UDP
    sport       : int
    dport       : int
    label       : str             # "benign" | "attack"
    attack_class: Optional[str]   # None for benign; else e.g. "syn_scan"
    generator   : str             # "iperf3_tcp" | "http" | "sqli_post" | ...
    start_ts    : float
    end_ts      : Optional[float] = None


class GroundTruthRecorder:

    def __init__(self):
        self._lock = threading.Lock()
        self.flows: List[FlowRecord] = []

    def add(self, src: str, dst: str, proto: int, sport: int, dport: int,
            label: str, generator: str, attack_class: Optional[str] = None,
            start_ts: Optional[float] = None) -> FlowRecord:
        assert label in ("benign", "attack")
        rec = FlowRecord(src, dst, proto, sport, dport, label, attack_class,
                         generator, start_ts if start_ts is not None else time.time())
        with self._lock:
            self.flows.append(rec)
        return rec

    @staticmethod
    def end(rec: FlowRecord, end_ts: Optional[float] = None) -> None:
        rec.end_ts = end_ts if end_ts is not None else time.time()

    def write(self, path: str) -> None:
        with self._lock:
            with open(path, "w") as f:
                for r in self.flows:
                    f.write(json.dumps(asdict(r)) + "\n")

    def summary(self) -> dict:
        with self._lock:
            by_class = {}
            for r in self.flows:
                key = r.attack_class or "benign"
                by_class[key] = by_class.get(key, 0) + 1
            return {"n_flows": len(self.flows), "by_class": by_class}
