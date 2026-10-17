"""
controller/features.py
Feature extraction from raw register snapshots: per-flow vectors for the
flow-level Isolation Forest, and per-(src IP, dst IP) aggregate vectors for
the host-pair Isolation Forest (Decision B).

Per occupied register slot (= flow, keyed by the 5-tuple hash) per poll
window, the flow-level feature vector is:

  [0] pkt_count      packets in the slot since the flow started (pkt_counter)
  [1] byte_count     bytes since the flow started (byte_counter)
  [2] duration_s     flow_last_ts - flow_first_ts (data-plane timestamps)
  [3] pkt_rate       packets in this window / window length (pkt/s)
  [4] mean_pkt_size  byte_count / pkt_count
  [5] entropy        payload byte entropy from this window's sampled clones
                      (controller/dpi.py's live snapshot) when at least one
                      sampled byte of this flow has been seen; otherwise the
                      flow-distribution entropy below. controller/monitor.py
                      does the substitution (it owns the DPI snapshot) and
                      logs which source each row used (flow_windows.csv:
                      entropy_source column).

ENTROPY_FALLBACK_SOURCE = "flow_distribution": the Shannon entropy (bits) of
this window's per-flow packet deltas across all active slots of the switch,
H = -sum p_i log2 p_i with p_i = dpkts_i / sum(dpkts). It is a window-level
value, identical for every flow of that switch in that window, computed here
because it needs no payload visibility. It is used only until a flow's first
sampled clone arrives (adaptive sampling means every flow is eventually
sampled, including at NORMAL: see p4/adaptmon.p4), and during any window
where controller/dpi.py is not running.

Per (src IP, dst IP) pair per window, the host-pair feature vector is:

  [0] new_flows          slots for this pair newly occupied this window
                          (previous pkt_counter was 0, or the slot's 5-tuple
                          changed under a running counter: a collision)
  [1] distinct_dst_ports  distinct dport values among this pair's slots
  [2] syn_only_ratio      sum(syn_delta) / sum(pkt_delta) over the pair's
                          flows this window (0 if the pair has no TCP)
  [3] pkt_count           sum(pkt_delta) over the pair's flows this window
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import ip_int_to_str

logger = logging.getLogger(__name__)

MAX_FLOWS = 4096

FEATURE_NAMES = [
    "pkt_count", "byte_count", "duration_s",
    "pkt_rate", "mean_pkt_size", "entropy",
]
N_FEATURES = len(FEATURE_NAMES)
ENTROPY_IDX = FEATURE_NAMES.index("entropy")
ENTROPY_FALLBACK_SOURCE = "flow_distribution"
ENTROPY_PRIMARY_SOURCE = "payload_sampled"

PAIR_FEATURE_NAMES = ["new_flows", "distinct_dst_ports", "syn_only_ratio", "pkt_count"]
N_PAIR_FEATURES = len(PAIR_FEATURE_NAMES)

FiveTuple = Tuple[str, str, int, int, int]
PairKey = Tuple[int, str, str]      # (switch_id, src_ip, dst_ip)


@dataclass
class SwitchWindow:
    """Per-flow features of one switch for one poll window."""
    switch_id : int
    timestamp : float
    dt        : float               # seconds since the previous poll
    slots     : np.ndarray          # occupied slot indices
    tuples    : List[FiveTuple]     # (src, dst, proto, sport, dport) per slot
    X         : np.ndarray          # (n, N_FEATURES)
    pkt_delta : np.ndarray          # packets in this window per slot
    syn_delta : np.ndarray          # SYN-only packets in this window per slot
    is_new    : np.ndarray          # bool: slot newly occupied (or collided) this window
    flow_level: np.ndarray          # flow_level register as read
    entropy   : float               # flow-distribution entropy (fallback source)
    n_tuple_changes: int            # slots whose 5-tuple changed without reset


def shannon_entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0].astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(-(p * np.log2(p)).sum())


class FlowFeatureExtractor:
    """
    Keeps the previous snapshot of one switch and turns each new register
    snapshot into per-flow feature vectors.
    """

    def __init__(self, switch_id: int):
        self.switch_id = switch_id
        self._prev_pkts: Optional[np.ndarray] = None
        self._prev_syn : Optional[np.ndarray] = None
        self._prev_ts  : Optional[float] = None
        self._prev_tuples: Optional[np.ndarray] = None

    def forget(self, slots: List[int]) -> None:
        """Slots cleared by the controller (aging): next delta starts at 0."""
        if self._prev_pkts is not None and slots:
            self._prev_pkts[slots] = 0
            self._prev_syn[slots] = 0

    def extract(self, regs: Dict[str, List[int]], timestamp: float) -> Optional[SwitchWindow]:
        """
        regs: whole register arrays from SwitchInterface.read_registers().
        Returns None for the first snapshot (no previous one to diff
        against) or if the read was incomplete.
        """
        try:
            pkts  = np.asarray(regs["pkt_counter"],  dtype=np.int64)
            bytes_= np.asarray(regs["byte_counter"], dtype=np.int64)
            first = np.asarray(regs["flow_first_ts"], dtype=np.int64)
            last  = np.asarray(regs["flow_last_ts"],  dtype=np.int64)
            src   = np.asarray(regs["flow_src_ip"],   dtype=np.int64)
            dst   = np.asarray(regs["flow_dst_ip"],   dtype=np.int64)
            proto = np.asarray(regs["flow_protocol"], dtype=np.int64)
            sport = np.asarray(regs["flow_src_port"], dtype=np.int64)
            dport = np.asarray(regs["flow_dst_port"], dtype=np.int64)
            level = np.asarray(regs["flow_level"],    dtype=np.int64)
            syn   = np.asarray(regs["flow_syn_pkts"], dtype=np.int64)
        except KeyError as e:
            logger.warning(f"S{self.switch_id}: register missing from read: {e}")
            return None
        if not all(len(a) == MAX_FLOWS for a in (pkts, bytes_, first, last, src, dst,
                                                 proto, sport, dport, level, syn)):
            logger.warning(f"S{self.switch_id}: incomplete register read")
            return None

        prev, prev_ts, prev_syn = self._prev_pkts, self._prev_ts, self._prev_syn
        tuples_now = np.vstack([src, dst, proto, sport, dport])
        prev_tuples = self._prev_tuples
        self._prev_pkts, self._prev_ts = pkts.copy(), timestamp
        self._prev_syn = syn.copy()
        self._prev_tuples = tuples_now
        if prev is None:
            return None

        dt = max(timestamp - prev_ts, 1e-3)
        occupied = np.nonzero(pkts > 0)[0]
        # A counter below its previous value means the slot was cleared
        # (aging) since the last poll: everything counted arrived after it
        reset = pkts[occupied] < prev[occupied]
        delta = np.where(~reset, pkts[occupied] - prev[occupied], pkts[occupied])
        syn_delta = np.where(~(syn[occupied] < prev_syn[occupied]),
                             syn[occupied] - prev_syn[occupied], syn[occupied])
        # 5-tuple changed while the counter kept growing: another flow hashed
        # into an occupied slot (collision); the slot now mixes two flows
        tuple_changed = (prev[occupied] > 0) & ~reset & \
            (tuples_now[:, occupied] != prev_tuples[:, occupied]).any(axis=0)
        changed = int(np.count_nonzero(tuple_changed))
        # "New" = this slot was empty last poll, or the above collision case
        is_new = (prev[occupied] == 0) | tuple_changed

        entropy = shannon_entropy(delta)
        p = pkts[occupied].astype(np.float64)
        b = bytes_[occupied].astype(np.float64)
        X = np.column_stack([
            p,
            b,
            np.maximum(last[occupied] - first[occupied], 0) / 1e6,
            delta / dt,
            np.divide(b, p, out=np.zeros_like(b), where=p > 0),
            np.full(len(occupied), entropy),
        ]) if len(occupied) else np.zeros((0, N_FEATURES))

        tuples = [
            (ip_int_to_str(int(src[i])), ip_int_to_str(int(dst[i])),
             int(proto[i]), int(sport[i]), int(dport[i]))
            for i in occupied
        ]
        return SwitchWindow(
            switch_id=self.switch_id, timestamp=timestamp, dt=dt,
            slots=occupied, tuples=tuples, X=X, pkt_delta=delta,
            syn_delta=syn_delta, is_new=is_new,
            flow_level=level[occupied], entropy=entropy,
            n_tuple_changes=changed,
        )


def aggregate_host_pairs(w: SwitchWindow) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Group a SwitchWindow's occupied flows by (src IP, dst IP) and build one
    PAIR_FEATURE_NAMES vector per pair for this window (Decision B).
    """
    pairs: Dict[Tuple[str, str], Dict[str, object]] = {}
    for j, tup in enumerate(w.tuples):
        src, dst, proto, _sport, dport = tup
        key = (src, dst)
        agg = pairs.setdefault(key, {
            "new_flows": 0, "ports": set(), "syn": 0, "pkts": 0, "tcp_pkts": 0,
        })
        agg["new_flows"] += int(w.is_new[j])
        agg["ports"].add(dport)
        agg["pkts"] += int(w.pkt_delta[j])
        if proto == 6:   # TCP: syn_delta only meaningful for TCP flows
            agg["tcp_pkts"] += int(w.pkt_delta[j])
            agg["syn"] += int(w.syn_delta[j])
    out = {}
    for key, agg in pairs.items():
        ratio = agg["syn"] / agg["tcp_pkts"] if agg["tcp_pkts"] > 0 else 0.0
        out[key] = np.array([
            agg["new_flows"], len(agg["ports"]), ratio, agg["pkts"],
        ], dtype=np.float64)
    return out


# --------------------------------------------------------------------------
# DPI live entropy snapshot (Decision C): controller/dpi.py periodically
# dumps per-flow payload-byte entropy from the clones it has sniffed so far;
# the monitor loop reads it each poll and prefers it over the flow-
# distribution entropy above, for every flow it has data for.
# --------------------------------------------------------------------------

def entropy_snapshot_key(switch_id: int, tup: FiveTuple) -> str:
    return f"{switch_id}|{tup[0]}|{tup[1]}|{tup[2]}|{tup[3]}|{tup[4]}"


class EntropySnapshotReader:
    """Re-reads out_dir/dpi_entropy_live.json only when its mtime changes."""

    def __init__(self, path: str):
        self.path = path
        self._mtime: Optional[float] = None
        self._data: Dict[str, dict] = {}

    def refresh(self) -> Dict[str, dict]:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self._data = {}
            return self._data
        if mtime == self._mtime:
            return self._data
        try:
            with open(self.path) as f:
                self._data = json.load(f)
            self._mtime = mtime
        except (OSError, json.JSONDecodeError):
            pass   # keep the previous snapshot; writer uses atomic replace
        return self._data

    def lookup(self, switch_id: int, tup: FiveTuple) -> Optional[dict]:
        return self._data.get(entropy_snapshot_key(switch_id, tup))
