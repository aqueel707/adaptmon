"""
controller/features.py
Per-flow feature extraction from raw register snapshots.

One feature vector per occupied register slot (= flow, keyed by the 5-tuple
hash) per poll window:

  [0] pkt_count      packets in the slot since the flow started (pkt_counter)
  [1] byte_count     bytes since the flow started (byte_counter)
  [2] duration_s     flow_last_ts - flow_first_ts (data-plane timestamps)
  [3] pkt_rate       packets in this window / window length (pkt/s)
  [4] mean_pkt_size  byte_count / pkt_count
  [5] entropy        see ENTROPY_SOURCE

ENTROPY_SOURCE = "flow_distribution": the Shannon entropy (bits) of this
window's per-flow packet deltas across all active slots of the switch,
H = -sum p_i log2 p_i with p_i = dpkts_i / sum(dpkts). It is a window-level
value, identical for every flow of that switch in that window.
Payload byte entropy is NOT an Isolation Forest input: it exists only for
cloned flows (level >= CLONE), never during the benign warm-up the model is
trained on, so the model could not learn it. controller/dpi.py computes it
for cloned packets and reports it in the alerts instead.
"""

import logging
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
ENTROPY_SOURCE = "flow_distribution"

FiveTuple = Tuple[str, str, int, int, int]


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
    flow_level: np.ndarray          # flow_level register as read
    entropy   : float
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
        self._prev_ts  : Optional[float] = None
        self._prev_tuples: Optional[np.ndarray] = None

    def forget(self, slots: List[int]) -> None:
        """Slots cleared by the controller (aging): next delta starts at 0."""
        if self._prev_pkts is not None and slots:
            self._prev_pkts[slots] = 0

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
        except KeyError as e:
            logger.warning(f"S{self.switch_id}: register missing from read: {e}")
            return None
        if not all(len(a) == MAX_FLOWS for a in (pkts, bytes_, first, last, src, dst,
                                                 proto, sport, dport, level)):
            logger.warning(f"S{self.switch_id}: incomplete register read")
            return None

        prev, prev_ts = self._prev_pkts, self._prev_ts
        tuples_now = np.vstack([src, dst, proto, sport, dport])
        prev_tuples = self._prev_tuples
        self._prev_pkts, self._prev_ts = pkts.copy(), timestamp
        self._prev_tuples = tuples_now
        if prev is None:
            return None

        dt = max(timestamp - prev_ts, 1e-3)
        occupied = np.nonzero(pkts > 0)[0]
        # A counter below its previous value means the slot was cleared
        # (aging) since the last poll: everything counted arrived after it
        delta = np.where(pkts >= prev, pkts - prev, pkts)[occupied]
        # 5-tuple changed while the counter kept growing: another flow hashed
        # into an occupied slot (collision); the slot now mixes two flows
        changed = int(np.count_nonzero(
            (prev[occupied] > 0) & (pkts[occupied] >= prev[occupied])
            & (tuples_now[:, occupied] != prev_tuples[:, occupied]).any(axis=0)
        ))

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
            flow_level=level[occupied], entropy=entropy,
            n_tuple_changes=changed,
        )
