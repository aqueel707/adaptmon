"""
controller/features.py
Feature extraction from raw register snapshots.
Computes per-flow statistics: pps, bps, entropy, duration, etc.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

MAX_FLOWS = 4096


class FlowRecord:
    """Stores time-series data for a single flow."""

    def __init__(self, flow_index: int):
        self.flow_index   = flow_index
        self.timestamps   : List[float]  = []
        self.pkt_counts   : List[int]    = []
        self.byte_counts  : List[int]    = []
        self.first_seen   : float        = 0.0
        self.last_seen    : float        = 0.0
        self.active       : bool         = False

    def update(
        self,
        timestamp: float,
        pkt_count: int,
        byte_count: int
    ) -> None:
        self.timestamps.append(timestamp)
        self.pkt_counts.append(pkt_count)
        self.byte_counts.append(byte_count)
        if not self.active:
            self.first_seen = timestamp
            self.active = True
        self.last_seen = timestamp
        # Keep rolling window of last 60 samples
        if len(self.timestamps) > 60:
            self.timestamps.pop(0)
            self.pkt_counts.pop(0)
            self.byte_counts.pop(0)

    @property
    def duration(self) -> float:
        if self.first_seen == 0 or self.last_seen == 0:
            return 0.0
        return self.last_seen - self.first_seen

    @staticmethod
    def _counter_delta(counts: List[int]) -> int:
        if len(counts) < 2:
            return 0
        # A drop means the controller reset the registers (flow aging)
        # since the last poll; everything counted now arrived after it.
        if counts[-1] < counts[-2]:
            return counts[-1]
        return counts[-1] - counts[-2]

    @property
    def pkt_delta(self) -> int:
        return self._counter_delta(self.pkt_counts)

    @property
    def byte_delta(self) -> int:
        return self._counter_delta(self.byte_counts)

    @property
    def time_delta(self) -> float:
        if len(self.timestamps) < 2:
            return 1.0
        delta = self.timestamps[-1] - self.timestamps[-2]
        return delta if delta > 0 else 1.0

    def packets_per_second(self) -> float:
        return self.pkt_delta / self.time_delta

    def bytes_per_second(self) -> float:
        return self.byte_delta / self.time_delta

    def mean_pkt_size(self) -> float:
        if self.pkt_counts[-1] == 0 if self.pkt_counts else True:
            return 0.0
        total_pkts = self.pkt_counts[-1]
        total_bytes = self.byte_counts[-1]
        return total_bytes / total_pkts if total_pkts > 0 else 0.0

    def pkt_rate_variance(self) -> float:
        if len(self.pkt_counts) < 3:
            return 0.0
        deltas = [
            max(0, self.pkt_counts[i] - self.pkt_counts[i-1])
            for i in range(1, len(self.pkt_counts))
        ]
        return float(np.var(deltas))


class FeatureExtractor:
    """
    Maintains flow state and extracts feature vectors
    from periodic register snapshots.
    """

    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval  = poll_interval
        self.flow_records   : Dict[int, FlowRecord] = {}
        self.prev_pkts      : Dict[int, int]  = {}
        self.prev_bytes     : Dict[int, int]  = {}
        self.prev_timestamp : float            = 0.0
        self._global_entropy_history : List[float] = []

    def _get_or_create(self, idx: int) -> FlowRecord:
        if idx not in self.flow_records:
            self.flow_records[idx] = FlowRecord(idx)
        return self.flow_records[idx]

    def update_snapshot(
        self,
        pkt_counters  : Dict[int, int],
        byte_counters : Dict[int, int],
        timestamp     : float
    ) -> None:
        """Ingest a new register snapshot."""
        for idx in range(MAX_FLOWS):
            pkt_val  = pkt_counters.get(idx, 0)
            byte_val = byte_counters.get(idx, 0)
            if pkt_val > 0:
                rec = self._get_or_create(idx)
                rec.update(timestamp, pkt_val, byte_val)
            else:
                # Empty slot (aged out): a later flow hashing here is new
                self.flow_records.pop(idx, None)
        self.prev_timestamp = timestamp

    def compute_entropy(
        self,
        pkt_counters: Dict[int, int]
    ) -> float:
        """
        Shannon entropy of flow packet distribution.
        H = -sum(p_i * log2(p_i))
        """
        counts = np.array(
            [v for v in pkt_counters.values() if v > 0],
            dtype=np.float64
        )
        if len(counts) == 0:
            return 0.0
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy)

    def compute_flow_distribution(
        self,
        pkt_counters: Dict[int, int]
    ) -> Dict[str, float]:
        """Compute distribution statistics across flows."""
        vals = [v for v in pkt_counters.values() if v > 0]
        if not vals:
            return {
                "n_active_flows": 0,
                "mean_pkts"     : 0.0,
                "std_pkts"      : 0.0,
                "max_pkts"      : 0.0,
                "gini"          : 0.0,
            }
        arr = np.array(vals, dtype=np.float64)
        return {
            "n_active_flows": len(vals),
            "mean_pkts"     : float(arr.mean()),
            "std_pkts"      : float(arr.std()),
            "max_pkts"      : float(arr.max()),
            "gini"          : self._gini(arr),
        }

    @staticmethod
    def _gini(arr: np.ndarray) -> float:
        """Gini coefficient of array."""
        if arr.sum() == 0:
            return 0.0
        sorted_arr = np.sort(arr)
        n = len(sorted_arr)
        index = np.arange(1, n + 1)
        return float(
            (2 * np.sum(index * sorted_arr)) / (n * sorted_arr.sum()) - (n + 1) / n
        )

    def get_active_flows(self) -> List[int]:
        """Return indices of currently active flows."""
        return [
            idx for idx, rec in self.flow_records.items()
            if rec.active and len(rec.pkt_counts) > 0
            and rec.pkt_counts[-1] > 0
        ]

    def extract_feature_vector(
        self,
        pkt_counters  : Dict[int, int],
        byte_counters : Dict[int, int],
        timestamp     : float
    ) -> np.ndarray:
        """
        Build the global feature vector used by anomaly detection.
        Features:
          [0] entropy
          [1] n_active_flows
          [2] mean_pps (avg packets/sec across active flows)
          [3] mean_bps (avg bytes/sec across active flows)
          [4] max_pps
          [5] max_bps
          [6] std_pps
          [7] std_bps
          [8] gini_pkts
          [9] top_flow_fraction
        """
        entropy = self.compute_entropy(pkt_counters)
        dist    = self.compute_flow_distribution(pkt_counters)

        active_flows = self.get_active_flows()
        n_active = len(active_flows)

        pps_list = []
        bps_list = []
        for idx in active_flows:
            rec = self.flow_records[idx]
            if len(rec.timestamps) >= 2:
                pps_list.append(rec.packets_per_second())
                bps_list.append(rec.bytes_per_second())

        pps_arr = np.array(pps_list, dtype=np.float64) if pps_list else np.array([0.0])
        bps_arr = np.array(bps_list, dtype=np.float64) if bps_list else np.array([0.0])

        total_pkts = sum(pkt_counters.values())
        top_count  = max(pkt_counters.values(), default=0)
        top_fraction = top_count / total_pkts if total_pkts > 0 else 0.0

        features = np.array([
            entropy,                          # [0]
            float(n_active),                  # [1]
            float(pps_arr.mean()),            # [2]
            float(bps_arr.mean()),            # [3]
            float(pps_arr.max()),             # [4]
            float(bps_arr.max()),             # [5]
            float(pps_arr.std()),             # [6]
            float(bps_arr.std()),             # [7]
            dist["gini"],                     # [8]
            top_fraction,                     # [9]
        ], dtype=np.float64)

        return features

    def extract_per_flow_features(
        self,
        indices: List[int]
    ) -> Dict[int, np.ndarray]:
        """
        Extract per-flow feature vectors.
        Returns dict: flow_index -> feature_vector
        """
        result = {}
        for idx in indices:
            rec = self.flow_records.get(idx)
            if rec is None or len(rec.timestamps) < 2:
                continue
            fv = np.array([
                rec.packets_per_second(),
                rec.bytes_per_second(),
                rec.mean_pkt_size(),
                rec.pkt_rate_variance(),
                float(rec.duration),
                float(rec.pkt_counts[-1]),
                float(rec.byte_counts[-1]),
            ], dtype=np.float64)
            result[idx] = fv
        return result

    def get_entropy_history(self) -> List[float]:
        return self._global_entropy_history.copy()

    def clear_inactive_flows(self, idle_threshold: float = 30.0) -> int:
        """Remove flows idle for more than idle_threshold seconds."""
        now = time.time()
        to_remove = [
            idx for idx, rec in self.flow_records.items()
            if rec.active and (now - rec.last_seen) > idle_threshold
        ]
        for idx in to_remove:
            self.flow_records[idx].active = False
        return len(to_remove)

    def get_top_talkers(
        self,
        pkt_counters: Dict[int, int],
        n: int = 10
    ) -> List[Tuple[int, int]]:
        """Return top-n flows by packet count."""
        active = {
            idx: cnt for idx, cnt in pkt_counters.items()
            if cnt > 0
        }
        sorted_flows = sorted(
            active.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_flows[:n]

    def summary(
        self,
        pkt_counters  : Dict[int, int],
        byte_counters : Dict[int, int],
        timestamp     : float
    ) -> Dict:
        """Return a summary dict for logging."""
        entropy = self.compute_entropy(pkt_counters)
        dist    = self.compute_flow_distribution(pkt_counters)
        top     = self.get_top_talkers(pkt_counters, 5)
        return {
            "timestamp"     : timestamp,
            "entropy"       : entropy,
            "n_active_flows": dist["n_active_flows"],
            "mean_pkts"     : dist["mean_pkts"],
            "std_pkts"      : dist["std_pkts"],
            "max_pkts"      : dist["max_pkts"],
            "gini"          : dist["gini"],
            "top_talkers"   : top,
        }
