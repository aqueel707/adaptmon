"""
controller/aging.py
Flow table aging.
The P4 flow registers are hash-indexed and never cleared by the data
plane, so slots filled by short-lived flows (e.g. spoofed-source SYN
floods) stay occupied forever. FlowAger tracks, per hash slot, how many
consecutive polls its packet counter has not changed and reports slots
idle for idle_polls polls so the controller can zero them.
"""

from typing import Dict, List


class FlowAger:

    def __init__(self, idle_polls: int = 3):
        self.idle_polls  = idle_polls
        self._last_pkts  : Dict[int, int] = {}
        self._idle       : Dict[int, int] = {}

    def update(self, pkt_counters: Dict[int, int]) -> List[int]:
        """
        Ingest one register snapshot; return indices of expired slots.
        A slot reading 0 is empty (never used, or already cleared) and is
        no longer tracked. Expired slots are forgotten immediately: once
        zeroed in the data plane they read 0, and if a new flow hashes
        there it is tracked afresh.
        """
        expired = []
        for idx, val in pkt_counters.items():
            if val == 0:
                self._last_pkts.pop(idx, None)
                self._idle.pop(idx, None)
                continue
            if self._last_pkts.get(idx) != val:
                self._last_pkts[idx] = val
                self._idle[idx]      = 0
                continue
            self._idle[idx] += 1
            if self._idle[idx] >= self.idle_polls:
                expired.append(idx)
                del self._last_pkts[idx]
                del self._idle[idx]
        return expired

    def n_tracked(self) -> int:
        return len(self._last_pkts)
