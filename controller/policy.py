"""
controller/policy.py
Per-flow (and per-host-pair, Decision B) monitoring policy:
NORMAL -> MONITOR -> CLONE -> DROP.

The Isolation Forest score alone sets a raw level of at most CLONE
(Decision A: thresholds are the warm-up 5th percentile for MONITOR and the
warm-up 1st percentile for CLONE; score-based DROP and the z-score trigger
of the earlier switch-global design are both removed).

A DPI confirmation (on a head, sampled or level clone alike) escalates
the flow straight to DROP, bypassing the up-hysteresis -- a content/pattern
match is already strong evidence on its own, and waiting for K_UP anomalous
score windows would let the attack's packets through meanwhile (this
matters most for a scan, whose probes are each a short-lived flow): the
controller calls force_drop() on the flow, and for a scan class on its host
pair too (controller/monitor.py section 6). force_clone() remains for
callers that only want full capture.

Hysteresis (applies identically to flow keys and host-pair keys):
  * escalate only after K_UP = 2 consecutive windows whose raw level is above
    the current level; the new level is the lowest of those raw levels
  * de-escalate only after K_DOWN = 3 consecutive windows below the current
    level; the new level is the highest of those raw levels
  * force_clone()/force_drop() clear the per-key window history, so leaving
    a DPI-forced level still needs K_DOWN fresh windows below it, counted
    from the moment it was forced (not windows from before the attack
    was confirmed). The IF score never sets DROP, so a forced DROP dwells
    exactly K_DOWN scored windows unless confirmed again, then its rule is
    removed.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, Dict, Hashable, Optional, Tuple

from .model import Thresholds

logger = logging.getLogger(__name__)

K_UP = 2
K_DOWN = 3


class MonitoringLevel(IntEnum):
    NORMAL  = 0
    MONITOR = 1
    CLONE   = 2
    DROP    = 3


def score_to_level(score: float, th: Thresholds) -> MonitoringLevel:
    """
    Lower score = more anomalous. Returns at most CLONE: DROP is never set
    from a score alone (Decision A) -- see force_drop() below.
    """
    if score > th.monitor:
        return MonitoringLevel.NORMAL
    if score > th.clone:
        return MonitoringLevel.MONITOR
    return MonitoringLevel.CLONE


@dataclass
class FlowState:
    level  : MonitoringLevel = MonitoringLevel.NORMAL
    recent : Deque[MonitoringLevel] = field(
        default_factory=lambda: deque(maxlen=max(K_UP, K_DOWN)))
    # The installed data-plane rule's handle: a drop_table handle for a
    # flow-key FlowState, or a host_clone_table handle for a pair-key one
    rule_handle: Optional[int] = None
    # Pair keys only: host_drop_table handle while the pair is at DROP
    drop_handle: Optional[int] = None
    max_level  : MonitoringLevel = MonitoringLevel.NORMAL
    # Which detector/signal last moved this key, for logging only
    escalated_by: Optional[str] = None


class FlowPolicy:
    """
    Hysteresis state keyed by whatever the caller uses to identify a flow
    or a host pair -- e.g. (switch_id, slot) or (switch_id, src_ip, dst_ip).
    """

    def __init__(self, k_up: int = K_UP, k_down: int = K_DOWN):
        self.k_up = k_up
        self.k_down = k_down
        self.flows: Dict[Hashable, FlowState] = {}

    def state(self, key: Hashable) -> FlowState:
        st = self.flows.get(key)
        if st is None:
            st = self.flows[key] = FlowState(
                recent=deque(maxlen=max(self.k_up, self.k_down)))
        return st

    def forget(self, key: Hashable) -> Optional[FlowState]:
        return self.flows.pop(key, None)

    def update(self, key: Hashable, raw: MonitoringLevel
               ) -> Tuple[MonitoringLevel, MonitoringLevel]:
        """Feed one window's raw level; return (old_level, new_level)."""
        st = self.state(key)
        st.recent.append(raw)
        old = st.level
        last_up = list(st.recent)[-self.k_up:]
        last_dn = list(st.recent)[-self.k_down:]
        if len(last_up) == self.k_up and all(r > old for r in last_up):
            st.level = MonitoringLevel(min(last_up))
        elif len(last_dn) == self.k_down and all(r < old for r in last_dn):
            st.level = MonitoringLevel(max(last_dn))
        if st.level > st.max_level:
            st.max_level = st.level
        return old, st.level

    def _force(self, key: Hashable, level: MonitoringLevel) -> MonitoringLevel:
        st = self.state(key)
        old = st.level
        st.level = level
        st.recent.clear()
        if st.level > st.max_level:
            st.max_level = st.level
        return old

    def force_drop(self, key: Hashable) -> MonitoringLevel:
        """
        Escalate directly to DROP (a DPI-confirmed attack on a flow already
        at CLONE), bypassing the up-hysteresis. Clears the window history:
        leaving DROP needs K_DOWN fresh sub-DROP windows counted from now,
        not from before the confirmation.
        """
        return self._force(key, MonitoringLevel.DROP)

    def force_clone(self, key: Hashable) -> MonitoringLevel:
        """
        Escalate directly to CLONE, bypassing the up-hysteresis: a DPI
        confirmation on a flow/pair below CLONE is itself strong enough
        evidence (a content/pattern match, not a score) that waiting for
        K_UP consecutive anomalous windows would just let the attack's
        own packets -- already in hand -- go uninspected in the meantime.
        Clears the window history like force_drop(), for the same reason.
        """
        return self._force(key, MonitoringLevel.CLONE)
