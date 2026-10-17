"""
controller/policy.py
Per-flow monitoring policy: NORMAL -> MONITOR -> CLONE -> DROP.

The Isolation Forest score alone sets a flow's raw level each window
(thresholds from controller.model.Thresholds). The z-score trigger of the
earlier switch-global design is removed.

Hysteresis per flow:
  * escalate only after K_UP = 2 consecutive windows whose raw level is above
    the current level; the new level is the lowest of those raw levels
  * de-escalate only after K_DOWN = 3 consecutive windows below the current
    level; the new level is the highest of those raw levels
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
    """Lower score = more anomalous."""
    if score > th.normal:
        return MonitoringLevel.NORMAL
    if score > th.monitor:
        return MonitoringLevel.MONITOR
    if score > th.clone:
        return MonitoringLevel.CLONE
    return MonitoringLevel.DROP


@dataclass
class FlowState:
    level  : MonitoringLevel = MonitoringLevel.NORMAL
    recent : Deque[MonitoringLevel] = field(
        default_factory=lambda: deque(maxlen=max(K_UP, K_DOWN)))
    drop_handle: Optional[int] = None
    max_level  : MonitoringLevel = MonitoringLevel.NORMAL


class FlowPolicy:

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
