"""
controller/policy.py
Monitoring policy engine.
Implements adaptive state machine: NORMAL -> MONITOR -> CLONE -> DROP
"""

import logging
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MonitoringLevel(Enum):
    NORMAL  = 0
    MONITOR = 1
    CLONE   = 2
    DROP    = 3


@dataclass
class PolicyConfig:
    """Thresholds for monitoring level transitions."""
    # Score thresholds (IsolationForest scores are negative for anomalies)
    # Lower score = more anomalous
    normal_threshold  : float = -0.10   # score > this => NORMAL
    monitor_threshold : float = -0.20   # score > this => MONITOR
    clone_threshold   : float = -0.30   # score > this => CLONE
    # score <= clone_threshold => DROP

    # Hysteresis: prevent rapid oscillation
    hysteresis_window : int   = 3       # consecutive readings to confirm transition
    drop_duration     : float = 30.0   # seconds to keep DROP state
    monitor_duration  : float = 60.0   # seconds to stay in MONITOR before reverting

    # Per-flow z-score for individual flow flagging
    flow_zscore_threshold : float = 4.0

    # Minimum flows to trigger DROP
    min_flows_for_drop : int = 5


@dataclass
class SwitchState:
    """Current monitoring state for one switch."""
    switch_id      : int
    level          : MonitoringLevel = MonitoringLevel.NORMAL
    last_score     : float           = 0.0
    level_history  : List[Tuple[float, MonitoringLevel]] = field(
        default_factory=list
    )
    level_since    : float           = field(default_factory=time.time)
    score_buffer   : List[float]     = field(default_factory=list)
    dropped_flows  : Set[str]        = field(default_factory=set)
    cloned_flows   : Set[str]        = field(default_factory=set)
    monitored_flows: Set[str]        = field(default_factory=set)
    transitions    : int             = 0

    def add_score(self, score: float) -> None:
        self.score_buffer.append(score)
        self.last_score = score
        if len(self.score_buffer) > 10:
            self.score_buffer.pop(0)

    def mean_recent_score(self, n: int = 3) -> float:
        recent = self.score_buffer[-n:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def set_level(self, level: MonitoringLevel) -> None:
        if level != self.level:
            logger.info(
                f"Switch {self.switch_id}: "
                f"{self.level.name} -> {level.name} "
                f"(score={self.last_score:.4f})"
            )
            self.transitions += 1
            self.level_since = time.time()
        self.level = level
        self.level_history.append((time.time(), level))
        if len(self.level_history) > 1000:
            self.level_history.pop(0)


class PolicyEngine:
    """
    Adaptive monitoring policy engine.
    Decides monitoring level per switch based on anomaly scores.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config  = config or PolicyConfig()
        self.states  : Dict[int, SwitchState] = {}
        self._action_log : List[Dict] = []

    def get_state(self, switch_id: int) -> SwitchState:
        if switch_id not in self.states:
            self.states[switch_id] = SwitchState(switch_id=switch_id)
        return self.states[switch_id]

    def evaluate(
        self,
        switch_id    : int,
        score        : float,
        n_active_flows: int = 0,
    ) -> MonitoringLevel:
        """
        Evaluate anomaly score and return recommended monitoring level.
        Uses hysteresis to avoid rapid transitions.
        """
        state = self.get_state(switch_id)
        state.add_score(score)

        avg_score = state.mean_recent_score(
            n=self.config.hysteresis_window
        )

        new_level = self._score_to_level(avg_score, n_active_flows)
        old_level = state.level

        # Only allow escalation immediately; de-escalation requires timeout
        if new_level.value > old_level.value:
            # Escalate
            state.set_level(new_level)
        elif new_level.value < old_level.value:
            # De-escalate: check if we've been in current state long enough
            time_in_state = time.time() - state.level_since
            if old_level == MonitoringLevel.DROP:
                min_stay = self.config.drop_duration
            elif old_level == MonitoringLevel.MONITOR:
                min_stay = self.config.monitor_duration
            else:
                min_stay = 5.0

            if time_in_state >= min_stay:
                state.set_level(new_level)

        self._log_action(switch_id, score, avg_score, state.level)
        return state.level

    def _score_to_level(
        self,
        score         : float,
        n_active_flows: int
    ) -> MonitoringLevel:
        """Map anomaly score to monitoring level."""
        cfg = self.config
        if score > cfg.normal_threshold:
            return MonitoringLevel.NORMAL
        elif score > cfg.monitor_threshold:
            return MonitoringLevel.MONITOR
        elif score > cfg.clone_threshold:
            return MonitoringLevel.CLONE
        else:
            # Only DROP if enough flows are active
            if n_active_flows >= cfg.min_flows_for_drop:
                return MonitoringLevel.DROP
            return MonitoringLevel.CLONE

    def evaluate_per_flow(
        self,
        switch_id  : int,
        flow_index : int,
        pps_zscore : float,
        bps_zscore : float
    ) -> MonitoringLevel:
        """
        Evaluate per-flow anomaly and return action.
        Returns CLONE for suspicious flows, DROP for extreme.
        """
        max_z = max(abs(pps_zscore), abs(bps_zscore))
        thresh = self.config.flow_zscore_threshold
        if max_z > thresh * 2:
            return MonitoringLevel.DROP
        elif max_z > thresh:
            return MonitoringLevel.CLONE
        elif max_z > thresh * 0.5:
            return MonitoringLevel.MONITOR
        return MonitoringLevel.NORMAL

    def should_drop_flow(
        self,
        switch_id  : int,
        flow_key   : str
    ) -> bool:
        state = self.get_state(switch_id)
        return flow_key in state.dropped_flows

    def add_dropped_flow(self, switch_id: int, flow_key: str) -> None:
        state = self.get_state(switch_id)
        state.dropped_flows.add(flow_key)

    def remove_dropped_flow(self, switch_id: int, flow_key: str) -> None:
        state = self.get_state(switch_id)
        state.dropped_flows.discard(flow_key)

    def add_cloned_flow(self, switch_id: int, flow_key: str) -> None:
        state = self.get_state(switch_id)
        state.cloned_flows.add(flow_key)

    def get_level(self, switch_id: int) -> MonitoringLevel:
        return self.get_state(switch_id).level

    def get_level_numeric(self, switch_id: int) -> int:
        return self.get_state(switch_id).level.value

    def get_transitions(self, switch_id: int) -> int:
        return self.get_state(switch_id).transitions

    def _log_action(
        self,
        switch_id   : int,
        raw_score   : float,
        smooth_score: float,
        level       : MonitoringLevel
    ) -> None:
        self._action_log.append({
            "timestamp"   : time.time(),
            "switch_id"   : switch_id,
            "raw_score"   : raw_score,
            "smooth_score": smooth_score,
            "level"       : level.value,
            "level_name"  : level.name,
        })
        if len(self._action_log) > 10000:
            self._action_log = self._action_log[-5000:]

    def get_action_log(self) -> List[Dict]:
        return self._action_log.copy()

    def get_level_distribution(
        self,
        switch_id : int
    ) -> Dict[str, int]:
        """Count time spent in each monitoring level."""
        state = self.get_state(switch_id)
        counts = {lvl.name: 0 for lvl in MonitoringLevel}
        for _, lvl in state.level_history:
            counts[lvl.name] += 1
        return counts

    def reset_switch(self, switch_id: int) -> None:
        """Reset a switch to NORMAL state."""
        if switch_id in self.states:
            state = self.states[switch_id]
            state.set_level(MonitoringLevel.NORMAL)
            state.dropped_flows.clear()
            state.cloned_flows.clear()
            state.score_buffer.clear()

    def get_summary(self) -> Dict:
        summary = {}
        for sw_id, state in self.states.items():
            summary[sw_id] = {
                "level"        : state.level.name,
                "last_score"   : state.last_score,
                "transitions"  : state.transitions,
                "dropped_flows": len(state.dropped_flows),
                "cloned_flows" : len(state.cloned_flows),
            }
        return summary
