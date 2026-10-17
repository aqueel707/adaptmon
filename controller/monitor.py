"""
controller/monitor.py
Main monitoring loop.
Orchestrates register polling, feature extraction, anomaly detection,
and policy enforcement.
"""

import threading
import time
import logging
import os
import csv
import json
from dataclasses import asdict
import psutil
from typing import Dict, List, Optional, Callable

from .dataplane  import DataPlaneManager
from .features   import FeatureExtractor
from .model      import AnomalyDetector, PerFlowDetector
from .aging      import FlowAger
from .policy     import PolicyEngine, MonitoringLevel, PolicyConfig, Thresholds
from .utils      import ip_int_to_str, flow_key

logger = logging.getLogger(__name__)

# Above this many expired slots in one poll, resetting the whole flow
# arrays (one CLI call) is cheaper than per-slot writes (~0.2 s per CLI
# call plus ~1.7 ms per slot), keeping a poll well under poll_interval
MAX_SLOT_CLEARS = 128


class MonitoringSession:
    """
    Encapsulates a single monitoring session for one switch.
    """

    def __init__(
        self,
        switch_id       : int,
        dpm             : DataPlaneManager,
        feature_extractor: FeatureExtractor,
        detector        : AnomalyDetector,
        per_flow_det    : PerFlowDetector,
        policy          : PolicyEngine,
        log_dir         : str,
        poll_interval   : float = 1.0,
        flow_idle_polls : int   = 3,
    ):
        self.switch_id        = switch_id
        self.dpm              = dpm
        self.features         = feature_extractor
        self.detector         = detector
        self.per_flow_det     = per_flow_det
        self.policy           = policy
        self.poll_interval    = poll_interval
        self.log_dir          = log_dir
        self.ager             = FlowAger(idle_polls=flow_idle_polls)
        self._n_aged          = 0

        self._running         = False
        self._thread          : Optional[threading.Thread] = None
        self._prev_pkts       : Dict[int, int] = {}
        self._prev_bytes      : Dict[int, int] = {}
        self._iteration       = 0
        self._csv_path        = os.path.join(log_dir, f"switch_{switch_id}_metrics.csv")
        self._thresholds_path = os.path.join(log_dir, f"switch_{switch_id}_thresholds.json")
        self._thresholds_set  = False
        self._csv_file        = None
        self._csv_writer      = None
        self._lock            = threading.Lock()

        os.makedirs(log_dir, exist_ok=True)

    def _open_csv(self) -> None:
        self._csv_file = open(self._csv_path, "w", newline="")
        fieldnames = [
            "timestamp", "iteration", "switch_id",
            "n_active_flows", "entropy", "mean_pps", "mean_bps",
            "max_pps", "max_bps", "std_pps", "std_bps",
            "gini", "top_flow_fraction",
            "anomaly_score", "z_max", "if_level", "z_level",
            "monitoring_level", "monitoring_level_name",
            "controller_cpu_pct", "controller_mem_mb",
            "n_dropped_flows", "n_cloned_flows", "n_aged_flows",
        ]
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=fieldnames
        )
        self._csv_writer.writeheader()

    def _close_csv(self) -> None:
        with self._lock:
            if self._csv_file:
                self._csv_file.flush()
                self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

    def start(self) -> None:
        self._open_csv()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name=f"monitor-sw{self.switch_id}",
            daemon=True
        )
        self._thread.start()
        logger.info(f"Monitor started for switch {self.switch_id}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        self._close_csv()
        logger.info(f"Monitor stopped for switch {self.switch_id}")

    def _loop(self) -> None:
        proc = psutil.Process(os.getpid())
        while self._running:
            t_start = time.time()
            try:
                self._poll_and_process(proc)
            except Exception as e:
                logger.error(
                    f"Monitor loop error (sw={self.switch_id}): {e}",
                    exc_info=True
                )
            elapsed = time.time() - t_start
            sleep_time = max(0.0, self.poll_interval - elapsed)
            time.sleep(sleep_time)

    def _poll_and_process(self, proc: psutil.Process) -> None:
        timestamp = time.time()
        self._iteration += 1

        # 1. Read registers
        pkts, bytes_ = self.dpm.collect_all_counters(self.switch_id)
        if not pkts:
            logger.debug(f"No counters from switch {self.switch_id}")
            return

        # 1b. Age out flow slots idle for flow_idle_polls polls
        self._age_flows(pkts, bytes_)

        # 2. Update feature extractor
        self.features.update_snapshot(pkts, bytes_, timestamp)

        # 3. Extract global feature vector
        fv = self.features.extract_feature_vector(pkts, bytes_, timestamp)

        # 4. Run anomaly detection (the first poll has no previous
        #    snapshot, so its rate features are all zero; skip it)
        score = (
            self.detector.ingest(fv, timestamp)
            if self._iteration > 1 else None
        )

        # 4b. Once the warm-up ends, derive this switch's thresholds
        if not self._thresholds_set:
            self._maybe_set_thresholds()

        # 4c. Complementary z-score signal vs. the benign warm-up
        z_max = self.detector.z_max(fv) if score is not None else None

        # 5. Evaluate policy
        n_active = int(fv[1])
        level = MonitoringLevel.NORMAL
        if score is not None:
            level = self.policy.evaluate(
                self.switch_id, score, n_active, z_max=z_max
            )
        else:
            score = 0.0

        # 6. Enforce policy actions
        self._enforce_policy(level, pkts, bytes_)

        # 7. Measure controller overhead
        cpu_pct = proc.cpu_percent(interval=None)
        mem_mb  = proc.memory_info().rss / 1024 / 1024

        # 8. Log metrics
        state = self.policy.get_state(self.switch_id)
        row = {
            "timestamp"           : timestamp,
            "iteration"           : self._iteration,
            "switch_id"           : self.switch_id,
            "n_active_flows"      : n_active,
            "entropy"             : fv[0],
            "mean_pps"            : fv[2],
            "mean_bps"            : fv[3],
            "max_pps"             : fv[4],
            "max_bps"             : fv[5],
            "std_pps"             : fv[6],
            "std_bps"             : fv[7],
            "gini"                : fv[8],
            "top_flow_fraction"   : fv[9],
            "anomaly_score"       : score,
            # Blank until the model is trained (like anomaly_score = 0.0)
            "z_max"               : z_max if z_max is not None else "",
            "if_level"            : state.last_if_level.value,
            "z_level"             : state.last_z_level.value,
            "monitoring_level"    : level.value,
            "monitoring_level_name": level.name,
            "controller_cpu_pct"  : cpu_pct,
            "controller_mem_mb"   : mem_mb,
            "n_dropped_flows"     : len(state.dropped_flows),
            "n_cloned_flows"      : len(state.cloned_flows),
            "n_aged_flows"        : self._n_aged,
        }
        with self._lock:
            if self._csv_writer:
                self._csv_writer.writerow(row)
                if self._iteration % 10 == 0:
                    self._csv_file.flush()

        logger.debug(
            f"SW{self.switch_id} iter={self._iteration} "
            f"score={score:.4f} level={level.name} "
            f"flows={n_active} cpu={cpu_pct:.1f}%"
        )

    def _maybe_set_thresholds(self) -> None:
        """
        After the detector freezes, set thresholds from its scores on the
        benign warm-up windows (fixed PolicyConfig thresholds if the
        warm-up was too short). Saved next to the metrics CSV for the
        figure scripts.
        """
        if self.detector.warmup_scores is not None:
            th = Thresholds.from_warmup(
                self.detector.warmup_scores, self.policy.config
            )
        elif self.detector.warmup_failed():
            th = Thresholds.fixed(
                self.policy.config, n_warmup=len(self.detector._buffer)
            )
        else:
            return
        self.policy.set_thresholds(self.switch_id, th)
        with open(self._thresholds_path, "w") as f:
            json.dump(asdict(th), f, indent=2)
        self._thresholds_set = True
        logger.info(
            f"SW{self.switch_id} thresholds ({th.source}, "
            f"n_warmup={th.n_warmup}): NORMAL > {th.normal:.4f}, "
            f"MONITOR > {th.monitor:.4f}, CLONE > {th.clone:.4f}, "
            f"else DROP"
        )

    def _age_flows(
        self,
        pkts  : Dict[int, int],
        bytes_: Dict[int, int]
    ) -> None:
        """
        Zero expired slots in the data plane and in this poll's snapshot,
        so the features see the table as it is after aging.
        Mass expiry (e.g. when a spoofed-source flood stops) resets the
        whole flow arrays instead: live flows lose their accumulated
        counts, which FlowRecord treats as a counter reset.
        """
        expired = self.ager.update(pkts)
        if not expired:
            return
        sw = self.dpm.get_switch(self.switch_id)
        if sw is None:
            return
        if len(expired) > MAX_SLOT_CLEARS:
            sw.reset_flow_registers()
            logger.info(
                f"SW{self.switch_id}: {len(expired)} flows expired, "
                f"reset flow registers"
            )
        else:
            sw.clear_flow_entries(expired)
        for idx in expired:
            pkts[idx]   = 0
            bytes_[idx] = 0
        self._n_aged += len(expired)

    def _enforce_policy(
        self,
        level : MonitoringLevel,
        pkts  : Dict[int, int],
        bytes_: Dict[int, int]
    ) -> None:
        """
        Apply data-plane actions based on current monitoring level.
        """
        sw = self.dpm.get_switch(self.switch_id)
        if sw is None:
            return

        if level == MonitoringLevel.DROP:
            # Find top talkers and install drop rules
            top = self.features.get_top_talkers(pkts, n=5)
            for flow_idx, _ in top:
                src_ip_int = sw.read_register("flow_src_ip", flow_idx)
                dst_ip_int = sw.read_register("flow_dst_ip", flow_idx)
                proto_val  = sw.read_register("flow_protocol", flow_idx)
                if src_ip_int and dst_ip_int:
                    src_ip = ip_int_to_str(src_ip_int)
                    dst_ip = ip_int_to_str(dst_ip_int)
                    proto  = proto_val if proto_val else 6
                    fk = flow_key(src_ip, dst_ip, proto)
                    if not self.policy.should_drop_flow(self.switch_id, fk):
                        ok = self.dpm.install_drop_rule(
                            self.switch_id, src_ip, dst_ip, proto
                        )
                        if ok:
                            self.policy.add_dropped_flow(self.switch_id, fk)
                            logger.info(
                                f"DROP rule installed: {src_ip} -> {dst_ip}"
                            )

        elif level == MonitoringLevel.CLONE:
            top = self.features.get_top_talkers(pkts, n=3)
            for flow_idx, _ in top:
                src_ip_int = sw.read_register("flow_src_ip", flow_idx)
                dst_ip_int = sw.read_register("flow_dst_ip", flow_idx)
                if src_ip_int and dst_ip_int:
                    src_ip = ip_int_to_str(src_ip_int)
                    dst_ip = ip_int_to_str(dst_ip_int)
                    fk = flow_key(src_ip, dst_ip, 0)
                    if fk not in self.policy.get_state(self.switch_id).cloned_flows:
                        ok = self.dpm.install_clone_rule(
                            self.switch_id, src_ip, dst_ip
                        )
                        if ok:
                            self.policy.add_cloned_flow(self.switch_id, fk)

        # In NORMAL / MONITOR: no special data-plane actions

    def get_metrics_path(self) -> str:
        return self._csv_path


class AdaptMonController:
    """
    Top-level controller managing all monitoring sessions.
    """

    def __init__(
        self,
        switch_configs : List[Dict],
        poll_interval  : float = 1.0,
        log_dir        : str   = "logs",
        policy_config  : Optional[PolicyConfig] = None,
        train_until    : Optional[float] = None,
        flow_idle_polls: int   = 3,
    ):
        self.switch_configs = switch_configs
        self.train_until    = train_until
        self.flow_idle_polls = flow_idle_polls
        self.poll_interval  = poll_interval
        self.log_dir        = log_dir
        self.policy_config  = policy_config or PolicyConfig()

        self.dpm      = DataPlaneManager(switch_configs)
        self.policy   = PolicyEngine(self.policy_config)
        self.sessions : Dict[int, MonitoringSession] = {}

        self._setup_sessions()

    def _setup_sessions(self) -> None:
        for cfg in self.switch_configs:
            sid = cfg["switch_id"]
            features    = FeatureExtractor(poll_interval=self.poll_interval)
            # A sample stamped t reads counters slightly after t, so stop
            # the warm-up one poll early to keep attack packets out of it.
            detector_train_until = (
                self.train_until - self.poll_interval
                if self.train_until is not None else None
            )
            detector    = AnomalyDetector(
                window_size      = 200,
                n_estimators     = 100,
                contamination    = 0.05,
                retrain_interval = 30,
                min_samples_to_train = 15,
                train_until      = detector_train_until,
            )
            per_flow_det = PerFlowDetector(zscore_threshold=3.5)
            session = MonitoringSession(
                switch_id        = sid,
                dpm              = self.dpm,
                feature_extractor= features,
                detector         = detector,
                per_flow_det     = per_flow_det,
                policy           = self.policy,
                log_dir          = self.log_dir,
                poll_interval    = self.poll_interval,
                flow_idle_polls  = self.flow_idle_polls,
            )
            self.sessions[sid] = session

    def start(self) -> None:
        cfg = self.policy_config
        if self.train_until is not None:
            logger.info(
                f"Thresholds: derived per switch from the benign warm-up; "
                f"fallback if warm-up < {cfg.min_warmup_for_thresholds} "
                f"samples: NORMAL > {cfg.normal_threshold}, "
                f"MONITOR > {cfg.monitor_threshold}, "
                f"CLONE > {cfg.clone_threshold}"
            )
            logger.info(
                f"z_max escalation (vs. warm-up, log1p space): "
                f"MONITOR > {cfg.z_monitor_threshold}, "
                f"CLONE > {cfg.z_clone_threshold}, "
                f"DROP > {cfg.z_drop_threshold}"
            )
        logger.info(f"Starting AdaptMon controller ({len(self.sessions)} switches)")
        for session in self.sessions.values():
            session.start()

    def stop(self) -> None:
        logger.info("Stopping AdaptMon controller")
        # Signal every session first so they wind down in parallel
        for session in self.sessions.values():
            session._running = False
        for session in self.sessions.values():
            session.stop()

    def wait_for_switches(self, timeout: int = 60) -> bool:
        return self.dpm.wait_for_switches(timeout)

    def install_forwarding(self, rules: List[Dict]) -> None:
        self.dpm.install_base_forwarding(rules)

    def get_metrics_paths(self) -> Dict[int, str]:
        return {
            sid: s.get_metrics_path()
            for sid, s in self.sessions.items()
        }

    def get_policy_summary(self) -> Dict:
        return self.policy.get_summary()

    def get_detector_stats(self) -> Dict:
        return {
            sid: s.detector.get_stats()
            for sid, s in self.sessions.items()
        }

    def reset_all(self) -> None:
        for sid in self.sessions:
            self.policy.reset_switch(sid)
            sw = self.dpm.get_switch(sid)
            if sw:
                sw.reset_all_counters()
