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
import psutil
from typing import Dict, List, Optional, Callable

from .dataplane  import DataPlaneManager
from .features   import FeatureExtractor
from .model      import AnomalyDetector, PerFlowDetector
from .policy     import PolicyEngine, MonitoringLevel, PolicyConfig
from .utils      import ip_int_to_str, flow_key

logger = logging.getLogger(__name__)


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
    ):
        self.switch_id        = switch_id
        self.dpm              = dpm
        self.features         = feature_extractor
        self.detector         = detector
        self.per_flow_det     = per_flow_det
        self.policy           = policy
        self.poll_interval    = poll_interval
        self.log_dir          = log_dir

        self._running         = False
        self._thread          : Optional[threading.Thread] = None
        self._prev_pkts       : Dict[int, int] = {}
        self._prev_bytes      : Dict[int, int] = {}
        self._iteration       = 0
        self._csv_path        = os.path.join(log_dir, f"switch_{switch_id}_metrics.csv")
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
            "anomaly_score", "monitoring_level", "monitoring_level_name",
            "controller_cpu_pct", "controller_mem_mb",
            "n_dropped_flows", "n_cloned_flows",
        ]
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=fieldnames
        )
        self._csv_writer.writeheader()

    def _close_csv(self) -> None:
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()

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

        # 2. Update feature extractor
        self.features.update_snapshot(pkts, bytes_, timestamp)

        # 3. Extract global feature vector
        fv = self.features.extract_feature_vector(pkts, bytes_, timestamp)

        # 4. Run anomaly detection
        score = self.detector.ingest(fv, timestamp)

        # 5. Evaluate policy
        n_active = int(fv[1])
        level = MonitoringLevel.NORMAL
        if score is not None:
            level = self.policy.evaluate(
                self.switch_id, score, n_active
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
            "monitoring_level"    : level.value,
            "monitoring_level_name": level.name,
            "controller_cpu_pct"  : cpu_pct,
            "controller_mem_mb"   : mem_mb,
            "n_dropped_flows"     : len(state.dropped_flows),
            "n_cloned_flows"      : len(state.cloned_flows),
        }
        if self._csv_writer:
            self._csv_writer.writerow(row)
            if self._iteration % 10 == 0:
                self._csv_file.flush()

        logger.debug(
            f"SW{self.switch_id} iter={self._iteration} "
            f"score={score:.4f} level={level.name} "
            f"flows={n_active} cpu={cpu_pct:.1f}%"
        )

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
    ):
        self.switch_configs = switch_configs
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
            detector    = AnomalyDetector(
                window_size      = 200,
                n_estimators     = 100,
                contamination    = 0.05,
                retrain_interval = 30,
                min_samples_to_train = 15,
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
            )
            self.sessions[sid] = session

    def start(self) -> None:
        logger.info(f"Starting AdaptMon controller ({len(self.sessions)} switches)")
        for session in self.sessions.values():
            session.start()

    def stop(self) -> None:
        logger.info("Stopping AdaptMon controller")
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
