"""
controller/monitor.py
Main monitoring loop (per-flow detection).

Every poll window, for every switch:
  1. read the flow registers (simple_switch_CLI register_read over Thrift)
  2. build one feature vector per occupied slot (controller/features.py)
  3. before the warm-up ends: buffer the vectors; at the first poll after it,
     fit the Isolation Forest once on them (controller/model.py)
  4. score every flow; map score -> raw level; per-flow hysteresis
     (controller/policy.py)
  5. write changed levels to the flow_level register; install a 5-tuple drop
     rule on entering DROP, delete it on leaving DROP; emit an anomaly alert
     when a flow reaches CLONE or above
  6. age out slots idle for flow_idle_polls polls (zero the slot, reset its
     level, delete its drop rule)

Outputs in out_dir:
  flow_windows.csv  one row per (window, switch, flow slot)
  windows.csv       one row per window: timings, counts, CPU/RSS
  alerts.jsonl      anomaly alerts (DPI appends confirmed alerts)
  model.json        training summary + thresholds
"""

import csv
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import psutil

from .aging import FlowAger
from .alerts import AlertLog
from .dataplane import DataPlaneManager, FLOW_REGISTERS
from .features import FEATURE_NAMES, ENTROPY_SOURCE, FlowFeatureExtractor, SwitchWindow
from .model import FlowIsolationForest
from .policy import FlowPolicy, MonitoringLevel, score_to_level, K_UP, K_DOWN

logger = logging.getLogger(__name__)

LEVEL_NAMES = [lv.name for lv in MonitoringLevel]


class AdaptMonController:
    """
    Top-level controller: one polling loop over all switches.
    """

    def __init__(
        self,
        switch_configs : List[Dict],
        poll_interval  : float = 1.0,
        out_dir        : str   = "logs",
        train_until    : Optional[float] = None,
        flow_idle_polls: int   = 3,
        dpi_pid        : Optional[int] = None,
    ):
        self.switch_configs  = switch_configs
        self.poll_interval   = poll_interval
        self.out_dir         = out_dir
        self.train_until     = train_until
        self.flow_idle_polls = flow_idle_polls
        self.dpi_pid         = dpi_pid
        os.makedirs(out_dir, exist_ok=True)

        self.dpm      = DataPlaneManager(switch_configs)
        self.model    = FlowIsolationForest()
        self.policy   = FlowPolicy()
        self.alerts   = AlertLog(os.path.join(out_dir, "alerts.jsonl"))
        sids = [c["switch_id"] for c in switch_configs]
        self.extractors = {sid: FlowFeatureExtractor(sid) for sid in sids}
        self.agers      = {sid: FlowAger(idle_polls=flow_idle_polls) for sid in sids}
        # Per (switch, slot): the 5-tuple the policy state belongs to
        self._slot_tuple: Dict[tuple, tuple] = {}

        self._warmup_X: List[np.ndarray] = []
        self._n_warmup_vectors = 0
        self._train_failed = False
        self._window = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid())
        self._dpi_proc = None
        if dpi_pid:
            try:
                self._dpi_proc = psutil.Process(dpi_pid)
                self._dpi_proc.cpu_percent(None)
            except psutil.Error:
                self._dpi_proc = None

        self._flow_f = open(os.path.join(out_dir, "flow_windows.csv"), "w", newline="")
        self._flow_w = csv.writer(self._flow_f)
        self._flow_w.writerow(
            ["ts", "window", "switch", "slot", "src", "dst", "proto", "sport", "dport"]
            + FEATURE_NAMES + ["pkt_delta", "scored", "score", "raw_level", "level"]
        )
        self._win_f = open(os.path.join(out_dir, "windows.csv"), "w", newline="")
        self._win_fields = [
            "ts", "window", "phase", "n_flows_total",
            "t_read_ms", "t_feat_ms", "t_infer_ms", "t_policy_ms", "t_write_ms",
            "t_total_ms", "cpu_pct", "rss_mb", "dpi_cpu_pct", "dpi_rss_mb",
        ]
        for sid in sids:
            self._win_fields += [
                f"s{sid}_{k}" for k in
                ["n_flows", "n_normal", "n_monitor", "n_clone", "n_drop",
                 "fwd", "cloned", "dropped", "n_aged", "n_tuple_changes",
                 "n_drop_rules", "entropy"]
            ]
        self._win_w = csv.DictWriter(self._win_f, fieldnames=self._win_fields)
        self._win_w.writeheader()

    # ------------------------------------------------------------------
    def wait_for_switches(self, timeout: int = 60) -> bool:
        return self.dpm.wait_for_switches(timeout)

    def install_forwarding(self, rules: List[Dict]) -> None:
        self.dpm.install_base_forwarding(rules)

    def start(self) -> None:
        logger.info(
            "z-score trigger: DISABLED (removed); the IsolationForest score "
            "alone sets each flow's level"
        )
        logger.info(
            f"Per-flow detection: features={FEATURE_NAMES}, "
            f"entropy source={ENTROPY_SOURCE}, hysteresis up={K_UP} / down={K_DOWN} windows, "
            f"train_until={self.train_until}"
        )
        for sw in self.dpm.all_switches():
            sw.reset_dp_counters()
        self._proc.cpu_percent(None)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="monitor", daemon=True)
        self._thread.start()
        logger.info(f"Monitor started for switches {sorted(self.extractors)}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=15)
        self._flow_f.close()
        self._win_f.close()
        logger.info("Monitor stopped")

    def _loop(self) -> None:
        next_t = time.time()
        while self._running:
            try:
                self._poll()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)
            next_t += self.poll_interval
            time.sleep(max(0.0, next_t - time.time()))
            if next_t < time.time() - self.poll_interval:
                next_t = time.time()   # overran by more than a window

    # ------------------------------------------------------------------
    def _poll(self, ts: Optional[float] = None) -> None:
        """One window. ts (epoch) is the read time; tests may pass it."""
        self._window += 1
        t0 = time.perf_counter()
        ts = time.time() if ts is None else ts
        row = {"ts": ts, "window": self._window}

        # 1. Read registers + data-plane counters
        regs, dp = {}, {}
        for sid, sw in self.dpm.switches.items():
            regs[sid] = sw.read_registers(FLOW_REGISTERS)
            dp[sid] = sw.read_dp_counters()
        t1 = time.perf_counter()

        # 2. Features
        wins: Dict[int, SwitchWindow] = {}
        for sid, r in regs.items():
            w = self.extractors[sid].extract(r, ts)
            if w is not None:
                wins[sid] = w
        t2 = time.perf_counter()

        # 3. Warm-up buffering / one-time training
        in_warmup = self.train_until is not None and ts < self.train_until
        if not self.model.trained and not in_warmup and not self._train_failed \
                and self._warmup_X:
            th = self.model.fit_once(np.vstack(self._warmup_X))
            if th is None:
                self._train_failed = True
            with open(os.path.join(self.out_dir, "model.json"), "w") as f:
                json.dump({"trained": th is not None, "train_until": self.train_until,
                           "entropy_source": ENTROPY_SOURCE, **self.model.train_info},
                          f, indent=2)
        if in_warmup:
            for w in wins.values():
                if len(w.X):
                    self._warmup_X.append(w.X)
                    self._n_warmup_vectors += len(w.X)

        # 4. Score
        scores: Dict[int, Optional[np.ndarray]] = {}
        for sid, w in wins.items():
            scores[sid] = self.model.score(w.X) if self.model.trained else None
        t3 = time.perf_counter()

        # 5. Policy
        level_writes: Dict[int, Dict[int, int]] = {sid: {} for sid in wins}
        drop_adds, drop_dels = [], []
        new_alerts = []
        for sid, w in wins.items():
            counts = [0, 0, 0, 0]
            sc = scores[sid]
            for j, slot in enumerate(w.slots):
                slot = int(slot)
                key = (sid, slot)
                tup = w.tuples[j]
                if self._slot_tuple.get(key) != tup:
                    # New flow in this slot (or a colliding one overwrote the
                    # 5-tuple): start its policy state afresh
                    old_st = self.policy.forget(key)
                    if old_st is not None and old_st.drop_handle is not None:
                        drop_dels.append((sid, old_st.drop_handle))
                    if old_st is not None and old_st.level != MonitoringLevel.NORMAL:
                        level_writes[sid][slot] = 0
                    self._slot_tuple[key] = tup
                raw = lvl = MonitoringLevel.NORMAL
                score = ""
                if sc is not None:
                    score = float(sc[j])
                    raw = score_to_level(score, self.model.thresholds)
                    old, lvl = self.policy.update(key, raw)
                    if lvl != old:
                        level_writes[sid][slot] = int(lvl)
                        st = self.policy.state(key)
                        if lvl == MonitoringLevel.DROP and st.drop_handle is None:
                            drop_adds.append((sid, key, tup))
                        if old == MonitoringLevel.DROP and lvl < MonitoringLevel.DROP \
                                and st.drop_handle is not None:
                            drop_dels.append((sid, st.drop_handle))
                            st.drop_handle = None
                        if lvl > old and lvl >= MonitoringLevel.CLONE:
                            new_alerts.append({
                                "type": "anomaly", "ts": ts, "window": self._window,
                                "switch": sid, "flow_idx": slot,
                                "five_tuple": dict(zip(
                                    ("src", "dst", "proto", "sport", "dport"), tup)),
                                "score": score, "raw_level": raw.name,
                                "level": int(lvl), "level_name": lvl.name,
                                "prev_level": old.name,
                            })
                        logger.debug(f"S{sid} slot {slot} {tup}: {old.name} -> {lvl.name} "
                                     f"(score={score:.4f})")
                counts[lvl] += 1
                self._flow_w.writerow(
                    [f"{ts:.6f}", self._window, sid, slot, *tup]
                    + [f"{v:.6g}" for v in w.X[j]]
                    + [int(w.pkt_delta[j]), int(sc is not None),
                       f"{score:.6f}" if score != "" else "", int(raw), int(lvl)]
                )
            row.update({
                f"s{sid}_n_flows": len(w.slots), f"s{sid}_n_normal": counts[0],
                f"s{sid}_n_monitor": counts[1], f"s{sid}_n_clone": counts[2],
                f"s{sid}_n_drop": counts[3], f"s{sid}_entropy": f"{w.entropy:.4f}",
                f"s{sid}_n_tuple_changes": w.n_tuple_changes,
            })
        for a in new_alerts:
            self.alerts.write(a)
        t4 = time.perf_counter()

        # 6. Write back: levels, drop rules, aging
        for sid, w in wins.items():
            sw = self.dpm.get_switch(sid)
            for dsid, handle in [d for d in drop_dels if d[0] == sid]:
                sw.delete_drop_rule(handle)
            for asid, key, tup in [d for d in drop_adds if d[0] == sid]:
                handle = sw.add_drop_rule(*tup)
                self.policy.state(key).drop_handle = handle
            expired = self.agers[sid].update(
                {int(s): int(p) for s, p in zip(w.slots, w.X[:, 0])})
            for slot in expired:
                st = self.policy.forget((sid, slot))
                self._slot_tuple.pop((sid, slot), None)
                level_writes[sid].pop(slot, None)
                if st is not None and st.drop_handle is not None:
                    sw.delete_drop_rule(st.drop_handle)
            if expired:
                sw.clear_flow_entries(expired)      # also resets flow_level
                self.extractors[sid].forget(expired)
            if level_writes[sid]:
                sw.write_flow_levels(level_writes[sid])
            row[f"s{sid}_n_aged"] = len(expired)
            row[f"s{sid}_n_drop_rules"] = sum(
                1 for (s, _), st in self.policy.flows.items()
                if s == sid and st.drop_handle is not None)
        t5 = time.perf_counter()

        for sid, c in dp.items():
            row.update({f"s{sid}_fwd": c["forwarded"], f"s{sid}_cloned": c["cloned"],
                        f"s{sid}_dropped": c["dropped"]})
        row.update({
            "phase": "warmup" if in_warmup else ("scored" if self.model.trained else "untrained"),
            "n_flows_total": sum(len(w.slots) for w in wins.values()),
            "t_read_ms": (t1 - t0) * 1e3, "t_feat_ms": (t2 - t1) * 1e3,
            "t_infer_ms": (t3 - t2) * 1e3, "t_policy_ms": (t4 - t3) * 1e3,
            "t_write_ms": (t5 - t4) * 1e3, "t_total_ms": (t5 - t0) * 1e3,
            "cpu_pct": self._proc.cpu_percent(None),
            "rss_mb": self._proc.memory_info().rss / 2**20,
        })
        if self._dpi_proc is not None:
            try:
                row["dpi_cpu_pct"] = self._dpi_proc.cpu_percent(None)
                row["dpi_rss_mb"] = self._dpi_proc.memory_info().rss / 2**20
            except psutil.Error:
                self._dpi_proc = None
        self._win_w.writerow(row)
        if self._window % 5 == 0:
            self._win_f.flush()
            self._flow_f.flush()

    def summary(self) -> Dict:
        return {
            "windows": self._window,
            "trained": self.model.trained,
            "train_info": self.model.train_info,
            "flows_tracked": len(self.policy.flows),
            "alerts_written": self.alerts.n_written,
        }
