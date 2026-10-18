"""
controller/monitor.py
Main monitoring loop: two detectors plus DPI confirmation.

Every poll window, for every switch:
  1. read the flow registers (simple_switch_CLI register_read over Thrift)
  2. build one feature vector per occupied slot (controller/features.py);
     merge in payload-sampled entropy from DPI's live snapshot where
     available, else keep the flow-distribution entropy (Decision C)
  3. before the warm-up ends: buffer the flow vectors AND the per-
     (src,dst) host-pair aggregates; at the first poll after it, fit both
     Isolation Forests once (controller/model.py)
  4. score every flow and every host pair; map score -> raw level (at most
     CLONE: Decision A); per-key hysteresis (controller/policy.py)
  5. a host pair escalating to CLONE installs a src-IP clone rule (every
     packet from that source is cloned from then on: Decision B)
  6. tail alerts.jsonl for DPI-confirmed attacks; a confirmed attack
     forces the flow to DROP (5-tuple drop rule) and, for a scan class, its
     host pair (source-IP drop rule); write back changed flow_level entries,
     (de)install drop/host-clone/host-drop rules
  7. age out slots idle for flow_idle_polls polls

At start() the controller writes head_clone_n (flow-head cloning: the
first N packets of every new flow go to DPI) to every switch and logs the
value read back.

Outputs in out_dir:
  flow_windows.csv  one row per (window, switch, flow slot)
  pair_windows.csv  one row per (window, switch, src, dst) host pair
  windows.csv       one row per window: timings, counts, CPU/RSS, and the
                    data-plane counters (cumulative since start)
  alerts.jsonl       anomaly / pair_escalation / forced_clone / forced_drop alerts
                     (DPI appends "confirmed" alerts to the same file)
  model.json        training summary + thresholds for both models
"""

import csv
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

from .aging import FlowAger
from .alerts import AlertLog, AlertTailer
from .dataplane import DataPlaneManager, FLOW_REGISTERS
from .features import (
    FEATURE_NAMES, PAIR_FEATURE_NAMES, ENTROPY_IDX,
    ENTROPY_FALLBACK_SOURCE, ENTROPY_PRIMARY_SOURCE,
    FlowFeatureExtractor, SwitchWindow, aggregate_host_pairs,
    EntropySnapshotReader,
)
from .model import FlowIsolationForest, HostPairIsolationForest
from .policy import FlowPolicy, MonitoringLevel, score_to_level, K_UP, K_DOWN

logger = logging.getLogger(__name__)

LEVEL_NAMES = [lv.name for lv in MonitoringLevel]

# How many poll windows a DPI-confirmed alert is retried for before a flow
# that never reaches/returns to CLONE (or goes inactive) gives up on it
CONFIRMATION_RETRY_WINDOWS = 10

# escalated_by trigger taxonomy: "IF-flow" (the flow's own Isolation Forest
# score), "IF-pair" (Decision B's host-pair aggregate score), "DPI-head" /
# "DPI-sample" (a DPI confirmation on a flow/pair below CLONE, force_clone()'d
# straight there, on a flow-head clone / an adaptive-sampling clone -- see
# section 6 of _poll() and _dpi_trigger()). force_drop() keeps its own
# "dpi_confirmed:<class>" tag, since that is a second, distinct event on a
# flow already at CLONE, not an escalation trigger by itself.
SCAN_ATTACK_CLASSES = {"syn_scan", "null_scan"}

# Extra per-switch windows.csv columns from read_dp_counters_full(), next to
# s<sid>_fwd/_cloned/_dropped (packets). ALL data-plane counter columns are
# cumulative since start() reset them -- a run's totals are its last row,
# not a sum over rows.
DP_COUNTER_FIELDS = [
    "fwd_bytes", "cloned_bytes", "dropped_bytes",
    "clone_level", "clone_level_bytes", "clone_sample", "clone_sample_bytes",
    "clone_hostpair", "clone_hostpair_bytes", "clone_head", "clone_head_bytes",
]

# cpu_hdr.reason (p4/adaptmon.p4) of the clone that let DPI see a flow ->
# escalated_by for a DPI confirmation below CLONE. Reason 3 (host-pair rule)
# is resolved at runtime to whatever escalated that pair -- see
# _dpi_trigger(). Reason 1 means the packet was a level clone: the flow was
# already fully captured at CLONE (e.g. escalated by its IF score).
DPI_TRIGGER_BY_REASON = {2: "DPI-sample", 4: "DPI-head", 1: "DPI-level"}


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
        head_clone_n   : int   = 0,
    ):
        self.switch_configs  = switch_configs
        self.poll_interval   = poll_interval
        self.out_dir         = out_dir
        self.train_until     = train_until
        self.flow_idle_polls = flow_idle_polls
        self.dpi_pid         = dpi_pid
        self.head_clone_n    = head_clone_n
        self.head_clone_n_readback: Dict[int, Optional[int]] = {}
        os.makedirs(out_dir, exist_ok=True)

        self.dpm        = DataPlaneManager(switch_configs)
        self.model      = FlowIsolationForest()
        self.pair_model = HostPairIsolationForest()
        self.policy      = FlowPolicy()      # keyed (switch_id, slot)
        self.pair_policy = FlowPolicy()      # keyed (switch_id, src_ip, dst_ip)
        self.alerts  = AlertLog(os.path.join(out_dir, "alerts.jsonl"))
        self._alert_tailer = AlertTailer(os.path.join(out_dir, "alerts.jsonl"))
        self._entropy_reader = EntropySnapshotReader(
            os.path.join(out_dir, "dpi_entropy_live.json"))
        sids = [c["switch_id"] for c in switch_configs]
        self.extractors = {sid: FlowFeatureExtractor(sid) for sid in sids}
        self.agers      = {sid: FlowAger(idle_polls=flow_idle_polls) for sid in sids}
        # Per (switch, slot): the 5-tuple the policy state belongs to
        self._slot_tuple: Dict[tuple, tuple] = {}
        # DPI confirmations not yet actionable (flow not at CLONE yet, or
        # not active this window): (alert, window_first_seen)
        self._pending_confirmations: List[Tuple[dict, int]] = []

        self._warmup_X: List[np.ndarray] = []
        self._n_warmup_vectors = 0
        self._warmup_pair_X: List[np.ndarray] = []
        self._n_warmup_pair_vectors = 0
        self._training_attempted = False
        self._window = 0
        self._n_forced_drops = 0
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
            + FEATURE_NAMES
            + ["entropy_source", "pkt_delta", "drop_pkts", "scored", "score", "raw_level", "level",
               "escalated_by"]
        )
        self._pair_f = open(os.path.join(out_dir, "pair_windows.csv"), "w", newline="")
        self._pair_w = csv.writer(self._pair_f)
        self._pair_w.writerow(
            ["ts", "window", "switch", "src", "dst"] + PAIR_FEATURE_NAMES
            + ["scored", "score", "raw_level", "level", "escalated_by"]
        )
        self._win_f = open(os.path.join(out_dir, "windows.csv"), "w", newline="")
        self._win_fields = [
            "ts", "window", "phase", "n_flows_total",
            "t_read_ms", "t_feat_ms", "t_infer_ms", "t_policy_ms", "t_write_ms",
            "t_total_ms", "cpu_pct", "rss_mb", "dpi_cpu_pct", "dpi_rss_mb",
            "n_forced_drops_total", "n_pending_confirmations",
        ]
        for sid in sids:
            self._win_fields += [
                f"s{sid}_{k}" for k in
                ["n_flows", "n_normal", "n_monitor", "n_clone", "n_drop",
                 "fwd", "cloned", "dropped", "n_aged", "n_tuple_changes",
                 "n_drop_rules", "entropy",
                 "n_pairs", "n_host_clone_rules", "n_host_drop_rules",
                 "n_entropy_payload", "n_entropy_fallback"]
                + DP_COUNTER_FIELDS
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
            "Decision A: score -> level thresholds are per-model warm-up "
            "percentiles (MONITOR < p5, CLONE < p1); DROP is never set from "
            "a score -- only by DPI. Any DPI confirmation forces the flow to DROP (5-tuple drop "
            "rule) and, for a scan class, its source (source-IP drop rule). "
            "The z-score trigger of "
            "the earlier switch-global design is removed."
        )
        logger.info(
            f"Decision B: a second IsolationForest scores host-pair "
            f"aggregates {PAIR_FEATURE_NAMES}; a pair escalating to CLONE "
            f"installs a src-IP clone rule (every packet from that source "
            f"is cloned from then on)."
        )
        logger.info(
            "Decision C: data-plane adaptive sampling (NORMAL 1-in-64, "
            "MONITOR 1-in-8, CLONE/DROP every packet); the entropy feature "
            f"is payload byte entropy from sampled clones ({ENTROPY_PRIMARY_SOURCE}) "
            f"when available this window, else flow-distribution entropy "
            f"({ENTROPY_FALLBACK_SOURCE})."
        )
        logger.info(
            f"Flow features={FEATURE_NAMES}, hysteresis up={K_UP} / down={K_DOWN} "
            f"windows, train_until={self.train_until}"
        )
        for sid, sw in self.dpm.switches.items():
            self.head_clone_n_readback[sid] = sw.set_head_clone_n(self.head_clone_n)
        ok = all(v == self.head_clone_n for v in self.head_clone_n_readback.values())
        (logger.info if ok else logger.error)(
            f"Flow-head cloning: the first N={self.head_clone_n} packets of every "
            f"new flow are cloned to the CPU port (clone_reason 4; N=0 disables). "
            f"head_clone_n read back per switch: {self.head_clone_n_readback}"
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
        self._pair_f.close()
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
            dp[sid] = sw.read_dp_counters_full()
        t1 = time.perf_counter()

        # 2. Features
        wins: Dict[int, SwitchWindow] = {}
        for sid, r in regs.items():
            w = self.extractors[sid].extract(r, ts)
            if w is not None:
                wins[sid] = w

        # 2b. Entropy feature (Decision C): payload-sampled when DPI has
        # seen at least one sampled byte of this flow, else keep the
        # flow-distribution value features.py already filled in
        self._entropy_reader.refresh()
        entropy_source: Dict[int, List[str]] = {}
        for sid, w in wins.items():
            sources = []
            for j, tup in enumerate(w.tuples):
                snap = self._entropy_reader.lookup(sid, tup)
                if snap is not None and snap.get("n", 0) > 0:
                    w.X[j, ENTROPY_IDX] = snap["entropy"]
                    sources.append(ENTROPY_PRIMARY_SOURCE)
                else:
                    sources.append(ENTROPY_FALLBACK_SOURCE)
            entropy_source[sid] = sources
        t2 = time.perf_counter()

        # 2c. Host-pair aggregates (Decision B)
        pairs: Dict[int, Dict[Tuple[str, str], np.ndarray]] = {
            sid: aggregate_host_pairs(w) for sid, w in wins.items()
        }

        # 3. Warm-up buffering / one-time training. The flow model and the
        # host-pair model are attempted exactly once, together, but
        # INDEPENDENTLY: this lab topology has few distinct host pairs, so
        # the pair model is the one at real risk of too few warm-up samples
        # (MIN_TRAIN_SAMPLES is per-sample, and there is at most one sample
        # per active pair per window, vs. one per active flow for the flow
        # model). A pair-model failure must not block flow detection, and
        # must not be retried (no more warm-up samples will ever arrive).
        in_warmup = self.train_until is not None and ts < self.train_until
        if not self._training_attempted and not in_warmup and self._warmup_X:
            self._training_attempted = True
            th = self.model.fit_once(np.vstack(self._warmup_X))
            pair_X = (np.vstack(self._warmup_pair_X) if self._warmup_pair_X
                     else np.zeros((0, len(PAIR_FEATURE_NAMES))))
            pth = self.pair_model.fit_once(pair_X)
            if pth is None:
                logger.error(
                    "Host-pair model NOT trained (insufficient warm-up pair "
                    "samples: host pairs are counted once per window, not "
                    "once per flow); host-pair detection disabled for this "
                    "run. Flow-level detection is unaffected."
                )
            with open(os.path.join(self.out_dir, "model.json"), "w") as f:
                json.dump({
                    "trained": th is not None and pth is not None,
                    "train_until": self.train_until,
                    "entropy_primary_source": ENTROPY_PRIMARY_SOURCE,
                    "entropy_fallback_source": ENTROPY_FALLBACK_SOURCE,
                    "data_plane": {"head_clone_n": self.head_clone_n,
                                   "dpi_confirmation_action":
                                       "DROP (flow 5-tuple; + source IP for syn/null scans)",
                                   "head_clone_n_readback": self.head_clone_n_readback},
                    "flow_model": {"trained": th is not None, **self.model.train_info},
                    "host_pair_model": {"trained": pth is not None, **self.pair_model.train_info},
                }, f, indent=2)
        if in_warmup:
            for w in wins.values():
                if len(w.X):
                    self._warmup_X.append(w.X)
                    self._n_warmup_vectors += len(w.X)
            for p in pairs.values():
                if p:
                    self._warmup_pair_X.append(np.vstack(list(p.values())))
                    self._n_warmup_pair_vectors += len(p)

        # 4. Score
        scores: Dict[int, Optional[np.ndarray]] = {}
        for sid, w in wins.items():
            scores[sid] = self.model.score(w.X) if self.model.trained else None
        pair_scores: Dict[int, Dict[Tuple[str, str], float]] = {}
        for sid, p in pairs.items():
            if self.pair_model.trained and p:
                keys = list(p.keys())
                sc = self.pair_model.score(np.vstack([p[k] for k in keys]))
                pair_scores[sid] = dict(zip(keys, sc))
            else:
                pair_scores[sid] = {}
        t3 = time.perf_counter()

        # Dropped packets per slot (measurement only: p4/adaptmon.p4 drop_pkts)
        drop_regs = {sid: r.get("drop_pkts") for sid, r in regs.items()}

        # 5a. Flow-level policy
        level_writes: Dict[int, Dict[int, int]] = {sid: {} for sid in wins}
        drop_adds, drop_dels = [], []
        new_alerts = []
        tuple_to_slot: Dict[int, Dict[tuple, int]] = {}
        # (policy_key, row_prefix) pairs; the final level/escalated_by -- set
        # after section 6 may still change them -- are appended just before
        # writing, at the end of _poll() (see "flush buffered rows" below)
        flow_row_buffer: List[tuple] = []
        pair_row_buffer: List[tuple] = []
        for sid, w in wins.items():
            counts = [0, 0, 0, 0]
            sc = scores[sid]
            tuple_to_slot[sid] = {}
            for j, slot in enumerate(w.slots):
                slot = int(slot)
                key = (sid, slot)
                tup = w.tuples[j]
                tuple_to_slot[sid][tup] = slot
                if self._slot_tuple.get(key) != tup:
                    # New flow in this slot (or a colliding one overwrote the
                    # 5-tuple): start its policy state afresh
                    old_st = self.policy.forget(key)
                    if old_st is not None and old_st.rule_handle is not None:
                        drop_dels.append((sid, old_st.rule_handle))
                    if old_st is not None and old_st.level != MonitoringLevel.NORMAL:
                        level_writes[sid][slot] = 0
                    self._slot_tuple[key] = tup
                raw = lvl = MonitoringLevel.NORMAL
                score = ""
                escalated_by = self.policy.state(key).escalated_by
                if sc is not None:
                    score = float(sc[j])
                    raw = score_to_level(score, self.model.thresholds)
                    old, lvl = self.policy.update(key, raw)
                    if lvl != old:
                        level_writes[sid][slot] = int(lvl)
                        st = self.policy.state(key)
                        if lvl > old:
                            st.escalated_by = "IF-flow"
                        if old == MonitoringLevel.DROP and lvl < MonitoringLevel.DROP \
                                and st.rule_handle is not None:
                            drop_dels.append((sid, st.rule_handle))
                            st.rule_handle = None
                            st.escalated_by = None
                        escalated_by = st.escalated_by
                        if lvl > old and lvl >= MonitoringLevel.CLONE:
                            new_alerts.append({
                                "type": "anomaly", "detector": "flow", "ts": ts,
                                "window": self._window, "switch": sid, "flow_idx": slot,
                                "five_tuple": dict(zip(
                                    ("src", "dst", "proto", "sport", "dport"), tup)),
                                "score": score, "raw_level": raw.name,
                                "level": int(lvl), "level_name": lvl.name,
                                "prev_level": old.name,
                            })
                        logger.debug(f"S{sid} slot {slot} {tup}: {old.name} -> {lvl.name} "
                                     f"(score={score:.4f}, by=IF-flow)")
                counts[lvl] += 1
                src_count = (entropy_source[sid][j] if sid in entropy_source
                            and j < len(entropy_source[sid]) else ENTROPY_FALLBACK_SOURCE)
                # Buffered, not written yet: section 6 (DPI confirmation) can
                # still force_clone/force_drop this same key later in this
                # same poll, and the row must reflect that final state, not
                # this mid-poll snapshot -- see flow_row_buffer below.
                flow_row_buffer.append((
                    key,
                    [f"{ts:.6f}", self._window, sid, slot, *tup]
                    + [f"{v:.6g}" for v in w.X[j]]
                    + [src_count, int(w.pkt_delta[j]),
                       drop_regs[sid][slot] if drop_regs.get(sid) else "",
                       int(sc is not None),
                       f"{score:.6f}" if score != "" else "", int(raw)]
                ))
            row.update({
                f"s{sid}_n_flows": len(w.slots), f"s{sid}_n_normal": counts[0],
                f"s{sid}_n_monitor": counts[1], f"s{sid}_n_clone": counts[2],
                f"s{sid}_n_drop": counts[3], f"s{sid}_entropy": f"{w.entropy:.4f}",
                f"s{sid}_n_tuple_changes": w.n_tuple_changes,
                f"s{sid}_n_entropy_payload": entropy_source.get(sid, []).count(
                    ENTROPY_PRIMARY_SOURCE),
                f"s{sid}_n_entropy_fallback": entropy_source.get(sid, []).count(
                    ENTROPY_FALLBACK_SOURCE),
            })

        # 5b. Host-pair policy (Decision B)
        host_clone_adds, host_clone_dels = [], []
        host_drop_adds, host_drop_dels = [], []
        for sid, p in pairs.items():
            sc = pair_scores.get(sid, {})
            n_pair_clone_rules = n_pair_drop_rules = 0
            for pkey, vec in p.items():
                src, dst = pkey
                pstate_key = (sid, src, dst)
                score = float(sc[pkey]) if pkey in sc else 0.0
                raw = (score_to_level(score, self.pair_model.thresholds)
                      if self.pair_model.trained else MonitoringLevel.NORMAL)
                old, lvl = self.pair_policy.update(pstate_key, raw)
                st = self.pair_policy.state(pstate_key)
                if lvl > old:
                    st.escalated_by = "IF-pair"
                    if lvl >= MonitoringLevel.CLONE:
                        host_clone_adds.append((sid, pstate_key, src))
                        new_alerts.append({
                            "type": "pair_escalation", "detector": "host_pair",
                            "ts": ts, "window": self._window, "switch": sid,
                            "src": src, "dst": dst, "score": score,
                            "raw_level": raw.name, "level": int(lvl),
                            "level_name": lvl.name, "prev_level": old.name,
                        })
                        logger.info(f"S{sid} pair {src}->{dst}: {old.name} -> {lvl.name} "
                                   f"(score={score:.4f}, by=IF-pair)")
                elif lvl < old:
                    if old == MonitoringLevel.DROP and st.drop_handle is not None:
                        # Left a DPI-forced DROP after K_DOWN windows: lift the
                        # source drop; a pair still at CLONE gets its full
                        # capture back (host-clone rule), as any CLONE pair has
                        host_drop_dels.append((sid, st.drop_handle))
                        st.drop_handle = None
                        if lvl >= MonitoringLevel.CLONE and st.rule_handle is None:
                            host_clone_adds.append((sid, pstate_key, src))
                        logger.info(f"S{sid} pair {src}->{dst}: DROP -> {lvl.name} "
                                    f"(source drop lifted)")
                    if old >= MonitoringLevel.CLONE > lvl:
                        if st.rule_handle is not None:
                            host_clone_dels.append((sid, st.rule_handle))
                            st.rule_handle = None
                        st.escalated_by = None
                if lvl >= MonitoringLevel.CLONE:
                    n_pair_clone_rules += 1
                if lvl == MonitoringLevel.DROP:
                    n_pair_drop_rules += 1
                pair_row_buffer.append((
                    pstate_key,
                    [f"{ts:.6f}", self._window, sid, src, dst]
                    + [f"{v:.6g}" for v in vec]
                    + [int(self.pair_model.trained), f"{score:.6f}", int(raw)]
                ))
            row[f"s{sid}_n_pairs"] = len(p)
            row[f"s{sid}_n_host_clone_rules"] = n_pair_clone_rules
            row[f"s{sid}_n_host_drop_rules"] = n_pair_drop_rules

        # 6. DPI confirmation -> DROP. A confirmation (on a head, sampled or
        # level clone alike) forces the flow straight to DROP (force_drop,
        # bypassing up-hysteresis: the content/pattern match is strong
        # evidence on its own) and installs its 5-tuple drop rule. A
        # confirmed scan class also forces the host pair to DROP and drops
        # every packet from the source (host_drop_table), since each probe is
        # its own short-lived flow. Leaving DROP is the usual K_DOWN dwell
        # (sections 5a/5b), which also removes the rules.
        confirmed = [a for a in self._alert_tailer.read_new()
                    if a.get("type") == "confirmed"]
        pending = self._pending_confirmations + [(a, self._window) for a in confirmed]
        still_pending = []
        for alert, added_window in pending:
            sid = alert["switch"]
            ft = alert["five_tuple"]
            cls = alert["attack_class"]
            tup = (ft["src"], ft["dst"], ft["proto"], ft["sport"], ft["dport"])
            slot = tuple_to_slot.get(sid, {}).get(tup)
            if slot is None:
                expired = (self._window - added_window) >= CONFIRMATION_RETRY_WINDOWS
                if not expired:
                    still_pending.append((alert, added_window))
                else:
                    logger.warning(
                        f"S{sid} DPI confirmed {cls} on inactive "
                        f"flow {tup}: expired unactioned after "
                        f"{CONFIRMATION_RETRY_WINDOWS} windows")
                continue
            key = (sid, slot)
            st = self.policy.state(key)
            if st.level == MonitoringLevel.DROP:
                continue   # already forced (by this or an earlier confirmation); consumed
            escalated_by = self._dpi_trigger(sid, alert)
            old, new_level = self.policy.force_drop(key), MonitoringLevel.DROP
            drop_adds.append((sid, key, tup))
            self._n_forced_drops += 1
            st.escalated_by = escalated_by
            level_writes[sid][slot] = int(new_level)
            new_alerts.append({
                "type": "forced_drop" if new_level == MonitoringLevel.DROP else "forced_clone",
                "detector": "dpi_confirmation", "ts": ts,
                "window": self._window, "switch": sid, "flow_idx": slot,
                "five_tuple": ft, "attack_class": cls,
                "clone_reason": alert.get("clone_reason"), "trigger": escalated_by,
                "prev_level": old.name, "level": int(new_level),
                "level_name": new_level.name,
            })
            logger.info(f"S{sid} flow {tup}: {old.name} -> {new_level.name} by DPI "
                       f"confirmation (class={cls}, trigger={escalated_by})")
            if cls in SCAN_ATTACK_CLASSES:
                pstate_key = (sid, ft["src"], ft["dst"])
                pst = self.pair_policy.state(pstate_key)
                if pst.level < MonitoringLevel.DROP:
                    pold = self.pair_policy.force_drop(pstate_key)
                    pst.escalated_by = escalated_by
                    if pst.rule_handle is not None:   # the drop rule clones too
                        host_clone_dels.append((sid, pst.rule_handle))
                        pst.rule_handle = None
                    host_drop_adds.append((sid, pstate_key, ft["src"]))
                    new_alerts.append({
                        "type": "pair_escalation", "detector": "dpi_confirmation",
                        "ts": ts, "window": self._window, "switch": sid,
                        "src": ft["src"], "dst": ft["dst"], "attack_class": cls,
                        "clone_reason": alert.get("clone_reason"), "trigger": escalated_by,
                        "prev_level": pold.name, "level": int(MonitoringLevel.DROP),
                        "level_name": "DROP",
                    })
                    logger.info(f"S{sid} pair {ft['src']}->{ft['dst']}: {pold.name} -> "
                               f"DROP by DPI confirmation (class={cls}, trigger={escalated_by})")
        self._pending_confirmations = still_pending
        for a in new_alerts:
            self.alerts.write(a)

        # Flush the buffered rows now that section 6 has had its chance to
        # force_clone/force_drop a key -- each row gets that key's *final*
        # level/escalated_by for this window, not the pre-DPI snapshot from
        # when it was buffered (see flow_row_buffer/pair_row_buffer above).
        for key, prefix in flow_row_buffer:
            st = self.policy.state(key)
            self._flow_w.writerow(prefix + [int(st.level), st.escalated_by or ""])
        for pstate_key, prefix in pair_row_buffer:
            pst = self.pair_policy.state(pstate_key)
            self._pair_w.writerow(prefix + [int(pst.level), pst.escalated_by or ""])
        t4 = time.perf_counter()

        # 7. Write back: levels, drop/host-clone rules, aging
        for sid, w in wins.items():
            sw = self.dpm.get_switch(sid)
            for dsid, handle in [d for d in drop_dels if d[0] == sid]:
                sw.delete_drop_rule(handle)
            for asid, key, tup in [d for d in drop_adds if d[0] == sid]:
                handle = sw.add_drop_rule(*tup)
                self.policy.state(key).rule_handle = handle
            for dsid, handle in [d for d in host_clone_dels if d[0] == sid]:
                sw.delete_host_clone_rule(handle)
            for asid, pkey, src in [d for d in host_clone_adds if d[0] == sid]:
                handle = sw.add_host_clone_rule(src)
                self.pair_policy.state(pkey).rule_handle = handle
            for dsid, handle in [d for d in host_drop_dels if d[0] == sid]:
                sw.delete_host_drop_rule(handle)
            for asid, pkey, src in [d for d in host_drop_adds if d[0] == sid]:
                handle = sw.add_host_drop_rule(src)
                self.pair_policy.state(pkey).drop_handle = handle
            expired = self.agers[sid].update(
                {int(s): int(p) for s, p in zip(w.slots, w.X[:, 0])})
            for slot in expired:
                st = self.policy.forget((sid, slot))
                self._slot_tuple.pop((sid, slot), None)
                level_writes[sid].pop(slot, None)
                if st is not None and st.rule_handle is not None:
                    sw.delete_drop_rule(st.rule_handle)
            if expired:
                sw.clear_flow_entries(expired)      # also resets flow_level
                self.extractors[sid].forget(expired)
            if level_writes[sid]:
                sw.write_flow_levels(level_writes[sid])
            row[f"s{sid}_n_aged"] = len(expired)
            row[f"s{sid}_n_drop_rules"] = sum(
                1 for (s, _), st in self.policy.flows.items()
                if s == sid and st.rule_handle is not None)
        t5 = time.perf_counter()

        for sid, c in dp.items():
            row.update({f"s{sid}_fwd": c["forwarded"], f"s{sid}_cloned": c["cloned"],
                        f"s{sid}_dropped": c["dropped"],
                        f"s{sid}_fwd_bytes": c["forwarded_bytes"],
                        f"s{sid}_cloned_bytes": c["cloned_bytes"],
                        f"s{sid}_dropped_bytes": c["dropped_bytes"]})
            row.update({f"s{sid}_{k}": c[k] for k in DP_COUNTER_FIELDS if k.startswith("clone_")})
        row.update({
            "phase": "warmup" if in_warmup else ("scored" if self.model.trained else "untrained"),
            "n_flows_total": sum(len(w.slots) for w in wins.values()),
            "t_read_ms": (t1 - t0) * 1e3, "t_feat_ms": (t2 - t1) * 1e3,
            "t_infer_ms": (t3 - t2) * 1e3, "t_policy_ms": (t4 - t3) * 1e3,
            "t_write_ms": (t5 - t4) * 1e3, "t_total_ms": (t5 - t0) * 1e3,
            "cpu_pct": self._proc.cpu_percent(None),
            "rss_mb": self._proc.memory_info().rss / 2**20,
            "n_forced_drops_total": self._n_forced_drops,
            "n_pending_confirmations": len(self._pending_confirmations),
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
            self._pair_f.flush()

    def _dpi_trigger(self, sid: int, alert: dict) -> str:
        """
        escalated_by for a DPI confirmation on a flow below CLONE: why DPI
        had the packet at all (the confirmed alert's clone_reason). A
        host-pair clone rule (reason 3) is credited to whatever escalated
        the pair that installed it -- IF-pair, or a DPI-head/DPI-sample scan
        confirmation -- since that is what put this source under full
        capture. An alert without clone_reason (older DPI) counts as
        DPI-sample, the only non-level path there was before flow heads.
        """
        reason = alert.get("clone_reason")
        if reason == 3:
            src = alert["five_tuple"]["src"]
            for (psid, psrc, _), pst in self.pair_policy.flows.items():
                if psid == sid and psrc == src and pst.escalated_by \
                        and (pst.rule_handle is not None or pst.drop_handle is not None):
                    return pst.escalated_by
            return "IF-pair"
        return DPI_TRIGGER_BY_REASON.get(reason, "DPI-sample")

    def summary(self) -> Dict:
        return {
            "windows": self._window,
            "head_clone_n": self.head_clone_n,
            "head_clone_n_readback": self.head_clone_n_readback,
            "trained": self.model.trained,
            "pair_trained": self.pair_model.trained,
            "flow_train_info": self.model.train_info,
            "pair_train_info": self.pair_model.train_info,
            "flows_tracked": len(self.policy.flows),
            "pairs_tracked": len(self.pair_policy.flows),
            "forced_drops": self._n_forced_drops,
            "pending_confirmations": len(self._pending_confirmations),
            "alerts_written": self.alerts.n_written,
        }
