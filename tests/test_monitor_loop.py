"""
Unit tests of the full poll loop (controller/monitor.py) against an
in-memory fake switch, covering Decisions A-C:

  A. thresholds (warm-up p5 / min) and DROP only via a DPI-confirmed alert
     on a flow already at CLONE, including the pending/expiry retry path
  B. the host-pair detector escalating to CLONE and installing/removing a
     src-IP host-clone rule
  C. the entropy feature preferring DPI's live payload-sampled snapshot
     over the flow-distribution fallback, logged per row
"""

import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller import monitor as mon
from controller.dataplane import FLOW_REGISTERS, MAX_FLOWS
from controller.features import entropy_snapshot_key
from controller.utils import ip_str_to_int


class FakeSwitch:
    def __init__(self):
        self.r = {k: [0] * MAX_FLOWS for k in FLOW_REGISTERS}
        self.drop_rules = {}
        self.host_clone_rules = {}
        self._h = 0
        self.level_writes = []

    def set_flow(self, slot, tup, pkts, nbytes, first_us, last_us, syn=0):
        src, dst, proto, sp, dp = tup
        self.r["pkt_counter"][slot] = pkts
        self.r["byte_counter"][slot] = nbytes
        self.r["flow_src_ip"][slot] = ip_str_to_int(src)
        self.r["flow_dst_ip"][slot] = ip_str_to_int(dst)
        self.r["flow_protocol"][slot] = proto
        self.r["flow_src_port"][slot] = sp
        self.r["flow_dst_port"][slot] = dp
        self.r["flow_syn_pkts"][slot] = syn
        self.r["flow_first_ts"][slot] = first_us
        self.r["flow_last_ts"][slot] = last_us

    # SwitchInterface API used by the monitor
    def read_registers(self, names):
        return {n: list(self.r[n]) for n in names}

    def read_dp_counters(self):
        return {"forwarded": 0, "cloned": 0, "dropped": 0}

    def reset_dp_counters(self):
        pass

    def write_flow_levels(self, levels):
        for i, v in levels.items():
            self.r["flow_level"][i] = v
        self.level_writes.append(dict(levels))

    def add_drop_rule(self, src, dst, proto, sp, dp):
        self._h += 1
        self.drop_rules[self._h] = (src, dst, proto, sp, dp)
        return self._h

    def delete_drop_rule(self, h):
        return self.drop_rules.pop(h, None) is not None

    def add_host_clone_rule(self, src_ip):
        self._h += 1
        self.host_clone_rules[self._h] = src_ip
        return self._h

    def delete_host_clone_rule(self, h):
        return self.host_clone_rules.pop(h, None) is not None

    def clear_flow_entries(self, idx):
        for i in idx:
            for k in ("pkt_counter", "byte_counter", "flow_level", "flow_syn_pkts"):
                self.r[k][i] = 0


class FakeDPM:
    def __init__(self, sw):
        self.switches = {1: sw}

    def get_switch(self, sid):
        return self.switches[sid]

    def all_switches(self):
        return list(self.switches.values())


@pytest.fixture
def ctl(tmp_path, monkeypatch):
    sw = FakeSwitch()
    monkeypatch.setattr(mon, "DataPlaneManager", lambda cfg: FakeDPM(sw))
    c = mon.AdaptMonController([{"switch_id": 1, "thrift_port": 0}],
                               out_dir=str(tmp_path), train_until=10**12,
                               flow_idle_polls=3)
    return c, sw, tmp_path


T0 = 1_000_000.0
RATE = np.random.default_rng(1).uniform(5, 400, 60)       # pkt/s per flow
SIZE = np.random.default_rng(2).uniform(80, 1400, 60)     # bytes per packet
START = np.random.default_rng(3).uniform(0, 8, 60)        # flow start (s)
# Spread the 60 benign flows over several destinations so the host-pair
# model's warm-up gets more than one sample per window (one per active
# pair, not per flow -- see monitor.py's training-attempt comment)
PAIR_DST = ["10.0.2.1", "10.0.2.2", "10.0.2.3", "10.0.2.4", "10.0.2.5", "10.0.2.6"]


def benign_step(sw, rng, t, slots):
    # 60 benign flows with varied rates, packet sizes, start times and
    # destinations (10 flows per destination, 6 destinations)
    for k, s in enumerate(slots):
        if t < START[k]:
            continue
        prev = sw.r["pkt_counter"][s]
        n = prev + max(1, int(rng.poisson(RATE[k])))
        dst = PAIR_DST[k % len(PAIR_DST)]
        sw.set_flow(s, ("10.0.1.1", dst, 6, 30000 + k, 5001), n,
                    int(n * SIZE[k] * rng.uniform(0.95, 1.05)),
                    int(START[k] * 1e6), int(t * 1e6))


def write_confirmed_alert(out_dir, **kw):
    alert = {"type": "confirmed", "detector": "dpi", "attack_class": "syn_scan",
             **kw}
    with open(os.path.join(out_dir, "alerts.jsonl"), "a") as f:
        f.write(json.dumps(alert) + "\n")


def train(c, sw, rng, t, slots, n_warmup=12):
    for _ in range(n_warmup):
        t += 1
        benign_step(sw, rng, t, slots)
        c._poll(T0 + t)
    c.train_until = 0
    t += 1
    benign_step(sw, rng, t, slots)
    c._poll(T0 + t)
    return t


def test_both_models_train_once_on_warmup_only(ctl):
    c, sw, out = ctl
    rng = np.random.default_rng(0)
    slots = list(range(100, 160))
    t = train(c, sw, rng, 0.0, slots)
    assert c.model.trained and c.pair_model.trained
    info = json.load(open(out / "model.json"))
    assert info["trained"]
    assert info["flow_model"]["n_train"] == c._n_warmup_vectors
    assert info["host_pair_model"]["n_train"] == c._n_warmup_pair_vectors
    assert info["flow_model"]["params"] == {"n_estimators": 100, "max_samples": 256,
                                            "contamination": 0.05, "random_state": 42}
    assert info["host_pair_model"]["params"] == info["flow_model"]["params"]
    with pytest.raises(RuntimeError):
        c.model.fit_once(np.zeros((100, 6)))
    with pytest.raises(RuntimeError):
        c.pair_model.fit_once(np.zeros((100, 4)))


def test_pair_model_failure_does_not_block_flow_model(ctl, monkeypatch):
    c, sw, out = ctl
    rng = np.random.default_rng(0)
    slots = list(range(100, 160))
    # Force the pair model's warm-up buffer to look too small, as if this
    # topology only ever had one active host pair per window
    orig_fit = c.pair_model.fit_once
    monkeypatch.setattr(c.pair_model, "fit_once", lambda X: None if len(X) else orig_fit(X))
    t = train(c, sw, rng, 0.0, slots)
    assert c.model.trained
    assert not c.pair_model.trained
    info = json.load(open(out / "model.json"))
    assert info["flow_model"]["trained"] and not info["host_pair_model"]["trained"]
    assert not info["trained"]       # overall flag still reflects the partial failure
    # Scoring must not crash with an untrained pair model
    sw.set_flow(7, ("10.0.1.9", "10.0.2.9", 6, 1, 2), 1, 60, 0, int((t + 1) * 1e6))
    c._poll(T0 + t + 1)


def test_flow_escalates_to_clone_via_score_not_drop(ctl):
    """Decision A: the score path tops out at CLONE."""
    c, sw, out = ctl
    rng = np.random.default_rng(0)
    slots = list(range(100, 160))
    t = train(c, sw, rng, 0.0, slots)
    th = c.model.thresholds
    c.model.score = lambda X: np.where(X[:, 3] > 10_000, th.clone - 1.0, th.monitor + 1.0)

    atk = ("10.0.1.3", "10.0.2.1", 17, 4444, 9999)
    pk, levels = 0, []
    for w in range(3):
        t += 1
        benign_step(sw, rng, t, slots)
        pk += 50000
        sw.set_flow(7, atk, pk, pk * 60, 50_000_000, 50_000_000 + int((w + 1) * 1e6))
        c._poll(T0 + t)
        levels.append(sw.r["flow_level"][7])
    assert levels == [0, 2, 2]              # escalates to CLONE on window 2, holds
    assert sw.drop_rules == {}              # never DROP from score alone
    alerts = [json.loads(l) for l in open(out / "alerts.jsonl")]
    anomaly = [a for a in alerts if a["type"] == "anomaly"]
    assert len(anomaly) == 1
    assert anomaly[0]["detector"] == "flow" and anomaly[0]["level_name"] == "CLONE"
    assert max(sw.r["flow_level"][s] for s in slots) < 2   # benign flows stay below


def test_dpi_confirmation_forces_drop_only_at_clone(ctl):
    c, sw, out = ctl
    rng = np.random.default_rng(0)
    slots = list(range(100, 160))
    t = train(c, sw, rng, 0.0, slots)
    th = c.model.thresholds
    c.model.score = lambda X: np.where(X[:, 3] > 10_000, th.clone - 1.0, th.monitor + 1.0)
    atk = ("10.0.1.3", "10.0.2.1", 6, 4444, 9999)

    # A confirmation that arrives while the flow is still NORMAL/MONITOR
    # must NOT force DROP
    write_confirmed_alert(out, ts=T0 + t, switch=1, flow_idx=7,
                          five_tuple={"src": atk[0], "dst": atk[1], "proto": atk[2],
                                     "sport": atk[3], "dport": atk[4]},
                          level=2)
    t += 1
    benign_step(sw, rng, t, slots)
    sw.set_flow(7, atk, 50000, 50000 * 60, 50_000_000, 50_000_000 + int(1e6))
    c._poll(T0 + t)
    assert sw.r["flow_level"][7] == 0       # one score window: not CLONE yet
    assert sw.drop_rules == {}
    assert len(c._pending_confirmations) == 1

    # Second anomalous window: escalates to CLONE; the pending confirmation
    # is retried this same poll and now forces DROP
    t += 1
    benign_step(sw, rng, t, slots)
    sw.set_flow(7, atk, 100000, 100000 * 60, 50_000_000, 50_000_000 + int(2e6))
    c._poll(T0 + t)
    assert sw.r["flow_level"][7] == 3
    assert list(sw.drop_rules.values()) == [atk]
    assert c._pending_confirmations == []
    forced = [json.loads(l) for l in open(out / "alerts.jsonl")
             if json.loads(l)["type"] == "forced_drop"]
    assert len(forced) == 1
    assert forced[0]["attack_class"] == "syn_scan"
    assert forced[0]["detector"] == "dpi_confirmation"
    assert c.policy.state((1, 7)).escalated_by == "dpi_confirmed:syn_scan"


def test_confirmation_on_inactive_flow_expires_after_retry_window(ctl):
    c, sw, out = ctl
    rng = np.random.default_rng(0)
    slots = list(range(100, 160))
    t = train(c, sw, rng, 0.0, slots)
    write_confirmed_alert(out, ts=T0 + t, switch=1, flow_idx=999,
                          five_tuple={"src": "1.2.3.4", "dst": "5.6.7.8", "proto": 6,
                                     "sport": 1, "dport": 2}, level=2)
    for _ in range(mon.CONFIRMATION_RETRY_WINDOWS + 1):
        t += 1
        benign_step(sw, rng, t, slots)
        c._poll(T0 + t)
    assert c._pending_confirmations == []   # expired, not stuck forever
    assert sw.drop_rules == {}


def test_host_pair_escalation_installs_and_removes_clone_rule(ctl):
    """Decision B."""
    c, sw, out = ctl
    rng = np.random.default_rng(0)
    slots = list(range(100, 160))
    t = train(c, sw, rng, 0.0, slots)
    th = c.pair_model.thresholds
    # Force the pair scorer deterministic: ratio (feature idx 2) > 0.5 -> CLONE
    c.pair_model.score = lambda X: np.where(X[:, 2] > 0.5, th.clone - 1.0, th.monitor + 1.0)

    scan_src = "10.0.1.7"
    levels = []
    for w in range(3):
        t += 1
        benign_step(sw, rng, t, slots)
        for i, (slot, dport) in enumerate(zip(range(10, 15), (21, 22, 23, 80, 443))):
            sw.set_flow(slot, (scan_src, "10.0.2.9", 6, 50000 + i, dport),
                       w + 1, (w + 1) * 60, 0, int((w + 1) * 1e6), syn=w + 1)
        c._poll(T0 + t)
        levels.append(c.pair_policy.state((1, scan_src, "10.0.2.9")).level)
    assert [int(l) for l in levels] == [0, 2, 2]
    assert list(sw.host_clone_rules.values()) == [scan_src]
    alerts = [json.loads(l) for l in open(out / "alerts.jsonl")]
    pe = [a for a in alerts if a["type"] == "pair_escalation"]
    assert len(pe) == 1 and pe[0]["detector"] == "host_pair" and pe[0]["src"] == scan_src

    # Stop scanning: the pair's score goes back to normal (syn ratio drops
    # once non-SYN packets dominate) -> de-escalate after K_DOWN windows,
    # rule removed
    for w in range(4):
        t += 1
        benign_step(sw, rng, t, slots)
        for i, (slot, dport) in enumerate(zip(range(10, 15), (21, 22, 23, 80, 443))):
            n = 5 + w * 50
            sw.set_flow(slot, (scan_src, "10.0.2.9", 6, 50000 + i, dport),
                       n, n * 800, 0, int((5 + w) * 1e6), syn=5)   # syn count frozen
        c._poll(T0 + t)
    assert sw.host_clone_rules == {}


def test_entropy_prefers_dpi_payload_snapshot_over_fallback(ctl):
    """Decision C."""
    c, sw, out = ctl
    rng = np.random.default_rng(0)
    slots = list(range(100, 160))
    t = train(c, sw, rng, 0.0, slots)

    tup = ("10.0.1.1", PAIR_DST[0], 6, 30000, 5001)
    key = entropy_snapshot_key(1, tup)
    with open(os.path.join(out, "dpi_entropy_live.json"), "w") as f:
        json.dump({key: {"entropy": 6.75, "n": 512, "clones": 3}}, f)

    t += 1
    benign_step(sw, rng, t, slots)
    c._poll(T0 + t)

    rows = list(csv.reader(open(out / "flow_windows.csv")))
    header = rows[0]
    ent_idx = header.index("entropy")
    src_idx = header.index("entropy_source")
    sport_idx = header.index("sport")
    window_idx = header.index("window")
    last_window = max(int(r[window_idx]) for r in rows[1:])
    this_window = [r for r in rows[1:] if int(r[window_idx]) == last_window]
    row = next(r for r in this_window if r[sport_idx] == "30000")
    assert float(row[ent_idx]) == pytest.approx(6.75)
    assert row[src_idx] == "payload_sampled"
    # A flow DPI has no entry for still uses the flow-distribution fallback
    other = next(r for r in this_window if r[sport_idx] == "30001")
    assert other[src_idx] == "flow_distribution"


def test_aging_clears_slot_and_policy_state(ctl):
    c, sw, out = ctl
    rng = np.random.default_rng(0)
    slots = list(range(100, 160))
    t = train(c, sw, rng, 0.0, slots)
    sw.set_flow(7, ("10.0.1.3", "10.0.2.1", 17, 4444, 9999), 10, 600,
               50_000_000, 50_000_000 + int(1e6))
    t += 1
    benign_step(sw, rng, t, slots)
    c._poll(T0 + t)
    assert (1, 7) in c.policy.flows
    for _ in range(4):
        t += 1
        benign_step(sw, rng, t, slots)
        c._poll(T0 + t)
    assert sw.r["pkt_counter"][7] == 0 and sw.r["flow_level"][7] == 0
    assert (1, 7) not in c.policy.flows
    c.stop()
    rows = open(out / "flow_windows.csv").read().splitlines()
    assert rows[0].startswith(
        "ts,window,switch,slot,src,dst,proto,sport,dport,pkt_count")
    assert "entropy_source" in rows[0]
