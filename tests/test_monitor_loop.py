"""
Unit test of the full poll loop (controller/monitor.py) against an
in-memory fake switch: warm-up buffering, one-time training, per-flow
escalation with hysteresis, flow_level write-back, drop-rule install and
removal on de-escalation, anomaly alerts, aging.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller import monitor as mon
from controller.dataplane import FLOW_REGISTERS, MAX_FLOWS
from controller.utils import ip_str_to_int


class FakeSwitch:
    def __init__(self):
        self.r = {k: [0] * MAX_FLOWS for k in FLOW_REGISTERS}
        self.drop_rules = {}
        self._h = 0
        self.level_writes = []

    def set_flow(self, slot, tup, pkts, nbytes, first_us, last_us):
        src, dst, proto, sp, dp = tup
        self.r["pkt_counter"][slot] = pkts
        self.r["byte_counter"][slot] = nbytes
        self.r["flow_src_ip"][slot] = ip_str_to_int(src)
        self.r["flow_dst_ip"][slot] = ip_str_to_int(dst)
        self.r["flow_protocol"][slot] = proto
        self.r["flow_src_port"][slot] = sp
        self.r["flow_dst_port"][slot] = dp
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

    def clear_flow_entries(self, idx):
        for i in idx:
            for k in ("pkt_counter", "byte_counter", "flow_level"):
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


def benign_step(sw, rng, t, slots):
    # 60 benign flows with varied rates, packet sizes and start times
    for k, s in enumerate(slots):
        if t < START[k]:
            continue
        prev = sw.r["pkt_counter"][s]
        n = prev + max(1, int(rng.poisson(RATE[k])))
        sw.set_flow(s, ("10.0.1.1", "10.0.2.1", 6, 30000 + k, 5001), n,
                    int(n * SIZE[k] * rng.uniform(0.95, 1.05)),
                    int(START[k] * 1e6), int(t * 1e6))


def test_poll_loop_escalation_writeback_and_removal(ctl):
    c, sw, out = ctl
    rng = np.random.default_rng(0)
    slots = list(range(100, 160))
    t = 0.0
    # Warm-up: 12 windows of benign flows
    for _ in range(12):
        t += 1
        benign_step(sw, rng, t, slots)
        c._poll(T0 + t)
    assert not c.model.trained
    # End warm-up: next poll trains once
    c.train_until = 0
    t += 1
    benign_step(sw, rng, t, slots)
    c._poll(T0 + t)
    assert c.model.trained
    info = json.load(open(out / "model.json"))
    # first poll has no previous snapshot to diff against
    assert info["trained"] and info["n_train"] == c._n_warmup_vectors
    assert info["params"] == {"n_estimators": 100, "max_samples": 256,
                              "contamination": 0.05, "random_state": 42}
    with pytest.raises(RuntimeError):
        c.model.fit_once(np.zeros((100, 6)))

    # The plumbing below needs deterministic scores, so the frozen model's
    # scorer is replaced: rate > 10k pkt/s -> far below the DROP threshold,
    # else well inside NORMAL. (How the real model scores real attack flows
    # is measured on the bmv2 runs, not here.)
    th = c.model.thresholds
    c.model.score = lambda X: np.where(X[:, 3] > 10_000, th.clone - 1.0, th.normal + 1.0)

    # Attack flow: one tiny-packet flow at a huge rate, slot 7
    atk = ("10.0.1.3", "10.0.2.1", 17, 4444, 9999)
    pk = 0
    levels = []
    for w in range(3):
        t += 1
        benign_step(sw, rng, t, slots)
        pk += 50000
        sw.set_flow(7, atk, pk, pk * 60, 50_000_000, 50_000_000 + int((w + 1) * 1e6))
        c._poll(T0 + t)
        levels.append(sw.r["flow_level"][7])
    # Escalation only on the second anomalous window, then held
    assert levels[0] == 0 and levels[1] == 3 and levels[2] == 3
    assert list(sw.drop_rules.values()) == [atk]
    alerts = [json.loads(l) for l in open(out / "alerts.jsonl")]
    assert len(alerts) == 1 and alerts[0]["type"] == "anomaly"
    assert alerts[0]["flow_idx"] == 7 and alerts[0]["level_name"] == "DROP"
    assert alerts[0]["five_tuple"]["sport"] == 4444
    # Benign flows stay below CLONE
    assert max(sw.r["flow_level"][s] for s in slots) < 2

    # Attack flow starts behaving like the benign flows -> 3 windows to drop
    levels = []
    for w in range(4):
        t += 1
        benign_step(sw, rng, t, slots)
        pk += 100
        sw.set_flow(7, atk, pk, pk * 800, 50_000_000, 50_000_000 + int((w + 4) * 1e6))
        c._poll(T0 + t)
        levels.append(sw.r["flow_level"][7])
    assert levels[0] == 3 and levels[1] == 3, levels
    assert levels[2] < 3, levels           # third window below DROP
    assert sw.drop_rules == {}             # drop rule removed on de-escalation

    # Aging: flow stops; after 3 idle polls the slot is cleared
    for w in range(4):
        t += 1
        benign_step(sw, rng, t, slots)
        c._poll(T0 + t)
    assert sw.r["pkt_counter"][7] == 0 and sw.r["flow_level"][7] == 0
    assert (1, 7) not in c.policy.flows
    c.stop()
    rows = open(out / "flow_windows.csv").read().splitlines()
    assert rows[0].startswith("ts,window,switch,slot,src,dst,proto,sport,dport,pkt_count")
