"""
Unit tests for controller/features.py: SYN-packet tracking, the "new flow"
flag, host-pair aggregation (Decision B), and the DPI entropy-snapshot
reader (Decision C).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.dataplane import FLOW_REGISTERS, MAX_FLOWS
from controller.features import (
    FlowFeatureExtractor, aggregate_host_pairs, EntropySnapshotReader,
    entropy_snapshot_key, ENTROPY_FALLBACK_SOURCE, ENTROPY_PRIMARY_SOURCE,
)
from controller.utils import ip_str_to_int


def _empty_regs():
    return {k: [0] * MAX_FLOWS for k in FLOW_REGISTERS}


def _set_flow(r, slot, src, dst, proto, sport, dport, pkts, nbytes, syn, first, last):
    r["pkt_counter"][slot] = pkts
    r["byte_counter"][slot] = nbytes
    r["flow_src_ip"][slot] = ip_str_to_int(src)
    r["flow_dst_ip"][slot] = ip_str_to_int(dst)
    r["flow_protocol"][slot] = proto
    r["flow_src_port"][slot] = sport
    r["flow_dst_port"][slot] = dport
    r["flow_syn_pkts"][slot] = syn
    r["flow_first_ts"][slot] = first
    r["flow_last_ts"][slot] = last


def test_syn_delta_tracked_across_polls():
    fx = FlowFeatureExtractor(1)
    r = _empty_regs()
    _set_flow(r, 5, "10.0.1.1", "10.0.2.1", 6, 1, 80, 1, 60, 1, 0, 1_000_000)
    assert fx.extract(r, 1.0) is None     # first poll: no previous snapshot
    _set_flow(r, 5, "10.0.1.1", "10.0.2.1", 6, 1, 80, 4, 240, 4, 0, 4_000_000)
    w = fx.extract(r, 2.0)
    assert w.pkt_delta[0] == 3 and w.syn_delta[0] == 3


def test_is_new_flag_for_fresh_and_collided_slots():
    fx = FlowFeatureExtractor(1)
    r = _empty_regs()
    _set_flow(r, 5, "10.0.1.1", "10.0.2.1", 6, 1, 80, 2, 120, 0, 0, 1e6)
    fx.extract(r, 1.0)
    # Slot 9 appears for the first time this poll -> is_new
    _set_flow(r, 5, "10.0.1.1", "10.0.2.1", 6, 1, 80, 3, 180, 0, 0, 2e6)
    _set_flow(r, 9, "10.0.1.2", "10.0.2.2", 17, 2, 53, 1, 50, 0, 0, 2e6)
    w = fx.extract(r, 2.0)
    by_slot = dict(zip(w.slots.tolist(), w.is_new.tolist()))
    assert by_slot[5] is False or not by_slot[5]
    assert by_slot[9]
    # Slot 5 now reused by a different 5-tuple while the counter kept
    # growing (collision): also counts as "new"
    _set_flow(r, 5, "10.0.9.9", "10.0.9.8", 6, 3, 443, 4, 240, 0, 0, 3e6)
    w2 = fx.extract(r, 3.0)
    assert dict(zip(w2.slots.tolist(), w2.is_new.tolist()))[5]
    assert w2.n_tuple_changes == 1


def test_aggregate_host_pairs_counts_and_ratio():
    fx = FlowFeatureExtractor(1)
    r = _empty_regs()
    # Two flows h1->h3 (one TCP SYN-heavy, one TCP data) + one unrelated pair
    _set_flow(r, 1, "10.0.1.1", "10.0.2.1", 6, 100, 21, 1, 60, 1, 0, 1e6)
    _set_flow(r, 2, "10.0.1.1", "10.0.2.1", 6, 101, 22, 1, 60, 1, 0, 1e6)
    _set_flow(r, 3, "10.0.1.1", "10.0.2.1", 6, 102, 23, 1, 60, 1, 0, 1e6)
    _set_flow(r, 4, "10.0.1.1", "10.0.2.1", 6, 103, 24, 1, 60, 1, 0, 1e6)
    _set_flow(r, 5, "10.0.1.1", "10.0.2.1", 6, 104, 25, 1, 60, 1, 0, 1e6)
    _set_flow(r, 6, "9.9.9.9", "8.8.8.8", 17, 1, 2, 10, 1000, 0, 0, 1e6)
    fx.extract(r, 1.0)
    for slot, dport, syn in ((1, 21, 2), (2, 22, 2), (3, 23, 2), (4, 24, 2), (5, 25, 2)):
        _set_flow(r, slot, "10.0.1.1", "10.0.2.1", 6, 100 + slot - 1, dport, 2, 120, syn, 0, 2e6)
    _set_flow(r, 6, "9.9.9.9", "8.8.8.8", 17, 1, 2, 20, 2000, 0, 0, 2e6)
    w = fx.extract(r, 2.0)
    pairs = aggregate_host_pairs(w)
    scan = pairs[("10.0.1.1", "10.0.2.1")]
    # [new_flows, distinct_dst_ports, syn_only_ratio, pkt_count]
    assert scan[0] == 0          # none newly-occupied this window (all seen before)
    assert scan[1] == 5          # 5 distinct dst ports
    assert scan[2] == 1.0        # every TCP packet this window was SYN-only
    assert scan[3] == 5          # 1 packet/flow x 5 flows this window
    other = pairs[("9.9.9.9", "8.8.8.8")]
    assert other[1] == 1 and other[2] == 0.0 and other[3] == 10   # UDP: ratio 0


def test_entropy_snapshot_reader_refresh_and_lookup(tmp_path):
    path = tmp_path / "dpi_entropy_live.json"
    tup = ("10.0.1.1", "10.0.2.1", 6, 1, 80)
    key = entropy_snapshot_key(1, tup)
    path.write_text(json.dumps({key: {"entropy": 3.5, "n": 42, "clones": 7}}))
    reader = EntropySnapshotReader(str(path))
    assert reader.lookup(1, tup) is None     # not read until refresh()
    reader.refresh()
    snap = reader.lookup(1, tup)
    assert snap == {"entropy": 3.5, "n": 42, "clones": 7}
    assert reader.lookup(2, tup) is None     # different switch, no match


def test_entropy_snapshot_reader_missing_file_is_empty(tmp_path):
    reader = EntropySnapshotReader(str(tmp_path / "nope.json"))
    reader.refresh()
    assert reader.lookup(1, ("a", "b", 6, 1, 2)) is None


def test_entropy_snapshot_reader_skips_reread_when_mtime_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"x": {"entropy": 1.0, "n": 1, "clones": 1}}))
    reader = EntropySnapshotReader(str(path))
    reader.refresh()
    calls = []
    orig_open = open

    def spy_open(p, *a, **kw):
        calls.append(p)
        return orig_open(p, *a, **kw)

    monkeypatch.setattr("builtins.open", spy_open)
    reader.refresh()   # mtime unchanged -> must not reopen the file
    assert str(path) not in calls


def test_primary_and_fallback_entropy_source_constants_distinct():
    assert ENTROPY_PRIMARY_SOURCE != ENTROPY_FALLBACK_SOURCE
