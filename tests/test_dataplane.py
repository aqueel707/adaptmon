"""Unit tests: controller/dataplane.py's counter/register parsing, against a
canned simple_switch_CLI transcript (tests/smoke_headclone.py checks the
same calls against a live bmv2)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.dataplane import SwitchInterface  # noqa: E402

CLI_OUT = """\
RuntimeCmd: this is the counter array AdaptMonEgress.fwd_counter
AdaptMonEgress.fwd_counter[0]=  (12800 bytes, 100 packets)
AdaptMonEgress.clone_counter[0]=  (3840 bytes, 30 packets)
AdaptMonEgress.drop_counter[0]=  (0 bytes, 0 packets)
AdaptMonEgress.clone_reason_counter[1]=  (1280 bytes, 10 packets)
AdaptMonEgress.clone_reason_counter[2]=  (256 bytes, 2 packets)
AdaptMonEgress.clone_reason_counter[3]=  (0 bytes, 0 packets)
AdaptMonEgress.clone_reason_counter[4]=  (2304 bytes, 18 packets)
"""


def test_read_dp_counters_full_parses_packets_bytes_and_reasons(monkeypatch):
    sw = SwitchInterface(1, 9090)
    monkeypatch.setattr(sw, "_cli", lambda cmds, timeout=15: CLI_OUT)
    c = sw.read_dp_counters_full()
    assert (c["forwarded"], c["forwarded_bytes"]) == (100, 12800)
    assert (c["cloned"], c["cloned_bytes"]) == (30, 3840)
    assert (c["dropped"], c["dropped_bytes"]) == (0, 0)
    assert (c["clone_level"], c["clone_sample"], c["clone_hostpair"], c["clone_head"]) \
        == (10, 2, 0, 18)
    assert c["clone_head_bytes"] == 2304
    # per-reason clones add up to the total
    assert sum(c[f"clone_{r}"] for r in ("level", "sample", "hostpair", "head")) == c["cloned"]


def test_read_dp_counters_keeps_its_packets_only_shape(monkeypatch):
    """smoke_dataplane.py compares this dict for exact equality."""
    sw = SwitchInterface(1, 9090)
    monkeypatch.setattr(sw, "_cli", lambda cmds, timeout=15: CLI_OUT)
    assert sw.read_dp_counters() == {"forwarded": 100, "cloned": 30, "dropped": 0}


def test_unreadable_counter_is_minus_one(monkeypatch):
    sw = SwitchInterface(1, 9090)
    monkeypatch.setattr(sw, "_cli", lambda cmds, timeout=15: "Could not connect")
    c = sw.read_dp_counters_full()
    assert c["forwarded"] == -1 and c["clone_head_bytes"] == -1


def test_set_head_clone_n_writes_then_reads_back(monkeypatch):
    sw = SwitchInterface(1, 9090)
    sent = []

    def fake_cli(cmds, timeout=15):
        sent.append(cmds)
        return "head_clone_n[0]= 3\n" if cmds.startswith("register_read") else ""
    monkeypatch.setattr(sw, "_cli", fake_cli)
    assert sw.set_head_clone_n(3) == 3
    assert sent[0] == "register_write head_clone_n 0 3\n"
    assert sent[1] == "register_read head_clone_n 0\n"
