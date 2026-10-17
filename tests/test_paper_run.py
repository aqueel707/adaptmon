"""Unit tests: experiments/paper_run.py's seeded schedule and seed registry."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import paper_run as pr  # noqa: E402


def test_schedule_is_a_pure_function_of_the_seed():
    assert pr.build_schedule(123, 8) == pr.build_schedule(123, 8)
    assert pr.build_schedule(123, 8) != pr.build_schedule(124, 8)


def test_seeded_cycle_shape():
    for c in pr.build_schedule(99, 20)["cycles"]:
        assert len(set(c["scan_ports"])) == pr.N_SCAN_PORTS
        assert all(1 <= p <= 1023 for p in c["scan_ports"])
        assert sorted(c["order"]) == sorted(pr.ATTACK_PHASES)
        assert 6.0 <= c["benign_s"] <= 10.0 and 1.0 <= c["iperf_mbps"] <= 3.0
        assert sorted(c["sqli_order"]) == list(range(len(pr.SQLI_BODIES)))


def test_unseeded_schedule_is_the_fixed_one_runs_a_and_b_used():
    s = pr.build_schedule(None, 3)
    assert s["seed"] is None
    for c in s["cycles"]:
        assert c["scan_ports"] == [20, 120, 220, 320, 420, 520, 620, 720]
        assert c["order"] == ["syn", "null", "sqli", "spike"]
        assert (c["benign_s"], c["iperf_mbps"], c["http_seed"]) == (8.0, 2.0, None)


def test_registry_records_seeds_and_refuses_reuse(tmp_path):
    reg = str(tmp_path / "seeds.json")
    pr.register_seeds([11, 12], "dev", str(tmp_path / "series_a"), path=reg)
    assert set(pr.load_seed_registry(reg)["seeds"]) == {"11", "12"}
    with pytest.raises(SystemExit, match="already used"):
        pr.register_seeds([13, 12], "final", str(tmp_path / "series_b"), path=reg)
    # nothing from the refused call was recorded
    assert set(pr.load_seed_registry(reg)["seeds"]) == {"11", "12"}
    with pytest.raises(SystemExit, match="Duplicate"):
        pr.register_seeds([20, 20], "final", str(tmp_path / "series_c"), path=reg)
