"""
Unit tests: per-key hysteresis, score -> level mapping (Decision A), and
force_drop() (the DPI-confirmation path, Decision A).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.model import Thresholds
from controller.policy import FlowPolicy, MonitoringLevel as L, score_to_level

TH = Thresholds(monitor=0.0, clone=-0.2, warmup_min=-0.3, warmup_p1=-0.2,
                warmup_p5=0.0, warmup_p10=0.05, warmup_mean=0.1, warmup_std=0.1,
                n_warmup=100)


def test_score_to_level_boundaries():
    assert score_to_level(0.01, TH) == L.NORMAL
    assert score_to_level(0.0, TH) == L.MONITOR     # not above p5
    assert score_to_level(-0.1, TH) == L.MONITOR
    assert score_to_level(-0.2, TH) == L.CLONE       # not above p1
    assert score_to_level(-5, TH) == L.CLONE


def test_score_to_level_never_returns_drop():
    """Decision A: DROP is only reached via force_drop(), never from a score."""
    for s in (0.5, 0.0, -0.2, -1.0, -1000.0):
        assert score_to_level(s, TH) != L.DROP


def run(seq):
    p = FlowPolicy()
    return [p.update("f", r)[1] for r in seq]


def test_escalate_needs_two_consecutive_windows():
    assert run([L.CLONE]) == [L.NORMAL]
    assert run([L.CLONE, L.CLONE]) == [L.NORMAL, L.CLONE]
    # interrupted by a NORMAL window: no escalation
    assert run([L.CLONE, L.NORMAL, L.CLONE]) == [L.NORMAL] * 3


def test_escalate_to_lowest_of_the_k_windows():
    assert run([L.CLONE, L.MONITOR]) == [L.NORMAL, L.MONITOR]
    assert run([L.CLONE, L.CLONE]) == [L.NORMAL, L.CLONE]


def test_deescalate_needs_three_consecutive_windows():
    seq = [L.CLONE, L.CLONE, L.NORMAL, L.NORMAL, L.NORMAL]
    assert run(seq) == [L.NORMAL, L.CLONE, L.CLONE, L.CLONE, L.NORMAL]
    seq = [L.CLONE, L.CLONE, L.NORMAL, L.NORMAL, L.CLONE, L.NORMAL, L.NORMAL]
    assert run(seq) == [L.NORMAL, L.CLONE, L.CLONE, L.CLONE, L.CLONE, L.CLONE, L.CLONE]


def test_deescalate_to_highest_of_the_k_windows():
    seq = [L.CLONE, L.CLONE, L.NORMAL, L.MONITOR, L.NORMAL]
    assert run(seq)[-1] == L.MONITOR


def test_max_level_tracked():
    p = FlowPolicy()
    for r in [L.CLONE, L.CLONE, L.NORMAL, L.NORMAL, L.NORMAL]:
        p.update("f", r)
    assert p.state("f").level == L.NORMAL
    assert p.state("f").max_level == L.CLONE


def test_force_drop_sets_drop_and_clears_history():
    p = FlowPolicy()
    p.update("f", L.CLONE)
    p.update("f", L.CLONE)
    assert p.state("f").level == L.CLONE
    old = p.force_drop("f")
    assert old == L.CLONE
    assert p.state("f").level == L.DROP
    assert p.state("f").max_level == L.DROP
    assert len(p.state("f").recent) == 0


def test_leaving_forced_drop_needs_fresh_k_down_windows():
    p = FlowPolicy()
    p.update("f", L.CLONE)
    p.update("f", L.CLONE)          # two CLONE windows already in history
    p.force_drop("f")               # history cleared on forcing
    # A single post-force NORMAL window must NOT be enough to de-escalate,
    # even though 2 old CLONE + 1 new NORMAL would satisfy a naive "3 below
    # current" check if the old history had not been cleared.
    _, lvl = p.update("f", L.NORMAL)
    assert lvl == L.DROP
    _, lvl = p.update("f", L.NORMAL)
    assert lvl == L.DROP
    _, lvl = p.update("f", L.NORMAL)
    assert lvl == L.NORMAL          # 3 fresh sub-DROP windows after forcing


def test_force_drop_is_independent_per_key():
    p = FlowPolicy()
    p.update("a", L.CLONE)
    p.update("a", L.CLONE)
    p.force_drop("a")
    assert p.state("a").level == L.DROP
    assert p.state("b").level == L.NORMAL   # untouched, freshly created
