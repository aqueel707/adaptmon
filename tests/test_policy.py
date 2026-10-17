"""Unit tests: per-flow hysteresis and score -> level mapping."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.model import Thresholds
from controller.policy import FlowPolicy, MonitoringLevel as L, score_to_level

TH = Thresholds(normal=0.0, monitor=-0.1, clone=-0.2,
                warmup_min=0.0, warmup_std=0.1, n_warmup=100)


def test_score_to_level_boundaries():
    assert score_to_level(0.01, TH) == L.NORMAL
    assert score_to_level(0.0, TH) == L.MONITOR      # not above warm-up min
    assert score_to_level(-0.1, TH) == L.CLONE
    assert score_to_level(-0.2, TH) == L.DROP
    assert score_to_level(-5, TH) == L.DROP


def run(seq):
    p = FlowPolicy()
    return [p.update("f", r)[1] for r in seq]


def test_escalate_needs_two_consecutive_windows():
    assert run([L.CLONE]) == [L.NORMAL]
    assert run([L.CLONE, L.CLONE]) == [L.NORMAL, L.CLONE]
    # interrupted by a NORMAL window: no escalation
    assert run([L.CLONE, L.NORMAL, L.CLONE]) == [L.NORMAL] * 3


def test_escalate_to_lowest_of_the_k_windows():
    assert run([L.DROP, L.CLONE]) == [L.NORMAL, L.CLONE]
    assert run([L.DROP, L.DROP]) == [L.NORMAL, L.DROP]
    assert run([L.MONITOR, L.DROP]) == [L.NORMAL, L.MONITOR]


def test_deescalate_needs_three_consecutive_windows():
    seq = [L.DROP, L.DROP, L.NORMAL, L.NORMAL, L.NORMAL]
    assert run(seq) == [L.NORMAL, L.DROP, L.DROP, L.DROP, L.NORMAL]
    seq = [L.DROP, L.DROP, L.NORMAL, L.NORMAL, L.DROP, L.NORMAL, L.NORMAL]
    assert run(seq) == [L.NORMAL, L.DROP, L.DROP, L.DROP, L.DROP, L.DROP, L.DROP]


def test_deescalate_to_highest_of_the_k_windows():
    seq = [L.DROP, L.DROP, L.NORMAL, L.MONITOR, L.NORMAL]
    assert run(seq)[-1] == L.MONITOR


def test_max_level_tracked():
    p = FlowPolicy()
    for r in [L.DROP, L.DROP, L.NORMAL, L.NORMAL, L.NORMAL]:
        p.update("f", r)
    assert p.state("f").level == L.NORMAL
    assert p.state("f").max_level == L.DROP
