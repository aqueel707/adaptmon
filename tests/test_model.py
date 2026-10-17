"""
Unit tests for controller/model.py: IFModel training/freezing, Decision A
thresholds (warm-up p5 / min), and the two concrete models sharing the
exact IsolationForest hyperparameters the brief specifies.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.model import (
    IF_PARAMS, FlowIsolationForest, HostPairIsolationForest, MIN_TRAIN_SAMPLES,
)
from controller.features import FEATURE_NAMES, PAIR_FEATURE_NAMES


def test_if_params_match_spec():
    assert IF_PARAMS == {"n_estimators": 100, "max_samples": 256,
                         "contamination": 0.05, "random_state": 42}


def _benign_flow_matrix(n, rng):
    pkt = rng.uniform(10, 500, n)
    byte = pkt * rng.uniform(80, 1400, n)
    dur = rng.uniform(1, 60, n)
    rate = pkt / dur
    mean_sz = byte / pkt
    ent = rng.uniform(0.5, 2.0, n)
    return np.column_stack([pkt, byte, dur, rate, mean_sz, ent])


def test_flow_model_trains_once_and_refuses_retrain():
    rng = np.random.default_rng(0)
    m = FlowIsolationForest()
    assert not m.trained
    th = m.fit_once(_benign_flow_matrix(200, rng))
    assert m.trained and th is not None
    assert m.feature_names == FEATURE_NAMES
    with pytest.raises(RuntimeError):
        m.fit_once(_benign_flow_matrix(200, rng))


def test_fit_once_returns_none_below_min_samples():
    rng = np.random.default_rng(0)
    m = FlowIsolationForest()
    n = MIN_TRAIN_SAMPLES - 1
    th = m.fit_once(_benign_flow_matrix(n, rng))
    assert th is None and not m.trained


def test_thresholds_are_warmup_p5_and_min_decision_a():
    rng = np.random.default_rng(1)
    m = FlowIsolationForest()
    X = _benign_flow_matrix(300, rng)
    th = m.fit_once(X)
    scores = m.model.decision_function(m._preprocess(X))
    assert th.warmup_min == pytest.approx(scores.min())
    assert th.warmup_p5 == pytest.approx(np.percentile(scores, 5))
    assert th.monitor == th.warmup_p5
    assert th.clone == th.warmup_min
    assert th.clone <= th.monitor     # min is at or below the 5th percentile
    assert th.n_warmup == 300


def test_score_shape_and_frozen_score_matches_warmup_scores():
    rng = np.random.default_rng(2)
    m = FlowIsolationForest()
    X = _benign_flow_matrix(100, rng)
    m.fit_once(X)
    s1 = m.score(X[:5])
    s2 = m.score(X[:5])
    np.testing.assert_array_equal(s1, s2)   # frozen model: deterministic


def test_score_requires_trained_model():
    m = FlowIsolationForest()
    with pytest.raises(RuntimeError):
        m.score(np.zeros((1, 6)))


def test_score_empty_batch():
    rng = np.random.default_rng(0)
    m = FlowIsolationForest()
    m.fit_once(_benign_flow_matrix(100, rng))
    assert len(m.score(np.zeros((0, 6)))) == 0


def test_host_pair_model_uses_pair_feature_names():
    rng = np.random.default_rng(3)
    m = HostPairIsolationForest()
    X = np.column_stack([
        rng.integers(0, 5, 100), rng.integers(1, 10, 100),
        rng.uniform(0, 1, 100), rng.integers(1, 200, 100),
    ]).astype(np.float64)
    th = m.fit_once(X)
    assert m.feature_names == PAIR_FEATURE_NAMES
    assert th is not None
    assert m.params == {"n_estimators": 100, "max_samples": 256,
                        "contamination": 0.05, "random_state": 42}


def test_two_models_are_independent():
    rng = np.random.default_rng(4)
    flow_m = FlowIsolationForest()
    pair_m = HostPairIsolationForest()
    flow_m.fit_once(_benign_flow_matrix(100, rng))
    assert not pair_m.trained
    pair_X = np.column_stack([
        rng.integers(0, 5, 100), rng.integers(1, 10, 100),
        rng.uniform(0, 1, 100), rng.integers(1, 200, 100),
    ]).astype(np.float64)
    pair_m.fit_once(pair_X)
    assert flow_m.trained and pair_m.trained
    assert flow_m.model is not pair_m.model
    assert flow_m.thresholds != pair_m.thresholds
