"""
controller/model.py
Per-flow anomaly detection with one Isolation Forest.

The model is fitted exactly once, on the per-flow feature vectors of the
benign-only warm-up windows (every occupied slot of every switch in every
warm-up poll), and is never refitted afterwards. Scores are
IsolationForest.decision_function(): lower = more anomalous.

Preprocessing: log1p on the heavy-tailed count/rate/duration columns, so a
split is not spent separating a few multi-megabyte bulk flows from
everything else. No scaling: Isolation Forest splits are invariant to
per-feature affine transforms.
"""

import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

from .features import FEATURE_NAMES, N_FEATURES

logger = logging.getLogger(__name__)

IF_PARAMS = dict(
    n_estimators  = 100,
    max_samples   = 256,
    contamination = 0.05,
    random_state  = 42,
)
LOG1P_FEATURES = [FEATURE_NAMES.index(n) for n in
                  ("pkt_count", "byte_count", "duration_s", "pkt_rate")]
MIN_TRAIN_SAMPLES = 50


def preprocess(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(np.array(X, dtype=np.float64, copy=True),
                      nan=0.0, posinf=0.0, neginf=0.0)
    X[:, LOG1P_FEATURES] = np.log1p(np.clip(X[:, LOG1P_FEATURES], 0.0, None))
    return X


@dataclass
class Thresholds:
    """
    Score cut-offs from the frozen model's scores on its own warm-up
    samples: score > normal => NORMAL, > monitor => MONITOR,
    > clone => CLONE, else DROP.
    """
    normal    : float     # warm-up min
    monitor   : float     # min - 1 sigma
    clone     : float     # min - 2 sigma
    warmup_min: float
    warmup_std: float
    n_warmup  : int


class FlowIsolationForest:

    def __init__(self, **params):
        self.params = {**IF_PARAMS, **params}
        self.model: Optional[IsolationForest] = None
        self.thresholds: Optional[Thresholds] = None
        self.train_info: Dict = {}

    @property
    def trained(self) -> bool:
        return self.model is not None

    def fit_once(self, X_warmup: np.ndarray) -> Optional[Thresholds]:
        """Fit on the warm-up vectors and derive the thresholds. Refuses a
        second call: the model must stay frozen for the evaluation."""
        if self.model is not None:
            raise RuntimeError("model already trained; retraining is disabled")
        X_warmup = np.asarray(X_warmup, dtype=np.float64)
        n = len(X_warmup)
        if n < MIN_TRAIN_SAMPLES:
            logger.error(f"Warm-up produced {n} flow vectors (< {MIN_TRAIN_SAMPLES}); "
                         f"model NOT trained")
            return None
        t0 = time.perf_counter()
        model = IsolationForest(**self.params)
        Xp = preprocess(X_warmup)
        model.fit(Xp)
        scores = model.decision_function(Xp)
        fit_s = time.perf_counter() - t0
        lo, sd = float(scores.min()), float(scores.std())
        self.model = model
        self.thresholds = Thresholds(
            normal=lo, monitor=lo - sd, clone=lo - 2 * sd,
            warmup_min=lo, warmup_std=sd, n_warmup=n,
        )
        self.train_info = {
            "sklearn_version": sklearn.__version__,
            "params": self.params,
            "features": FEATURE_NAMES,
            "log1p_features": [FEATURE_NAMES[i] for i in LOG1P_FEATURES],
            "n_train": n,
            "max_samples_effective": min(self.params["max_samples"], n),
            "fit_seconds": fit_s,
            "warmup_score_min": lo,
            "warmup_score_std": sd,
            "warmup_score_mean": float(scores.mean()),
            "thresholds": asdict(self.thresholds),
            "train_feature_min": X_warmup.min(axis=0).tolist(),
            "train_feature_max": X_warmup.max(axis=0).tolist(),
        }
        logger.info(
            f"IsolationForest trained once on {n} warm-up flow vectors "
            f"({fit_s*1000:.0f} ms); frozen. Thresholds: NORMAL > {lo:.4f}, "
            f"MONITOR > {lo - sd:.4f}, CLONE > {lo - 2*sd:.4f}, else DROP"
        )
        return self.thresholds

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not trained")
        if len(X) == 0:
            return np.zeros(0)
        assert X.shape[1] == N_FEATURES
        return self.model.decision_function(preprocess(X))
