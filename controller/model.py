"""
controller/model.py
Two Isolation Forests, each fitted exactly once on benign warm-up vectors
and never refitted afterwards (Decision B adds the second model):

  FlowIsolationForest      one flow (5-tuple) per sample (controller/
                            features.py FEATURE_NAMES)
  HostPairIsolationForest  one (src IP, dst IP) aggregate per sample
                            (controller/features.py PAIR_FEATURE_NAMES)

Both use the same IsolationForest hyperparameters (IF_PARAMS) and the same
preprocessing shape (log1p on their own heavy-tailed columns; no scaling,
since Isolation Forest splits are invariant to per-feature affine
transforms). Scores are IsolationForest.decision_function(): lower = more
anomalous.

Thresholds (Decision A, revised): derived from each model's own scores on
its own warm-up samples -- MONITOR below the warm-up 5th percentile, CLONE
below the warm-up 1st percentile. DROP is never reached from a score: a
flow or pair already at CLONE is escalated to DROP only when
controller/dpi.py confirms an attack on it (see controller/monitor.py and
controller/policy.py).

CLONE was originally the raw warm-up minimum (the single most extreme
score seen anywhere in warm-up). That is fragile: it takes only one
unusually quiet or unusually bursty benign sample during warm-up to set a
bar that no later attack sample needs to beat to go undetected -- measured
on a real run, the warm-up minimum was *more* extreme than the best score
any attack flow reached, for every attack class. The 1st percentile keeps
the same "far out in the warm-up's own tail" intent without anchoring to
whichever single sample happened to be the worst.
"""

import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

from .features import FEATURE_NAMES, N_FEATURES, PAIR_FEATURE_NAMES, N_PAIR_FEATURES

logger = logging.getLogger(__name__)

IF_PARAMS = dict(
    n_estimators  = 100,
    max_samples   = 256,
    contamination = 0.05,
    random_state  = 42,
)
FLOW_LOG1P_FEATURES = ("pkt_count", "byte_count", "duration_s", "pkt_rate")
PAIR_LOG1P_FEATURES = ("new_flows", "distinct_dst_ports", "pkt_count")
MIN_TRAIN_SAMPLES = 50


@dataclass
class Thresholds:
    """
    Score thresholds derived from a model's own scores on its warm-up
    samples (Decision A): score > monitor => NORMAL, > clone => MONITOR,
    else => CLONE. DROP is never reached from the score alone.
    """
    monitor    : float     # warm-up 5th percentile
    clone      : float     # warm-up 1st percentile
    warmup_min : float
    warmup_p1  : float
    warmup_p5  : float
    warmup_p10 : float
    warmup_mean: float
    warmup_std : float
    n_warmup   : int


class IFModel:
    """
    One Isolation Forest, fit_once() on a batch of warm-up feature vectors,
    frozen after. `feature_names`/`log1p_features` describe the vectors it
    scores (flow-level or host-pair-level; see module docstring).
    """

    def __init__(self, feature_names: Iterable[str], log1p_features: Iterable[str],
                 **params):
        self.feature_names = list(feature_names)
        self.log1p_idx = [self.feature_names.index(n) for n in log1p_features]
        self.params = {**IF_PARAMS, **params}
        self.model: Optional[IsolationForest] = None
        self.thresholds: Optional[Thresholds] = None
        self.train_info: Dict = {}

    @property
    def trained(self) -> bool:
        return self.model is not None

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.array(X, dtype=np.float64, copy=True),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if self.log1p_idx:
            X[:, self.log1p_idx] = np.log1p(np.clip(X[:, self.log1p_idx], 0.0, None))
        return X

    def fit_once(self, X_warmup: np.ndarray) -> Optional[Thresholds]:
        """Fit on the warm-up vectors and derive the thresholds. Refuses a
        second call: the model must stay frozen for the evaluation."""
        if self.model is not None:
            raise RuntimeError("model already trained; retraining is disabled")
        X_warmup = np.asarray(X_warmup, dtype=np.float64)
        n = len(X_warmup)
        if n < MIN_TRAIN_SAMPLES:
            logger.error(f"Warm-up produced {n} samples for {self.feature_names} "
                         f"(< {MIN_TRAIN_SAMPLES}); model NOT trained")
            return None
        t0 = time.perf_counter()
        model = IsolationForest(**self.params)
        Xp = self._preprocess(X_warmup)
        model.fit(Xp)
        scores = model.decision_function(Xp)
        mn  = float(scores.min())
        p1  = float(np.percentile(scores, 1))
        p5  = float(np.percentile(scores, 5))
        p10 = float(np.percentile(scores, 10))
        std = float(scores.std())
        fit_s = time.perf_counter() - t0
        self.model = model
        self.thresholds = Thresholds(
            monitor=p5, clone=p1, warmup_min=mn, warmup_p1=p1, warmup_p5=p5,
            warmup_p10=p10, warmup_mean=float(scores.mean()), warmup_std=std,
            n_warmup=n,
        )
        self.train_info = {
            "sklearn_version": sklearn.__version__,
            "params": self.params,
            "features": self.feature_names,
            "log1p_features": [self.feature_names[i] for i in self.log1p_idx],
            "n_train": n,
            "max_samples_effective": min(self.params["max_samples"], n),
            "fit_seconds": fit_s,
            "warmup_score_min": mn,
            "warmup_score_p1": p1,
            "warmup_score_p5": p5,
            "warmup_score_p10": p10,
            "warmup_score_mean": float(scores.mean()),
            "warmup_score_std": std,
            "warmup_scores": scores.tolist(),
            "thresholds": asdict(self.thresholds),
            "train_feature_min": X_warmup.min(axis=0).tolist(),
            "train_feature_max": X_warmup.max(axis=0).tolist(),
        }
        logger.info(
            f"IsolationForest[{','.join(self.feature_names)}] trained once on "
            f"{n} warm-up vectors ({fit_s*1000:.0f} ms); frozen. "
            f"Thresholds: NORMAL > {p5:.4f} (p5), MONITOR > {p1:.4f} (p1), "
            f"else CLONE (warm-up min was {mn:.4f}); DROP only on DPI confirmation"
        )
        return self.thresholds

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not trained")
        if len(X) == 0:
            return np.zeros(0)
        assert X.shape[1] == len(self.feature_names)
        return self.model.decision_function(self._preprocess(X))


class FlowIsolationForest(IFModel):
    def __init__(self, **params):
        super().__init__(FEATURE_NAMES, FLOW_LOG1P_FEATURES, **params)


class HostPairIsolationForest(IFModel):
    def __init__(self, **params):
        super().__init__(PAIR_FEATURE_NAMES, PAIR_LOG1P_FEATURES, **params)
