"""
controller/model.py
Anomaly detection using IsolationForest.
Handles training, online scoring, and model persistence.
"""

import numpy as np
import os
import pickle
import logging
import time
from typing import List, Optional, Tuple, Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "entropy",
    "n_active_flows",
    "mean_pps",
    "mean_bps",
    "max_pps",
    "max_bps",
    "std_pps",
    "std_bps",
    "gini_pkts",
    "top_flow_fraction",
]

N_FEATURES = len(FEATURE_NAMES)

# Count and rate features, unbounded and heavy-tailed under a flood:
# log1p-compressed before scaling so benign values are not squashed into
# a sliver of the range by the flood's extremes.
LOG_FEATURES = [
    FEATURE_NAMES.index(name) for name in (
        "n_active_flows",
        "mean_pps", "mean_bps", "max_pps", "max_bps", "std_pps", "std_bps",
    )
]


# Floor on the warm-up std used for z-scores, so a feature that was
# (nearly) constant during warm-up doesn't turn tiny changes into huge z
def z_std_floor(mean: np.ndarray) -> np.ndarray:
    return 0.05 * np.abs(mean) + 0.01


def log_count_features(X: np.ndarray) -> np.ndarray:
    """Apply log1p to the LOG_FEATURES columns (clipped at 0)."""
    X = np.array(X, dtype=np.float64, copy=True)
    X[:, LOG_FEATURES] = np.log1p(np.clip(X[:, LOG_FEATURES], 0.0, None))
    return X


class AnomalyDetector:
    """
    Online anomaly detection using IsolationForest.

    Two training modes:
      * train_until set (experiments): buffer every sample whose timestamp
        is before train_until (the benign warm-up), train exactly once on
        that buffer when the first later sample arrives, then freeze. No
        sample at or after train_until is ever used for training.
      * train_until None (standalone): sliding window, periodic retraining.
    """

    def __init__(
        self,
        window_size          : int   = 100,
        n_estimators         : int   = 100,
        contamination        : float = 0.05,
        retrain_interval     : int   = 50,
        min_samples_to_train : int   = 20,
        train_until          : Optional[float] = None,
    ):
        self.window_size          = window_size
        self.n_estimators         = n_estimators
        self.contamination        = contamination
        self.retrain_interval     = retrain_interval
        self.min_samples_to_train = min_samples_to_train
        self.train_until          = train_until

        self._buffer         : List[np.ndarray] = []
        self._scores         : List[float]       = []
        self._timestamps     : List[float]       = []
        self._pipeline       : Optional[Pipeline] = None
        self._is_trained     : bool              = False
        self._samples_since_retrain : int        = 0
        self._training_count        : int        = 0
        self._warmup_failed         : bool       = False
        # Frozen model's scores for its own warm-up samples (frozen mode)
        self.warmup_scores          : Optional[np.ndarray] = None
        # Per-feature warm-up mean/std after log1p, for z_max (frozen mode)
        self._z_mean                : Optional[np.ndarray] = None
        self._z_std                 : Optional[np.ndarray] = None

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        # The log step is part of the pipeline, so fit and every
        # decision_function call apply the same transform.
        self._pipeline = Pipeline([
            ("log",    FunctionTransformer(log_count_features)),
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators   = self.n_estimators,
                contamination  = self.contamination,
                random_state   = 42,
                n_jobs         = -1,
                warm_start     = False,
            ))
        ])

    def ingest(
        self,
        feature_vector : np.ndarray,
        timestamp      : float
    ) -> Optional[float]:
        """
        Ingest a new feature vector.
        Returns anomaly score if model is trained, else None.
        The score is IsolationForest.decision_function(): 0 is the
        contamination boundary, negative = anomalous, and lower = more
        anomalous. (Raw score_samples() sits around -0.45 even for benign
        traffic, below every PolicyConfig threshold.)
        """
        if feature_vector.shape[0] != N_FEATURES:
            logger.warning(
                f"Expected {N_FEATURES} features, "
                f"got {feature_vector.shape[0]}"
            )
            return None

        # Replace NaN/Inf
        fv = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.train_until is not None:
            return self._ingest_frozen(fv, timestamp)

        self._buffer.append(fv)
        self._timestamps.append(timestamp)
        self._samples_since_retrain += 1

        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
            self._timestamps.pop(0)

        # Retrain if needed
        if (
            len(self._buffer) >= self.min_samples_to_train
            and self._samples_since_retrain >= self.retrain_interval
        ):
            self._retrain()

        # Score current sample
        if self._is_trained:
            score = self._score(fv)
            self._scores.append(score)
            return score

        return None

    def _ingest_frozen(
        self,
        fv        : np.ndarray,
        timestamp : float
    ) -> Optional[float]:
        """Warm-up-then-freeze mode (see class docstring)."""
        if timestamp < self.train_until:
            self._buffer.append(fv)
            self._timestamps.append(timestamp)
            return None

        if not self._is_trained and not self._warmup_failed:
            if len(self._buffer) >= self.min_samples_to_train:
                self._retrain()
                if self._is_trained:
                    X = np.array(self._buffer, dtype=np.float64)
                    self.warmup_scores = self._pipeline.decision_function(X)
                    X_log = log_count_features(X)
                    self._z_mean = X_log.mean(axis=0)
                    self._z_std  = np.maximum(
                        X_log.std(axis=0), z_std_floor(self._z_mean)
                    )
                logger.info("Model frozen after benign warm-up")
            else:
                # Never fall back to training on post-warm-up (attack) data
                self._warmup_failed = True
                logger.error(
                    f"Warm-up produced only {len(self._buffer)} samples "
                    f"(need {self.min_samples_to_train}); "
                    f"detector stays untrained for this run"
                )

        if self._is_trained:
            score = self._score(fv)
            self._scores.append(score)
            return score
        return None

    def z_max(self, feature_vector: np.ndarray) -> Optional[float]:
        """
        Largest per-feature |z| against the benign warm-up (log1p space).
        Complements the IF score, which saturates on gross outliers.
        None until the warm-up statistics exist (frozen mode only).
        """
        if self._z_mean is None:
            return None
        fv = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        x  = log_count_features(fv.reshape(1, -1))[0]
        return float(np.max(np.abs(x - self._z_mean) / self._z_std))

    def _retrain(self) -> None:
        """Retrain the IsolationForest on the current buffer."""
        X = np.array(self._buffer, dtype=np.float64)
        try:
            self._pipeline.fit(X)
            self._is_trained = True
            self._samples_since_retrain = 0
            self._training_count += 1
            logger.info(
                f"Model retrained (count={self._training_count}) "
                f"on {len(X)} samples"
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")

    def _score(self, fv: np.ndarray) -> float:
        """Score a single feature vector."""
        try:
            X = fv.reshape(1, -1)
            score = float(self._pipeline.decision_function(X)[0])
            return score
        except Exception as e:
            logger.warning(f"Scoring failed: {e}")
            return 0.0

    def score_batch(
        self,
        feature_matrix: np.ndarray
    ) -> np.ndarray:
        """Score a batch of feature vectors (decision_function scale)."""
        if not self._is_trained:
            return np.zeros(len(feature_matrix))
        X = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            return self._pipeline.decision_function(X)
        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
            return np.zeros(len(feature_matrix))

    def predict_batch(
        self,
        feature_matrix: np.ndarray
    ) -> np.ndarray:
        """Predict labels: 1=normal, -1=anomaly."""
        if not self._is_trained:
            return np.ones(len(feature_matrix))
        X = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            return self._pipeline.predict(X)
        except Exception as e:
            logger.warning(f"Batch predict failed: {e}")
            return np.ones(len(feature_matrix))

    def get_latest_score(self) -> Optional[float]:
        if self._scores:
            return self._scores[-1]
        return None

    def get_score_history(self) -> List[float]:
        return self._scores.copy()

    def get_timestamp_history(self) -> List[float]:
        return self._timestamps.copy()

    def is_trained(self) -> bool:
        return self._is_trained

    def warmup_failed(self) -> bool:
        return self._warmup_failed

    def save(self, path: str) -> None:
        """Persist model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline"       : self._pipeline,
                "buffer"         : self._buffer,
                "scores"         : self._scores,
                "timestamps"     : self._timestamps,
                "is_trained"     : self._is_trained,
                "training_count" : self._training_count,
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> bool:
        """Load persisted model."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._pipeline       = state["pipeline"]
            self._buffer         = state["buffer"]
            self._scores         = state["scores"]
            self._timestamps     = state["timestamps"]
            self._is_trained     = state["is_trained"]
            self._training_count = state["training_count"]
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    def get_stats(self) -> Dict:
        """Return model statistics."""
        scores = np.array(self._scores) if self._scores else np.array([0.0])
        return {
            "is_trained"     : self._is_trained,
            "training_count" : self._training_count,
            "buffer_size"    : len(self._buffer),
            "n_scores"       : len(self._scores),
            "mean_score"     : float(scores.mean()),
            "min_score"      : float(scores.min()),
            "max_score"      : float(scores.max()),
            "std_score"      : float(scores.std()),
        }


class PerFlowDetector:
    """
    Per-flow anomaly detector for flagging individual flows.
    Uses lightweight statistics without full IsolationForest
    to avoid per-flow model overhead.
    """

    def __init__(self, zscore_threshold: float = 3.0):
        self.zscore_threshold = zscore_threshold
        self._flow_pps_history : Dict[int, List[float]] = {}
        self._flow_bps_history : Dict[int, List[float]] = {}

    def update(
        self,
        flow_index : int,
        pps        : float,
        bps        : float
    ) -> Tuple[float, float]:
        """
        Update per-flow statistics.
        Returns (pps_zscore, bps_zscore).
        """
        if flow_index not in self._flow_pps_history:
            self._flow_pps_history[flow_index] = []
            self._flow_bps_history[flow_index] = []

        self._flow_pps_history[flow_index].append(pps)
        self._flow_bps_history[flow_index].append(bps)

        # Keep window
        if len(self._flow_pps_history[flow_index]) > 30:
            self._flow_pps_history[flow_index].pop(0)
            self._flow_bps_history[flow_index].pop(0)

        return (
            self._zscore(self._flow_pps_history[flow_index]),
            self._zscore(self._flow_bps_history[flow_index]),
        )

    @staticmethod
    def _zscore(history: List[float]) -> float:
        if len(history) < 3:
            return 0.0
        arr = np.array(history)
        mean = arr.mean()
        std  = arr.std()
        if std == 0:
            return 0.0
        return float((history[-1] - mean) / std)

    def is_anomalous(
        self,
        flow_index : int,
        pps        : float,
        bps        : float
    ) -> bool:
        pps_z, bps_z = self.update(flow_index, pps, bps)
        return (
            abs(pps_z) > self.zscore_threshold
            or abs(bps_z) > self.zscore_threshold
        )
