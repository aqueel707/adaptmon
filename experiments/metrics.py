"""
experiments/metrics.py
Metrics computation and aggregation for AdaptMon experiments.
Reads CSV logs from the controller and computes summary statistics.
"""

import os
import csv
import time
import logging
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_metrics_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """Load a controller metrics CSV file into a DataFrame."""
    if not os.path.exists(csv_path):
        logger.warning(f"Metrics file not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f"Empty metrics file: {csv_path}")
            return None
        # Handle both unix-epoch floats and datetime strings
        ts = df["timestamp"]
        try:
            # Try numeric unix epoch first
            df["timestamp"] = pd.to_datetime(pd.to_numeric(ts), unit="s")
        except (ValueError, TypeError):
            try:
                # Fall back to string parsing
                df["timestamp"] = pd.to_datetime(ts)
            except Exception:
                # Last resort: integer index as seconds
                df["timestamp"] = pd.to_datetime(
                    pd.RangeIndex(len(df)), unit="s",
                    origin="2024-01-01"
                )
        return df
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return None


def aggregate_run_metrics(csv_path: str) -> Dict:
    """Compute summary statistics for a single run."""
    df = load_metrics_csv(csv_path)
    if df is None:
        return {}

    metrics = {}

    # Anomaly scores
    if "anomaly_score" in df.columns:
        scores = df["anomaly_score"].dropna()
        metrics["anomaly_score_mean"] = float(scores.mean())
        metrics["anomaly_score_std"]  = float(scores.std())
        metrics["anomaly_score_min"]  = float(scores.min())
        metrics["anomaly_score_max"]  = float(scores.max())

    # Monitoring levels
    if "monitoring_level" in df.columns:
        level_counts = df["monitoring_level"].value_counts().to_dict()
        metrics["level_normal_count"]  = int(level_counts.get(0, 0))
        metrics["level_monitor_count"] = int(level_counts.get(1, 0))
        metrics["level_clone_count"]   = int(level_counts.get(2, 0))
        metrics["level_drop_count"]    = int(level_counts.get(3, 0))
        metrics["level_transitions"]   = int(
            (df["monitoring_level"].diff().abs() > 0).sum()
        )

    # Controller CPU
    if "controller_cpu_pct" in df.columns:
        cpu = df["controller_cpu_pct"].dropna()
        metrics["cpu_mean"]  = float(cpu.mean())
        metrics["cpu_std"]   = float(cpu.std())
        metrics["cpu_max"]   = float(cpu.max())
        metrics["cpu_p95"]   = float(np.percentile(cpu, 95))

    # Memory
    if "controller_mem_mb" in df.columns:
        mem = df["controller_mem_mb"].dropna()
        metrics["mem_mean_mb"] = float(mem.mean())
        metrics["mem_max_mb"]  = float(mem.max())

    # Flow counts
    if "n_active_flows" in df.columns:
        flows = df["n_active_flows"].dropna()
        metrics["flows_mean"] = float(flows.mean())
        metrics["flows_max"]  = float(flows.max())

    # Entropy
    if "entropy" in df.columns:
        ent = df["entropy"].dropna()
        metrics["entropy_mean"] = float(ent.mean())
        metrics["entropy_std"]  = float(ent.std())
        metrics["entropy_min"]  = float(ent.min())
        metrics["entropy_max"]  = float(ent.max())

    # Throughput features
    for col in ["mean_pps", "mean_bps", "max_pps", "max_bps"]:
        if col in df.columns:
            vals = df[col].dropna()
            metrics[f"{col}_mean"] = float(vals.mean())
            metrics[f"{col}_max"]  = float(vals.max())

    metrics["n_samples"] = len(df)
    metrics["duration_s"] = (
        df["timestamp"].max() - df["timestamp"].min()
    ).total_seconds() if len(df) > 1 else 0.0

    return metrics


def aggregate_series_metrics(
    csv_paths : List[str]
) -> Dict:
    """Aggregate metrics across all runs in a series."""
    all_metrics = []
    for path in csv_paths:
        m = aggregate_run_metrics(path)
        if m:
            all_metrics.append(m)

    if not all_metrics:
        return {}

    # Compute mean and std across runs for each metric
    aggregated = {}
    keys = set()
    for m in all_metrics:
        keys.update(m.keys())

    for key in keys:
        vals = [m.get(key, 0) for m in all_metrics if key in m]
        if vals and isinstance(vals[0], (int, float)):
            aggregated[f"{key}_mean_across_runs"] = float(np.mean(vals))
            aggregated[f"{key}_std_across_runs"]  = float(np.std(vals))

    aggregated["n_runs"] = len(all_metrics)
    return aggregated


def compute_detection_metrics(
    df              : pd.DataFrame,
    attack_start_s  : float,
    attack_end_s    : float,
    threshold       : float = -0.2,
) -> Dict:
    """
    Compute attack detection performance metrics.

    Args:
        df             : Metrics DataFrame
        attack_start_s : Seconds from start when attack begins
        attack_end_s   : Seconds from start when attack ends
        threshold      : Score threshold for anomaly detection
    """
    if df is None or df.empty:
        return {}

    # Normalize timestamps
    t0 = df["timestamp"].min()
    df = df.copy()
    df["elapsed"] = (df["timestamp"] - t0).dt.total_seconds()

    # Ground truth labels
    df["is_attack"] = (
        (df["elapsed"] >= attack_start_s)
        & (df["elapsed"] <= attack_end_s)
    )

    # Predicted labels based on anomaly score
    if "anomaly_score" not in df.columns:
        return {}

    df["predicted_anomaly"] = df["anomaly_score"] < threshold

    # Confusion matrix
    tp = int(((df["is_attack"]) & (df["predicted_anomaly"])).sum())
    fp = int(((~df["is_attack"]) & (df["predicted_anomaly"])).sum())
    tn = int(((~df["is_attack"]) & (~df["predicted_anomaly"])).sum())
    fn = int(((df["is_attack"]) & (~df["predicted_anomaly"])).sum())

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Detection latency: time from attack start to first detection
    attack_rows = df[
        df["is_attack"] & df["predicted_anomaly"]
    ]
    if not attack_rows.empty:
        first_detect = attack_rows["elapsed"].min()
        detect_latency = first_detect - attack_start_s
    else:
        detect_latency = float("inf")

    return {
        "tp"              : tp,
        "fp"              : fp,
        "tn"              : tn,
        "fn"              : fn,
        "precision"       : precision,
        "recall"          : recall,
        "f1_score"        : f1,
        "fpr"             : fpr,
        "detection_latency_s": detect_latency,
    }


def compute_overhead_vs_flows(n_flows: int) -> Dict:
    """
    Synthetic benchmark: estimate controller CPU overhead for n_flows.
    Simulates the register polling operation timing.
    """
    import subprocess
    import time as _time

    proc = psutil.Process(os.getpid())

    # Simulate processing n_flows feature vectors
    fake_pkts  = {i: np.random.randint(0, 1000) for i in range(n_flows)}
    fake_bytes = {i: np.random.randint(0, 100000) for i in range(n_flows)}

    cpu_before = proc.cpu_percent(interval=0.1)
    mem_before = proc.memory_info().rss / 1024 / 1024

    t_start = _time.time()

    # Feature extraction simulation
    counts = np.array(list(fake_pkts.values()), dtype=np.float64)
    total  = counts.sum()
    if total > 0:
        probs   = counts / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    else:
        entropy = 0.0

    # Isolation Forest scoring simulation
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        X = np.array([
            [entropy, n_flows,
             float(counts.mean()), float(counts.std()),
             float(counts.max()), 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(
            np.random.randn(max(n_flows, 20), 10)
        )
        model = IsolationForest(n_estimators=100, random_state=42)
        model.fit(X_scaled)
        _ = model.score_samples(X[:1])
    except Exception:
        pass

    elapsed = _time.time() - t_start

    cpu_after = proc.cpu_percent(interval=0.1)
    mem_after = proc.memory_info().rss / 1024 / 1024

    return {
        "n_flows"         : n_flows,
        "processing_ms"   : elapsed * 1000,
        "cpu_delta_pct"   : cpu_after - cpu_before,
        "mem_delta_mb"    : mem_after - mem_before,
        "entropy"         : entropy,
    }


def load_all_series_csvs(
    series_dir : str
) -> List[pd.DataFrame]:
    """Load all CSV files from a series directory."""
    dataframes = []
    for root, _, files in os.walk(series_dir):
        for fname in files:
            if fname.endswith("_metrics.csv"):
                fpath = os.path.join(root, fname)
                df = load_metrics_csv(fpath)
                if df is not None:
                    dataframes.append(df)
    return dataframes


def combined_metrics_df(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Combine multiple metrics DataFrames into one."""
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    return combined
