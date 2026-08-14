"""
analysis/stats.py
Statistical analysis for AdaptMon experiment results.
Computes significance tests, confidence intervals, and summary tables.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def compute_confidence_interval(
    data       : np.ndarray,
    confidence : float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a sample.
    Returns (mean, lower_bound, upper_bound).
    """
    from scipy import stats as scipy_stats
    n    = len(data)
    mean = np.mean(data)
    if n < 2:
        return mean, mean, mean
    se   = scipy_stats.sem(data)
    h    = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def compute_ci_bootstrap(
    data       : np.ndarray,
    n_boot     : int   = 1000,
    confidence : float = 0.95,
    statistic  = np.mean,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval."""
    np.random.seed(42)
    boot_stats = [
        statistic(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ]
    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return statistic(data), lower, upper


def mann_whitney_test(
    group_a : np.ndarray,
    group_b : np.ndarray,
) -> Dict[str, float]:
    """
    Non-parametric Mann-Whitney U test for comparing two groups.
    """
    try:
        from scipy import stats as sp
        stat, p_value = sp.mannwhitneyu(
            group_a, group_b, alternative="two-sided"
        )
        return {
            "statistic"   : float(stat),
            "p_value"     : float(p_value),
            "significant" : p_value < 0.05,
        }
    except ImportError:
        return {"error": "scipy not available"}
    except Exception as e:
        return {"error": str(e)}


def ks_test(
    group_a : np.ndarray,
    group_b : np.ndarray,
) -> Dict[str, float]:
    """Kolmogorov-Smirnov test."""
    try:
        from scipy import stats as sp
        stat, p_value = sp.ks_2samp(group_a, group_b)
        return {
            "statistic"   : float(stat),
            "p_value"     : float(p_value),
            "significant" : p_value < 0.05,
        }
    except ImportError:
        return {"error": "scipy not available"}


def shapiro_wilk_test(data: np.ndarray) -> Dict[str, float]:
    """Test for normality using Shapiro-Wilk."""
    try:
        from scipy import stats as sp
        stat, p = sp.shapiro(data[:min(len(data), 5000)])
        return {
            "statistic" : float(stat),
            "p_value"   : float(p),
            "is_normal" : p > 0.05,
        }
    except ImportError:
        return {"error": "scipy not available"}


def compute_summary_table(
    dfs : List[pd.DataFrame],
    metrics : List[str] = None,
) -> pd.DataFrame:
    """
    Compute a summary statistics table across all runs.
    """
    if metrics is None:
        metrics = [
            "anomaly_score", "entropy",
            "n_active_flows", "controller_cpu_pct",
            "controller_mem_mb", "mean_pps",
        ]

    rows = []
    for run_idx, df in enumerate(dfs):
        row = {"run_id": run_idx + 1}
        for m in metrics:
            if m in df.columns:
                vals = df[m].dropna()
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"]  = float(vals.std())
                row[f"{m}_p50"]  = float(vals.quantile(0.5))
                row[f"{m}_p95"]  = float(vals.quantile(0.95))
        rows.append(row)

    return pd.DataFrame(rows)


def compute_anomaly_detection_roc(
    df              : pd.DataFrame,
    attack_start_s  : float,
    attack_end_s    : float,
    n_thresholds    : int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve for anomaly detection.
    Returns dict with 'fpr', 'tpr', 'thresholds', 'auc'.
    """
    if "anomaly_score" not in df.columns:
        return {}

    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        t0 = df["timestamp"].min()
        elapsed = (df["timestamp"] - t0).dt.total_seconds()
    else:
        elapsed = pd.Series(np.arange(len(df)))

    y_true = (
        (elapsed >= attack_start_s) & (elapsed <= attack_end_s)
    ).astype(int).values

    scores = df["anomaly_score"].fillna(0).values

    # Negate scores so higher = more anomalous (for ROC)
    scores_neg = -scores

    thresholds = np.linspace(scores_neg.min(), scores_neg.max(), n_thresholds)
    fprs = []
    tprs = []

    for thresh in thresholds:
        y_pred = (scores_neg >= thresh).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fprs.append(fpr)
        tprs.append(tpr)

    fprs = np.array(fprs)
    tprs = np.array(tprs)

    # AUC via trapezoidal rule
    sort_idx = np.argsort(fprs)
    auc = float(np.trapz(tprs[sort_idx], fprs[sort_idx]))

    return {
        "fpr"        : fprs,
        "tpr"        : tprs,
        "thresholds" : thresholds,
        "auc"        : auc,
    }


def compute_transition_statistics(
    df : pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze monitoring level transition patterns."""
    if "monitoring_level" not in df.columns:
        return {}

    levels = df["monitoring_level"].fillna(0).astype(int)

    n_transitions = int((levels.diff().abs() > 0).sum())
    transition_matrix = np.zeros((4, 4), dtype=int)

    for i in range(1, len(levels)):
        prev = min(3, max(0, int(levels.iloc[i-1])))
        curr = min(3, max(0, int(levels.iloc[i])))
        transition_matrix[prev, curr] += 1

    level_durations = {}
    for lvl in range(4):
        mask     = levels == lvl
        duration = int(mask.sum())
        level_durations[lvl] = duration

    return {
        "n_transitions"     : n_transitions,
        "transition_matrix" : transition_matrix.tolist(),
        "level_durations"   : level_durations,
        "dominant_level"    : int(levels.mode().iloc[0]),
    }


def generate_report(
    dfs         : List[pd.DataFrame],
    output_path : str,
    attack_start_s : float = 30.0,
    attack_end_s   : float = 90.0,
) -> None:
    """
    Generate a text/markdown report of experiment results.
    """
    lines = [
        "# AdaptMon Experiment Report",
        "",
        f"Runs analyzed: {len(dfs)}",
        "",
    ]

    if not dfs:
        lines.append("No data available.")
    else:
        # Summary table
        summary = compute_summary_table(dfs)
        lines.append("## Summary Statistics Across Runs")
        lines.append("")
        lines.append(summary.to_markdown(index=False))
        lines.append("")

        # Anomaly detection per run
        lines.append("## Anomaly Detection Performance")
        lines.append("")

        from experiments.metrics import compute_detection_metrics

        for i, df in enumerate(dfs):
            dm = compute_detection_metrics(
                df, attack_start_s, attack_end_s
            )
            if dm:
                lines.append(
                    f"Run {i+1}: "
                    f"Precision={dm.get('precision', 0):.3f} "
                    f"Recall={dm.get('recall', 0):.3f} "
                    f"F1={dm.get('f1_score', 0):.3f} "
                    f"Latency={dm.get('detection_latency_s', 'N/A'):.1f}s"
                )
        lines.append("")

        # Mean detection metrics
        precision_vals = []
        recall_vals    = []
        f1_vals        = []
        latency_vals   = []

        for df in dfs:
            dm = compute_detection_metrics(df, attack_start_s, attack_end_s)
            if dm:
                precision_vals.append(dm.get("precision", 0))
                recall_vals.append(dm.get("recall", 0))
                f1_vals.append(dm.get("f1_score", 0))
                if dm.get("detection_latency_s", float("inf")) < float("inf"):
                    latency_vals.append(dm["detection_latency_s"])

        if precision_vals:
            lines.append(f"Mean Precision: {np.mean(precision_vals):.3f} ± {np.std(precision_vals):.3f}")
            lines.append(f"Mean Recall: {np.mean(recall_vals):.3f} ± {np.std(recall_vals):.3f}")
            lines.append(f"Mean F1: {np.mean(f1_vals):.3f} ± {np.std(f1_vals):.3f}")
        if latency_vals:
            lines.append(f"Mean Detection Latency: {np.mean(latency_vals):.2f}s ± {np.std(latency_vals):.2f}s")

        lines.append("")

        # Transition analysis
        if len(dfs) > 0:
            lines.append("## Monitoring Level Transitions")
            combined_df = pd.concat(dfs, ignore_index=True)
            ts = compute_transition_statistics(combined_df)
            lines.append(f"Total transitions: {ts.get('n_transitions', 0)}")
            lines.append(f"Dominant level: {['NORMAL','MONITOR','CLONE','DROP'][ts.get('dominant_level', 0)]}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {output_path}")
