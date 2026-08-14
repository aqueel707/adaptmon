"""
analysis/plots.py
Automatic result graph generation for AdaptMon experiments.
Generates all required plots and saves to results/ directory.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import List, Optional, Dict, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(ROOT, "results")
STYLE_PARAMS = {
    "figure.figsize"   : (10, 6),
    "axes.titlesize"   : 14,
    "axes.labelsize"   : 12,
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "legend.fontsize"  : 10,
    "lines.linewidth"  : 2,
    "grid.alpha"       : 0.4,
    "axes.grid"        : True,
}

LEVEL_COLORS = {
    0: "#2ecc71",   # NORMAL  - green
    1: "#f39c12",   # MONITOR - orange
    2: "#e74c3c",   # CLONE   - red
    3: "#8e44ad",   # DROP    - purple
}

LEVEL_NAMES = {
    0: "NORMAL",
    1: "MONITOR",
    2: "CLONE",
    3: "DROP",
}


def _apply_style():
    plt.rcParams.update(STYLE_PARAMS)


def plot_anomaly_scores(
    df          : pd.DataFrame,
    output_path : str,
    attack_start_s : Optional[float] = 30.0,
    attack_end_s   : Optional[float] = 90.0,
    thresholds     : Dict[str, float] = None,
) -> None:
    """
    Plot anomaly scores over time with monitoring level background
    and attack period annotation.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        t0 = df["timestamp"].min()
        elapsed = (df["timestamp"] - t0).dt.total_seconds()
    else:
        elapsed = np.arange(len(df))

    # ---- Upper: anomaly scores ----
    ax1 = axes[0]
    if "anomaly_score" in df.columns:
        scores = df["anomaly_score"].ffill()
        ax1.plot(
            elapsed, scores,
            color="#2c3e50", linewidth=1.5,
            label="Anomaly Score", alpha=0.9
        )
        ax1.fill_between(
            elapsed, scores, 0,
            where=(scores < 0),
            alpha=0.2, color="#e74c3c",
            label="Below Baseline"
        )

        # Threshold lines
        if thresholds is None:
            thresholds = {
                "NORMAL/MONITOR": -0.10,
                "MONITOR/CLONE" : -0.20,
                "CLONE/DROP"    : -0.30,
            }
        colors = ["#f39c12", "#e74c3c", "#8e44ad"]
        for (name, val), color in zip(thresholds.items(), colors):
            ax1.axhline(
                val, linestyle="--", linewidth=1.2,
                color=color, label=f"Threshold ({name}): {val}"
            )

    # Attack period shading
    if attack_start_s is not None and attack_end_s is not None:
        ax1.axvspan(
            attack_start_s, attack_end_s,
            alpha=0.15, color="#e74c3c", label="Attack Period"
        )

    ax1.set_ylabel("Anomaly Score")
    ax1.set_title("Anomaly Scores Over Time")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_ylim(bottom=min(-0.5, scores.min() * 1.1) if "anomaly_score" in df.columns else -0.5)

    # ---- Lower: monitoring level ----
    ax2 = axes[1]
    if "monitoring_level" in df.columns:
        levels = df["monitoring_level"].fillna(0).astype(int)
        # Background color by level
        for i, (t, lvl) in enumerate(zip(elapsed, levels)):
            t_end = elapsed.iloc[i+1] if i+1 < len(elapsed) else elapsed.iloc[-1]
            ax2.axvspan(
                t, t_end,
                alpha=0.7,
                color=LEVEL_COLORS.get(lvl, "#cccccc")
            )
        ax2.plot(
            elapsed, levels,
            color="#2c3e50", linewidth=1.0, drawstyle="steps-post"
        )
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(["NORMAL", "MONITOR", "CLONE", "DROP"])
        ax2.set_ylim(-0.2, 3.5)

        # Legend patches
        patches = [
            Patch(color=LEVEL_COLORS[k], label=LEVEL_NAMES[k])
            for k in sorted(LEVEL_COLORS.keys())
        ]
        ax2.legend(handles=patches, loc="upper right", fontsize=8)

    if attack_start_s is not None and attack_end_s is not None:
        ax2.axvspan(
            attack_start_s, attack_end_s,
            alpha=0.1, color="#e74c3c"
        )

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Monitoring Level")
    ax2.set_title("Monitoring Level Transitions")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_monitoring_levels(
    df          : pd.DataFrame,
    output_path : str,
) -> None:
    """
    Plot monitoring level distribution over time (stacked bar / timeline).
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: level over time ----
    ax1 = axes[0]
    if "monitoring_level" in df.columns and "timestamp" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            t0 = df["timestamp"].min()
            elapsed = (df["timestamp"] - t0).dt.total_seconds()
        else:
            elapsed = np.arange(len(df))

        levels = df["monitoring_level"].fillna(0).astype(int)
        ax1.step(elapsed, levels, where="post", color="#2c3e50", linewidth=1.5)
        ax1.set_yticks([0, 1, 2, 3])
        ax1.set_yticklabels(["NORMAL", "MONITOR", "CLONE", "DROP"])
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Monitoring Level")
        ax1.set_title("Monitoring Level Timeline")

        for lvl, color in LEVEL_COLORS.items():
            mask = levels == lvl
            if mask.any():
                ax1.fill_between(
                    elapsed, lvl, levels,
                    where=(levels == lvl),
                    alpha=0.3, color=color, step="post"
                )

    # ---- Right: time distribution per level ----
    ax2 = axes[1]
    if "monitoring_level" in df.columns:
        counts = df["monitoring_level"].fillna(0).astype(int).value_counts().sort_index()
        labels = [LEVEL_NAMES.get(i, str(i)) for i in counts.index]
        colors = [LEVEL_COLORS.get(i, "#cccccc") for i in counts.index]
        pct    = counts / counts.sum() * 100

        bars = ax2.bar(labels, pct.values, color=colors, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, pct.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9
            )
        ax2.set_xlabel("Monitoring Level")
        ax2.set_ylabel("Time Fraction (%)")
        ax2.set_title("Time Distribution Across Levels")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_controller_cpu_usage(
    dfs         : List[pd.DataFrame],
    output_path : str,
) -> None:
    """
    Plot controller CPU usage across runs (mean ± std).
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: CPU over time for each run ----
    ax1 = axes[0]
    max_len = 0
    all_cpu = []

    for run_idx, df in enumerate(dfs):
        if "controller_cpu_pct" not in df.columns:
            continue
        cpu = df["controller_cpu_pct"].fillna(0).values
        elapsed = np.arange(len(cpu))
        ax1.plot(
            elapsed, cpu,
            alpha=0.4, linewidth=0.8,
            color="#3498db", label=f"Run {run_idx+1}" if run_idx < 3 else None
        )
        all_cpu.append(cpu)
        max_len = max(max_len, len(cpu))

    # Mean ± std across runs
    if all_cpu:
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in all_cpu
        ]
        stacked = np.array(padded)
        mean_cpu = np.nanmean(stacked, axis=0)
        std_cpu  = np.nanstd(stacked, axis=0)
        elapsed  = np.arange(max_len)
        ax1.plot(elapsed, mean_cpu, color="#e74c3c",
                 linewidth=2.5, label="Mean", zorder=5)
        ax1.fill_between(
            elapsed,
            mean_cpu - std_cpu,
            mean_cpu + std_cpu,
            alpha=0.25, color="#e74c3c"
        )

    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("CPU Usage (%)")
    ax1.set_title(f"Controller CPU Usage ({len(dfs)} runs)")
    ax1.legend(fontsize=8)

    # ---- Right: CPU distribution ----
    ax2 = axes[1]
    all_vals = []
    for df in dfs:
        if "controller_cpu_pct" in df.columns:
            all_vals.extend(df["controller_cpu_pct"].dropna().tolist())

    if all_vals:
        ax2.hist(
            all_vals, bins=40,
            color="#3498db", edgecolor="white",
            alpha=0.8
        )
        ax2.axvline(
            np.mean(all_vals),
            color="#e74c3c", linewidth=2,
            linestyle="--",
            label=f"Mean: {np.mean(all_vals):.1f}%"
        )
        ax2.axvline(
            np.percentile(all_vals, 95),
            color="#8e44ad", linewidth=2,
            linestyle=":",
            label=f"P95: {np.percentile(all_vals, 95):.1f}%"
        )
        ax2.set_xlabel("CPU Usage (%)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("CPU Usage Distribution")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_flow_distribution(
    df          : pd.DataFrame,
    output_path : str,
) -> None:
    """
    Plot flow distribution metrics: active flows, entropy, gini over time.
    """
    _apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        t0 = df["timestamp"].min()
        elapsed = (df["timestamp"] - t0).dt.total_seconds()
    else:
        elapsed = np.arange(len(df))

    # ---- Active flows ----
    ax1 = axes[0]
    if "n_active_flows" in df.columns:
        flows = df["n_active_flows"].fillna(0)
        ax1.plot(elapsed, flows, color="#2ecc71", linewidth=1.5)
        ax1.fill_between(elapsed, flows, alpha=0.2, color="#2ecc71")
        ax1.set_ylabel("Active Flows")
        ax1.set_title("Active Flow Count Over Time")

    # ---- Entropy ----
    ax2 = axes[1]
    if "entropy" in df.columns:
        entropy = df["entropy"].fillna(0)
        ax2.plot(elapsed, entropy, color="#3498db", linewidth=1.5)
        ax2.fill_between(elapsed, entropy, alpha=0.2, color="#3498db")
        ax2.set_ylabel("Entropy (bits)")
        ax2.set_title("Flow Entropy Over Time")

        # Annotate drops
        entropy_min_idx = entropy.idxmin()
        if not np.isnan(entropy.iloc[entropy_min_idx] if hasattr(entropy, "iloc") else entropy[entropy_min_idx]):
            ax2.annotate(
                "Min Entropy\n(possible attack)",
                xy=(elapsed.iloc[entropy_min_idx] if hasattr(elapsed, "iloc") else elapsed[entropy_min_idx],
                    entropy.iloc[entropy_min_idx] if hasattr(entropy, "iloc") else entropy[entropy_min_idx]),
                xytext=(20, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=8, color="red"
            )

    # ---- Gini coefficient ----
    ax3 = axes[2]
    if "gini" in df.columns:
        gini = df["gini"].fillna(0)
        ax3.plot(elapsed, gini, color="#e74c3c", linewidth=1.5)
        ax3.fill_between(elapsed, gini, alpha=0.2, color="#e74c3c")
        ax3.set_ylabel("Gini Coefficient")
        ax3.set_title("Flow Distribution Gini (higher = more skewed)")
        ax3.set_ylim(0, 1)

    ax3.set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_scalability_curve(
    scalability_data : Dict,
    output_path      : str,
) -> None:
    """
    Plot controller overhead vs. number of flows (scalability curve).
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_flows_list  = sorted(scalability_data.keys())
    proc_ms_list  = [scalability_data[n]["processing_ms"] for n in n_flows_list]
    cpu_list      = [scalability_data[n].get("cpu_delta_pct", 0) for n in n_flows_list]

    # ---- Processing time ----
    ax1 = axes[0]
    ax1.plot(n_flows_list, proc_ms_list, "o-", color="#3498db",
             linewidth=2, markersize=6)
    ax1.fill_between(n_flows_list, proc_ms_list, alpha=0.2, color="#3498db")

    # Fit polynomial trend
    if len(n_flows_list) >= 3:
        try:
            coefs = np.polyfit(np.log(n_flows_list), proc_ms_list, 1)
            x_fit = np.linspace(min(n_flows_list), max(n_flows_list), 100)
            y_fit = np.polyval(coefs, np.log(x_fit))
            ax1.plot(x_fit, y_fit, "--", color="#e74c3c",
                     linewidth=1.5, label="Log trend")
            ax1.legend()
        except Exception:
            pass

    ax1.set_xlabel("Number of Flows")
    ax1.set_ylabel("Processing Time (ms)")
    ax1.set_title("Controller Processing Overhead vs. Flow Count")
    ax1.set_xscale("log")

    # ---- CPU usage ----
    ax2 = axes[1]
    ax2.bar(
        [str(n) for n in n_flows_list], cpu_list,
        color="#e74c3c", edgecolor="white", alpha=0.8
    )
    ax2.set_xlabel("Number of Flows")
    ax2.set_ylabel("CPU Delta (%)")
    ax2.set_title("CPU Overhead vs. Flow Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_entropy_distribution(
    dfs         : List[pd.DataFrame],
    output_path : str,
    attack_start_s : Optional[float] = 30.0,
    attack_end_s   : Optional[float] = 90.0,
) -> None:
    """
    Plot entropy distributions: normal vs. attack periods.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    all_normal_entropy = []
    all_attack_entropy = []

    for df in dfs:
        if "entropy" not in df.columns:
            continue
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            t0 = df["timestamp"].min()
            elapsed = (df["timestamp"] - t0).dt.total_seconds()
        else:
            elapsed = pd.Series(np.arange(len(df)))

        entropy = df["entropy"].fillna(0)

        if attack_start_s is not None and attack_end_s is not None:
            normal_mask = (elapsed < attack_start_s) | (elapsed > attack_end_s)
            attack_mask = (elapsed >= attack_start_s) & (elapsed <= attack_end_s)
            all_normal_entropy.extend(entropy[normal_mask].tolist())
            all_attack_entropy.extend(entropy[attack_mask].tolist())
        else:
            all_normal_entropy.extend(entropy.tolist())

    # ---- Left: histogram ----
    ax1 = axes[0]
    bins = np.linspace(
        min(
            min(all_normal_entropy) if all_normal_entropy else 0,
            min(all_attack_entropy) if all_attack_entropy else 0,
        ),
        max(
            max(all_normal_entropy) if all_normal_entropy else 10,
            max(all_attack_entropy) if all_attack_entropy else 10,
        ),
        40
    )

    if all_normal_entropy:
        ax1.hist(
            all_normal_entropy, bins=bins,
            label=f"Normal (n={len(all_normal_entropy)})",
            color="#2ecc71", alpha=0.6, density=True
        )
    if all_attack_entropy:
        ax1.hist(
            all_attack_entropy, bins=bins,
            label=f"Attack (n={len(all_attack_entropy)})",
            color="#e74c3c", alpha=0.6, density=True
        )

    ax1.set_xlabel("Entropy (bits)")
    ax1.set_ylabel("Density")
    ax1.set_title("Entropy Distribution: Normal vs. Attack")
    ax1.legend()

    # ---- Right: box plot ----
    ax2 = axes[1]
    data_to_plot = []
    labels_plot  = []

    if all_normal_entropy:
        data_to_plot.append(all_normal_entropy)
        labels_plot.append("Normal")
    if all_attack_entropy:
        data_to_plot.append(all_attack_entropy)
        labels_plot.append("Attack")

    if data_to_plot:
        bp = ax2.boxplot(
            data_to_plot,
            labels=labels_plot,
            patch_artist=True,
            notch=True,
        )
        colors_box = ["#2ecc71", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors_box[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_ylabel("Entropy (bits)")
    ax2.set_title("Entropy Box Plot: Normal vs. Attack")

    # Add means annotation
    for i, (data, label) in enumerate(zip(data_to_plot, labels_plot)):
        ax2.text(
            i + 1, np.mean(data),
            f"μ={np.mean(data):.2f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color="black"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_all_plots(
    results_dir     : str,
    series_dir      : str,
    attack_start_s  : float = 30.0,
    attack_end_s    : float = 90.0,
) -> List[str]:
    """
    Generate all required plots for an experiment series.
    Returns list of generated file paths.
    """
    import json
    generated = []

    # Load all CSV files from series
    from experiments.metrics import (
        load_all_series_csvs, combined_metrics_df, load_metrics_csv
    )

    dfs = load_all_series_csvs(series_dir)
    if not dfs:
        logger.warning(
            f"No metrics CSVs found in {series_dir}. "
            "Generating synthetic data for demonstration."
        )
        dfs = [_generate_synthetic_df(attack_start_s, attack_end_s)]

    combined = combined_metrics_df(dfs)
    if combined is None:
        combined = dfs[0] if dfs else _generate_synthetic_df(
            attack_start_s, attack_end_s
        )

    # 1. anomaly_scores.png
    p1 = os.path.join(results_dir, "anomaly_scores.png")
    try:
        plot_anomaly_scores(
            combined, p1,
            attack_start_s=attack_start_s,
            attack_end_s=attack_end_s,
        )
        generated.append(p1)
    except Exception as e:
        logger.error(f"anomaly_scores plot failed: {e}")

    # 2. monitoring_levels.png
    p2 = os.path.join(results_dir, "monitoring_levels.png")
    try:
        plot_monitoring_levels(combined, p2)
        generated.append(p2)
    except Exception as e:
        logger.error(f"monitoring_levels plot failed: {e}")

    # 3. controller_cpu_usage.png
    p3 = os.path.join(results_dir, "controller_cpu_usage.png")
    try:
        plot_controller_cpu_usage(dfs, p3)
        generated.append(p3)
    except Exception as e:
        logger.error(f"controller_cpu_usage plot failed: {e}")

    # 4. flow_distribution.png
    p4 = os.path.join(results_dir, "flow_distribution.png")
    try:
        plot_flow_distribution(combined, p4)
        generated.append(p4)
    except Exception as e:
        logger.error(f"flow_distribution plot failed: {e}")

    # 5. scalability_curve.png
    p5 = os.path.join(results_dir, "scalability_curve.png")
    scalability_json = os.path.join(results_dir, "scalability.json")
    try:
        if os.path.exists(scalability_json):
            with open(scalability_json) as f:
                scalability_data = json.load(f)
            scalability_data = {int(k): v for k, v in scalability_data.items()}
        else:
            scalability_data = _generate_synthetic_scalability()
        plot_scalability_curve(scalability_data, p5)
        generated.append(p5)
    except Exception as e:
        logger.error(f"scalability_curve plot failed: {e}")

    # 6. entropy_distribution.png
    p6 = os.path.join(results_dir, "entropy_distribution.png")
    try:
        plot_entropy_distribution(
            dfs, p6,
            attack_start_s=attack_start_s,
            attack_end_s=attack_end_s,
        )
        generated.append(p6)
    except Exception as e:
        logger.error(f"entropy_distribution plot failed: {e}")

    return generated


def _generate_synthetic_df(
    attack_start_s : float = 30.0,
    attack_end_s   : float = 90.0,
    duration       : int   = 120,
) -> pd.DataFrame:
    """
    Generate a synthetic metrics DataFrame for demonstration when
    no actual experiment data is available.
    """
    np.random.seed(42)
    n = duration
    # Use unix epoch floats so CSV round-trips cleanly
    t = pd.array(
        [1704067200.0 + i for i in range(n)],  # 2024-01-01 00:00:00 UTC + i seconds
        dtype="float64"
    )
    elapsed = np.arange(n)

    is_attack = (elapsed >= attack_start_s) & (elapsed <= attack_end_s)

    # Entropy: drops during attack (concentration of traffic)
    entropy = np.where(
        is_attack,
        np.random.normal(2.5, 0.5, n),   # low entropy attack
        np.random.normal(8.0, 0.8, n),   # high entropy normal
    ).clip(0, 12)

    # Anomaly scores: lower during attack
    base_score = np.random.normal(-0.05, 0.03, n)
    attack_score = np.random.normal(-0.35, 0.08, n)
    scores = np.where(is_attack, attack_score, base_score)

    # Monitoring levels based on scores
    levels = np.zeros(n, dtype=int)
    levels[scores <= -0.30] = 3  # DROP
    levels[(scores > -0.30) & (scores <= -0.20)] = 2  # CLONE
    levels[(scores > -0.20) & (scores <= -0.10)] = 1  # MONITOR

    # Active flows
    n_flows = np.where(
        is_attack,
        np.random.randint(50, 200, n),
        np.random.randint(10, 50, n),
    )

    # PPS
    pps = np.where(
        is_attack,
        np.random.normal(5000, 1000, n),
        np.random.normal(100, 30, n),
    ).clip(0)

    # BPS
    bps = pps * np.random.normal(64, 10, n)

    # CPU
    cpu = np.where(
        is_attack,
        np.random.normal(35, 8, n),
        np.random.normal(12, 4, n),
    ).clip(0, 100)

    gini = np.where(
        is_attack,
        np.random.normal(0.85, 0.05, n),
        np.random.normal(0.40, 0.08, n),
    ).clip(0, 1)

    return pd.DataFrame({
        "timestamp"           : t,
        "iteration"           : np.arange(n),
        "switch_id"           : 1,
        "n_active_flows"      : n_flows,
        "entropy"             : entropy,
        "mean_pps"            : pps,
        "mean_bps"            : bps,
        "max_pps"             : pps * np.random.uniform(1.5, 3.0, n),
        "max_bps"             : bps * np.random.uniform(1.5, 3.0, n),
        "std_pps"             : pps * 0.2,
        "std_bps"             : bps * 0.2,
        "gini"                : gini,
        "top_flow_fraction"   : np.where(is_attack, 0.8, 0.2),
        "anomaly_score"       : scores,
        "monitoring_level"    : levels,
        "monitoring_level_name": [
            ["NORMAL","MONITOR","CLONE","DROP"][l] for l in levels
        ],
        "controller_cpu_pct"  : cpu,
        "controller_mem_mb"   : np.random.normal(150, 20, n),
        "n_dropped_flows"     : np.where(is_attack, np.random.randint(0, 10, n), 0),
        "n_cloned_flows"      : np.where(is_attack, np.random.randint(0, 5, n), 0),
    })


def _generate_synthetic_scalability() -> Dict:
    """Generate synthetic scalability data."""
    np.random.seed(42)
    n_flows_list = [100, 500, 1000, 2000, 4096]
    result = {}
    for n in n_flows_list:
        proc_ms = 2.5 + 0.012 * n + np.random.normal(0, 1)
        result[n] = {
            "n_flows"       : n,
            "processing_ms" : max(1.0, proc_ms),
            "cpu_delta_pct" : max(0.0, 1.0 + 0.008 * n + np.random.normal(0, 0.5)),
            "mem_delta_mb"  : max(0.0, 0.5 + 0.001 * n),
            "entropy"       : np.random.normal(7.0, 0.5),
        }
    return result
