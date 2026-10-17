#!/usr/bin/env python3
"""Generate result figures for the paper as SVGs in paper/figs/.

Both figures are built from the switch_*_metrics.csv files and
attack_window.json of each run in an experiment series. The unit is one
controller polling window per switch (the Isolation Forest scores the
switch-wide feature vector, not individual flows).

fig-confusion.svg          - window-level confusion matrix: ground truth from
                             the recorded attack window, prediction = the
                             monitoring level the controller assigned
                             (CLONE or DROP = predicted attack)
fig-score-distribution.svg - Isolation Forest decision_function score per
                             window, split by traffic phase. Each switch in
                             each run has its own thresholds (derived from
                             its benign warm-up), so scores are plotted
                             relative to them:
                               margin = (score - NORMAL) / (NORMAL - MONITOR)
                             i.e. in warm-up standard deviations below the
                             lowest warm-up score. Boundaries sit at 0, -1, -2.
fig-hybrid-detection.svg   - the complementary z-score signal: (a) z_max per
                             window by phase, (b) IF margin vs z_max, showing
                             attack windows the IF score misses (it saturates
                             on gross outliers) but z_max catches

Usage: make_result_figs.py [SERIES_DIR]   (default: latest series in results/)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from controller.policy import MonitoringLevel, PolicyConfig, Thresholds  # noqa: E402

OUT_DIR = ROOT / "paper" / "figs"
RESULTS_DIR = ROOT / "results"

# Fixed fallback thresholds (run_series passes no overrides); used for
# series recorded before per-switch thresholds were saved
POLICY = PolicyConfig()

# Levels at which the controller acts in the data plane count as a detection
ATTACK_LEVELS = {MonitoringLevel.CLONE.value, MonitoringLevel.DROP.value}

BENIGN_COLOR = "#1976D2"
ATTACK_COLOR = "#E53935"

plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",  # keep text as text in the SVG
})


def save_svg(fig, name):
    """White background everywhere, no outer margin beyond a hairline."""
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, format="svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_confusion(windows):
    actual = windows["is_attack"].to_numpy()
    predicted = windows["monitoring_level"].isin(ATTACK_LEVELS).to_numpy()
    tp = int(np.sum(actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    print(f"Confusion: TP={tp} FN={fn} FP={fp} TN={tn}  "
          f"TPR={tpr:.3f} FPR={fpr:.3f}")

    # Rows: actual (attack, benign); cols: predicted (attack, benign)
    cm = np.array([[tp, fn], [fp, tn]])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    # Cells as vector rectangles (imshow would embed a raster PNG, which
    # some SVG viewers draw with a dark frame)
    cmap = plt.get_cmap("Blues")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(cm.max(), 1))
    for i in range(2):
        for j in range(2):
            ax.add_patch(matplotlib.patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, facecolor=cmap(norm(cm[i, j])),
                edgecolor="white", lw=2))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=28, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted: Attack", "Predicted: Benign"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual: Attack", "Actual: Benign"],
                       rotation=90, va="center")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"Window-Level Detection Confusion Matrix (n={cm.sum()})")
    ax.text(0.5, -0.14, f"TPR = {tpr:.2f}    FPR = {fpr:.2f}    "
            "(predicted attack = CLONE or DROP)",
            transform=ax.transAxes, ha="center", va="top")

    save_svg(fig, "fig-confusion.svg")


def latest_series_dir():
    """Most recent results/<series>/ that has an attack_window.json."""
    candidates = sorted(p.parent.parent for p in
                        RESULTS_DIR.glob("*/run_*/attack_window.json"))
    if not candidates:
        sys.exit("No experiment series with attack_window.json found in "
                 f"{RESULTS_DIR}; run the experiment series first.")
    return candidates[-1]


def load_thresholds(metrics_csv):
    """Thresholds the controller used for this switch in this run."""
    path = metrics_csv.with_name(
        metrics_csv.name.replace("_metrics.csv", "_thresholds.json"))
    if path.exists():
        return Thresholds(**json.loads(path.read_text()))
    return Thresholds.fixed(POLICY)


def load_windows(series_dir):
    """Return a DataFrame of scored windows: anomaly_score, margin,
    monitoring_level, is_attack, threshold_source.

    A window is labelled attack if its timestamp falls inside the run's
    recorded attack window. Rows with score exactly 0.0 are dropped: the
    monitor logs 0.0 (and level NORMAL) before the model is first trained.
    """
    frames = []
    for run_dir in sorted(Path(series_dir).glob("run_*")):
        window_file = run_dir / "attack_window.json"
        csvs = sorted(run_dir.glob("ctrl_logs/**/switch_*_metrics.csv"))
        if not window_file.exists() or not csvs:
            print(f"Skipping {run_dir.name}: missing attack window or CSVs")
            continue
        window = json.loads(window_file.read_text())
        for csv in csvs:
            df = pd.read_csv(csv)
            # z_max / if_level / z_level only exist in newer series
            df = df[[c for c in ["timestamp", "anomaly_score", "z_max",
                                 "if_level", "z_level", "monitoring_level"]
                     if c in df.columns]]
            df = df[df["anomaly_score"] != 0.0].copy()
            df["is_attack"] = df["timestamp"].between(window["start"],
                                                      window["end"])
            th = load_thresholds(csv)
            step = (th.normal - th.monitor) or 1.0
            df["margin"] = (df["anomaly_score"] - th.normal) / step
            df["threshold_source"] = th.source
            print(f"  {run_dir.name}/{csv.stem}: {th.source} thresholds "
                  f"(n_warmup={th.n_warmup}) NORMAL>{th.normal:.4f} "
                  f"MONITOR>{th.monitor:.4f} CLONE>{th.clone:.4f}")
            frames.append(df)
    windows = pd.concat(frames, ignore_index=True) if frames else None
    if windows is None or windows["is_attack"].all() or \
            not windows["is_attack"].any():
        sys.exit(f"Need scored windows from both phases in {series_dir}")
    return windows


def make_score_distribution(windows):
    benign = windows.loc[~windows["is_attack"], "margin"].to_numpy()
    attack = windows.loc[windows["is_attack"], "margin"].to_numpy()

    # Per-switch boundaries all map to these on the margin scale
    t_normal, t_monitor, t_clone = 0.0, -1.0, -2.0

    # Report how well the thresholds separate the phases (no hard assertion:
    # these are measurements, not a guaranteed result)
    print(f"Benign windows: n={len(benign)}, "
          f"{np.mean(benign > t_normal):.1%} above normal threshold")
    print(f"Attack windows: n={len(attack)}, "
          f"{np.mean(attack <= t_normal):.1%} at or below normal threshold")

    scores = np.concatenate([benign, attack])
    span = max(scores.max(), t_normal) - min(scores.min(), t_clone)
    x_min = min(scores.min(), t_clone) - 0.05 * span
    x_max = max(scores.max(), t_normal) + 0.05 * span

    fig, ax = plt.subplots(figsize=(8, 5))

    # Lower score = more anomalous, so DROP is on the left
    ax.axvspan(x_min, t_clone, color="#E53935", alpha=0.08, lw=0)
    ax.axvspan(t_clone, t_monitor, color="#FB8C00", alpha=0.08, lw=0)
    ax.axvspan(t_monitor, t_normal, color="#FDD835", alpha=0.08, lw=0)
    ax.axvspan(t_normal, x_max, color="#43A047", alpha=0.08, lw=0)

    bins = np.linspace(x_min, x_max, 41)
    ax.hist(benign, bins=bins, alpha=0.7, color="#1976D2",
            edgecolor="white", label=f"Benign phase (n={len(benign)})")
    ax.hist(attack, bins=bins, alpha=0.7, color="#E53935",
            edgecolor="white", label=f"Attack phase (n={len(attack)})")

    for t, color in [(t_clone, "#C62828"), (t_monitor, "#FB8C00"),
                     (t_normal, "#9E9D24")]:
        ax.axvline(t, color=color, ls="--", lw=1.5)

    # Headroom on top for the threshold labels, a strip at the bottom
    # (below the bars) for the zone labels
    ymax = ax.get_ylim()[1] * 1.35
    strip = ymax * 0.06
    ax.set_ylim(-strip, ymax)
    ax.axhline(0, color="0.6", lw=0.8)
    # Threshold labels run up the dashed lines, so they can't collide
    for t, color, label in [(t_clone, "#C62828", "min − 2σ"),
                            (t_monitor, "#FB8C00", "min − 1σ"),
                            (t_normal, "#9E9D24", "warm-up min")]:
        ax.text(t, ymax * 0.98, f"{label} ", color=color, rotation=90,
                ha="right", va="top", fontsize=8)
    for x, label in [((x_min + t_clone) / 2, "DROP"),
                     ((t_clone + t_monitor) / 2, "CLONE"),
                     ((t_monitor + t_normal) / 2, "MONITOR"),
                     ((t_normal + x_max) / 2, "NORMAL")]:
        ax.text(x, -strip / 2, label, ha="center", va="center",
                color="0.45", fontsize=7)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Score relative to warm-up minimum, in warm-up σ  "
                  "(lower = more anomalous)")
    ax.set_ylabel("Number of polling windows")
    ax.set_title("Isolation Forest Score Distribution by Traffic Phase",
                 pad=14)
    ax.legend(loc="upper right", frameon=False)

    save_svg(fig, "fig-score-distribution.svg")


def make_hybrid_detection(windows):
    """z_max by phase, and IF margin vs z_max. Raw per-window values; the
    controller escalates on 3-window means of each signal."""
    if "z_max" not in windows or windows["z_max"].isna().all():
        print("No z_max in this series; skipping fig-hybrid-detection.svg")
        return
    w = windows.dropna(subset=["z_max"])
    zc = POLICY  # z thresholds are fixed (not derived from the warm-up)
    z_lines = [(zc.z_monitor_threshold, "#9E9D24", "MONITOR"),
               (zc.z_clone_threshold, "#FB8C00", "CLONE"),
               (zc.z_drop_threshold, "#C62828", "DROP")]
    phases = [(False, "Benign phase", BENIGN_COLOR, "o"),
              (True, "Attack phase", ATTACK_COLOR, "^")]

    # Detection = CLONE or DROP for each signal on its own
    if_hit = w["margin"] <= -1.0
    z_hit = w["z_max"] > zc.z_clone_threshold
    atk = w["is_attack"]
    print(f"Attack windows (raw values, n={atk.sum()}): "
          f"IF only={int((atk & if_hit & ~z_hit).sum())}  "
          f"z only={int((atk & ~if_hit & z_hit).sum())}  "
          f"both={int((atk & if_hit & z_hit).sum())}  "
          f"neither={int((atk & ~if_hit & ~z_hit).sum())}")
    print(f"Benign windows (raw values, n={(~atk).sum()}): "
          f"IF flags={int((~atk & if_hit).sum())}  "
          f"z flags={int((~atk & z_hit).sum())}")
    if {"if_level", "z_level"} <= set(w.columns):
        lv_if = w["if_level"].isin(ATTACK_LEVELS)
        lv_z = w["z_level"].isin(ATTACK_LEVELS)
        print(f"Attack windows (controller levels, smoothed): "
              f"IF CLONE/DROP={int((atk & lv_if).sum())}  "
              f"z CLONE/DROP={int((atk & lv_z).sum())}  "
              f"combined={int((atk & (lv_if | lv_z)).sum())}")

    z = w["z_max"].clip(lower=0.1)   # log axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) z_max distribution by phase
    bins = np.logspace(np.log10(z.min()), np.log10(z.max() * 1.05), 41)
    for is_attack, label, color, _ in phases:
        vals = z[w["is_attack"] == is_attack]
        ax1.hist(vals, bins=bins, alpha=0.7, color=color, edgecolor="white",
                 label=f"{label} (n={len(vals)})")
    ax1.set_xscale("log")
    ymax = ax1.get_ylim()[1] * 1.12
    ax1.set_ylim(0, ymax)
    # Stagger labels: the lines are close together on the log axis
    for (t, color, label), y in zip(z_lines, [0.98, 0.90, 0.82]):
        ax1.axvline(t, color=color, ls="--", lw=1.5)
        ax1.text(t, ymax * y, f" {label} (z>{t:g})", color=color,
                 ha="left", va="top", fontsize=8)
    ax1.set_xlabel("z_max: largest per-feature |z| vs. benign warm-up")
    ax1.set_ylabel("Number of polling windows")
    ax1.set_title("(a) z-score signal by traffic phase")
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.70), frameon=False)

    # (b) IF margin vs z_max
    for is_attack, label, color, marker in phases:
        sel = w["is_attack"] == is_attack
        ax2.scatter(w.loc[sel, "margin"], z[sel], s=28, marker=marker,
                    color=color, alpha=0.7, edgecolor="white", lw=0.5,
                    label=label)
    ax2.set_yscale("log")
    ax2.axvline(-1.0, color="#FB8C00", ls="--", lw=1.5)
    ax2.axhline(zc.z_clone_threshold, color="#FB8C00", ls="--", lw=1.5)
    n_z_only = int((atk & ~if_hit & z_hit).sum())
    # Upper right is empty: attack windows sit at high z but moderate IF
    # margin, benign windows at lower z
    ax2.text(0.98, 0.98, f"IF misses, z_max catches:\n"
             f"{n_z_only} of {int(atk.sum())} attack windows",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8,
             color="0.25")
    ylo, yhi = ax2.get_ylim()
    ax2.text(-1.0, np.sqrt(ylo * zc.z_clone_threshold), "IF CLONE boundary ",
             rotation=90, color="#FB8C00", ha="right", va="center",
             fontsize=8)
    ax2.text(ax2.get_xlim()[1], zc.z_clone_threshold, "z CLONE boundary ",
             color="#FB8C00", ha="right", va="bottom", fontsize=8)
    ax2.set_xlabel("IF score vs. warm-up minimum, in warm-up σ\n"
                   "(lower = more anomalous)")
    ax2.set_ylabel("z_max (log scale)")
    ax2.set_title("(b) IF score vs z-score, per window")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88),
               frameon=False)

    save_svg(fig, "fig-hybrid-detection.svg")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_series_dir()
    print(f"Reading scores from {series_dir}")
    windows = load_windows(series_dir)
    if (windows["threshold_source"] == "fallback").any():
        print("Note: some switches used the fixed fallback thresholds; "
              "their margin unit is the fixed threshold step, not warm-up σ")
    make_confusion(windows)
    make_score_distribution(windows)
    make_hybrid_detection(windows)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
