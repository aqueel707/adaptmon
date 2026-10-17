"""
analysis/score_analysis.py — per-run/per-series report on the raw
Isolation Forest scores: warm-up percentiles, per-attack-class score
distributions, and per-class ROC-AUC (flow-level and host-pair-level).

    python3 -m analysis.score_analysis results/<series or run>

This is deliberately separate from analysis/evaluate.py's confusion-matrix
scoring: it is diagnostic (why does/doesn't a class get detected), not the
headline TPR/FPR/recall numbers.

Warm-up percentiles come from controller/model.py's model.json (p1, p5,
p10, min of the model's own scores on its own warm-up samples). A run from
before controller/model.py stored p1/p10/the raw warm-up score array only
has min and p5 available -- this is reported as a gap, not backfilled or
guessed, since there is no way to recover the missing percentiles from a
run that has already finished (the raw warm-up vectors were never
persisted, only kept in memory for the run's own fit_once() call).

Per-class AUC, flow-level: label = 1 for this class's ground-truth flows,
0 for all benign flows; score = each matched flow_windows.csv row's score
(lower = more anomalous, so AUC is computed on -score). Pair-level: same
idea, but a (src,dst) pair is not identified with one flow, so rows are
labelled by time instead -- a pair_windows.csv row counts as a positive
sample for class C if its timestamp falls inside [start_ts, end_ts] of any
ground-truth flow of class C on that same (src,dst) pair (+/- one poll
interval of slack), and negative otherwise, restricted to the pairs that
ever carried class C (comparing against a pair's own quiet baseline, not
a different pair's).
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.evaluate import (  # noqa: E402
    load_ground_truth, load_flow_windows, load_pair_windows, load_model_info,
)

POLL_SLACK_S = 1.5   # >= one poll interval either side of a flow's window


def warmup_percentiles(model_info: dict) -> Dict[str, Dict]:
    out = {}
    for key, label in (("flow_model", "flow"), ("host_pair_model", "host-pair")):
        m = model_info.get(key, {})
        if not m.get("trained"):
            out[label] = {"trained": False}
            continue
        th = m["thresholds"]
        row = {"trained": True, "n_warmup": m["n_train"], "min": th["warmup_min"],
              "p5": th["warmup_p5"]}
        if "warmup_p1" in th:
            row["p1"] = th["warmup_p1"]
            row["p10"] = th["warmup_p10"]
        else:
            row["p1"] = row["p10"] = None   # run predates p1/p10 persistence
        out[label] = row
    return out


def flow_score_distribution(gt: pd.DataFrame, fw: pd.DataFrame) -> Dict[str, Dict]:
    fw = fw.copy()
    fw["k"] = list(zip(fw.src, fw.dst, fw.proto.astype(int),
                       fw.sport.astype(int), fw.dport.astype(int)))
    scored = fw[fw["scored"] == 1]
    tuple_to_class = {}
    for r in gt.itertuples():
        tuple_to_class[(r.src, r.dst, int(r.proto), int(r.sport), int(r.dport))] = \
            r.attack_class or "benign"
    scored = scored[scored["k"].isin(tuple_to_class)].copy()
    scored["cls"] = scored["k"].map(tuple_to_class)

    out = {}
    for cls, g in scored.groupby("cls"):
        out[cls] = {"n_rows": len(g), "min": float(g["score"].min()),
                   "median": float(g["score"].median()),
                   "max": float(g["score"].max())}
    return out


def flow_level_auc(gt: pd.DataFrame, fw: pd.DataFrame) -> Dict[str, Optional[float]]:
    fw = fw.copy()
    fw["k"] = list(zip(fw.src, fw.dst, fw.proto.astype(int),
                       fw.sport.astype(int), fw.dport.astype(int)))
    scored = fw[fw["scored"] == 1].copy()
    tuple_to_label = {(r.src, r.dst, int(r.proto), int(r.sport), int(r.dport)): r.label
                      for r in gt.itertuples()}
    tuple_to_class = {(r.src, r.dst, int(r.proto), int(r.sport), int(r.dport)):
                      r.attack_class for r in gt.itertuples()}
    scored = scored[scored["k"].isin(tuple_to_label)].copy()
    benign_scores = scored[scored["k"].map(tuple_to_label) == "benign"]["score"].to_numpy()

    out = {}
    classes = {c for c in tuple_to_class.values() if c}
    for cls in sorted(classes):
        cls_tuples = {k for k, c in tuple_to_class.items() if c == cls}
        pos_scores = scored[scored["k"].isin(cls_tuples)]["score"].to_numpy()
        if len(pos_scores) == 0 or len(benign_scores) == 0:
            out[cls] = None
            continue
        y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(benign_scores))])
        s = np.concatenate([pos_scores, benign_scores])
        out[cls] = float(roc_auc_score(y, -s))
    return out


def pair_score_distribution(gt: pd.DataFrame, pw: pd.DataFrame) -> Dict[str, Dict]:
    """Same idea as flow_score_distribution, but a pair isn't identified
    with one flow -- see the module docstring's note on pair-level AUC."""
    attack = gt[gt["label"] == "attack"]
    out = {}
    for cls, g in attack.groupby("attack_class"):
        pairs = set(zip(g["src"], g["dst"]))
        rows = pw[pw.apply(lambda r: (r["src"], r["dst"]) in pairs, axis=1)]
        windows = list(zip(g["start_ts"], g["end_ts"]))
        in_window = rows["ts"].apply(
            lambda ts: any(s - POLL_SLACK_S <= ts <= e + POLL_SLACK_S for s, e in windows))
        sel = rows[in_window]
        if sel.empty:
            out[cls] = {"n_rows": 0, "min": None, "median": None, "max": None}
            continue
        out[cls] = {"n_rows": len(sel), "min": float(sel["score"].min()),
                   "median": float(sel["score"].median()),
                   "max": float(sel["score"].max())}
    return out


def pair_level_auc(gt: pd.DataFrame, pw: pd.DataFrame) -> Dict[str, Optional[float]]:
    out = {}
    attack = gt[gt["label"] == "attack"]
    for cls, g in attack.groupby("attack_class"):
        pairs = set(zip(g["src"], g["dst"]))
        rows = pw[pw.apply(lambda r: (r["src"], r["dst"]) in pairs, axis=1)]
        if rows.empty:
            out[cls] = None
            continue
        windows = list(zip(g["start_ts"], g["end_ts"]))

        def _in_any_window(ts):
            return any(s - POLL_SLACK_S <= ts <= e + POLL_SLACK_S for s, e in windows)

        labels = rows["ts"].apply(_in_any_window).astype(int).to_numpy()
        scores = rows["score"].to_numpy()
        if labels.sum() == 0 or labels.sum() == len(labels):
            out[cls] = None   # need both classes present to score an AUC
            continue
        out[cls] = float(roc_auc_score(labels, -scores))
    return out


def analyze_run(run_dir: str) -> Dict:
    gt = load_ground_truth(run_dir)
    fw = load_flow_windows(run_dir)
    pw = load_pair_windows(run_dir)
    model_info = load_model_info(run_dir)
    return {
        "run_dir": run_dir,
        "warmup_percentiles": warmup_percentiles(model_info),
        "flow_score_distribution": flow_score_distribution(gt, fw),
        "pair_score_distribution": pair_score_distribution(gt, pw),
        "flow_level_auc": flow_level_auc(gt, fw),
        "pair_level_auc": pair_level_auc(gt, pw),
    }


def analyze_series(series_dir: str) -> Dict:
    run_dirs = sorted(glob.glob(os.path.join(series_dir, "run_*")))
    per_run = {os.path.basename(rd): analyze_run(rd) for rd in run_dirs}
    # Pool flow/pair rows across runs for series-level AUC and score stats
    gts, fws, pws = [], [], []
    for rd in run_dirs:
        gts.append(load_ground_truth(rd))
        fws.append(load_flow_windows(rd))
        pws.append(load_pair_windows(rd))
    gt = pd.concat(gts, ignore_index=True)
    fw = pd.concat(fws, ignore_index=True)
    pw = pd.concat(pws, ignore_index=True)
    return {
        "series_dir": series_dir,
        "n_runs": len(run_dirs),
        "per_run_warmup_percentiles": {
            name: r["warmup_percentiles"] for name, r in per_run.items()
        },
        "pooled_flow_score_distribution": flow_score_distribution(gt, fw),
        "pooled_pair_score_distribution": pair_score_distribution(gt, pw),
        "pooled_flow_level_auc": flow_level_auc(gt, fw),
        "pooled_pair_level_auc": pair_level_auc(gt, pw),
    }


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x is not None and not (
        isinstance(x, float) and np.isnan(x)) else "n/a"


def to_markdown(result: Dict) -> str:
    lines = [f"# Score analysis: `{result.get('run_dir') or result.get('series_dir')}`", ""]

    if "warmup_percentiles" in result:
        lines += ["## Warm-up score percentiles", "",
                  "| model | n | min | p1 | p5 | p10 |",
                  "|---|---|---|---|---|---|"]
        for label, m in result["warmup_percentiles"].items():
            if not m.get("trained"):
                lines.append(f"| {label} | not trained | | | | |")
                continue
            lines.append(f"| {label} | {m['n_warmup']} | {_fmt(m['min'])} | "
                         f"{_fmt(m['p1'])} | {_fmt(m['p5'])} | {_fmt(m['p10'])} |")
        if any(m.get("trained") and m.get("p1") is None
              for m in result["warmup_percentiles"].values()):
            lines.append("")
            lines.append("_p1/p10 unavailable: this run predates "
                         "controller/model.py persisting them "
                         "(warmup_scores/warmup_p1/warmup_p10 in model.json)._")
        lines.append("")

    if "per_run_warmup_percentiles" in result:
        lines += ["## Warm-up score percentiles, per run", "",
                  "| run | model | n | min | p1 | p5 | p10 |",
                  "|---|---|---|---|---|---|---|"]
        for run_name, models in result["per_run_warmup_percentiles"].items():
            for label, m in models.items():
                if not m.get("trained"):
                    lines.append(f"| {run_name} | {label} | not trained | | | | |")
                    continue
                lines.append(f"| {run_name} | {label} | {m['n_warmup']} | "
                             f"{_fmt(m['min'])} | {_fmt(m['p1'])} | {_fmt(m['p5'])} | "
                             f"{_fmt(m['p10'])} |")
        lines.append("")

    flow_dist_key = "flow_score_distribution" if "flow_score_distribution" in result \
        else "pooled_flow_score_distribution"
    pair_dist_key = "pair_score_distribution" if "pair_score_distribution" in result \
        else "pooled_pair_score_distribution"
    lines += [f"## Per-class score distribution ({flow_dist_key} / {pair_dist_key})", "",
             "| class | flow n_rows | flow min | flow median | flow max | "
             "pair n_rows | pair min | pair median | pair max |",
             "|---|---|---|---|---|---|---|---|---|"]
    classes = sorted(set(result[flow_dist_key]) | set(result[pair_dist_key]))
    for cls in classes:
        fd = result[flow_dist_key].get(cls, {})
        pd_ = result[pair_dist_key].get(cls, {})
        lines.append(f"| {cls} | {fd.get('n_rows', 0)} | {_fmt(fd.get('min'))} | "
                     f"{_fmt(fd.get('median'))} | {_fmt(fd.get('max'))} | "
                     f"{pd_.get('n_rows', 0)} | {_fmt(pd_.get('min'))} | "
                     f"{_fmt(pd_.get('median'))} | {_fmt(pd_.get('max'))} |")
    lines.append("")

    auc_flow_key = "flow_level_auc" if "flow_level_auc" in result else "pooled_flow_level_auc"
    auc_pair_key = "pair_level_auc" if "pair_level_auc" in result else "pooled_pair_level_auc"
    lines += ["## Per-class ROC-AUC (class vs. benign, by -score)", "",
             "| class | flow-level AUC | pair-level AUC |",
             "|---|---|---|"]
    classes = sorted(set(result[auc_flow_key]) | set(result[auc_pair_key]))
    for cls in classes:
        fa = result[auc_flow_key].get(cls)
        pa = result[auc_pair_key].get(cls)
        lines.append(f"| {cls} | {_fmt(fa)} | {_fmt(pa)} |")
    lines.append("")
    lines.append("_AUC 0.5 = score carries no separating information; 1.0 = perfect "
                 "separation. `n/a` means one class (positive or benign) had no "
                 "matching rows to compare against._")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    is_single_run = os.path.exists(os.path.join(args.path, "ground_truth.jsonl"))
    result = analyze_run(args.path) if is_single_run else analyze_series(args.path)

    out_path = args.out or os.path.join(args.path, "score_analysis.md")
    md = to_markdown(result)
    with open(out_path, "w") as f:
        f.write(md)
    json_path = os.path.splitext(out_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(md)
    print(f"\n-> {out_path}\n-> {json_path}")


if __name__ == "__main__":
    main()
