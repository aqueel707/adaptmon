"""
analysis/evaluate.py — Stage 6: score a completed experiments/paper_run.py
run (or a whole series of them) against its own
traffic.ground_truth.GroundTruthRecorder output.

    python3 -m analysis.evaluate results/paper_<ts>/run_01     # one run
    python3 -m analysis.evaluate results/paper_<ts>            # pools all run_*/

Methodology
-----------
experiments/paper_run.py guarantees every flow in a run has a globally
unique 5-tuple (src, dst, proto, sport, dport) -- see its module docstring
-- so ground truth joins to controller output by exact 5-tuple, never by
inferring identity from timestamps:

  * flow_windows.csv rows matching a tuple -> that flow's own per-flow
    verdict is the max 'level' any row reached. A tuple with no matching
    rows was never active during a 1 Hz register poll (too short-lived --
    this is itself a real property of the pipeline, not excluded from the
    confusion matrix).
  * a host pair escalation (controller/monitor.py) only ever touches the
    pair's own policy state, so a flow also counts as detected if its
    (src, dst) pair was at >= CLONE while the flow was active (+/- one poll
    interval of slack).
  * "predicted attack"  = detected either way (Decision A's CLONE level).
  * "blocked (DROP)"    = the flow's own level reached DROP -- under
    Decision A only via a DPI confirmation of a flow already at CLONE, so a
    nonzero *benign* DROP rate would mean that invariant broke.
  * "trigger" = what escalated it: IF-flow / IF-pair / DPI-head /
    DPI-sample (controller/monitor.py's escalated_by), DPI-drop (a second
    confirmation on a flow already at CLONE), DPI-level (a confirmation the
    data plane cloned for a level the controller no longer held -- a race).
  * alerts.jsonl's "confirmed" entries (controller/dpi.py) are reported
    separately as DPI's own verdict, keyed by attack_class.

Reported, overall and per attack_class: confusion matrix and TPR/FPR with
Clopper-Pearson 95% CIs, recall split by trigger (each with its own CI),
benign false positives split by trigger, DPI confirmation recall, benign
DROP rate, detection latency, plus run-level inspection reduction,
data-plane load (capacity) and controller/DPI resource use.

Data-plane counters in windows.csv are CUMULATIVE since the controller
reset them at start (controller/monitor.py) -- a run's totals are its last
row; series totals are the sum of each run's last row.

Inspection reduction = 1 - cloned / forwarded (packets; bytes likewise),
where cloned counts every copy sent to a CPU port (flow-head, sampling,
level and host-pair clones) and forwarded counts originals sent out, both
summed over every switch hop.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from controller.policy import MonitoringLevel  # noqa: E402

CLONE = MonitoringLevel.CLONE.value
DROP = MonitoringLevel.DROP.value

# Slack matches experiments/paper_run.py's PHASE_GAP_S / one poll interval
PAIR_MATCH_SLACK_S = 1.5
# Decision D: originals + clones per switch should stay under this
CAPACITY_BUDGET_PPS = 2500

TRIGGERS = ["IF-flow", "IF-pair", "DPI-head", "DPI-sample", "DPI-drop", "DPI-level",
            "unattributed"]
CLONE_REASON_COLS = ["head", "sample", "level", "hostpair"]


def _key(src, dst, proto, sport, dport) -> str:
    return f"{src}|{dst}|{int(proto)}|{int(sport)}|{int(dport)}"


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact binomial CI for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def _rate(k: int, n: int) -> Dict:
    lo, hi = clopper_pearson(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "ci95": [lo, hi]}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_ground_truth(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "ground_truth.jsonl")
    rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame(rows)
    df["key"] = df.apply(lambda r: _key(r.src, r.dst, r.proto, r.sport, r.dport), axis=1)
    if df["key"].duplicated().any():
        dupes = df[df["key"].duplicated(keep=False)]
        raise ValueError(
            f"{path}: {dupes['key'].nunique()} 5-tuple(s) reused within one run "
            f"(ground truth cannot be joined unambiguously): "
            f"{dupes['key'].unique()[:5].tolist()}"
        )
    return df


def load_flow_windows(run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(run_dir, "ctrl", "flow_windows.csv"))
    df["key"] = df.apply(lambda r: _key(r.src, r.dst, r.proto, r.sport, r.dport), axis=1)
    return df


def load_confirmed_alerts(run_dir: str) -> pd.DataFrame:
    rows = []
    for line in open(os.path.join(run_dir, "ctrl", "alerts.jsonl")):
        a = json.loads(line)
        if a.get("type") != "confirmed":
            continue
        ft = a["five_tuple"]
        rows.append({"key": _key(ft["src"], ft["dst"], ft["proto"], ft["sport"], ft["dport"]),
                    "attack_class": a["attack_class"], "ts": a["ts"],
                    "clone_reason": a.get("clone_reason")})
    return pd.DataFrame(rows, columns=["key", "attack_class", "ts", "clone_reason"])


def load_windows(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "ctrl", "windows.csv"))


def load_pair_windows(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "ctrl", "pair_windows.csv"))


def load_model_info(run_dir: str) -> dict:
    return json.load(open(os.path.join(run_dir, "ctrl", "model.json")))


def load_dpi_stats(run_dir: str) -> dict:
    path = os.path.join(run_dir, "ctrl", "dpi_stats.json")
    return json.load(open(path)) if os.path.exists(path) else {}


# --------------------------------------------------------------------------
# Run-level: inspection, capacity, resources
# --------------------------------------------------------------------------

def _switch_ids(windows_df: pd.DataFrame) -> List[int]:
    return sorted(int(c[1:].split("_")[0]) for c in windows_df.columns
                  if c.startswith("s") and c.endswith("_fwd"))


def run_totals(windows_df: pd.DataFrame) -> Dict:
    """Final cumulative data-plane counters of one run, summed over switches.
    Byte / per-reason fields are None for runs recorded before those
    counters existed."""
    last = windows_df.iloc[-1]
    sids = _switch_ids(windows_df)

    def total(field):
        cols = [f"s{s}_{field}" for s in sids]
        if not all(c in windows_df.columns for c in cols):
            return None
        return int(sum(last[c] for c in cols))

    out = {"forwarded": total("fwd"), "cloned": total("cloned"),
           "forwarded_bytes": total("fwd_bytes"), "cloned_bytes": total("cloned_bytes")}
    for r in CLONE_REASON_COLS:
        out[f"clone_{r}"] = total(f"clone_{r}")
        out[f"clone_{r}_bytes"] = total(f"clone_{r}_bytes")
    return out


def inspection(totals: List[Dict]) -> Dict:
    """1 - cloned / forwarded, packets and bytes, over the given runs."""
    def s(field):
        vals = [t[field] for t in totals]
        return None if any(v is None for v in vals) else sum(vals)

    fwd, cl, fwd_b, cl_b = s("forwarded"), s("cloned"), s("forwarded_bytes"), s("cloned_bytes")
    out = {
        "forwarded_packets": fwd, "cloned_packets": cl,
        "cloned_per_forwarded": cl / fwd if fwd else float("nan"),
        "inspection_reduction_packets": 1 - cl / fwd if fwd else float("nan"),
        "forwarded_bytes": fwd_b, "cloned_bytes": cl_b,
        "inspection_reduction_bytes": (1 - cl_b / fwd_b) if fwd_b else None,
        "per_run_reduction_packets": [1 - t["cloned"] / t["forwarded"] for t in totals
                                      if t["forwarded"]],
    }
    by_reason = {}
    for r in CLONE_REASON_COLS:
        p, b = s(f"clone_{r}"), s(f"clone_{r}_bytes")
        if p is not None:
            by_reason[r] = {"packets": p, "bytes": b,
                            "share_of_forwarded_packets": p / fwd if fwd else float("nan"),
                            "share_of_forwarded_bytes": b / fwd_b if fwd_b else None}
    out["clones_by_reason"] = by_reason or None
    return out


def capacity(windows_df: pd.DataFrame, dpi_stats: dict) -> Dict:
    """Per-switch offered load (originals + clones per second, from the
    per-window deltas of the cumulative counters) and DPI socket drops."""
    dt = windows_df["ts"].diff()
    per_switch = {}
    for s in _switch_ids(windows_df):
        pps = (windows_df[f"s{s}_fwd"].diff() + windows_df[f"s{s}_cloned"].diff()) / dt
        pps = pps.replace([np.inf, -np.inf], np.nan).dropna()
        per_switch[s] = {
            "max_pps": float(pps.max()) if len(pps) else float("nan"),
            "p99_pps": float(np.percentile(pps, 99)) if len(pps) else float("nan"),
            "windows_over_budget": int((pps > CAPACITY_BUDGET_PPS).sum()),
            "n_windows": int(len(pps)),
        }
    socks = dpi_stats.get("sockets", {})
    return {
        "budget_pps": CAPACITY_BUDGET_PPS,
        "per_switch": per_switch,
        "max_pps": max((v["max_pps"] for v in per_switch.values()), default=float("nan")),
        "windows_over_budget": sum(v["windows_over_budget"] for v in per_switch.values()),
        "dpi_kernel_received": sum(v.get("kernel_received", 0) for v in socks.values()),
        "dpi_kernel_drops": sum(v.get("kernel_drops", 0) for v in socks.values()),
    }


def _dist(x) -> Dict:
    x = pd.Series(x).dropna()
    if x.empty:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def resources(windows_df: pd.DataFrame) -> Dict:
    """Controller poll time and controller/DPI CPU/RSS over scored windows
    (after training -- the steady state the detector runs in)."""
    w = windows_df[windows_df["phase"] == "scored"] if "phase" in windows_df else windows_df
    return {
        "poll_total_ms": _dist(w["t_total_ms"]),
        "poll_read_ms": _dist(w["t_read_ms"]),
        "controller_cpu_pct": _dist(w["cpu_pct"]),
        "controller_rss_mb": _dist(w["rss_mb"]),
        "dpi_cpu_pct": _dist(w["dpi_cpu_pct"]) if "dpi_cpu_pct" in w else {"n": 0},
        "dpi_rss_mb": _dist(w["dpi_rss_mb"]) if "dpi_rss_mb" in w else {"n": 0},
    }


def _pool_dists(dists: List[Dict]) -> Dict:
    """Series-level resource summary: worst p95/max over runs, mean of means."""
    dists = [d for d in dists if d.get("n")]
    if not dists:
        return {"n": 0}
    return {"n": sum(d["n"] for d in dists),
            "mean": float(np.mean([d["mean"] for d in dists])),
            "p95_max_over_runs": max(d["p95"] for d in dists),
            "max": max(d["max"] for d in dists)}


# --------------------------------------------------------------------------
# Flow-level join
# --------------------------------------------------------------------------

# escalated_by tags written by runs recorded before the IF-*/DPI-* names
LEGACY_TRIGGERS = {"flow": "IF-flow", "host_pair": "IF-pair"}


def _trigger_bucket(t: str) -> str:
    t = LEGACY_TRIGGERS.get(t, t)
    if t in ("IF-flow", "IF-pair", "DPI-head", "DPI-sample", "DPI-level"):
        return t
    if t.startswith("dpi_confirmed:"):
        return "DPI-drop"
    return "unattributed"


def join_run(run_dir: str) -> pd.DataFrame:
    """
    One row per ground-truth flow, with its detection verdict, trigger,
    detection time and DPI confirmation time attached (see module
    docstring for the definitions).
    """
    gt = load_ground_truth(run_dir)
    fw = load_flow_windows(run_dir)
    pw = load_pair_windows(run_dir)
    confirmed = load_confirmed_alerts(run_dir)

    # Flow-level: level and trigger at the row where this flow peaked, and
    # the first time it was at >= CLONE
    peak_idx = fw.groupby("key")["level"].idxmax()
    flow_peak = fw.loc[peak_idx, ["key", "level", "escalated_by"]].rename(
        columns={"level": "flow_max_level", "escalated_by": "flow_trigger"})
    first_clone = fw[fw["level"] >= CLONE].groupby("key")["ts"].min().rename("flow_detect_ts")
    gt = gt.merge(flow_peak, on="key", how="left").merge(first_clone, on="key", how="left")
    gt["observed"] = gt["flow_max_level"].notna()
    gt["flow_max_level"] = gt["flow_max_level"].fillna(MonitoringLevel.NORMAL.value).astype(int)
    gt["flow_trigger"] = gt["flow_trigger"].fillna("")

    # Pair-level: the first time this flow's (src, dst) pair was at >= CLONE
    # while the flow itself was active. Few distinct pairs exist in this
    # topology, so group pair_windows once instead of filtering per row.
    pw_clone = pw[pw["level"] >= CLONE].sort_values("ts")
    pair_clone_events = {
        pair: list(zip(g["ts"], g["escalated_by"].fillna("")))
        for pair, g in pw_clone.groupby(["src", "dst"])
    }

    def _pair_hit(row) -> pd.Series:
        lo, hi = row["start_ts"] - PAIR_MATCH_SLACK_S, row["end_ts"] + PAIR_MATCH_SLACK_S
        for ts, trigger in pair_clone_events.get((row["src"], row["dst"]), []):
            if lo <= ts <= hi:
                return pd.Series({"pair_detected": True, "pair_trigger": trigger,
                                  "pair_detect_ts": ts})
        return pd.Series({"pair_detected": False, "pair_trigger": "",
                          "pair_detect_ts": np.nan})

    gt = pd.concat([gt, gt.apply(_pair_hit, axis=1)], axis=1)

    flow_det = gt["flow_max_level"] >= CLONE
    gt["predicted_attack"] = flow_det | gt["pair_detected"]
    gt["blocked"] = gt["flow_max_level"] >= DROP
    raw_trigger = np.where(flow_det, gt["flow_trigger"],
                           np.where(gt["pair_detected"], gt["pair_trigger"], ""))
    gt["trigger"] = [(_trigger_bucket(t) if d else "none")
                     for t, d in zip(raw_trigger, gt["predicted_attack"])]
    detect_ts = np.where(flow_det, gt["flow_detect_ts"], gt["pair_detect_ts"])
    # A pair already at CLONE before this flow started has caught it at
    # once, not at a negative latency
    gt["detect_latency_s"] = np.where(gt["predicted_attack"],
                                      np.maximum(0.0, detect_ts - gt["start_ts"]), np.nan)

    first_conf = confirmed.groupby(["key", "attack_class"])["ts"].min() \
        if not confirmed.empty else pd.Series(dtype=float)
    conf_ts = [first_conf.get((k, c), np.nan) if c else np.nan
               for k, c in zip(gt["key"], gt["attack_class"])]
    gt["dpi_confirmed"] = pd.notna(conf_ts)
    gt["dpi_latency_s"] = np.array(conf_ts, dtype=float) - gt["start_ts"]
    return gt


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _confusion(df: pd.DataFrame) -> Dict:
    actual = df["label"] == "attack"
    predicted = df["predicted_attack"]
    tp = int((actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    tpr, fpr = _rate(tp, tp + fn), _rate(fp, fp + tn)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tpr": tpr["rate"], "tpr_ci95": tpr["ci95"],
            "fpr": fpr["rate"], "fpr_ci95": fpr["ci95"]}


def _by_trigger(g: pd.DataFrame) -> Dict:
    n = len(g)
    return {t: _rate(int((g["trigger"] == t).sum()), n) for t in TRIGGERS
            if (g["trigger"] == t).any()}


def evaluate(df: pd.DataFrame) -> Dict:
    out = {"n_flows": len(df), "overall": _confusion(df)}

    benign = df[df["label"] == "benign"]
    out["benign"] = {
        "n": len(benign),
        "n_escalated_clone_or_above": int(benign["predicted_attack"].sum()),
        "false_positives_by_trigger": _by_trigger(benign),
        "n_dropped": int(benign["blocked"].sum()),
        "drop_rate": float(benign["blocked"].mean()) if len(benign) else float("nan"),
    }

    per_class = {}
    for cls, g in df[df["label"] == "attack"].groupby("attack_class"):
        n = len(g)
        detected = int(g["predicted_attack"].sum())
        confirmed = int(g["dpi_confirmed"].sum())
        det = _rate(detected, n)
        conf = _rate(confirmed, n)
        per_class[cls] = {
            "n": n,
            "n_observed": int(g["observed"].sum()),
            "detected_clone_or_above": detected,
            "detected_recall": det["rate"],
            "detected_recall_ci95": det["ci95"],
            "recall_by_trigger": _by_trigger(g),
            "detect_latency_s": _dist(g["detect_latency_s"]),
            "dpi_confirmed": confirmed,
            "dpi_confirmed_recall": conf["rate"],
            "dpi_confirmed_recall_ci95": conf["ci95"],
            "dpi_latency_s": _dist(g["dpi_latency_s"]),
            "dropped": int(g["blocked"].sum()),
        }
    out["per_attack_class"] = per_class
    attacks = df[df["label"] == "attack"]
    out["detect_latency_s_by_trigger"] = {
        t: _dist(attacks.loc[attacks["trigger"] == t, "detect_latency_s"])
        for t in TRIGGERS if (attacks["trigger"] == t).any()}
    return out


def _run_level(run_dir: str) -> Dict:
    w = load_windows(run_dir)
    return {"totals": run_totals(w), "capacity": capacity(w, load_dpi_stats(run_dir)),
            "resources": resources(w)}


def evaluate_run(run_dir: str) -> Dict:
    df = join_run(run_dir)
    rl = _run_level(run_dir)
    result = evaluate(df)
    result["inspection"] = inspection([rl["totals"]])
    result["capacity"] = rl["capacity"]
    result["resources"] = rl["resources"]
    result["run_dir"] = run_dir
    return result


def evaluate_series(series_dir: str) -> Dict:
    run_dirs = sorted(glob.glob(os.path.join(series_dir, "run_*")))
    if not run_dirs:
        raise SystemExit(f"No run_* directories under {series_dir}")
    dfs, run_levels, per_run = [], [], {}
    for rd in run_dirs:
        name = os.path.basename(rd)
        try:
            df = join_run(rd)
            rl = _run_level(rd)
        except Exception as e:
            per_run[name] = {"error": str(e)}
            continue
        dfs.append(df)
        run_levels.append(rl)
        per_run[name] = {**evaluate(df), "inspection": inspection([rl["totals"]]),
                         "capacity": rl["capacity"], "resources": rl["resources"]}
    result = evaluate(pd.concat(dfs, ignore_index=True))
    result["inspection"] = inspection([rl["totals"] for rl in run_levels])
    result["capacity"] = {
        "budget_pps": CAPACITY_BUDGET_PPS,
        "max_pps": max(rl["capacity"]["max_pps"] for rl in run_levels),
        "windows_over_budget": sum(rl["capacity"]["windows_over_budget"] for rl in run_levels),
        "dpi_kernel_drops": sum(rl["capacity"]["dpi_kernel_drops"] for rl in run_levels),
        "dpi_kernel_received": sum(rl["capacity"]["dpi_kernel_received"] for rl in run_levels),
    }
    result["resources"] = {k: _pool_dists([rl["resources"][k] for rl in run_levels])
                           for k in run_levels[0]["resources"]}
    result["series_dir"] = series_dir
    result["n_runs"] = len(run_dirs)
    result["per_run"] = per_run
    return result


def _fmt_rate(r: Dict) -> str:
    return f"{r['k']}/{r['n']}={r['rate']:.2f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="A run_NN directory or a series directory containing them")
    p.add_argument("--out", default=None, help="Write metrics JSON here "
                   "(default <path>/metrics.json)")
    args = p.parse_args()

    is_single_run = os.path.exists(os.path.join(args.path, "ground_truth.jsonl"))
    result = evaluate_run(args.path) if is_single_run else evaluate_series(args.path)

    out_path = args.out or os.path.join(args.path, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    o = result["overall"]
    print(f"n_flows={result['n_flows']}  "
          f"TPR={o['tpr']:.3f} [{o['tpr_ci95'][0]:.3f},{o['tpr_ci95'][1]:.3f}]  "
          f"FPR={o['fpr']:.3f} [{o['fpr_ci95'][0]:.3f},{o['fpr_ci95'][1]:.3f}]")
    b = result["benign"]
    fp_trig = ", ".join(f"{t}={v['k']}" for t, v in b["false_positives_by_trigger"].items()
                        if t != "none")
    print(f"benign: n={b['n']} escalated(>=CLONE)={b['n_escalated_clone_or_above']} "
          f"[{fp_trig}] dropped={b['n_dropped']}")
    for cls, m in result["per_attack_class"].items():
        trig = ", ".join(f"{t} {_fmt_rate(v)}" for t, v in m["recall_by_trigger"].items()
                         if t != "none")
        lat = m["detect_latency_s"]
        lat_s = f" latency median {lat['median']:.2f}s" if lat.get("n") else ""
        print(f"{cls}: n={m['n']} detected={m['detected_clone_or_above']} "
              f"({m['detected_recall']:.2f}) [{trig}]{lat_s} "
              f"dpi_confirmed={m['dpi_confirmed']}")
    insp = result["inspection"]
    byte_s = (f", bytes {insp['inspection_reduction_bytes']:.1%}"
              if insp["inspection_reduction_bytes"] is not None else "")
    print(f"inspection reduction: packets {insp['inspection_reduction_packets']:.1%}{byte_s} "
          f"({insp['cloned_packets']} cloned / {insp['forwarded_packets']} forwarded)")
    cap = result["capacity"]
    print(f"capacity: max {cap['max_pps']:.0f} pkt/s/switch, "
          f"{cap['windows_over_budget']} windows > {cap['budget_pps']}, "
          f"DPI kernel drops {cap['dpi_kernel_drops']}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
