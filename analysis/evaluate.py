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
    verdict is the max 'level' any row reached; its trigger is the
    escalated_by of the first row at >= CLONE (what detected it -- a later
    forced DROP on a flow the IF already had at CLONE does not re-attribute
    the detection). A tuple with no matching
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

Drop metrics (runs whose flow_windows.csv has the per-slot drop_pkts
register, i.e. Run D on; None for earlier runs):
  * packets dropped per flow = sum over switches of the flow's drop_pkts,
    each switch's series summed over slot occupancies (the register restarts
    at 0 when a slot is aged and the flow re-occupies it); packets per flow
    likewise from pkt_count, max over switches (a packet dropped on the
    first hop never reaches the second). Polls are 1 Hz, so packets dropped
    after a flow's last poll before its slot was cleared are not seen: the
    per-flow sums are cross-checked against the switches' own cumulative
    drop counters ("attributed_share").
  * benign packets wrongly dropped = the same sum over benign flows (a
    source-IP drop rule for a scan also hits that host's benign flows).
  * delivered after drop = packets of a dropped flow that a switch still
    forwarded in polls wholly after the drop was installed (the rule was
    lifted by the K_DOWN dwell or by slot aging), min over its switches.
  * time to drop = first poll at which a drop applied to the flow (its own
    level DROP, or a pair with its source IP at DROP within +/- one poll of
    its lifetime) minus its first packet; per phase (one attack class in
    one cycle, the primary figure) = first drop on any of the phase's flows
    minus the phase's first packet (0 when a source drop from an earlier
    phase was still in force; counted separately). The rule is installed
    at the end of that poll, a few ms later.

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

# One attack phase = one class in one cycle; experiments/paper_run.py's
# _SportAllocator gives each (generator, cycle) its own BLOCK of sports
PHASE_SPORT_BASES = {"sqli": 56000, "syn_scan": 57000, "null_scan": 60000,
                     "rate_spike": 63000}
PHASE_SPORT_BLOCK = 20


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
           "forwarded_bytes": total("fwd_bytes"), "cloned_bytes": total("cloned_bytes"),
           "dropped": total("dropped"), "dropped_bytes": total("dropped_bytes")}
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
        # secondary, once the data plane drops: per packet entering the switch
        "dropped_packets": s("dropped"),
        "inspection_reduction_packets_incl_dropped":
            (1 - cl / (fwd + s("dropped"))) if fwd and s("dropped") is not None else None,
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


def _increments(fw: pd.DataFrame, col: str) -> pd.Series:
    """Per row: the increase of a per-slot cumulative register since the
    flow's previous row on that switch -- a drop in value means the slot was
    cleared and re-occupied, so that row's value counts from 0 again."""
    f = fw.sort_values(["key", "switch", "ts"], kind="stable")
    d = f.groupby(["key", "switch"])[col].diff()
    return pd.Series(np.where(d.isna() | (d < 0), f[col], d), index=f.index).reindex(fw.index)


def _occupancy_sum(fw: pd.DataFrame, col: str) -> pd.Series:
    """Per (key, switch): total of a per-slot cumulative register over the
    flow's slot occupancies."""
    return _increments(fw, col).groupby([fw["key"], fw["switch"]]).sum()


def _join_drops(gt: pd.DataFrame, fw: pd.DataFrame, pw: pd.DataFrame) -> pd.DataFrame:
    """Attach pkts_total, pkts_dropped and drop_ts (see the module docstring);
    NaN for runs recorded before the drop_pkts register existed."""
    if "drop_pkts" not in fw.columns:
        for c in ("pkts_total", "pkts_dropped", "drop_ts", "drop_latency_s",
                  "pkts_delivered_after_drop"):
            gt[c] = np.nan
        return gt
    pkts = _occupancy_sum(fw, "pkt_count").groupby(level="key").max().rename("pkts_total")
    dropped = _occupancy_sum(fw, "drop_pkts").groupby(level="key").sum().rename("pkts_dropped")
    flow_drop = fw[fw["level"] >= DROP].groupby("key")["ts"].min().rename("flow_drop_ts")
    gt = gt.merge(pkts, left_on="key", right_index=True, how="left") \
           .merge(dropped, left_on="key", right_index=True, how="left") \
           .merge(flow_drop, left_on="key", right_index=True, how="left")
    gt["pkts_dropped"] = gt["pkts_dropped"].fillna(0).astype(int)

    src_drops = {src: np.sort(g["ts"].to_numpy())
                 for src, g in pw[pw["level"] >= DROP].groupby("src")}

    def _src_drop(row) -> float:
        ts = src_drops.get(row["src"])
        if ts is None:
            return np.nan
        hit = ts[(ts >= row["start_ts"] - PAIR_MATCH_SLACK_S)
                 & (ts <= row["end_ts"] + PAIR_MATCH_SLACK_S)]
        return float(hit[0]) if len(hit) else np.nan

    gt["drop_ts"] = np.fmin(gt["flow_drop_ts"], gt.apply(_src_drop, axis=1))
    # a source drop already in force when the flow started drops it at once
    gt["drop_latency_s"] = np.maximum(0.0, gt["drop_ts"] - gt["start_ts"])

    # Delivered after the drop: packets a switch still forwarded in polls
    # wholly after the drop was installed (rows from drop_ts + 1.5 polls on),
    # i.e. once the dwell or slot aging lifted the rule; min over the
    # switches that saw the flow, since a delivered packet crossed them all
    fwd_inc = _increments(fw, "pkt_count") - _increments(fw, "drop_pkts")
    dts = fw["key"].map(gt.set_index("key")["drop_ts"])
    after = fwd_inc[fw["ts"] > dts + PAIR_MATCH_SLACK_S]
    per_sw = after.groupby([fw["key"], fw["switch"]]).sum().clip(lower=0)
    seen = fw[fw["key"].isin(per_sw.index.get_level_values(0))].groupby("key")["switch"].nunique()
    n_sw = per_sw.groupby(level="key").size()
    # a switch with no row after the drop forwarded nothing more of the flow
    leak = per_sw.groupby(level="key").min().where(n_sw >= seen.reindex(n_sw.index), 0)
    gt["pkts_delivered_after_drop"] = np.where(
        gt["drop_ts"].notna(), gt["key"].map(leak).fillna(0), np.nan)
    return gt.drop(columns="flow_drop_ts")


def _phase_id(row) -> Optional[str]:
    base = PHASE_SPORT_BASES.get(row["attack_class"])
    if row["label"] != "attack" or base is None:
        return None
    return f"{row['run']}|{row['attack_class']}|{(int(row['sport']) - base) // PHASE_SPORT_BLOCK}"


def join_run(run_dir: str) -> pd.DataFrame:
    """
    One row per ground-truth flow, with its detection verdict, trigger,
    detection time and DPI confirmation time attached (see module
    docstring for the definitions).
    """
    gt = load_ground_truth(run_dir)
    gt["run"] = os.path.basename(os.path.normpath(run_dir))
    gt["phase"] = gt.apply(_phase_id, axis=1)
    fw = load_flow_windows(run_dir)
    pw = load_pair_windows(run_dir)
    confirmed = load_confirmed_alerts(run_dir)

    # Flow-level: the level this flow peaked at; the time and trigger of its
    # first row at >= CLONE
    flow_peak = fw.groupby("key")["level"].max().rename("flow_max_level")
    first_clone = (fw[fw["level"] >= CLONE].sort_values("ts", kind="stable")
                   .drop_duplicates("key")[["key", "ts", "escalated_by"]]
                   .rename(columns={"ts": "flow_detect_ts", "escalated_by": "flow_trigger"}))
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
    gt = _join_drops(gt, fw, pw)

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


def _has_drop_data(df: pd.DataFrame) -> bool:
    return "pkts_total" in df and df["pkts_total"].notna().any()


def _packet_drops(g: pd.DataFrame) -> Optional[Dict]:
    """Packets dropped / seen over a set of flows (None before Run D)."""
    if not _has_drop_data(g):
        return None
    total, dropped = int(g["pkts_total"].fillna(0).sum()), int(g["pkts_dropped"].sum())
    after = g["pkts_delivered_after_drop"]
    return {"flows_with_dropped_packets": int((g["pkts_dropped"] > 0).sum()),
            "packets_dropped": dropped, "packets_seen": total,
            "dropped_share": dropped / total if total else float("nan"),
            "packets_delivered_after_drop": int(after.fillna(0).sum()),
            "flows_with_delivery_after_drop": int((after > 0).sum())}


def _time_to_drop(g: pd.DataFrame) -> Optional[Dict]:
    """Per phase (primary) and per flow (secondary); see the module docstring."""
    if not _has_drop_data(g):
        return None
    ph = g[g["phase"].notna()].groupby("phase").agg(
        start=("start_ts", "min"), drop=("drop_ts", "min"))
    # a source drop still in force from an earlier phase of the same host
    # (e.g. a null scan's, when the syn scan follows) drops it at once
    return {"n_phases": int(len(ph)), "n_phases_dropped": int(ph["drop"].notna().sum()),
            "n_phases_drop_in_force_at_start": int((ph["drop"] < ph["start"]).sum()),
            "per_phase_s": _dist(np.maximum(0.0, ph["drop"] - ph["start"])),
            "n_flows_dropped": int(g["drop_ts"].notna().sum()),
            "per_flow_s": _dist(g["drop_latency_s"])}


def evaluate(df: pd.DataFrame) -> Dict:
    out = {"n_flows": len(df), "overall": _confusion(df)}

    benign = df[df["label"] == "benign"]
    out["benign"] = {
        "n": len(benign),
        "n_escalated_clone_or_above": int(benign["predicted_attack"].sum()),
        "false_positives_by_trigger": _by_trigger(benign),
        "n_dropped": int(benign["blocked"].sum()),
        "drop_rate": float(benign["blocked"].mean()) if len(benign) else float("nan"),
        "packet_drops": _packet_drops(benign),
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
            "packet_drops": _packet_drops(g),
            "time_to_drop": _time_to_drop(g),
        }
    out["per_attack_class"] = per_class
    attacks = df[df["label"] == "attack"]
    out["attack_packet_drops"] = _packet_drops(attacks)
    out["attack_time_to_drop"] = _time_to_drop(attacks)
    out["detect_latency_s_by_trigger"] = {
        t: _dist(attacks.loc[attacks["trigger"] == t, "detect_latency_s"])
        for t in TRIGGERS if (attacks["trigger"] == t).any()}
    return out


def drop_accounting(df: pd.DataFrame, totals: List[Dict]) -> Optional[Dict]:
    """The switches' own cumulative drop counters vs the per-flow sums."""
    if not _has_drop_data(df) or any(t.get("dropped") is None for t in totals):
        return None
    sw = sum(t["dropped"] for t in totals)
    att = int(df.loc[df["label"] == "attack", "pkts_dropped"].sum())
    ben = int(df.loc[df["label"] == "benign", "pkts_dropped"].sum())
    return {"switch_dropped_packets": sw, "attributed_attack": att, "attributed_benign": ben,
            "attributed_share": (att + ben) / sw if sw else float("nan"),
            "switch_dropped_bytes": sum(t["dropped_bytes"] or 0 for t in totals)}


def _run_level(run_dir: str) -> Dict:
    w = load_windows(run_dir)
    return {"totals": run_totals(w), "capacity": capacity(w, load_dpi_stats(run_dir)),
            "resources": resources(w)}


def evaluate_run(run_dir: str) -> Dict:
    df = join_run(run_dir)
    rl = _run_level(run_dir)
    result = evaluate(df)
    result["inspection"] = inspection([rl["totals"]])
    result["drop_accounting"] = drop_accounting(df, [rl["totals"]])
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
                         "drop_accounting": drop_accounting(df, [rl["totals"]]),
                         "capacity": rl["capacity"], "resources": rl["resources"]}
    pooled = pd.concat(dfs, ignore_index=True)
    result = evaluate(pooled)
    result["inspection"] = inspection([rl["totals"] for rl in run_levels])
    result["drop_accounting"] = drop_accounting(pooled, [rl["totals"] for rl in run_levels])
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
    if result.get("drop_accounting"):
        da, bd = result["drop_accounting"], b["packet_drops"]
        print(f"drops: switches {da['switch_dropped_packets']} pkts; attack flows "
              f"{da['attributed_attack']}, benign flows {da['attributed_benign']} "
              f"({bd['flows_with_dropped_packets']} benign flows hit); "
              f"attributed {da['attributed_share']:.1%}")
        for cls, m in result["per_attack_class"].items():
            pd_, td = m["packet_drops"], m["time_to_drop"]
            ph = td["per_phase_s"]
            ph_s = f", median {ph['median']:.2f}s" if ph.get("n") else ""
            print(f"  {cls}: dropped {pd_['packets_dropped']}/{pd_['packets_seen']} pkts "
                  f"({pd_['dropped_share']:.1%}), delivered after drop "
                  f"{pd_['packets_delivered_after_drop']}; phases dropped "
                  f"{td['n_phases_dropped']}/{td['n_phases']}{ph_s} to drop")
    cap = result["capacity"]
    print(f"capacity: max {cap['max_pps']:.0f} pkt/s/switch, "
          f"{cap['windows_over_budget']} windows > {cap['budget_pps']}, "
          f"DPI kernel drops {cap['dpi_kernel_drops']}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
