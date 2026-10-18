#!/usr/bin/env python3
"""
analysis/paper_runC.py — every paper number and figure from one 5-run
series (by default Run C, results/paper_run_C), plus the separate
latency-only benchmark (results/latency_bench_runC) for per-poll latency at
fixed active-flow counts, which the series itself (<= 80 active flows) does
not reach.

    python3 -m analysis.paper_runC                       # Run C -> paper/runC
    python3 -m analysis.paper_runD                       # Run D -> paper/runD
    python3 -m analysis.paper_runC --series results/<s> --out paper/<x> --name "Run X"

Writes <out>/PAPER_NUMBERS.md (each number with its source file),
<out>/numbers.json, and fig_*.pdf + fig_*.png. What a DPI confirmation does
(CLONE in Run C, DROP from Run D on) is read from each run's model.json
(data_plane.dpi_confirmation_action); for DROP series, item 10 reports the
drops.
"""

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis import evaluate as ev                               # noqa: E402
from analysis.replay import replay_levels, estimate_clones, CLONE, DROP  # noqa: E402

SERIES = "results/paper_run_C"   # set by configure()
RUN = "Run C"
DPI_ACTION = "CLONE"             # set in main() from model.json
BENCH_FALLBACK = "results/latency_bench_runC"


def pick_bench():
    """$ADAPTMON_BENCH, else the newest complete results/latency_bench_idle_* run
    (scripts/run_latency_idle.sh) not marked superseded in
    experiments/seeds_used.json, else the contended-host run."""
    if os.environ.get("ADAPTMON_BENCH"):
        return os.environ["ADAPTMON_BENCH"]
    reg = json.load(open(os.path.join(ROOT, "experiments", "seeds_used.json")))["seeds"]
    superseded = {v["series"] for v in reg.values() if v.get("superseded_reason")}
    idle = [d for d in sorted(glob.glob(os.path.join(ROOT, "results", "latency_bench_idle_*")))
            if os.path.exists(os.path.join(d, "bench_config.json"))
            and os.path.exists(os.path.join(d, "ctrl", "windows.csv"))
            and rel(d) not in superseded]
    return rel(idle[-1]) if idle else BENCH_FALLBACK


BENCH = BENCH_FALLBACK    # set in main()
OUT = "paper/runC"


def configure(series, out, name):
    global SERIES, OUT, RUN
    SERIES, OUT, RUN = series, out, name
LEVELS = ["NORMAL", "MONITOR", "CLONE", "DROP"]
CLASSES = ["sqli", "null_scan", "syn_scan", "rate_spike"]
TRIGGERS = ["IF-flow", "IF-pair", "DPI-head", "DPI-sample"]

# Reference palette (dataviz skill, references/palette.md), light mode.
# Categorical slots in fixed order; ordinal levels on the blue ramp.
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LEVEL_RAMP = {"MONITOR": "#86b6ef", "CLONE": "#2a78d6", "DROP": "#104281"}
NEUTRAL = "#a3a29c"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False, "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

NUM = {}    # everything also goes to numbers.json


def rel(p):
    return os.path.relpath(p, ROOT)


def code_ref(path, pattern):
    """'path:line' of the first line matching pattern (cites a constant)."""
    if re.fullmatch(r"\^[A-Z_0-9]+", pattern):
        pattern += r"\s*="       # a constant's definition, not a mention of it
    for i, line in enumerate(open(os.path.join(ROOT, path)), 1):
        if re.search(pattern, line):
            return f"`{path}:{i}`"
    raise ValueError(f"{pattern!r} not found in {path}")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pct(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}%"


# --------------------------------------------------------------------------
# Load the series
# --------------------------------------------------------------------------

def load_runs():
    runs = []
    for rd in sorted(glob.glob(os.path.join(ROOT, SERIES, "run_*"))):
        r = {"dir": rd, "name": os.path.basename(rd)}
        r["cfg"] = json.load(open(os.path.join(rd, "run_config.json")))
        r["seed"] = r["cfg"]["seed"]
        r["gt"] = ev.join_run(rd)
        r["fw"] = ev.load_flow_windows(rd)
        r["pw"] = ev.load_pair_windows(rd)
        r["w"] = ev.load_windows(rd)
        r["model"] = ev.load_model_info(rd)
        r["alerts"] = [json.loads(l) for l in open(os.path.join(rd, "ctrl", "alerts.jsonl"))]
        r["dpi_stats"] = ev.load_dpi_stats(rd)
        runs.append(r)
    return runs


# --------------------------------------------------------------------------
# 1. Per-poll latency
# --------------------------------------------------------------------------

def host_paragraph(bench):
    """Describe the benchmark host: from host_info.json (scripts/run_latency_idle.sh)
    when present, else the known contention caveat of the run made in-session."""
    path = os.path.join(ROOT, bench, "host_info.json")
    if not os.path.exists(path):
        return ("**Host contention — read before citing absolute values.** This benchmark "
                "ran on a 3-vCPU desktop VM that also ran the GUI and the IDE driving the "
                "session (sampled right after the run: the IDE renderer ~72% and gnome-shell "
                "~50% of a core). The same 100-flow workload took 97 ms per poll in an "
                "earlier attempt (`results/latency_bench_runC_overbudget`) and ~350 ms here, "
                "so absolute per-poll times on that machine vary ~3x with background load. "
                "The scaling across tiers within the run is the more reliable reading; "
                "absolute numbers should come from `scripts/run_latency_idle.sh` on an idle "
                "host.")
    h = json.load(open(path))
    lim = h.get("idle_check", {}).get("load_1min_max", 0.5)
    ides = h.get("ide_processes_running") or []
    ide_s = ("no IDE process" if not ides else
             "IDE check overridden: " + ", ".join(ides) + " running, using "
             + ", ".join(f"{k} {v:.0f}%" for k, v in h.get("ide_cpu_pct_before", {}).items())
             + " of a core in a 5 s sample before the run — absolute per-poll times include "
             "that background load")
    return (f"**Host** (`{bench}/host_info.json`, recorded by `scripts/run_latency_idle.sh`): "
            f"{h.get('cpu_model', '?')}, {h.get('cores', '?')} cores, kernel "
            f"{h.get('kernel', '?')}; 1-min load average {h.get('load_before', '?')} before "
            f"the run (idle check: < {lim}; {ide_s}) and {h.get('load_after', '?')} "
            f"after. The benchmark itself (2 bmv2 switches, controller, DPI, sender) is the "
            f"load during the run.")


def host_caption(bench):
    if os.path.exists(os.path.join(ROOT, bench, "host_info.json")):
        h = json.load(open(os.path.join(ROOT, bench, "host_info.json")))
        ide = " with the IDE open" if h.get("ide_processes_running") else ""
        return (f"Host: {h.get('cpu_model', '?')}, {h.get('cores', '?')} cores{ide}, load "
                f"{h.get('load_before', '?')} before the run. Hollow = not steady state.")
    return ("3-vCPU desktop host shared with the GUI/IDE: absolute values vary ~3x with "
            "background load (see PAPER_NUMBERS.md, item 1). Hollow = not steady state.")


def item1_latency(runs, md):
    cols = [("t_read_ms", "register poll"), ("t_feat_ms", "feature extraction"),
            ("t_infer_ms", "IF inference (flow + pair)"), ("t_policy_ms", "policy + DPI handling"),
            ("t_write_ms", "write-back"), ("t_total_ms", "total")]
    md += ["## 1. Per-poll controller latency", ""]
    bench_cfg_path = os.path.join(ROOT, BENCH, "bench_config.json")
    rows = []
    if os.path.exists(bench_cfg_path):
        cfg = json.load(open(bench_cfg_path))
        bw = pd.read_csv(os.path.join(ROOT, BENCH, "ctrl", "windows.csv"))
        bstats = ev.load_dpi_stats(os.path.join(ROOT, BENCH))
        cap = ev.capacity(bw, bstats)
        settle = cfg["settle_s"]
        bw = bw.assign(pps=(bw.s1_fwd.diff() + bw.s1_cloned.diff()) / bw.ts.diff())
        md += [f"**Latency-only benchmark, not {RUN}** (`{BENCH}`, seed {cfg['seed']}, "
               f"registered as `latency-bench` in `experiments/seeds_used.json`). {RUN} never "
               f"exceeds {int(max(r['w'].n_flows_total.max() for r in runs))} active flows per "
               f"poll, so the tiers below come from `experiments/latency_bench.py`: the same "
               f"controller + DPI + head cloning (N = {cfg['head_clone_n']}) on real bmv2, "
               f"1 s polls, crafted UDP flows into s1, each in its own register slot "
               f"(the 4096 tier uses every slot), each sending one {cfg['payload_b']}-byte "
               f"packet every {cfg['flow_period_s']} s ({4096 / cfg['flow_period_s']:.0f} pkt/s "
               f"at 4096); {cfg['flows_per_source']} flows per spoofed source host, so the "
               f"number of host pairs scales with the tier; a stationary flow population "
               f"(each flow lives a random {0.5 * cfg['lifetime_mean_s']:.0f}–"
               f"{1.5 * cfg['lifetime_mean_s']:.0f} s, idles {cfg['idle_s']:.0f} s so its slot "
               f"ages out, then restarts as a new flow), so training and every tier see the "
               f"same mix of flow ages; tiers are cumulative, {cfg['tier_s']} s each, the first "
               f"{settle} s of each excluded (new flows join over the first {cfg['ramp_s']} s); "
               f"models trained on a "f"{cfg['warmup_s']} s warm-up at {cfg['tiers'][0]} flows. Source: "
               f"`{BENCH}/ctrl/windows.csv` (per-poll timings measured by the controller, "
               f"`controller/monitor.py`), tier spans from `{BENCH}/bench_config.json`.", "",
               "| tier (flows) | occupied slots, median | polls | polls over 1 s | "
               "switch rate / offered | "
               + " | ".join(f"{lbl} mean / P95 (ms)" for _, lbl in cols) + " |",
               "|" + "---|" * (5 + len(cols))]
        empty_tiers = []
        for t in [p for p in cfg["plan"] if p["phase"] == "measure"]:
            sel = bw[(bw.ts >= t["t_start"] + settle) & (bw.ts <= t["t_end"])
                     & (bw.phase == "scored")]
            if len(sel) == 0:
                md.append(f"| {t['n_flows']} † | — | 0 | — | — | "
                          + " | ".join("—" for _ in cols) + " |")
                empty_tiers.append(t["n_flows"])
                continue
            row = {"tier": t["n_flows"], "n_flows_median": float(sel.n_flows_total.median()),
                   "polls": int(len(sel)),
                   "read_median": float(sel.t_read_ms.median()),
                   "read_slow_polls": int((sel.t_read_ms > 150).sum()),
                   "feat_infer_median": float((sel.t_feat_ms + sel.t_infer_ms).median()),
                   "total_median": float(sel.t_total_ms.median()),
                   "pps_median": float(sel.pps.median()), "pps_max": float(sel.pps.max()),
                   "over_budget": int((sel.pps > ev.CAPACITY_BUDGET_PPS).sum()),
                   "overruns": int((sel.t_total_ms > 1000).sum()),
                   "n_flows_min": int(sel.n_flows_total.min()),
                   "n_flows_max": int(sel.n_flows_total.max()),
                   "max_poll_gap_s": float(sel.ts.diff().max())}
            duty = cfg["lifetime_mean_s"] / (cfg["lifetime_mean_s"] + cfg["idle_s"])
            row["offered_pps"] = row["n_flows_median"] / cfg["flow_period_s"] * duty
            row["processed_share"] = row["pps_median"] / row["offered_pps"]
            # Steady state = the controller keeps up and the switch keeps up
            row["steady_state"] = bool(row["overruns"] == 0 and row["max_poll_gap_s"] < 1.5
                                       and row["processed_share"] >= 0.9
                                       and row["over_budget"] == 0)
            for c, _ in cols:
                row[c] = {"mean": float(sel[c].mean()), "p95": float(np.percentile(sel[c], 95))}
            rows.append(row)
            md.append(f"| {t['n_flows']}{'' if row['steady_state'] else ' †'} | "
                      f"{row['n_flows_median']:.0f} | {row['polls']} | {row['overruns']} | "
                      f"{pct(row['processed_share'], 0)} | "
                      + " | ".join(f"{row[c]['mean']:.1f} / {row[c]['p95']:.1f}" for c, _ in cols)
                      + " |")
        md += ["", "Reading the table: the register poll reads all 11 flow-register arrays in "
               "full (4096 cells each) every poll (`controller/dataplane.py` "
               "`read_registers`), so its cost does not depend on the number of active flows; "
               "its median per tier is " + ", ".join(
                   f"{r['read_median']:.0f}" for r in rows) + " ms. What scales "
               "with active flows is feature extraction + inference, median "
               + " → ".join(f"{r['feat_infer_median']:.1f}" for r in rows)
               + " ms; median total " + ", ".join(f"{r['total_median']:.0f}" for r in rows)
               + " ms. Polls over the 1 s poll interval (total > 1000 ms): " + ", ".join(
                   f"{r['overruns']}" for r in rows) + ". `fig_latency_tiers.pdf/.png` "
               "plots the means.",
               "", "**† = not a steady-state measurement.** A tier counts as steady state "
               "only if no measured poll overran the 1 s interval, the longest gap between "
               "polls stayed < 1.5 s, the switch's packet rate (originals + clones) reached "
               ">= 90% of the offered rate of originals, "
               "and no window exceeded the load budget. Per tier: " + "; ".join(
                   f"{r['tier']}: {'steady' if r['steady_state'] else 'NOT steady'} "
                   f"(overruns {r['overruns']}, max gap {r['max_poll_gap_s']:.1f} s, "
                   f"switch rate {pct(r['processed_share'], 0)} of offered)" for r in rows) + "."
               + (f" Tiers with no measured poll at all (reported, not dropped): "
                  f"{', '.join(map(str, empty_tiers))}." if empty_tiers else ""),
               "", host_paragraph(BENCH),
               "", "**Validity:** active flows in the measured polls, min–max per tier: "
               + ", ".join(f"{r['n_flows_min']}–{r['n_flows_max']}" for r in rows)
               + "; longest gap between consecutive measured polls per tier: "
               + ", ".join(f"{r['max_poll_gap_s']:.2f} s" for r in rows) + ".",
               "", "**Load during the benchmark** (s1, originals + clones per second, from the "
               "data-plane counters in windows.csv), in the measured windows per tier: median "
               + ", ".join(f"{r['pps_median']:.0f}" for r in rows) + "; max "
               + ", ".join(f"{r['pps_max']:.0f}" for r in rows) + "; windows over the "
               f"{ev.CAPACITY_BUDGET_PPS} pkt/s budget " + ", ".join(
                   f"{r['over_budget']}" for r in rows) + f". Whole run (incl. ramps): peak "
               f"{cap['max_pps']:.0f} pkt/s, {cap['windows_over_budget']} windows over budget; "
               f"DPI capture drops {cap['dpi_kernel_drops']} of "
               f"{cap['dpi_kernel_received']} (`{BENCH}/ctrl/dpi_stats.json`).",
               "", "_Superseded or aborted benchmark attempts, kept on disk but not used here "
               "(`experiments/seeds_used.json`):_"]
        reg = json.load(open(os.path.join(ROOT, "experiments", "seeds_used.json")))["seeds"]
        superseded = {sd: v for sd, v in reg.items()
                      if v["purpose"] == "latency-bench" and v["series"] != BENCH}
        for sd, v in superseded.items():
            reason = v.get("note") or v.get("superseded_reason") or "superseded by the run above"
            md.append(f"- `{v['series']}` (seed {sd}): {reason}")
        NUM["latency_bench_superseded"] = superseded
        NUM["latency_bench"] = {"seed": cfg["seed"], "tiers": rows, "capacity": cap}
    else:
        md += ["_Benchmark results not found — run `experiments/latency_bench.py`._"]
    # the series' own, as reference
    allw = pd.concat([r["w"][r["w"].phase == "scored"] for r in runs])
    rc = {c: {"mean": float(allw[c].mean()), "p95": float(np.percentile(allw[c], 95))}
          for c, _ in cols}
    md += ["", f"For reference, **{RUN} itself** (all 5 runs, {len(allw)} scored polls, "
           f"active flows per poll median {allw.n_flows_total.median():.0f}, max "
           f"{allw.n_flows_total.max():.0f}; `{SERIES}/run_0N/ctrl/windows.csv`): "
           + "; ".join(f"{lbl} {rc[c]['mean']:.1f} / {rc[c]['p95']:.1f} ms" for c, lbl in cols)
           + " (mean / P95).", ""]
    bench_model = os.path.join(ROOT, BENCH, "ctrl", "model.json")
    bench_prefix = (os.path.exists(bench_model) and "dpi_confirmation_action"
                    not in json.load(open(bench_model)).get("data_plane", {}))
    if os.path.exists(bench_model) and (DPI_ACTION == "DROP") == bench_prefix:
        ab = sorted(glob.glob(os.path.join(ROOT, "results", "reg_read_ab_*", "report.json")))
        ab_s = "not measured"
        if ab:
            a = json.load(open(ab[-1]))
            sm = a["summary"]
            ab_s = (f"{sm['with_drop_pkts']['median_ms']:.0f} vs "
                    f"{sm['without_drop_pkts']['median_ms']:.0f} ms median per read on a live "
                    f"s1, paired difference {a['paired_diff_ms_median']:+.1f} ms "
                    f"(`{rel(os.path.dirname(ab[-1]))}/report.json`, "
                    "`experiments/reg_read_ab.py`)")
        which = ("before the DROP change (Run D), which reads one register array fewer per "
                 f"poll than {RUN}'s" if bench_prefix else
                 f"after the DROP change (Run D), which reads one register array more per poll "
                 f"than {RUN}'s")
        md += [f"**{RUN} vs the benchmark:** the latency-only benchmark ran the controller "
               f"{which} (`drop_pkts`, `controller/dataplane.py` FLOW_REGISTERS). "
               f"That array's cost: {ab_s}. {RUN}'s own per-poll times are in the paragraph "
               "above.", ""]
    NUM[f"latency_{os.path.basename(OUT)}"] = rc
    if rows:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        steady = [r for r in rows if r["steady_state"]]
        series = [(c, lbl, CAT[i]) for i, (c, lbl) in enumerate(cols[:3])] + \
                 [("t_total_ms", "total", INK)]
        for c, lbl, col in series:
            ax.plot([r["tier"] for r in steady], [r[c]["mean"] for r in steady], marker="o",
                    ms=4, lw=1.6, color=col, label=lbl)
            for prev, r in zip(rows, rows[1:]):
                if not r["steady_state"]:     # dashed, hollow: not a steady-state point
                    ax.plot([prev["tier"], r["tier"]], [prev[c]["mean"], r[c]["mean"]],
                            ls="--", lw=1.0, color=col)
                    ax.plot([r["tier"]], [r[c]["mean"]], marker="o", ms=5, mfc=SURF,
                            mec=col, lw=0)
        for r in rows:
            if not r["steady_state"]:
                ax.annotate("not steady state\n(polls overran 1 s)", (r["tier"], r["t_total_ms"]["mean"]),
                            xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                            fontsize=7, color=INK2)
        x = [r["tier"] for r in rows]
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.set_xlabel("flows (tier)")
        ax.set_ylabel("mean per-poll time (ms)")
        ax.set_title("Controller per-poll latency (latency-only benchmark)", loc="left")
        ax.legend(fontsize=8, loc="upper left")
        fig.text(0.01, -0.04, host_caption(BENCH), fontsize=6.5, color=INK2)
        save(fig, "fig_latency_tiers")
    return md


# --------------------------------------------------------------------------
# 2. Time share per monitoring level
# --------------------------------------------------------------------------

def item2_level_share(runs, md):
    fws, pws, n_scored_windows, periods = [], [], 0, []
    for r in runs:
        gtk = dict(zip(r["gt"].key, r["gt"].label))
        f = r["fw"][r["fw"].scored == 1].copy()
        f["gt_label"] = f.key.map(gtk).fillna("not in ground truth")
        fws.append(f)
        pws.append(r["pw"][r["pw"].scored == 1])
        sw = r["w"][r["w"].phase == "scored"]
        n_scored_windows += len(sw)
        periods += sw.ts.diff().dropna().tolist()
    f = pd.concat(fws)
    p = pd.concat(pws)

    def shares(s):
        c = s.value_counts()
        n = int(c.sum())
        return {LEVELS[k]: {"n": int(c.get(k, 0)), "share": c.get(k, 0) / n} for k in range(4)}, n

    overall, n_f = shares(f.level)
    by_label = {lbl: shares(g.level) for lbl, g in f.groupby("gt_label")}
    by_switch = {f"s{s}": shares(g.level) for s, g in f.groupby("switch")}
    pairs, n_p = shares(p.level)
    NUM["level_share"] = {"flow_polls": overall, "n_flow_polls": n_f, "by_gt_label":
                          {k: v[0] for k, v in by_label.items()},
                          "by_switch": {k: v[0] for k, v in by_switch.items()},
                          "pair_polls": pairs, "n_pair_polls": n_p}
    md += ["## 2. Time share per monitoring level", "",
           f"**Which windows:** the scored phase only — every poll after the one-time "
           f"training at the end of the 30 s warm-up (`phase == scored` in windows.csv, "
           f"`scored == 1` in flow_windows.csv), all 5 runs, both switches: "
           f"{n_scored_windows} poll windows (mean poll period "
           f"{np.mean(periods):.3f} s). Warm-up polls are excluded (every flow is NORMAL "
           f"before the models exist). **Unit:** one occupied flow slot in one poll "
           f"(a flow-poll), so a share is the fraction of flow-time spent at that level; "
           f"host pairs likewise per pair-poll. Sources: "
           f"`{SERIES}/run_0N/ctrl/flow_windows.csv`, `pair_windows.csv`, `windows.csv`.", "",
           "| population | flow-polls | " + " | ".join(LEVELS) + " |", "|---|---|---|---|---|---|"]

    def line(name, sh, n):
        return f"| {name} | {n} | " + " | ".join(
            f"{pct(sh[l]['share'], 2)} ({sh[l]['n']})" for l in LEVELS) + " |"

    md.append(line("all flows", overall, n_f))
    for lbl in ["benign", "attack", "not in ground truth"]:
        if lbl in by_label:
            md.append(line(f"ground-truth {lbl}" if lbl != "not in ground truth"
                           else "not in ground truth (reverse direction etc.)", *by_label[lbl]))
    for s, v in by_switch.items():
        md.append(line(f"switch {s}", *v))
    md.append(line("host pairs (pair-polls)", pairs, n_p))
    if overall["DROP"]["n"] == 0:
        md += ["", f"DROP never occurred in {RUN}: it needs a second DPI confirmation on a flow "
               "already at CLONE, and DPI confirms each (flow, class) once "
               "(`controller/dpi.py` `_confirm`); every confirmation therefore arrived with "
               "the flow below CLONE and force-cloned it (`forced_clone` in alerts.jsonl, "
               "no `forced_drop`)."]
    md.append("")
    return md


# --------------------------------------------------------------------------
# 3. Window-level confusion matrix
# --------------------------------------------------------------------------

def item3_window_confusion(runs, md):
    tp = fp = fn = tn = 0
    fp_tail = tn_tail = fn_lead = 0
    per_switch = Counter()
    for r in runs:
        gt = r["gt"]
        att = gt[gt.label == "attack"][["start_ts", "end_ts"]].to_numpy()
        w = r["w"][r["w"].phase == "scored"]
        ts_all = r["w"].ts.to_numpy()
        win_all = r["w"].window.to_numpy()
        prev_ts = dict(zip(win_all[1:], ts_all[:-1]))
        hi_f = r["fw"][r["fw"].level >= CLONE].groupby(["switch", "window"]).size()
        hi_p = r["pw"][r["pw"].level >= CLONE].groupby(["switch", "window"]).size()
        for s in (1, 2):
            actual_seq, pred_seq = [], []
            for wr in w.itertuples(index=False):
                lo, hi = prev_ts.get(wr.window, wr.ts - 1.0), wr.ts
                actual = bool(((att[:, 0] <= hi) & (att[:, 1] > lo)).any())
                pred = (s, wr.window) in hi_f.index or (s, wr.window) in hi_p.index
                actual_seq.append(actual)
                pred_seq.append(pred)
                if actual and pred:
                    tp += 1; per_switch[(s, "tp")] += 1
                elif pred:
                    fp += 1; per_switch[(s, "fp")] += 1
                    if any(actual_seq[-4:-1]):
                        fp_tail += 1
                elif actual:
                    fn += 1; per_switch[(s, "fn")] += 1
                else:
                    tn += 1; per_switch[(s, "tn")] += 1
                    if any(actual_seq[-4:-1]):
                        tn_tail += 1
            fn_lead += sum(1 for i in range(len(actual_seq) - 1)
                           if actual_seq[i] and not pred_seq[i] and pred_seq[i + 1])
    tpr, fpr = ev._rate(tp, tp + fn), ev._rate(fp, fp + tn)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    fpr_x = ev._rate(fp - fp_tail, (fp - fp_tail) + (tn - tn_tail))
    NUM["window_confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "tpr": tpr, "fpr": fpr,
                               "precision": prec, "fp_within_3_windows_after_attack": fp_tail,
                               "fn_followed_by_detection_next_window": fn_lead,
                               "tn_within_3_windows_after_attack": tn_tail,
                               "fpr_excluding_deescalation": fpr_x,
                               "per_switch": {f"s{s}_{k}": v for (s, k), v in per_switch.items()}}
    md += ["## 3. Window-level confusion matrix", "",
           "**Unit:** one poll window on one switch, scored phase, all 5 runs. **Predicted "
           "attack window:** any flow slot or host pair on that switch at level >= CLONE in "
           "that window (`level` in flow_windows.csv / pair_windows.csv). **Actual attack "
           "window:** a ground-truth attack flow (`ground_truth.jsonl`) was active during "
           "the window's interval (previous poll, this poll]; every attack flow in this "
           "topology crosses both switches. Sources: `" + SERIES +
           "/run_0N/ctrl/{windows,flow_windows,pair_windows}.csv`, "
           "`run_0N/ground_truth.jsonl`.", "",
           "| | predicted attack | predicted benign |", "|---|---|---|",
           f"| **actual attack** | TP {tp} | FN {fn} |",
           f"| **actual benign** | FP {fp} | TN {tn} |", "",
           f"TPR {fmt(tpr['rate'])} [{fmt(tpr['ci95'][0])}, {fmt(tpr['ci95'][1])}], FPR "
           f"{fmt(fpr['rate'])} [{fmt(fpr['ci95'][0])}, {fmt(fpr['ci95'][1])}], precision "
           f"{fmt(prec)}. Per switch: "
           + "; ".join(f"s{s} TP {per_switch[(s,'tp')]} FN {per_switch[(s,'fn')]} "
                       f"FP {per_switch[(s,'fp')]} TN {per_switch[(s,'tn')]}" for s in (1, 2))
           + f". Of the {fp} FP windows, {fp_tail} fall within 3 windows after an actual "
           f"attack window (hysteresis: leaving CLONE needs K_DOWN = 3 windows below it), "
           f"the rest are benign-phase escalations (iperf3 false positives, see "
           f"`RESULTS.md`). Of the {fn} FN windows, {fn_lead} are followed by a predicted "
           f"window: detection one poll late (a confirmation or score is acted on at the "
           f"next poll), not a miss of that attack phase.", "",
           f"**Secondary (not the headline):** the de-escalation period is the 3 windows "
           f"after an actual attack window on the same switch (K_DOWN = 3, "
           f"`controller/policy.py`). It holds {fp_tail} of the {fp} FP windows and "
           f"{tn_tail} TN windows. Excluding that period from the benign windows "
           f"altogether (its FPs and its TNs alike): FP {fp - fp_tail}, TN {tn - tn_tail}, "
           f"FPR {fmt(fpr_x['rate'])} [{fmt(fpr_x['ci95'][0])}, {fmt(fpr_x['ci95'][1])}]. "
           f"The headline FPR above ({fmt(fpr['rate'])}) counts those windows, because the "
           f"system is still cloning/inspecting during them.", ""]
    return md


# --------------------------------------------------------------------------
# 4. Score distributions
# --------------------------------------------------------------------------

def item4_scores(runs, md):
    data = {"flow": {"warm-up": [], "benign eval": [], "attack eval": []},
            "host pair": {"warm-up": [], "benign eval": [], "attack eval": []}}
    th = {"flow": [], "host pair": []}
    for r in runs:
        m = r["model"]
        data["flow"]["warm-up"] += m["flow_model"]["warmup_scores"]
        data["host pair"]["warm-up"] += m["host_pair_model"]["warmup_scores"]
        th["flow"].append((m["flow_model"]["thresholds"]["monitor"],
                           m["flow_model"]["thresholds"]["clone"]))
        th["host pair"].append((m["host_pair_model"]["thresholds"]["monitor"],
                                m["host_pair_model"]["thresholds"]["clone"]))
        gtk = dict(zip(r["gt"].key, r["gt"].label))
        f = r["fw"][r["fw"].scored == 1]
        lab = f.key.map(gtk)
        data["flow"]["benign eval"] += f.score[lab == "benign"].tolist()
        data["flow"]["attack eval"] += f.score[lab == "attack"].tolist()
        att = r["gt"][r["gt"].label == "attack"]
        win = defaultdict(list)
        for a in att.itertuples(index=False):
            win[(a.src, a.dst)].append((a.start_ts, a.end_ts))
        p = r["pw"][r["pw"].scored == 1]
        is_att = [any(s - ev.PAIR_MATCH_SLACK_S <= t <= e + ev.PAIR_MATCH_SLACK_S
                      for s, e in win.get((src, dst), []))
                  for src, dst, t in zip(p.src, p.dst, p.ts)]
        is_att = np.array(is_att, dtype=bool)
        data["host pair"]["attack eval"] += p.score[is_att].tolist()
        data["host pair"]["benign eval"] += p.score[~is_att].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    stats = {}
    for ax, model in zip(axes, ["flow", "host pair"]):
        allv = np.concatenate([np.array(v) for v in data[model].values()])
        bins = np.linspace(np.percentile(allv, 0.2), np.percentile(allv, 99.8), 40)
        stats[model] = {}
        for i, (name, v) in enumerate(data[model].items()):
            v = np.array(v)
            ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                    lw=1.6, color=CAT[i], label=name)
            stats[model][name] = {"n": int(len(v)), "median": float(np.median(v)),
                                  "p1": float(np.percentile(v, 1)), "p5": float(np.percentile(v, 5))}
        for k, (p5, p1) in enumerate(th[model]):
            ax.axvline(p5, color=INK2, lw=0.8, ls="--", label="p5 (MONITOR), 5 runs" if k == 0 else None)
            ax.axvline(p1, color=INK, lw=0.8, ls=":", label="p1 (CLONE), 5 runs" if k == 0 else None)
        ax.set_title(f"{model} model", loc="left")
        ax.set_ylabel("density")
        ax.set_xlabel("IF score (lower = more anomalous)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.12))
    fig.text(0.99, -0.16, "n (flow / pair): " + ", ".join(
        f"{k} {stats['flow'][k]['n']} / {stats['host pair'][k]['n']}" for k in stats["flow"]),
        ha="right", fontsize=6.5, color=INK2)
    save(fig, "fig_score_distribution")
    NUM["score_distribution"] = {"stats": stats, "thresholds_p5_p1_per_run": th}
    md += ["## 4. Score distributions", "",
           "`fig_score_distribution.pdf/.png`. Warm-up = each model's own scores on its own "
           "warm-up samples (`warmup_scores` in `run_0N/ctrl/model.json`, i.e. the samples "
           "the thresholds were derived from); benign eval / attack eval = scored-phase rows "
           "of flow_windows.csv joined to ground truth by exact 5-tuple (flows), and "
           "pair_windows.csv rows labelled attack when a ground-truth attack flow of that "
           "(src, dst) pair was active within ±1.5 s (pairs). Flow rows of 5-tuples not in "
           "ground truth (e.g. reverse direction) are left out. Pooled over the 5 runs; the "
           "dashed / dotted lines are each run's p5 / p1 (one model per run, shared by both "
           "switches). Histograms are density-normalised (the populations differ in size) "
           "and clipped to the pooled 0.2–99.8th percentile range.", "",
           "| model | population | n | median | p5 | p1 |", "|---|---|---|---|---|---|"]
    for model, d in stats.items():
        for name, s in d.items():
            md.append(f"| {model} | {name} | {s['n']} | {fmt(s['median'], 4)} | "
                      f"{fmt(s['p5'], 4)} | {fmt(s['p1'], 4)} |")
    md.append("")
    return md


# --------------------------------------------------------------------------
# 5. Per-class detection by trigger
# --------------------------------------------------------------------------

def _with_other(rbt):
    """recall_by_trigger with every trigger outside TRIGGERS (DPI-level,
    DPI-drop, unattributed) folded into one "other" entry."""
    out = {t: v for t, v in rbt.items() if t in TRIGGERS}
    rest = [v for t, v in rbt.items() if t not in TRIGGERS and t != "none"]
    if rest:
        k, n = sum(v["k"] for v in rest), rest[0]["n"]
        out["other"] = {"k": k, "n": n, "rate": k / n, "from": sorted(
            t for t in rbt if t not in TRIGGERS and t != "none")}
    return out


def item5_trigger_bars(metrics, md):
    pc = {c: {**v, "recall_by_trigger": _with_other(v["recall_by_trigger"])}
          for c, v in metrics["per_attack_class"].items()}
    b_all = _with_other(metrics["benign"]["false_positives_by_trigger"])
    trig = TRIGGERS + (["other"] if any("other" in v["recall_by_trigger"] for v in pc.values())
                       or "other" in b_all else [])
    colors = CAT + [NEUTRAL]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0),
                                  gridspec_kw={"width_ratios": [4, 1.3]})
    x = np.arange(len(CLASSES))
    bottom = np.zeros(len(CLASSES))
    for i, t in enumerate(trig):
        vals = np.array([pc[c]["recall_by_trigger"].get(t, {}).get("rate", 0.0) for c in CLASSES])
        lbl = t if vals.any() else f"{t} (0 in every class)"
        ax.bar(x, vals, bottom=bottom, width=0.6, color=colors[i], label=lbl,
               edgecolor=SURF, linewidth=1.0)
        bottom += vals
    tot = np.array([pc[c]["detected_recall"] for c in CLASSES])
    lo = np.array([pc[c]["detected_recall_ci95"][0] for c in CLASSES])
    hi = np.array([pc[c]["detected_recall_ci95"][1] for c in CLASSES])
    ax.errorbar(x, tot, yerr=[tot - lo, hi - tot], fmt="none", ecolor=INK, elinewidth=0.8,
                capsize=3)
    for xi, c in zip(x, CLASSES):
        ax.text(xi, min(hi[xi], 1.0) + 0.03, f"{pc[c]['detected_clone_or_above']}/{pc[c]['n']}",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("detection rate (recall)")
    ax.set_title("Attack flows detected, by trigger", loc="left")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    b = b_all
    bottom = 0.0
    for i, t in enumerate(trig):
        v = b.get(t, {}).get("rate", 0.0)
        ax2.bar([0], [v], bottom=bottom, width=0.6, color=colors[i], edgecolor=SURF, linewidth=1.0)
        bottom += v
    o = metrics["overall"]
    ax2.errorbar([0], [o["fpr"]], yerr=[[o["fpr"] - o["fpr_ci95"][0]], [o["fpr_ci95"][1] - o["fpr"]]],
                 fmt="none", ecolor=INK, elinewidth=0.8, capsize=3)
    ax2.text(0, o["fpr_ci95"][1] + 0.004, f"{o['fp']}/{o['fp'] + o['tn']}", ha="center",
             fontsize=8)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["benign"])
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(0, 0.13)
    ax2.set_ylabel("false-positive rate")
    ax2.set_title("Benign flows", loc="left")
    save(fig, "fig_detection_by_trigger")
    md += ["## 5. Per-class detection by trigger", "",
           "`fig_detection_by_trigger.pdf/.png` — stacked segments are the share of each "
           "class's ground-truth flows detected via that trigger; whisker = Clopper-Pearson "
           "95% CI of the class total. Separate panel (own axis) for benign false positives. "
           f"Source: `{SERIES}/metrics.json` (`analysis/evaluate.py`).", "",
           "| class | n | " + " | ".join(trig) + " | total [95% CI] |",
           "|---|---|" + "---|" * len(trig) + "---|"]
    for c in CLASSES:
        rbt = pc[c]["recall_by_trigger"]
        md.append(f"| {c} | {pc[c]['n']} | " + " | ".join(
            f"{rbt[t]['k']} ({pct(rbt[t]['rate'])})" if t in rbt else "0" for t in trig)
            + f" | {pc[c]['detected_clone_or_above']} ({pct(pc[c]['detected_recall'])}) "
            f"[{pct(pc[c]['detected_recall_ci95'][0])}, {pct(pc[c]['detected_recall_ci95'][1])}] |")
    md.append(f"| benign (FP) | {metrics['benign']['n']} | " + " | ".join(
        f"{b[t]['k']} ({pct(b[t]['rate'])})" if t in b else "0" for t in trig)
        + f" | {o['fp']} ({pct(o['fpr'])}) [{pct(o['fpr_ci95'][0])}, {pct(o['fpr_ci95'][1])}] |")
    others = sorted({t for v in list(pc.values()) + [{"recall_by_trigger": b}]
                     for t in v["recall_by_trigger"].get("other", {}).get("from", [])})
    if others:
        md += ["", "\"other\" = " + ", ".join(others) + " (`analysis/evaluate.py` "
               "module docstring)."]
    md += ["", "Trigger = what first put the flow at >= CLONE (`escalated_by` on its first "
           "such row in flow_windows.csv, else its host pair's); a later forced DROP does "
           "not re-attribute it."]
    md.append("")
    return md


# --------------------------------------------------------------------------
# 6. Monitoring-level timeline
# --------------------------------------------------------------------------

def pick_representative(metrics):
    pr = metrics["per_run"]
    names = sorted(pr)
    vals = {n: (pr[n]["overall"]["tpr"], pr[n]["overall"]["fpr"],
                pr[n]["inspection"]["inspection_reduction_packets"]) for n in names}
    arr = np.array([vals[n] for n in names])
    med = np.median(arr, axis=0)
    rng = arr.max(axis=0) - arr.min(axis=0)
    d = (np.abs(arr - med) / np.where(rng > 0, rng, 1)).sum(axis=1)
    return names[int(np.argmin(d))], dict(zip(names, d.round(3)))


SPORT_BASES = {"syn_scan": 57000, "null_scan": 60000, "sqli": 56000, "rate_spike": 63000}


def item6_timeline(runs, metrics, md):
    name, dist = pick_representative(metrics)
    r = next(x for x in runs if x["name"] == name)
    t0 = r["w"].ts.iloc[0]
    train_until = r["model"]["train_until"]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [0.9, 2.2, 1.4]})
    ax0, ax1, ax2 = axes
    # (a) phases from ground truth
    gt = r["gt"]
    ben = gt[gt.generator.str.startswith("iperf3")]
    for b in ben.itertuples(index=False):
        ax0.axvspan(b.start_ts - t0, b.end_ts - t0, ymin=0.5, ymax=0.95, color=NEUTRAL, lw=0)
    for i, c in enumerate(CLASSES):
        g = gt[gt.attack_class == c].copy()
        g["cycle"] = (g.sport - SPORT_BASES[c]) // 20
        for _, ph in g.groupby("cycle"):
            ax0.axvspan(ph.start_ts.min() - t0, max(ph.end_ts.max(), ph.start_ts.min() + 0.6) - t0,
                        ymin=0.5, ymax=0.95, color=CAT[i], lw=0,
                        label=c if _ == g.cycle.min() else None)
    ax0.set_yticks([])
    ax0.grid(False)
    ax0.set_title(f"(a) traffic phases — {name}, seed {r['seed']}", loc="left")
    from matplotlib.patches import Patch
    h, l = ax0.get_legend_handles_labels()
    h.append(Patch(color=NEUTRAL)); l.append("benign (iperf3 + HTTP)")
    ax0.legend(h, l, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               handlelength=1.2, columnspacing=1.2)
    # (b) flow slots per level on s1
    f = r["fw"][r["fw"].switch == 1]
    cnt = f.groupby(["ts", "level"]).size().unstack(fill_value=0).reindex(columns=range(4),
                                                                          fill_value=0)
    xs = cnt.index - t0
    ax1.plot(xs, cnt.sum(axis=1), color=NEUTRAL, lw=1.2, label="all active flows")
    for k in (1, 2, 3):
        ax1.plot(xs, cnt[k], color=LEVEL_RAMP[LEVELS[k]], lw=1.6, label=LEVELS[k])
    ax1.set_ylabel("flow slots (s1)")
    ax1.set_title("(b) flows per monitoring level, switch 1", loc="left")
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.3)
    ax1.legend(fontsize=7, ncol=4, loc="upper right")
    # (c) host pairs on s1
    p = r["pw"][r["pw"].switch == 1]
    for i, (src, dst, lbl) in enumerate([("10.0.1.1", "10.0.2.1", "h1→h3 (iperf3, scans)"),
                                          ("10.0.1.2", "10.0.2.2", "h2→h4 (HTTP, SQLi, spike)")]):
        g = p[(p.src == src) & (p.dst == dst)].sort_values("ts")
        ax2.step(g.ts - t0, g.level + (i - 0.5) * 0.08, where="post", color=CAT[i], lw=1.4,
                 label=lbl)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(LEVELS)
    ax2.set_ylim(-0.3, 4.1)
    ax2.set_title("(c) host-pair level, switch 1", loc="left")
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.set_xlabel("time since controller start (s)")
    for ax in axes:
        ax.axvline(train_until - t0, color=INK, lw=0.8, ls="--")
    ax2.text(train_until - t0, 3.75, " models trained", fontsize=7, va="center", color=INK)
    save(fig, "fig_level_timeline")
    NUM["timeline_run"] = {"run": name, "seed": r["seed"], "distance_to_median": dist}
    md += ["## 6. Monitoring-level timeline", "",
           f"`fig_level_timeline.pdf/.png` — **{name}, seed {r['seed']}**. Chosen as the run "
           f"closest to the 5-run median on TPR, FPR and inspection reduction together "
           f"(sum of |value − median| / range: "
           + ", ".join(f"{n} {v}" for n, v in dist.items()) + "). (a) phases from "
           f"`{SERIES}/{name}/ground_truth.jsonl`; (b) flow slots per level on switch 1 per "
           f"poll from `flow_windows.csv` (gray = all occupied slots); (c) the two host pairs "
           f"that carry traffic, from `pair_windows.csv`. Dashed line: the one-time training "
           f"(`train_until` in `model.json`).", ""]
    return md


# --------------------------------------------------------------------------
# 7. Ablation from logs
# --------------------------------------------------------------------------

def _detect(gt, fw, pw, flow_levels, pair_levels):
    """Same detection definition as analysis/evaluate.join_run, on given levels."""
    hi = set(fw.key[flow_levels >= CLONE])
    pc = pw[pair_levels >= CLONE]
    ev_pairs = {k: g.ts.to_numpy() for k, g in pc.groupby(["src", "dst"])}
    det = []
    for g in gt.itertuples(index=False):
        d = g.key in hi
        if not d:
            ts = ev_pairs.get((g.src, g.dst))
            if ts is not None:
                d = bool(((ts >= g.start_ts - ev.PAIR_MATCH_SLACK_S)
                          & (ts <= g.end_ts + ev.PAIR_MATCH_SLACK_S)).any())
        det.append(d)
    return np.array(det)


def _dpi_head_forces(r):
    """Logged DPI confirmations attributable to flow-head clones: clone_reason
    4, or 3 when the host-clone rule that captured the packet was installed by
    a DPI-head escalation (the controller's own attribution, _dpi_trigger)."""
    pw = r["pw"]
    flow_f, pair_f, kinds = [], [], Counter()
    for a in r["alerts"]:
        if a.get("type") != "confirmed":
            continue
        reason, sw, ft = a.get("clone_reason"), a["switch"], a["five_tuple"]
        ok = reason == 4
        if reason == 3:
            prior = pw[(pw.switch == sw) & (pw.src == ft["src"]) & (pw.ts <= a["ts"] + 1.0)
                       & (pw.level >= CLONE)]
            ok = len(prior) > 0 and prior.sort_values("ts").escalated_by.iloc[-1] == "DPI-head"
        kinds[(reason, ok)] += 1
        if not ok:
            continue
        tup = (ft["src"], ft["dst"], int(ft["proto"]), int(ft["sport"]), int(ft["dport"]))
        flow_f.append((sw, tup, a["ts"]))
        if a["attack_class"] in ("syn_scan", "null_scan"):
            pair_f.append((sw, ft["src"], ft["dst"], a["ts"]))
    return flow_f, pair_f, kinds


def item7_ablation(runs, metrics, md):
    head_n = runs[0]["cfg"]["head_clone_n"]
    res = {k: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "fwd": 0, "cl": 0, "per_run": [],
               "per_class": Counter(), "n_class": Counter(), "fwd_est": 0, "cl_est": 0}
           for k in ("IF only", "DPI-head only", "full system")}
    validation, kinds_all = [], Counter()
    cons = Counter()
    fp_trig, if_trig = Counter(), Counter()    # full-system triggers, see the text
    for r in runs:
        gt, fw, pw, w = r["gt"], r["fw"], r["pw"], r["w"]
        actual = (gt.label == "attack").to_numpy()
        tot = ev.run_totals(w)
        lf, lp = replay_levels(fw, pw, w, "LOG")
        est_full = estimate_clones(fw, pw, lf, lp, head_n, True)
        validation.append({"run": r["name"], "est": est_full, "actual": tot,
                           "measured_dropped": tot["dropped"]})
        scen = {}
        ifl, ipl = replay_levels(fw, pw, w, "IF")
        scen["IF only"] = (_detect(gt, fw, pw, ifl, ipl),
                           estimate_clones(fw, pw, ifl, ipl, 0, True))
        ff, pf, kinds = _dpi_head_forces(r)
        kinds_all.update(kinds)
        dfl, dpl = replay_levels(fw, pw, w, "DPI", ff, pf, dpi_action=DPI_ACTION)
        scen["DPI-head only"] = (_detect(gt, fw, pw, dfl, dpl),
                                 estimate_clones(fw, pw, dfl, dpl, head_n, False))
        full_det = gt.predicted_attack.to_numpy()
        assert (_detect(gt, fw, pw, lf, lp) == full_det).all(), "detection replay mismatch"
        scen["full system"] = (full_det, {"forwarded": tot["forwarded"], "cloned": tot["cloned"]})
        est = {"IF only": scen["IF only"][1], "DPI-head only": scen["DPI-head only"][1],
               "full system": est_full}
        cls_arr = gt.attack_class.fillna("benign").to_numpy()
        ben = (gt.label == "benign").to_numpy()
        dd = scen["DPI-head only"][0]
        cons["dpi_fp"] += int((dd & ben).sum())
        cons["dpi_fp_also_full"] += int((dd & ben & full_det).sum())
        cons["dpi_new_tp"] += int((dd & ~ben & ~full_det).sum())
        trig = gt.trigger.to_numpy()
        fp_trig.update(trig[dd & ben & full_det])
        if_trig.update(trig[scen["IF only"][0] & ~ben])
        for k, (det, cl) in scen.items():
            c = res[k]
            run_tp = int((actual & det).sum()); run_fn = int((actual & ~det).sum())
            run_fp = int((~actual & det).sum()); run_tn = int((~actual & ~det).sum())
            c["tp"] += run_tp; c["fn"] += run_fn; c["fp"] += run_fp; c["tn"] += run_tn
            c["fwd"] += cl["forwarded"]; c["cl"] += cl["cloned"]
            c["fwd_est"] += est[k]["forwarded"]; c["cl_est"] += est[k]["cloned"]
            for cls_name, d in zip(cls_arr, det):
                c["n_class"][cls_name] += 1
                c["per_class"][cls_name] += int(d)
            c["per_run"].append({"tpr": run_tp / (run_tp + run_fn), "fpr": run_fp / (run_fp + run_tn),
                                 "insp": 1 - cl["cloned"] / cl["forwarded"]})
    out = {}
    for k, c in res.items():
        out[k] = {"tpr": ev._rate(c["tp"], c["tp"] + c["fn"]), "fpr": ev._rate(c["fp"], c["fp"] + c["tn"]),
                  "inspection_reduction": 1 - c["cl"] / c["fwd"], "per_run": c["per_run"],
                  "counts": {x: c[x] for x in ("tp", "fn", "fp", "tn")},
                  "inspection_reduction_estimated": 1 - c["cl_est"] / c["fwd_est"],
                  "per_class": {k2: {"detected": c["per_class"][k2], "n": c["n_class"][k2]}
                                for k2 in c["n_class"]}}
    err = [(v["est"]["cloned"] - v["actual"]["cloned"]) / v["actual"]["cloned"] for v in validation]
    err_fwd = [abs(v["est"]["forwarded"] - v["actual"]["forwarded"]) / v["actual"]["forwarded"]
               for v in validation]
    drops_est = sum(v["est"]["dropped"] for v in validation)
    drops_meas = sum(v["measured_dropped"] or 0 for v in validation)
    NUM["ablation"] = out
    NUM["ablation_consistency"] = dict(cons)
    NUM["ablation_full_trigger_of_dpi_fp"] = dict(fp_trig)
    NUM["ablation_full_trigger_of_if_only_tp"] = dict(if_trig)
    NUM["ablation_estimator_validation"] = validation
    md += [f"## 7. Ablation on {RUN} logs", "",
           f"Computed offline from {RUN}'s logs — no new runs (`analysis/replay.py`, applied "
           "per run, pooled over the 5). Detection uses the same definition as the full "
           "evaluation (flow level, or its host pair while active, >= CLONE).", "",
           "- **(a) IF only** — the logged per-window raw IF levels (flow and host-pair "
           "models) replayed through the controller's own hysteresis "
           "(`controller/policy.FlowPolicy`), every DPI confirmation ignored; no flow-head "
           "clones (N = 0); adaptive sampling kept (it feeds the IF's payload-entropy "
           "feature, Decision C).",
           "- **(b) DPI-head only** — no IF: raw level NORMAL every window; each logged DPI "
           "confirmation attributable to a flow-head clone (clone_reason 4, or 3 under a "
           "host rule that a DPI-head escalation installed) "
           + ("forces the flow, and for syn/null scans its host pair, to DROP (the "
              f"controller's own action in {RUN})" if DPI_ACTION == "DROP" else
              "force-clones the flow, and for syn/null scans its host pair")
           + ", at the poll that processes it, then "
           "de-escalates by the same hysteresis; head clones kept (N = "
           f"{head_n}); no sampling. Confirmations used: "
           + ", ".join(f"reason {k[0]} {'kept' if k[1] else 'dropped'}: {v}"
                       for k, v in sorted(kinds_all.items(), key=str)) + ".",
           "- **(c) full system** — the logged run as it happened (`analysis/evaluate.py`; "
           "inspection from the data-plane counters).", "",
           "| configuration | TPR [95% CI] | FPR [95% CI] | inspection reduction (packets) |",
           "|---|---|---|---|"]
    for k in ("IF only", "DPI-head only", "full system"):
        v = out[k]
        md.append(f"| {k} | {fmt(v['tpr']['rate'])} [{fmt(v['tpr']['ci95'][0])}, "
                  f"{fmt(v['tpr']['ci95'][1])}] | {fmt(v['fpr']['rate'])} "
                  f"[{fmt(v['fpr']['ci95'][0])}, {fmt(v['fpr']['ci95'][1])}] | "
                  f"{pct(v['inspection_reduction'])}"
                  + (" (estimated)" if k != "full system" else " (measured)") + " |")
    io, do, fs = out["IF only"], out["DPI-head only"], out["full system"]
    if_cls = {k2: v for k2, v in io["per_class"].items() if k2 != "benign" and v["detected"]}
    if_tp = sum(v["detected"] for v in if_cls.values())
    md += ["", "**What IF-only detects.** Its " + str(if_tp) + " detected attack flows "
           "come from " + ", ".join(
               f"{k2} {v['detected']}/{v['n']}" for k2, v in sorted(if_cls.items()))
           + " (`per_class` in `numbers.json` → ablation): the flow model catches the rate "
           "spike, and the host-pair model catches some scan probes. In the full system "
           "these " + str(sum(if_trig.values())) + " flows are credited to "
           + ", ".join(f"{t} {n}" for t, n in if_trig.most_common())
           + " (the trigger is whatever first put the flow at >= CLONE, so a DPI-head "
           "escalation that comes first takes the credit). SQLi is never caught without DPI "
           f"({io['per_class'].get('sqli', {}).get('detected', 0)}/"
           f"{io['per_class'].get('sqli', {}).get('n', 0)}).",
           "", "**What IF adds on top of DPI-head** (full system − DPI-head only; "
           "inspection compared estimate-to-estimate, since (b) has no measured counters):",
           "",
           "| | DPI-head only | full system | Δ (full − DPI-head) |", "|---|---|---|---|",
           f"| TPR | {fmt(do['tpr']['rate'])} | {fmt(fs['tpr']['rate'])} | "
           f"{fs['tpr']['rate'] - do['tpr']['rate']:+.3f} "
           f"({fs['counts']['tp'] - do['counts']['tp']:+d} attack flows) |",
           f"| FPR | {fmt(do['fpr']['rate'])} | {fmt(fs['fpr']['rate'])} | "
           f"{fs['fpr']['rate'] - do['fpr']['rate']:+.3f} "
           f"({fs['counts']['fp'] - do['counts']['fp']:+d} benign flows) |",
           f"| inspection reduction (estimated) | {pct(do['inspection_reduction_estimated'])} | "
           f"{pct(fs['inspection_reduction_estimated'])} | "
           f"{100 * (fs['inspection_reduction_estimated'] - do['inspection_reduction_estimated']):+.1f} pp |",
           "", "Per class, attack flows detected (DPI-head only → full system): " + ", ".join(
               f"{k2} {do['per_class'][k2]['detected']} → {fs['per_class'][k2]['detected']}"
               for k2 in sorted(fs["per_class"]) if k2 != "benign")
           + f". (Measured full-system inspection reduction: {pct(fs['inspection_reduction'])}.)"]
    md += ["", "**How far to trust (a) and (b).**",
           f"- *Level replay:* on flow slots and host pairs that no DPI action ever touched, "
           f"the IF-only replay reproduces the logged level on "
           f"{pct(NUM.get('replay_agreement_flows', float('nan')), 2)} of flow rows and "
           f"{pct(NUM.get('replay_agreement_pairs', float('nan')), 2)} of pair rows, and replaying "
           f"the logged levels reproduces the full system's per-flow detections exactly. "
           f"Of the {NUM['ablation_consistency']['dpi_fp']} benign flows (b) flags, "
           f"{NUM['ablation_consistency']['dpi_fp_also_full']} are also flagged by the full "
           f"system (credited there to "
           + ", ".join(f"{t} {n}" for t, n in fp_trig.most_common()) + ")"
           + ("" if NUM["ablation_consistency"]["dpi_fp"] ==
              NUM["ablation_consistency"]["dpi_fp_also_full"] else
              "; the rest were flagged only in the replay, whose de-escalation and timing "
              "differ from the logged run's")
           + ", and (b) detects "
           + ("no attack flow the full system missed" if not NUM["ablation_consistency"]
              ["dpi_new_tp"] else f"{NUM['ablation_consistency']['dpi_new_tp']} attack "
              "flows the full system missed") + ".",
           f"- *Inspection estimate:* the clone estimator, run on the full system's logged "
           f"levels, matches the measured data-plane counters to within "
           f"{min(err) * 100:+.1f}% to {max(err) * 100:+.1f}% of total clones per run "
           f"(forwarded packets within {max(err_fwd) * 100:.2f}%"
           + (f"; dropped packets {drops_est} estimated vs {drops_meas} measured over the "
              f"{len(validation)} runs" if DPI_ACTION == "DROP" else "")
           + ")"
           + ("; it overestimates drops because it treats a rule as in force from the start "
              "of the next poll window, while the controller installs it at the end of the "
              "poll that decided it (up to a few hundred ms later) — chiefly a source drop "
              "landing as a benign iperf3 flow starts, most of whose first-window packets "
              "pass before the rule is in place" if DPI_ACTION == "DROP" and drops_est >
              1.2 * drops_meas else "")
           + "; (a) and (b) use the same estimator."
           + (" In (a) nothing is dropped (the IF never sets DROP), so packets the full "
              "system dropped count as forwarded there." if DPI_ACTION == "DROP" else ""),
           "- *Not captured:* the traffic, the IF scores and DPI's own decisions are the "
           "logged ones. With a different clone stream DPI could decide differently — e.g. "
           "without sampled iperf3 clones, its syn-scan ratio rule (suppressed by them in "
           "Run C, see RESULTS.md) might fire more often in (b). (b) is therefore a lower "
           "bound on what a DPI-head-only system would confirm, not an exact replay.", ""]
    return md


def replay_agreement(runs):
    agree_f, n_f, agree_p, n_p = 0, 0, 0, 0
    for r in runs:
        fw, pw, w = r["fw"], r["pw"], r["w"]
        ifl, ipl = replay_levels(fw, pw, w, "IF")
        eb = fw.escalated_by.fillna("")
        touched = set(map(tuple, fw.loc[eb.str.startswith("DPI") | eb.str.startswith("dpi"),
                                        ["switch", "slot"]].values))
        for a in r["alerts"]:
            if a["type"] in ("forced_clone", "forced_drop"):
                touched.add((a["switch"], a["flow_idx"]))
        m = ~fw[["switch", "slot"]].apply(tuple, axis=1).isin(touched).to_numpy()
        peb = pw.escalated_by.fillna("")
        pt = set(map(tuple, pw.loc[peb.str.startswith("DPI"), ["switch", "src", "dst"]].values))
        pm = ~pw[["switch", "src", "dst"]].apply(tuple, axis=1).isin(pt).to_numpy()
        agree_f += int((ifl[m] == fw.level.to_numpy()[m]).sum()); n_f += int(m.sum())
        agree_p += int((ipl[pm] == pw.level.to_numpy()[pm]).sum()); n_p += int(pm.sum())
    NUM["replay_agreement_flows"] = agree_f / n_f
    NUM["replay_agreement_pairs"] = agree_p / n_p
    NUM["replay_agreement_rows"] = {"flows": n_f, "pairs": n_p}


# --------------------------------------------------------------------------
# 8. Run-to-run spread
# --------------------------------------------------------------------------

def item8_spread(runs, metrics, md):
    pr = metrics["per_run"]
    rows = []
    for r in runs:
        m = pr[r["name"]]
        rows.append({"run": r["name"], "seed": r["seed"], "tpr": m["overall"]["tpr"],
                     "fpr": m["overall"]["fpr"],
                     "insp_pkts": m["inspection"]["inspection_reduction_packets"],
                     "insp_bytes": m["inspection"]["inspection_reduction_bytes"],
                     "by_reason": {k: v["share_of_forwarded_packets"] for k, v in
                                   (m["inspection"]["clones_by_reason"] or {}).items()},
                     "spikes": f"{m['per_attack_class']['rate_spike']['detected_clone_or_above']}/"
                               f"{m['per_attack_class']['rate_spike']['n']}"})
    df = pd.DataFrame(rows)
    agg = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1)),
               "min": float(df[c].min()), "max": float(df[c].max())}
           for c in ("tpr", "fpr", "insp_pkts", "insp_bytes")}
    NUM["spread"] = {"per_run": rows, "summary": agg,
                     "pooled_insp_pkts": metrics["inspection"]["inspection_reduction_packets"]}
    md += ["## 8. Run-to-run spread (5 seeds)", "",
           f"Per run from `{SERIES}/metrics.json` → `per_run` (`analysis/evaluate.py`); std is "
           "the sample standard deviation (ddof = 1, n = 5).", "",
           "| run | seed | TPR | FPR | inspection reduction (packets) | (bytes) |",
           "|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['run']} | {x['seed']} | {fmt(x['tpr'])} | {fmt(x['fpr'])} | "
                  f"{pct(x['insp_pkts'])} | {pct(x['insp_bytes'])} |")
    md.append("| **mean ± std** | | " + " | ".join(
        f"{fmt(agg[c]['mean'])} ± {fmt(agg[c]['std'])}" if c in ("tpr", "fpr")
        else f"{pct(agg[c]['mean'])} ± {pct(agg[c]['std'])}" for c in
        ("tpr", "fpr", "insp_pkts", "insp_bytes")) + " |")
    md.append("| **min / max** | | " + " | ".join(
        f"{fmt(agg[c]['min'])} / {fmt(agg[c]['max'])}" if c in ("tpr", "fpr")
        else f"{pct(agg[c]['min'])} / {pct(agg[c]['max'])}" for c in
        ("tpr", "fpr", "insp_pkts", "insp_bytes")) + " |")
    ip, ib = agg["insp_pkts"], agg["insp_bytes"]
    md += ["", f"**Inspection reduction across the 5 seeds:** packets {pct(ip['mean'])} ± "
           f"{pct(ip['std'])} (min {pct(ip['min'])}, max {pct(ip['max'])}); bytes "
           f"{pct(ib['mean'])} ± {pct(ib['std'])} (min {pct(ib['min'])}, max "
           f"{pct(ib['max'])}). Per run, clones by reason as a share of forwarded packets "
           "(level / host-pair / sampling / head) and rate spikes caught: "
           + "; ".join(
               f"{x['run']} " + " / ".join(pct(x['by_reason'].get(k, float('nan')))
                                           for k in ("level", "hostpair", "sample", "head"))
               + f", spikes {x['spikes']}" for x in rows)
           + " (`metrics.json` → per_run → inspection → clones_by_reason). The pooled value "
           "in the headline "
           f"({pct(NUM['spread'].get('pooled_insp_pkts', float('nan')))}) weights runs by "
           "their traffic; the mean here weights each seed equally."]
    md.append("")
    return md


# --------------------------------------------------------------------------
# 9. Parameters
# --------------------------------------------------------------------------

def item9_params(runs, md):
    model_ref = code_ref("controller/monitor.py", r"self.model\s+= FlowIsolationForest")
    md += ["## 9. Exact parameter values", "",
           "### Thresholds (per run)", "",
           "One flow model and one host-pair model per run, **shared by both switches** "
           f"(`controller/monitor.py` builds a single `FlowIsolationForest` and "
           f"`HostPairIsolationForest`, {model_ref}), "
           "trained once on warm-up vectors from both switches — so there are no per-switch "
           "thresholds; the values below apply to s1 and s2 alike. MONITOR if score < p5, "
           "CLONE if score < p1 of the model's own warm-up scores "
           f"({code_ref('controller/model.py', r'monitor=p5, clone=p1')}). Source: "
           f"`{SERIES}/run_0N/ctrl/model.json`.", "",
           "| run | seed | flow n_warmup | flow p5 (MONITOR) | flow p1 (CLONE) | flow min | "
           "pair n_warmup | pair p5 | pair p1 | pair min | head N read back (s1, s2) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    thr = []
    for r in runs:
        fm, pm = r["model"]["flow_model"], r["model"]["host_pair_model"]
        rb = r["model"]["data_plane"]["head_clone_n_readback"]
        thr.append({"run": r["name"], "flow": fm["thresholds"], "pair": pm["thresholds"],
                    "head_readback": rb})
        md.append(f"| {r['name']} | {r['seed']} | {fm['n_train']} | "
                  f"{fmt(fm['thresholds']['monitor'], 4)} | {fmt(fm['thresholds']['clone'], 4)} | "
                  f"{fmt(fm['thresholds']['warmup_min'], 4)} | {pm['n_train']} | "
                  f"{fmt(pm['thresholds']['monitor'], 4)} | {fmt(pm['thresholds']['clone'], 4)} | "
                  f"{fmt(pm['thresholds']['warmup_min'], 4)} | {rb.get('1')}, {rb.get('2')} |")
    NUM["thresholds"] = thr
    md += ["", "**p5 = 0 in every run** (to floating-point precision) because the "
           "MONITOR threshold coincides with sklearn's `contamination = 0.05` boundary: "
           "`IsolationForest.decision_function` is the raw score minus `offset_`, and "
           "`offset_` is set to the 5th percentile of the training scores when "
           "contamination = 0.05. The warm-up 5th percentile of `decision_function` is "
           "therefore 0 by construction — the MONITOR threshold is exactly the boundary "
           "sklearn itself would use to call a sample an outlier. The CLONE threshold "
           "(p1) is the only one that varies by run.", ""]

    cfg = runs[0]["cfg"]
    spans = [p for r in runs for p in r["cfg"]["schedule"]["cycles"]]
    iperf = [p["iperf_mbps"] for p in spans]
    benign = [p["benign_s"] for p in spans]
    wu_iperf = [r["cfg"]["schedule"]["warmup"]["iperf_mbps"] for r in runs]
    periods = np.concatenate([r["w"][r["w"].phase == "scored"].ts.diff().dropna().to_numpy()
                              for r in runs])
    n_warm = [int((r["w"].phase == "warmup").sum()) for r in runs]
    # measured per-class packet rates on s1 (rate spike, iperf3) and packets per flow
    spike_rates, iperf_rates, pkts = [], [], defaultdict(list)
    for r in runs:
        f1 = r["fw"][r["fw"].switch == 1]
        mx = f1.groupby("key").pkt_count.max()
        for g in r["gt"].itertuples(index=False):
            n = mx.get(g.key)
            if n is None:
                continue
            cls = g.attack_class or g.generator
            pkts[cls].append(n)
            dur = g.end_ts - g.start_ts
            if g.attack_class == "rate_spike" and dur > 0:
                spike_rates.append(n / dur)
            if g.generator == "iperf3_tcp" and dur > 0:
                iperf_rates.append(n / dur)
    p = [
        ("Flow-head clone N", f"{cfg['head_clone_n']} packets per new flow; read back "
         f"{{1: 3, 2: 3}} in every run (table above)", f"`{SERIES}/run_0N/run_config.json`, "
         f"`model.json` → data_plane; {code_ref('p4/adaptmon.p4', r'head_val <= head_n')}"),
        ("Adaptive sampling", "NORMAL 1-in-64, MONITOR 1-in-8 (of the slot's packet "
         "counter), CLONE/DROP every packet",
         f"{code_ref('p4/adaptmon.p4', r'pkt_val & 63')}, {code_ref('p4/adaptmon.p4', r'pkt_val & 7')}"),
        ("Hysteresis", "escalate after K_UP = 2 consecutive windows above the current "
         "level; de-escalate after K_DOWN = 3 below; DPI confirmations bypass it",
         f"{code_ref('controller/policy.py', r'^K_UP')}, {code_ref('controller/policy.py', r'^K_DOWN')}"),
        ("DPI syn-scan rule", "SYN-only share > 0.9 AND >= 5 distinct dst ports over the "
         "host pair's cloned TCP packets of the last 3.0 s",
         f"{code_ref('controller/dpi.py', r'^SYN_ONLY_RATIO')}, {code_ref('controller/dpi.py', r'^SYN_MIN_PORTS')}, "
         f"{code_ref('controller/dpi.py', r'^SYN_PAIR_WINDOW_SECS')}"),
        ("DPI other rules", "null_scan: TCP flags == 0; sqli: 4 regexes on HTTP payload "
         "(OR 1=1, UNION SELECT, --, /*)", code_ref("controller/dpi.py", r"^SQLI_PATTERNS")),
        ("DPI confirmation retry", "a confirmation for a flow not active this poll is "
         "retried for 10 polls", code_ref("controller/monitor.py", r"^CONFIRMATION_RETRY_WINDOWS")),
        ("IsolationForest", "n_estimators = 100, max_samples = 256, contamination = 0.05, "
         "random_state = 42; log1p on flow pkt_count, byte_count, duration_s, pkt_rate and on "
         "pair new_flows, distinct_dst_ports, pkt_count; >= 50 warm-up samples to train",
         f"{code_ref('controller/model.py', r'^IF_PARAMS')}, "
         f"{code_ref('controller/model.py', r'^FLOW_LOG1P_FEATURES')}, "
         f"{code_ref('controller/model.py', r'^MIN_TRAIN_SAMPLES')}"),
        ("IF features", "flow: pkt_count, byte_count, duration_s, pkt_rate, mean_pkt_size, "
         "entropy (payload byte entropy of the flow's DPI clones when any, else "
         "flow-distribution entropy -- Decision C); host pair: new_flows, "
         "distinct_dst_ports, syn_only_ratio, pkt_count (per poll window)",
         f"{code_ref('controller/features.py', r'^FEATURE_NAMES')}, "
         f"{code_ref('controller/features.py', r'^PAIR_FEATURE_NAMES')}"),
        ("Poll interval", f"1.0 s configured; measured mean {periods.mean():.3f} s, P95 "
         f"{np.percentile(periods, 95):.3f} s over scored polls",
         f"{code_ref('experiments/paper_run.py', r'--poll-interval')}; `windows.csv` ts"),
        ("Slot aging", "a slot unchanged for 3 polls is cleared",
         code_ref("controller/controller.py", r"--flow-idle-polls")),
        ("Warm-up", f"{cfg['warmup_s']} s benign-only before the one-time training "
         f"(controller start-up 4 s before that); "
         + (f"{n_warm[0]} warm-up polls in every run" if len(set(n_warm)) == 1
            else f"{min(n_warm)}–{max(n_warm)} warm-up polls per run"), f"{code_ref('experiments/paper_run.py', r'^WARMUP_S')}, "
         f"{code_ref('experiments/paper_run.py', r'^CTRL_STARTUP_S')}; `windows.csv` phase"),
        ("Cycles / phase gap", f"{cfg['n_cycles']} cycles per run; {cfg['phase_gap_s']} s "
         "between phases", f"`run_config.json`; {code_ref('experiments/paper_run.py', r'^PHASE_GAP_S')}"),
        ("Benign: iperf3 TCP h1→h3", f"seeded per cycle, {min(iperf):.2f}–{max(iperf):.2f} "
         f"Mbit/s (warm-up {min(wu_iperf):.2f}–{max(wu_iperf):.2f}); measured "
         f"{np.median(iperf_rates):.0f} pkt/s median per flow on s1; benign phase "
         f"{min(benign):.2f}–{max(benign):.2f} s", "`run_config.json` → schedule; "
         "`flow_windows.csv` pkt_count"),
        ("Benign: HTTP POST h2→h4:8000", "one request per new TCP connection, every 1.0 s "
         "± 30% (warm-up every 1.5 s ± 30%), random benign body",
         f"{code_ref('experiments/paper_run.py', r'^HTTP_JITTER')}; `traffic/paper_traffic.py` HttpBenign"),
        ("Attack: nmap -sS / -sN h1→h3", f"8 ports per scan (seeded, 1–1023), --max-rate "
         f"{cfg['nmap_max_rate']} pkt/s, 1 probe per port per flow (median "
         f"{np.median(pkts['syn_scan']):.0f} / {np.median(pkts['null_scan']):.0f} packets per "
         f"flow)", f"`run_config.json`; {code_ref('experiments/paper_run.py', r'max_rate=500')}"),
        ("Attack: SQLi POST h2→h4:8000", f"4 requests, {cfg['sqli_gap_s']} s apart, one per "
         f"SQLi pattern, seeded order (median {np.median(pkts['sqli']):.0f} packets per flow)",
         code_ref("experiments/paper_run.py", r"^SQLI_GAP_S")),
        ("Attack: UDP rate spike h2→h4:9999", f"{cfg['rate_spike_pps']} pkt/s for "
         f"{cfg['rate_spike_s']} s, 64-byte payload; measured {np.median(spike_rates):.0f} "
         "pkt/s median on s1", f"`run_config.json`; {code_ref('traffic/paper_traffic.py', r'payload_size: int = 64')}"),
    ]
    if DPI_ACTION == "DROP":     # Run D on; cites the current code
        p[3:3] = [
            ("DPI confirmation action", (
                "DROP: the flow's 5-tuple drop rule, plus a source-IP drop rule for syn/null "
                "scans (a dropped packet is still cloned to DPI)"),
             f"`{SERIES}/run_0N/ctrl/model.json` → data_plane; "
             f"{code_ref('controller/monitor.py', r'force_drop[(]key[)]')}"),
            ("Drop rule removal", "after K_DOWN = 3 scored windows below DROP (the IF never "
             "sets DROP, so a forced DROP dwells 3 scored windows unless confirmed again), or "
             "when the flow's slot ages out (3 idle polls), whichever comes first; a source "
             "drop after its host pair's K_DOWN windows",
             f"{code_ref('controller/policy.py', r'^K_DOWN')}, "
             f"{code_ref('controller/monitor.py', r'sw.delete_drop_rule[(]st.rule_handle[)]')}"),
        ]
    md += ["### System and traffic", "", "| parameter | value | source |", "|---|---|---|"]
    for k, v, s in p:
        md.append(f"| {k} | {v} | {s} |")
    md.append("")
    NUM["params"] = {k: v for k, v, _ in p}
    return md


# --------------------------------------------------------------------------
# 10. Drops (series where a DPI confirmation forces DROP)
# --------------------------------------------------------------------------

def _drop_episodes(r):
    """Consecutive polls a flow slot (per switch) or a host pair spent at
    DROP, and how each episode ended: "dwell" = the level fell below DROP
    while the slot was still occupied (K_DOWN windows, controller/policy.py),
    "slot aged" = the slot had no row at the next poll (cleared after 3 idle
    polls, which also deletes the rule), "run end"."""
    last_w = int(r["w"].window.max())
    cls = dict(zip(r["gt"].key, r["gt"].attack_class.fillna("benign")))
    out = []
    for kind, df, keys in (("flow", r["fw"], ["switch", "key"]),
                           ("source", r["pw"], ["switch", "src", "dst"])):
        for k, g in df.sort_values("window").groupby(keys):
            wins, lv = g.window.to_numpy(), g.level.to_numpy()
            i = 0
            while i < len(g):
                if lv[i] < DROP:
                    i += 1
                    continue
                j = i
                while j + 1 < len(g) and lv[j + 1] >= DROP and wins[j + 1] == wins[j] + 1:
                    j += 1
                if j + 1 < len(g) and wins[j + 1] == wins[j] + 1:
                    end = "dwell"
                elif wins[j] >= last_w:
                    end = "run end"
                else:
                    end = "slot aged"
                out.append({"kind": kind, "class": cls.get(k[1], "not in ground truth")
                            if kind == "flow" else "host pair",
                            "polls": int(wins[j] - wins[i] + 1), "end": end})
                i = j + 1
    return out


def item10_drops(runs, metrics, md):
    pc, b = metrics["per_attack_class"], metrics["benign"]
    da = metrics["drop_accounting"]
    md += [f"## 10. Drops ({RUN}: DPI confirmation → DROP)", "",
           "Per ground-truth flow, from the switch's own per-slot drop counter (`drop_pkts` in "
           f"`{SERIES}/run_0N/ctrl/flow_windows.csv`, a P4 register incremented for every "
           "packet the 5-tuple or source drop rule discarded; `p4/adaptmon.p4`), summed over "
           "both switches and over slot re-occupations; packets seen = the slot packet counter, "
           "max over switches. **Time to drop** = first poll at which a drop applied to the "
           "flow (its own level DROP, or a host pair of its source at DROP) minus the first "
           "packet — per phase (one class in one cycle, the primary figure: from the phase's "
           "first packet to the first drop on any of its flows; 0 when a source drop from an "
           "earlier phase was still in force) and per flow (secondary). **Delivered after "
           "drop** = packets of a dropped flow still forwarded (by every switch on its path) "
           "in polls wholly after its drop was installed, i.e. after the dwell or slot aging "
           f"lifted the rule. All pooled over the {len(runs)} runs; definitions in "
           f"`analysis/evaluate.py` (module docstring). Source: `{SERIES}/metrics.json`.", "",
           "| class | flows | flows with a dropped packet | packets dropped / seen | "
           "delivered after drop (flows) | phases with a drop | time to drop per phase, "
           "median / P95 / max (s) | per flow, median / P95 (s) |",
           "|---|---|---|---|---|---|---|---|"]
    rows = {}
    for c in CLASSES:
        d, t = pc[c]["packet_drops"], pc[c]["time_to_drop"]
        ph, fl = t["per_phase_s"], t["per_flow_s"]
        rows[c] = {"packet_drops": d, "time_to_drop": t}
        ph_s = (f"{ph['median']:.2f} / {ph['p95']:.2f} / {ph['max']:.2f}" if ph.get("n")
                else "—")
        fl_s = f"{fl['median']:.2f} / {fl['p95']:.2f}" if fl.get("n") else "—"
        in_force = t["n_phases_drop_in_force_at_start"]
        md.append(f"| {c} | {pc[c]['n']} | {d['flows_with_dropped_packets']} | "
                  f"{d['packets_dropped']:,} / {d['packets_seen']:,} "
                  f"({pct(d['dropped_share'])}) | {d['packets_delivered_after_drop']:,} "
                  f"({d['flows_with_delivery_after_drop']}) | {t['n_phases_dropped']} / "
                  f"{t['n_phases']}" + (f" ({in_force} already in force)" if in_force else "")
                  + f" | {ph_s} | {fl_s} |")
    bd = b["packet_drops"]
    md.append(f"| **benign** | {b['n']} | {bd['flows_with_dropped_packets']} | "
              f"{bd['packets_dropped']:,} / {bd['packets_seen']:,} ({pct(bd['dropped_share'], 2)}) "
              "| — | — | — | — |")
    md += ["", f"**Benign packets wrongly dropped:** {bd['packets_dropped']:,} of "
           f"{bd['packets_seen']:,} benign packets seen, in {bd['flows_with_dropped_packets']} of "
           f"{b['n']} benign flows (flows whose own level reached DROP: {b['n_dropped']})."
           + (" A source drop for a scan discards everything from that host, but in this "
              "schedule benign traffic runs only in each cycle's benign phase, before its "
              "attack phases (`experiments/paper_run.py` `run_cycle`), so a source drop could "
              "overlap it only at the start of the next cycle; this result therefore does not "
              "show what a source drop costs a host that carries benign traffic at the same "
              "time." if bd["packets_dropped"] == 0 else
              " A source drop for a scan discards everything from that host: benign flows of "
              "the scanning host active while it was in force are dropped too."),
           "", f"**Accounting:** the switches' own cumulative drop counters (`drop_counter`, "
           f"`windows.csv` s1/s2_dropped, final values) total {da['switch_dropped_packets']:,} "
           f"packets ({da['switch_dropped_bytes']:,} bytes); the per-flow sums above attribute "
           f"{da['attributed_attack']:,} to attack flows and {da['attributed_benign']:,} to "
           f"benign flows ({pct(da['attributed_share'])} of the switch total)"
           + (". The rest are packets dropped after a flow's last poll (the slot was cleared "
              "before the next read) or in flows not in ground truth." if
              da["attributed_attack"] + da["attributed_benign"] < da["switch_dropped_packets"]
              else "."), ""]
    # classes whose phases got a drop that came after their flows had ended
    late = []
    for c in CLASSES:
        d, t = pc[c]["packet_drops"], pc[c]["time_to_drop"]
        if t["n_phases_dropped"] and d["packets_seen"] and d["dropped_share"] < 0.05:
            g = pd.concat([r["gt"][r["gt"].attack_class == c] for r in runs])
            late.append(f"{c} (flow duration median {(g.end_ts - g.start_ts).median():.2f} s, "
                        f"first packet → drop median {t['per_flow_s']['median']:.2f} s; "
                        f"{pct(d['dropped_share'])} of its packets dropped)")
    if late:
        md += ["**Drop after the fact:** " + "; ".join(late) + ". Each of these flows is one "
               "short connection or probe that had ended before its confirmation was acted on "
               "at the next poll, so its 5-tuple drop rule had nothing left to drop; and a "
               "5-tuple rule does not match the next connection or probe (a new source port). "
               "Only a source drop (syn/null scans) catches later flows.", ""]
    NUM["drop_after_the_fact"] = late
    eps = [dict(e, run=r["name"]) for r in runs for e in _drop_episodes(r)]
    ep = pd.DataFrame(eps)
    md += ["**How long a DROP lasted** (per switch; flow slots in `flow_windows.csv`, "
           "source drops as host pairs in `pair_windows.csv`):", "",
           "| population | DROP episodes | polls at DROP, median (min–max) | ended by dwell | "
           "ended by slot aging | still at DROP at run end |", "|---|---|---|---|---|---|"]
    ep_sum = {}
    if len(ep):
        for name, g in ep.groupby("class"):
            ends = g.end.value_counts()
            ep_sum[name] = {"n": int(len(g)), "polls_median": float(g.polls.median()),
                            "polls_min": int(g.polls.min()), "polls_max": int(g.polls.max()),
                            **{e: int(ends.get(e, 0)) for e in ("dwell", "slot aged", "run end")}}
            v = ep_sum[name]
            md.append(f"| {name} | {v['n']} | {v['polls_median']:.0f} ({v['polls_min']}–"
                      f"{v['polls_max']}) | {v['dwell']} | {v['slot aged']} | {v['run end']} |")
    md += ["", "A forced DROP is lifted after K_DOWN = 3 scored windows below DROP (every "
           "occupied slot is scored each poll and the IF never sets DROP), or earlier if the "
           "slot ages out — the removal logic is the same as before the change. A TCP flow "
           "that keeps retransmitting after the rule is lifted gets through: that is the "
           "\"delivered after drop\" column (on a real switch: `tests/smoke_drop.py`, whose "
           "report records the retransmission delivered once aging lifted the rule).", ""]
    NUM["drops"] = {"per_class": rows, "benign": bd, "accounting": da, "episodes": ep_sum,
                    "attack": {"packet_drops": metrics["attack_packet_drops"],
                               "time_to_drop": metrics["attack_time_to_drop"]}}

    # figure: time to drop per phase, by class
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    rng = np.random.default_rng(0)
    ymax = 0.0
    for i, c in enumerate(CLASSES):
        vals = []
        for r in runs:
            g = r["gt"][r["gt"].attack_class == c]
            ph = g.groupby("phase").agg(start=("start_ts", "min"), drop=("drop_ts", "min"))
            vals += np.maximum(0.0, (ph["drop"] - ph["start"]).dropna()).tolist()
        n_ph = sum(r["gt"][r["gt"].attack_class == c].phase.nunique() for r in runs)
        if vals:
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=14, color=CAT[i],
                       edgecolors=SURF, linewidths=0.8, zorder=3)
            ax.hlines(np.median(vals), i - 0.28, i + 0.28, color=INK, lw=1.6, zorder=4)
            ymax = max(ymax, max(vals))
        ax.text(i, -0.08, f"{len(vals)}/{n_ph} phases", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color=INK2)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, max(1.0, ymax * 1.15))
    ax.set_ylabel("first packet → drop (s)")
    ax.set_title("Time to drop, per attack phase (bar = median)", loc="left")
    save(fig, "fig_time_to_drop")
    md += ["`fig_time_to_drop.pdf/.png` — one dot per attack phase that got a drop (all "
           f"{len(runs)} runs); bar = median; below each class, phases with a drop / all "
           "phases.", ""]
    return md


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default=SERIES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--name", default=RUN, help='e.g. "Run D"')
    a = ap.parse_args(argv)
    configure(a.series, a.out, a.name)
    global BENCH, DPI_ACTION
    BENCH = pick_bench()
    os.makedirs(os.path.join(ROOT, OUT), exist_ok=True)
    runs = load_runs()
    actions = {r["model"]["data_plane"].get("dpi_confirmation_action", "CLONE").split()[0]
               for r in runs}
    assert len(actions) == 1, f"mixed DPI confirmation actions in {SERIES}: {actions}"
    DPI_ACTION = actions.pop()
    metrics = ev.evaluate_series(os.path.join(ROOT, SERIES))
    replay_agreement(runs)
    seeds = ", ".join(f"{r['name']} {r['seed']}" for r in runs)
    n_gt = sum(len(r["gt"]) for r in runs)
    md = [f"# {RUN} — paper numbers", "",
          f"Generated by `analysis/paper_runC.py` from **{RUN} only** (`{SERIES}`, {len(runs)} "
          f"runs, seeds {seeds}; {n_gt:,} ground-truth flows) — except item 1's latency tiers, "
          f"which come from a separate, labelled latency-only benchmark (`{BENCH}`). Every "
          f"number names its source. Definitions as in `RESULTS.md` / `analysis/evaluate.py`; "
          "CIs are Clopper-Pearson 95%.", ""]
    if DPI_ACTION == "DROP":
        md += [f"**What changed from Run C:** only the action on a DPI confirmation — a "
               "confirmation on a head, sampled or level clone now forces the flow straight to "
               "DROP (5-tuple drop rule) and, for syn/null scans, its source IP (source drop "
               "rule); Run C force-cloned it instead. IF thresholds, head N, sampling, "
               "hysteresis and traffic are unchanged (item 9; `model.json` → data_plane → "
               "`dpi_confirmation_action`). Drop metrics: item 10.", ""]
    md += [f"Headline (for orientation; `{SERIES}/metrics.json`): flow-level TPR "
           f"{fmt(metrics['overall']['tpr'])}, FPR {fmt(metrics['overall']['fpr'])}, inspection "
           f"reduction {pct(metrics['inspection']['inspection_reduction_packets'])} packets / "
           f"{pct(metrics['inspection']['inspection_reduction_bytes'])} bytes.", ""]
    md = item1_latency(runs, md)
    md = item2_level_share(runs, md)
    md = item3_window_confusion(runs, md)
    md = item4_scores(runs, md)
    md = item5_trigger_bars(metrics, md)
    md = item6_timeline(runs, metrics, md)
    md = item7_ablation(runs, metrics, md)
    md = item8_spread(runs, metrics, md)
    md = item9_params(runs, md)
    if DPI_ACTION == "DROP":
        md = item10_drops(runs, metrics, md)
    with open(os.path.join(ROOT, OUT, "PAPER_NUMBERS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(ROOT, OUT, "numbers.json"), "w") as f:
        json.dump(NUM, f, indent=1, default=str)
    print(os.path.join(OUT, "PAPER_NUMBERS.md"))


if __name__ == "__main__":
    main()
