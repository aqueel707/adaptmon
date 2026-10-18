"""Unit tests for analysis/evaluate.py on small synthetic run directories."""

import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import evaluate as ev  # noqa: E402

FLOW_HDR = (["ts", "window", "switch", "slot", "src", "dst", "proto", "sport", "dport"]
            + ["entropy_source", "pkt_delta", "scored", "score", "raw_level", "level",
               "escalated_by"])
PAIR_HDR = ["ts", "window", "switch", "src", "dst", "scored", "score", "raw_level", "level",
            "escalated_by"]
A, B, C, D = "10.0.1.1", "10.0.2.1", "10.0.1.2", "10.0.2.2"


def gt(src, dst, sport, dport, label, cls, start, end, proto=6):
    return dict(src=src, dst=dst, proto=proto, sport=sport, dport=dport, label=label,
                attack_class=cls, generator="g", start_ts=start, end_ts=end)


def frow(ts, src, dst, sport, dport, level, by="", proto=6):
    return dict(ts=ts, window=int(ts), switch=1, slot=sport, src=src, dst=dst, proto=proto,
                sport=sport, dport=dport, entropy_source="x", pkt_delta=1, scored=1,
                score=0.0, raw_level=level, level=level, escalated_by=by)


def windows(rows):
    """rows: (ts, s1_fwd, s1_cloned, s2_fwd, s2_cloned) cumulative"""
    return pd.DataFrame([dict(ts=t, window=i + 1, phase="scored", s1_fwd=a, s1_cloned=b,
                              s2_fwd=c, s2_cloned=d, t_total_ms=10, t_read_ms=5,
                              cpu_pct=1.0, rss_mb=100.0, dpi_cpu_pct=2.0, dpi_rss_mb=50.0)
                         for i, (t, a, b, c, d) in enumerate(rows)])


def write_run(d, gts, flows, pairs, alerts=(), win=None):
    os.makedirs(os.path.join(d, "ctrl"), exist_ok=True)
    with open(os.path.join(d, "ground_truth.jsonl"), "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gts)
    for name, hdr, rows in (("flow_windows.csv", FLOW_HDR, flows),
                            ("pair_windows.csv", PAIR_HDR, pairs)):
        with open(os.path.join(d, "ctrl", name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(d, "ctrl", "alerts.jsonl"), "w") as f:
        f.writelines(json.dumps(a) + "\n" for a in alerts)
    (win if win is not None else windows([(0, 0, 0, 0, 0), (1, 100, 10, 0, 0)])).to_csv(
        os.path.join(d, "ctrl", "windows.csv"), index=False)


# --- inspection: cumulative counters --------------------------------------

def test_inspection_uses_final_cumulative_counters_not_their_sum():
    """Regression: windows.csv counters are cumulative; summing them over
    windows (what the first version of this module did) weights later
    windows more and gives the wrong ratio."""
    w = windows([(0, 0, 0, 0, 0), (1, 100, 90, 0, 0), (2, 1000, 100, 0, 0)])
    insp = ev.inspection([ev.run_totals(w)])
    assert insp["forwarded_packets"] == 1000 and insp["cloned_packets"] == 100
    assert insp["inspection_reduction_packets"] == pytest.approx(0.9)
    # the old, wrong computation for contrast: 1 - 190/(1100+190)
    assert insp["inspection_reduction_packets"] != pytest.approx(1 - 190 / 1290)


def test_inspection_sums_switches_and_runs_final_values():
    r1 = ev.run_totals(windows([(0, 0, 0, 0, 0), (1, 100, 10, 50, 5)]))
    r2 = ev.run_totals(windows([(0, 0, 0, 0, 0), (1, 300, 30, 50, 15)]))
    insp = ev.inspection([r1, r2])
    assert (insp["forwarded_packets"], insp["cloned_packets"]) == (500, 60)
    assert insp["inspection_reduction_packets"] == pytest.approx(1 - 60 / 500)
    assert insp["per_run_reduction_packets"] == pytest.approx([1 - 15 / 150, 1 - 45 / 350])
    assert insp["inspection_reduction_bytes"] is None   # no byte columns in this run


def test_inspection_bytes_and_clone_reasons_when_recorded():
    w = windows([(0, 0, 0, 0, 0), (1, 100, 20, 0, 0)])
    for col, v in (("s1_fwd_bytes", 10000), ("s1_cloned_bytes", 1500), ("s2_fwd_bytes", 0),
                   ("s2_cloned_bytes", 0)):
        w[col] = [0, v]
    for r, (p, b) in {"head": (12, 900), "sample": (2, 100), "level": (6, 500),
                      "hostpair": (0, 0)}.items():
        for s in (1, 2):
            w[f"s{s}_clone_{r}"] = [0, p if s == 1 else 0]
            w[f"s{s}_clone_{r}_bytes"] = [0, b if s == 1 else 0]
    insp = ev.inspection([ev.run_totals(w)])
    assert insp["inspection_reduction_bytes"] == pytest.approx(1 - 1500 / 10000)
    assert insp["clones_by_reason"]["head"]["packets"] == 12
    assert insp["clones_by_reason"]["head"]["share_of_forwarded_packets"] == pytest.approx(0.12)


def test_capacity_from_counter_deltas():
    w = windows([(0, 0, 0, 0, 0), (1, 1000, 1000, 10, 0), (2, 2500, 2600, 20, 0),
                 (3, 2600, 2610, 30, 0)])
    cap = ev.capacity(w, {"sockets": {"a": {"kernel_received": 9, "kernel_drops": 2}}})
    assert cap["per_switch"][1]["max_pps"] == pytest.approx(3100)
    assert cap["per_switch"][1]["windows_over_budget"] == 1
    assert cap["windows_over_budget"] == 1 and cap["dpi_kernel_drops"] == 2


# --- join / triggers / latency -------------------------------------------

def test_trigger_buckets_including_legacy_tags():
    assert ev._trigger_bucket("DPI-head") == "DPI-head"
    assert ev._trigger_bucket("dpi_confirmed:sqli") == "DPI-drop"
    assert ev._trigger_bucket("flow") == "IF-flow"
    assert ev._trigger_bucket("host_pair") == "IF-pair"
    assert ev._trigger_bucket("") == "unattributed"


def test_join_triggers_latency_and_pair_only_detection(tmp_path):
    d = str(tmp_path / "run_01")
    gts = [
        gt(A, B, 1, 21, "attack", "syn_scan", 10.0, 10.2),      # DPI-head, flow
        gt(A, B, 2, 22, "attack", "syn_scan", 10.0, 10.2),      # pair only (never polled)
        gt(C, D, 3, 8000, "attack", "sqli", 20.0, 20.5),        # missed
        gt(C, D, 4, 8000, "benign", None, 30.0, 31.0),          # benign, IF-flow FP
        gt(C, D, 5, 8000, "benign", None, 40.0, 41.0),          # benign, clean
    ]
    flows = [frow(11.0, A, B, 1, 21, 2, "DPI-head"), frow(12.0, A, B, 1, 21, 2, "DPI-head"),
             frow(21.0, C, D, 3, 8000, 0), frow(31.0, C, D, 4, 8000, 2, "IF-flow"),
             frow(41.0, C, D, 5, 8000, 0)]
    pairs = [dict(ts=9.5, window=9, switch=1, src=A, dst=B, scored=1, score=-1, raw_level=2,
                  level=2, escalated_by="DPI-head")]
    alerts = [{"type": "confirmed", "ts": 10.4, "attack_class": "syn_scan", "clone_reason": 4,
               "five_tuple": {"src": A, "dst": B, "proto": 6, "sport": 1, "dport": 21}}]
    write_run(d, gts, flows, pairs, alerts)
    j = ev.join_run(d).set_index("sport")
    assert j.loc[1, "trigger"] == "DPI-head" and j.loc[1, "detect_latency_s"] == pytest.approx(1.0)
    assert j.loc[1, "dpi_confirmed"] and j.loc[1, "dpi_latency_s"] == pytest.approx(0.4)
    # pair already at CLONE (ts 9.5) before the flow started (10.0): latency 0, not -0.5
    assert not j.loc[2, "observed"] and j.loc[2, "trigger"] == "DPI-head"
    assert j.loc[2, "detect_latency_s"] == 0.0
    assert j.loc[3, "trigger"] == "none" and np.isnan(j.loc[3, "detect_latency_s"])
    assert j.loc[4, "trigger"] == "IF-flow"

    m = ev.evaluate(j.reset_index())
    assert m["overall"]["tp"] == 2 and m["overall"]["fn"] == 1 and m["overall"]["fp"] == 1
    assert m["per_attack_class"]["syn_scan"]["recall_by_trigger"]["DPI-head"]["k"] == 2
    assert m["benign"]["false_positives_by_trigger"]["IF-flow"]["k"] == 1


def test_trigger_is_the_first_detection_not_a_later_forced_drop(tmp_path):
    """IF-flow had the flow at CLONE; a DPI-level confirmation then forced it
    to DROP. The detection is still the IF's; the flow is still blocked."""
    d = str(tmp_path / "run_01")
    gts = [gt(C, D, 1, 8000, "attack", "rate_spike", 10.0, 20.0)]
    flows = [frow(11.0, C, D, 1, 8000, 1), frow(12.0, C, D, 1, 8000, 2, "IF-flow"),
             frow(13.0, C, D, 1, 8000, 3, "DPI-level")]
    write_run(d, gts, flows, [])
    j = ev.join_run(d).iloc[0]
    assert j["trigger"] == "IF-flow" and j["flow_max_level"] == 3 and j["blocked"]
    assert j["detect_latency_s"] == pytest.approx(2.0)


# --- drop metrics (Run D: drop_pkts register) -----------------------------

DROP_HDR = FLOW_HDR + ["pkt_count", "drop_pkts"]


def drow(ts, src, dst, sport, dport, level, pkts, drops, switch=1, by=""):
    return {**frow(ts, src, dst, sport, dport, level, by), "switch": switch,
            "pkt_count": pkts, "drop_pkts": drops}


def write_drop_run(d, gts, flows, pairs, switch_dropped):
    write_run(d, gts, [], pairs)
    with open(os.path.join(d, "ctrl", "flow_windows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DROP_HDR)
        w.writeheader()
        w.writerows(flows)
    win = windows([(0, 0, 0, 0, 0), (40, 100, 10, 50, 5)])
    win["s1_dropped"], win["s2_dropped"] = [0, switch_dropped], [0, 0]
    win["s1_dropped_bytes"], win["s2_dropped_bytes"] = [0, 64 * switch_dropped], [0, 0]
    win.to_csv(os.path.join(d, "ctrl", "windows.csv"), index=False)


def test_drop_packets_time_to_drop_and_accounting(tmp_path):
    d = str(tmp_path / "run_01")
    sq1, sq2, syn, ben = 56000, 56001, 57000, 51000
    gts = [gt(C, D, sq1, 8001, "attack", "sqli", 10.0, 25.0),
           gt(C, D, sq2, 8002, "attack", "sqli", 10.5, 11.0),
           gt(A, B, syn, 22, "attack", "syn_scan", 30.0, 30.1),
           gt(A, B, ben, 8000, "benign", None, 30.5, 33.0)]
    flows = [
        # forced DROP at 11; drops 0 -> 2 -> 4, slot aged, re-occupied: 1 more
        drow(11.0, C, D, sq1, 8001, 3, 3, 0, by="DPI-head"),
        drow(12.0, C, D, sq1, 8001, 3, 7, 2, by="DPI-head"),
        drow(13.0, C, D, sq1, 8001, 3, 9, 4, by="DPI-head"),
        drow(20.0, C, D, sq1, 8001, 0, 2, 1),
        drow(11.0, C, D, sq1, 8001, 3, 3, 0, switch=2, by="DPI-head"),
        drow(20.0, C, D, sq1, 8001, 0, 4, 0, switch=2),     # the 1 s1 let through
        drow(11.0, C, D, sq2, 8002, 0, 4, 0),
        # the scan source is dropped at 31 (pair DROP); its benign flow is hit
        drow(31.0, A, B, syn, 22, 0, 1, 1),
        drow(31.0, A, B, ben, 8000, 0, 5, 2),
    ]
    pairs = [dict(ts=31.0, window=31, switch=1, src=A, dst=B, scored=1, score=-1,
                  raw_level=0, level=3, escalated_by="DPI-head")]
    write_drop_run(d, gts, flows, pairs, switch_dropped=9)
    j = ev.join_run(d).set_index("sport")
    assert (j.loc[sq1, "pkts_dropped"], j.loc[sq1, "pkts_total"]) == (5, 11)
    assert j.loc[sq1, "drop_latency_s"] == pytest.approx(1.0)
    assert j.loc[sq2, "pkts_dropped"] == 0 and np.isnan(j.loc[sq2, "drop_ts"])
    assert j.loc[syn, "drop_latency_s"] == pytest.approx(1.0)
    assert j.loc[ben, "drop_latency_s"] == pytest.approx(0.5)

    m = ev.evaluate_run(d)
    sq = m["per_attack_class"]["sqli"]
    assert sq["packet_drops"] == {"flows_with_dropped_packets": 1, "packets_dropped": 5,
                                  "packets_seen": 15, "dropped_share": 5 / 15,
                                  # re-occupied slot at 20: 2 pkts, 1 dropped, 1 through
                                  "packets_delivered_after_drop": 1,
                                  "flows_with_delivery_after_drop": 1}
    td = sq["time_to_drop"]
    assert (td["n_phases"], td["n_phases_dropped"], td["n_flows_dropped"]) == (1, 1, 1)
    assert td["per_phase_s"]["median"] == pytest.approx(1.0)
    bd = m["benign"]["packet_drops"]
    assert (bd["packets_dropped"], bd["flows_with_dropped_packets"]) == (2, 1)
    da = m["drop_accounting"]
    assert (da["switch_dropped_packets"], da["attributed_attack"], da["attributed_benign"]) \
        == (9, 6, 2)
    assert da["attributed_share"] == pytest.approx(8 / 9)
    assert m["inspection"]["dropped_packets"] == 9
    assert m["inspection"]["inspection_reduction_packets_incl_dropped"] == \
        pytest.approx(1 - 15 / 159)


def test_drop_metrics_are_none_for_runs_without_the_register(tmp_path):
    d = str(tmp_path / "run_01")
    write_run(d, [gt(C, D, 56000, 8000, "attack", "sqli", 10.0, 11.0)],
              [frow(11.0, C, D, 56000, 8000, 3, "DPI-head")], [])
    m = ev.evaluate_run(d)
    assert m["per_attack_class"]["sqli"]["packet_drops"] is None
    assert m["per_attack_class"]["sqli"]["time_to_drop"] is None
    assert m["drop_accounting"] is None and m["benign"]["packet_drops"] is None
