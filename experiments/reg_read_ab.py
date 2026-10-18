#!/usr/bin/env python3
"""
experiments/reg_read_ab.py -- cost of the drop_pkts register (Run D) in the
controller's per-poll register read, on a live bmv2 (run inside
scripts/sandbox.sh):

    scripts/sandbox.sh python3 -m experiments.reg_read_ab [--reps 30]

controller/dataplane.py reads every flow register array in full each poll.
Run D added one array (drop_pkts) to FLOW_REGISTERS; the latency-only
benchmark (experiments/latency_bench.py) predates it. This times
read_registers() on s1 with and without drop_pkts, interleaved (A, B, A,
B, ...) so host load affects both alike. No traffic: the read cost does not
depend on the number of active flows (whole arrays are read).
Output: results/reg_read_ab_<timestamp>/report.json
"""

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from topo.namespace_topo import NetworkNamespaceTopo                 # noqa: E402
from controller.dataplane import DataPlaneManager, FLOW_REGISTERS    # noqa: E402
from controller.utils import get_timestamp_str                       # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=30)
    args = p.parse_args()
    out = os.path.join(ROOT, "results", f"reg_read_ab_{get_timestamp_str()}")
    os.makedirs(out, exist_ok=True)

    topo = NetworkNamespaceTopo(log_dir=os.path.join(out, "switches"))
    topo.teardown()
    topo.build()
    assert topo.start_switches()
    time.sleep(1)
    sw = DataPlaneManager([{"switch_id": 1, "thrift_port": 9090}]).get_switch(1)
    variants = {"with_drop_pkts": list(FLOW_REGISTERS),
                "without_drop_pkts": [r for r in FLOW_REGISTERS if r != "drop_pkts"]}
    ms = {k: [] for k in variants}
    sw.read_registers(variants["with_drop_pkts"])      # warm-up, not timed
    for _ in range(args.reps):
        for k, regs in variants.items():
            t = time.perf_counter()
            sw.read_registers(regs)
            ms[k].append((time.perf_counter() - t) * 1e3)
    topo.stop_switches()
    topo.teardown()

    summary = {k: {"n_registers": len(variants[k]), "reps": len(v),
                   "median_ms": statistics.median(v), "mean_ms": statistics.mean(v),
                   "min_ms": min(v), "max_ms": max(v)} for k, v in ms.items()}
    diff = [a - b for a, b in zip(ms["with_drop_pkts"], ms["without_drop_pkts"])]
    report = {"summary": summary, "paired_diff_ms_median": statistics.median(diff),
              "load_avg_after": os.getloadavg(), "raw_ms": ms}
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    for k, v in summary.items():
        print(f"{k}: {v['n_registers']} registers, median {v['median_ms']:.1f} ms "
              f"(min {v['min_ms']:.1f}, max {v['max_ms']:.1f})")
    print(f"paired difference, median: {report['paired_diff_ms_median']:+.1f} ms")
    print(f"report: {os.path.join(out, 'report.json')}")


if __name__ == "__main__":
    main()
