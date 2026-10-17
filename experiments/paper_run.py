#!/usr/bin/env python3
"""
experiments/paper_run.py — paper evaluation runner.

Drives a full run of the real system -- namespaced bmv2 topology, the real
controller (controller/controller.py, i.e. controller/monitor.py's
AdaptMonController) and DPI (controller/dpi.py) as the production CLI would
run them, and traffic.paper_traffic's generators -- and writes everything
analysis/evaluate.py needs to score it:

    scripts/sandbox.sh python3 -m experiments.paper_run --out results/<name> \\
        --runs 5 --seeds S1,S2,S3,S4,S5 --purpose final [--cycles 8]

Unlike tests/smoke_traffic.py and tests/smoke_nmap.py, this installs no
host_clone_table shortcut: detection has to happen the way the deployed
system would do it (per-flow escalation, Decision B's host-pair escalation,
flow-head cloning, and adaptive sampling) -- the run measures how well that
works unassisted.

Traffic schedule. A WARMUP_S benign-only period runs first (iperf3 h1->h3 +
HTTP h2->h4; controller --train-until), then each cycle:
  benign: iperf3 (h1->h3) + HTTP benign (h2->h4), concurrently
  four attack phases, one at a time (Decision D), in the cycle's order:
    syn    nmap -sS h1->h3 on N_SCAN_PORTS ports
    null   nmap -sN h1->h3 on the same ports
    sqli   4 SQLi POSTs h2->h4, SQLI_GAP_S apart (one poll window)
    spike  UDP rate spike h2->h4, --rate-spike-pps for RATE_SPIKE_S
  with PHASE_GAP_S between phases so no two land in one poll window.

Seeds (--seeds, one per run). A seed fixes that run's schedule via
build_schedule(): per cycle the scan ports (8 distinct, 1-1023), the order
of the four attack phases, the benign phase length (6-10 s), the iperf3
bitrate (1-3 Mbit/s), the benign HTTP bodies and gap jitter (+/-30%), and
the SQLi body order; plus the warm-up iperf3 bitrate and HTTP jitter.
Detector parameters are never seeded (IsolationForest random_state stays
fixed in controller/model.py). Without --seeds the fixed schedule runs A
and B used is reproduced (ports 20,120,..,720; order syn,null,sqli,spike;
8 s benign; 2 Mbit/s; no jitter). Every seed used is recorded in
experiments/seeds_used.json at launch, and a seed already recorded there is
refused -- so a final evaluation's seeds provably never ran before.

Rate spike: 1000 pkt/s by default (runs A and B: 1500). At CLONE every
packet is also cloned, so 1500 pkt/s put ~3,100-3,300 pkt/s through a
switch in run B -- over Decision D's ~2,500 budget -- and DPI's capture
sockets dropped clones. 1000 pkt/s keeps CLONE-level load near 2,000.

Every generator call uses a source-port block no other call in the run
uses (_SportAllocator), so every flow has a unique 5-tuple and
analysis/evaluate.py joins ground truth by exact 5-tuple.
"""

import argparse
import datetime
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from controller.utils import ensure_dirs, get_timestamp_str, setup_logging  # noqa: E402
from topo.namespace_topo import NetworkNamespaceTopo, HOSTS                # noqa: E402
from traffic.ground_truth import GroundTruthRecorder                      # noqa: E402
from traffic.paper_traffic import (                                       # noqa: E402
    HttpServer, HttpBenign, HttpSqliAttack, Iperf3Benign, NmapScan,
    RateSpikeAttack, SQLI_BODIES,
)

logger = logging.getLogger(__name__)
H = {h["name"]: h for h in HOSTS}

SWITCH_CONFIGS = [{"switch_id": 1, "thrift_port": 9090},
                  {"switch_id": 2, "thrift_port": 9091}]
DPI_IFACES = "1:am_s1dpi,2:am_s2dpi"
SEED_REGISTRY = os.path.join(ROOT, "experiments", "seeds_used.json")

CTRL_STARTUP_S  = 4     # time for the controller to connect + mirror
WARMUP_S        = 30    # benign-only, before --train-until
SETTLE_S        = 6     # after the last attack, before stopping the controller
PHASE_GAP_S     = 1.5   # >= one poll_interval, keeps phases in separate windows
N_SCAN_PORTS    = 8
SQLI_GAP_S      = 0.1
RATE_SPIKE_S    = 5
ATTACK_PHASES   = ["syn", "null", "sqli", "spike"]
HTTP_JITTER     = 0.3

FIXED_CYCLE = {   # the unseeded schedule runs A and B used
    "scan_ports": [20 + i * 100 for i in range(N_SCAN_PORTS)],
    "order": list(ATTACK_PHASES), "benign_s": 8.0, "iperf_mbps": 2.0,
    "http_seed": None, "sqli_order": list(range(len(SQLI_BODIES))),
}


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------

def build_schedule(seed: Optional[int], n_cycles: int) -> Dict:
    """Everything a run's traffic varies by, as a pure function of the seed."""
    if seed is None:
        return {"seed": None, "warmup": {"iperf_mbps": 1.0, "http_interval": 1.5,
                                         "http_seed": None},
                "cycles": [dict(FIXED_CYCLE) for _ in range(n_cycles)]}
    rng = random.Random(seed)
    warmup = {"iperf_mbps": round(rng.uniform(1.0, 3.0), 2), "http_interval": 1.5,
              "http_seed": rng.randrange(2**31)}
    cycles = []
    for _ in range(n_cycles):
        cycles.append({
            "scan_ports": sorted(rng.sample(range(1, 1024), N_SCAN_PORTS)),
            "order": rng.sample(ATTACK_PHASES, len(ATTACK_PHASES)),
            "benign_s": round(rng.uniform(6.0, 10.0), 2),
            "iperf_mbps": round(rng.uniform(1.0, 3.0), 2),
            "http_seed": rng.randrange(2**31),
            "sqli_order": rng.sample(range(len(SQLI_BODIES)), len(SQLI_BODIES)),
        })
    return {"seed": seed, "warmup": warmup, "cycles": cycles}


def load_seed_registry(path: str = SEED_REGISTRY) -> Dict:
    if not os.path.exists(path):
        return {"seeds": {}}
    with open(path) as f:
        return json.load(f)


def register_seeds(seeds: List[int], purpose: str, series_dir: str,
                   path: str = SEED_REGISTRY) -> None:
    """Refuse any seed already recorded; record the new ones before any run
    starts (a crashed run's seed counts as used too)."""
    reg = load_seed_registry(path)
    reused = [s for s in seeds if str(s) in reg["seeds"]]
    if reused:
        raise SystemExit(f"Seed(s) {reused} already used ({path}): "
                         f"{[reg['seeds'][str(s)] for s in reused]}")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"Duplicate seeds in {seeds}")
    for i, s in enumerate(seeds, 1):
        reg["seeds"][str(s)] = {"purpose": purpose, "series": os.path.relpath(series_dir, ROOT),
                                "run": f"run_{i:02d}",
                                "registered": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(path, "w") as f:
        json.dump(reg, f, indent=2)


# --------------------------------------------------------------------------
# Traffic
# --------------------------------------------------------------------------

class _SportAllocator:
    """
    One disjoint source-port block per (generator, cycle), so no two flows
    in a run ever share a 5-tuple -- see the module docstring.
    """
    # BLOCK must stay well under the smallest gap between consecutive BASES
    # divided by the max cycle count, or two generators' blocks collide once
    # enough cycles run (e.g. nmap_syn=54000 + 5*200 == nmap_null's own base
    # 55000 -- exactly what broke the first full run of this script).
    BLOCK = 20
    BASES = {"iperf3": 50000, "http_benign": 51000, "sqli": 56000,
            "nmap_syn": 57000, "nmap_null": 60000, "rate_spike": 63000}

    def __init__(self, cycle: int):
        self.cycle = cycle

    def __getitem__(self, generator: str) -> int:
        return self.BASES[generator] + self.cycle * self.BLOCK


def run_cycle(gt: GroundTruthRecorder, cycle: int, plan: Dict, spike_pps: int) -> None:
    sport = _SportAllocator(cycle)
    logger.info(f"-- cycle {cycle}: benign {plan['benign_s']}s, "
                f"iperf3 {plan['iperf_mbps']}M; then {plan['order']} --")
    iperf = Iperf3Benign(gt, H["h1"]["ns"], H["h1"]["ip"], H["h3"]["ns"], H["h3"]["ip"],
                        port=5201, duration=plan["benign_s"],
                        bitrate=f"{plan['iperf_mbps']}M", cport=sport["iperf3"])
    http_rng = random.Random(plan["http_seed"]) if plan["http_seed"] is not None else None
    http_benign = HttpBenign(gt, H["h2"]["ns"], H["h2"]["ip"], H["h4"]["ip"],
                             duration=plan["benign_s"], interval=1.0,
                             base_sport=sport["http_benign"], rng=http_rng,
                             interval_jitter=HTTP_JITTER if http_rng else 0.0)
    iperf.start()
    http_benign.start()
    iperf.wait(timeout=plan["benign_s"] + 5)
    iperf.stop()
    http_benign.stop()

    for phase in plan["order"]:
        # Before every attack phase: let the previous phase's packets (incl.
        # TCP teardown) clear the poll window, so each phase's host-pair
        # window-local features are its own (controller/features.py)
        time.sleep(PHASE_GAP_S)
        logger.info(f"-- cycle {cycle}: {phase} --")
        if phase in ("syn", "null"):
            NmapScan(gt, phase, H["h1"]["ns"], H["h1"]["ip"], H["h3"]["ip"],
                     ports=plan["scan_ports"], max_rate=500,
                     base_sport=sport[f"nmap_{phase}"]).run(timeout=20)
        elif phase == "sqli":
            sqli = HttpSqliAttack(gt, H["h2"]["ns"], H["h2"]["ip"], H["h4"]["ip"],
                                  base_sport=sport["sqli"], gap=SQLI_GAP_S,
                                  bodies=[SQLI_BODIES[i] for i in plan["sqli_order"]])
            sqli.start()
            sqli.join(timeout=10)
        else:
            spike = RateSpikeAttack(gt, H["h2"]["ns"], H["h2"]["ip"], H["h4"]["ip"],
                                    dport=9999, duration=RATE_SPIKE_S, rate=spike_pps,
                                    sport=sport["rate_spike"])
            spike.start()
            spike.wait(timeout=RATE_SPIKE_S + 5)
            spike.stop()


def run_once(run_dir: str, schedule: Dict, head_clone_n: int, spike_pps: int) -> Dict:
    ensure_dirs([run_dir])
    n_cycles = len(schedule["cycles"])
    sw_log = os.path.join(run_dir, "switches")
    ensure_dirs([sw_log])
    topo = NetworkNamespaceTopo(log_dir=sw_log)
    topo.teardown()
    topo.build()
    if not topo.start_switches():
        topo.teardown()
        raise RuntimeError("switches failed to start")
    time.sleep(1)
    topo.install_forwarding_rules()

    sw_cfg_path = os.path.join(run_dir, "switch_configs.json")
    with open(sw_cfg_path, "w") as f:
        json.dump(SWITCH_CONFIGS, f)
    config = {"seed": schedule["seed"], "n_cycles": n_cycles, "head_clone_n": head_clone_n,
              "rate_spike_pps": spike_pps, "rate_spike_s": RATE_SPIKE_S,
              "warmup_s": WARMUP_S, "phase_gap_s": PHASE_GAP_S, "sqli_gap_s": SQLI_GAP_S,
              "nmap_max_rate": 500, "schedule": schedule}
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    ctrl_out_dir = os.path.join(run_dir, "ctrl")
    ensure_dirs([ctrl_out_dir])
    attack_at = time.time() + CTRL_STARTUP_S + WARMUP_S
    est_cycle_s = 10 + 15 + RATE_SPIKE_S + 4 * PHASE_GAP_S   # rough safety net
    total_s = CTRL_STARTUP_S + WARMUP_S + n_cycles * est_cycle_s + SETTLE_S + 60

    ctrl_cmd = [
        sys.executable, "-m", "controller.controller",
        "--switches", sw_cfg_path,
        "--out-dir", ctrl_out_dir,
        "--poll-interval", "1.0",
        "--duration", str(total_s),
        "--train-until", str(attack_at),
        "--dpi-ifaces", DPI_IFACES,
        "--head-clone-n", str(head_clone_n),
    ]
    ctrl_stdout = open(os.path.join(run_dir, "controller_out.log"), "w")
    ctrl_proc = subprocess.Popen(ctrl_cmd, stdout=ctrl_stdout, stderr=subprocess.STDOUT,
                                 cwd=ROOT)
    time.sleep(CTRL_STARTUP_S)
    if ctrl_proc.poll() is not None:
        topo.stop_switches()
        topo.teardown()
        raise RuntimeError(f"controller exited early (code {ctrl_proc.returncode}); "
                           f"see {ctrl_stdout.name}")

    logger.info(f"Warm-up: {WARMUP_S}s benign-only "
               f"(train_until={attack_at:.0f}, now={time.time():.0f})")
    gt = GroundTruthRecorder()
    httpd = HttpServer(H["h4"]["ns"])
    httpd.start()

    wu = schedule["warmup"]
    wu_rng = random.Random(wu["http_seed"]) if wu["http_seed"] is not None else None
    warmup_iperf = Iperf3Benign(gt, H["h1"]["ns"], H["h1"]["ip"], H["h3"]["ns"],
                                H["h3"]["ip"], port=5201, duration=WARMUP_S,
                                bitrate=f"{wu['iperf_mbps']}M", cport=49000)
    warmup_http = HttpBenign(gt, H["h2"]["ns"], H["h2"]["ip"], H["h4"]["ip"],
                             duration=WARMUP_S, interval=wu["http_interval"],
                             base_sport=49100, rng=wu_rng,
                             interval_jitter=HTTP_JITTER if wu_rng else 0.0)
    time.sleep(max(0.0, attack_at - time.time() - WARMUP_S))
    warmup_iperf.start()
    warmup_http.start()
    warmup_iperf.wait(timeout=WARMUP_S + 5)
    warmup_iperf.stop()
    warmup_http.stop()
    time.sleep(max(0.0, attack_at - time.time()))

    logger.info(f"Attack phase: {n_cycles} cycles")
    for c, plan in enumerate(schedule["cycles"]):
        run_cycle(gt, c, plan, spike_pps)

    logger.info(f"Settling {SETTLE_S}s for the last confirmations/forced drops...")
    time.sleep(SETTLE_S)
    httpd.stop()

    ctrl_proc.send_signal(signal.SIGTERM)
    try:
        ctrl_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("Controller did not exit on SIGTERM, killing it")
        ctrl_proc.kill()
        ctrl_proc.wait()
    ctrl_stdout.close()

    topo.stop_switches()
    topo.teardown()

    gt_path = os.path.join(run_dir, "ground_truth.jsonl")
    gt.write(gt_path)
    summary = gt.summary()
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump({"n_cycles": n_cycles, "seed": schedule["seed"], "ground_truth": summary},
                  f, indent=2)
    logger.info(f"Run complete: {summary}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="Series directory "
                   "(default results/paper_<timestamp>)")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--cycles", type=int, default=8, help="Attack/benign cycles per run")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated, one per run; omit for the fixed (unseeded) schedule")
    p.add_argument("--purpose", choices=["dev", "final"], default="dev",
                   help="Recorded with the seeds in experiments/seeds_used.json")
    p.add_argument("--head-clone-n", type=int, default=3)
    p.add_argument("--rate-spike-pps", type=int, default=1000)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    seeds: List[Optional[int]] = [None] * args.runs
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        if len(seeds) != args.runs:
            raise SystemExit(f"--seeds has {len(seeds)} values for --runs {args.runs}")

    series_dir = args.out or os.path.join(ROOT, "results", f"paper_{get_timestamp_str()}")
    ensure_dirs([series_dir])
    if args.seeds:
        register_seeds(seeds, args.purpose, series_dir)
    setup_logging(series_dir, log_file="paper_run.log",
                 level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"Series: {series_dir} ({args.runs} run(s) x {args.cycles} cycles), "
                f"seeds={seeds}, purpose={args.purpose}, head_clone_n={args.head_clone_n}, "
                f"rate_spike_pps={args.rate_spike_pps}")

    for i, seed in enumerate(seeds, 1):
        run_dir = os.path.join(series_dir, f"run_{i:02d}")
        logger.info(f"=== run {i}/{args.runs} (seed {seed}) -> {run_dir} ===")
        try:
            run_once(run_dir, build_schedule(seed, args.cycles), args.head_clone_n,
                     args.rate_spike_pps)
        except Exception:
            logger.exception(f"Run {i} failed")
        if i < args.runs:
            time.sleep(5)

    print(series_dir)


if __name__ == "__main__":
    main()
