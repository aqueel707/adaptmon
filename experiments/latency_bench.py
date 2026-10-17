#!/usr/bin/env python3
"""
experiments/latency_bench.py — controller per-poll latency vs. number of
active flows, on real bmv2 switches. A LATENCY-ONLY benchmark, separate
from the detection runs: it measures how long one poll takes, nothing about
detection quality.

    scripts/sandbox.sh python3 -m experiments.latency_bench --seed <new seed> \\
        --out results/latency_bench_<name>

The real system runs as in experiments/paper_run.py (controller/controller.py
+ controller/dpi.py on both CPU ports, --head-clone-n 3, 1 s polls). Instead
of the paper traffic, crafted UDP flows (h1 side -> h2) pass through s1:

  * tiers of 100, 500, 1000, 2000, 4096 flows, cumulative, TIER_S seconds
    each; every flow has its own register slot (5-tuples chosen with
    controller.utils.flow_slot so no two collide; 4096 uses every slot)
  * each flow sends one PAYLOAD_B-byte packet every FLOW_PERIOD_S (2,048
    pkt/s at 4096 flows, under Decision D's ~2,500 budget); a tier's new
    flows join over RAMP_S, and timings are taken only after SETTLE_S
  * the flow population is STATIONARY: each flow lives a random LIFETIME_S
    (uniform 0.5-1.5 x the mean; the first one uniform 0-1.5 x, so ages are
    spread from the start), goes idle IDLE_S so the controller ages its slot
    out, then restarts as a new flow on the same 5-tuple. The models train
    on the same age mix the tiers later see. (Without churn, attempts with
    all flows started together trained on almost no young flows, then
    scored each tier's newly joined flows as anomalous; the escalations'
    CLONE-level cloning saturated the host -- see experiments/seeds_used.json.)
    Cost: a slot is empty for part of each idle gap, so a tier's occupied
    slots sit slightly under its target -- the measured count is reported.
  * FLOWS_PER_SOURCE flows per spoofed source IP, so host-pair aggregates
    keep the same shape at every tier and the number of pairs scales with
    the tier (50 ... 2048) -- pair inference is timed realistically too
  * the models train once on a WARMUP_S warm-up at the first tier

The seed fixes the 5-tuples, join order and every lifetime; it is recorded
in experiments/seeds_used.json (purpose "latency-bench") and refused if
already used. Per-poll timings are the controller's own windows.csv
columns (t_read_ms = register poll, t_feat_ms = feature extraction,
t_infer_ms = both IF models' inference, t_policy_ms, t_write_ms,
t_total_ms); bench_config.json records each tier's time span.
"""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scapy.all import Ether, IP, UDP, Raw   # noqa: E402

from controller.utils import ensure_dirs, flow_slot, get_timestamp_str  # noqa: E402
from experiments.paper_run import register_seeds                        # noqa: E402
from topo.namespace_topo import NetworkNamespaceTopo, HOSTS             # noqa: E402

H = {h["name"]: h for h in HOSTS}
TIERS = [100, 500, 1000, 2000, 4096]
TIER_S = 60
WARMUP_S = 200          # > 2 x LIFETIME_S: training sees the stationary age mix
CTRL_STARTUP_S = 4
FLOW_PERIOD_S = 2.0
LIFETIME_S = 90.0       # mean flow lifetime
IDLE_S = 5.0            # > 3 polls + margin: the controller ages the slot out
RAMP_S = 15
SETTLE_S = RAMP_S + 3 * FLOW_PERIOD_S + 1
PAYLOAD_B = 64
FLOWS_PER_SOURCE = 2
DPORT = 9000
SWITCH_CONFIGS = [{"switch_id": 1, "thrift_port": 9090},
                  {"switch_id": 2, "thrift_port": 9091}]

SENDER = r"""
import heapq, json, random, socket, sys, time
frames = [bytes.fromhex(l.strip()) for l in open(sys.argv[1])]
plan = json.load(open(sys.argv[2]))      # [[t_start, t_end, n_flows], ...]
period, ramp, life, idle = (float(x) for x in sys.argv[4:8])
rng = random.Random(int(sys.argv[8]))
s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
s.bind((sys.argv[3], 0))
# join time of every flow: warm-up flows at the start, a tier's new flows
# spread over its first `ramp` seconds
join, prev_n = {}, 0
for i, (t0, t1, n) in enumerate(plan):
    for f in range(prev_n, n):
        join[f] = t0 if i == 0 else t0 + ramp * (f - prev_n) / max(n - prev_n, 1)
    prev_n = max(prev_n, n)
t_end = plan[-1][1]
heap, until = [], {}
for f, tj in join.items():
    until[f] = tj + rng.uniform(0, 1.5 * life)          # first lifetime: spread ages
    heapq.heappush(heap, (tj + rng.uniform(0, period), f))
sent = restarts = 0
while heap:
    due, f = heap[0]
    now = time.time()
    if now >= t_end:
        break
    if due > now:
        time.sleep(min(due - now, 0.002))
        continue
    heapq.heappop(heap)
    if due >= until[f]:                                   # lifetime over: go idle
        restart = until[f] + idle
        until[f] = restart + rng.uniform(0.5 * life, 1.5 * life)
        heapq.heappush(heap, (restart, f))
        restarts += 1
        continue
    s.send(frames[f]); sent += 1
    heapq.heappush(heap, (due + period, f))
print(json.dumps({"sent": sent, "lifetime_ends": restarts}))
"""


def source_ip(k: int) -> str:
    return f"10.100.{k // 256}.{k % 256}"


def choose_flows(seed: int):
    """max(TIERS) 5-tuples into distinct s1 slots; flow i from source i // FLOWS_PER_SOURCE."""
    rng = random.Random(seed)
    dst = H["h2"]["ip"]
    sports = list(range(1024, 65536))
    rng.shuffle(sports)
    taken, flows = set(), []
    for i in range(max(TIERS)):
        src = source_ip(i // FLOWS_PER_SOURCE)
        start = rng.randrange(len(sports))
        for j in range(len(sports)):
            sp = sports[(start + j) % len(sports)]
            slot = flow_slot(src, dst, 17, sp, DPORT)
            if slot not in taken:
                taken.add(slot)
                flows.append((src, sp))
                break
        else:
            raise RuntimeError(f"no free slot for flow {i}")
    return flows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.join(ROOT, "results", f"latency_bench_{get_timestamp_str()}")
    ensure_dirs([out])
    register_seeds([args.seed], "latency-bench", out)

    flows = choose_flows(args.seed)
    frames = [bytes(Ether(src=H["h1"]["mac"], dst=H["h2"]["mac"])
                    / IP(src=s, dst=H["h2"]["ip"], ttl=64)
                    / UDP(sport=sp, dport=DPORT) / Raw(b"\x00" * PAYLOAD_B))
              for s, sp in flows]
    with open(os.path.join(out, "frames.hex"), "w") as f:
        f.writelines(fr.hex() + "\n" for fr in frames)

    topo = NetworkNamespaceTopo(log_dir=os.path.join(out, "switches"))
    topo.teardown()
    topo.build()
    assert topo.start_switches(), "switches failed to start"
    time.sleep(1)
    topo.install_forwarding_rules()
    sink = subprocess.Popen(
        ["ip", "netns", "exec", H["h2"]["ns"], sys.executable, "-c",
         f"import socket\ns=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n"
         f"s.bind(('0.0.0.0',{DPORT}))\nwhile True: s.recv(4096)"])

    t_launch = time.time()
    warm_start = t_launch + CTRL_STARTUP_S
    train_until = warm_start + WARMUP_S
    plan = [[warm_start, train_until, TIERS[0]]]
    for i, n in enumerate(TIERS):
        t0 = train_until + i * TIER_S
        plan.append([t0, t0 + TIER_S, n])
    t_end = plan[-1][1]
    cfg = {"seed": args.seed, "tiers": TIERS, "tier_s": TIER_S, "warmup_s": WARMUP_S,
           "flow_period_s": FLOW_PERIOD_S, "lifetime_mean_s": LIFETIME_S, "idle_s": IDLE_S,
           "payload_b": PAYLOAD_B, "flows_per_source": FLOWS_PER_SOURCE,
           "ramp_s": RAMP_S, "settle_s": SETTLE_S,
           "head_clone_n": 3, "poll_interval_s": 1.0, "train_until": train_until,
           "plan": [{"t_start": a, "t_end": b, "n_flows": n,
                     "phase": "warmup" if i == 0 else "measure"}
                    for i, (a, b, n) in enumerate(plan)],
           "flows": flows}
    with open(os.path.join(out, "bench_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(plan, f)
    with open(os.path.join(out, "switch_configs.json"), "w") as f:
        json.dump(SWITCH_CONFIGS, f)

    ctrl_dir = os.path.join(out, "ctrl")
    ensure_dirs([ctrl_dir])
    ctrl_log = open(os.path.join(out, "controller_out.log"), "w")
    ctrl = subprocess.Popen(
        [sys.executable, "-m", "controller.controller",
         "--switches", os.path.join(out, "switch_configs.json"), "--out-dir", ctrl_dir,
         "--poll-interval", "1.0", "--duration", str(t_end - t_launch + 30),
         "--train-until", str(train_until), "--dpi-ifaces", "1:am_s1dpi,2:am_s2dpi",
         "--head-clone-n", "3"],
        stdout=ctrl_log, stderr=subprocess.STDOUT, cwd=ROOT)
    sender = subprocess.run(
        ["ip", "netns", "exec", H["h1"]["ns"], sys.executable, "-c", SENDER,
         os.path.join(out, "frames.hex"), os.path.join(out, "plan.json"),
         H["h1"]["veth_host"], str(FLOW_PERIOD_S), str(RAMP_S), str(LIFETIME_S),
         str(IDLE_S), str(args.seed)],
        capture_output=True, text=True)
    time.sleep(3)
    ctrl.send_signal(signal.SIGTERM)
    try:
        ctrl.wait(timeout=60)
    except subprocess.TimeoutExpired:
        ctrl.kill()
    ctrl_log.close()
    sink.kill()
    topo.stop_switches()
    topo.teardown()
    with open(os.path.join(out, "sender.json"), "w") as f:
        json.dump({"stdout": sender.stdout.strip(), "stderr": sender.stderr[-2000:],
                   "returncode": sender.returncode}, f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
