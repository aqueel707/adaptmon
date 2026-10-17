#!/usr/bin/env python3
"""
tests/smoke_controller.py — Stage-2 smoke test of the per-flow controller
on real bmv2 (run inside scripts/sandbox.sh):

    scripts/sandbox.sh python3 tests/smoke_controller.py

Timeline (s from traffic start): 0-25 benign only (warm-up, model trained
once at the first poll after 25 s), 30-40 an extra UDP rate-spike flow
h2 -> h4 (~2000 pkt/s), controller stops at 55 s.

Checks the controller started ("Monitor started"), trained exactly once on
warm-up vectors only, logged per-flow rows, and that the flow_level register
read back from the switches equals the level the controller last assigned
to each live flow. Prints a short summary of what levels flows reached.
Output: results/smoke_stage2_<timestamp>/
"""

import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from topo.namespace_topo import NetworkNamespaceTopo, HOSTS   # noqa: E402
from traffic.ns_traffic import NsTrafficManager               # noqa: E402
from controller.dataplane import DataPlaneManager            # noqa: E402
from controller.utils import get_timestamp_str               # noqa: E402

H = {h["name"]: h for h in HOSTS}
SPIKE = """
import socket, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('{src}', 45000))
end = time.time() + {dur}
payload = b'x' * 64
n = 0
t0 = time.time()
while time.time() < end:
    s.sendto(payload, ('{dst}', 9999))
    n += 1
    lag = t0 + n / {rate} - time.time()
    if lag > 0:
        time.sleep(lag)
"""


def main():
    out = os.path.join(ROOT, "results", f"smoke_stage2_{get_timestamp_str()}")
    os.makedirs(out, exist_ok=True)
    topo = NetworkNamespaceTopo(log_dir=os.path.join(out, "switches"))
    topo.teardown()
    topo.build()
    assert topo.start_switches()
    time.sleep(1)
    topo.install_forwarding_rules()
    sw_cfg = os.path.join(out, "switch_configs.json")
    json.dump([{"switch_id": 1, "thrift_port": 9090},
               {"switch_id": 2, "thrift_port": 9091}], open(sw_cfg, "w"))

    # Traffic begins ~3 s after launch, after the controller's first poll
    t_traffic = time.time() + 3
    ctrl = subprocess.Popen(
        [sys.executable, "-m", "controller.controller", "--switches", sw_cfg,
         "--out-dir", out, "--duration", "57", "--train-until", str(t_traffic + 25)],
        cwd=ROOT, stdout=open(os.path.join(out, "controller_out.log"), "w"),
        stderr=subprocess.STDOUT)
    time.sleep(max(0, t_traffic - time.time()))
    traffic = NsTrafficManager({h: {"ns": H[h]["ns"], "ip": H[h]["ip"]} for h in H})
    traffic.start_normal_traffic(duration=55)
    sink = subprocess.Popen(
        ["ip", "netns", "exec", H["h4"]["ns"], sys.executable, "-c",
         "import socket\ns=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n"
         "s.bind(('0.0.0.0',9999))\nwhile True: s.recv(2048)"])
    time.sleep(max(0, t_traffic + 30 - time.time()))
    spike = subprocess.Popen(
        ["ip", "netns", "exec", H["h2"]["ns"], sys.executable, "-c",
         SPIKE.format(src=H["h2"]["ip"], dst=H["h4"]["ip"], dur=10, rate=2000)])
    spike.wait()
    ctrl.wait(timeout=60)
    traffic.stop_all()
    sink.kill()

    dpm = DataPlaneManager([{"switch_id": 1, "thrift_port": 9090},
                            {"switch_id": 2, "thrift_port": 9091}])
    regs = {sid: dpm.get_switch(sid).read_registers(["flow_level", "pkt_counter"])
            for sid in (1, 2)}
    topo.stop_switches()
    topo.teardown()

    log = open(os.path.join(out, "controller.log")).read()
    model = json.load(open(os.path.join(out, "model.json")))
    rows = list(csv.DictReader(open(os.path.join(out, "flow_windows.csv"))))
    wins = list(csv.DictReader(open(os.path.join(out, "windows.csv"))))
    last = {}
    for r in rows:
        last[(int(r["switch"]), int(r["slot"]))] = r
    final_w = max(int(r["window"]) for r in rows)
    mismatch = [(k, int(r["level"]), regs[k[0]]["flow_level"][k[1]])
                for k, r in last.items()
                if int(r["window"]) == final_w and regs[k[0]]["pkt_counter"][k[1]] > 0
                and int(r["level"]) != regs[k[0]]["flow_level"][k[1]]]
    warm_rows = [r for r in rows if float(r["ts"]) < t_traffic + 25]
    checks = {
        "monitor_started_logged": "Monitor started" in log,
        "z_score_disabled_logged": "z-score trigger: DISABLED" in log,
        "mirroring_configured_logged": log.count("mirroring session 100 -> CPU port 255 configured") == 2,
        "trained_once": log.count("IsolationForest trained once") == 1 and model["trained"],
        "trained_on_warmup_vectors_only": model["n_train"] == sum(
            1 for r in warm_rows if int(r["window"]) > 1),
        "no_scores_during_warmup": all(r["scored"] == "0" for r in warm_rows),
        "flow_level_register_matches_controller": not mismatch,
    }
    max_level = defaultdict(int)
    for r in rows:
        k = (r["src"], r["dst"], r["proto"], r["sport"], r["dport"])
        max_level[k] = max(max_level[k], int(r["level"]))
    spike_key = (H["h2"]["ip"], H["h4"]["ip"], "17", "45000", "9999")
    summary = {
        "checks": checks,
        "mismatches": mismatch[:10],
        "model": {k: model[k] for k in ("n_train", "thresholds", "fit_seconds")},
        "n_windows": len(wins),
        "max_level_per_flow_hist": dict(Counter(max_level.values())),
        "spike_flow_max_level": max_level.get(spike_key),
        "spike_flow_scores": [(r["window"], r["switch"], r["score"], r["raw_level"], r["level"])
                              for r in rows if (r["src"], r["dst"], r["proto"], r["sport"],
                                                r["dport"]) == spike_key][:30],
        "poll_ms_mean": sum(float(w["t_total_ms"]) for w in wins) / len(wins),
        "poll_ms_max": max(float(w["t_total_ms"]) for w in wins),
    }
    json.dump(summary, open(os.path.join(out, "report.json"), "w"), indent=2)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("checks", "spike_flow_scores")},
                     indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
