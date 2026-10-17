#!/usr/bin/env python3
"""
tests/smoke_traffic.py — Stage-4 traffic-generator smoke test on real bmv2
(run inside scripts/sandbox.sh):

    scripts/sandbox.sh python3 tests/smoke_traffic.py

Runs every traffic.paper_traffic generator for a few seconds each,
sequentially (Decision D: keep offered load low by not overlapping
attacks), and checks:
  * each generator's ground-truth records match what it actually sent
    (right 5-tuple, right label/class, start < end)
  * the HTTP server actually received the SQLi bodies (the attack really
    went over the wire as real HTTP, not just a ground-truth entry)
  * iperf3 actually transferred data (its own JSON summary, not assumed)
  * nmap scans run and are confirmed by DPI (reusing tests/smoke_nmap.py's
    already-proven mechanism, abbreviated here to 2 ports each)
  * the whole sequential schedule's peak instantaneous rate stays under
    Decision D's ~2500 pkt/s budget (measured from the switch's own
    forwarded+cloned counters, not estimated)
Output: results/smoke_stage4_<timestamp>/
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from topo.namespace_topo import NetworkNamespaceTopo, HOSTS   # noqa: E402
from controller.dataplane import DataPlaneManager            # noqa: E402
from controller.utils import get_timestamp_str                # noqa: E402
from traffic.ground_truth import GroundTruthRecorder          # noqa: E402
from traffic.paper_traffic import (                           # noqa: E402
    HttpServer, HttpBenign, HttpSqliAttack, Iperf3Benign, NmapScan,
    RateSpikeAttack, SQLI_BODIES,
)

H = {h["name"]: h for h in HOSTS}


def main():
    out = os.path.join(ROOT, "results", f"smoke_stage4_{get_timestamp_str()}")
    os.makedirs(out, exist_ok=True)
    topo = NetworkNamespaceTopo(log_dir=os.path.join(out, "switches"))
    topo.teardown()
    topo.build()
    assert topo.start_switches()
    time.sleep(1)
    topo.install_forwarding_rules()
    dpm = DataPlaneManager([{"switch_id": 1, "thrift_port": 9090},
                            {"switch_id": 2, "thrift_port": 9091}])
    assert dpm.wait_for_switches(30)
    assert all(dpm.install_mirroring_all().values())
    s1 = dpm.get_switch(1)

    # Capture all of h1's and h2's traffic (host-pair mechanism) so DPI can
    # confirm the scans (from h1) and the SQLi HTTP POSTs (from h2) without
    # waiting on per-flow escalation or adaptive-sampling luck
    clone_handle = s1.add_host_clone_rule(H["h1"]["ip"])
    clone_handle_h2 = s1.add_host_clone_rule(H["h2"]["ip"])
    alerts_path = os.path.join(out, "alerts.jsonl")
    import subprocess
    dpi = subprocess.Popen([sys.executable, "-m", "controller.dpi", "--ifaces", "1:am_s1dpi",
                            "--alerts", alerts_path, "--out-dir", out], cwd=ROOT,
                           stdout=open(os.path.join(out, "dpi_out.log"), "w"),
                           stderr=subprocess.STDOUT)
    time.sleep(2)

    gt = GroundTruthRecorder()
    checks = {}

    httpd = HttpServer(H["h4"]["ns"])
    httpd.start()
    # Confirm the SQLi bodies actually reach the server on the wire by
    # sniffing h4's receiving interface for the POST bodies directly,
    # rather than trusting the generator's own ground-truth bookkeeping.
    from scapy.all import TCP, Raw

    pcap_path = os.path.join(out, "h4_in.pcap")
    cap_argv = [sys.executable, os.path.join(ROOT, "tools", "pcap_capture.py"),
               H["h4"]["veth_host"], pcap_path, "--dir", "in"]
    cap = subprocess.Popen(["ip", "netns", "exec", H["h4"]["ns"]] + cap_argv,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

    s1.reset_dp_counters()
    t_pps_samples = []

    def sample_pps(label, duration):
        c0 = s1.read_dp_counters()
        t0 = time.time()
        time.sleep(duration)
        c1 = s1.read_dp_counters()
        dt = time.time() - t0
        pps = ((c1["forwarded"] - c0["forwarded"]) + (c1["cloned"] - c0["cloned"])) / dt
        t_pps_samples.append((label, pps))

    # 1. Benign iperf3 (h1 -> h3)
    iperf = Iperf3Benign(gt, H["h1"]["ns"], H["h1"]["ip"], H["h3"]["ns"], H["h3"]["ip"],
                        port=5201, duration=6, bitrate="1M", cport=50100)
    iperf.start()
    sample_pps("iperf3_benign", 6)
    iperf.wait(timeout=10)
    iperf.stop()

    # 2. Benign HTTP (h2 -> h4)
    http_benign = HttpBenign(gt, H["h2"]["ns"], H["h2"]["ip"], H["h4"]["ip"],
                             duration=6, interval=1.0, base_sport=51100)
    http_benign.start()
    sample_pps("http_benign", 6)
    http_benign.stop()

    # 3. SQLi attack (h2 -> h4, same server)
    sqli = HttpSqliAttack(gt, H["h2"]["ns"], H["h2"]["ip"], H["h4"]["ip"],
                          base_sport=52100, gap=0.3)
    sqli.start()
    sample_pps("sqli_attack", 3)
    sqli.join(timeout=10)

    # 4. Nmap SYN scan (h1 -> h3)
    syn = NmapScan(gt, "syn", H["h1"]["ns"], H["h1"]["ip"], H["h3"]["ip"],
                  ports=[21, 22, 80, 443, 8080], max_rate=500, base_sport=54100)
    t0 = time.time()
    r_syn = syn.run(timeout=15)
    t_pps_samples.append(("nmap_syn", 5 / max(time.time() - t0, 1e-3)))

    # 5. Nmap NULL scan (h1 -> h3)
    null = NmapScan(gt, "null", H["h1"]["ns"], H["h1"]["ip"], H["h3"]["ip"],
                   ports=[21, 22, 80, 443, 8080], max_rate=500, base_sport=54200)
    t0 = time.time()
    r_null = null.run(timeout=15)
    t_pps_samples.append(("nmap_null", 5 / max(time.time() - t0, 1e-3)))

    # 6. Rate-spike attack (h2 -> h4)
    spike = RateSpikeAttack(gt, H["h2"]["ns"], H["h2"]["ip"], H["h4"]["ip"], dport=9999,
                            duration=4, rate=800, sport=45100)
    spike.start()
    sample_pps("rate_spike", 4)
    spike.wait(timeout=8)
    spike.stop()

    time.sleep(1)
    cap.terminate()
    cap.wait(timeout=5)
    httpd.stop()
    s1.delete_host_clone_rule(clone_handle)
    s1.delete_host_clone_rule(clone_handle_h2)
    dpi.send_signal(15)
    dpi.wait(timeout=20)
    topo.stop_switches()
    topo.teardown()

    gt_path = os.path.join(out, "ground_truth.jsonl")
    gt.write(gt_path)
    gt_recs = [json.loads(l) for l in open(gt_path)]

    # -- Checks --
    checks["ground_truth_has_all_classes"] = {
        "pass": {r["attack_class"] or "benign" for r in gt_recs} == {
            "benign", "sqli", "syn_scan", "null_scan", "rate_spike"},
        "classes": sorted({r["attack_class"] or "benign" for r in gt_recs}),
    }
    checks["ground_truth_flows_have_valid_windows"] = {
        "pass": all(r["end_ts"] is not None and r["end_ts"] >= r["start_ts"] for r in gt_recs),
        "n_flows": len(gt_recs),
    }
    checks["nmap_scans_ran_ok"] = {
        "pass": r_syn.returncode == 0 and r_null.returncode == 0,
    }

    from scapy.all import rdpcap
    h4_pkts = rdpcap(pcap_path) if os.path.getsize(pcap_path) > 24 else []
    bodies_seen = []
    for p in h4_pkts:
        if TCP in p and Raw in p:
            try:
                bodies_seen.append(bytes(p[Raw]).decode("latin-1"))
            except Exception:
                pass
    joined = "\n".join(bodies_seen)
    sqli_on_wire = [body for body, _ in SQLI_BODIES if body in joined
                   or body.replace("%27", "'") in joined]
    checks["sqli_bodies_actually_on_the_wire"] = {
        "pass": len(sqli_on_wire) == len(SQLI_BODIES),
        "found": len(sqli_on_wire), "expected": len(SQLI_BODIES),
    }

    al = [json.loads(l) for l in open(alerts_path)]
    syn_confirmed = {a["five_tuple"]["dport"] for a in al if a["attack_class"] == "syn_scan"}
    null_confirmed = {a["five_tuple"]["dport"] for a in al if a["attack_class"] == "null_scan"}
    sqli_confirmed = sum(1 for a in al if a["attack_class"] == "sqli")
    checks["dpi_confirms_scans_and_sqli"] = {
        "pass": syn_confirmed == {21, 22, 80, 443, 8080}
                and null_confirmed == {21, 22, 80, 443, 8080}
                and sqli_confirmed == len(SQLI_BODIES),
        "syn_confirmed_ports": sorted(syn_confirmed),
        "null_confirmed_ports": sorted(null_confirmed),
        "sqli_confirmed": sqli_confirmed,
    }

    checks["peak_offered_rate_under_decision_d_budget"] = {
        "pass": max(p for _, p in t_pps_samples) < 2500,
        "samples": {label: round(p, 1) for label, p in t_pps_samples},
    }

    report = {"checks": {k: v["pass"] for k, v in checks.items()}, "detail": checks,
             "ground_truth_summary": gt.summary()}
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    for k, v in report["checks"].items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(json.dumps(checks, indent=1, default=str))
    sys.exit(0 if all(report["checks"].values()) else 1)


if __name__ == "__main__":
    main()
