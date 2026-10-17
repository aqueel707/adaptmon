#!/usr/bin/env python3
"""
tests/smoke_nmap.py — confirms real nmap SYN/NULL scans are detected
end-to-end through bmv2 + DPI (run inside scripts/sandbox.sh):

    scripts/sandbox.sh python3 tests/smoke_nmap.py

Installs a host_clone_table rule for h1 (Decision B's mechanism: every
probe of a scan is its own 5-tuple/slot, so per-flow escalation never
triggers in time -- a scan is exactly the "many flows, one source" pattern
that rule exists for), then runs real `nmap -sS` and `nmap -sN` from h1 at
h3 and checks DPI confirms syn_scan / null_scan on nmap's real traffic
(not a synthetic stand-in -- see tests/test_dpi.py and tests/smoke_dpi.py
for the crafted-packet versions of these same rules).
Output: results/smoke_nmap_<timestamp>/
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from topo.namespace_topo import NetworkNamespaceTopo, HOSTS   # noqa: E402
from controller.dataplane import DataPlaneManager            # noqa: E402
from controller.utils import get_timestamp_str                # noqa: E402

H = {h["name"]: h for h in HOSTS}


def main():
    out = os.path.join(ROOT, "results", f"smoke_nmap_{get_timestamp_str()}")
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

    # Capture every packet from h1 regardless of its own flow's level
    # (Decision B's intended use case)
    handle = s1.add_host_clone_rule(H["h1"]["ip"])
    assert handle is not None

    alerts = os.path.join(out, "alerts.jsonl")
    dpi = subprocess.Popen([sys.executable, "-m", "controller.dpi", "--ifaces", "1:am_s1dpi",
                            "--alerts", alerts, "--out-dir", out], cwd=ROOT,
                           stdout=open(os.path.join(out, "dpi_out.log"), "w"),
                           stderr=subprocess.STDOUT)
    time.sleep(3)

    syn_log = os.path.join(out, "nmap_syn.log")
    null_log = os.path.join(out, "nmap_null.log")
    r1 = subprocess.run(
        ["ip", "netns", "exec", H["h1"]["ns"], "nmap", "-sS", "-Pn", "-n", "--send-ip", "-T4",
         "--max-rate", "500", "-p", "21,22,25,80,443,8080", H["h3"]["ip"]],
        capture_output=True, text=True)
    open(syn_log, "w").write(r1.stdout + r1.stderr)
    r2 = subprocess.run(
        ["ip", "netns", "exec", H["h1"]["ns"], "nmap", "-sN", "-Pn", "-n", "--send-ip", "-T4",
         "--max-rate", "500", "-p", "21,22,25,80,443,8080", H["h3"]["ip"]],
        capture_output=True, text=True)
    open(null_log, "w").write(r2.stdout + r2.stderr)
    time.sleep(1)

    s1.delete_host_clone_rule(handle)
    dpi.send_signal(15)
    dpi.wait(timeout=20)
    topo.stop_switches()
    topo.teardown()

    al = [json.loads(l) for l in open(alerts)]
    syn_classes = {a["five_tuple"]["dport"] for a in al if a["attack_class"] == "syn_scan"}
    null_classes = {a["five_tuple"]["dport"] for a in al if a["attack_class"] == "null_scan"}
    ports = {21, 22, 25, 80, 443, 8080}
    checks = {
        "nmap_syn_scan_ran": r1.returncode == 0,
        "nmap_null_scan_ran": r2.returncode == 0,
        "syn_scan_confirmed_for_all_ports": syn_classes == ports,
        "null_scan_confirmed_for_all_ports": null_classes == ports,
        "all_confirmed_alerts_from_h1": all(
            a["five_tuple"]["src"] == H["h1"]["ip"] for a in al if a["type"] == "confirmed"),
    }
    report = {"checks": checks, "n_alerts": len(al),
             "syn_dports_confirmed": sorted(syn_classes), "null_dports_confirmed": sorted(null_classes)}
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
