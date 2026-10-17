#!/usr/bin/env python3
"""
tests/smoke_dpi.py — Stage-3 DPI integration test on real bmv2
(run inside scripts/sandbox.sh):

    scripts/sandbox.sh python3 tests/smoke_dpi.py

Sets flow_level = CLONE (register_write) for the slots of crafted flows
h1 -> h3, sends them through s1, and checks that controller/dpi.py (as a
separate process sniffing am_s1dpi) writes the right confirmed alerts:
  * 8 SYN-only probes to 8 ports       -> syn_scan x8
  * 3 NULL probes                      -> null_scan x3
  * HTTP POST with a SQLi body         -> sqli
  * benign HTTP POST, benign UDP flow  -> nothing
Then a burst of 3000 clones measures DPI capture loss (clones counted by
the switch vs clones the DPI engine processed).
Output: results/smoke_stage3_<timestamp>/
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap   # noqa: E402

from topo.namespace_topo import NetworkNamespaceTopo, HOSTS   # noqa: E402
from controller.dataplane import DataPlaneManager            # noqa: E402
from controller.utils import flow_slot, get_timestamp_str    # noqa: E402

H = {h["name"]: h for h in HOSTS}
SENDER = (
    "import sys,time\n"
    "from scapy.all import rdpcap,conf\n"
    "sock=conf.L2socket(iface=sys.argv[2]); gap=float(sys.argv[3])\n"
    "for p in rdpcap(sys.argv[1]):\n"
    "    sock.send(p)\n"
    "    if gap: time.sleep(gap)\n"
)


def pkt(l4):
    return Ether(src=H["h1"]["mac"], dst=H["h3"]["mac"]) / IP(src=H["h1"]["ip"], dst=H["h3"]["ip"]) / l4


def tup(p):
    l4 = p[TCP] if TCP in p else p[UDP]
    return (p[IP].src, p[IP].dst, p[IP].proto, l4.sport, l4.dport)


def send(pkts, path, gap):
    wrpcap(path, pkts)
    subprocess.run(["ip", "netns", "exec", H["h1"]["ns"], sys.executable, "-c", SENDER,
                    path, H["h1"]["veth_host"], str(gap)], check=True)


def main():
    out = os.path.join(ROOT, "results", f"smoke_stage3_{get_timestamp_str()}")
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

    syn = [pkt(TCP(sport=50000, dport=p, flags="S")) for p in (21, 22, 23, 25, 53, 80, 110, 443)]
    null = [pkt(TCP(sport=50001, dport=p, flags=0)) for p in (22, 80, 443)]
    body = "user=admin%27+OR+1%3D1--&pw=x"
    sqli = [pkt(TCP(sport=50002, dport=80, flags="PA") / Raw(
        (f"POST /login HTTP/1.1\r\nHost: 10.0.2.1\r\nContent-Type: application/"
         f"x-www-form-urlencoded\r\nContent-Length: {len(body)}\r\n\r\n{body}").encode()))]
    good = "name=Jane&comment=nice+service&rating=4"
    benign = [pkt(TCP(sport=50003, dport=80, flags="PA") / Raw(
        (f"POST /feedback HTTP/1.1\r\nHost: 10.0.2.1\r\nContent-Length: {len(good)}"
         f"\r\n\r\n{good}").encode())),
        pkt(UDP(sport=50004, dport=5555) / Raw(b"--/* OR 1=1"))]
    allp = syn + null + sqli + benign
    slots = {tup(p): flow_slot(*tup(p)) for p in allp}
    s1.write_flow_levels({s: 2 for s in slots.values()})

    alerts = os.path.join(out, "alerts.jsonl")
    dpi = subprocess.Popen([sys.executable, "-m", "controller.dpi", "--ifaces", "1:am_s1dpi",
                            "--alerts", alerts, "--out-dir", out], cwd=ROOT,
                           stdout=open(os.path.join(out, "dpi_out.log"), "w"),
                           stderr=subprocess.STDOUT)
    time.sleep(3)
    s1.reset_dp_counters()
    send(allp, os.path.join(out, "rules.pcap"), 0.01)
    time.sleep(1)
    n_rule_clones = s1.read_dp_counters()["cloned"]

    # Throughput: 3000 clones of one UDP flow, sent as fast as the sender can
    burst = [pkt(UDP(sport=50005, dport=5556) / Raw(b"y" * 64)) for _ in range(3000)]
    s1.write_flow_levels({flow_slot(*tup(burst[0])): 2})
    t0 = time.time()
    send(burst, os.path.join(out, "burst.pcap"), 0)
    send_s = time.time() - t0
    time.sleep(2)
    burst_clones = s1.read_dp_counters()["cloned"] - n_rule_clones
    dpi.send_signal(15)
    dpi.wait(timeout=20)
    topo.stop_switches()
    topo.teardown()

    al = [json.loads(l) for l in open(alerts)]
    got = {}
    for a in al:
        t = a["five_tuple"]
        got.setdefault((t["src"], t["dst"], t["proto"], t["sport"], t["dport"]), set()).add(
            a["attack_class"])
    stats = json.load(open(os.path.join(out, "dpi_stats.json")))
    checks = {
        "syn_scan_all_8": all(got.get(tup(p)) == {"syn_scan"} for p in syn),
        "null_scan_all_3": all(got.get(tup(p)) == {"null_scan"} for p in null),
        "sqli": got.get(tup(sqli[0])) == {"sqli"},
        "benign_not_flagged": all(tup(p) not in got for p in benign),
        "all_rule_clones_seen": n_rule_clones == len(allp),
        "flow_idx_in_alert_matches_slot": all(
            a["flow_idx"] == slots[(a["five_tuple"]["src"], a["five_tuple"]["dst"],
                                    a["five_tuple"]["proto"], a["five_tuple"]["sport"],
                                    a["five_tuple"]["dport"])] for a in al),
    }
    processed = stats["engine"]["clones"]
    report = {
        "checks": checks,
        "n_alerts": len(al),
        "rule_clones_switch": n_rule_clones,
        "burst": {"sent": len(burst), "send_seconds": send_s,
                  "offered_pps": len(burst) / send_s,
                  "clones_by_switch": burst_clones,
                  "dpi_processed_total": processed,
                  "dpi_processed_burst": processed - n_rule_clones,
                  "dpi_loss_fraction": 1 - (processed - n_rule_clones) / max(burst_clones, 1)},
        "dpi_stats": stats,
    }
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(json.dumps(report["burst"], indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
