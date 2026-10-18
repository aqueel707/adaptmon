#!/usr/bin/env python3
"""
tests/smoke_drop.py -- DPI confirmation -> DROP on real bmv2 (run inside
scripts/sandbox.sh):

    scripts/sandbox.sh python3 tests/smoke_drop.py

The real controller (controller/controller.py) and DPI run with flow-head
cloning (N = 3). The Isolation Forest is deliberately never trained here
(--train-until an hour away), so nothing but a DPI confirmation can change
a flow's level: the test isolates the confirmation -> DROP path. (The K_DOWN
dwell and rule removal need scored windows; tests/test_monitor_loop.py
covers them.)

Three TCP flows to sinks on h4, each sending a chunk every CHUNK_GAP_S:
  attack  h2 -> h4:8001  first segment is an HTTP POST with a SQLi body
                         (caught by DPI on its flow-head clones), then keeps
                         sending on the same connection
  benign  h1 -> h4:8002  another host
  benign  h2 -> h4:8003  the attacker's own host, a different 5-tuple
Checks:
  1. DPI confirmed sqli on the attack flow and the controller forced it to
     DROP (forced_drop alert, trigger DPI-head)
  2. the SQLi request reached the server, but nothing sent after the drop
     did while the drop rule was installed: no attack byte arrived between
     drop + 1 s and the rule's removal (windows.csv s1_n_drop_rules back to
     0), while the client kept sending in that interval. The rule's removal
     is the unchanged removal logic: here no window is scored, so it is slot
     aging -- TCP backs off once its segments are dropped, the slot goes
     idle for 3 polls and is reclaimed with its rule. A retransmission
     after that is delivered; the report records it (not a failure).
  3. the switch's own per-slot drop counter (drop_pkts, flow_windows.csv)
     is > 0 for the attack flow, and s1's drop counter is > 0
  4. both benign flows were delivered in full (bytes received == sent) and
     have no drop rule and drop_pkts == 0
  5. sqli is not a scan class: no source-IP drop rule
Output: results/smoke_drop_<timestamp>/report.json
"""

import csv
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from topo.namespace_topo import NetworkNamespaceTopo, HOSTS   # noqa: E402
from controller.dataplane import DataPlaneManager            # noqa: E402
from controller.utils import get_timestamp_str                # noqa: E402

H = {h["name"]: h for h in HOSTS}
CHUNK_GAP_S = 0.25
SEND_S = 8.0
SQLI_BODY = "user=admin%27+OR+1%3D1&pw=x"
BENIGN_BODY = "name=Jane+Doe&comment=great+product&rating=5"

SINK = r"""
import json, socket, sys, threading, time
ports, dur, out = [int(p) for p in sys.argv[1].split(",")], float(sys.argv[2]), sys.argv[3]
log = {p: [] for p in ports}
def serve(p):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", p)); srv.listen(4); srv.settimeout(dur)
    try:
        conn, _ = srv.accept()
    except socket.timeout:
        return
    conn.settimeout(0.5)
    end = time.time() + dur
    while time.time() < end:
        try:
            d = conn.recv(65535)
        except socket.timeout:
            continue
        if not d:
            break
        log[p].append([time.time(), d.decode("latin-1")])
ts = [threading.Thread(target=serve, args=(p,)) for p in ports]
[t.start() for t in ts]; [t.join() for t in ts]
json.dump({str(p): v for p, v in log.items()}, open(out, "w"))
"""

CLIENT = r"""
import json, socket, sys, time
src, sport, dst, dport, body, dur, gap, out = sys.argv[1:9]
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((src, int(sport))); s.settimeout(3); s.connect((dst, int(dport)))
s.setblocking(False)
req = ("POST /login HTTP/1.1\r\nHost: " + dst + "\r\nContent-Type: "
       "application/x-www-form-urlencoded\r\nContent-Length: " + str(len(body))
       + "\r\n\r\n" + body)
sent = []
def send(data):
    try:
        n = s.send(data.encode())
    except BlockingIOError:
        n = 0
    sent.append([time.time(), data, n])
send(req)
end, i = time.time() + float(dur), 0
while time.time() < end:
    time.sleep(float(gap)); i += 1
    send("chunk-%04d\n" % i)
json.dump(sent, open(out, "w"))
"""


def ns_py(host, script, *args, **kw):
    return subprocess.Popen(["ip", "netns", "exec", H[host]["ns"], sys.executable, "-c", script,
                             *map(str, args)], **kw)


def main():
    out = os.path.join(ROOT, "results", f"smoke_drop_{get_timestamp_str()}")
    ctrl_dir = os.path.join(out, "ctrl")
    os.makedirs(ctrl_dir, exist_ok=True)
    checks, report = {}, {"out_dir": out}

    topo = NetworkNamespaceTopo(log_dir=os.path.join(out, "switches"))
    topo.teardown()
    topo.build()
    assert topo.start_switches()
    time.sleep(1)
    topo.install_forwarding_rules()
    with open(os.path.join(out, "switch_configs.json"), "w") as f:
        json.dump([{"switch_id": 1, "thrift_port": 9090}, {"switch_id": 2, "thrift_port": 9091}], f)
    ctrl_log = open(os.path.join(out, "controller_out.log"), "w")
    ctrl = subprocess.Popen(
        [sys.executable, "-m", "controller.controller", "--switches",
         os.path.join(out, "switch_configs.json"), "--out-dir", ctrl_dir,
         "--poll-interval", "1.0", "--duration", "120",
         "--train-until", str(time.time() + 3600),
         "--dpi-ifaces", "1:am_s1dpi,2:am_s2dpi", "--head-clone-n", "3"],
        stdout=ctrl_log, stderr=subprocess.STDOUT, cwd=ROOT)
    time.sleep(6)

    flows = {   # name: (src host, sport, dport, body, start delay)
        "attack_sqli_h2": ("h2", 47001, 8001, SQLI_BODY, 2.0),
        "benign_h1": ("h1", 47002, 8002, BENIGN_BODY, 0.0),
        "benign_h2_other_flow": ("h2", 47003, 8003, BENIGN_BODY, 0.0),
    }
    sink_file = os.path.join(out, "sink.json")
    sink = ns_py("h4", SINK, ",".join(str(f[2]) for f in flows.values()), SEND_S + 8, sink_file)
    time.sleep(1)
    clients = {}
    t0 = time.time()
    for name, (host, sport, dport, body, delay) in sorted(flows.items(), key=lambda x: x[1][4]):
        time.sleep(max(0.0, t0 + delay - time.time()))
        clients[name] = ns_py(host, CLIENT, H[host]["ip"], sport, H["h4"]["ip"], dport, body,
                              SEND_S, CHUNK_GAP_S, os.path.join(out, f"client_{name}.json"))
    for p in clients.values():
        p.wait(timeout=SEND_S + 20)
    sink.wait(timeout=SEND_S + 20)
    time.sleep(2)

    dpm = DataPlaneManager([{"switch_id": 1, "thrift_port": 9090},
                            {"switch_id": 2, "thrift_port": 9091}])
    s1 = dpm.get_switch(1)
    c1 = s1.read_dp_counters_full()
    host_drop_rules = s1.count_host_drop_rules()
    ctrl.send_signal(signal.SIGTERM)
    ctrl.wait(timeout=60)
    ctrl_log.close()
    topo.stop_switches()
    topo.teardown()

    recv = json.load(open(sink_file))
    alerts = [json.loads(l) for l in open(os.path.join(ctrl_dir, "alerts.jsonl"))]
    rows = list(csv.DictReader(open(os.path.join(ctrl_dir, "flow_windows.csv"))))
    win_rows = list(csv.DictReader(open(os.path.join(ctrl_dir, "windows.csv"))))

    def tuple_of(name):
        host, sport, dport, _, _ = flows[name]
        return (H[host]["ip"], H["h4"]["ip"], "6", str(sport), str(dport))

    def drop_pkts(name):
        vals = [int(r["drop_pkts"] or 0) for r in rows
                if (r["src"], r["dst"], r["proto"], r["sport"], r["dport"]) == tuple_of(name)]
        return max(vals, default=0)

    sent = {n: json.load(open(os.path.join(out, f"client_{n}.json"))) for n in flows}
    got = {n: recv.get(str(flows[n][2]), []) for n in flows}
    sent_bytes = {n: sum(len(d) for _, d, k in sent[n] if k) for n in flows}
    got_bytes = {n: sum(len(d) for _, d in got[n]) for n in flows}

    a_name = "attack_sqli_h2"
    a_ip, a_sport = H["h2"]["ip"], flows[a_name][1]
    fd = [a for a in alerts if a["type"] == "forced_drop"
          and a["five_tuple"]["src"] == a_ip and a["five_tuple"]["sport"] == a_sport]
    conf = [a for a in alerts if a["type"] == "confirmed" and a["attack_class"] == "sqli"
            and a["five_tuple"]["sport"] == a_sport]
    checks["dpi_confirmed_sqli_and_forced_drop"] = {
        "pass": bool(conf) and bool(fd) and fd[0]["level_name"] == "DROP"
                and fd[0]["trigger"] == "DPI-head",
        "confirmed": len(conf), "forced_drop": fd[:2],
    }
    t_drop = min(a["ts"] for a in fd) if fd else float("inf")
    # poll in which s1's drop rule was deleted (or the end of the run)
    t_removed = min((float(r["ts"]) for r in win_rows
                     if float(r["ts"]) > t_drop and int(r["s1_n_drop_rules"] or 0) == 0),
                    default=float(win_rows[-1]["ts"]) if win_rows else t_drop)
    in_force = lambda t: t_drop + 1.0 < t <= t_removed     # noqa: E731
    recv_in_force = [t for t, _ in got[a_name] if in_force(t)]
    sent_in_force = [t for t, _, _ in sent[a_name] if in_force(t)]
    recv_after = [(t, len(d)) for t, d in got[a_name] if t > t_removed]
    first_seg = got[a_name][0][1] if got[a_name] else ""
    checks["attack_later_packets_dropped"] = {
        "pass": SQLI_BODY in first_seg and not recv_in_force and len(sent_in_force) >= 5,
        "sqli_request_reached_server": SQLI_BODY in first_seg,
        "drop_rule_lifetime_s": round(t_removed - t_drop, 3),
        "client_sends_while_rule_installed": len(sent_in_force),
        "bytes_received_while_rule_installed": sum(
            len(d) for t, d in got[a_name] if in_force(t)),
        # informational: delivered after the removal logic lifted the rule
        "delivered_after_rule_removed": [
            {"s_after_drop": round(t - t_drop, 3), "bytes": n} for t, n in recv_after],
        "bytes_sent": sent_bytes[a_name], "bytes_received": got_bytes[a_name],
    }
    checks["switch_counted_the_drops"] = {
        "pass": drop_pkts(a_name) > 0 and c1["dropped"] > 0,
        "attack_flow_drop_pkts": drop_pkts(a_name), "s1_dropped": c1["dropped"],
    }
    benign = [n for n in flows if n.startswith("benign")]
    checks["benign_flows_delivered_in_full"] = {
        "pass": all(got_bytes[n] == sent_bytes[n] > 0 and drop_pkts(n) == 0 for n in benign)
                and not any(a["type"] == "forced_drop" and a["five_tuple"]["sport"] in
                            [flows[n][1] for n in benign] for a in alerts),
        **{n: {"sent": sent_bytes[n], "received": got_bytes[n], "drop_pkts": drop_pkts(n)}
           for n in benign},
    }
    checks["sqli_is_not_a_scan_no_source_drop"] = {
        "pass": host_drop_rules == 0 and not any(
            a["type"] == "pair_escalation" and a.get("level_name") == "DROP" for a in alerts),
        "host_drop_rules_on_s1": host_drop_rules,
    }

    report["checks"] = checks
    report["all_pass"] = all(c["pass"] for c in checks.values())
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    for k, c in checks.items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {k}")
    print(f"report: {os.path.join(out, 'report.json')}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
