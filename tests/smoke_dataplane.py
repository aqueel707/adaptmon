#!/usr/bin/env python3
"""
tests/smoke_dataplane.py — Stage-1 data-plane smoke test on real bmv2.

Run inside the sandbox (namespaced root):
    scripts/sandbox.sh python3 tests/smoke_dataplane.py

Builds the 2-switch namespace topology, configures clone session 100 via
DataPlaneManager.install_mirroring_all() (mirroring_add 100 255), then sends
crafted UDP flows h1 -> h2 (one switch) and h1 -> h3 (two switches) with
per-flow levels written into flow_level via register_write. Checks:

  1. mirroring_get 100 reports port 255 on both switches
  2. utils.flow_slot() predicts the register slot bmv2 uses for each 5-tuple
  3. clones reach the CPU port only for flows with flow_level >= CLONE (2)
  4. every clone is cpu_hdr (14 B: flow_index, level, pkt_count, byte_count,
     reason) + the original frame, with flow_index/level from the field list
  5. forwarded originals are byte-identical to what h1 sent (pcap diff):
       h1 -> h2: s1 rule switched to forward_l2 (no rewrite) -> strict equality
       h1 -> h3: forward_l3 on 2 switches -> equal after applying the
                 documented rewrite (TTL - 2, IPv4 checksum recomputed);
                 no extra bytes (no cpu_hdr)
  6. a DROP flow (level 3 + 5-tuple drop rule) is not delivered but is still
     cloned
  7. the data-plane fwd/clone/drop counters match the packets sent exactly

Writes pcaps + report.json under results/smoke_stage1_<timestamp>/ and exits
non-zero if any check fails.
"""

import json
import os
import random
import re
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scapy.all import Ether, IP, UDP, Raw, rdpcap, wrpcap   # noqa: E402

from topo.namespace_topo import NetworkNamespaceTopo, HOSTS   # noqa: E402
from controller.dataplane import DataPlaneManager            # noqa: E402
from controller.utils import flow_slot, get_timestamp_str    # noqa: E402

CPU_HDR_LEN = 14
H = {h["name"]: h for h in HOSTS}

SINK = (
    "import socket,select\n"
    "ss=[]\n"
    "for p in {ports}:\n"
    "    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind(('0.0.0.0',p)); ss.append(s)\n"
    "while True:\n"
    "    r,_,_=select.select(ss,[],[])\n"
    "    for s in r: s.recv(65535)\n"
)

SENDER = (
    "import sys,time\n"
    "from scapy.all import rdpcap,sendp,conf\n"
    "sock=conf.L2socket(iface=sys.argv[2])\n"
    "for p in rdpcap(sys.argv[1]):\n"
    "    sock.send(p); time.sleep(0.004)\n"
)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def ns_exec(host, cmd):
    return f"ip netns exec {H[host]['ns']} {cmd}"


CAPTURE = os.path.join(ROOT, "tools", "pcap_capture.py")


class Capture:
    """tools/pcap_capture.py on iface (direction 'in' or 'out')."""

    def __init__(self, direction, iface, path, host=None):
        self.path = path
        cmd = [sys.executable, CAPTURE, iface, path, "--dir", direction]
        if host:
            cmd = ["ip", "netns", "exec", H[host]["ns"]] + cmd
        # argv list (no shell), so terminate() reaches the capture process
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

    def stop(self):
        time.sleep(0.5)
        self.proc.terminate()
        self.proc.wait(timeout=5)
        with open(self.path + ".stats.json") as f:
            self.stats = json.load(f)
        return rdpcap(self.path) if os.path.getsize(self.path) > 24 else []


def mk_flow(src, dst, sport, dport, n, rng):
    pkts = []
    for _ in range(n):
        payload = bytes(rng.getrandbits(8) for _ in range(rng.randint(18, 1000)))
        pkts.append(
            Ether(src=H[src]["mac"], dst=H[dst]["mac"])
            / IP(src=H[src]["ip"], dst=H[dst]["ip"], ttl=64, id=rng.randint(0, 65535))
            / UDP(sport=sport, dport=dport) / Raw(payload)
        )
    # Build once so checksums are fixed in the bytes we send and compare
    return [Ether(bytes(p)) for p in pkts]


def send(host, pkts, path):
    wrpcap(path, pkts)
    r = sh(ns_exec(host, f"python3 -c '{SENDER}' {path} {H[host]['veth_host']}"))
    if r.returncode != 0:
        raise RuntimeError(r.stderr)


def l4key(p):
    return (p[IP].src, p[IP].dst, p[IP].proto, p[UDP].sport, p[UDP].dport)


def parse_cpu(raw):
    idx, lvl, pc, bc, reason = struct.unpack("!IBIIB", raw[:CPU_HDR_LEN])
    return {"flow_index": idx, "level": lvl, "pkt_count": pc,
            "byte_count": bc, "reason": reason}, raw[CPU_HDR_LEN:]


def main():
    out = os.path.join(ROOT, "results", f"smoke_stage1_{get_timestamp_str()}")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(1)
    checks = {}
    report = {"out_dir": out, "checks": checks}

    topo = NetworkNamespaceTopo(log_dir=os.path.join(out, "switches"))
    topo.teardown()
    topo.build()
    assert topo.start_switches(), "switches failed to start"
    time.sleep(1)
    topo.install_forwarding_rules()
    dpm = DataPlaneManager([{"switch_id": 1, "thrift_port": 9090},
                            {"switch_id": 2, "thrift_port": 9091}])
    assert dpm.wait_for_switches(30)
    s1, s2 = dpm.get_switch(1), dpm.get_switch(2)

    # 1. mirroring session 100 -> CPU port 255
    mir = dpm.install_mirroring_all()
    checks["mirroring_session_100_port_255"] = {
        "pass": all(mir.values()),
        "mirroring_get": {sid: dpm.get_switch(sid).run_commands(["mirroring_get 100"]).strip()
                          for sid in mir},
    }

    # h1 -> h2 on s1: use forward_l2 so s1 forwards without any rewrite
    dump = s1.run_commands(["table_dump AdaptMonIngress.ipv4_forward"])
    handle = None
    for block in dump.split("Dumping entry")[1:]:
        if "0a000102/32" in block:
            handle = int(re.search(r"0x([0-9a-f]+)", block).group(1), 16)
    s1.run_commands([f"table_modify AdaptMonIngress.ipv4_forward "
                     f"AdaptMonIngress.forward_l2 {handle} 2"])

    # UDP sinks so the receivers send nothing back (no ICMP unreachable)
    sinks = [subprocess.Popen(ns_exec(h, f"python3 -c \"{SINK.format(ports=ports)}\""),
                              shell=True)
             for h, ports in (("h2", [7000, 7001, 7002, 7003]), ("h3", [7100]))]
    time.sleep(1)

    # Flows (sport chosen so the 5 flows use 5 distinct slots)
    flows = {   # name: (src, dst, sport, dport, level on s1)
        "L0_normal" : ("h1", "h2", 41000, 7000, 0),
        "L1_monitor": ("h1", "h2", 41001, 7001, 1),
        "L2_clone"  : ("h1", "h2", 41002, 7002, 2),
        "L3_drop"   : ("h1", "h2", 41003, 7003, 3),
        "X2_clone_2hop": ("h1", "h3", 41004, 7100, 2),
    }
    slots = {n: flow_slot(H[s]["ip"], H[d]["ip"], 17, sp, dp)
             for n, (s, d, sp, dp, _) in flows.items()}
    assert len(set(slots.values())) == len(slots), f"slot collision {slots}"

    # 2. one probe packet per flow, then find the slots bmv2 actually used
    probes = []
    for n, (s, d, sp, dp, _) in flows.items():
        probes += mk_flow(s, d, sp, dp, 1, rng)
    send("h1", probes, os.path.join(out, "probes.pcap"))
    time.sleep(0.5)
    regs = s1.read_registers(["pkt_counter", "flow_src_port", "flow_dst_port"])
    found = {n: [i for i, v in enumerate(regs["flow_src_port"])
                 if v == sp and regs["flow_dst_port"][i] == dp and regs["pkt_counter"][i] > 0]
             for n, (_, _, sp, dp, _) in flows.items()}
    checks["flow_slot_matches_bmv2_crc32"] = {
        "pass": all(found[n] == [slots[n]] for n in flows),
        "predicted": slots, "observed": found,
    }

    # Levels via register_write (Thrift); DROP also gets its 5-tuple rule
    s1.write_flow_levels({slots[n]: f[4] for n, f in flows.items()})
    s, d, sp, dp, _ = flows["L3_drop"]
    drop_handle = s1.add_drop_rule(H[s]["ip"], H[d]["ip"], 17, sp, dp)
    lv = s1.read_registers(["flow_level"])["flow_level"]
    checks["flow_level_written"] = {
        "pass": all(lv[slots[n]] == f[4] for n, f in flows.items()) and drop_handle is not None,
        "levels": {n: lv[slots[n]] for n in flows}, "drop_rule_handle": drop_handle,
    }
    s1.reset_dp_counters()
    s2.reset_dp_counters()

    # Main traffic: 30 packets per flow, interleaved
    N = 30
    per_flow = {n: mk_flow(s, d, sp, dp, N, rng) for n, (s, d, sp, dp, _) in flows.items()}
    sent = [p for i in range(N) for n in flows for p in [per_flow[n][i]]]
    caps = {
        "h1_out": Capture("out", H["h1"]["veth_host"], os.path.join(out, "h1_out.pcap"), "h1"),
        "h2_in" : Capture("in", H["h2"]["veth_host"], os.path.join(out, "h2_in.pcap"), "h2"),
        "h3_in" : Capture("in", H["h3"]["veth_host"], os.path.join(out, "h3_in.pcap"), "h3"),
        "s1_cpu": Capture("in", "am_s1dpi", os.path.join(out, "s1_cpu.pcap")),
        "s2_cpu": Capture("in", "am_s2dpi", os.path.join(out, "s2_cpu.pcap")),
    }
    time.sleep(1.5)
    send("h1", sent, os.path.join(out, "sent.pcap"))
    got = {k: c.stop() for k, c in caps.items()}
    report["capture_stats"] = {k: c.stats for k, c in caps.items()}
    for p in sinks:
        p.kill()

    by_flow = lambda pkts: {n: [bytes(p) for p in pkts if UDP in p and
                                l4key(p) == (H[f[0]]["ip"], H[f[1]]["ip"], 17, f[2], f[3])]
                            for n, f in flows.items()}
    h1_out = by_flow(got["h1_out"])
    sent_b = {n: [bytes(p) for p in per_flow[n]] for n in flows}
    checks["capture_h1_matches_sent"] = {
        "pass": all(h1_out[n] == sent_b[n] for n in flows),
        "counts": {n: len(h1_out[n]) for n in flows},
    }

    # 5. originals byte-identical
    h2_in = by_flow(got["h2_in"])
    strict = {n: h2_in[n] == sent_b[n] for n in ("L0_normal", "L1_monitor", "L2_clone")}
    checks["originals_byte_identical_l2_path"] = {
        "pass": all(strict.values()),
        "per_flow": strict,
        "packets_compared": sum(len(h2_in[n]) for n in strict),
        "non_udp_or_foreign_packets_at_h2": len(got["h2_in"]) - sum(len(v) for v in h2_in.values()),
    }
    h3_in = by_flow(got["h3_in"])["X2_clone_2hop"]
    expected_h3 = []
    for b in sent_b["X2_clone_2hop"]:
        e = Ether(b)
        e[IP].ttl -= 2          # two forward_l3 hops
        del e[IP].chksum
        expected_h3.append(bytes(Ether(bytes(e))))
    checks["originals_identical_after_l3_rewrite_2hop"] = {
        "pass": h3_in == expected_h3,
        "packets_compared": len(h3_in),
        "lengths_equal": [len(a) for a in h3_in] == [len(b) for b in sent_b["X2_clone_2hop"]],
        "rewrite_applied": "ttl-=2, ipv4 checksum recomputed; dst MAC unchanged "
                           "(forward_l3 sets the same host MAC h1 already used)",
    }

    # 6. DROP flow
    checks["drop_flow_not_delivered"] = {
        "pass": len(h2_in["L3_drop"]) == 0, "delivered": len(h2_in["L3_drop"]),
    }

    # 3 + 4. clones
    def clone_check(cap, expect_flows, frames_expected):
        res = {"n_clones": len(cap), "per_flow": {}}
        ok = True
        hdrs_by_flow = {n: [] for n in flows}
        for p in cap:
            hdr, frame = parse_cpu(bytes(p))
            e = Ether(frame)
            name = next((n for n, f in flows.items() if UDP in e and
                         l4key(e) == (H[f[0]]["ip"], H[f[1]]["ip"], 17, f[2], f[3])), None)
            if name is None:
                ok = False
                res.setdefault("unknown", 0)
                res["unknown"] = res["unknown"] + 1
                continue
            hdrs_by_flow[name].append((hdr, frame))
        for n in flows:
            got_n = hdrs_by_flow[n]
            want = n in expect_flows
            entry = {"clones": len(got_n)}
            if want:
                hdr_ok = all(h["flow_index"] == slots[n] and h["level"] == flows[n][4]
                             and h["reason"] == 1 for h, _ in got_n)
                counts = [h["pkt_count"] for h, _ in got_n]
                frames_ok = [fr for _, fr in got_n] == frames_expected[n]
                entry.update(hdr_fields_ok=hdr_ok,
                             pkt_count_seq=f"{counts[0]}..{counts[-1]}" if counts else "",
                             pkt_count_consecutive=counts == list(range(2, 2 + N)),
                             frames_equal_forwarded_original=frames_ok)
                ok &= len(got_n) == N and hdr_ok and frames_ok and counts == list(range(2, 2 + N))
            else:
                ok &= len(got_n) == 0
            res["per_flow"][n] = entry
        res["pass"] = ok
        return res

    # Frame inside a clone = the original as it left s1's egress
    s1_frames = {n: sent_b[n] for n in ("L2_clone", "L3_drop")}
    s1_frames["X2_clone_2hop"] = []
    for b in sent_b["X2_clone_2hop"]:
        e = Ether(b)
        e[IP].ttl -= 1
        del e[IP].chksum
        s1_frames["X2_clone_2hop"].append(bytes(Ether(bytes(e))))
    checks["clones_only_for_level_ge_clone_s1"] = clone_check(
        got["s1_cpu"], {"L2_clone", "L3_drop", "X2_clone_2hop"}, s1_frames)
    checks["no_clones_s2_all_levels_normal"] = {
        "pass": len(got["s2_cpu"]) == 0, "n_clones": len(got["s2_cpu"]),
    }

    # 7. counters (reset after the probes)
    c1, c2 = s1.read_dp_counters(), s2.read_dp_counters()
    exp1 = {"forwarded": 4 * N, "cloned": 3 * N, "dropped": N}
    exp2 = {"forwarded": N, "cloned": 0, "dropped": 0}
    checks["dataplane_counters_exact"] = {
        "pass": c1 == exp1 and c2 == exp2,
        "s1": c1, "s1_expected": exp1, "s2": c2, "s2_expected": exp2,
    }

    # De-escalation: drop rule removed, level reset -> clones stop, delivery resumes
    s1.delete_drop_rule(drop_handle)
    s1.write_flow_levels({slots["L3_drop"]: 0, slots["L2_clone"]: 0})
    s1.reset_dp_counters()
    again = {n: mk_flow(*flows[n][:4], 10, rng) for n in ("L2_clone", "L3_drop")}
    caps2 = {"h2_in": Capture("in", H["h2"]["veth_host"], os.path.join(out, "h2_in_deesc.pcap"), "h2"),
             "s1_cpu": Capture("in", "am_s1dpi", os.path.join(out, "s1_cpu_deesc.pcap"))}
    sinks = [subprocess.Popen(ns_exec("h2", f"python3 -c \"{SINK.format(ports=[7002, 7003])}\""),
                              shell=True)]
    time.sleep(1.5)
    send("h1", again["L2_clone"] + again["L3_drop"], os.path.join(out, "sent_deesc.pcap"))
    got2 = {k: c.stop() for k, c in caps2.items()}
    for p in sinks:
        p.kill()
    checks["deescalation_removes_clone_and_drop"] = {
        "pass": len(got2["s1_cpu"]) == 0 and len(got2["h2_in"]) == 20
                and s1.count_drop_rules() == 0,
        "clones_after": len(got2["s1_cpu"]), "delivered_after": len(got2["h2_in"]),
        "drop_rules_left": s1.count_drop_rules(),
    }

    topo.stop_switches()
    topo.teardown()
    report["all_pass"] = all(c["pass"] for c in checks.values())
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    for k, c in checks.items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {k}")
    print(f"report: {os.path.join(out, 'report.json')}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
