#!/usr/bin/env python3
"""
tests/smoke_dataplane.py — Stage-1 data-plane smoke test on real bmv2,
covering Decisions B (host-pair clone rule) and C (adaptive sampling).

Run inside the sandbox (namespaced root):
    scripts/sandbox.sh python3 tests/smoke_dataplane.py

Builds the 2-switch namespace topology, configures clone session 100 via
DataPlaneManager.install_mirroring_all() (mirroring_add 100 255), then sends
crafted UDP flows h1 -> h2 (one switch) and h1 -> h3 (two switches) with
per-flow levels written into flow_level via register_write. `expected_samples()`
below is an independent re-implementation of p4/adaptmon.p4's sampling rule
(NORMAL 1-in-64, MONITOR 1-in-8, CLONE/DROP every packet), used as the
oracle throughout, exactly like utils.flow_slot() independently
re-implements the CRC32 hash. Checks:

  1. mirroring_get 100 reports port 255 on both switches
  2. flow_slot() predicts the register slot bmv2 uses for each 5-tuple
  3. clones match expected_samples() exactly per flow level, with the right
     cpu_hdr.reason (1 = full capture at CLONE/DROP, 2 = per-flow sampling)
     and flow_index/level from the field list
  4. s2 (untouched flow_level register, so NORMAL by default) independently
     samples the one flow crossing it at the same 1-in-64 rate -- clone
     sampling is a per-switch decision, not a global one
  5. forwarded originals are byte-identical to what h1 sent (pcap diff):
       h1 -> h2: s1 rule switched to forward_l2 (no rewrite) -> strict equality
       h1 -> h3: forward_l3 on 2 switches -> equal after applying the
                 documented rewrite (TTL - 2, IPv4 checksum recomputed)
  6. a DROP flow (level 3 + 5-tuple drop rule) is not delivered but is still
     fully cloned (reason 1)
  7. the data-plane fwd/clone/drop counters match the packets sent exactly
  8. a host_clone_table rule (Decision B) forces full capture (reason 3) of
     every packet from one source IP, regardless of that flow's own level,
     and stops forcing it once removed
  9. de-escalating a flow back to NORMAL (clearing its level, removing any
     drop rule) resumes delivery and drops back to 1-in-64 sampling

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
    "from scapy.all import rdpcap,conf\n"
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
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

    def stop(self):
        time.sleep(0.5)
        self.proc.terminate()
        self.proc.wait(timeout=5)
        with open(self.path + ".stats.json") as f:
            self.stats = json.load(f)
        return rdpcap(self.path) if os.path.getsize(self.path) > 24 else []


def expected_samples(level, start, n):
    """
    Independent re-implementation of p4/adaptmon.p4's sampling rule: the
    pkt_counter VALUES (not positions) among the next n packets -- i.e.
    start+1 .. start+n -- that the data plane clones at this flow level.
    """
    vals = range(start + 1, start + n + 1)
    if level >= 2:
        return list(vals)
    mod = 8 if level == 1 else 64
    return [v for v in vals if v % mod == 0]


def mk_flow(src, dst, sport, dport, n, rng, src_ip_override=None):
    pkts = []
    for _ in range(n):
        payload = bytes(rng.getrandbits(8) for _ in range(rng.randint(18, 1000)))
        pkts.append(
            Ether(src=H[src]["mac"], dst=H[dst]["mac"])
            / IP(src=src_ip_override or H[src]["ip"], dst=H[dst]["ip"],
                ttl=64, id=rng.randint(0, 65535))
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


def clones_by_tuple(cap, want_tuple):
    out = []
    for p in cap:
        hdr, frame = parse_cpu(bytes(p))
        e = Ether(frame)
        if UDP in e and l4key(e) == want_tuple:
            out.append((hdr, frame))
    return out


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
    # The sampling oracle below assumes each flow's pkt_counter is exactly 1
    # after its single probe packet (a fresh switch, no prior traffic)
    starts = {n: regs["pkt_counter"][slots[n]] for n in flows}
    assert all(v == 1 for v in starts.values()), f"unexpected probe counts: {starts}"

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

    # Main traffic: 128 packets per flow, interleaved (128 so NORMAL's
    # 1-in-64 sampling has two guaranteed hits: pkt_val 64 and 128)
    N = 128
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

    # 3. s1 clones: exact match against the sampling oracle, per flow
    s1_result = {}
    ok_s1 = True
    for n, (s, d, sp, dp, level) in flows.items():
        want_vals = expected_samples(level, starts[n], N)
        want_reason = 1 if level >= 2 else 2
        tup = (H[s]["ip"], H[d]["ip"], 17, sp, dp)
        got_n = clones_by_tuple(got["s1_cpu"], tup)
        got_counts = sorted(h["pkt_count"] for h, _ in got_n)
        hdr_ok = all(h["flow_index"] == slots[n] and h["level"] == level
                     and h["reason"] == want_reason for h, _ in got_n)
        # Frame carried by the clone == the original as it left s1's egress
        # (index i (0-based) of the main batch has pkt_count i+2)
        rewrite = (lambda b: b) if n != "X2_clone_2hop" else (lambda b: _ttl_minus(b, 1))
        frames_expected = [rewrite(sent_b[n][pc - 2]) for pc in got_counts]
        frames_ok = [fr for _, fr in sorted(got_n, key=lambda x: x[0]["pkt_count"])] == frames_expected
        entry = {"expected_pkt_counts": want_vals, "observed_pkt_counts": got_counts,
                 "expected_reason": want_reason, "hdr_fields_ok": hdr_ok, "frames_ok": frames_ok}
        ok_s1 &= got_counts == want_vals and hdr_ok and frames_ok
        s1_result[n] = entry
    checks["s1_clones_match_sampling_oracle"] = {"pass": ok_s1, "per_flow": s1_result}

    # 4. s2: independent sampling of the one flow crossing it (its own
    # flow_level register for this 5-tuple was never written -> NORMAL)
    regs2 = s2.read_registers(["pkt_counter"])
    s2_tup = (H["h1"]["ip"], H["h3"]["ip"], 17,
             flows["X2_clone_2hop"][2], flows["X2_clone_2hop"][3])
    s2_slot = flow_slot(*s2_tup[:2], 17, s2_tup[3], s2_tup[4])
    s2_start = regs2["pkt_counter"][s2_slot] - N   # before this batch
    want_s2 = expected_samples(0, s2_start, N)
    got_s2 = clones_by_tuple(got["s2_cpu"], s2_tup)
    got_s2_counts = sorted(h["pkt_count"] for h, _ in got_s2)
    checks["s2_samples_independently_at_normal_rate"] = {
        "pass": got_s2_counts == want_s2 and all(h["reason"] == 2 for h, _ in got_s2),
        "s2_slot": s2_slot, "expected": want_s2, "observed": got_s2_counts,
    }

    # 7. counters (reset after the probes)
    n_s1_clones = sum(len(expected_samples(f[4], starts[n], N)) for n, f in flows.items())
    c1, c2 = s1.read_dp_counters(), s2.read_dp_counters()
    exp1 = {"forwarded": 4 * N, "cloned": n_s1_clones, "dropped": N}
    exp2 = {"forwarded": N, "cloned": len(want_s2), "dropped": 0}
    checks["dataplane_counters_exact"] = {
        "pass": c1 == exp1 and c2 == exp2,
        "s1": c1, "s1_expected": exp1, "s2": c2, "s2_expected": exp2,
    }

    # 8. host_clone_table (Decision B): forces full capture of one source
    # IP regardless of its own flow's level
    host_src = "10.0.1.55"
    host_tup = (host_src, H["h3"]["ip"], 17, 42000, 7200)
    host_slot = flow_slot(host_src, H["h3"]["ip"], 17, 42000, 7200)
    assert host_slot not in slots.values() and host_slot != s2_slot, \
        f"test 5-tuple collides with an earlier one at slot {host_slot}"
    s1.reset_dp_counters()
    pre = mk_flow("h1", "h3", 42000, 7200, 20, rng, src_ip_override=host_src)
    send("h1", pre, os.path.join(out, "host_pre.pcap"))
    cap_pre = Capture("in", "am_s1dpi", os.path.join(out, "host_pre_cpu.pcap"))
    time.sleep(0.3)
    before = cap_pre.stop()
    start_before = s1.read_registers(["pkt_counter"])["pkt_counter"][
        flow_slot(host_src, H["h3"]["ip"], 17, 42000, 7200)] - 20

    clone_handle = s1.add_host_clone_rule(host_src)
    cap_mid = Capture("in", "am_s1dpi", os.path.join(out, "host_mid_cpu.pcap"))
    mid = mk_flow("h1", "h3", 42000, 7200, 20, rng, src_ip_override=host_src)
    send("h1", mid, os.path.join(out, "host_mid.pcap"))
    time.sleep(0.3)
    during = cap_mid.stop()

    s1.delete_host_clone_rule(clone_handle)
    cap_post = Capture("in", "am_s1dpi", os.path.join(out, "host_post_cpu.pcap"))
    # 30, not 20: start_after=40 (0+20+20), so this run of pkt_val 41..70
    # crosses 64 and gives a genuine positive expectation, not just "zero
    # clones" (which an incompletely-removed rule would also produce, by
    # coincidence, if checked only against a too-short run)
    post = mk_flow("h1", "h3", 42000, 7200, 30, rng, src_ip_override=host_src)
    send("h1", post, os.path.join(out, "host_post.pcap"))
    time.sleep(0.3)
    after = cap_post.stop()

    before_c = clones_by_tuple(before, host_tup)
    during_c = clones_by_tuple(during, host_tup)
    after_c = clones_by_tuple(after, host_tup)
    want_before = expected_samples(0, start_before, 20)
    want_after_start = start_before + 20 + 20
    want_after = expected_samples(0, want_after_start, 30)
    checks["host_clone_rule_forces_full_capture"] = {
        "pass": (sorted(h["pkt_count"] for h, _ in before_c) == want_before
                 and len(during_c) == 20
                 and all(h["reason"] == 3 and h["level"] == 0 for h, _ in during_c)
                 and sorted(h["pkt_count"] for h, _ in after_c) == want_after
                 and s1.count_host_clone_rules() == 0),
        "before_sampled": sorted(h["pkt_count"] for h, _ in before_c),
        "before_expected": want_before,
        "during_count": len(during_c), "during_reasons": [h["reason"] for h, _ in during_c],
        "after_sampled": sorted(h["pkt_count"] for h, _ in after_c),
        "after_expected": want_after,
    }

    # 9. de-escalation: drop rule removed, level reset -> delivery resumes,
    # sampling returns to the per-level rate (not to zero: NORMAL still
    # samples 1-in-64)
    pre_counts = s1.read_registers(["pkt_counter"])["pkt_counter"]
    start_clone = pre_counts[slots["L2_clone"]]
    start_drop = pre_counts[slots["L3_drop"]]
    s1.delete_drop_rule(drop_handle)
    s1.write_flow_levels({slots["L3_drop"]: 0, slots["L2_clone"]: 0})
    s1.reset_dp_counters()
    N2 = 10
    again = {n: mk_flow(*flows[n][:4], N2, rng) for n in ("L2_clone", "L3_drop")}
    caps2 = {"h2_in": Capture("in", H["h2"]["veth_host"], os.path.join(out, "h2_in_deesc.pcap"), "h2"),
             "s1_cpu": Capture("in", "am_s1dpi", os.path.join(out, "s1_cpu_deesc.pcap"))}
    sinks = [subprocess.Popen(ns_exec("h2", f"python3 -c \"{SINK.format(ports=[7002, 7003])}\""),
                              shell=True)]
    time.sleep(1.5)
    send("h1", again["L2_clone"] + again["L3_drop"], os.path.join(out, "sent_deesc.pcap"))
    got2 = {k: c.stop() for k, c in caps2.items()}
    for p in sinks:
        p.kill()
    want_clone2 = expected_samples(0, start_clone, N2)
    want_drop2 = expected_samples(0, start_drop, N2)
    clone2_tup = (H["h1"]["ip"], H["h2"]["ip"], 17, flows["L2_clone"][2], flows["L2_clone"][3])
    drop2_tup = (H["h1"]["ip"], H["h2"]["ip"], 17, flows["L3_drop"][2], flows["L3_drop"][3])
    got_clone2 = sorted(h["pkt_count"] for h, _ in clones_by_tuple(got2["s1_cpu"], clone2_tup))
    got_drop2 = sorted(h["pkt_count"] for h, _ in clones_by_tuple(got2["s1_cpu"], drop2_tup))
    checks["deescalation_resumes_delivery_and_returns_to_normal_sampling"] = {
        "pass": (len(got2["h2_in"]) == 2 * N2 and s1.count_drop_rules() == 0
                 and got_clone2 == want_clone2 and got_drop2 == want_drop2),
        "delivered_after": len(got2["h2_in"]), "drop_rules_left": s1.count_drop_rules(),
        "clone_flow_sampled": got_clone2, "clone_flow_expected": want_clone2,
        "drop_flow_sampled": got_drop2, "drop_flow_expected": want_drop2,
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


def _ttl_minus(frame_bytes, n):
    e = Ether(frame_bytes)
    e[IP].ttl -= n
    del e[IP].chksum
    return bytes(Ether(bytes(e)))


if __name__ == "__main__":
    main()
