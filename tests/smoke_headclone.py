#!/usr/bin/env python3
"""
tests/smoke_headclone.py -- flow-head cloning on real bmv2 (run inside
scripts/sandbox.sh):

    scripts/sandbox.sh python3 tests/smoke_headclone.py

head_clone_n = 3 on s1, 0 on s2. Crafted UDP flows from h1 through s1 (one
also through s2), each clone parsed off the CPU port and compared, packet by
packet, with an independent oracle (SlotOracle below re-implements the
p4/adaptmon.p4 rule: a flow is new on an empty slot or a 5-tuple change; a
packet is cloned for its level (reason 1) if the slot is at CLONE/DROP, else
for the head (4) if it is among the first N of its flow, else for sampling
(2) at 1-in-8 MONITOR / 1-in-64 NORMAL of the slot's pkt_counter). Checks:

  1. head_clone_n reads back as written on both switches
  2. NORMAL flow: packets 1-3 cloned (reason 4), 4th not; 64th sampled (2)
  3. MONITOR flow: 1-3 head, then 8 and 16 sampled
  4. CLONE flow: every packet cloned for its level (reason 1), incl. 1-3
  5. hash collision: a new 5-tuple in an occupied slot gets its own head
     clones, and so does the original flow when it comes back (ping-pong)
  6. a slot the controller aged out (clear_flow_entries) restarts the head
  7. s2 (N=0): no head clones for the flow crossing it
  8. N changed at runtime (3 -> 5) applies to the next new flow
  9. per-reason clone counters, cloned/forwarded packets AND bytes on s1
     equal what the CPU-port capture and the sent frames say, exactly

Output: results/smoke_headclone_<timestamp>/report.json
"""

import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from scapy.all import Ether, IP, UDP   # noqa: E402

from smoke_dataplane import H, Capture, mk_flow, send, parse_cpu, ns_exec, SINK, l4key  # noqa: E402
from topo.namespace_topo import NetworkNamespaceTopo           # noqa: E402
from controller.dataplane import DataPlaneManager              # noqa: E402
from controller.utils import flow_slot, get_timestamp_str      # noqa: E402


class SlotOracle:
    """Per-switch model of the slot state p4/adaptmon.p4 keeps (pkt_counter,
    stored 5-tuple, head_pkts, flow_level) and of its clone decision."""

    def __init__(self, head_n):
        self.head_n = head_n
        self.slots = {}      # slot -> dict(tup, pkt, head)
        self.level = {}      # slot -> level

    def packet(self, tup):
        slot = flow_slot(*tup)
        st = self.slots.setdefault(slot, {"tup": None, "pkt": 0, "head": 0})
        new = st["pkt"] == 0 or st["tup"] != tup
        st["head"] = 1 if new else st["head"] + 1
        st["pkt"] += 1
        st["tup"] = tup
        lv = self.level.get(slot, 0)
        if lv >= 2:
            reason = 1
        elif st["head"] <= self.head_n:
            reason = 4
        elif lv == 1 and st["pkt"] % 8 == 0:
            reason = 2
        elif lv == 0 and st["pkt"] % 64 == 0:
            reason = 2
        else:
            reason = None
        return (st["pkt"], reason) if reason else None

    def clear(self, slot):
        self.slots.setdefault(slot, {"tup": None, "pkt": 0, "head": 0})["pkt"] = 0


def main():
    out = os.path.join(ROOT, "results", f"smoke_headclone_{get_timestamp_str()}")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(7)
    checks, report = {}, {"out_dir": out}

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
    s1, s2 = dpm.get_switch(1), dpm.get_switch(2)

    rb = {1: s1.set_head_clone_n(3), 2: s2.set_head_clone_n(0)}
    checks["head_clone_n_reads_back"] = {"pass": rb == {1: 3, 2: 0}, "readback": rb}

    h2_ports, h3_port = list(range(7300, 7307)), 7400
    sinks = [subprocess.Popen(ns_exec("h2", f"python3 -c \"{SINK.format(ports=h2_ports)}\""),
                              shell=True),
             subprocess.Popen(ns_exec("h3", f"python3 -c \"{SINK.format(ports=[h3_port])}\""),
                              shell=True)]
    time.sleep(1)

    ip1, ip2 = H["h1"]["ip"], H["h2"]["ip"]
    F = {   # name: (dst host, sport, dport)
        "A_normal":  ("h2", 43000, 7300),
        "B_monitor": ("h2", 43001, 7301),
        "C_clone":   ("h2", 43002, 7302),
        "E_aged":    ("h2", 43004, 7304),
        "F_2hop":    ("h3", 43005, h3_port),
        "G_n5":      ("h2", 43006, 7305),
    }
    tup = {n: (ip1, H[d]["ip"], 17, sp, dp) for n, (d, sp, dp) in F.items()}
    slot = {n: flow_slot(*t) for n, t in tup.items()}
    assert len(set(slot.values())) == len(slot), f"slot collision among test flows: {slot}"
    # D: a different 5-tuple that hashes to A's slot
    d_sport = next(sp for sp in range(44000, 65000)
                   if flow_slot(ip1, ip2, 17, sp, 7303) == slot["A_normal"])
    tup["D_collide"] = (ip1, ip2, 17, d_sport, 7303)
    slot["D_collide"] = slot["A_normal"]
    report["slots"] = slot

    o1, o2 = SlotOracle(3), SlotOracle(0)
    o1.level[slot["B_monitor"]] = 1
    o1.level[slot["C_clone"]] = 2
    s1.write_flow_levels({slot["B_monitor"]: 1, slot["C_clone"]: 2})

    def pk(name, n):
        d, sp, dp = F.get(name, ("h2", d_sport, 7303))
        return mk_flow("h1", d, sp, dp, n, rng)

    # Phases in send order: (name, packets, action-before)
    phases = [
        ("A_normal", pk("A_normal", 70), None),
        ("B_monitor", pk("B_monitor", 20), None),
        ("C_clone", pk("C_clone", 10), None),
        ("D_collide", pk("D_collide", 5), None),
        ("A_normal", pk("A_normal", 2), None),        # A returns: ping-pong
        ("E_aged", pk("E_aged", 5), None),
        ("E_aged", pk("E_aged", 5), "clear_E"),       # slot aged out in between
        ("F_2hop", pk("F_2hop", 10), None),
        ("G_n5", pk("G_n5", 8), "n5"),                # N 3 -> 5 at runtime
    ]

    s1.reset_dp_counters()
    s2.reset_dp_counters()
    caps = {1: Capture("in", "am_s1dpi", os.path.join(out, "s1_cpu.pcap")),
            2: Capture("in", "am_s2dpi", os.path.join(out, "s2_cpu.pcap"))}
    time.sleep(1.5)

    expected = {1: {}, 2: {}}       # switch -> name -> [(pkt_count, reason)]
    sent_frames = []
    for i, (name, pkts, action) in enumerate(phases):
        if action == "clear_E":
            s1.clear_flow_entries([slot["E_aged"]])
            o1.clear(slot["E_aged"])
        elif action == "n5":
            rb5 = s1.set_head_clone_n(5)
            o1.head_n = 5
            report["n5_readback"] = rb5
        for p in pkts:
            e1 = o1.packet(tup[name])
            if e1:
                expected[1].setdefault(name, []).append(e1)
            if name == "F_2hop":
                e2 = o2.packet(tup[name])
                if e2:
                    expected[2].setdefault(name, []).append(e2)
        send("h1", pkts, os.path.join(out, f"phase_{i}.pcap"))
        sent_frames += pkts
        time.sleep(0.3)

    got = {sid: c.stop() for sid, c in caps.items()}
    for p in sinks:
        p.kill()

    observed = {1: {}, 2: {}}
    clone_bytes = {1: 0, 2: 0}
    by_reason = {1: {}, 2: {}}
    name_of = {t: n for n, t in tup.items()}
    for sid in (1, 2):
        for p in got[sid]:
            hdr, frame = parse_cpu(bytes(p))
            e = Ether(frame)
            if UDP not in e or l4key(e) not in name_of:
                continue
            observed[sid].setdefault(name_of[l4key(e)], []).append((hdr["pkt_count"], hdr["reason"]))
            clone_bytes[sid] += len(frame)
            by_reason[sid][hdr["reason"]] = by_reason[sid].get(hdr["reason"], 0) + 1

    def flow_check(sid, name, note):
        exp, obs = expected[sid].get(name, []), observed[sid].get(name, [])
        return {"pass": exp == obs, "expected": exp, "observed": obs, "note": note}

    checks["normal_flow_head_1_3_cloned_4th_not_64th_sampled"] = flow_check(
        1, "A_normal", "expected[:3] are reason 4 at pkt 1-3; pkt 4 absent; pkt 64 reason 2")
    checks["normal_flow_head_shape"] = {
        "pass": expected[1]["A_normal"][:4] == [(1, 4), (2, 4), (3, 4), (64, 2)],
        "first4_expected": expected[1]["A_normal"][:4]}
    checks["monitor_flow_head_then_1_in_8"] = flow_check(1, "B_monitor", "1-3 head, 8 & 16 sampled")
    checks["clone_level_overrides_head"] = flow_check(1, "C_clone", "all 10 reason 1")
    checks["collision_new_tuple_gets_its_own_head"] = flow_check(
        1, "D_collide", f"sport {d_sport} -> A's slot {slot['A_normal']}")
    checks["aged_slot_restarts_head"] = flow_check(1, "E_aged", "heads 1-3 twice, pkt_count restarts")
    checks["two_hop_flow_head_on_s1"] = flow_check(1, "F_2hop", "s1 N=3")
    checks["s2_n0_no_head_clones"] = {
        "pass": observed[2].get("F_2hop", []) == expected[2].get("F_2hop", []) == [],
        "observed": observed[2].get("F_2hop", [])}
    checks["runtime_n_change_applies"] = flow_check(1, "G_n5", "N=5: pkts 1-5 reason 4")

    # A's two returning packets (after D took the slot) -- the ping-pong case
    a_back = observed[1].get("A_normal", [])[-2:]
    checks["collision_ping_pong_documented"] = {
        "pass": a_back == [(76, 4), (77, 4)],
        "observed_last_two_A_clones": a_back,
        "note": "a flow returning to a slot another 5-tuple has since taken is "
                "a new flow again: its next N packets are head-cloned",
    }

    c1, c2 = s1.read_dp_counters_full(), s2.read_dp_counters_full()
    report["s1_counters"], report["s2_counters"] = c1, c2
    s1_sent_bytes = sum(len(p) for p in sent_frames)
    n_exp = {r: sum(1 for v in expected[1].values() for _, rr in v if rr == r) for r in (1, 2, 3, 4)}
    checks["s1_counters_exact"] = {
        "pass": (c1["forwarded"] == len(sent_frames)
                 and c1["forwarded_bytes"] == s1_sent_bytes
                 and c1["cloned"] == sum(n_exp.values()) == sum(by_reason[1].values())
                 and c1["cloned_bytes"] == clone_bytes[1]
                 and (c1["clone_level"], c1["clone_sample"], c1["clone_hostpair"], c1["clone_head"])
                 == (n_exp[1], n_exp[2], n_exp[3], n_exp[4])),
        "forwarded": (c1["forwarded"], len(sent_frames)),
        "forwarded_bytes": (c1["forwarded_bytes"], s1_sent_bytes),
        "cloned": (c1["cloned"], sum(n_exp.values())),
        "cloned_bytes": (c1["cloned_bytes"], clone_bytes[1]),
        "per_reason_counter": {r: c1[f"clone_{n}"] for r, n in
                               ((1, "level"), (2, "sample"), (3, "hostpair"), (4, "head"))},
        "per_reason_expected": n_exp,
    }
    checks["s2_counters_exact"] = {
        "pass": c2["forwarded"] == 10 and c2["cloned"] == 0 and c2["clone_head"] == 0,
        "s2": {k: c2[k] for k in ("forwarded", "cloned", "clone_head")},
    }

    topo.stop_switches()
    topo.teardown()
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
