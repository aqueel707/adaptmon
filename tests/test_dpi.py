"""Unit tests for the DPI rules (controller/dpi.py) with crafted Scapy packets."""

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scapy.all import Ether, IP, TCP, UDP, Raw

from controller.dpi import DPIEngine, CPU_HDR

ATK, TGT = "10.0.1.11", "10.0.2.1"


def clone(pkt, idx=7, level=2):
    return CPU_HDR.pack(idx, level, 1, len(pkt), 1) + bytes(Ether() / pkt)


def feed(engine, pkts, switch=1):
    out = []
    for i, p in enumerate(pkts):
        out += engine.process(clone(p), switch, 1000.0 + i)
    return out


def classes(alerts):
    return sorted({a["attack_class"] for a in alerts})


def test_cpu_header_is_stripped_and_parsed():
    e = DPIEngine()
    raw = CPU_HDR.pack(1234, 3, 42, 5000, 1) + bytes(
        Ether() / IP(src=ATK, dst=TGT) / TCP(sport=1, dport=2, flags=0))
    a = e.process(raw, 2, 5.0)
    assert len(a) == 1 and a[0]["flow_idx"] == 1234 and a[0]["level"] == 3
    assert a[0]["switch"] == 2
    assert a[0]["five_tuple"] == {"src": ATK, "dst": TGT, "proto": 6, "sport": 1, "dport": 2}


def test_syn_scan_detected_over_5_ports():
    e = DPIEngine()
    pkts = [IP(src=ATK, dst=TGT) / TCP(sport=40000, dport=p, flags="S")
            for p in (21, 22, 23, 25, 80, 443, 8080)]
    a = feed(e, pkts)
    assert classes(a) == ["syn_scan"]
    # all 7 SYN-only 5-tuples confirmed exactly once
    assert len(a) == 7
    assert a[0]["evidence"]["distinct_dst_ports"] >= 5


def test_syn_scan_needs_5_distinct_ports():
    e = DPIEngine()
    pkts = [IP(src=ATK, dst=TGT) / TCP(sport=40000, dport=p, flags="S")
            for p in (21, 22, 23, 25)] * 3
    assert feed(e, pkts) == []


def test_syn_scan_needs_ratio_above_0_9():
    e = DPIEngine()
    # 6 ports but a normal handshake mix: SYN, ACK, PSH/ACK ... (ratio 1/3)
    pkts = []
    for p in (21, 22, 23, 25, 80, 443):
        pkts += [IP(src=ATK, dst=TGT) / TCP(sport=40000 + p, dport=p, flags="S"),
                 IP(src=ATK, dst=TGT) / TCP(sport=40000 + p, dport=p, flags="A"),
                 IP(src=ATK, dst=TGT) / TCP(sport=40000 + p, dport=p, flags="PA") / Raw(b"x")]
    assert feed(e, pkts) == []


def test_null_scan():
    e = DPIEngine()
    a = feed(e, [IP(src=ATK, dst=TGT) / TCP(sport=40000, dport=p, flags=0) for p in (22, 80)])
    assert classes(a) == ["null_scan"] and len(a) == 2


def test_null_rule_ignores_flagged_tcp():
    e = DPIEngine()
    assert feed(e, [IP(src=ATK, dst=TGT) / TCP(sport=1, dport=80, flags="A")]) == []


def http_post(body, path="/login"):
    req = (f"POST {path} HTTP/1.1\r\nHost: 10.0.2.1\r\n"
           f"Content-Type: application/x-www-form-urlencoded\r\n"
           f"Content-Length: {len(body)}\r\n\r\n{body}").encode()
    return IP(src=ATK, dst=TGT) / TCP(sport=40001, dport=80, flags="PA") / Raw(req)


@pytest.mark.parametrize("body,pattern", [
    ("user=admin%27+OR+1%3D1&pw=x", "or_1_eq_1"),
    ("user=admin' or 1=1&pw=x", "or_1_eq_1"),
    ("id=1+UNION+SELECT+username%2Cpassword+FROM+users", "union_select"),
    ("id=1 union all select null", "union_select"),
    ("user=admin%27--&pw=x", "sql_comment_dashes"),
    ("user=admin%27%2F%2A&pw=x", "sql_comment_block"),
])
def test_sqli_patterns(body, pattern):
    e = DPIEngine()
    a = feed(e, [http_post(body)])
    assert classes(a) == ["sqli"]
    assert pattern in a[0]["evidence"]["patterns"]


def test_sqli_in_get_query():
    e = DPIEngine()
    req = b"GET /item?id=1%20UNION%20SELECT%201 HTTP/1.1\r\nHost: x\r\n\r\n"
    a = feed(e, [IP(src=ATK, dst=TGT) / TCP(sport=4, dport=80, flags="PA") / Raw(req)])
    assert classes(a) == ["sqli"]


def test_sqli_body_in_later_segment_of_http_flow():
    e = DPIEngine()
    hdr = b"POST /login HTTP/1.1\r\nHost: x\r\nContent-Length: 20\r\n\r\n"
    p1 = IP(src=ATK, dst=TGT) / TCP(sport=5, dport=80, flags="PA") / Raw(hdr)
    p2 = IP(src=ATK, dst=TGT) / TCP(sport=5, dport=80, flags="PA") / Raw(b"u=a%27+OR+1%3D1")
    assert classes(feed(e, [p1, p2])) == ["sqli"]


def test_benign_http_post_not_flagged():
    e = DPIEngine()
    assert feed(e, [http_post("name=John+Smith&comment=great+product%21&rating=5")]) == []


def test_sqli_only_in_http_payloads():
    e = DPIEngine()
    # Non-HTTP TCP payload (e.g. iperf3 bulk data) containing "--" and "/*"
    p = IP(src="10.0.1.1", dst=TGT) / TCP(sport=5, dport=5201, flags="PA") / Raw(b"ab--cd/*ef OR 1=1")
    assert feed(e, [p]) == []


def test_udp_not_matched_and_entropy_reported():
    e = DPIEngine()
    assert feed(e, [IP(src=ATK, dst=TGT) / UDP(sport=1, dport=9999) / Raw(bytes(range(256)))]) == []
    rec = next(iter(e.flows.values()))
    from controller.dpi import byte_entropy
    assert abs(byte_entropy(rec.hist) - 8.0) < 1e-9


def test_each_flow_class_confirmed_once():
    e = DPIEngine()
    p = IP(src=ATK, dst=TGT) / TCP(sport=1, dport=80, flags=0)
    assert len(feed(e, [p, p, p])) == 1


def test_confirmed_alert_tags_dpi_detector():
    e = DPIEngine()
    a = feed(e, [IP(src=ATK, dst=TGT) / TCP(sport=1, dport=2, flags=0)])
    assert a[0]["detector"] == "dpi"


def test_entropy_snapshot_keyed_for_monitor_lookup():
    """Decision C: controller/features.py.entropy_snapshot_key() must match
    the keys entropy_snapshot() produces."""
    from controller.dpi import byte_entropy
    from controller.features import entropy_snapshot_key

    e = DPIEngine()
    feed(e, [IP(src=ATK, dst=TGT) / UDP(sport=1, dport=2) / Raw(bytes(range(256)))])
    snap = e.entropy_snapshot()
    key = entropy_snapshot_key(1, (ATK, TGT, 17, 1, 2))
    assert key in snap
    assert snap[key]["n"] == 256 and snap[key]["clones"] == 1
    assert abs(snap[key]["entropy"] - 8.0) < 1e-3


def test_entropy_snapshot_n_zero_before_any_payload():
    e = DPIEngine()
    feed(e, [IP(src=ATK, dst=TGT) / TCP(sport=1, dport=2, flags="S")])
    snap = e.entropy_snapshot()
    from controller.features import entropy_snapshot_key
    key = entropy_snapshot_key(1, (ATK, TGT, 6, 1, 2))
    assert snap[key]["n"] == 0 and snap[key]["entropy"] == 0.0
