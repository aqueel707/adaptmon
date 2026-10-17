#!/usr/bin/env python3
"""
tools/pcap_capture.py — minimal AF_PACKET capture to a pcap file.

Replaces tcpdump inside scripts/sandbox.sh: tcpdump insists on dropping
privileges (setgroups/chown), which an unprivileged user namespace forbids,
and exits right after writing the pcap header.

    pcap_capture.py IFACE OUT.pcap [--dir in|out|any] [--snaplen N]

--dir in  : packets received on IFACE (PACKET_HOST/BROADCAST/MULTICAST/OTHERHOST)
--dir out : packets sent from IFACE   (PACKET_OUTGOING)
Timestamps are the kernel receive time (SIOCGSTAMPNS-equivalent via
SO_TIMESTAMPNS). On SIGTERM/SIGINT the file is flushed and a JSON line
{"captured": n, "kernel_drops": d} is written to stderr (and to
OUT.pcap.stats.json), so callers can verify the capture was complete.
"""

import argparse
import json
import signal
import socket
import struct
import sys

ETH_P_ALL = 0x0003
PACKET_OUTGOING = 4
SOL_PACKET = 263
PACKET_STATISTICS = 6
SO_TIMESTAMPNS = 35
SCM_TIMESTAMPNS = SO_TIMESTAMPNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("iface")
    ap.add_argument("out")
    ap.add_argument("--dir", choices=["in", "out", "any"], default="any")
    ap.add_argument("--snaplen", type=int, default=65535)
    args = ap.parse_args()

    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    s.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
    s.bind((args.iface, 0))
    s.settimeout(0.2)   # lets the loop notice SIGTERM

    f = open(args.out, "wb", buffering=1 << 20)
    # pcap global header, microsecond resolution, LINKTYPE_ETHERNET
    f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, args.snaplen, 1))
    f.flush()
    n = 0
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while not stop:
        try:
            data, anc, _, addr = s.recvmsg(65535, socket.CMSG_SPACE(16))
        except (socket.timeout, InterruptedError):
            continue
        pkttype = addr[2]
        if args.dir == "out" and pkttype != PACKET_OUTGOING:
            continue
        if args.dir == "in" and pkttype == PACKET_OUTGOING:
            continue
        sec, nsec = 0, 0
        for lvl, typ, cdata in anc:
            if lvl == socket.SOL_SOCKET and typ == SCM_TIMESTAMPNS:
                sec, nsec = struct.unpack("qq", cdata[:16])
        cap = data[:args.snaplen]
        f.write(struct.pack("<IIII", sec, nsec // 1000, len(cap), len(data)))
        f.write(cap)
        n += 1

    # tp_packets, tp_drops (counters since bind; reading resets them)
    try:
        st = s.getsockopt(SOL_PACKET, PACKET_STATISTICS, 8)
        _, drops = struct.unpack("II", st)
    except OSError:
        drops = -1
    f.close()
    stats = {"iface": args.iface, "dir": args.dir, "captured": n, "kernel_drops": drops}
    with open(args.out + ".stats.json", "w") as sf:
        json.dump(stats, sf)
    print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main()
