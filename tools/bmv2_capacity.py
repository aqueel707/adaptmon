#!/usr/bin/env python3
"""tools/bmv2_capacity.py OUT_DIR — paced UDP h1->h3 through s1 at 500..8000 pkt/s,
64/1000-byte payloads, with and without cloning; reads the fwd/clone counters.
Run inside scripts/sandbox.sh. Writes OUT_DIR/capacity.json."""
import os, sys, time, subprocess, json
sys.path.insert(0, os.getcwd())
from scapy.all import Ether, IP, UDP, Raw, wrpcap
from topo.namespace_topo import NetworkNamespaceTopo, HOSTS
from controller.dataplane import DataPlaneManager
from controller.utils import flow_slot
H = {h["name"]: h for h in HOSTS}
SENDER = ("import sys,time\nfrom scapy.all import rdpcap,conf\n"
          "sock=conf.L2socket(iface=sys.argv[2]); rate=float(sys.argv[3]); ps=rdpcap(sys.argv[1])\n"
          "t0=time.time()\nfor i,p in enumerate(ps):\n    sock.send(p)\n"
          "    d=t0+i/rate-time.time()\n    if d>0: time.sleep(d)\nprint(time.time()-t0)\n")
topo = NetworkNamespaceTopo(log_dir=sys.argv[1]); topo.teardown(); topo.build(); topo.start_switches(); time.sleep(1); topo.install_forwarding_rules()
dpm = DataPlaneManager([{"switch_id":1,"thrift_port":9090},{"switch_id":2,"thrift_port":9091}]); dpm.wait_for_switches(30); dpm.install_mirroring_all()
s1 = dpm.get_switch(1)
sink = subprocess.Popen(["ip","netns","exec",H["h3"]["ns"],sys.executable,"-c","import socket\ns=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\ns.bind(('0.0.0.0',6000))\nwhile True: s.recv(4096)"])
res = []
for size in (64, 1000):
  for clone in (0, 1):
    for rate in (500, 1000, 2000, 4000, 8000):
        sport = 20000 + rate + clone*7 + size
        p = Ether(src=H["h1"]["mac"], dst=H["h3"]["mac"])/IP(src=H["h1"]["ip"], dst=H["h3"]["ip"])/UDP(sport=sport, dport=6000)/Raw(b"z"*size)
        n = int(rate*3)
        wrpcap(sys.argv[1]+"/b.pcap", [p]*n)
        s1.write_flow_levels({flow_slot(H["h1"]["ip"], H["h3"]["ip"], 17, sport, 6000): 2 if clone else 0})
        s1.reset_dp_counters()
        r = subprocess.run(["ip","netns","exec",H["h1"]["ns"],sys.executable,"-c",SENDER,sys.argv[1]+"/b.pcap",H["h1"]["veth_host"],str(rate)],capture_output=True,text=True)
        time.sleep(2)
        c = s1.read_dp_counters()
        el = float(r.stdout.strip().splitlines()[-1])
        res.append(dict(size=size, clone=clone, target_pps=rate, sent=n, achieved_send_pps=round(n/el), fwd=c["forwarded"], cloned=c["cloned"], fwd_frac=round(c["forwarded"]/n,3)))
        print(res[-1], flush=True)
json.dump(res, open(sys.argv[1]+"/capacity.json","w"), indent=1)
sink.kill(); topo.stop_switches(); topo.teardown()
