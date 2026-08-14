"""
topo/fat_tree.py
Multi-switch Mininet topology for AdaptMon experiments.
Topology: 4 hosts, 2 core switches, inter-switch link.

h1 -- s1 -- s2 -- h3
h2 --/            \-- h4

Usage:
    sudo python3 topo/fat_tree.py [--json-config <path>]
"""

import argparse
import json
import os
import sys
import time
import subprocess
import logging

try:
    from mininet.net    import Mininet
    from mininet.node   import OVSSwitch, DefaultController, RemoteController
    from mininet.link   import TCLink
    from mininet.log    import setLogLevel, info
    from mininet.cli    import CLI
    from mininet.topo   import Topo
    MININET_AVAILABLE = True
except ImportError:
    MININET_AVAILABLE = False

logger = logging.getLogger(__name__)

# BMv2 simple_switch binary
SIMPLE_SWITCH = os.environ.get(
    "SIMPLE_SWITCH", "simple_switch"
)

P4_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "p4", "build", "adaptmon.json"
)

THRIFT_BASE = 9090
LOG_DIR     = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs"
)


class AdaptMonTopo:
    """
    Two-switch four-host topology using BMv2 simple_switch.
    Does NOT use Mininet OVS switches — uses P4 software switches.
    """

    def __init__(
        self,
        p4_json     : str  = P4_JSON,
        thrift_base : int  = THRIFT_BASE,
        log_dir     : str  = LOG_DIR,
        pcap        : bool = False,
    ):
        self.p4_json     = p4_json
        self.thrift_base = thrift_base
        self.log_dir     = log_dir
        self.pcap        = pcap
        self._procs      = {}   # switch_id -> Popen
        os.makedirs(log_dir, exist_ok=True)

    def _start_switch(
        self,
        switch_id   : int,
        interfaces  : list,   # list of (iface_name, port_number) tuples
    ) -> subprocess.Popen:
        """Start a single BMv2 simple_switch process."""
        thrift_port = self.thrift_base + switch_id - 1
        log_file    = os.path.join(
            self.log_dir, f"switch_{switch_id}.log"
        )

        cmd = [SIMPLE_SWITCH]
        cmd += ["--thrift-port", str(thrift_port)]
        cmd += ["--log-console"]
        cmd += ["--log-level", "warn"]

        for iface, port_num in interfaces:
            cmd += ["-i", f"{port_num}@{iface}"]

        if self.pcap:
            cmd += ["--pcap", self.log_dir]

        cmd += [self.p4_json]

        logger.info(f"Starting switch {switch_id}: {' '.join(cmd)}")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=lf,
                preexec_fn=os.setsid
            )
        self._procs[switch_id] = proc
        return proc

    def stop_all(self) -> None:
        """Stop all switch processes."""
        import signal
        for sid, proc in self._procs.items():
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                logger.info(f"Switch {sid} stopped")
            except Exception as e:
                logger.warning(f"Could not stop switch {sid}: {e}")
        self._procs.clear()

    def get_switch_proc(self, switch_id: int) -> subprocess.Popen:
        return self._procs.get(switch_id)


class MininetAdaptMonTopo(Topo):
    """
    Mininet topology definition for AdaptMon.
    Uses standard OVS for host connectivity; P4 switches run separately.
    This class creates the namespace/veth infrastructure.
    """

    def build(
        self,
        n_switches : int = 2,
        n_hosts    : int = 4,
        **opts
    ):
        switches = []
        for i in range(1, n_switches + 1):
            s = self.addSwitch(f"s{i}")
            switches.append(s)

        hosts = []
        for i in range(1, n_hosts + 1):
            # Assign hosts to switches round-robin
            sw_idx = (i - 1) % n_switches
            sw_name = switches[sw_idx]
            ip  = f"10.0.{sw_idx+1}.{i}/24"
            mac = f"00:00:00:00:{sw_idx+1:02d}:{i:02d}"
            h = self.addHost(
                f"h{i}",
                ip=ip,
                mac=mac,
                defaultRoute=f"via 10.0.{sw_idx+1}.254"
            )
            hosts.append(h)
            self.addLink(h, sw_name, bw=100, delay="1ms")

        # Inter-switch link
        if n_switches > 1:
            self.addLink(
                switches[0], switches[1],
                bw=1000, delay="0.5ms"
            )


def start_mininet_topo(
    p4_json       : str  = P4_JSON,
    thrift_base   : int  = THRIFT_BASE,
    log_dir       : str  = LOG_DIR,
    interactive   : bool = False,
    duration      : int  = 300,
) -> dict:
    """
    Start the Mininet topology with P4 switches.
    Returns dict with host/switch info for experiment framework.
    """
    if not MININET_AVAILABLE:
        raise RuntimeError(
            "Mininet not available. Install mininet package."
        )

    setLogLevel("info")

    topo = MininetAdaptMonTopo(n_switches=2, n_hosts=4)

    net = Mininet(
        topo       = topo,
        link       = TCLink,
        controller = None,
        switch     = OVSSwitch,
    )

    net.start()

    # Retrieve host info
    host_info = {}
    for h in net.hosts:
        host_info[h.name] = {
            "ip"   : h.IP(),
            "mac"  : h.MAC(),
            "intf" : h.intfNames(),
        }

    switch_info = {}
    for s in net.switches:
        switch_info[s.name] = {
            "intfs": s.intfNames(),
        }

    logger.info(f"Hosts: {host_info}")
    logger.info(f"Switches: {switch_info}")

    if interactive:
        CLI(net)
    else:
        info(f"Running for {duration} seconds...\n")
        time.sleep(duration)

    net.stop()

    return {
        "hosts"   : host_info,
        "switches": switch_info,
    }


def generate_switch_config(
    n_switches  : int = 2,
    thrift_base : int = THRIFT_BASE,
) -> list:
    """Generate switch config list for the controller."""
    configs = []
    for i in range(1, n_switches + 1):
        configs.append({
            "switch_id"   : i,
            "thrift_port" : thrift_base + i - 1,
        })
    return configs


def generate_forwarding_rules(
    n_switches : int = 2,
    n_hosts    : int = 4,
) -> list:
    """
    Generate forwarding rules for the topology.
    Hosts h1,h2 on s1; h3,h4 on s2.
    Inter-switch port = 3 on each switch.
    """
    rules = []

    host_assignments = {}
    for i in range(1, n_hosts + 1):
        sw_id = ((i - 1) % n_switches) + 1
        sw_port = ((i - 1) // n_switches) + 1
        sw_net = sw_id
        host_ip = f"10.0.{sw_net}.{i}"
        host_assignments[f"h{i}"] = {
            "ip"     : host_ip,
            "sw_id"  : sw_id,
            "sw_port": sw_port,
        }

    # For each switch, add local host routes
    for hname, hinfo in host_assignments.items():
        sw_id   = hinfo["sw_id"]
        ip      = hinfo["ip"]
        port    = hinfo["sw_port"]
        rules.append({
            "switch_id": sw_id,
            "dst_ip"   : ip,
            "port"     : port,
        })
        # For remote hosts, route through inter-switch port (3)
        for hname2, hinfo2 in host_assignments.items():
            if hinfo2["sw_id"] != sw_id:
                rules.append({
                    "switch_id": sw_id,
                    "dst_ip"   : hinfo2["ip"],
                    "port"     : 3,  # inter-switch port
                })

    # Deduplicate
    seen = set()
    unique_rules = []
    for r in rules:
        key = (r["switch_id"], r["dst_ip"], r["port"])
        if key not in seen:
            seen.add(key)
            unique_rules.append(r)

    return unique_rules


def write_p4app_config(
    output_path : str,
    p4_json     : str  = P4_JSON,
    thrift_base : int  = THRIFT_BASE,
    n_switches  : int  = 2,
    n_hosts     : int  = 4,
) -> None:
    """Write a JSON config file for the experiment framework."""
    config = {
        "p4_json"        : p4_json,
        "thrift_base"    : thrift_base,
        "n_switches"     : n_switches,
        "n_hosts"        : n_hosts,
        "switch_configs" : generate_switch_config(n_switches, thrift_base),
        "forwarding"     : generate_forwarding_rules(n_switches, n_hosts),
        "hosts"          : {
            f"h{i}": {
                "ip" : f"10.0.{((i-1)%n_switches)+1}.{i}",
                "gw" : f"10.0.{((i-1)%n_switches)+1}.254",
            }
            for i in range(1, n_hosts + 1)
        }
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Topology config written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaptMon Mininet Topology")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--p4-json", type=str, default=P4_JSON)
    parser.add_argument(
        "--write-config", type=str, default=None,
        help="Write topology config to JSON file"
    )
    args = parser.parse_args()

    if args.write_config:
        write_p4app_config(args.write_config, args.p4_json)
    else:
        start_mininet_topo(
            p4_json     = args.p4_json,
            interactive = args.interactive,
            duration    = args.duration,
        )
