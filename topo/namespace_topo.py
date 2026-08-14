"""
topo/namespace_topo.py
Linux network namespace topology for AdaptMon.
Fully self-contained — creates, wires, and tears down the topology.
Must be run as root (sudo).

Topology:
  h1 (10.0.1.1) ─── s1 (BMv2, port 9090) ─── s2 (BMv2, port 9091) ─── h3 (10.0.2.1)
  h2 (10.0.1.2) ───/                                                 \── h4 (10.0.2.2)

Interface naming (all in root namespace except host-side veths):
  h1 side : am_h1     (inside ns adaptmon_h1)
  s1 port1: am_s1p1   (root ns, bound to s1 port 1)
  s1 port2: am_s1p2   (root ns, bound to s1 port 2)
  s1 port3: am_s1p3   (root ns, inter-switch to s2)
  s2 port1: am_s2p1   (root ns, bound to s2 port 1)
  s2 port2: am_s2p2   (root ns, bound to s2 port 2)
  s2 port3: am_s2p3   (root ns, inter-switch to s1)
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P4_JSON    = os.path.join(ROOT, "p4", "build", "adaptmon.json")
SIMPLE_SW  = os.environ.get("SIMPLE_SWITCH",     "simple_switch")
SIMPLE_CLI = os.environ.get("SIMPLE_SWITCH_CLI", "simple_switch_CLI")


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------

def _run(cmd: str, check: bool = False, silent: bool = True) -> Tuple[int, str, str]:
    """Run a shell command. Returns (returncode, stdout, stderr)."""
    logger.debug(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed [{r.returncode}]: {cmd}\n{r.stderr.strip()}")
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def _iface_exists(name: str) -> bool:
    rc, _, _ = _run(f"ip link show {name}")
    return rc == 0


def _ns_exists(name: str) -> bool:
    rc, out, _ = _run("ip netns list")
    return name in out


def _ns_run(ns: str, cmd: str, check: bool = False) -> Tuple[int, str, str]:
    return _run(f"ip netns exec {ns} {cmd}", check=check)


# --------------------------------------------------------------------------
# Topology definition
# --------------------------------------------------------------------------

HOSTS = [
    {
        "name"   : "h1",
        "ns"     : "adaptmon_h1",
        "ip"     : "10.0.1.1",
        "prefix" : "24",
        "mac"    : "00:00:00:01:00:01",
        "gw"     : "10.0.1.254",
        "switch" : 1,
        "sw_port": 1,
        "veth_host": "am_h1",      # inside namespace
        "veth_sw"  : "am_s1p1",    # in root namespace, s1 port 1
    },
    {
        "name"   : "h2",
        "ns"     : "adaptmon_h2",
        "ip"     : "10.0.1.2",
        "prefix" : "24",
        "mac"    : "00:00:00:01:00:02",
        "gw"     : "10.0.1.254",
        "switch" : 1,
        "sw_port": 2,
        "veth_host": "am_h2",
        "veth_sw"  : "am_s1p2",
    },
    {
        "name"   : "h3",
        "ns"     : "adaptmon_h3",
        "ip"     : "10.0.2.1",
        "prefix" : "24",
        "mac"    : "00:00:00:02:00:01",
        "gw"     : "10.0.2.254",
        "switch" : 2,
        "sw_port": 1,
        "veth_host": "am_h3",
        "veth_sw"  : "am_s2p1",
    },
    {
        "name"   : "h4",
        "ns"     : "adaptmon_h4",
        "ip"     : "10.0.2.2",
        "prefix" : "24",
        "mac"    : "00:00:00:02:00:02",
        "gw"     : "10.0.2.254",
        "switch" : 2,
        "sw_port": 2,
        "veth_host": "am_h4",
        "veth_sw"  : "am_s2p2",
    },
]

SWITCHES = [
    {"id": 1, "thrift_port": 9090},
    {"id": 2, "thrift_port": 9091},
]

# Inter-switch link: s1-port3 <--> s2-port3
INTER_SW = {
    "veth_s1": "am_s1p3",   # s1 port 3 side
    "veth_s2": "am_s2p3",   # s2 port 3 side
    "s1_port": 3,
    "s2_port": 3,
}

# Switch interface map: switch_id -> [(iface_name, port_number), ...]
SW_IFACES = {
    1: [
        ("am_s1p1", 1),
        ("am_s1p2", 2),
        ("am_s1p3", 3),
    ],
    2: [
        ("am_s2p1", 1),
        ("am_s2p2", 2),
        ("am_s2p3", 3),
    ],
}

# Forwarding rules for each switch
FWD_RULES = {
    1: [
        # Local hosts — rewrite dst MAC to actual host MAC
        "table_add AdaptMonIngress.ipv4_forward AdaptMonIngress.forward_l3 10.0.1.1/32 => 1 00:00:00:01:00:01",
        "table_add AdaptMonIngress.ipv4_forward AdaptMonIngress.forward_l3 10.0.1.2/32 => 2 00:00:00:01:00:02",
        # Remote hosts via inter-switch — rewrite dst MAC to remote host MAC
        # s2 will forward based on IP, so MAC here is what s2 will see as src
        "table_add AdaptMonIngress.ipv4_forward AdaptMonIngress.forward_l3 10.0.2.1/32 => 3 00:00:00:02:00:01",
        "table_add AdaptMonIngress.ipv4_forward AdaptMonIngress.forward_l3 10.0.2.2/32 => 3 00:00:00:02:00:02",
    ],
    2: [
        # Local hosts
        "table_add AdaptMonIngress.ipv4_forward AdaptMonIngress.forward_l3 10.0.2.1/32 => 1 00:00:00:02:00:01",
        "table_add AdaptMonIngress.ipv4_forward AdaptMonIngress.forward_l3 10.0.2.2/32 => 2 00:00:00:02:00:02",
        # Remote hosts via inter-switch
        "table_add AdaptMonIngress.ipv4_forward AdaptMonIngress.forward_l3 10.0.1.1/32 => 3 00:00:00:01:00:01",
        "table_add AdaptMonIngress.ipv4_forward AdaptMonIngress.forward_l3 10.0.1.2/32 => 3 00:00:00:01:00:02",
    ],
}


# --------------------------------------------------------------------------
# Main topology class
# --------------------------------------------------------------------------

class NetworkNamespaceTopo:

    def __init__(
        self,
        p4_json : str = P4_JSON,
        log_dir : str = "/tmp/adaptmon_sw",
    ):
        self.p4_json   = p4_json
        self.log_dir   = log_dir
        self._sw_procs : Dict[int, subprocess.Popen] = {}
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Teardown — always run first to get a clean slate
    # ------------------------------------------------------------------
    def teardown(self) -> None:
        """Remove all topology state from a previous run."""
        logger.info("Tearing down previous topology...")

        # Kill switch processes we own
        for sid, proc in list(self._sw_procs.items()):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=3)
            except Exception:
                pass
        self._sw_procs.clear()

        # Kill any leftover simple_switch processes bound to our JSON
        _run("pkill -f 'simple_switch.*adaptmon.json' 2>/dev/null || true")
        time.sleep(0.5)

        # Delete host namespaces
        for h in HOSTS:
            _run(f"ip netns del {h['ns']} 2>/dev/null || true")

        # Delete all our veth interfaces (deleting one side removes the pair)
        all_ifaces = (
            [h["veth_sw"] for h in HOSTS]
            + [INTER_SW["veth_s1"]]
        )
        for iface in all_ifaces:
            _run(f"ip link del {iface} 2>/dev/null || true")

        # Remove BMv2 IPC socket files left behind
        for sw in SWITCHES:
            ipc = f"/tmp/bmv2-{sw['id']}-notifications.ipc"
            try:
                os.remove(ipc)
                logger.debug(f"Removed IPC socket: {ipc}")
            except FileNotFoundError:
                pass

        logger.info("Teardown complete")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Create all namespaces, veth pairs, and IP assignments."""
        logger.info("Building topology...")

        # 1. Host namespaces
        for h in HOSTS:
            _run(f"ip netns add {h['ns']}", check=True)
            _ns_run(h["ns"], "ip link set lo up")
            logger.info(f"  Namespace: {h['ns']} ({h['ip']})")

        # 2. Host <-> switch veth pairs
        for h in HOSTS:
            hv  = h["veth_host"]   # goes into namespace
            sv  = h["veth_sw"]     # stays in root ns, plugged into switch
            ns  = h["ns"]
            ip  = h["ip"]
            pfx = h["prefix"]
            mac = h["mac"]

            # Create pair in root namespace
            _run(f"ip link add {hv} type veth peer name {sv}", check=True)

            # Move host side into namespace
            _run(f"ip link set {hv} netns {ns}", check=True)

            # Bring up switch side in root namespace
            _run(f"ip link set {sv} up", check=True)

            # Configure host side inside namespace
            _ns_run(ns, f"ip link set {hv} address {mac}")
            _ns_run(ns, f"ip link set {hv} up")
            _ns_run(ns, f"ip addr add {ip}/{pfx} dev {hv}")
            _ns_run(ns, f"ip link set lo up")
            # Disable reverse path filtering so cross-subnet packets are accepted
            _ns_run(ns, f"sysctl -w net.ipv4.conf.all.rp_filter=0 2>/dev/null || true")
            _ns_run(ns, f"sysctl -w net.ipv4.conf.{hv}.rp_filter=0 2>/dev/null || true")

            logger.info(f"  {h['name']}: {ip} <--> {sv} (s{h['switch']} port {h['sw_port']})")

        # 3. Inter-switch veth pair (both sides stay in root namespace)
        vs1 = INTER_SW["veth_s1"]
        vs2 = INTER_SW["veth_s2"]
        _run(f"ip link add {vs1} type veth peer name {vs2}", check=True)
        _run(f"ip link set {vs1} up", check=True)
        _run(f"ip link set {vs2} up", check=True)
        logger.info(f"  Inter-switch: {vs1} <--> {vs2}")

        # Install static ARP entries + host routes in every namespace
        # Do this in two passes: routes first, then ARP
        all_hosts_info = [(h["ip"], h["mac"], h["veth_host"]) for h in HOSTS]

        # Pass 1: add all host routes
        for h in HOSTS:
            hv = h["veth_host"]
            for ip, mac, _ in all_hosts_info:
                if ip == h["ip"]:
                    continue
                rc, out, err = _run(
                    f"ip netns exec {h['ns']} ip route add {ip}/32 dev {hv}",
                    check=False
                )
                if rc != 0 and "File exists" not in err:
                    logger.debug(f"  route {h['name']}->{ip}: {err}")

        # Pass 2: add all ARP/neighbour entries
        for h in HOSTS:
            hv = h["veth_host"]
            for ip, mac, _ in all_hosts_info:
                if ip == h["ip"]:
                    continue
                # Try replace first (works whether entry exists or not)
                rc, out, err = _run(
                    f"ip netns exec {h['ns']} "
                    f"ip neigh replace {ip} lladdr {mac} dev {hv} nud permanent",
                    check=False
                )
                if rc != 0:
                    # Fallback to add
                    _run(
                        f"ip netns exec {h['ns']} "
                        f"ip neigh add {ip} lladdr {mac} dev {hv} nud permanent",
                        check=False
                    )
                logger.debug(f"  ARP {h['name']}: {ip} -> {mac}")

            # Verify entries were set
            rc, out, _ = _run(f"ip netns exec {h['ns']} ip neigh show")
            logger.info(f"  {h['name']} ARP table: {out.strip()}")

        logger.info("Topology built successfully")

    # ------------------------------------------------------------------
    # Start switches
    # ------------------------------------------------------------------
    def start_switches(self) -> bool:
        """Start both BMv2 simple_switch processes."""
        logger.info("Starting BMv2 switches...")

        if not os.path.exists(self.p4_json):
            logger.error(f"P4 JSON not found: {self.p4_json}")
            return False

        for sw in SWITCHES:
            sid         = sw["id"]
            thrift_port = sw["thrift_port"]
            log_file    = os.path.join(self.log_dir, f"switch_{sid}.log")
            ifaces      = SW_IFACES[sid]

            cmd = [
                SIMPLE_SW,
                "--thrift-port", str(thrift_port),
                "--log-level",   "warn",
                "--no-p4",       # start without P4 then load JSON
            ]
            # Actually load P4 directly:
            # Each switch needs a unique device-id and IPC socket path
            # to avoid "Address already in use" on bmv2-notifications.ipc
            ipc_path = f"/tmp/bmv2-{sid}-notifications.ipc"
            # Remove stale socket file if present
            try:
                import os as _os
                _os.remove(ipc_path)
            except FileNotFoundError:
                pass

            cmd = [
                SIMPLE_SW,
                "--thrift-port",        str(thrift_port),
                "--device-id",          str(sid),
                "--notifications-addr", f"ipc://{ipc_path}",
                "--log-level",          "warn",
            ]
            for iface, port_num in ifaces:
                cmd += ["-i", f"{port_num}@{iface}"]
            cmd.append(self.p4_json)

            logger.info(f"  s{sid}: {' '.join(cmd)}")

            try:
                with open(log_file, "w") as lf:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=lf,
                        stderr=lf,
                        preexec_fn=os.setsid,
                    )
                self._sw_procs[sid] = proc
            except FileNotFoundError:
                logger.error("simple_switch not found. Install BMv2.")
                return False
            except Exception as e:
                logger.error(f"Switch {sid} failed to start: {e}")
                return False

            # Give this switch 2s to initialize before starting the next
            time.sleep(2)

            # Check it's still alive
            if proc.poll() is not None:
                logger.error(
                    f"Switch {sid} exited immediately "
                    f"(code={proc.returncode}). Check {log_file}"
                )
                try:
                    with open(log_file) as lf:
                        tail = lf.readlines()[-15:]
                    for line in tail:
                        logger.error(f"  LOG: {line.rstrip()}")
                except Exception:
                    pass
                return False

            logger.info(f"  s{sid}: running (pid={proc.pid}, thrift={thrift_port})")

        return True

    # ------------------------------------------------------------------
    # Install forwarding rules
    # ------------------------------------------------------------------
    def install_forwarding_rules(self) -> bool:
        """Install IPv4 LPM forwarding rules on both switches."""
        logger.info("Installing forwarding rules...")
        all_ok = True

        for sw in SWITCHES:
            sid         = sw["id"]
            thrift_port = sw["thrift_port"]
            rules       = FWD_RULES[sid]
            cmds        = "\n".join(rules) + "\nexit\n"

            try:
                result = subprocess.run(
                    [SIMPLE_CLI, "--thrift-port", str(thrift_port)],
                    input   = cmds,
                    capture_output = True,
                    text    = True,
                    timeout = 15,
                )
                if result.returncode != 0 or "error" in result.stderr.lower():
                    logger.warning(
                        f"  s{sid} rule install warning: {result.stderr[:200]}"
                    )
                    all_ok = False
                else:
                    logger.info(f"  s{sid}: {len(rules)} rules installed")
            except subprocess.TimeoutExpired:
                logger.error(f"  s{sid}: CLI timeout")
                all_ok = False
            except FileNotFoundError:
                logger.error("simple_switch_CLI not found")
                all_ok = False

        return all_ok

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------
    def verify_connectivity(self) -> bool:
        """Ping between namespaces to confirm the topology is working."""
        logger.info("Verifying connectivity...")
        tests = [
            ("adaptmon_h1", "10.0.1.2", "h1->h2 (same switch)"),
            ("adaptmon_h1", "10.0.2.1", "h1->h3 (cross switch)"),
            ("adaptmon_h3", "10.0.1.1", "h3->h1 (cross switch)"),
        ]
        all_ok = True
        for ns, dst, label in tests:
            rc, _, _ = _run(
                f"ip netns exec {ns} ping -c 3 -W 2 {dst}"
            )
            ok = rc == 0
            logger.info(f"  {label}: {'OK' if ok else 'FAIL'}")
            if not ok:
                all_ok = False
        return all_ok

    # ------------------------------------------------------------------
    # Run command inside a host namespace
    # ------------------------------------------------------------------
    def run_in_host(
        self,
        host_name : str,
        cmd       : str,
        background: bool = False,
    ) -> Optional[subprocess.Popen]:
        h = next((h for h in HOSTS if h["name"] == host_name), None)
        if not h:
            logger.error(f"Unknown host: {host_name}")
            return None
        full = f"ip netns exec {h['ns']} {cmd}"
        if background:
            return subprocess.Popen(
                full, shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        _run(full)
        return None

    # ------------------------------------------------------------------
    # Teardown switches only
    # ------------------------------------------------------------------
    def stop_switches(self) -> None:
        for sid, proc in list(self._sw_procs.items()):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            logger.info(f"  s{sid} stopped")
        self._sw_procs.clear()

    # ------------------------------------------------------------------
    # Write config for controller
    # ------------------------------------------------------------------
    def write_config(self, path: str) -> None:
        config = {
            "switch_configs": [
                {"switch_id": sw["id"], "thrift_port": sw["thrift_port"]}
                for sw in SWITCHES
            ],
            "host_namespaces": {
                h["name"]: {
                    "ns" : h["ns"],
                    "ip" : h["ip"],
                }
                for h in HOSTS
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Topology config: {path}")

    def get_host_info(self) -> Dict:
        return {
            h["name"]: {"ns": h["ns"], "ip": h["ip"]}
            for h in HOSTS
        }
