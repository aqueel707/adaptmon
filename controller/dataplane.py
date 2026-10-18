"""
controller/dataplane.py
Abstraction layer for BMv2 simple_switch_CLI interactions.
Handles register reads, table entry installation, and switch management.

Control channel: every command is a simple_switch_CLI command
(register_read / register_write / table_add / table_delete / mirroring_add /
counter_read) sent over bmv2's Thrift runtime API, NOT P4Runtime. By default
the commands run through simple_switch_CLI's own command interpreter
(sswitch_CLI.SimpleSwitchAPI) loaded in-process, which keeps one Thrift
connection open per switch instead of spawning a CLI process per call.
ADAPTMON_CLI=subprocess pipes them into a simple_switch_CLI process instead.
"""

import contextlib
import io
import os
import re
import subprocess
import sys
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_FLOWS = 4096
CPU_PORT = 255
CLONE_SESSION = 100

# Per-slot flow state written by the P4 ingress for every IPv4 packet
FLOW_REGISTERS = [
    "pkt_counter", "byte_counter",
    "flow_src_ip", "flow_dst_ip", "flow_src_port", "flow_dst_port",
    "flow_protocol", "flow_first_ts", "flow_last_ts", "flow_level",
    "flow_syn_pkts", "drop_pkts",
]
# Zeroed per slot on expiry. pkt_counter == 0 makes the next packet that
# hashes to the slot restart flow_first_ts; the 5-tuple registers are
# overwritten by that packet. flow_level is reset so a new flow does not
# inherit the old one's level.
SLOT_CLEAR_REGISTERS = ["pkt_counter", "byte_counter", "flow_active", "flow_level",
                        "flow_syn_pkts", "drop_pkts"]
DP_COUNTERS = {
    "forwarded": "AdaptMonEgress.fwd_counter",
    "cloned"   : "AdaptMonEgress.clone_counter",
    "dropped"  : "AdaptMonEgress.drop_counter",
}
CLONE_REASON_COUNTER = "AdaptMonEgress.clone_reason_counter"
# p4/adaptmon.p4 clone_reason values
CLONE_REASONS = {1: "level", 2: "sample", 3: "hostpair", 4: "head"}
THRIFT_BASE_PORT = 9090

# simple_switch_CLI's Python modules (installed by bmv2 next to the wrapper)
CLI_LIB_DIRS = [
    os.environ.get("BMV2_CLI_LIB", ""),
    "/usr/local/lib/python3.10/site-packages",
    "/usr/lib/python3/dist-packages",
]


def _import_cli_modules():
    for d in CLI_LIB_DIRS:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)
    import runtime_CLI   # noqa: E402
    import sswitch_CLI   # noqa: E402
    return runtime_CLI, sswitch_CLI


class SwitchInterface:
    """
    Interface to a single BMv2 simple_switch instance via simple_switch_CLI
    commands (Thrift).
    """

    def __init__(self, switch_id: int, thrift_port: int,
                 thrift_ip: str = "127.0.0.1"):
        self.switch_id   = switch_id
        self.thrift_port = thrift_port
        self.thrift_ip   = thrift_ip
        self._lock       = threading.Lock()
        self._mode       = os.environ.get("ADAPTMON_CLI", "inprocess")
        self._api        = None
        self._cli_path   = "simple_switch_CLI"

    # ------------------------------------------------------------------
    # Command transport
    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        runtime_CLI, sswitch_CLI = _import_cli_modules()
        pre = runtime_CLI.PreType.SimplePreLAG
        services = runtime_CLI.RuntimeAPI.get_thrift_services(pre)
        services.extend(sswitch_CLI.SimpleSwitchAPI.get_thrift_services())
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                std, mc, ss = runtime_CLI.thrift_connect(
                    self.thrift_ip, self.thrift_port, services
                )
                # Fetches the P4 JSON from the switch (both switches run the
                # same program, so runtime_CLI's module-level tables agree)
                runtime_CLI.load_json_config(std, None)
        except (SystemExit, Exception) as e:
            logger.debug(f"S{self.switch_id} thrift connect failed: {e!r} {buf.getvalue()[-200:]}")
            return False
        self._api = sswitch_CLI.SimpleSwitchAPI(pre, std, mc, ss)
        return True

    def _cli(self, commands: str, timeout: int = 15) -> str:
        """Run newline-separated simple_switch_CLI commands; return output."""
        with self._lock:
            if self._mode == "subprocess":
                return self._cli_subprocess(commands, timeout)
            if self._api is None and not self._connect():
                return "Could not connect to thrift client"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                for line in commands.splitlines():
                    if line.strip():
                        try:
                            self._api.onecmd(line)
                        except Exception as e:
                            # Connection lost (switch restarted): reconnect
                            # on the next call
                            print(f"Error: {e!r}")
                            self._api = None
                            break
            return buf.getvalue()

    def _cli_subprocess(self, commands: str, timeout: int) -> str:
        try:
            proc = subprocess.run(
                [self._cli_path, "--thrift-port", str(self.thrift_port)],
                input=commands, capture_output=True, text=True, timeout=timeout
            )
            return proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            logger.warning(f"CLI timeout on switch {self.switch_id}")
            return ""

    # ------------------------------------------------------------------
    # Registers
    # ------------------------------------------------------------------
    def read_register(self, reg_name: str, index: int) -> Optional[int]:
        """Read a single register cell."""
        out = self._cli(f"register_read {reg_name} {index}\n")
        # Parse: "pkt_counter[42]= 17"
        m = re.search(rf"{reg_name}\[{index}\]\s*=\s*(\d+)", out)
        return int(m.group(1)) if m else None

    def read_registers(self, reg_names: List[str]) -> Dict[str, List[int]]:
        """
        Read whole register arrays (`register_read <name>` without an index
        dumps the array: "pkt_counter= 0, 3, ...").
        Missing/unparsable arrays come back empty.
        """
        cmds = "".join(f"register_read {r}\n" for r in reg_names)
        out = self._cli(cmds, timeout=30)
        results = {}
        for reg_name in reg_names:
            m = re.search(rf"{reg_name}\s*=\s*([\d,\s]*\d)", out)
            results[reg_name] = [int(v) for v in m.group(1).split(",")] if m else []
        return results

    def write_registers(self, writes: List[Tuple[str, int, int]]) -> str:
        """Batch of (register, index, value) writes in one call."""
        if not writes:
            return ""
        cmds = "".join(f"register_write {r} {i} {v}\n" for r, i, v in writes)
        return self._cli(cmds, timeout=60)

    def write_flow_levels(self, levels: Dict[int, int]) -> None:
        """flow_level[idx] = level for each slot (register_write, Thrift)."""
        self.write_registers([("flow_level", i, v) for i, v in levels.items()])

    def clear_flow_entries(self, indices: List[int]) -> None:
        """Zero the counters, active bit and level at the given slots."""
        self.write_registers([
            (reg, i, 0) for i in indices for reg in SLOT_CLEAR_REGISTERS
        ])

    def reset_flow_registers(self) -> None:
        """Zero every slot of every flow register (one call)."""
        self._cli("".join(f"register_reset {reg}\n" for reg in FLOW_REGISTERS))

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    def read_dp_counters(self) -> Dict[str, int]:
        """Global forwarded / cloned / dropped packet counters."""
        cmds = "".join(f"counter_read {c} 0\n" for c in DP_COUNTERS.values())
        out = self._cli(cmds)
        res = {}
        for key, cname in DP_COUNTERS.items():
            # "AdaptMonEgress.fwd_counter[0]= (1234 bytes, 17 packets)"
            m = re.search(rf"{re.escape(cname)}\[0\]=\s*\(\d+ bytes, (\d+) packets\)", out)
            res[key] = int(m.group(1)) if m else -1
        return res

    def read_dp_counters_full(self) -> Dict[str, int]:
        """
        Cumulative (since the last reset) packets AND bytes of every
        data-plane counter, plus clones per clone_reason:
        forwarded/_bytes, cloned/_bytes, dropped/_bytes,
        clone_<level|sample|hostpair|head>/_bytes. -1 = unreadable.
        """
        cmds = "".join(f"counter_read {c} 0\n" for c in DP_COUNTERS.values())
        cmds += "".join(f"counter_read {CLONE_REASON_COUNTER} {i}\n" for i in CLONE_REASONS)
        out = self._cli(cmds)
        res = {}
        for key, cname in DP_COUNTERS.items():
            m = re.search(re.escape(cname) + r"\[0\]=\s*\((\d+) bytes, (\d+) packets\)", out)
            res[key], res[f"{key}_bytes"] = (int(m.group(2)), int(m.group(1))) if m else (-1, -1)
        for i, name in CLONE_REASONS.items():
            m = re.search(re.escape(CLONE_REASON_COUNTER) + rf"\[{i}\]=\s*\((\d+) bytes, (\d+) packets\)", out)
            res[f"clone_{name}"], res[f"clone_{name}_bytes"] = \
                (int(m.group(2)), int(m.group(1))) if m else (-1, -1)
        return res

    def reset_dp_counters(self) -> None:
        self._cli("".join(f"counter_reset {c}\n"
                          for c in list(DP_COUNTERS.values()) + [CLONE_REASON_COUNTER]))

    # ------------------------------------------------------------------
    # Flow-head cloning
    # ------------------------------------------------------------------
    def set_head_clone_n(self, n: int) -> Optional[int]:
        """Clone the first n packets of every new flow (0 = off); returns
        the value read back from the switch, None if unreadable."""
        self._cli(f"register_write head_clone_n 0 {int(n)}\n")
        return self.read_register("head_clone_n", 0)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def add_drop_rule(self, src_ip: str, dst_ip: str, protocol: int,
                      sport: int, dport: int) -> Optional[int]:
        """Install a 5-tuple drop rule; return its entry handle."""
        out = self._cli(
            f"table_add AdaptMonIngress.drop_table AdaptMonIngress.do_drop_flow "
            f"{src_ip} {dst_ip} {protocol} {sport} {dport} => \n"
        )
        m = re.search(r"added with handle (\d+)", out)
        if not m:
            logger.warning(f"S{self.switch_id} drop rule not added: {out.strip()[-200:]}")
        return int(m.group(1)) if m else None

    def delete_drop_rule(self, handle: int) -> bool:
        out = self._cli(f"table_delete AdaptMonIngress.drop_table {handle}\n")
        return "Error" not in out and "Invalid" not in out

    def count_drop_rules(self) -> int:
        out = self._cli("table_num_entries AdaptMonIngress.drop_table\n")
        m = re.search(r"(\d+)\s*$", out.strip())
        return int(m.group(1)) if m else -1

    # ------------------------------------------------------------------
    # Host-pair clone rule (Decision B): forces full capture of every
    # packet from one source IP once the host-pair detector escalates it
    # ------------------------------------------------------------------
    def add_host_clone_rule(self, src_ip: str) -> Optional[int]:
        out = self._cli(
            f"table_add AdaptMonIngress.host_clone_table "
            f"AdaptMonIngress.do_force_clone {src_ip} => \n"
        )
        m = re.search(r"added with handle (\d+)", out)
        if not m:
            logger.warning(f"S{self.switch_id} host-clone rule not added for "
                           f"{src_ip}: {out.strip()[-200:]}")
        return int(m.group(1)) if m else None

    def delete_host_clone_rule(self, handle: int) -> bool:
        out = self._cli(f"table_delete AdaptMonIngress.host_clone_table {handle}\n")
        return "Error" not in out and "Invalid" not in out

    def count_host_clone_rules(self) -> int:
        out = self._cli("table_num_entries AdaptMonIngress.host_clone_table\n")
        m = re.search(r"(\d+)\s*$", out.strip())
        return int(m.group(1)) if m else -1

    def add_host_drop_rule(self, src_ip: str) -> Optional[int]:
        """Drop (and clone) every packet from src_ip: a DPI-confirmed scan source."""
        out = self._cli(
            f"table_add AdaptMonIngress.host_drop_table "
            f"AdaptMonIngress.do_host_drop {src_ip} => \n"
        )
        m = re.search(r"added with handle (\d+)", out)
        if not m:
            logger.warning(f"S{self.switch_id} host-drop rule not added for "
                           f"{src_ip}: {out.strip()[-200:]}")
        return int(m.group(1)) if m else None

    def delete_host_drop_rule(self, handle: int) -> bool:
        out = self._cli(f"table_delete AdaptMonIngress.host_drop_table {handle}\n")
        return "Error" not in out and "Invalid" not in out

    def count_host_drop_rules(self) -> int:
        out = self._cli("table_num_entries AdaptMonIngress.host_drop_table\n")
        m = re.search(r"(\d+)\s*$", out.strip())
        return int(m.group(1)) if m else -1

    def add_forward_rule(self, dst_ip: str, port: int) -> bool:
        """Install an IPv4 LPM forwarding rule."""
        out = self._cli(
            f"table_add AdaptMonIngress.ipv4_forward AdaptMonIngress.forward_l2 "
            f"{dst_ip}/32 => {port}\n"
        )
        return "added with handle" in out

    def run_commands(self, commands: List[str]) -> str:
        return self._cli("\n".join(commands) + "\n")

    # ------------------------------------------------------------------
    # Mirroring
    # ------------------------------------------------------------------
    def set_mirroring_session(self, session_id: int, egress_port: int) -> bool:
        """mirroring_add <session> <port>, then confirm with mirroring_get."""
        self._cli(f"mirroring_delete {session_id}\n")
        self._cli(f"mirroring_add {session_id} {egress_port}\n")
        out = self._cli(f"mirroring_get {session_id}\n")
        # Prints the thrift struct: MirroringSessionConfig(port=255, mgid=None)
        return f"port={egress_port}," in out.replace(" ", "")

    def is_alive(self) -> bool:
        """Check if the switch is responding."""
        out = self._cli("show_tables\n", timeout=5)
        return "drop_table" in out


class DataPlaneManager:
    """
    Manages multiple SwitchInterface instances.
    """

    def __init__(self, switch_configs: List[Dict]):
        """
        switch_configs: list of dicts with keys:
            switch_id, thrift_port
        """
        self.switches: Dict[int, SwitchInterface] = {}
        for cfg in switch_configs:
            sid  = cfg["switch_id"]
            port = cfg.get("thrift_port", THRIFT_BASE_PORT + sid)
            self.switches[sid] = SwitchInterface(sid, port)
        logger.info(
            f"DataPlaneManager initialized with "
            f"{len(self.switches)} switches"
        )

    def get_switch(self, switch_id: int) -> Optional[SwitchInterface]:
        return self.switches.get(switch_id)

    def all_switches(self) -> List[SwitchInterface]:
        return list(self.switches.values())

    def wait_for_switches(self, timeout: int = 60) -> bool:
        """Wait until all switches are alive."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(sw.is_alive() for sw in self.switches.values()):
                logger.info("All switches are alive")
                return True
            time.sleep(1)
        logger.error("Timed out waiting for switches")
        return False

    def install_base_forwarding(self, forwarding_rules: List[Dict]) -> None:
        """
        Install base L3 forwarding rules.
        forwarding_rules: list of dicts:
            {switch_id, dst_ip, port}
        """
        for rule in forwarding_rules:
            sw = self.get_switch(rule["switch_id"])
            if sw:
                sw.add_forward_rule(rule["dst_ip"], rule["port"])

    def install_mirroring(
        self,
        switch_id : int,
        session_id: int = CLONE_SESSION,
        cpu_port  : int = CPU_PORT,
    ) -> bool:
        sw = self.get_switch(switch_id)
        return sw.set_mirroring_session(session_id, cpu_port) if sw else False

    def install_mirroring_all(self) -> Dict[int, bool]:
        """Configure clone session 100 -> CPU port on every switch."""
        return {sid: self.install_mirroring(sid) for sid in self.switches}
