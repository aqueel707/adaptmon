"""
controller/dataplane.py
Abstraction layer for BMv2 simple_switch_CLI interactions.
Handles register reads, table entry installation, and switch management.
"""

import subprocess
import re
import time
import logging
import threading
import os
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

MAX_FLOWS = 4096
THRIFT_BASE_PORT = 9090


class SwitchInterface:
    """
    Interface to a single BMv2 simple_switch instance via simple_switch_CLI.
    """

    def __init__(self, switch_id: int, thrift_port: int):
        self.switch_id   = switch_id
        self.thrift_port = thrift_port
        self._lock       = threading.Lock()
        self._cli_path   = self._find_cli()

    def _find_cli(self) -> str:
        candidates = [
            "simple_switch_CLI",
            "/usr/local/bin/simple_switch_CLI",
            "/usr/bin/simple_switch_CLI",
        ]
        for c in candidates:
            try:
                result = subprocess.run(
                    [c, "--help"],
                    capture_output=True, timeout=5
                )
                return c
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return "simple_switch_CLI"

    def _cli(self, commands: str, timeout: int = 15) -> str:
        """Send commands to simple_switch_CLI and return output."""
        with self._lock:
            try:
                proc = subprocess.run(
                    [self._cli_path,
                     "--thrift-port", str(self.thrift_port)],
                    input=commands,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                return proc.stdout + proc.stderr
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"CLI timeout on switch {self.switch_id} "
                    f"port {self.thrift_port}"
                )
                return ""
            except Exception as e:
                logger.error(f"CLI error: {e}")
                return ""

    def read_register(self, reg_name: str, index: int) -> Optional[int]:
        """Read a single register cell."""
        cmd = f"register_read {reg_name} {index}\n"
        out = self._cli(cmd)
        # Parse: "AdaptMonIngress.pkt_counter[42]= 17"
        pattern = rf"{reg_name}\[{index}\]\s*=\s*(\d+)"
        m = re.search(pattern, out)
        if m:
            return int(m.group(1))
        # Fallback: last number
        nums = re.findall(r"\d+", out.split("=")[-1]) if "=" in out else []
        if nums:
            return int(nums[-1])
        return None

    def read_register_range(
        self,
        reg_name: str,
        start: int,
        end: int
    ) -> Dict[int, int]:
        """Read a range of register cells efficiently."""
        results = {}
        # BMv2 CLI doesn't support bulk reads natively;
        # we batch commands for efficiency
        batch_size = 128
        for batch_start in range(start, end, batch_size):
            batch_end = min(batch_start + batch_size, end)
            cmds = "\n".join(
                f"register_read {reg_name} {i}"
                for i in range(batch_start, batch_end)
            ) + "\n"
            out = self._cli(cmds, timeout=30)
            for i in range(batch_start, batch_end):
                pattern = rf"{reg_name}\[{i}\]\s*=\s*(\d+)"
                m = re.search(pattern, out)
                if m:
                    results[i] = int(m.group(1))
                else:
                    results[i] = 0
        return results

    def read_all_pkt_counters(self) -> Dict[int, int]:
        return self.read_register_range("pkt_counter", 0, MAX_FLOWS)

    def read_all_byte_counters(self) -> Dict[int, int]:
        return self.read_register_range("byte_counter", 0, MAX_FLOWS)

    def read_flow_active(self) -> Dict[int, int]:
        return self.read_register_range("flow_active", 0, MAX_FLOWS)

    def read_flow_src_ip(self, index: int) -> Optional[int]:
        return self.read_register("flow_src_ip", index)

    def read_flow_dst_ip(self, index: int) -> Optional[int]:
        return self.read_register("flow_dst_ip", index)

    def read_flow_protocol(self, index: int) -> Optional[int]:
        return self.read_register("flow_protocol", index)

    def reset_register(self, reg_name: str, index: int) -> None:
        cmd = f"register_write {reg_name} {index} 0\n"
        self._cli(cmd)

    def reset_all_counters(self) -> None:
        for reg in ["pkt_counter", "byte_counter"]:
            cmd = f"register_reset {reg}\n"
            self._cli(cmd)

    def add_drop_rule(
        self,
        src_ip: str,
        dst_ip: str,
        protocol: int
    ) -> bool:
        """Install a drop rule in the drop_table."""
        cmd = (
            f"table_add AdaptMonIngress.drop_table "
            f"AdaptMonIngress.do_drop_flow "
            f"{src_ip} {dst_ip} {protocol} => \n"
        )
        out = self._cli(cmd)
        return "Entry has been added" in out or "handle" in out.lower()

    def delete_drop_rule(
        self,
        src_ip: str,
        dst_ip: str,
        protocol: int
    ) -> bool:
        """Delete a drop rule."""
        # First find the handle
        cmd = "table_dump AdaptMonIngress.drop_table\n"
        out = self._cli(cmd)
        # Look for handle matching this entry
        pattern = (
            rf"Dumping entry.*?handle\s+(\d+).*?"
            rf"{re.escape(src_ip)}.*?"
            rf"{re.escape(dst_ip)}"
        )
        m = re.search(pattern, out, re.DOTALL)
        if m:
            handle = m.group(1)
            del_cmd = (
                f"table_delete AdaptMonIngress.drop_table {handle}\n"
            )
            del_out = self._cli(del_cmd)
            return "Deleted" in del_out or "deleted" in del_out
        return False

    def add_clone_rule(self, src_ip: str, dst_ip: str) -> bool:
        """Install a clone rule in clone_table."""
        cmd = (
            f"table_add AdaptMonIngress.clone_table "
            f"AdaptMonIngress.do_clone_to_cpu "
            f"{src_ip} {dst_ip} => \n"
        )
        out = self._cli(cmd)
        return "Entry has been added" in out or "handle" in out.lower()

    def add_forward_rule(self, dst_ip: str, port: int) -> bool:
        """Install an IPv4 LPM forwarding rule."""
        cmd = (
            f"table_add AdaptMonIngress.ipv4_forward "
            f"AdaptMonIngress.forward_l2 "
            f"{dst_ip}/32 => {port}\n"
        )
        out = self._cli(cmd)
        return "Entry has been added" in out or "handle" in out.lower()

    def add_default_forward(self, port: int) -> bool:
        cmd = (
            f"table_set_default AdaptMonIngress.ipv4_forward "
            f"AdaptMonIngress.forward_l2 {port}\n"
        )
        out = self._cli(cmd)
        return True

    def set_mirroring_session(
        self,
        session_id: int,
        egress_port: int
    ) -> bool:
        cmd = (
            f"mirroring_add {session_id} {egress_port}\n"
        )
        out = self._cli(cmd)
        return True

    def is_alive(self) -> bool:
        """Check if the switch is responding."""
        out = self._cli("show_tables\n", timeout=5)
        return len(out) > 10


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
            all_alive = all(
                sw.is_alive() for sw in self.switches.values()
            )
            if all_alive:
                logger.info("All switches are alive")
                return True
            time.sleep(2)
        logger.error("Timed out waiting for switches")
        return False

    def install_base_forwarding(
        self,
        forwarding_rules: List[Dict]
    ) -> None:
        """
        Install base L3 forwarding rules.
        forwarding_rules: list of dicts:
            {switch_id, dst_ip, port}
        """
        for rule in forwarding_rules:
            sw = self.get_switch(rule["switch_id"])
            if sw:
                sw.add_forward_rule(rule["dst_ip"], rule["port"])
                logger.debug(
                    f"S{rule['switch_id']}: forward "
                    f"{rule['dst_ip']} -> port {rule['port']}"
                )

    def install_mirroring(
        self,
        switch_id: int,
        session_id: int,
        cpu_port: int
    ) -> None:
        sw = self.get_switch(switch_id)
        if sw:
            sw.set_mirroring_session(session_id, cpu_port)

    def collect_all_counters(
        self,
        switch_id: int
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Return (pkt_counters, byte_counters) for a switch."""
        sw = self.get_switch(switch_id)
        if sw is None:
            return {}, {}
        pkts  = sw.read_all_pkt_counters()
        bytes_ = sw.read_all_byte_counters()
        return pkts, bytes_

    def install_drop_rule(
        self,
        switch_id: int,
        src_ip: str,
        dst_ip: str,
        protocol: int = 6
    ) -> bool:
        sw = self.get_switch(switch_id)
        if sw:
            return sw.add_drop_rule(src_ip, dst_ip, protocol)
        return False

    def install_clone_rule(
        self,
        switch_id: int,
        src_ip: str,
        dst_ip: str
    ) -> bool:
        sw = self.get_switch(switch_id)
        if sw:
            return sw.add_clone_rule(src_ip, dst_ip)
        return False
