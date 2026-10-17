"""
experiments/run_series.py
Experiment series runner.

Real mode  : Linux namespaces + BMv2 simple_switch + real register polling
Mock mode  : synthetic CSVs for pipeline validation (no switches needed)
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from controller.utils import ensure_dirs, get_timestamp_str

logger = logging.getLogger(__name__)

P4_SRC   = os.path.join(ROOT, "p4", "adaptmon.p4")
P4_BUILD = os.path.join(ROOT, "p4", "build")
P4_JSON  = os.path.join(P4_BUILD, "adaptmon.json")
LOG_DIR  = os.path.join(ROOT, "logs")
RESULTS  = os.path.join(ROOT, "results")

# Seconds the run waits after launching the controller before continuing
CTRL_STARTUP_S = 4

EXPERIMENT_CONFIG = {
    "n_runs"         : 10,
    "run_duration"   : 120,
    "attack_start"   : 30,
    "attack_duration": 60,
    "attack_rate"    : 5000,
    "poll_interval"  : 1.0,
    "target_ip"      : "10.0.2.1",
    "switch_configs" : [
        {"switch_id": 1, "thrift_port": 9090},
        {"switch_id": 2, "thrift_port": 9091},
    ],
}


def compile_p4(p4_src: str, build_dir: str) -> bool:
    os.makedirs(build_dir, exist_ok=True)
    p4json = os.path.join(build_dir, "adaptmon.json")
    p4info = os.path.join(build_dir, "adaptmon.p4info.pb.txt")
    for cmd in [
        ["p4c", "--target", "bmv2", "--arch", "v1model",
         "--std", "p4-16", "-o", build_dir,
         "--p4runtime-files", p4info, p4_src],
        ["p4c-bm2-ss", "--std", "p4-16", "-o", p4json, p4_src],
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(p4json):
                logger.info("P4 compiled successfully")
                return True
        except FileNotFoundError:
            continue
    logger.error("P4 compilation failed")
    return False


# --------------------------------------------------------------------------
# Real experiment run
# --------------------------------------------------------------------------

class RealExperimentRun:

    def __init__(self, run_id: int, config: Dict, run_dir: str):
        self.run_id  = run_id
        self.config  = config
        self.run_dir = run_dir
        ensure_dirs([run_dir])

    def execute(self) -> Dict:
        logger.info(f"=== Real Run {self.run_id} ===")
        t0     = time.time()
        result = {
            "run_id"       : self.run_id,
            "status"       : "failed",
            "metrics_files": [],
        }

        from topo.namespace_topo import NetworkNamespaceTopo
        from traffic.ns_traffic  import NsTrafficManager

        sw_log = os.path.join(self.run_dir, "switches")
        ensure_dirs([sw_log])
        topo = NetworkNamespaceTopo(p4_json=P4_JSON, log_dir=sw_log)

        # ---- 1. Clean slate ----
        topo.teardown()

        # ---- 2. Build namespace topology ----
        try:
            topo.build()
        except Exception as e:
            logger.error(f"Topology build failed: {e}")
            topo.teardown()
            return result

        # ---- 3. Start switches ----
        if not topo.start_switches():
            logger.error("Switches failed to start")
            topo.teardown()
            return result

        # ---- 4. Wait for thrift to be ready ----
        time.sleep(3)

        # ---- 5. Install forwarding rules ----
        topo.install_forwarding_rules()

        # ---- 6. Quick connectivity check ----
        if not topo.verify_connectivity():
            logger.warning(
                "Connectivity check failed — "
                "traffic may not flow through switches"
            )

        # ---- 7. Write topo config ----
        topo_cfg = os.path.join(self.run_dir, "topo_config.json")
        topo.write_config(topo_cfg)

        # ---- 8. Start benign traffic ----
        # Started before the controller so every warm-up sample the
        # detector trains on contains normal traffic (not an idle network).
        with open(topo_cfg) as f:
            topo_info = json.load(f)

        traffic = NsTrafficManager(topo_info["host_namespaces"])
        traffic.start_normal_traffic(
            duration=self.config["run_duration"] + CTRL_STARTUP_S
        )

        # ---- 9. Start controller ----
        # The attack start is fixed as an absolute time up front and given
        # to the controller as the end of the benign warm-up: the detector
        # trains once on samples before it, then freezes.
        attack_at = time.time() + CTRL_STARTUP_S + self.config["attack_start"]
        sw_cfg = [
            {"switch_id": 1, "thrift_port": 9090},
            {"switch_id": 2, "thrift_port": 9091},
        ]
        sw_cfg_path = os.path.join(self.run_dir, "switch_configs.json")
        with open(sw_cfg_path, "w") as f:
            json.dump(sw_cfg, f)

        ctrl_log_dir = os.path.join(self.run_dir, "ctrl_logs")
        ensure_dirs([ctrl_log_dir])

        ctrl_cmd = [
            sys.executable, "-m", "controller.controller",
            "--switches",      sw_cfg_path,
            "--poll-interval", str(self.config["poll_interval"]),
            "--log-dir",       ctrl_log_dir,
            "--run-id",        str(self.run_id),
            "--duration",      str(self.config["run_duration"]),
            "--train-until",   str(attack_at),
        ]
        ctrl_stdout = os.path.join(self.run_dir, "controller_out.log")
        with open(ctrl_stdout, "w") as cf:
            ctrl_proc = subprocess.Popen(
                ctrl_cmd, stdout=cf, stderr=cf, cwd=ROOT
            )

        # Give controller time to connect to switches
        time.sleep(CTRL_STARTUP_S)

        if ctrl_proc.poll() is not None:
            logger.error(
                f"Controller exited early (code {ctrl_proc.returncode}) "
                f"— see {ctrl_stdout}"
            )
            traffic.stop_all()
            topo.stop_switches()
            topo.teardown()
            result["duration"] = time.time() - t0
            return result

        # ---- 10. Schedule attack ----
        attack_window_path = os.path.join(self.run_dir, "attack_window.json")

        def _attack():
            time.sleep(max(0.0, attack_at - time.time()))
            logger.info("ATTACK START")
            attack_start_ts = time.time()
            traffic.start_attack(
                src_host  = "h1",
                target_ip = self.config["target_ip"],
                duration  = self.config["attack_duration"],
                rate      = self.config["attack_rate"],
            )
            time.sleep(self.config["attack_duration"])
            traffic.stop_attacks()
            logger.info("ATTACK END")
            # Epoch times, comparable to the controller CSV timestamps
            with open(attack_window_path, "w") as f:
                json.dump({"start": attack_start_ts, "end": time.time()}, f)

        atk = threading.Thread(target=_attack, daemon=True)
        atk.start()

        # ---- 11. Wait ----
        logger.info(
            f"Run {self.run_id} in progress "
            f"({self.config['run_duration']}s)..."
        )
        time.sleep(self.config["run_duration"])

        # ---- 12. Teardown ----
        traffic.stop_all()
        atk.join(timeout=5)

        # The controller stops itself after --duration; give it time to
        # flush its CSVs before escalating to SIGTERM / SIGKILL.
        try:
            ctrl_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            ctrl_proc.terminate()
            try:
                ctrl_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("Controller did not exit, killing it")
                ctrl_proc.kill()
                ctrl_proc.wait()

        topo.stop_switches()
        topo.teardown()

        # ---- 13. Collect metrics ----
        for dirpath, _, files in os.walk(ctrl_log_dir):
            for fname in files:
                if fname.endswith("_metrics.csv"):
                    result["metrics_files"].append(
                        os.path.join(dirpath, fname)
                    )

        result["duration"] = time.time() - t0
        if not result["metrics_files"]:
            logger.error(
                f"Run {self.run_id} produced no metrics CSVs "
                f"— see {ctrl_stdout}"
            )
            return result
        result["status"]   = "completed"
        logger.info(
            f"=== Run {self.run_id} complete "
            f"({result['duration']:.1f}s, "
            f"{len(result['metrics_files'])} CSVs) ==="
        )
        return result


# --------------------------------------------------------------------------
# Mock experiment run
# --------------------------------------------------------------------------

class MockExperimentRun:

    def __init__(self, run_id: int, config: Dict, run_dir: str):
        self.run_id  = run_id
        self.config  = config
        self.run_dir = run_dir
        ensure_dirs([run_dir])

    def execute(self) -> Dict:
        from analysis.plots import _generate_synthetic_df

        result = {
            "run_id"       : self.run_id,
            "status"       : "completed",
            "metrics_files": [],
            "duration"     : 0.1,
        }

        for sw_id in [1, 2]:
            df = _generate_synthetic_df(
                attack_start_s = float(self.config["attack_start"]),
                attack_end_s   = float(
                    self.config["attack_start"] +
                    self.config["attack_duration"]
                ),
                duration = self.config["run_duration"],
            )
            df["switch_id"] = sw_id
            csv_path = os.path.join(
                self.run_dir, f"switch_{sw_id}_metrics.csv"
            )
            df.to_csv(csv_path, index=False)
            result["metrics_files"].append(csv_path)
            logger.info(f"  Mock data: {csv_path}")

        return result


# --------------------------------------------------------------------------
# Series orchestrator
# --------------------------------------------------------------------------

class ExperimentSeries:

    def __init__(
        self,
        config      : Dict,
        results_dir : str  = RESULTS,
        mock        : bool = False,
    ):
        self.config      = config
        self.results_dir = results_dir
        self.mock        = mock
        self.series_id   = get_timestamp_str()
        self.series_dir  = os.path.join(results_dir, self.series_id)
        ensure_dirs([self.series_dir])

    def run(self) -> Dict:
        n = self.config["n_runs"]
        mode = "MOCK" if self.mock else "REAL"
        logger.info(f"Starting experiment series: {n} runs [{mode}]")

        all_runs = []
        for i in range(1, n + 1):
            run_dir = os.path.join(self.series_dir, f"run_{i:02d}")

            if self.mock:
                run = MockExperimentRun(i, self.config, run_dir)
            else:
                run = RealExperimentRun(i, self.config, run_dir)

            result = run.execute()
            all_runs.append(result)

            if i < n:
                pause = 3 if self.mock else 10
                logger.info(f"Pausing {pause}s between runs...")
                time.sleep(pause)

        summary = self._aggregate(all_runs)
        summary_path = os.path.join(self.series_dir, "series_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(
            f"Series complete: "
            f"{summary['n_completed']}/{summary['n_total']} runs succeeded"
        )
        return summary

    def _aggregate(self, runs: List[Dict]) -> Dict:
        completed   = [r for r in runs if r["status"] == "completed"]
        all_metrics = []
        for r in completed:
            all_metrics.extend(r.get("metrics_files", []))
        return {
            "series_id"   : self.series_id,
            "n_total"     : len(runs),
            "n_completed" : len(completed),
            "n_failed"    : len(runs) - len(completed),
            "all_metrics" : all_metrics,
            "series_dir"  : self.series_dir,
            "runs"        : runs,
        }


# --------------------------------------------------------------------------
# Scalability benchmark
# --------------------------------------------------------------------------

def run_scalability_experiment(
    max_flows_list : List[int] = [100, 500, 1000, 2000, 4096],
    results_dir    : str       = RESULTS,
) -> Dict:
    from experiments.metrics import compute_overhead_vs_flows
    logger.info("Scalability experiment starting...")
    results = {}
    for n in max_flows_list:
        logger.info(f"  Testing {n} flows...")
        results[n] = compute_overhead_vs_flows(n)
    ensure_dirs([results_dir])
    out = os.path.join(results_dir, "scalability.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Scalability results: {out}")
    return results
