"""
analysis/overhead.py
Controller CPU/memory overhead analysis for AdaptMon.
Measures overhead of register polling, feature extraction, and model inference.
"""

import os
import sys
import time
import logging
import threading
import numpy as np
import pandas as pd
import psutil
from typing import Dict, List, Optional, Callable, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def measure_overhead(label: str = "operation"):
    """Context manager to measure CPU and memory overhead."""
    proc = psutil.Process(os.getpid())
    cpu_before = proc.cpu_percent(interval=0.1)
    mem_before = proc.memory_info().rss / 1024 / 1024
    t_start = time.perf_counter()

    yield

    elapsed = time.perf_counter() - t_start
    cpu_after = proc.cpu_percent(interval=0.1)
    mem_after = proc.memory_info().rss / 1024 / 1024

    logger.debug(
        f"[{label}] time={elapsed*1000:.2f}ms "
        f"cpu_delta={cpu_after-cpu_before:.1f}% "
        f"mem_delta={mem_after-mem_before:.2f}MB"
    )


class OverheadProfiler:
    """
    Profiles the overhead of each controller component
    across multiple iterations.
    """

    def __init__(self, window: int = 100):
        self.window    = window
        self._records  : Dict[str, List[Dict]] = {}
        self._proc     = psutil.Process(os.getpid())
        self._lock     = threading.Lock()

    def record(
        self,
        component     : str,
        elapsed_ms    : float,
        cpu_pct       : float = 0.0,
        mem_mb        : float = 0.0,
        extra         : Dict  = None,
    ) -> None:
        with self._lock:
            if component not in self._records:
                self._records[component] = []
            self._records[component].append({
                "elapsed_ms" : elapsed_ms,
                "cpu_pct"    : cpu_pct,
                "mem_mb"     : mem_mb,
                "timestamp"  : time.time(),
                **(extra or {}),
            })
            # Keep rolling window
            if len(self._records[component]) > self.window:
                self._records[component].pop(0)

    def time_and_record(
        self,
        component : str,
        fn        : Callable,
        *args,
        **kwargs,
    ):
        """Call fn, record timing, return result."""
        cpu_before = self._proc.cpu_percent(interval=None)
        mem_before = self._proc.memory_info().rss / 1024 / 1024
        t_start = time.perf_counter()

        result = fn(*args, **kwargs)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        cpu_after  = self._proc.cpu_percent(interval=None)
        mem_after  = self._proc.memory_info().rss / 1024 / 1024

        self.record(
            component,
            elapsed_ms,
            cpu_pct = cpu_after - cpu_before,
            mem_mb  = mem_after - mem_before,
        )
        return result

    def get_stats(self, component: str) -> Dict:
        """Return statistics for a component."""
        records = self._records.get(component, [])
        if not records:
            return {}

        times = np.array([r["elapsed_ms"] for r in records])
        cpus  = np.array([r["cpu_pct"] for r in records])
        mems  = np.array([r["mem_mb"] for r in records])

        return {
            "component"    : component,
            "n_samples"    : len(records),
            "time_mean_ms" : float(times.mean()),
            "time_std_ms"  : float(times.std()),
            "time_p50_ms"  : float(np.percentile(times, 50)),
            "time_p95_ms"  : float(np.percentile(times, 95)),
            "time_p99_ms"  : float(np.percentile(times, 99)),
            "time_max_ms"  : float(times.max()),
            "cpu_mean_pct" : float(cpus.mean()),
            "cpu_max_pct"  : float(cpus.max()),
            "mem_mean_mb"  : float(mems.mean()),
            "mem_max_mb"   : float(mems.max()),
        }

    def get_all_stats(self) -> Dict[str, Dict]:
        """Return statistics for all tracked components."""
        return {
            component: self.get_stats(component)
            for component in self._records
        }

    def get_summary_df(self) -> pd.DataFrame:
        """Return all stats as a DataFrame."""
        rows = []
        for component, stats in self.get_all_stats().items():
            rows.append(stats)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def save_csv(self, path: str) -> None:
        """Save all records to CSV."""
        all_rows = []
        for component, records in self._records.items():
            for r in records:
                row = {"component": component, **r}
                all_rows.append(row)
        if all_rows:
            df = pd.DataFrame(all_rows)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"Overhead records saved to {path}")


def benchmark_register_poll(
    n_flows_list  : List[int] = None,
    n_iterations  : int       = 50,
) -> Dict[int, Dict]:
    """
    Benchmark the register polling phase for varying flow counts.
    Uses a simulated polling loop (no real switch needed).
    """
    if n_flows_list is None:
        n_flows_list = [100, 500, 1000, 2000, 4096]

    results = {}

    for n_flows in n_flows_list:
        times_ms = []
        for _ in range(n_iterations):
            # Simulate reading n_flows register values
            t_start = time.perf_counter()
            fake_data = {i: int(np.random.randint(0, 10000))
                         for i in range(n_flows)}
            _ = sum(fake_data.values())   # force evaluation
            elapsed = (time.perf_counter() - t_start) * 1000
            times_ms.append(elapsed)

        results[n_flows] = {
            "n_flows"        : n_flows,
            "mean_ms"        : float(np.mean(times_ms)),
            "std_ms"         : float(np.std(times_ms)),
            "p50_ms"         : float(np.percentile(times_ms, 50)),
            "p95_ms"         : float(np.percentile(times_ms, 95)),
            "max_ms"         : float(np.max(times_ms)),
        }
        logger.info(
            f"n_flows={n_flows}: "
            f"mean={results[n_flows]['mean_ms']:.2f}ms "
            f"p95={results[n_flows]['p95_ms']:.2f}ms"
        )

    return results


def benchmark_feature_extraction(
    n_flows_list : List[int] = None,
    n_iterations : int       = 50,
) -> Dict[int, Dict]:
    """Benchmark feature extraction for varying flow counts."""
    if n_flows_list is None:
        n_flows_list = [100, 500, 1000, 2000, 4096]

    results = {}

    for n_flows in n_flows_list:
        times_ms = []
        np.random.seed(42)

        for _ in range(n_iterations):
            # Simulate feature extraction
            pkt_counts = {i: np.random.randint(0, 1000) for i in range(n_flows)}
            counts_arr = np.array(list(pkt_counts.values()), dtype=np.float64)

            t_start = time.perf_counter()

            total = counts_arr.sum()
            if total > 0:
                probs   = counts_arr / total
                entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
                gini    = _compute_gini(counts_arr)
                mean_c  = float(counts_arr.mean())
                std_c   = float(counts_arr.std())
                max_c   = float(counts_arr.max())
                top_f   = max_c / total

            fv = np.array([
                entropy, float(n_flows), mean_c, mean_c * 64,
                max_c, max_c * 64, std_c, std_c * 64, gini, top_f
            ])

            elapsed = (time.perf_counter() - t_start) * 1000
            times_ms.append(elapsed)

        results[n_flows] = {
            "n_flows" : n_flows,
            "mean_ms" : float(np.mean(times_ms)),
            "std_ms"  : float(np.std(times_ms)),
            "p95_ms"  : float(np.percentile(times_ms, 95)),
        }

    return results


def benchmark_model_inference(
    n_iterations : int = 200,
) -> Dict:
    """Benchmark IsolationForest inference speed."""
    from sklearn.ensemble      import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline      import Pipeline

    np.random.seed(42)
    n_train = 100
    n_feat  = 10

    X_train = np.random.randn(n_train, n_feat)
    pipeline = Pipeline([
        ("scaler",  StandardScaler()),
        ("iforest", IsolationForest(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X_train)

    times_ms = []
    for _ in range(n_iterations):
        x = np.random.randn(1, n_feat)
        t_start = time.perf_counter()
        _ = pipeline.score_samples(x)
        elapsed = (time.perf_counter() - t_start) * 1000
        times_ms.append(elapsed)

    return {
        "n_iterations" : n_iterations,
        "mean_ms"      : float(np.mean(times_ms)),
        "std_ms"       : float(np.std(times_ms)),
        "p50_ms"       : float(np.percentile(times_ms, 50)),
        "p95_ms"       : float(np.percentile(times_ms, 95)),
        "p99_ms"       : float(np.percentile(times_ms, 99)),
        "max_ms"       : float(np.max(times_ms)),
    }


def _compute_gini(arr: np.ndarray) -> float:
    if arr.sum() == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    idx = np.arange(1, n + 1)
    return float(
        (2 * np.sum(idx * sorted_arr)) / (n * sorted_arr.sum()) - (n+1)/n
    )


def run_full_overhead_benchmark(
    results_dir : str,
) -> Dict:
    """
    Run all overhead benchmarks and save results.
    """
    import json

    os.makedirs(results_dir, exist_ok=True)
    results = {}

    logger.info("Benchmarking register polling overhead...")
    results["register_poll"] = benchmark_register_poll()

    logger.info("Benchmarking feature extraction overhead...")
    results["feature_extraction"] = benchmark_feature_extraction()

    logger.info("Benchmarking model inference overhead...")
    results["model_inference"] = benchmark_model_inference()

    out_path = os.path.join(results_dir, "overhead_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Overhead benchmark saved to {out_path}")
    return results


def plot_overhead_breakdown(
    overhead_data : Dict,
    output_path   : str,
) -> None:
    """Plot stacked overhead breakdown per component."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_flows_list = sorted(
        overhead_data.get("register_poll", {}).keys()
    )

    if not n_flows_list:
        return

    poll_ms    = [overhead_data["register_poll"].get(n, {}).get("mean_ms", 0)
                  for n in n_flows_list]
    feat_ms    = [overhead_data.get("feature_extraction", {}).get(n, {}).get("mean_ms", 0)
                  for n in n_flows_list]
    infer_ms_v = overhead_data.get("model_inference", {}).get("mean_ms", 0)
    infer_ms   = [infer_ms_v] * len(n_flows_list)

    x    = np.arange(len(n_flows_list))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x, poll_ms,  width, label="Register Poll",
                   color="#3498db", alpha=0.85)
    bars2 = ax.bar(x, feat_ms,  width, bottom=poll_ms,
                   label="Feature Extraction",
                   color="#2ecc71", alpha=0.85)
    bars3 = ax.bar(x, infer_ms, width,
                   bottom=[p+f for p,f in zip(poll_ms, feat_ms)],
                   label="Model Inference",
                   color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_flows_list])
    ax.set_xlabel("Number of Flows")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Controller Overhead Breakdown per Poll Cycle")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Overhead breakdown saved to {output_path}")
