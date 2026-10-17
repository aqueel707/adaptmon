#!/usr/bin/env python3
"""
run_experiment.py
Main entry point for AdaptMon experiments.
Orchestrates: P4 compilation, topology, controller, traffic, analysis.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from controller.utils       import setup_logging, ensure_dirs, get_timestamp_str
from experiments.run_series import (
    ExperimentSeries, compile_p4, run_scalability_experiment,
    EXPERIMENT_CONFIG, P4_SRC, P4_BUILD, P4_JSON, RESULTS
)
from experiments.metrics    import (
    load_all_series_csvs, combined_metrics_df
)
from analysis.plots         import generate_all_plots
from analysis.stats         import generate_report
from analysis.overhead      import run_full_overhead_benchmark, plot_overhead_breakdown

logger = logging.getLogger(__name__)


def setup_environment() -> bool:
    """Check and set up the experiment environment."""
    checks = {
        "p4c"              : ["p4c", "--version"],
        "simple_switch"    : ["simple_switch", "--version"],
        "python3"          : ["python3", "--version"],
    }

    all_ok = True
    for name, cmd in checks.items():
        try:
            r = subprocess.run(
                cmd, capture_output=True, timeout=5
            )
            status = "OK" if r.returncode == 0 else "WARNING"
        except FileNotFoundError:
            status = "MISSING"
            all_ok = False
        except Exception:
            status = "ERROR"
        logger.info(f"  {name}: {status}")

    # Check Python packages
    for pkg in ["sklearn", "numpy", "pandas", "matplotlib", "psutil"]:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
            logger.info(f"  python:{pkg}: OK")
        except ImportError:
            logger.warning(f"  python:{pkg}: MISSING (install with pip)")
            all_ok = False

    return all_ok


def run_full_experiment(
    n_runs       : int  = 10,
    duration     : int  = 120,
    attack_start : int  = 30,
    attack_dur   : int  = 60,
    attack_rate  : int  = 5000,
    results_dir  : str  = RESULTS,
    skip_compile : bool = False,
    mock_mode    : bool = False,
) -> str:
    """
    Full experiment pipeline.
    Returns path to results directory.
    """
    run_id = get_timestamp_str()
    exp_results = results_dir   # ExperimentSeries adds its own timestamp inside
    ensure_dirs([exp_results, "logs"])

    logger.info("=" * 60)
    logger.info("AdaptMon Full Experiment")
    logger.info(f"  runs={n_runs}, duration={duration}s")
    logger.info(f"  attack: start={attack_start}s, dur={attack_dur}s, rate={attack_rate}pps")
    logger.info(f"  results_dir={exp_results}")
    logger.info("=" * 60)

    # 1. Compile P4
    if not skip_compile:
        logger.info("Step 1: Compiling P4 program...")
        if not compile_p4(P4_SRC, P4_BUILD):
            logger.warning(
                "P4 compilation failed — "
                "experiment will use synthetic data if no real switch available"
            )
    else:
        logger.info("Step 1: Skipping P4 compilation (--skip-compile)")

    # 2. Run experiment series
    config = EXPERIMENT_CONFIG.copy()
    config["n_runs"]           = n_runs
    config["run_duration"]     = duration
    config["attack_start"]     = attack_start
    config["attack_duration"]  = attack_dur
    config["attack_rate"]      = attack_rate

    logger.info(f"Step 2: Running experiment series (mock={mock_mode})...")
    series = ExperimentSeries(
        config      = config,
        results_dir = exp_results,
        mock        = mock_mode,
    )
    summary = series.run()
    series_dir = summary["series_dir"]
    if summary["n_completed"] == 0:
        logger.error(
            f"All {summary['n_total']} runs failed — no data to analyse. "
            f"Check controller_out.log files in {series_dir}"
        )
        sys.exit(1)
    if summary["n_failed"]:
        logger.warning(
            f"{summary['n_failed']}/{summary['n_total']} runs failed; "
            "analysing the remaining runs only"
        )
    errors = []

    # 3. Scalability experiment
    logger.info("Step 3: Running scalability benchmark...")
    try:
        sc_data = run_scalability_experiment(
            max_flows_list = [100, 500, 1000, 2000, 4096],
            results_dir    = exp_results,
        )
    except Exception as e:
        logger.warning(f"Scalability experiment failed: {e}")

    # 4. Overhead benchmarks
    logger.info("Step 4: Running overhead benchmarks...")
    try:
        overhead_data = run_full_overhead_benchmark(exp_results)
        plot_overhead_breakdown(
            overhead_data,
            os.path.join(exp_results, "overhead_breakdown.png")
        )
    except Exception as e:
        logger.warning(f"Overhead benchmark failed: {e}")

    # 5. Generate plots
    logger.info("Step 5: Generating result graphs...")
    try:
        generated = generate_all_plots(
            results_dir    = exp_results,
            series_dir     = series_dir,
            attack_start_s = float(attack_start),
            attack_end_s   = float(attack_start + attack_dur),
            allow_synthetic = mock_mode,
        )
        for p in generated:
            logger.info(f"  Generated: {p}")
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        errors.append("plots")

    # 6. Generate report
    logger.info("Step 6: Generating report...")
    try:
        dfs = load_all_series_csvs(series_dir)
        if not dfs and not mock_mode:
            raise RuntimeError(f"No metrics CSVs found in {series_dir}")
        if not dfs:
            from analysis.plots import _generate_synthetic_df
            dfs = [
                _generate_synthetic_df(attack_start, attack_start + attack_dur)
                for _ in range(n_runs)
            ]
        report_path = os.path.join(exp_results, "report.md")
        generate_report(
            dfs,
            report_path,
            attack_start_s = float(attack_start),
            attack_end_s   = float(attack_start + attack_dur),
        )
        logger.info(f"  Report: {report_path}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        errors.append("report")

    logger.info("=" * 60)
    logger.info(f"Experiment complete. Results at: {exp_results}")
    logger.info("=" * 60)

    # Print file listing
    logger.info("Generated files:")
    for fname in sorted(os.listdir(exp_results)):
        fpath = os.path.join(exp_results, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            logger.info(f"  {fname} ({size:,} bytes)")

    if errors:
        logger.error(f"Failed steps: {', '.join(errors)}")
        sys.exit(1)
    return exp_results


def _generate_mock_series(
    config       : dict,
    base_dir     : str,
    n_runs       : int,
    attack_start : int,
    attack_dur   : int,
) -> str:
    """Generate synthetic data instead of running real switches."""
    from analysis.plots import _generate_synthetic_df

    series_dir = os.path.join(base_dir, "mock_series")
    ensure_dirs([series_dir])

    for run_idx in range(1, n_runs + 1):
        run_dir = os.path.join(series_dir, f"run_{run_idx:02d}")
        ensure_dirs([run_dir])

        # Generate synthetic data for 2 switches
        for sw_id in [1, 2]:
            df = _generate_synthetic_df(
                attack_start_s = float(attack_start),
                attack_end_s   = float(attack_start + attack_dur),
                duration       = config["run_duration"],
            )
            df["switch_id"] = sw_id
            csv_path = os.path.join(
                run_dir, f"switch_{sw_id}_metrics.csv"
            )
            df.to_csv(csv_path, index=False)
            logger.info(f"  Generated mock data: {csv_path}")

    return series_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="AdaptMon: Adaptive Monitoring System for Programmable Data Planes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--n-runs",       type=int,   default=10,
                   help="Number of experiment repetitions")
    p.add_argument("--duration",     type=int,   default=120,
                   help="Duration per run (seconds)")
    p.add_argument("--attack-start", type=int,   default=30,
                   help="Seconds after start to begin attack")
    p.add_argument("--attack-dur",   type=int,   default=60,
                   help="Attack duration (seconds)")
    p.add_argument("--attack-rate",  type=int,   default=5000,
                   help="Attack rate (packets/sec)")
    p.add_argument("--results-dir",  type=str,   default=RESULTS,
                   help="Directory for results")
    p.add_argument("--skip-compile", action="store_true",
                   help="Skip P4 compilation")
    p.add_argument("--mock",         action="store_true",
                   help="Use synthetic data (no real switches needed)")
    p.add_argument("--plots-only",   action="store_true",
                   help="Generate plots from existing data only")
    p.add_argument("--series-dir",   type=str,   default=None,
                   help="Existing series directory for --plots-only")
    p.add_argument("--check-env",    action="store_true",
                   help="Check environment and exit")
    p.add_argument("--verbose",      action="store_true",
                   help="Verbose logging")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = get_timestamp_str()

    setup_logging(
        log_dir  = os.path.join("logs", run_id),
        log_file = "main.log",
        level    = logging.DEBUG if args.verbose else logging.INFO
    )

    if args.check_env:
        logger.info("Checking environment...")
        ok = setup_environment()
        sys.exit(0 if ok else 1)

    if args.plots_only:
        if not args.series_dir:
            logger.error("--series-dir required with --plots-only")
            sys.exit(1)
        exp_results = args.results_dir
        ensure_dirs([exp_results])
        try:
            generated = generate_all_plots(
                results_dir    = exp_results,
                series_dir     = args.series_dir,
                attack_start_s = float(args.attack_start),
                attack_end_s   = float(args.attack_start + args.attack_dur),
            )
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
        print(f"Generated {len(generated)} plots in {exp_results}")
        return

    results_dir = run_full_experiment(
        n_runs       = args.n_runs,
        duration     = args.duration,
        attack_start = args.attack_start,
        attack_dur   = args.attack_dur,
        attack_rate  = args.attack_rate,
        results_dir  = args.results_dir,
        skip_compile = args.skip_compile,
        mock_mode    = args.mock,
    )

    print(f"\nResults saved to: {results_dir}")
    print("Graphs generated:")
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".png"):
            print(f"  {os.path.join(results_dir, fname)}")


if __name__ == "__main__":
    main()
