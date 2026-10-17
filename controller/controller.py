"""
controller/controller.py
Main controller entry point.
Parses CLI args and starts the AdaptMon monitoring system.
"""

import argparse
import logging
import os
import sys
import signal
import time
import json

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.monitor import AdaptMonController
from controller.policy  import PolicyConfig
from controller.utils   import setup_logging, ensure_dirs, get_timestamp_str

logger = logging.getLogger(__name__)

DEFAULT_SWITCH_CONFIGS = [
    {"switch_id": 1, "thrift_port": 9090},
    {"switch_id": 2, "thrift_port": 9091},
]

DEFAULT_FORWARDING_RULES = [
    # Switch 1 rules
    {"switch_id": 1, "dst_ip": "10.0.1.1", "port": 1},
    {"switch_id": 1, "dst_ip": "10.0.1.2", "port": 2},
    {"switch_id": 1, "dst_ip": "10.0.2.1", "port": 3},
    {"switch_id": 1, "dst_ip": "10.0.2.2", "port": 3},
    # Switch 2 rules
    {"switch_id": 2, "dst_ip": "10.0.2.1", "port": 1},
    {"switch_id": 2, "dst_ip": "10.0.2.2", "port": 2},
    {"switch_id": 2, "dst_ip": "10.0.1.1", "port": 3},
    {"switch_id": 2, "dst_ip": "10.0.1.2", "port": 3},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdaptMon: Adaptive Monitoring Controller"
    )
    parser.add_argument(
        "--switches", type=str, default=None,
        help="JSON file with switch configs"
    )
    parser.add_argument(
        "--forwarding", type=str, default=None,
        help="JSON file with forwarding rules"
    )
    parser.add_argument(
        "--poll-interval", type=float, default=1.0,
        help="Register polling interval in seconds"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for log files"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Experiment run identifier"
    )
    parser.add_argument(
        "--duration", type=int, default=300,
        help="Controller run duration in seconds (0 = infinite)"
    )
    parser.add_argument(
        "--train-until", type=float, default=None,
        help="Epoch time the benign warm-up ends (attack start): train the "
             "detector once on samples before it, then freeze. "
             "Omit for periodic retraining"
    )
    parser.add_argument(
        "--flow-idle-polls", type=int, default=3,
        help="Clear a flow register slot after its counter has not "
             "changed for this many polls"
    )
    parser.add_argument(
        "--contamination", type=float, default=0.05,
        help="IsolationForest contamination parameter"
    )
    parser.add_argument(
        "--normal-threshold", type=float, default=-0.10,
        help="Score threshold for NORMAL level"
    )
    parser.add_argument(
        "--monitor-threshold", type=float, default=-0.20,
        help="Score threshold for MONITOR level"
    )
    parser.add_argument(
        "--clone-threshold", type=float, default=-0.30,
        help="Score threshold for CLONE level"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_id  = args.run_id or get_timestamp_str()
    log_dir = os.path.join(args.log_dir, run_id)

    ensure_dirs([log_dir, "results"])

    setup_logging(
        log_dir,
        log_file="controller.log",
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    logger.info(f"AdaptMon Controller starting (run_id={run_id})")

    # Load switch configs
    if args.switches and os.path.exists(args.switches):
        with open(args.switches) as f:
            switch_configs = json.load(f)
    else:
        switch_configs = DEFAULT_SWITCH_CONFIGS

    # Load forwarding rules
    if args.forwarding and os.path.exists(args.forwarding):
        with open(args.forwarding) as f:
            forwarding_rules = json.load(f)
    else:
        forwarding_rules = DEFAULT_FORWARDING_RULES

    # Build policy config
    policy_cfg = PolicyConfig(
        normal_threshold  = args.normal_threshold,
        monitor_threshold = args.monitor_threshold,
        clone_threshold   = args.clone_threshold,
    )

    # Create controller
    controller = AdaptMonController(
        switch_configs = switch_configs,
        poll_interval  = args.poll_interval,
        log_dir        = log_dir,
        policy_config  = policy_cfg,
        train_until    = args.train_until,
        flow_idle_polls = args.flow_idle_polls,
    )

    # Signal handler for graceful shutdown
    def _sighandler(signum, frame):
        logger.info("Received signal, shutting down...")
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    # Wait for switches
    logger.info("Waiting for switches to come up...")
    if not controller.wait_for_switches(timeout=60):
        logger.error("Switches did not come up in time, proceeding anyway")

    # Install forwarding rules
    logger.info("Installing base forwarding rules...")
    controller.install_forwarding(forwarding_rules)

    # Start monitoring
    controller.start()

    # Run for specified duration
    try:
        if args.duration > 0:
            logger.info(f"Running for {args.duration} seconds")
            time.sleep(args.duration)
            logger.info("Duration elapsed, stopping")
        else:
            logger.info("Running indefinitely (Ctrl+C to stop)")
            while True:
                time.sleep(10)
                summary = controller.get_policy_summary()
                logger.info(f"Policy summary: {summary}")
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        controller.stop()
        logger.info("Controller terminated cleanly")

        # Save final stats
        stats = {
            "run_id"     : run_id,
            "policy"     : controller.get_policy_summary(),
            "detectors"  : controller.get_detector_stats(),
            "metrics"    : controller.get_metrics_paths(),
        }
        stats_path = os.path.join(log_dir, "final_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Final stats saved to {stats_path}")


if __name__ == "__main__":
    main()
