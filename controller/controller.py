"""
controller/controller.py
Main controller entry point.
Parses CLI args and starts the AdaptMon monitoring system.

    python3 -m controller.controller --switches sw.json --out-dir results/<run> \
        --train-until <epoch> --duration 120 [--dpi-ifaces 1:am_s1dpi,2:am_s2dpi]
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time

# Make sure local modules are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from controller.monitor import AdaptMonController
from controller.utils   import setup_logging, ensure_dirs, get_timestamp_str

logger = logging.getLogger(__name__)

DEFAULT_SWITCH_CONFIGS = [
    {"switch_id": 1, "thrift_port": 9090},
    {"switch_id": 2, "thrift_port": 9091},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdaptMon: Adaptive Monitoring Controller"
    )
    parser.add_argument("--switches", type=str, default=None,
                        help="JSON file with switch configs")
    parser.add_argument("--forwarding", type=str, default=None,
                        help="JSON file with forwarding rules (optional; the "
                             "topology normally installs them)")
    parser.add_argument("--poll-interval", type=float, default=1.0,
                        help="Register polling interval in seconds")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default logs/<run-id>)")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--duration", type=float, default=300,
                        help="Controller run duration in seconds (0 = infinite)")
    parser.add_argument("--train-until", type=float, default=None,
                        help="Epoch time the benign warm-up ends: the detector "
                             "is trained once on flow vectors before it")
    parser.add_argument("--warmup", type=float, default=None,
                        help="Alternative to --train-until: seconds from start")
    parser.add_argument("--flow-idle-polls", type=int, default=3,
                        help="Clear a flow register slot after its counter has "
                             "not changed for this many polls")
    parser.add_argument("--dpi-ifaces", type=str, default=None,
                        help="Start controller/dpi.py on these CPU-port "
                             "interfaces, e.g. 1:am_s1dpi,2:am_s2dpi")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_args()

    run_id  = args.run_id or get_timestamp_str()
    out_dir = args.out_dir or os.path.join("logs", run_id)
    ensure_dirs([out_dir])

    setup_logging(
        out_dir,
        log_file="controller.log",
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    logger.info(f"AdaptMon Controller starting (run_id={run_id}, out_dir={out_dir})")

    if args.switches and os.path.exists(args.switches):
        with open(args.switches) as f:
            switch_configs = json.load(f)
    else:
        switch_configs = DEFAULT_SWITCH_CONFIGS

    train_until = args.train_until
    if train_until is None and args.warmup is not None:
        train_until = time.time() + args.warmup

    # DPI sniffer on the CPU ports (separate process: own CPU/RSS figures)
    dpi_proc = None
    if args.dpi_ifaces:
        dpi_proc = subprocess.Popen(
            [sys.executable, "-m", "controller.dpi",
             "--ifaces", args.dpi_ifaces,
             "--alerts", os.path.join(out_dir, "alerts.jsonl"),
             "--out-dir", out_dir,
             "--snapshot-interval", str(args.poll_interval)],
            cwd=ROOT,
            stdout=open(os.path.join(out_dir, "dpi_out.log"), "w"),
            stderr=subprocess.STDOUT,
        )
        logger.info(f"DPI started (pid={dpi_proc.pid}) on {args.dpi_ifaces}")

    controller = AdaptMonController(
        switch_configs  = switch_configs,
        poll_interval   = args.poll_interval,
        out_dir         = out_dir,
        train_until     = train_until,
        flow_idle_polls = args.flow_idle_polls,
        dpi_pid         = dpi_proc.pid if dpi_proc else None,
    )

    stopping = {"flag": False}

    def _sighandler(signum, frame):
        logger.info("Received signal, shutting down...")
        stopping["flag"] = True

    signal.signal(signal.SIGINT,  _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    logger.info("Waiting for switches to come up...")
    if not controller.wait_for_switches(timeout=60):
        logger.error("Switches did not come up in time")
        sys.exit(2)

    if args.forwarding and os.path.exists(args.forwarding):
        with open(args.forwarding) as f:
            controller.install_forwarding(json.load(f))

    # Clone session 100 -> CPU port (mirroring_add 100 255) on every switch
    mirroring = controller.dpm.install_mirroring_all()
    for sid, ok in mirroring.items():
        if ok:
            logger.info(f"S{sid}: mirroring session 100 -> CPU port 255 configured")
        else:
            logger.error(f"S{sid}: mirroring session 100 could not be configured")

    controller.start()
    t_end = time.time() + args.duration if args.duration > 0 else float("inf")
    try:
        while not stopping["flag"] and time.time() < t_end:
            time.sleep(0.2)
        logger.info("Duration elapsed, stopping" if not stopping["flag"] else "Stopping")
    finally:
        controller.stop()
        if dpi_proc is not None:
            dpi_proc.send_signal(signal.SIGTERM)
            try:
                dpi_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                dpi_proc.kill()
        stats = {"run_id": run_id, **controller.summary()}
        with open(os.path.join(out_dir, "final_stats.json"), "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info("Controller terminated cleanly")


if __name__ == "__main__":
    main()
