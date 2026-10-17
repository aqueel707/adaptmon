#!/usr/bin/env bash
# run_latency_idle.sh — the controller per-poll latency benchmark
# (experiments/latency_bench.py) on an IDLE host, then regenerate the paper
# numbers from it (analysis/paper_runC.py).
#
#   scripts/run_latency_idle.sh
#
# Refuses to start unless the host is idle: 1-minute load average < 0.5 and
# no IDE process running (close the IDE; run this from a plain terminal or
# over ssh). Then:
#   1. picks a seed never used before (checked against
#      experiments/seeds_used.json; experiments/latency_bench.py records it
#      there, purpose "latency-bench", as the first thing it does, and
#      refuses a seed already listed)
#   2. records host info + the load average before the run in
#      results/latency_bench_idle_<timestamp>/host_info.json
#   3. runs the benchmark into that folder (~9 minutes)
#   4. records the load average after the run
#   5. regenerates paper/runC/ (PAPER_NUMBERS.md, numbers.json, figures)
#      from that run; earlier runs stay listed there as superseded
set -euo pipefail

cd "$(dirname "$0")/.."

LOAD_MAX=0.5
IDE_PROCS="code codium code-oss vscodium cursor windsurf idea pycharm clion goland webstorm \
rider rustrover phpstorm datagrip eclipse zed sublime_text subl atom"

# --- 1. idle check ---------------------------------------------------------
load1=$(cut -d' ' -f1 /proc/loadavg)
if ! awk -v l="$load1" -v m="$LOAD_MAX" 'BEGIN { exit !(l < m) }'; then
    echo "WARNING: 1-minute load average is $load1 (needs < $LOAD_MAX)." >&2
    echo "The host is not idle: wait, or stop whatever is running, then retry." >&2
    exit 2
fi
running_ides=$(ps -eo comm= | sort -u | grep -Fxf <(tr ' ' '\n' <<< "$IDE_PROCS") || true)
if [ -n "$running_ides" ]; then
    echo "WARNING: IDE process(es) running: $(echo $running_ides)." >&2
    echo "Close them (run this from a plain terminal or over ssh), then retry." >&2
    exit 2
fi

# --- 2. seed + output folder + host info -----------------------------------
SEED=$(python3 - <<'EOF'
import json, secrets
used = json.load(open("experiments/seeds_used.json"))["seeds"]
while True:
    s = secrets.randbits(31)
    if str(s) not in used:
        print(s)
        break
EOF
)
OUT="results/latency_bench_idle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

python3 - "$OUT" "$SEED" <<'EOF'
import json, os, platform, subprocess, sys
out, seed = sys.argv[1], int(sys.argv[2])
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
            if l.startswith("model name")), platform.processor())
def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return None
info = {
    "seed": seed,
    "cpu_model": cpu,
    "cores": os.cpu_count(),
    "kernel": platform.release(),
    "os": dict(l.rstrip().split("=", 1) for l in open("/etc/os-release") if "=" in l).get(
        "PRETTY_NAME", "").strip('"'),
    "mem_total_kb": int(next(l.split()[1] for l in open("/proc/meminfo")
                             if l.startswith("MemTotal"))),
    "python": platform.python_version(),
    "simple_switch": run(["simple_switch", "--version"]),
    "loadavg_before": open("/proc/loadavg").read().strip(),
    "load_before": float(open("/proc/loadavg").read().split()[0]),
    "idle_check": {"load_1min_max": 0.5, "ide_processes_checked": True},
}
json.dump(info, open(os.path.join(out, "host_info.json"), "w"), indent=2)
print(f"host: {info['cpu_model']}, {info['cores']} cores, kernel {info['kernel']}, "
      f"load {info['load_before']}")
EOF

# --- 3. benchmark ----------------------------------------------------------
echo "latency benchmark: seed $SEED -> $OUT (~9 minutes)"
scripts/sandbox.sh python3 -m experiments.latency_bench --seed "$SEED" --out "$OUT"

python3 - "$SEED" <<'EOF'
import json, sys
used = json.load(open("experiments/seeds_used.json"))["seeds"]
assert sys.argv[1] in used, f"seed {sys.argv[1]} missing from experiments/seeds_used.json"
print(f"seed {sys.argv[1]} recorded: {used[sys.argv[1]]}")
EOF

# --- 4. load after ---------------------------------------------------------
python3 - "$OUT" <<'EOF'
import json, os, sys
p = os.path.join(sys.argv[1], "host_info.json")
info = json.load(open(p))
info["loadavg_after"] = open("/proc/loadavg").read().strip()
info["load_after"] = float(info["loadavg_after"].split()[0])
json.dump(info, open(p, "w"), indent=2)
EOF

# --- 5. paper numbers from this run ----------------------------------------
ADAPTMON_BENCH="$OUT" python3 -m analysis.paper_runC
echo
grep -m1 '^\*\*† = not a steady-state measurement' paper/runC/PAPER_NUMBERS.md \
    | sed 's/\*\*//g' | fold -s -w 100
echo
echo "done: $OUT -> paper/runC/PAPER_NUMBERS.md, paper/runC/fig_latency_tiers.{pdf,png}"
