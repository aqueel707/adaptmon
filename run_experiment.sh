#!/usr/bin/env bash
# run_experiment.sh — AdaptMon Experiment Launcher
# Usage:
#   ./run_experiment.sh           # auto-detects real vs mock
#   ./run_experiment.sh --mock    # force mock mode
#   sudo ./run_experiment.sh      # force real mode (BMv2 + namespaces)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${GRN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YLW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[ERROR]${NC} $*"; }
section() { echo -e "\n${BLU}==== $* ====${NC}"; }

echo ""
echo "============================================================"
echo "  AdaptMon: Adaptive Monitoring for Programmable Data Planes"
echo "============================================================"
echo ""

# ---- Parse args ----
N_RUNS=10; DURATION=120; ATTACK_START=30; ATTACK_DUR=60; ATTACK_RATE=5000
MOCK=0; VERBOSE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock)         MOCK=1;            shift ;;
        --n-runs)       N_RUNS="$2";       shift 2 ;;
        --duration)     DURATION="$2";     shift 2 ;;
        --attack-start) ATTACK_START="$2"; shift 2 ;;
        --attack-dur)   ATTACK_DUR="$2";   shift 2 ;;
        --attack-rate)  ATTACK_RATE="$2";  shift 2 ;;
        --verbose)      VERBOSE="--verbose"; shift ;;
        --help)
            echo "  --mock            Synthetic data, no switches needed"
            echo "  --n-runs N        Runs (default 10)"
            echo "  --duration N      Seconds per run (default 120)"
            echo "  --attack-start N  Seconds before attack starts (default 30)"
            echo "  --attack-dur N    Attack duration in seconds (default 60)"
            echo "  --attack-rate N   Attack rate in pps (default 5000)"
            echo "  --verbose         Debug logging"
            exit 0 ;;
        *) shift ;;
    esac
done

# ---- Python ----
section "Environment"
PY=$(command -v python3 || command -v python)
[ -z "$PY" ] && { err "python3 not found"; exit 1; }
info "Python: $($PY --version 2>&1)"

# ---- Pip ----
$PY -m pip install -q --break-system-packages -r requirements.txt 2>/dev/null || \
$PY -m pip install -q -r requirements.txt 2>/dev/null || true
info "Python dependencies ready"
mkdir -p logs results p4/build

# ---- Real mode prerequisites ----
if [ $MOCK -eq 0 ]; then
    if ! command -v p4c &>/dev/null && ! command -v p4c-bm2-ss &>/dev/null; then
        warn "No P4 compiler (p4c / p4c-bm2-ss) found — falling back to mock mode"
        MOCK=1
    elif ! command -v simple_switch &>/dev/null; then
        warn "simple_switch not found — falling back to mock mode"
        MOCK=1
    fi
fi

# ---- Real mode needs root for network namespaces ----
# Done before compiling so root-owned files in p4/build from earlier runs
# cannot make the compile fail and silently force mock mode.
if [ $MOCK -eq 0 ] && [ "$EUID" -ne 0 ]; then
    warn "Real mode needs root for network namespaces."
    info "Re-running with sudo (you will be prompted once)..."
    echo ""
    exec sudo -E bash "$0" \
        --n-runs "$N_RUNS" --duration "$DURATION" \
        --attack-start "$ATTACK_START" --attack-dur "$ATTACK_DUR" \
        --attack-rate "$ATTACK_RATE" $VERBOSE
fi

# ---- P4 compilation ----
section "P4 Compilation"
if [ $MOCK -eq 0 ]; then
    COMPILED=0
    if command -v p4c &>/dev/null; then
        if p4c --target bmv2 --arch v1model --std p4-16 \
               -o p4/build p4/adaptmon.p4 2>p4/build/compile.log; then
            info "P4 compiled OK"
            COMPILED=1
        else
            warn "p4c failed — see p4/build/compile.log"
        fi
    fi
    if [ $COMPILED -eq 0 ] && command -v p4c-bm2-ss &>/dev/null; then
        if p4c-bm2-ss --std p4-16 -o p4/build/adaptmon.json \
                      p4/adaptmon.p4 2>>p4/build/compile.log; then
            info "P4 compiled OK (p4c-bm2-ss)"
            COMPILED=1
        fi
    fi
    if [ $COMPILED -eq 0 ]; then
        err "P4 compilation failed — see p4/build/compile.log"
        err "Fix the build, or pass --mock to run with synthetic data."
        exit 1
    fi
else
    info "Skipped (mock mode)"
fi

# ---- Summary ----
section "Experiment Configuration"
[ $MOCK -eq 1 ] && info "  Mode    : MOCK (synthetic data)" \
                || info "  Mode    : REAL (BMv2 + Linux namespaces)"
info "  n_runs  : $N_RUNS"
info "  duration: ${DURATION}s per run"
info "  attack  : starts at ${ATTACK_START}s, lasts ${ATTACK_DUR}s, ${ATTACK_RATE}pps"
echo ""

# ---- Launch ----
section "Running"
MOCK_ARG=""; [ $MOCK -eq 1 ] && MOCK_ARG="--mock"
START=$(date +%s)

$PY run_experiment.py \
    --n-runs       "$N_RUNS"       \
    --duration     "$DURATION"     \
    --attack-start "$ATTACK_START" \
    --attack-dur   "$ATTACK_DUR"   \
    --attack-rate  "$ATTACK_RATE"  \
    --results-dir  results         \
    --skip-compile                 \
    $MOCK_ARG $VERBOSE

CODE=$?
ELAPSED=$(( $(date +%s) - START ))

# Under sudo, hand generated files back to the invoking user so later
# non-root runs (and p4c) can overwrite them.
if [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER": results logs p4/build 2>/dev/null
    find . -name __pycache__ -type d -exec chown -R "$SUDO_USER": {} + 2>/dev/null
fi

section "Done"
info "Time: ${ELAPSED}s"
if [ $CODE -eq 0 ]; then
    info "Status: SUCCESS"
    LATEST=$(ls -td results/*/ 2>/dev/null | head -1)
    [ -n "$LATEST" ] && info "Series data: $LATEST"
    info "Graphs:"
    find results -maxdepth 1 -name "*.png" | sort | while read -r f; do
        echo "    $f"
    done
    info "Report: results/report.md"
else
    err "Failed (exit=$CODE) — check logs/ and results/<series>/run_*/controller_out.log"
fi
exit $CODE
