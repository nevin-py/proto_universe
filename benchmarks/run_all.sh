#!/usr/bin/env bash
# run_all.sh — Run all FiZK benchmarks sequentially (resumable)
#
# Total estimated time: 8-10 hours on RTX 4050 Laptop GPU
# Each script is independently resumable — re-run any script to continue.
#
# Usage:
#   bash benchmarks/run_all.sh              # run everything
#   bash benchmarks/run_all.sh overhead     # only overhead
#   bash benchmarks/run_all.sh scalability  # only scalability
#
# Run from workspace root.

set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
FILTER="${1:-all}"

run_bench() {
    local name="$1"
    local script="benchmarks/${name}.py"
    if [[ "$FILTER" != "all" && "$FILTER" != "$name" ]]; then
        echo "[SKIP] $name (not in filter)"
        return
    fi
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Starting: $name"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    $PYTHON "$script"
    echo ""
    echo "  Finished: $name  ($(date '+%H:%M:%S'))"
}

# Run in order: fast→slow, independent→dependent
run_bench bench_overhead           # ~45 min — quick sanity check
run_bench bench_scalability        # ~3 h
run_bench bench_zkp_value          # ~3 h
run_bench bench_attack_robustness  # ~5 h

echo ""
echo "All benchmarks complete. Results in Eval_results/benchmarks/"
