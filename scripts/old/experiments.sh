#!/bin/bash
# Working Experiment Suite - Fixed Version

set -e
set -o pipefail

DIR="outputs/exps_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${DIR}/logs" "${DIR}/results"

LOG="${DIR}/main.log"

# Log to both file and stdout
exec > >(tee -a "${LOG}") 2>&1

echo "=========================================="
echo "EXPERIMENT SUITE - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo "Directory: ${DIR}"
echo ""

TOTAL=0
PASSED=0

# Helper
run_test() {
    local name="$1"
    local cmd="$2"
    
    ((TOTAL++))
    echo ""
    echo "=========================================="
    echo "EXPERIMENT ${TOTAL}: ${name}"
    echo "=========================================="
    echo "Start: $(date '+%H:%M:%S')"
    echo ""
    
    if ${cmd}; then
        echo ""
        echo "✅ ${name}: PASS"
        ((PASSED++))
    else
        echo ""
        echo "❌ ${name}: FAIL"
    fi
}

# Run experiments
run_test "Byzantine Detection" "python scripts/run_benchmark_suite.py --benchmarks byzantine --output ${DIR}/results"
run_test "Linear Model Test" "python scripts/test_byzantine_multimodel.py"
run_test "MLP Model Test" "python scripts/test_byzantine_mlp.py"
run_test "CNN Model Test" "python scripts/test_byzantine_cnn.py"

# Summary
echo ""
echo "=========================================="
echo "SUITE COMPLETE - $(date '+%H:%M:%S')"
echo "=========================================="
echo "Total: ${TOTAL} | Passed: ${PASSED} | Failed: $((TOTAL - PASSED))"
echo ""
echo "Results: ${DIR}/results/"
echo "Logs: ${DIR}/logs/"
echo "Main log: ${LOG}"
echo ""
echo "Files created:"
find "${DIR}" -type f | while read f; do
    echo "  - $(basename $f) ($(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null) bytes)"
done
echo ""

if [ ${PASSED} -eq ${TOTAL} ]; then
    echo "🎉 ALL ${TOTAL} EXPERIMENTS PASSED!"
    exit 0
else
    echo "⚠️  $((TOTAL - PASSED)) failed"
    exit 1
fi
