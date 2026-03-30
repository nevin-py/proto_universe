#!/bin/bash
# Quick test of experiment infrastructure

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="outputs/test_run_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"/{logs,results}

echo "Testing experiment infrastructure..."

# Test 1: Byzantine detection
echo "▶  Test 1: Byzantine Detection"
python scripts/run_benchmark_suite.py \
    --benchmarks byzantine \
    --output "${RUN_DIR}/results" \
    > "${RUN_DIR}/logs/byzantine.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Byzantine detection: PASS"
else
    echo "❌ Byzantine detection: FAIL"
    tail -20 "${RUN_DIR}/logs/byzantine.log"
    exit 1
fi

echo ""
echo "✅ Infrastructure test complete!"
echo "Results in: ${RUN_DIR}"
ls -lh "${RUN_DIR}"/results/*.json
