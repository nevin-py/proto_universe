#!/bin/bash
# Simplified experiment runner with continuous logging

set -e
set -x  # Debug mode - print each command

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="outputs/experiments_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"/{logs,results}

LOG_FILE="${RUN_DIR}/main.log"

exec > >(tee -a "${LOG_FILE}") 2>&1  # Redirect all output to log + stdout

echo "========================================================================"
echo "EXPERIMENT SUITE - ${TIMESTAMP}"
echo "========================================================================"
echo "Directory: ${RUN_DIR}"
echo ""

# ============================================================================
# Experiment 1: Byzantine Detection
# ============================================================================

echo "========================================================================"
echo "EXPERIMENT 1: Byzantine Detection (All Architectures)"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

python scripts/run_benchmark_suite.py \
    --benchmarks byzantine \
    --output "${RUN_DIR}/results" \
    | tee "${RUN_DIR}/logs/exp1_byzantine.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Experiment 1: SUCCESS"
    echo "exp1_byzantine_detection" >> "${RUN_DIR}/completed.txt"
else
    echo "❌ Experiment 1: FAILED"
    exit 1
fi

echo ""
echo "End time: $(date)"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "========================================================================"
echo "EXPERIMENTS COMPLETE"
echo "========================================================================"
echo "Completed: $(wc -l < ${RUN_DIR}/completed.txt 2>/dev/null || echo 0)"
echo "Results: ${RUN_DIR}/results/"
ls -lh "${RUN_DIR}"/results/*.json 2>/dev/null || echo "No results found"
echo ""
echo "Full log: ${LOG_FILE}"
echo "========================================================================"
