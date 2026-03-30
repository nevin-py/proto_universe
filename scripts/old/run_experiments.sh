#!/bin/bash
# Working Experiment Runner with Continuous Logging
# Simple and tested version

# Exit on error
set -e

# Create timestamped directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="outputs/experiments_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/results"

# Log files
MAIN_LOG="${RESULTS_DIR}/main.log"

# Helper function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

# Start
log "=========================================="
log "EXPERIMENT SUITE STARTED"
log "=========================================="
log "Results directory: ${RESULTS_DIR}"
log ""

# ============================================================================
# EXPERIMENT 1: Byzantine Detection
# ============================================================================

log "Running: Byzantine Detection (All Architectures)"
log "Output: ${RESULTS_DIR}/logs/byzantine.log"

python scripts/run_benchmark_suite.py \
    --benchmarks byzantine \
    --output "${RESULTS_DIR}/results" \
    2>&1 | tee "${RESULTS_DIR}/logs/byzantine.log" | tee -a "${MAIN_LOG}"

BYZANTINE_EXIT=$?

if [ $BYZANTINE_EXIT -eq 0 ]; then
    log "✅ Byzantine Detection: PASS"
    echo "byzantine_detection" >> "${RESULTS_DIR}/completed.txt"
else
    log "❌ Byzantine Detection: FAILED (exit code: $BYZANTINE_EXIT)"
fi

log ""

# ============================================================================
# SUMMARY
# ============================================================================

log "=========================================="
log "EXPERIMENT SUITE COMPLETE"
log "=========================================="
log ""
log "Results saved to: ${RESULTS_DIR}/results/"
log "Logs saved to: ${RESULTS_DIR}/logs/"
log "Main log: ${MAIN_LOG}"
log ""

# List results
log "Generated files:"
ls -lh "${RESULTS_DIR}"/results/*.json 2>/dev/null | while read line; do
    log "  $line"
done

log ""
log "Completed experiments:"
if [ -f "${RESULTS_DIR}/completed.txt" ]; then
    cat "${RESULTS_DIR}/completed.txt" | while read exp; do
        log "  - $exp"
    done
else
    log "  (none)"
fi

log ""
log "=========================================="

if [ $BYZANTINE_EXIT -eq 0 ]; then
    log "✅ ALL EXPERIMENTS PASSED"
    exit 0
else
    log "❌ SOME EXPERIMENTS FAILED"
    exit 1
fi
