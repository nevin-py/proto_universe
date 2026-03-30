#!/bin/bash
# Complete Experiment Suite - Simplified and Working

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DIR="outputs/exps_${TIMESTAMP}"
mkdir -p "${DIR}/logs" "${DIR}/results"

LOG="${DIR}/main.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"
}

log "=========================================="
log "EXPERIMENT SUITE STARTED"
log "Directory: ${DIR}"
log "=========================================="

TOTAL=0
PASSED=0

# ============================================================================
# EXP 1: Byzantine Detection - All Architectures
# ============================================================================

((TOTAL++))
log ""
log "EXP 1/4: Byzantine Detection (Linear, MLP, CNN)"
log "=========================================="

python scripts/run_benchmark_suite.py \
    --benchmarks byzantine \
    --output "${DIR}/results" \
    2>&1 | tee "${DIR}/logs/exp1_byzantine.log" | tee -a "${LOG}"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "✅ EXP 1: PASS"
    ((PASSED++))
else
    log "❌ EXP 1: FAIL"
fi

# ============================================================================
# EXP 2: Linear Model Byzantine Test  
# ============================================================================

((TOTAL++))
log ""
log "EXP 2/4: Linear Model Byzantine Detection"
log "=========================================="

python scripts/test_byzantine_multimodel.py \
    2>&1 | tee "${DIR}/logs/exp2_linear.log" | tee -a "${LOG}"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "✅ EXP 2: PASS"
    ((PASSED++))
else
    log "❌ EXP 2: FAIL"
fi

# ============================================================================
# EXP 3: MLP Model Byzantine Test
# ============================================================================

((TOTAL++))
log ""
log "EXP 3/4: MLP Model Byzantine Detection"
log "=========================================="

python scripts/test_byzantine_mlp.py \
    2>&1 | tee "${DIR}/logs/exp3_mlp.log" | tee -a "${LOG}"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "✅ EXP 3: PASS"
    ((PASSED++))
else
    log "❌ EXP 3: FAIL"
fi

# ============================================================================
# EXP 4: CNN Model Byzantine Test
# ============================================================================

((TOTAL++))
log ""
log "EXP 4/4: CNN Model Byzantine Detection"
log "=========================================="

python scripts/test_byzantine_cnn.py \
    2>&1 | tee "${DIR}/logs/exp4_cnn.log" | tee -a "${LOG}"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "✅ EXP 4: PASS"
    ((PASSED++))
else
    log "❌ EXP 4: FAIL"
fi

# ============================================================================
# SUMMARY
# ============================================================================

log ""
log "=========================================="
log "SUITE COMPLETE"
log "=========================================="
log "Total: ${TOTAL} | Passed: ${PASSED} | Failed: $((TOTAL - PASSED))"
log "Success Rate: $((PASSED * 100 / TOTAL))%"
log ""
log "Results: ${DIR}/results/"
log "Logs: ${DIR}/logs/"
log "Main log: ${LOG}"
log ""

# List results
log "Files created:"
find "${DIR}" -name "*.json" -o -name "*.log" | sort | while read f; do
    size=$(ls -lh "$f" | awk '{print $5}')
    log "  $(basename $f) - ${size}"
done

log ""
log "=========================================="

if [ ${PASSED} -eq ${TOTAL} ]; then
    log "🎉 ALL ${TOTAL} EXPERIMENTS PASSED!"
    exit 0
else
    log "⚠️  $((TOTAL - PASSED)) experiments failed"
    exit 1
fi
