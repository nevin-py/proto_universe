#!/bin/bash
# Complete Experiment Suite - All Configurations
# Byzantine detection + Multiple architectures + IID/non-IID + Attacks + Defenses

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="outputs/full_experiments_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/results"

MAIN_LOG="${RESULTS_DIR}/main.log"
CHECKPOINT="${RESULTS_DIR}/completed.txt"

# Log function with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

# Experiment counter
TOTAL=0
PASSED=0
FAILED=0

# Run experiment helper
run_exp() {
    local name="$1"
    local cmd="$2"
    local logfile="${RESULTS_DIR}/logs/${name}.log"
    
    ((TOTAL++))
    
    log ""
    log "=========================================="
    log "EXPERIMENT ${TOTAL}: ${name}"
    log "=========================================="
    log "Start: $(date '+%H:%M:%S')"
    log "Command: ${cmd}"
    log "Log: ${logfile}"
    log ""
    
    if eval "${cmd}" 2>&1 | tee "${logfile}" | tee -a "${MAIN_LOG}"; then
        log ""
        log "✅ ${name}: PASS"
        echo "${name}" >> "${CHECKPOINT}"
        ((PASSED++))
        return 0
    else
        log ""
        log "❌ ${name}: FAIL"
        ((FAILED++))
        return 1
    fi
}

# Start
log "========================================================================"
log "COMPREHENSIVE EXPERIMENT SUITE"
log "========================================================================"
log "Timestamp: ${TIMESTAMP}"
log "Directory: ${RESULTS_DIR}"
log ""

# ============================================================================
# PART 1: Byzantine Detection Validation (Multi-Architecture)
# ============================================================================

log ""
log "========================================================================"
log "PART 1: BYZANTINE DETECTION VALIDATION"
log "========================================================================"

run_exp \
    "byzantine_detection_all_architectures" \
    "python scripts/run_benchmark_suite.py --benchmarks byzantine --output ${RESULTS_DIR}/results"

# ============================================================================
# PART 2: Individual Architecture Tests
# ============================================================================

log ""
log "========================================================================"
log "PART 2: INDIVIDUAL ARCHITECTURE TESTS"
log "========================================================================"

run_exp \
    "byzantine_linear_model" \
    "python scripts/test_byzantine_multimodel.py"

run_exp \
    "byzantine_mlp_model" \
    "python scripts/test_byzantine_mlp.py"

run_exp \
    "byzantine_cnn_model" \
    "python scripts/test_byzantine_cnn.py"

# ============================================================================
# PART 3: Defense Comparison (if comprehensive evaluation script exists)
# ============================================================================

log ""
log "========================================================================"
log "PART 3: DEFENSE MECHANISMS EVALUATION"
log "========================================================================"

# Check if comprehensive evaluation script has proper interface
if python scripts/run_comprehensive_evaluation.py --help 2>&1 | grep -q "datasets"; then
    log "Running defense comparison experiments..."
    
    # MNIST IID Linear Model Poisoning
    run_exp \
        "defense_mnist_iid_linear_poisoning" \
        "python scripts/run_comprehensive_evaluation.py --datasets mnist --partition iid --models linear --attacks modelpoisoning --alpha 0.3 --rounds 5 --clients 10 --output ${RESULTS_DIR}/results/defense_mnist_iid.json" || true
    
    # MNIST non-IID Linear Model Poisoning  
    run_exp \
        "defense_mnist_noniid_linear_poisoning" \
        "python scripts/run_comprehensive_evaluation.py --datasets mnist --partition noniid --models linear --attacks modelpoisoning --alpha 0.3 --rounds 5 --clients 10 --output ${RESULTS_DIR}/results/defense_mnist_noniid.json" || true
else
    log "⚠️  Skipping defense comparison - script interface not compatible"
    log "   Using Byzantine detection as primary validation"
fi

# ============================================================================
# PART 4: Scalability Analysis (varying malicious fraction)
# ============================================================================

log ""
log "========================================================================"
log "PART 4: SCALABILITY ANALYSIS"
log "========================================================================"

# Test with different alpha values using benchmark suite
for alpha in 0.2 0.3 0.4 0.5; do
    log "Testing with α=${alpha} (${alpha}% malicious clients)"
    
    # Note: benchmark suite doesn't support varying alpha yet
    # So we log this for future implementation
    log "⚠️  Scalability test α=${alpha} - requires benchmark suite enhancement"
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

log ""
log "========================================================================"
log "EXPERIMENT SUITE COMPLETE"
log "========================================================================"
log ""
log "Statistics:"
log "  Total experiments: ${TOTAL}"
log "  Passed: ${PASSED}"
log "  Failed: ${FAILED}"
log "  Success rate: $((PASSED * 100 / TOTAL))%"
log ""
log "Output:"
log "  Results directory: ${RESULTS_DIR}/results/"
log "  Logs directory: ${RESULTS_DIR}/logs/"
log "  Main log: ${MAIN_LOG}"
log ""

# List all results
log "Generated result files:"
find "${RESULTS_DIR}/results" -name "*.json" -type f 2>/dev/null | while read f; do
    size=$(ls -lh "$f" | awk '{print $5}')
    log "  - $(basename $f) (${size})"
done

log ""
log "Completed experiments:"
if [ -f "${CHECKPOINT}" ]; then
    cat "${CHECKPOINT}" | nl | while read line; do
        log "  ${line}"
    done
else
    log "  (none)"
fi

log ""
log "========================================================================"

# Exit status
if [ ${FAILED} -eq 0 ]; then
    log "🎉 ALL EXPERIMENTS PASSED!"
    log "========================================================================"
    exit 0
else
    log "⚠️  ${FAILED} experiment(s) failed - check logs above"
    log "========================================================================"
    exit 1
fi
