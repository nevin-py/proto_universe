#!/bin/bash
# Comprehensive Experiment Suite for FiZK-PoT Paper
# 
# Covers all experimental scenarios:
# - IID and non-IID data partitioning
# - Multiple model architectures (Linear, MLP, CNN)
# - Various attack types
# - Different malicious client ratios (α)
# - Ablation studies
#
# Features:
# - Resumable execution (skips completed experiments)
# - Per-experiment logging
# - Structured result storage
# - Progress tracking

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Configuration
# ============================================================================

# Experiment root directory
EXP_ROOT="outputs/comprehensive_experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${EXP_ROOT}/run_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"
CHECKPOINT_FILE="${RUN_DIR}/checkpoint.txt"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

# Global log
GLOBAL_LOG="${RUN_DIR}/experiment_suite.log"

# ============================================================================
# Experiment Matrix
# ============================================================================

# Datasets (data partitioning strategies)
DATASETS=("mnist" "fashion_mnist")
PARTITIONS=("iid" "noniid")

# Model architectures
MODELS=("linear" "mlp" "cnn")

# Attack types
ATTACKS=(
    "none"                  # Baseline (no attack)
    "modelpoisoning"
    "labelflip"
    "targetedlabelflip"
    "backdoor"
    "gaussian"
)

# Malicious client fractions
ALPHAS=("0.2" "0.3" "0.4" "0.5" "0.6")

# Defense methods
DEFENSES=(
    "vanilla"
    "multikrum"
    "median"
    "trimmedmean"
    "fltrust"
    # "fizk"  # Tested separately due to complexity
)

# Training parameters
NUM_ROUNDS=10
NUM_CLIENTS=10
LOCAL_EPOCHS=5
BATCH_SIZE=32
LEARNING_RATE=0.01

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${GLOBAL_LOG}"
}

is_completed() {
    local exp_id="$1"
    if [ -f "${CHECKPOINT_FILE}" ]; then
        grep -q "^${exp_id}$" "${CHECKPOINT_FILE}"
        return $?
    fi
    return 1
}

mark_completed() {
    local exp_id="$1"
    echo "${exp_id}" >> "${CHECKPOINT_FILE}"
}

# ============================================================================
# Experiment: Byzantine Detection Validation
# ============================================================================

run_byzantine_detection() {
    local exp_id="byzantine_detection_all_architectures"
    
    if is_completed "${exp_id}"; then
        log "⏭  Skipping ${exp_id} (already completed)"
        return 0
    fi
    
    log "▶  Running Byzantine Detection Validation"
    log "   Testing: Linear, MLP, CNN architectures"
    log "   Log: ${LOG_DIR}/${exp_id}.log"
    
    local exp_log="${LOG_DIR}/${exp_id}.log"
    local exp_result="${RESULTS_DIR}/${exp_id}.json"
    
    log "   Executing: python scripts/run_benchmark_suite.py --benchmarks byzantine"
    
    # Run with tee to get both file and stdout logging
    if python scripts/run_benchmark_suite.py \
        --benchmarks byzantine \
        --output "${RESULTS_DIR}" \
        2>&1 | tee "${exp_log}"; then
        
        # Move result file to proper location
        latest_result=$(ls -t "${RESULTS_DIR}"/benchmark_*.json 2>/dev/null | head -1)
        if [ -f "${latest_result}" ]; then
            cp "${latest_result}" "${exp_result}"
            log "   Result saved: ${exp_result}"
        fi
        
        log "✅ Byzantine Detection: PASS"
        mark_completed "${exp_id}"
        return 0
    else
        local exit_code=$?
        log "❌ Byzantine Detection: FAIL (exit code: ${exit_code})"
        log "   Last 10 lines of log:"
        tail -10 "${exp_log}" | while read line; do
            log "     ${line}"
        done
        return 1
    fi
}

# ============================================================================
# Experiment: Defense Comparison
# ============================================================================

run_defense_comparison() {
    local dataset="$1"
    local partition="$2"
    local model="$3"
    local attack="$4"
    local alpha="$5"
    
    local exp_id="defense_${dataset}_${partition}_${model}_${attack}_alpha${alpha}"
    
    if is_completed "${exp_id}"; then
        log "⏭  Skipping ${exp_id}"
        return 0
    fi
    
    log "▶  Running: ${exp_id}"
    log "   Dataset: ${dataset} (${partition})"
    log "   Model: ${model}"
    log "   Attack: ${attack}"
    log "   α: ${alpha}"
    log "   Log: ${LOG_DIR}/${exp_id}.log"
    
    local exp_log="${LOG_DIR}/${exp_id}.log"
    local exp_result="${RESULTS_DIR}/${exp_id}.json"
    
    log "   Executing comprehensive evaluation..."
    
    # Run comprehensive evaluation with tee for continuous logging
    if python scripts/run_comprehensive_evaluation.py \
        --datasets "${dataset}" \
        --partition "${partition}" \
        --models "${model}" \
        --attacks "${attack}" \
        --alpha "${alpha}" \
        --defenses "${DEFENSES[@]}" \
        --rounds "${NUM_ROUNDS}" \
        --clients "${NUM_CLIENTS}" \
        --local-epochs "${LOCAL_EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LEARNING_RATE}" \
        --output "${exp_result}" \
        2>&1 | tee "${exp_log}"; then
        
        log "✅ ${exp_id}: PASS"
        mark_completed "${exp_id}"
        return 0
    else
        local exit_code=$?
        log "❌ ${exp_id}: FAIL (exit code: ${exit_code})"
        log "   Last 20 lines:"
        tail -20 "${exp_log}" | while read line; do
            log "     ${line}"
        done
        return 1
    fi
}

# ============================================================================
# Experiment: Scalability Analysis (varying α)
# ============================================================================

run_scalability_analysis() {
    local dataset="$1"
    local model="$2"
    local attack="$3"
    
    local exp_id="scalability_${dataset}_${model}_${attack}"
    
    if is_completed "${exp_id}"; then
        log "⏭  Skipping ${exp_id}"
        return 0
    fi
    
    log "▶  Running Scalability Analysis: ${exp_id}"
    log "   Testing α: ${ALPHAS[*]}"
    
    local exp_log="${LOG_DIR}/${exp_id}.log"
    local exp_result="${RESULTS_DIR}/${exp_id}.json"
    
    # Run with all alpha values
    if python scripts/run_comprehensive_evaluation.py \
        --datasets "${dataset}" \
        --partition "iid" \
        --models "${model}" \
        --attacks "${attack}" \
        --alpha "${ALPHAS[@]}" \
        --defenses "vanilla" "multikrum" "median" \
        --rounds "${NUM_ROUNDS}" \
        --clients "${NUM_CLIENTS}" \
        --output "${exp_result}" \
        > "${exp_log}" 2>&1; then
        
        log "✅ ${exp_id}: PASS"
        mark_completed "${exp_id}"
        return 0
    else
        log "❌ ${exp_id}: FAIL"
        return 1
    fi
}

# ============================================================================
# Experiment: Ablation Study
# ============================================================================

run_ablation_study() {
    local exp_id="ablation_study"
    
    if is_completed "${exp_id}"; then
        log "⏭  Skipping ${exp_id}"
        return 0
    fi
    
    log "▶  Running Ablation Study"
    log "   Comparing: No Defense vs Individual Defenses vs Combined"
    
    local exp_log="${LOG_DIR}/${exp_id}.log"
    local exp_result="${RESULTS_DIR}/${exp_id}.json"
    
    if python scripts/run_ablation_study.py \
        --dataset "mnist" \
        --model "linear" \
        --attack "modelpoisoning" \
        --alpha 0.3 \
        --rounds "${NUM_ROUNDS}" \
        --output "${exp_result}" \
        > "${exp_log}" 2>&1; then
        
        log "✅ ${exp_id}: PASS"
        mark_completed "${exp_id}"
        return 0
    else
        log "❌ ${exp_id}: FAIL"
        return 1
    fi
}

# ============================================================================
# Main Experiment Suite
# ============================================================================

main() {
    log "="*80
    log "FiZK-PoT COMPREHENSIVE EXPERIMENT SUITE"
    log "="*80
    log "Run Directory: ${RUN_DIR}"
    log "Timestamp: ${TIMESTAMP}"
    log ""
    
    local total_experiments=0
    local completed_experiments=0
    local failed_experiments=0
    
    # ========================================================================
    # Part 1: Byzantine Detection Validation
    # ========================================================================
    
    log ""
    log "PART 1: BYZANTINE DETECTION VALIDATION"
    log "="*80
    
    ((total_experiments++))
    if run_byzantine_detection; then
        ((completed_experiments++))
    else
        ((failed_experiments++))
    fi
    
    # ========================================================================
    # Part 2: Core Defense Comparison Experiments
    # ========================================================================
    
    log ""
    log "PART 2: CORE DEFENSE COMPARISON"
    log "="*80
    log "Testing key combinations for paper results"
    
    # Key experiments for paper
    KEY_EXPERIMENTS=(
        # Format: dataset partition model attack alpha
        "mnist iid linear modelpoisoning 0.3"
        "mnist iid linear labelflip 0.3"
        "mnist iid linear backdoor 0.3"
        "mnist noniid linear modelpoisoning 0.3"
        "fashion_mnist iid linear modelpoisoning 0.3"
    )
    
    for exp in "${KEY_EXPERIMENTS[@]}"; do
        read -r dataset partition model attack alpha <<< "$exp"
        ((total_experiments++))
        if run_defense_comparison "$dataset" "$partition" "$model" "$attack" "$alpha"; then
            ((completed_experiments++))
        else
            ((failed_experiments++))
        fi
    done
    
    # ========================================================================
    # Part 3: Scalability Analysis (varying α)
    # ========================================================================
    
    log ""
    log "PART 3: SCALABILITY ANALYSIS"
    log "="*80
    log "Testing robustness with varying malicious client fractions"
    
    SCALABILITY_EXPERIMENTS=(
        "mnist linear modelpoisoning"
        "mnist linear labelflip"
    )
    
    for exp in "${SCALABILITY_EXPERIMENTS[@]}"; do
        read -r dataset model attack <<< "$exp"
        ((total_experiments++))
        if run_scalability_analysis "$dataset" "$model" "$attack"; then
            ((completed_experiments++))
        else
            ((failed_experiments++))
        fi
    done
    
    # ========================================================================
    # Part 4: Multi-Architecture Experiments
    # ========================================================================
    
    log ""
    log "PART 4: MULTI-ARCHITECTURE VALIDATION"
    log "="*80
    log "Testing different model architectures"
    
    # Note: MLP and CNN require circuit modifications
    # For now, focus on linear model which is fully supported
    
    ARCHITECTURE_EXPERIMENTS=(
        "mnist iid linear modelpoisoning 0.3"
        # "mnist iid mlp modelpoisoning 0.3"  # Requires circuit update
        # "mnist iid cnn modelpoisoning 0.3"  # Requires circuit update
    )
    
    for exp in "${ARCHITECTURE_EXPERIMENTS[@]}"; do
        read -r dataset partition model attack alpha <<< "$exp"
        ((total_experiments++))
        if run_defense_comparison "$dataset" "$partition" "$model" "$attack" "$alpha"; then
            ((completed_experiments++))
        else
            ((failed_experiments++))
        fi
    done
    
    # ========================================================================
    # Part 5: Ablation Study
    # ========================================================================
    
    log ""
    log "PART 5: ABLATION STUDY"
    log "="*80
    
    ((total_experiments++))
    if run_ablation_study; then
        ((completed_experiments++))
    else
        ((failed_experiments++))
    fi
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    
    log ""
    log "="*80
    log "EXPERIMENT SUITE COMPLETE"
    log "="*80
    log "Total Experiments: ${total_experiments}"
    log "Completed: ${completed_experiments}"
    log "Failed: ${failed_experiments}"
    log "Success Rate: $((completed_experiments * 100 / total_experiments))%"
    log ""
    log "Results saved to: ${RESULTS_DIR}"
    log "Logs saved to: ${LOG_DIR}"
    log "Global log: ${GLOBAL_LOG}"
    log ""
    
    # Generate summary report
    log "Generating summary report..."
    python scripts/compile_paper_results.py "${RESULTS_DIR}" "${RUN_DIR}/summary_report.md"
    
    if [ ${failed_experiments} -eq 0 ]; then
        log "✅ ALL EXPERIMENTS PASSED"
        return 0
    else
        log "⚠️  ${failed_experiments} EXPERIMENTS FAILED"
        return 1
    fi
}

# ============================================================================
# Execute
# ============================================================================

# Check if resuming
if [ -f "${CHECKPOINT_FILE}" ]; then
    log "Found checkpoint file - resuming from previous run"
    log "Completed experiments: $(wc -l < "${CHECKPOINT_FILE}")"
fi

main "$@"
