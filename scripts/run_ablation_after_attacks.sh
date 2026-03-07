#!/bin/bash

set -e

LOG_DIR="eval_results/logs"
mkdir -p "$LOG_DIR"

echo "=================================="
echo "Ablation Study Scheduler"
echo "=================================="
echo "Start time: $(date)"
echo ""

echo "▶ Waiting for attacks benchmark to finish..."
echo "  Checking for process: run_evaluation.py --mode attacks"
echo ""

while pgrep -f "run_evaluation.py.*--mode attacks" > /dev/null; do
    sleep 60  # Check every minute
    echo "  [$(date +%H:%M:%S)] Attacks still running... waiting"
done

echo ""
echo ":) Attacks benchmark completed at $(date)"
echo ""

echo ""
echo ":) Wait complete at $(date)"
echo ""

echo "=================================="
echo "ABLATION STUDY 1/2: MNIST"
echo "=================================="
echo "Tests ZKP contribution by comparing:"
echo "  - merkle_only: Layers 1-4 only (no ZKP)"
echo "  - zk_merkle:  Full 5-layer defense (with ZKP)"
echo ""
echo "Configuration:"
echo "  Dataset:     mnist"
echo "  Model:       linear"
echo "  Clients:     20"
echo "  Galaxies:    4"
echo "  Attacks:     label_flip, model_poisoning, backdoor"
echo "  Byzantine:   20%, 30%"
echo "  Local epochs: 1"
echo "  Rounds:      20"
echo "  Trials:      3"
echo "  Total:       36 experiments"
echo "  Time:        ~1-2 hours"
echo "=================================="
echo ""

python scripts/run_evaluation.py \
    --mode ablation \
    --dataset mnist \
    --model-type linear \
    --trials 3 \
    --num-rounds 20 \
    --local-epochs 1 \
    --num-clients 20 \
    --num-galaxies 4 \
    --output-dir ./eval_results \
    --resume \
    --verbose \
    > "$LOG_DIR/ablation_mnist.log" 2>&1

echo ""
echo ":) MNIST ablation complete at $(date)"
echo ""

echo "=================================="
echo "ABLATION STUDY 2/2: CIFAR-10"
echo "=================================="
echo "Configuration:"
echo "  Dataset:     cifar10"
echo "  Model:       cnn"
echo "  Clients:     20"
echo "  Galaxies:    4"
echo "  Attacks:     label_flip, model_poisoning, backdoor"
echo "  Byzantine:   20%, 30%"
echo "  Local epochs: 1"
echo "  Rounds:      20"
echo "  Trials:      3"
echo "  Total:       36 experiments"
echo "  Time:        ~2-3 hours"
echo "=================================="
echo ""

python scripts/run_evaluation.py \
    --mode ablation \
    --dataset cifar10 \
    --model-type cnn \
    --trials 3 \
    --num-rounds 20 \
    --local-epochs 1 \
    --num-clients 20 \
    --num-galaxies 4 \
    --output-dir ./eval_results \
    --resume \
    --verbose \
    > "$LOG_DIR/ablation_cifar10.log" 2>&1

echo ""
echo ":) CIFAR-10 ablation complete at $(date)"
echo ""

echo "=================================="
echo "ALL ABLATION STUDIES COMPLETE!"
echo "=================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  eval_results/ablation/"
echo ""
echo "Logs:"
echo "  MNIST:    $LOG_DIR/ablation_mnist.log"
echo "  CIFAR-10: $LOG_DIR/ablation_cifar10.log"
echo ""
echo "Experiments per dataset: 36"
echo "Total experiments:       72"
echo ""
echo "View results:"
echo "  ls -lh eval_results/ablation/*mnist*.json | wc -l  # Should be 36"
echo "  ls -lh eval_results/ablation/*cifar10*.json | wc -l  # Should be 36"
echo "  cat eval_results/resource_usage_report.txt"
