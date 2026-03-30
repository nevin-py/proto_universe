# Comprehensive Experiments Guide

## Quick Start

Run the full experiment suite:

```bash
./scripts/run_comprehensive_experiments.sh
```

The script is **resumable** - if interrupted, simply run it again and it will skip completed experiments.

## Experiment Structure

### Part 1: Byzantine Detection Validation
- **Goal:** Validate 100% Byzantine detection across architectures
- **Models:** Linear (10×784), MLP (10×64), CNN (10×128)
- **Metric:** Detection rate, detection time
- **Expected Result:** 100% detection in <1ms

### Part 2: Core Defense Comparison
- **Goal:** Compare defense mechanisms under attacks
- **Configurations:**
  - MNIST + IID + Linear + Model Poisoning (α=0.3)
  - MNIST + IID + Linear + Label Flip (α=0.3)
  - MNIST + IID + Linear + Backdoor (α=0.3)
  - MNIST + non-IID + Linear + Model Poisoning (α=0.3)
  - Fashion-MNIST + IID + Linear + Model Poisoning (α=0.3)
- **Defenses:** Vanilla, Multi-Krum, Median, Trimmed Mean, FLTrust
- **Metrics:** Accuracy, convergence, overhead

### Part 3: Scalability Analysis
- **Goal:** Test robustness with varying malicious fractions
- **Alpha values:** 0.0, 0.2, 0.3, 0.4, 0.5
- **Attacks:** Model Poisoning, Label Flip
- **Metric:** Accuracy vs α curve

### Part 4: Multi-Architecture Validation
- **Goal:** Show Byzantine detection works across architectures
- **Models:** Linear, MLP (future), CNN (future)
- **Note:** MLP/CNN require separate circuit preprocessing

### Part 5: Ablation Study
- **Goal:** Analyze contribution of each defense component
- **Variants:** No defense, individual defenses, combined
- **Metric:** Attack success rate reduction

## Output Structure

```
outputs/comprehensive_experiments/run_YYYYMMDD_HHMMSS/
├── experiment_suite.log          # Global log
├── checkpoint.txt                 # Completed experiments
├── logs/                          # Per-experiment logs
│   ├── byzantine_detection_all_architectures.log
│   ├── defense_mnist_iid_linear_modelpoisoning_alpha0.3.log
│   └── ...
├── results/                       # JSON results
│   ├── byzantine_detection_all_architectures.json
│   ├── defense_mnist_iid_linear_modelpoisoning_alpha0.3.json
│   └── ...
└── summary_report.md              # Compiled summary
```

## Resuming Interrupted Runs

The script maintains a checkpoint file tracking completed experiments:

1. **First run:** Creates `checkpoint.txt` and logs each completed experiment
2. **Interrupted:** Press Ctrl+C or let it fail
3. **Resume:** Run the script again - it automatically skips completed experiments

```bash
# Check progress
cat outputs/comprehensive_experiments/run_*/checkpoint.txt | wc -l

# Resume
./scripts/run_comprehensive_experiments.sh
```

## Experiment Configurations

### Full Matrix (for reference)

**Datasets:** MNIST, Fashion-MNIST
**Partitions:** IID, non-IID
**Models:** Linear, MLP, CNN
**Attacks:** None, Model Poisoning, Label Flip, Targeted Label Flip, Backdoor, Gaussian
**Alpha values:** 0.0, 0.2, 0.3, 0.4, 0.5
**Defenses:** Vanilla, Multi-Krum, Median, Trimmed Mean, FLTrust, FiZK-PoT

**Total possible combinations:** ~600+

**Selected for paper:** ~15 key experiments covering main scenarios

## Running Individual Experiments

### Byzantine Detection Only

```bash
python scripts/run_benchmark_suite.py \
    --benchmarks byzantine \
    --output outputs/byzantine_only
```

### Specific Defense Comparison

```bash
python scripts/run_comprehensive_evaluation.py \
    --datasets mnist \
    --partition iid \
    --models linear \
    --attacks modelpoisoning \
    --alpha 0.3 \
    --defenses vanilla multikrum median \
    --rounds 10 \
    --clients 10 \
    --output outputs/custom_experiment.json
```

### Ablation Study Only

```bash
python scripts/run_ablation_study.py \
    --dataset mnist \
    --model linear \
    --attack modelpoisoning \
    --alpha 0.3 \
    --output outputs/ablation.json
```

## Expected Runtime

- **Byzantine Detection:** ~5 seconds
- **Single Defense Comparison:** ~1-2 minutes
- **Scalability Analysis (5 α values):** ~5-10 minutes
- **Full Suite (~15 experiments):** ~20-30 minutes

**With ZKP proofs enabled:** Add ~20-30s per experiment for proof generation

## Results Analysis

After experiments complete, generate paper-ready outputs:

```bash
# Compile results into tables
python scripts/compile_paper_results.py \
    outputs/comprehensive_experiments/run_*/results \
    paper_results/

# Generate figures
python scripts/plot_comprehensive_results.py \
    outputs/comprehensive_experiments/run_*/results \
    paper_figures/
```

## Troubleshooting

### Experiment Fails

Check the specific experiment log:
```bash
tail -50 outputs/comprehensive_experiments/run_*/logs/EXPERIMENT_NAME.log
```

### Out of Memory

Reduce batch size or number of clients:
```bash
# Edit run_comprehensive_experiments.sh
NUM_CLIENTS=5  # Reduce from 10
BATCH_SIZE=16  # Reduce from 32
```

### Skip Failed Experiments

Failed experiments are logged but don't stop the suite. To retry only failed:

```bash
# Remove checkpoint for failed experiment
grep -v "failed_experiment_id" checkpoint.txt > checkpoint_new.txt
mv checkpoint_new.txt checkpoint.txt

# Re-run
./scripts/run_comprehensive_experiments.sh
```

## Key Metrics Tracked

Each experiment logs:
- **Accuracy:** Test accuracy per round
- **Convergence:** Rounds to reach threshold
- **Detection Rate:** % of malicious clients caught
- **False Positives:** % of honest clients rejected  
- **Runtime:** Training time, proof time, verification time
- **Proof Size:** Bytes (for ZKP experiments)

## Paper Integration

Results are structured to directly populate paper tables:

- **Table 1:** Byzantine detection (all architectures)
- **Table 2:** Defense comparison (MNIST, α=0.3)
- **Table 3:** Scalability analysis (varying α)
- **Figure 1:** Accuracy vs rounds (multiple defenses)
- **Figure 2:** Detection rate vs α
- **Figure 3:** Ablation study breakdown

Generated files can be directly imported into LaTeX:
```latex
\input{paper_results/table_defense_comparison.tex}
```
