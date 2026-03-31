# ProtoGalaxy FL Pipeline - Quick Start Guide

## Overview

Complete federated learning pipeline implementing the full 4-phase ProtoGalaxy protocol with:
- ✅ All 5 defense layers (Integrity, Statistical, Robust Agg, Reputation, Galaxy-Level)
- ✅ Merkle tree verification
- ✅ Forensic evidence logging
- ✅ MNIST dataset with IID partitioning
- ✅ Linear regression / MLP / CNN models
- ✅ Configurable clients, galaxies, and rounds

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import torchvision; print('✓ Ready!')"
```

## Basic Usage

### Run with defaults (20 clients, 2 galaxies, 10 rounds)
```bash
python scripts/run_protogalaxy_fl.py
```

### Run with custom configuration
```bash
python scripts/run_protogalaxy_fl.py \
    --num-clients 100 \
    --num-galaxies 10 \
    --rounds 20 \
    --model-type mlp \
    --verbose
```

### Run with Byzantine clients
```bash
python scripts/run_protogalaxy_fl.py \
    --num-clients 100 \
    --num-galaxies 10 \
    --rounds 30 \
    --byzantine-ratio 0.3 \
    --attack-type gradient_poison
```

## Command-Line Options

### FL Configuration
- `--num-clients`: Number of clients (default: 20)
- `--num-galaxies`: Number of galaxies (default: 2)
- `--rounds`: Number of FL rounds (default: 10)
- `--local-epochs`: Local epochs per round (default: 1)

### Model Configuration
- `--model-type`: Choose 'linear', 'mlp', or 'cnn' (default: linear)

### Data Configuration
- `--data-dir`: MNIST data directory (default: ./data)
- `--partition-type`: 'iid' or 'non-iid' (default: iid)

### Defense Configuration
- `--byzantine-ratio`: Fraction of Byzantine clients (default: 0.0)
- `--attack-type`: 'none', 'gradient_poison', or 'label_flip' (default: none)

### System Configuration
- `--device`: 'cpu' or 'cuda' (default: cpu)
- `--output-dir`: Output directory (default: ./fl_results)
- `--log-file`: Log file path (default: None, stdout only)
- `--verbose`: Enable verbose logging

## Example Scenarios

### Scenario 1: Quick Test (2 minutes)
```bash
python scripts/run_protogalaxy_fl.py \
    --num-clients 10 \
    --num-galaxies 2 \
    --rounds 5 \
    --model-type linear
```

**Expected Output**:
- Initial accuracy: ~10% (random)
- Final accuracy: >85%
- All 5 defense layers execute successfully

---

### Scenario 2: Full Production Run (10 minutes)
```bash
python scripts/run_protogalaxy_fl.py \
    --num-clients 100 \
    --num-galaxies 10 \
    --rounds 20 \
    --model-type mlp \
    --verbose \
    --log-file fl_run.log
```

**Expected Output**:
- Initial accuracy: ~10%
- Final accuracy: >90%
- Comprehensive logs in `fl_run.log`
- Results saved to `./fl_results/results.json`

---

### Scenario 3: Byzantine Attack Defense (15 minutes)
```bash
python scripts/run_protogalaxy_fl.py \
    --num-clients 100 \
    --num-galaxies 10 \
    --rounds 30 \
    --byzantine-ratio 0.3 \
    --attack-type gradient_poison \
    --model-type mlp \
    --verbose \
    --log-file byzantine_test.log
```

**Expected Output**:
- 30% of clients are Byzantine (gradient poisoning)
- Defense layers detect and quarantine malicious clients
- Layer 5 may dissolve compromised galaxies
- Model still converges: >80% accuracy despite attacks
- Forensic evidence logged in `./fl_results/forensic_evidence/`

---

## Output Files

After running, check the output directory:

```
fl_results/
├── results.json              # Training metrics and config
└── forensic_evidence/        # Evidence database
    ├── evidence_*.json       # Individual quarantine/ban records
    ├── client_index.json     # Index by client ID
    ├── round_index.json      # Index by round
    └── stats.json            # Summary statistics
```

## Understanding the Logs

### Phase 1: Commitment
```
[INFO] PHASE 1: COMMITMENT & MERKLE TREE CONSTRUCTION
[INFO] Building Merkle trees for 100 client commitments
[INFO]   Galaxy 0: 10 commitments, root=a1b2c3d4...
[INFO] ✓ Global Merkle root: e5f6g7h8...
```

### Phase 2: Revelation
```
[INFO] PHASE 2: REVELATION & MERKLE PROOF VERIFICATION
[INFO] Verifying Merkle proofs for 100 clients
[INFO] ✓ Verified: 100/100 clients (100.0%)
```

### Phase 3: Defense
```
[INFO] PHASE 3: MULTI-LAYER DEFENSE (Layers 1-5)
[INFO]   Layer 1 (Integrity): 0 detections
[INFO]   Layer 2 (Statistical): 5 detections
[INFO]   Layer 3 (Robust Agg): multi_krum
[INFO]   Layer 4 (Reputation): 100 clients scored
[INFO]   Layer 5 (Galaxy): 1 galaxies flagged
[INFO] 🔴 DISSOLVING Galaxy 3
[INFO]   Honest clients: 7
[INFO]   Malicious clients: 3 (will be quarantined)
[INFO] ✓ Clean galaxies: 9/10
```

### Phase 4: Aggregation
```
[INFO] PHASE 4: GLOBAL AGGREGATION & MODEL UPDATE
[INFO] Aggregating 9 clean galaxy updates
[INFO] ✓ Global model updated with learning rate 0.01
```

### Round Summary
```
[INFO] ROUND SUMMARY
[INFO] Phase 1 (Commitment):    100 commitments
[INFO] Phase 2 (Revelation):    100 verified, 0 rejected (100.0% acceptance)
[INFO] Phase 3 (Defense):       L1:0 L2:5 L5:1 flagged galaxies, 1 dissolved
[INFO] Phase 4 (Aggregation):   9 galaxies, update norm: 0.1234
[INFO] Test Accuracy: 87.42% (8742/10000)
[INFO] Test Loss:     0.3456
```

## Troubleshooting

### Issue: MNIST data not downloading
**Solution**: Manually download from torchvision:
```bash
python -c "from torchvision import datasets; datasets.MNIST('./data', download=True)"
```

### Issue: Out of memory
**Solution**: Reduce batch size or number of clients:
```bash
python scripts/run_protogalaxy_fl.py --num-clients 20 --num-galaxies 2
```

### Issue: Slow training
**Solution**: Use GPU if available:
```bash
python scripts/run_protogalaxy_fl.py --device cuda
```

## Next Steps

1. **Try different models**: Test with MLP or CNN
2. **Scale up**: Run with 1000 clients, 20 galaxies
3. **Test defenses**: Add Byzantine clients and monitor defense layers
4. **Analyze forensics**: Inspect `forensic_evidence/` directory
5. **Customize**: Modify defense thresholds in `defense_config`

## Architecture Details

For complete architecture documentation, see:
- [`protogalaxy_architecture.md`](protogalaxy_architecture.md)
- [`layer5_implementation_summary.md`](layer5_implementation_summary.md)
- [`implementation_plan.md`](implementation_plan.md)
