# FiZK-PoT: Federated Learning with Zero-Knowledge Proof-of-Training

Byzantine-robust federated learning using Zero-Knowledge Proofs and ProtoGalaxy IVC.

## Project Structure

```
proto_universe/
├── src/                          # Core implementation
│   ├── client/                   # FL client training logic
│   ├── crypto/                   # ZKP integration (ProtoGalaxy)
│   ├── defense/                  # Byzantine defenses (Multi-Krum, FLTrust, etc.)
│   ├── models/                   # Neural network models (Linear, MLP, CNN)
│   └── orchestration/            # FL pipeline orchestration
├── scripts/                      # Evaluation and benchmark scripts
│   ├── run_comprehensive_evaluation.py  # Main evaluation script
│   ├── run_all_byzantine_tests.sh       # Byzantine detection tests
│   ├── test_byzantine_*.py              # Architecture-specific tests
│   └── analyze_results.py               # Result analysis
├── sonobe/                       # Rust ZKP backend (ProtoGalaxy)
│   └── fl-zkp-bridge/           # Python-Rust bridge
├── outputs/                      # Experiment results
├── old/                          # Archived test files
├── paper.tex                     # LaTeX paper
└── config.yaml                   # FL configuration
```

## Quick Start

### 1. Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build Rust ZKP bridge
cd sonobe/fl-zkp-bridge
maturin develop --release
cd ../..
```

### 2. Run Byzantine Detection Tests

Test multi-architecture Byzantine detection (Linear, MLP, CNN):

```bash
./scripts/run_all_byzantine_tests.sh
```

Expected output: **100% Byzantine detection across all architectures**

### 3. Run Comprehensive Evaluation

Full evaluation with multiple baselines, attacks, and datasets:

```bash
python scripts/run_comprehensive_evaluation.py \
    --datasets mnist fashion_mnist \
    --baselines vanilla multikrum median trimmedmean fltrust fizk \
    --attacks modelpoisoning labelflip backdoor \
    --alpha 0.2 0.3 0.4 \
    --rounds 10 \
    --clients 10 \
    --output outputs/comprehensive_results.json
```

### 4. Generate Paper Results

```bash
# Generate tables and figures
python scripts/generate_paper_tables.py outputs/comprehensive_results.json

# Compile LaTeX paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Key Features

### Multi-Architecture Support

- **Linear Regression**: 10×784 (MNIST flattened)
- **MLP**: 10×64 (fc3 output layer)
- **CNN**: 10×128 (fc2 output layer)

All architectures achieve **100% Byzantine detection** via external fingerprint verification.

### Byzantine Detection Mechanism

1. **Fingerprint Computation**: `fp = Σ r[k] × (b[k] + Σ W[k,j])`
2. **External Verification**: Recompute fingerprint from client weights before proof generation
3. **Immediate Rejection**: Mismatch → Byzantine detected (<1ms)
4. **ZK Proof**: Honest clients generate ProtoGalaxy IVC proof

### Supported Defenses

- **Vanilla FL**: No defense (baseline)
- **Multi-Krum**: Select k closest gradients
- **Coordinate-Wise Median**: Element-wise median aggregation
- **Trimmed Mean**: Remove extreme values before averaging
- **FLTrust**: Server-validated trust scores
- **FiZK-PoT**: Zero-knowledge proof-of-training (our method)

### Attack Types

- Model Poisoning
- Label Flipping
- Targeted Label Flipping
- Backdoor Attacks
- Gaussian Noise
- Gradient Substitution
- Adaptive Attacks

## Benchmark Results

### Byzantine Detection (All Architectures)

| Architecture | Malicious Detected | False Positives | Proof Size |
|--------------|-------------------|-----------------|------------|
| Linear 10×784| 100%              | 0%              | 4.5 MB     |
| MLP 10×64    | 100%              | 0%              | N/A        |
| CNN 10×128   | 100%              | 0%              | N/A        |

**Detection Time**: <1ms (fingerprint verification)  
**Security**: Schwartz-Zippel with 2^(-100) collision probability

### Accuracy Under Attack (α=0.3, MNIST)

| Defense      | Clean Acc | Under Attack | Overhead  |
|--------------|-----------|--------------|-----------|
| Vanilla      | 92.5%     | 23.4%        | 0ms       |
| Multi-Krum   | 91.8%     | 78.2%        | ~5ms      |
| FLTrust      | 92.1%     | 81.5%        | ~10ms     |
| **FiZK-PoT** | **92.3%** | **91.9%**    | ~25s      |

## Configuration

Edit `config.yaml`:

```yaml
federated_learning:
  num_clients: 10
  num_rounds: 20
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.01

byzantine:
  alpha: 0.3  # Fraction of malicious clients
  attack_type: "model_poisoning"

zkp:
  batch_size: 8  # Samples per proof
  enable_real_proofs: true
```

## Architecture Notes

### ProtoGalaxy Preprocessing Limitation

ProtoGalaxy IVC preprocessing locks constraint count based on circuit dimensions. Each architecture (10×784, 10×64, 10×128) requires separate preprocessing.

**Impact**: Cannot reuse same prover instance across different dimensions.

**Solution**: Create separate prover per architecture or preprocess multiple keys.

**Byzantine Detection**: Unaffected - happens before ProtoGalaxy preprocessing.

## Citation

```bibtex
@article{fizk-pot2024,
  title={FiZK-PoT: Byzantine-Robust Federated Learning via Zero-Knowledge Proof-of-Training},
  author={...},
  year={2024}
}
```

## License

See LICENSE file.
