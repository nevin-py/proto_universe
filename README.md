# ProtoGalaxy: Byzantine-Resilient Federated Learning with ZKP Verification

ProtoGalaxy is a privacy-preserving federated learning system with a 5-layer Byzantine-resilient defense mechanism and zero-knowledge proof verification using ProtoGalaxy IVC.

## Features

- **5-Layer Defense Architecture**
  - L1: Commitment + Merkle Tree
  - L2: Statistical Analysis (z-score, MAD)
  - L3: Robust Aggregation (Trimmed Mean, Multi-Krum, Coordinate-wise Median)
  - L4: Reputation Management
  - L5: ZKP Norm-Bounded Proofs (ProtoGalaxy IVC on BN254/Grumpkin)

- **Zero-Knowledge Proofs**: Bounded gradient sum verification with norm checking
- **Comprehensive Evaluation**: ZKP performance benchmarking, attack rejection tests, scalability analysis
- **Resource Monitoring**: Detailed CPU/GPU/RAM tracking during training

## Installation

### 1. Clone with Submodules

```bash
git clone --recurse-submodules https://github.com/Vanesor/proto_universe.git
cd proto_universe
```

Or if already cloned:
```bash
git submodule update --init --recursive
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in editable mode
pip install -e .
```

### 3. Build ZKP Rust Bindings

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build fl-zkp-bridge
cd sonobe/fl-zkp-bridge
maturin develop --release
cd ../..
```

### 4. Verify Installation

```bash
python -c "from fl_zkp_bridge import FLZKPBoundedProver; print('✓ ZKP module loaded')"
```

## Quick Start

### Run Evaluation Benchmarks

**ZKP Performance:**
```bash
python scripts/run_evaluation.py --mode zkp_performance --trials 3 --output-dir ./eval_results
```

**Attack Rejection:**
```bash
python scripts/run_evaluation.py --mode attack_rejection --trials 3 --output-dir ./eval_results
```

**Baseline FL:**
```bash
python scripts/run_evaluation.py --mode baseline --trials 3 --num-rounds 20 --output-dir ./eval_results
```

**Full Attack Matrix:**
```bash
python scripts/run_evaluation.py --mode attacks --trials 3 --num-rounds 20 --output-dir ./eval_results --resume
```

### Custom Experiment

```bash
python scripts/run_evaluation.py --mode custom \
  --defense protogalaxy_full \
  --attack model_poisoning \
  --byzantine-fraction 0.2 \
  --num-clients 20 \
  --num-galaxies 4 \
  --num-rounds 20 \
  --output-dir ./eval_results
```

## System Requirements

- **Python**: 3.9+
- **CUDA**: 11.0+ (optional, for GPU acceleration)
- **RAM**: 8GB minimum, 16GB recommended
- **Rust**: 1.70+ (for ZKP bindings)

## Updating

```bash
git pull origin main
git submodule update --recursive
```

## Documentation

- [Architecture](protogalaxy_architecture.md) - System design and 5-layer defense
- [FL Pipeline](FL_PIPELINE_COMPLETE.md) - Complete federated learning workflow
- [Sequential Folding](SEQUENTIAL_FOLDING_INTEGRATION_COMPLETE.md) - ZKP folding implementation
- [Evaluation Method](RESEARCH_EVALUATION_METHOD.md) - Benchmarking methodology

## Project Structure

```
proto_universe/
├── src/                        # Core source code
│   ├── aggregators/           # L3: Robust aggregation algorithms
│   ├── client/                # Client-side training
│   ├── crypto/                # L5: ZKP provers and folders
│   ├── defense/               # L2,L4: Statistical analysis, reputation
│   ├── orchestration/         # ProtoGalaxy pipeline coordinator
│   └── simulation/            # Byzantine attack simulators
├── sonobe/                    # ZKP library (submodule - Vanesor/sonobe fork)
│   └── fl-zkp-bridge/        # Rust-Python ZKP bindings
├── scripts/                   # Evaluation and analysis scripts
│   ├── run_evaluation.py     # Main evaluation orchestrator
│   ├── evaluate_zkp_performance.py
│   └── evaluate_attack_rejection.py
├── eval_results/             # Evaluation outputs (JSON + logs)
└── tests/                    # Test suite
```

## Citation

If you use ProtoGalaxy in your research, please cite:

```bibtex
@article{protogalaxy2026,
  title={ProtoGalaxy: Byzantine-Resilient Federated Learning with Zero-Knowledge Proof Verification},
  author={ProtoGalaxy Team},
  year={2026}
}
```

## License

[License details]
