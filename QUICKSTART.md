# ProtoGalaxy Quick Start Guide

Get ProtoGalaxy running in **5 minutes**.

## Prerequisites

- Python 3.9+
- Git
- Rust (for ZKP module) - [install here](https://rustup.rs/)
- CUDA 11.0+ (optional, for GPU)

## Installation

### Option 1: Using Makefile (Recommended)

```bash
# Clone
git clone --recurse-submodules https://github.com/Vanesor/proto_universe.git
cd proto_universe

# Full setup (install, build, test)
make setup
```

That's it! Skip to [Running Evaluations](#running-evaluations).

### Option 2: Manual Setup

```bash
# 1. Clone
git clone --recurse-submodules https://github.com/Vanesor/proto_universe.git
cd proto_universe

# 2. Install Python dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Build ZKP module
cd sonobe/fl-zkp-bridge
maturin develop --release
cd ../..

# 4. Verify
python -c "from fl_zkp_bridge import FLZKPBoundedProver; print('✓ OK')"
```

## Running Evaluations

### Quick Benchmarks

**ZKP Performance (7 architectures, 3 trials, ~13 min):**
```bash
make eval-zkp
# or: python scripts/run_evaluation.py --mode zkp_performance --trials 3
```

**Attack Rejection (6 attacks, 3 trials, ~2 min):**
```bash
make eval-attack
# or: python scripts/run_evaluation.py --mode attack_rejection --trials 3
```

**Baseline FL (9 experiments, ~15 min):**
```bash
make eval-baseline
# or: python scripts/run_evaluation.py --mode baseline --trials 3 --num-rounds 20
```

### Custom Experiment

```bash
python scripts/run_evaluation.py --mode custom \
  --defense protogalaxy_full \
  --attack model_poisoning \
  --byzantine-fraction 0.2 \
  --num-clients 20 \
  --num-rounds 20 \
  --verbose
```

### Full Evaluation Suite

**⚠️ Warning: Takes 8-12 hours**

```bash
make eval-attacks      # 135 experiments
make eval-ablation     # 36 experiments  
make eval-scalability  # 135 experiments
```

## View Results

Results are saved to `eval_results/`:

```
eval_results/
├── baseline/          # JSON files per experiment
├── attacks/
├── zkp_performance/
├── attack_rejection/
├── logs/              # Detailed logs
└── resource_usage_report.txt
```

**View summary:**
```bash
cat eval_results/resource_usage_report.txt
```

**Analyze results:**
```python
import json
with open('eval_results/zkp_performance/zkp_performance_results.json') as f:
    results = json.load(f)
print(results)
```

## Common Commands

```bash
make test              # Run all tests
make test-zkp          # Run ZKP tests only
make format            # Format code
make clean             # Clean build artifacts
make help              # Show all commands
```

## Troubleshooting

**ZKP module not found:**
```bash
cd sonobe/fl-zkp-bridge
maturin develop --release
cd ../..
```

**Submodule empty:**
```bash
git submodule update --init --recursive
```

**Out of memory:**
Reduce batch size or number of clients in config.

**GPU not detected:**
Check CUDA installation: `nvidia-smi`

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow
- See evaluation results in `eval_results/`
- Explore code in `src/` and `scripts/`

## Questions?

Open an issue or ask the team!
