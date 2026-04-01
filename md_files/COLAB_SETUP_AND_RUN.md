# Google Colab Setup and Run Guide

This guide is tailored for Google Colab and provides notebook-cell-friendly commands to set up dependencies, build the Rust bridge, run required tests, and execute experiments.

## Colab Runtime Recommendations

- Runtime type: Python 3
- Hardware accelerator: CPU is fine for sanity checks; GPU optional for faster training
- Keep all commands in the same notebook session

---

## Cell 1: (Optional) Mount Google Drive for persistent outputs

```python
from google.colab import drive
drive.mount('/content/drive')
```

If using Drive persistence, set an output root in later commands like:

```python
OUTPUT_ROOT = "/content/drive/MyDrive/proto_system_outputs"
```

---

## Cell 2: Clone repo with submodules

```bash
%%bash
set -e
cd /content
rm -rf proto_universe
git clone --recurse-submodules https://github.com/nevin-py/proto_universe.git
cd proto_universe/proto_system
git submodule sync --recursive
git submodule update --init --recursive
```

---

## Cell 3: System dependencies (Rust toolchain + build tools)

```bash
%%bash
set -e
apt-get update
apt-get install -y build-essential pkg-config libssl-dev curl

if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
source "$HOME/.cargo/env"
cargo --version
```

---

## Cell 4: Python dependencies

```bash
%%bash
set -e
cd /content/proto_universe/proto_system
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---

## Cell 5: Build Rust bridge for Python (`fl_zkp_bridge`)

```bash
%%bash
set -e
source "$HOME/.cargo/env"
cd /content/proto_universe/proto_system/sonobe/fl-zkp-bridge
cargo fetch
maturin develop --release
```

---

## Cell 6: Verify bridge import

```python
import importlib.util
print("bridge_import_ok=", importlib.util.find_spec("fl_zkp_bridge") is not None)
```

---

## Cell 7: Required tests from the plan

```bash
%%bash
set -e
cd /content/proto_universe/proto_system
python -m pytest tests/test_zkp_soundness.py -q
python -m pytest tests/test_zkp_prover.py -q
```

---

## Cell 8: Lightweight integration run (low-load)

Use this first to avoid Colab runtime overload/timeouts:

```bash
%%bash
set -e
cd /content/proto_universe/proto_system

OUT="outputs/paper_essential"
# If using Drive persistence, replace OUT with e.g. /content/drive/MyDrive/proto_system_outputs

python scripts/run_zkp_research.py \
  --quick \
  --max-experiments 1 \
  --rounds 1 \
  --local-epochs 1 \
  --torch-threads 1 \
  --grad-sample 500 \
  --aggregator multi_krum \
  --output "$OUT" \
  --quiet
```

---

## Cell 9: Full experiment matrix (72 runs)

Current full matrix in the script is 72 experiments:

- 1 model (`cnn`) × 2 client counts (`10,20`) × 3 Byzantine fractions (`0.2,0.3,0.5`) × 4 attacks × 3 aggregators

```bash
%%bash
set -e
cd /content/proto_universe/proto_system

OUT="outputs/paper_essential"
# If using Drive persistence, replace OUT with e.g. /content/drive/MyDrive/proto_system_outputs

python scripts/run_zkp_research.py \
  --rounds 5 \
  --local-epochs 2 \
  --torch-threads 1 \
  --grad-sample 1000 \
  --output "$OUT"
```

If runtime limits are tight, split by aggregator (24 each):

```bash
%%bash
set -e
cd /content/proto_universe/proto_system
OUT="outputs/paper_essential"

python scripts/run_zkp_research.py --aggregator fedavg --rounds 5 --local-epochs 2 --torch-threads 1 --grad-sample 1000 --output "$OUT"
python scripts/run_zkp_research.py --aggregator multi_krum --rounds 5 --local-epochs 2 --torch-threads 1 --grad-sample 1000 --output "$OUT"
python scripts/run_zkp_research.py --aggregator coord_median --rounds 5 --local-epochs 2 --torch-threads 1 --grad-sample 1000 --output "$OUT"
```

---

## Cell 10: Inspect outputs

```bash
%%bash
set -e
cd /content/proto_universe/proto_system
find outputs/paper_essential -maxdepth 2 -type f | head -n 40
```

Important output files are typically:

- `outputs/paper_essential/run_<timestamp>/main.log`
- `outputs/paper_essential/run_<timestamp>/summary.json`

---

## Common Colab Issues

- If `cargo` is not found in a later cell, run:
  - `source "$HOME/.cargo/env"`
- If the runtime disconnects on full runs:
  - use aggregator-split commands
  - keep `--torch-threads 1`
  - persist outputs to Drive
- If submodule errors appear:
  - rerun the submodule sync/update cell

---

## Optional: Apply preserved local `sonobe` patch (important custom changes)

If you need the custom bridge changes that cannot be pushed to upstream `privacy-scaling-explorations/sonobe`, apply the patch bundled in this repo:

```bash
%%bash
set -e
cd /content/proto_universe/proto_system/sonobe

git apply --check ../patches/sonobe/0001-fl-zkp-bridge-update-bridge-api-and-config.patch
git apply ../patches/sonobe/0001-fl-zkp-bridge-update-bridge-api-and-config.patch
```

Then rebuild the Python bridge:

```bash
%%bash
set -e
source "$HOME/.cargo/env"
cd /content/proto_universe/proto_system/sonobe/fl-zkp-bridge
maturin develop --release
```

This keeps the main repo reproducible while still preserving your important local `sonobe` edits.
