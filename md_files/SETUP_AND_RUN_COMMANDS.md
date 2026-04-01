# Setup and Run Commands (End-to-End)

This file provides copy-paste commands to fully set up the project, initialize submodules, build the Rust ZKP bridge, run required tests, and execute the full experiment suite.

## 0) Prerequisites

- Linux with `git`, `python3`, and `cargo` installed
- Recommended Python: `3.10` to `3.12`
- Network access for dependency/data downloads

Check tools:

```bash
git --version
python3 --version
cargo --version
```

---

## 1) Clone + Submodule Setup

```bash
git clone https://github.com/nevin-py/proto_universe.git
cd proto_universe/proto_system

git submodule sync --recursive
git submodule update --init --recursive
```

If submodule state looks dirty:

```bash
git submodule foreach --recursive 'git status --short'
```

---

## 2) Python Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---

## 3) Build Rust ZKP Python Bridge

```bash
cd sonobe/fl-zkp-bridge
cargo fetch
maturin develop --release
cd ../..
```

Quick sanity import:

```bash
python -c "import importlib.util; print('bridge_import_ok=', importlib.util.find_spec('fl_zkp_bridge') is not None)"
```

---

## 4) Required Soundness Tests (from plan)

```bash
python -m pytest tests/test_zkp_soundness.py -q
python -m pytest tests/test_zkp_prover.py -q
```

Optional full test pass:

```bash
python -m pytest -q
```

---

## 5) End-to-End Integration Run (lightweight)

Use this to validate full pipeline with low system load:

```bash
.venv/bin/python scripts/run_zkp_research.py \
  --quick \
  --max-experiments 1 \
  --rounds 1 \
  --local-epochs 1 \
  --torch-threads 1 \
  --grad-sample 500 \
  --aggregator multi_krum \
  --output outputs/paper_essential \
  --quiet
```

---

## 6) Run the Full Experiment Count

Current script full matrix is:

- `1` model (`cnn`) × `2` client counts (`10,20`) × `3` Byzantine fractions (`0.2,0.3,0.5`) × `4` attacks × `3` aggregators = **72 experiments**

Run all 72 in one command:

```bash
.venv/bin/python scripts/run_zkp_research.py \
  --rounds 5 \
  --local-epochs 2 \
  --torch-threads 1 \
  --grad-sample 1000 \
  --output outputs/paper_essential
```

Notes:
- Do **not** pass `--quick`.
- Do **not** pass `--max-experiments`.
- Keep `--torch-threads 1` to reduce system overload.

---

## 7) Per-Aggregator Full Runs (optional split)

If you want smaller chunks, run 24 experiments per aggregator:

```bash
.venv/bin/python scripts/run_zkp_research.py --aggregator fedavg --rounds 5 --local-epochs 2 --torch-threads 1 --grad-sample 1000 --output outputs/paper_essential
.venv/bin/python scripts/run_zkp_research.py --aggregator multi_krum --rounds 5 --local-epochs 2 --torch-threads 1 --grad-sample 1000 --output outputs/paper_essential
.venv/bin/python scripts/run_zkp_research.py --aggregator coord_median --rounds 5 --local-epochs 2 --torch-threads 1 --grad-sample 1000 --output outputs/paper_essential
```

---

## 8) Output Locations

Each run writes to:

- `outputs/paper_essential/run_<timestamp>/main.log`
- `outputs/paper_essential/run_<timestamp>/summary.json`
- Per-experiment JSON files in same run folder

---

## 9) Shell Restart / Fish Startup Issue Workaround

If VS Code terminal keeps restarting due fish startup sourcing errors, run commands in a clean bash shell:

```bash
bash --noprofile --norc
cd /home/ariva/work/proto_universe/proto_system
source .venv/bin/activate
```

Then rerun the commands above.

---

## 10) Re-run From Existing Workspace

If you already have the repo:

```bash
cd /home/ariva/work/proto_universe/proto_system
git pull
git submodule sync --recursive
git submodule update --init --recursive
source .venv/bin/activate
python -m pip install -r requirements.txt
cd sonobe/fl-zkp-bridge && maturin develop --release && cd ../..
python -m pytest tests/test_zkp_soundness.py -q
python -m pytest tests/test_zkp_prover.py -q
.venv/bin/python scripts/run_zkp_research.py --rounds 5 --local-epochs 2 --torch-threads 1 --grad-sample 1000 --output outputs/paper_essential
```
