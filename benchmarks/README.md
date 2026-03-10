# FiZK Benchmarks

Four self-contained benchmark scripts for evaluating FiZK (Federated
learning with Zero-Knowledge proofs).  Each script is independently
runnable, resumable, and generates a `--summary` table without re-running.

---

## Prerequisites

**Python environment**

```bash
pip install torch torchvision numpy psutil
# gpu monitoring (optional):
pip install gputil
```

**Rust / ZKP bridge** — required for `fizk_norm` and `protogalaxy_full`

```bash
cd sonobe/fl-zkp-bridge
maturin develop --release      # takes ~5 min, needs Rust + maturin
cd ../..
```

Verify the bridge installed correctly:

```bash
python -c "import fl_zkp_bridge; print('OK')"
```

**Run the circuit sanity check first** (instant for sections A/B, ~5 min for C):

```bash
python scripts/test_circuit_behavior.py
```

---

## Quick start

```bash
# See all experiments without running any
python benchmarks/bench_overhead.py --dry-run

# Run the full overhead benchmark (~45 min)
python benchmarks/bench_overhead.py

# Print summary from already-completed results
python benchmarks/bench_overhead.py --summary

# Run everything sequentially
bash benchmarks/run_all.sh
```

Interrupt at any time with `Ctrl+C` — progress is saved after each
experiment.  Restart the same command to continue from where it stopped.

---

## Scripts

### 1. `bench_overhead.py` — ~45 min

**What it measures:** Per-round timing breakdown with no Byzantine clients.
Shows exactly how much time ZKP adds relative to a plain federation.

| Defense | Config | Purpose |
|---|---|---|
| `vanilla` | FL + trimmed mean | true baseline |
| `fizk_norm` | + Merkle + L2 norm filter | cost of Merkle + filter |
| `protogalaxy_full` | + ProtoGalaxy IVC ZKP | full FiZK cost |

**Setup:** MLP (109 K params), 10 clients, 2 galaxies, 5 rounds, 3 trials.

**Output columns:** total round time, Phase 1/2/3, ZK prove, ZK verify, Merkle build.

```bash
python benchmarks/bench_overhead.py
python benchmarks/bench_overhead.py --defense vanilla fizk_norm  # subset
```

---

### 2. `bench_scalability.py` — ~3 h (can run `--clients 10 20` for ~1 h)

**What it measures:** How round time grows as client count increases.
ZKP time grows linearly with clients; vanilla barely changes.

| Client counts | Galaxies | Rounds | Model |
|---|---|---|---|
| 10, 20, 30, 50 | clients÷5 (min 2) | 5 | linear (2 tensors/client — isolates ZKP) |

**Setup:** No Byzantine clients (clean) — isolates scaling overhead.

```bash
python benchmarks/bench_scalability.py
python benchmarks/bench_scalability.py --clients 10 20     # shorter run (~45 min)
python benchmarks/bench_scalability.py --summary           # print table
python benchmarks/bench_scalability.py --plot              # save PNG chart (requires matplotlib)
```

**Expected output:**

```
Defense                10        20        30        50
──────────────────────────────────────────────────────
vanilla               4.1       4.3       4.6       5.0   (s/round)
fizk_norm             6.2       8.1      10.5      14.2
protogalaxy_full     42.0      84.5     130.1     220.3

ZKP overhead ratio vs vanilla:
   10 clients: 42.0s / 4.1s = 10.2x
   50 clients: 220.3s / 5.0s = 44.1x
```

---

### 3. `bench_zkp_value.py` — ~3 h

**What it measures:** The specific security advantage ZKP provides over a
plain norm filter (`fizk_norm`).

Two ZKP-specific attacks:

| Attack | Description | fizk_norm response | protogalaxy_full response |
|---|---|---|---|
| `gradient_substitution` | Commit honest gradient, submit poisoned in Phase 2 | undetectable (only checks final norm) | commitment hash mismatch → rejected |
| `compromised_aggregator` | Server silently re-admits 1 norm-violating client | no evidence | `zk_proofs_failed ≥ 1` per round — auditable |

Plus a clean baseline and `model_poisoning` as a reference.

**Key metric:** `ZKFail` column — number of ZKP proof failures across all
rounds.  A non-zero count means the circuit cryptographically witnessed
a violation.  `fizk_norm` always shows 0 (no ZKP).

```bash
python benchmarks/bench_zkp_value.py
python benchmarks/bench_zkp_value.py --defense fizk_norm protogalaxy_full --attack gradient_substitution
python benchmarks/bench_zkp_value.py --summary
```

---

### 4. `bench_attack_robustness.py` — ~5 h

**What it measures:** Model accuracy under standard attacks, across all
four defenses.

| Attacks | Byzantine fraction |
|---|---|
| `none` (clean baseline) | 0% |
| `model_poisoning` (10× scale) | 30% |
| `backdoor` (pattern injection) | 30% |
| `label_flip` (permuted labels) | 30% |

**Setup:** MLP model, 10 clients, 2 galaxies, 10 rounds, 3 trials.

```bash
python benchmarks/bench_attack_robustness.py
# Run only vanilla and multi_krum first (fast, ~30 min):
python benchmarks/bench_attack_robustness.py --defense vanilla multi_krum
# Then run the ZKP defenses:
python benchmarks/bench_attack_robustness.py --defense fizk_norm protogalaxy_full
python benchmarks/bench_attack_robustness.py --summary
```

---

## Output format

All results are saved as JSON files:

```
Eval_results/benchmarks/
  overhead/     custom/*.json   run.log
  scalability/  custom/*.json   run.log   scalability.png  (if --plot)
  zkp_value/    custom/*.json   run.log
  attack_robustness/  custom/*.json   run.log
```

Each JSON file is one complete experiment result with:

```json
{
  "config": { "defense": "protogalaxy_full", "attack": "model_poisoning", ... },
  "final_accuracy": 0.9216,
  "avg_round_time": 42.3,
  "avg_zk_prove_time": 38.1,
  "avg_f1": 1.0,
  "rounds": [ { "round_num": 1, "accuracy": 0.88, "zk_proofs_failed": 2, ... }, ... ],
  "resource_usage": { "cpu_percent": {...}, "ram_mb": {...}, "gpu_util_percent": {...} }
}
```

---

## Laptop / CPU-only notes

If you do not have a GPU, everything still runs — training on CPU is ~5×
slower.  Recommended subset:

```bash
# ~90 min on CPU:
python benchmarks/bench_overhead.py
python benchmarks/bench_scalability.py --clients 10 20 --defense vanilla protogalaxy_full

# Skip attack robustness (too slow on CPU without GPU)
```

If `fl_zkp_bridge` is not available (Rust not installed), the ZKP
defenses fall back to a SHA-256 commitment — you can still benchmark
`vanilla` and `fizk_norm` overhead without the Rust toolchain.

---

## Resuming interrupted runs

Every script writes JSON immediately after each experiment completes and
checks for existing files before running.  Just re-run the same command:

```bash
# Interrupted after 12/48 experiments? Just re-run:
python benchmarks/bench_attack_robustness.py
# Picks up at experiment 13.
```

---

## Reproducing results on another machine

1. Clone the repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Build the ZKP bridge: `cd sonobe/fl-zkp-bridge && maturin develop --release`
4. Confirm circuit works: `python scripts/test_circuit_behavior.py`
5. Run: `python benchmarks/bench_overhead.py`   (fastest, 45 min)

Seeds are fixed (42/43/44) so results should be numerically reproducible
on the same hardware.  GPU/CPU differences will affect timing but not accuracy.
