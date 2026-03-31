# ProtoGalaxy Research Prototype - Analysis & Roadmap

**Date:** 2026-03-30
**Status:** Proof-of-Concept (PoC) — Requires significant development for publication

---

## Executive Summary

ProtoGalaxy is a Byzantine-resilient hierarchical federated learning system with a 5-layer defense pipeline, ZK proof integration (sum-check via Rust backend), and galaxy-level aggregation. The codebase implements the _intent_ of the architecture but has **critical structural inconsistencies**, **incomplete tests**, **missing modules**, and **architectural misalignments** that prevent it from functioning as a coherent research prototype.

---

## 1. System Architecture (As Implemented)

```
┌─────────────────────────────────────────────────────────────────┐
│                     ProtoGalaxyOrchestrator                      │
│  (src/orchestration/protogalaxy_orchestrator.py)                │
│                                                                  │
│  Phase 1: Commitment → Merkle Trees + ZK Sum-Check Proofs      │
│  Phase 2: Revelation → Merkle Proof Verification               │
│  Phase 3: Defense → 5-Layer Pipeline (L1-L5)                   │
│  Phase 4: Aggregation → Global Model Update                     │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────────┐   ┌──────────────────┐
│  Galaxy 0    │    │    Galaxy 1      │   │    Galaxy N      │
│  - Clients   │    │    - Clients     │   │    - Clients     │
│  - MerkleTree│    │    - MerkleTree  │   │    - MerkleTree  │
│  - Aggregator│    │    - Aggregator   │   │    - Aggregator  │
└──────────────┘    └──────────────────┘   └──────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                   ┌──────────────────┐
                   │ GlobalAggregator  │
                   │ + GlobalMerkleTree│
                   └──────────────────┘

Defense Layers (L1-L5):
  L1: Cryptographic integrity (Merkle verification)
  L2: Statistical anomaly detection (4-metric StatisticalAnalyzer)
  L3: Byzantine-robust aggregation (TrimmedMean/Multi-Krum/FLTrust)
  L4: Reputation-based filtering (update_behavior_score)
  L5: Galaxy-level defense (layer5_galaxy.py)
```

---

## 2. Critical Issues

### 2.1 Two Separate, Incompatible Systems

**Problem:** The codebase contains two completely separate FL systems that do not share interfaces:

| Component        | System A (Orchestrator)      | System B (Simulation)              |
| ---------------- | ---------------------------- | ---------------------------------- |
| Entry point      | `ProtoGalaxyOrchestrator`    | `FLSimulation`                     |
| Aggregators      | `galaxy.py`, `global_agg.py` | Uses same files but different flow |
| Defense          | `DefenseCoordinator`         | Same `DefenseCoordinator`          |
| Client           | `Client` class               | `ClientState` dataclass            |
| Round management | Manual phase execution       | `RoundManager`                     |

**Impact:** Code in `simulation/runner.py` imports from `src.orchestration.round_manager` (which may not exist), and `coordinator.py` imports a non-existent `src.client.client`.

**Evidence:**

```python
# coordinator.py:3-7
from src.client.client import Client  # DOES NOT EXIST
from src.defense.coordinator import DefenseCoordinator

# simulation/runner.py:23-24
from src.orchestration.round_manager import RoundManager, RoundPhase  # NEEDS VERIFICATION
from src.logging import FLLogger, FLLoggerFactory  # Partially exists
```

**Fix Required:** Unify both systems or clearly separate them with a shared interface.

---

### 2.2 Missing Modules

The following imports in the codebase point to files that **do not exist** or are **empty stubs**:

| Import                                                                 | Expected Location                    | Status                                                                |
| ---------------------------------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------- |
| `from src.client.client import Client`                                 | `src/client/client.py`               | **MISSING** (only `trainer.py`, `commitment.py`, `verifier.py` exist) |
| `from src.orchestration.round_manager import RoundManager, RoundPhase` | `src/orchestration/round_manager.py` | **NEEDS VERIFICATION**                                                |
| `from src.defense.layer5_galaxy import Layer5GalaxyDefense`            | `src/defense/layer5_galaxy.py`       | **MISSING**                                                           |
| `fl_zkp_bridge` (Rust module)                                          | External                             | **NOT COMPILED** — falls back to SHA-256                              |
| `from src.orchestration.pipeline import ProtoGalaxyPipeline`           | `src/orchestration/pipeline.py`      | **NEEDS VERIFICATION**                                                |

**Fix Required:** Create missing modules or remove broken imports.

---

### 2.3 Gradient Shape Loss in Aggregation

**Problem:** All robust aggregators (`TrimmedMeanAggregator`, `MultiKrumAggregator`, `CoordinateWiseMedianAggregator`) **flatten all gradients to a single numpy array**, destroying layer structure:

```python
# robust_agg.py:86-92 — ALL gradient layers become ONE flat vector
flat = np.concatenate([
    g.detach().cpu().numpy().flatten() if isinstance(g, torch.Tensor)
    else np.array(g).flatten()
    for g in grads
])
flattened.append(flat)  # Shape: (n, total_dimensions)
```

**Impact:**

- Cannot reconstruct per-layer gradients for model update
- The aggregated result is a 1D array, but the model expects `List[torch.Tensor]` per layer
- Phase 4 in `ProtoGalaxyOrchestrator` tries to apply flattened gradients directly to model parameters, which is incorrect

**Evidence in `protogalaxy_orchestrator.py:600-617`:**

```python
# This assumes global_update is a single flat tensor, but aggregators
# return flattened arrays. The model update logic is broken.
global_update = torch.stack(list(clean_galaxy_updates.values())).mean(dim=0)
# ...
param_vector -= learning_rate * global_update  # Wrong dimensionality
```

**Fix Required:** Aggregators must preserve layer-wise structure or the model update code must be fixed.

---

### 2.4 Byzantine Tolerance Formulas Are Wrong

**Problem:** The Multi-Krum and Trimmed Mean configurations use `f` incorrectly:

```yaml
# config.yaml
layer3_krum_f: 1 # "Multi-Krum parameter" — WRONG interpretation
```

**Architecture requirement:** Krum requires `n > 2f + 2` (for n clients, f Byzantine). If you want to tolerate f Byzantine clients, you need `n >= 2f + 3`.

**Current code (`robust_agg.py:227`):**

```python
if n <= 2 * self.f + 2:
    # Not enough clients for Krum, fall back to simple average
    # This is BACKWARDS — should be n < 2f + 2 for fallback
```

**Trimmed Mean:** The trim ratio of 0.1 (10% each side) only handles 20% Byzantine clients. The architecture may claim higher tolerance but the math doesn't support it without proper parameterization.

**Fix Required:** Fix the fallback condition and add validation for minimum client counts.

---

### 2.5 Layer 5 Defense Module Does Not Exist

**Problem:** `DefenseCoordinator.run_galaxy_defense()` imports and calls `Layer5GalaxyDefense` which does not exist:

```python
# src/defense/coordinator.py:22
from src.defense.layer5_galaxy import Layer5GalaxyDefense

# src/defense/coordinator.py:126-133
self.layer5 = Layer5GalaxyDefense(
    num_galaxies=num_galaxies,
    galaxy_rep_decay=self.config.get('layer5_galaxy_decay', 0.9),
    norm_threshold=self.config.get('layer5_norm_threshold', 3.0),
    direction_threshold=self.config.get('layer5_direction_threshold', 0.5),
    consistency_threshold=self.config.get('layer5_consistency_threshold', 0.7),
    dissolution_streak=self.config.get('layer5_dissolution_streak', 3)
)
```

**Impact:** Any call to `run_galaxy_defense()` will crash with `ImportError` or `ModuleNotFoundError`.

**Fix Required:** Implement `src/defense/layer5_galaxy.py` with the `Layer5GalaxyDefense` class.

---

### 2.6 ZKP Backend Not Compiled — Falls Back to SHA-256

**Problem:** The Rust `fl_zkp_bridge` module is not available, causing all ZK proofs to use a SHA-256 fallback:

```python
# src/crypto/zkp_prover.py:31-40
try:
    from fl_zkp_bridge import FLZKPBoundedProver as _FLZKPBoundedProver
    _ZKP_AVAILABLE = True
except ImportError:
    logger.warning(
        "fl_zkp_bridge not available - ZK proofs will use fallback mode. "
        "Build with: cd sonobe/fl-zkp-bridge && maturin develop --release"
    )
```

**Impact:**

- The "bounded" prover (with norm enforcement) does NOT enforce bounds in fallback mode
- `bounds_enforced = False` even when the circuit would theoretically enforce it
- No actual zero-knowledge property — just a hash commitment
- The research claim of "ZK sum-check proofs" is only valid with the Rust backend

**Fix Required:** Either compile the Rust backend or update the research claims to clarify this is a "commitment scheme" not a true ZK proof system in the current state.

---

### 2.7 Layer 4 Reputation Score Has Undefined Weights

**Problem:** `DefenseCoordinator` references architecture Section 4.4 for behavior scoring:

```python
# src/defense/coordinator.py:243-244
# B(t) = w1*I_integrity + w2*I_statistical + w3*I_krum + w4*I_historical
# Uses update_behavior_score() instead of simple penalize_client()
```

But `ReputationManager.update_behavior_score()` has **undefined weight parameters** `w1, w2, w3, w4`. The method signature and computation need to be verified against the stated formula.

**Fix Required:** Check `src/defense/reputation.py` and verify the weight parameters.

---

## 3. Testing Gaps

### 3.1 Test Coverage Summary

| Test File                      | Coverage                                   | Status                                                                 |
| ------------------------------ | ------------------------------------------ | ---------------------------------------------------------------------- |
| `test_trimmed_mean.py`         | TrimmedMeanAggregator, MultiKrumAggregator | **Partially passing** — some edge cases fail                           |
| `test_merkle.py`               | MerkleTree, verify_proof                   | **Needs verification**                                                 |
| `test_fl_modules.py`           | Trainer, partitioner, communication        | **Module-level only, no integration**                                  |
| `test_pipeline_integration.py` | 4-phase pipeline                           | **Mostly passing** — tests CommitmentGenerator and ProofVerifier fixes |
| `test_bounded_zkp.py`          | ZK proofs                                  | **Needs Rust backend** — mostly tests fallback                         |
| `test_trimmed_mean.py:142-154` | Multi-gradient-component                   | **BROKEN** — returns flat array, not per-layer                         |

### 3.2 Specific Test Issues

**`test_trimmed_mean.py:142-154` — Multiple gradient components test:**

```python
def test_multiple_gradient_components(self):
    """Works with multiple gradient components per update."""
    updates = [
        {'gradients': [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]},  # 2 layers
        ...
    ]
    result = agg.aggregate(updates)
    assert len(result['gradients']) == 5  # EXPECTS 5 elements
```

This test **passes** because the aggregator flattens everything to 5 dimensions. But the **real issue** is that the original layer structure `(2, 3)` is lost — the test only checks the _flattened_ dimension count, not shape preservation.

**Missing tests:**

- End-to-end simulation with actual model training
- Byzantine attack effectiveness validation
- Layer 5 galaxy defense (module doesn't exist)
- ZK proof verification with real Rust backend
- Non-IID data distribution effects on defense layers
- Galaxy dissolution mechanism

---

## 4. Data Flow Issues

### 4.1 Client Update Format Inconsistency

**Problem:** Different parts of the code expect different gradient formats:

| Component                                   | Expected Format                                                |
| ------------------------------------------- | -------------------------------------------------------------- |
| `Trainer.get_gradients()`                   | `List[torch.Tensor]` (per-layer)                               |
| `GalaxyMerkleTree`                          | `List[torch.Tensor]` but flattened to single tensor in Phase 1 |
| `TrimmedMeanAggregator`                     | `List[gradient]` flattened to `(n, d)` array                   |
| `ProtoGalaxyOrchestrator._phase3_defense()` | Dict of `{client_id: update}` but gradients format varies      |

**Evidence in `protogalaxy_orchestrator.py:512-529`:**

```python
for u in galaxy_client_updates:
    grads = u['gradients']
    if isinstance(grads, dict):
        flat_grad = torch.cat([g.flatten() for g in grads.values()])
    elif isinstance(grads, list):
        flat_grad = torch.cat([g.flatten() for g in grads])
    elif isinstance(grads, torch.Tensor):
        flat_grad = grads.flatten()
```

This handles multiple formats but the format choice should be standardized.

### 4.2 Non-IID Partitioner Has Calculation Bug

**Problem:** `NonIIDPartitioner.partition()` uses incorrect calculation for sample assignment:

```python
# src/data/partition.py:166-171
n_samples = len(cls_idx) // (num_clients // classes_per_client + 1)
start = (client_id // classes_per_client) * n_samples
end = start + n_samples
```

This can cause index out of bounds or uneven distribution when `num_clients` is not evenly divisible.

---

## 5. Configuration Issues

### 5.1 Config.yaml Inconsistencies

```yaml
# config.yaml
fl:
  num_clients: 10
  clients_per_round: 8 # Not used anywhere

galaxy:
  num_galaxies: 3
  clients_per_galaxy: 3 # 3*3 = 9, but num_clients = 10
```

**Problem:** `num_clients=10` with `num_galaxies=3` and `clients_per_galaxy=3` gives 9 client slots, leaving 1 client unassigned.

### 5.2 Default Threshold Values Don't Match Architecture

The `StatisticalAnalyzer` defaults in `statistical.py:40-46`:

```python
norm_threshold_sigma: float = 4.5,  # Default 3.0 in config.yaml
cosine_threshold: float = 0.5,      # Should align with architecture spec
coordinate_threshold_sigma: float = 4.5,
kl_divergence_threshold: float = 2.0,
```

But `DefenseCoordinator.__init__` passes:

```python
norm_threshold_sigma=self.config['layer1_threshold'],  # 2.0 from config.yaml
```

This creates a mismatch between the analyzer's designed defaults and what's actually passed.

---

## 6. Research Publication Readiness Assessment

### 6.1 Strong Points

| Feature                                    | Implementation Quality | Notes                                            |
| ------------------------------------------ | ---------------------- | ------------------------------------------------ |
| Hierarchical aggregation (galaxy → global) | ✅ Good concept        | Galaxy dissolution in Layer 5 not implemented    |
| 5-layer defense pipeline                   | ⚠️ Partial             | Layer 5 missing, L4 reputation weights undefined |
| Statistical analyzer (4-metric)            | ✅ Good                | Median+MAD approach is Byzantine-resilient       |
| Commitment scheme (Merkle + ZK)            | ⚠️ Partial             | ZK only works with Rust backend                  |
| Trimmed Mean aggregator                    | ✅ Implemented         | Works but loses layer structure                  |
| Multiple partition strategies              | ✅ Implemented         | IID, Dirichlet, NonIID, Shard                    |

### 6.2 Weak Points for Publication

1. **The ZK proof claim is misleading** — Without the Rust backend, it's just SHA-256 commitments
2. **No formal security analysis** — The document references Architecture Section 4 but no actual security proof or analysis exists
3. **Layer 5 (galaxy defense) is missing** — Key differentiator of the architecture is unimplemented
4. **Inconsistent terminology** — "ProtoGalaxy" vs "Protogalaxy" vs "proto-galaxy" throughout
5. **No benchmark comparison** — Should compare against Krum, Trimmed Mean baselines
6. **Reputation weights not defined** — B(t) formula in comments has undefined parameters

### 6.3 Required for Publication

**Must fix (Critical):**

- [ ] Implement `Layer5GalaxyDefense` class
- [ ] Fix gradient shape preservation in aggregators
- [ ] Unify or separate `ProtoGalaxyOrchestrator` and `FLSimulation` systems
- [ ] Create missing `src/client/client.py` or fix imports in `coordinator.py`
- [ ] Fix Byzantine tolerance formula in Krum fallback

**Should fix (Important):**

- [ ] Compile Rust ZKP backend or update claims to reflect fallback mode
- [ ] Add comprehensive integration tests with real model training
- [ ] Validate Non-IID partitioner calculations
- [ ] Align config defaults with architecture specs
- [ ] Add end-to-end experiments with baseline comparisons

**Nice to have (For quality):**

- [ ] Add security proof outline in documentation
- [ ] Benchmark L4/L5 contributions separately (ablation study)
- [ ] Add formal API documentation
- [ ] Create experiment scripts that reproduce paper figures

---

## 7. Recommended Development Roadmap

### Phase 1: Core Fixes (1-2 weeks)

1. Fix all import errors (create missing modules or remove broken imports)
2. Implement `Layer5GalaxyDefense` — the galaxy-level anomaly detection
3. Fix gradient shape preservation in aggregators
4. Unify the two FL systems (Orchestrator vs Simulation)

### Phase 2: Testing & Validation (1-2 weeks)

1. Write integration tests for full pipeline with real model training
2. Test Byzantine resilience with controlled attacks
3. Benchmark against standard FedAvg, Krum, Trimmed Mean
4. Fix Non-IID partitioner bugs

### Phase 3: ZK Proof Integration (1 week)

1. Compile Rust `fl_zkp_bridge` backend, OR
2. Update research claims to clarify commitment scheme vs ZK proof

### Phase 4: Documentation & Experiments (1-2 weeks)

1. Write proper architecture documentation
2. Run experiments for paper figures
3. Add ablation studies for each defense layer
4. Formalize security analysis

---

## 8. File Manifest

### Key Source Files

| Path                                            | Purpose                      | Issues                                     |
| ----------------------------------------------- | ---------------------------- | ------------------------------------------ |
| `src/orchestration/protogalaxy_orchestrator.py` | Main 4-phase orchestrator    | Uses broken imports, gradient shape issues |
| `src/simulation/runner.py`                      | FL simulation framework      | Separate system, broken imports            |
| `src/defense/coordinator.py`                    | Defense layer coordination   | Calls non-existent Layer5                  |
| `src/defense/statistical.py`                    | 4-metric anomaly detection   | ✅ Good implementation                     |
| `src/defense/robust_agg.py`                     | Byzantine-robust aggregators | Loses layer structure                      |
| `src/crypto/merkle.py`                          | Merkle tree + commitments    | ✅ Generally good                          |
| `src/crypto/zkp_prover.py`                      | ZK proof wrapper             | Fallback only without Rust                 |
| `src/client/client.py`                          | **MISSING**                  | Needs to be created                        |
| `src/defense/layer5_galaxy.py`                  | **MISSING**                  | Needs to be created                        |
| `src/aggregators/galaxy.py`                     | Galaxy-level aggregation     | ✅ Basic implementation                    |
| `src/aggregators/global_agg.py`                 | Global aggregation           | ✅ Basic implementation                    |

### Test Files

| Path                                 | Coverage                  |
| ------------------------------------ | ------------------------- |
| `tests/test_trimmed_mean.py`         | Aggregators               |
| `tests/test_merkle.py`               | Merkle operations         |
| `tests/test_fl_modules.py`           | Individual modules        |
| `tests/test_pipeline_integration.py` | Phase 1-3 integration     |
| `tests/test_bounded_zkp.py`          | ZK proofs (fallback mode) |

### Configuration

| Path                         | Issues                                               |
| ---------------------------- | ---------------------------------------------------- |
| `config.yaml`                | Galaxy/client count mismatch, threshold misalignment |
| `experiments/configs/*.yaml` | Need to verify these work with fixed system          |

---

## 9. Appendix: Architecture Claims vs Implementation

| Architecture Section | Claim                              | Implementation Status               |
| -------------------- | ---------------------------------- | ----------------------------------- |
| Section 3.4          | 4-phase protocol                   | ✅ Phase 1-4 defined, some bugs     |
| Section 4.2          | 4-metric StatisticalAnalyzer       | ✅ Implemented correctly            |
| Section 4.3          | Trimmed Mean / Multi-Krum          | ✅ Implemented, shape bug           |
| Section 4.4          | Reputation B(t) scoring            | ⚠️ Formula defined, weights missing |
| Section 4.5          | Galaxy-level defense + dissolution | ❌ Layer 5 module missing           |
| Section 6.2          | 5 Byzantine attack types           | ✅ All 5 implemented in Client      |

---

_Document generated: 2026-03-30_
_Next action: Create GitHub issue tracking for each critical item above_
