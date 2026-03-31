# ProtoGalaxy Codebase Deep Audit Report

**Auditor:** Senior Code Reviewer  
**Date:** 2026-02-12  
**Scope:** Full codebase audit against `protogalaxy_architecture.md` specification  
**Verdict:** The codebase is substantially functional as a research prototype. Several concrete bugs, logic gaps, and missing implementations must be addressed before the system can be considered a complete, faithful implementation of the architecture.

---

## Audit Methodology

Every Python module under `src/` was read in full. Each function was evaluated for:

1. **Functional correctness** — Does the code do what it claims? Are there logic errors?
2. **Architecture fidelity** — Does it match the specification in `protogalaxy_architecture.md`?
3. **No mocks / no simulation cheats** — Is there any `time.sleep`, hardcoded `return True`, or fake data?
4. **Integration integrity** — When module A calls module B, are the interfaces compatible?
5. **Completeness** — Are all specified components present and wired into the pipeline?

---

## Executive Summary

| Category | Status | Details |
|----------|--------|---------|
| Merkle Tree (Layer 1) | ✅ Functional | Correct SHA-256 Merkle tree with proof generation/verification |
| Commit-Reveal Protocol | ⚠️ Has Bugs | `GradientCommitment.verify()` works, but `CommitmentGenerator.get_commitment_proof()` crashes |
| Statistical Detection (Layer 2) | ✅ Functional | All 4 metrics implemented (norm, direction, coordinate, KL-divergence) |
| Robust Aggregation (Layer 3) | ✅ Functional | Both Trimmed Mean and Multi-Krum implemented with correct algorithms |
| Reputation System (Layer 4) | ✅ Functional | Full EWMA with quarantine/ban/rehabilitation |
| Galaxy Defense (Layer 5) | ✅ Functional | Anomaly detection + reputation + adaptive re-clustering |
| ZKP Sum-Check Proofs | ✅ Functional (graceful fallback) | Real ZKP via Rust bridge with SHA-256 fallback |
| Pipeline Orchestration | ⚠️ Has Bugs | End-to-end pipeline works but has interface mismatches |
| Client Verifier | 🔴 Broken | `Client.verify_merkle_root()` passes wrong parameter names |
| Communication Layer | ⚠️ Unused | Fully implemented but not integrated into the pipeline |
| `time.sleep` / Mocks | ✅ None found | No artificial delays or mock returns in `src/` |
| Forensic Logger | ✅ Functional | Evidence database with integrity hashing and querying |

---

## 🔴 CRITICAL BUGS (Will Crash at Runtime)

### BUG-1: `Client.verify_merkle_root()` — Wrong Parameter Names

**File:** `src/client/client.py`, lines 81-84  
**Severity:** 🔴 Runtime crash (TypeError)

```python
# CURRENT (BROKEN):
return self.verifier.verify_proof(
    leaf_hash=own_commitment_hash,
    proof=proof,
    root_hash=published_root  # ← WRONG: parameter name is 'root', not 'root_hash'
)

# ProofVerifier.verify_proof signature:
def verify_proof(self, root: str, proof: list, leaf_hash: str, leaf_index: int = 0)
```

**Impact:** Any client attempting to verify a Merkle root (Architecture Section 3.4, Phase 2, Step 1) will crash with `TypeError: unexpected keyword argument 'root_hash'`. This means the client-side root consistency check specified in the architecture is completely non-functional.

**Fix Required:** Change `root_hash=published_root` to `root=published_root` and add `leaf_index` parameter.

---

### BUG-2: `CommitmentGenerator.get_commitment_proof()` — References Non-Existent Attribute

**File:** `src/client/commitment.py`, line 37  
**Severity:** 🔴 Runtime crash (AttributeError)

```python
def get_commitment_proof(self, index: int):
    if index < len(self.commitments):
        return self.commitments[index].merkle_tree.get_proof(index)
        #                            ^^^^^^^^^^^
        # GradientCommitment has NO 'merkle_tree' attribute
```

**Impact:** `GradientCommitment` (defined in `src/crypto/merkle.py`) has fields `gradients`, `client_id`, `round_number`, `nonce`, `timestamp`, `commitment`. There is no `merkle_tree` attribute. Calling this method will crash with `AttributeError`.

**Note:** This method is not called in the main pipeline (`run_pipeline.py`), so it doesn't block the demo, but it's dead/broken code that represents an incomplete feature.

---

### BUG-3: `GlobalMerkleTreeAdapter.build_from_galaxy_roots()` — Metadata Destroys Hash Consistency

**File:** `src/crypto/merkle_adapter.py`, lines 88-105  
**Severity:** ⚠️ Logic error (silent)

When building the global Merkle tree from galaxy roots, the adapter passes `metadata_list` alongside the galaxy root hashes. But in `MerkleTree.build()`, when an item is already a 64-char hex string, it's used directly as a leaf hash. However, the metadata is still applied in `compute_hash()` when the condition fails (e.g., if a root is not exactly 64 chars). This creates an inconsistency: Merkle proofs generated from the global tree will NOT be verifiable against the galaxy roots directly, because the hash includes extraneous metadata.

**Current behavior:** Galaxy root strings ARE 64-char hex strings, so the `if isinstance(item, str) and len(item) == 64` branch is taken and metadata is silently ignored. This works *by accident* — the metadata_list construction is dead code that creates false confidence.

**Recommendation:** Remove the metadata parameter from `GlobalMerkleTreeAdapter.build_from_galaxy_roots()` or document this behavior explicitly.

---

## ⚠️ LOGIC / INTEGRATION ISSUES

### ISSUE-1: Trust-Weighted Aggregation Applied Too Early

**File:** `src/orchestration/pipeline.py`, lines 380-385  
**Architecture Reference:** Section 4.4 — "Trust-Weighted Aggregation"

```python
# In phase2_galaxy_verify_and_collect:
reputation = defense_coordinator.layer4.get_reputation(client_id)
weighted_gradients = []
for g in submission.gradients:
    weighted_gradients.append(g * reputation)
```

**Problem:** The architecture specifies trust-weighted aggregation as `w̄ = Σ(Rᵢ · ∇wᵢ) / Σ(Rᵢ)`. The current code applies the weight (`Rᵢ · ∇wᵢ`) but **never divides by `Σ(Rᵢ)`**. This means the effective learning rate scales with cumulative reputation, causing gradients to be systematically attenuated. With all clients at initial reputation 1.0, this works okay, but after a few rounds of reputation decay, honest clients' gradients will be suppressed far below their true magnitude.

**Additionally:** This weighting is applied in Phase 2 (revelation) BEFORE the defense pipeline runs in Phase 3. This means the statistical analyzer and Krum aggregator in Phase 3 operate on already-reweighted gradients, which distorts their anomaly detection thresholds. The architecture intends weighting to happen AFTER defense filtering, not before.

---

### ISSUE-2: Reputation Scores Initialize to 1.0, Architecture Says 0.5

**File:** `src/defense/reputation.py`, line 87  
**Architecture Reference:** Section 4.4 — "Initialize: Rᵢ(0) = 0.5 (neutral)"

```python
self.reputation_scores = {i: 1.0 for i in range(num_clients)}
```

**Impact:** Starting at 1.0 instead of 0.5 means the reputation system takes significantly longer to detect Byzantine clients (they start with more "reputation credit" to burn through). This directly contradicts the theoretical bound in Theorem 4.4 which assumes initial reputation of 0.5.

---

### ISSUE-3: Reputation EWMA Formula Deviates from Architecture

**File:** `src/defense/reputation.py`, lines 130-155  
**Architecture Reference:** Section 4.4 — "R(t+1) = (1-λ)·R(t) + λ·B(t), λ = 0.1"

The architecture defines `λ = 0.1` (learning rate), meaning 90% of old reputation is retained. The code uses `decay_factor = 0.9` as the retention factor for each individual indicator, but then the final `new_score` is directly assigned from `BehaviorScore.compute_weighted_score()` — it does NOT apply EWMA to the overall reputation score. The per-indicator EWMA is applied correctly, but the top-level reputation is just overwritten:

```python
self.reputation_scores[client_id] = new_score  # ← Direct overwrite, not EWMA
```

**Should be:**
```python
old_score = self.reputation_scores.get(client_id, 0.5)
self.reputation_scores[client_id] = (1 - lambda_lr) * old_score + lambda_lr * new_score
```

This means reputation swings much more wildly than the architecture intends. A single bad round can crater a client's reputation, and a single good round can restore it — defeating the "gradual detection" property described in Section 4.4.

---

### ISSUE-4: `GalaxyAggregator` and `GlobalAggregator` Are Never Used in the Pipeline

**File:** `src/aggregators/galaxy.py`, `src/aggregators/global_agg.py`  
**Used in:** `src/orchestration/pipeline.py` (instantiated but never called)

The pipeline creates `GalaxyAggregator` and `GlobalAggregator` instances but never calls their `aggregate()` method. Instead, all aggregation is delegated to `DefenseCoordinator.run_defense_pipeline()` which internally uses `TrimmedMeanAggregator` or `MultiKrumAggregator`. The standalone aggregators are dead code in the main pipeline path.

**This is acceptable** since the defense coordinator subsumes the aggregation role, but it means two separate aggregator hierarchies exist and only one is used. The other one (in `FLCoordinator` and `ProtoGalaxyOrchestrator`) uses the aggregators directly, creating inconsistency between pipeline variants.

---

### ISSUE-5: Coordinate-Wise Median (CWMed) — Not Implemented

**Architecture Reference:** Section 4.3 — "Coordinate-Wise Median"

The architecture specifies three aggregation methods: Multi-Krum, Coordinate-Wise Median, and Trimmed Mean. Only Multi-Krum and Trimmed Mean are implemented. CWMed is referenced in the architecture as the preferred method "for computational efficiency with moderate threat."

**Impact:** Missing one of the three specified aggregation strategies. For research reproducibility, CWMed should be implemented.

---

### ISSUE-6: FLLogger Signature Mismatch — `extra` Parameter

**File:** `src/logging/__init__.py`  
**Called from:** `src/orchestration/pipeline.py` (multiple locations)

The pipeline calls logger methods with `extra=` keyword:
```python
self.logger.info("message", extra={'key': 'value'})
```

But `FLLogger.info()` signature is:
```python
def info(self, message: str, event_type: str = "info", **kwargs)
```

The `extra` keyword gets swallowed by `**kwargs` and passed to `_log()`, which doesn't know what to do with it (it accepts `metrics=`, `client_id=`, `galaxy_id=` but not `extra=`). The extra data is silently lost.

**Note:** The pipeline currently runs with `logger=None`, so this doesn't crash, but if logging is enabled, the structured metadata won't be recorded.

---

## 📋 MISSING IMPLEMENTATIONS (Architecture Specified, Not Present)

### MISSING-1: Digital Signatures for Client Identity

**Architecture Reference:** Section 2.2 — "Each client has a unique cryptographic identity (public/private key pair)"  
**Architecture Reference:** Section 5.2, Theorem 5.3 — "Digital signatures prevent impersonation"

**Status:** Not implemented. Clients are identified by integer IDs only. There are no public/private key pairs, no digital signatures on gradient submissions, and no signature verification. This means:

- A client could impersonate another client by using their ID
- The "attribution guarantee" (Theorem 5.3) is not cryptographically enforced
- The "non-repudiation" property is not achieved

**For research prototype:** This is the single largest gap between the architecture and implementation. The Merkle tree provides integrity, but without signatures, attribution is based on trust in the transport layer.

---

### MISSING-2: Replay Attack Prevention — Incomplete

**Architecture Reference:** Section 4.1 — "Replay attacks (via round number in metadata)"

The `GradientCommitment` includes `round_number` and `timestamp` in its metadata, and the commitment hash binds to them. However, **there is no round-number check in the verification path** — the pipeline does not verify that a submitted gradient's round number matches the current round. A valid commitment from round N could theoretically be replayed in round N+1 if the Merkle tree happens to accept it.

**Mitigation:** The Merkle tree is rebuilt each round, so replayed commitments would have different roots and fail verification. But this is an implicit defense, not explicit validation as described in the architecture.

---

### MISSING-3: Aggregator Tamper Detection — Not Exercised

**Architecture Reference:** Section 5.4, Scenario 3 — "Malicious Aggregator"

The architecture describes a scenario where clients detect aggregator tampering by verifying their own submissions against published Merkle roots. While `Client.verify_merkle_root()` exists (with the bug above), **it is never called in the pipeline**. The pipeline does not have a step where clients verify the galaxy Merkle root after it's published.

**In `run_pipeline.py`:** After `phase1_galaxy_collect_commitments()` publishes the galaxy root, clients never verify it. They proceed directly to Phase 2 submission.

---

### MISSING-4: Non-IID Experiments Not Wired

**Architecture Reference:** Section 6.1 — "Non-IID (Label Skew)" and "Non-IID (Quantity Skew)"

The data partitioning strategies (`NonIIDPartitioner`, `DirichletPartitioner`) are implemented in `src/data/partition.py`, but the main pipeline runner (`run_pipeline.py`) only uses IID partitioning. The architecture's experimental methodology requires Non-IID experiments for research validity.

**Status:** Components exist but are not integrated into any runnable experiment script.

---

### MISSING-5: CIFAR-10, FEMNIST, Shakespeare Datasets

**Architecture Reference:** Section 6.1 — Datasets

- **CIFAR-10:** Model (`CIFAR10CNN`) and loader (`load_cifar10`) exist but no experiment script uses them
- **FEMNIST:** Not implemented at all
- **Shakespeare (LSTM):** Not implemented at all

**For research prototype:** MNIST-only experiments are insufficient for the claims in the architecture. At minimum, CIFAR-10 should be wired into experiment scripts.

---

### MISSING-6: Convergence Tracking and Evaluation Metrics

**Architecture Reference:** Section 6.4 — Evaluation Metrics

The following metrics are specified but not computed:

| Metric | Status |
|--------|--------|
| Model Accuracy | ✅ Computed in `run_pipeline.py` |
| Attack Success Rate (ASR) | ❌ Not computed — no backdoor trigger test |
| Byzantine Detection Rate (TPR) | ❌ Not computed — no ground-truth comparison |
| False Positive Rate (FPR) | ❌ Not computed |
| Convergence Speed | ❌ Not tracked |
| Communication Overhead | ❌ Not measured |
| Proof Verification Time | ⚠️ Logged in ZKP prover but not aggregated |
| Attribution Accuracy | ❌ Not computed |
| Evidence Quality | ❌ Not computed |

**Impact:** Without these metrics, the system cannot validate the claims in Sections 5.1-5.4 of the architecture.

---

### MISSING-7: FLTrust Baseline

**Architecture Reference:** Section 6.3 — "FLTrust: Server maintains root dataset for validation"

FLTrust is listed as a baseline comparison method but is not implemented. For research completeness, at least the primary baselines (Vanilla FL, Krum, Trimmed Mean, Median, FLTrust) should be available.

---

## 🟡 DESIGN CONCERNS (Functional but Suboptimal)

### CONCERN-1: Three Separate Orchestrators

The codebase has **three** separate orchestration systems:
1. `src/orchestration/pipeline.py` — `ProtoGalaxyPipeline` (used by `run_pipeline.py`)
2. `src/orchestration/protogalaxy_orchestrator.py` — `ProtoGalaxyOrchestrator` (alternative)
3. `src/orchestration/coordinator.py` — `FLCoordinator` (legacy)

Each implements a different subset of the architecture. `ProtoGalaxyPipeline` is the most complete and is used by the main runner. `ProtoGalaxyOrchestrator` adds ZKP proof integration but is not used by any runner script. `FLCoordinator` is a legacy implementation that doesn't use Merkle trees.

**Recommendation:** Consolidate into a single orchestrator that includes all features (Merkle trees + ZKP + full defense pipeline).

---

### CONCERN-2: Communication Layer Exists But Is Unused

The `src/communication/` package contains:
- `message.py` — Message types and serialization (fully implemented)
- `channel.py` — In-memory channels with queues, handlers, buffers (fully implemented)
- `client_comm.py` — Client communicator with registration and gradient submission (fully implemented)
- `server.py` — FL server with round management (fully implemented)
- `rest_api.py` — Galaxy API server with Flask endpoints (fully implemented, Flask-dependent)

**None of these are used in the pipeline.** The pipeline uses direct function calls instead of message passing. This is acceptable for a single-process research prototype, but it means the communication protocol specified in Architecture Section 3.4 is not exercised.

---

### CONCERN-3: `ByzantineClientSimulator` Trains with `None` DataLoader

**File:** `src/simulation/clients.py`, line 56

```python
def simulate_round(self, num_epochs: int = 1):
    self.client.train_local(None, 0)  # No real training
```

This calls `Trainer.train()` with `train_loader=None`, which returns `{'loss': 0.0, 'samples': 0}` and sets `_initial_weights = get_weights()`. Then `get_gradients()` returns all-zero gradients (since initial == current). The attack is then applied to zero gradients.

**Impact:** For `label_flip`, negating zeros gives zeros. For `model_poisoning`, scaling zeros gives zeros. Only `gaussian_noise` and `backdoor` (which add noise) produce non-zero gradients. This means the `ByzantineClientSimulator` produces unrealistic attacks for most attack types.

**Note:** `run_pipeline.py` handles Byzantine attacks directly on real gradients (lines 155-185), so the main pipeline is unaffected. This only affects the `simulation/clients.py` module.

---

### CONCERN-4: Gradient Sign Convention

**File:** `src/client/trainer.py`, lines 200-206

```python
gradients = [
    (init_w - curr_w).detach()  # ← init - current
    for init_w, curr_w in zip(self._initial_weights, current_weights)
]
```

And in `model_sync.py`:
```python
param.data -= learning_rate * grad.to(param.device)  # ← subtract gradient
```

Since `gradient = init_w - curr_w`, and the update is `param -= lr * gradient`, the effective update is `param -= lr * (init - current) = param + lr * (current - init)`, which moves the global model toward the locally-trained weights. This is correct FedAvg behavior. ✅

However, this means the "gradients" are actually **negative weight deltas** (pointing from current → initial), not true loss gradients. The label_flip attack (negating these) would push the model *toward* the locally-trained direction, which is the OPPOSITE of what a label-flip attack should do.

**Impact:** In `run_pipeline.py`, `gradients = [-g for g in gradients]` for label_flip would make `gradient = -(init - current) = current - init`, and `param -= lr * (current - init)` moves AWAY from local weights. This is correct attack behavior. ✅ (Verified — the convention is consistent.)

---

## ✅ VERIFIED WORKING COMPONENTS

### 1. Merkle Tree — `src/crypto/merkle.py`
- `MerkleTree`: Correct bottom-up construction with odd-leaf duplication
- `GalaxyMerkleTree`: Per-galaxy tree with client-to-leaf mapping
- `GlobalMerkleTree`: Two-level hierarchy from galaxy trees
- `GradientCommitment`: Binding commitment with nonce, timestamp, round_number
- `compute_hash`: SHA-256 with metadata binding
- `verify_proof`: Correct sibling-path reconstruction
- **Test:** Proof generation and verification are mathematically correct

### 2. Statistical Analyzer — `src/defense/statistical.py`
- 4-metric analyzer (norm, cosine direction, coordinate-wise, KL-divergence)
- Configurable thresholds matching architecture (k=3σ, cosine=0.5)
- Flagging rule: ≥2 of 4 metrics failed (matches architecture)
- KL-divergence uses histogram binning with epsilon smoothing ✅

### 3. Robust Aggregation — `src/defense/robust_agg.py`
- **Trimmed Mean:** Coordinate-wise sorting, symmetric trimming, correct O(n log n · d)
- **Multi-Krum:** Correct pairwise distance computation, k-nearest scoring, m-selection
- Both return flat numpy arrays, correctly flattened/reconstructed by pipeline

### 4. Reputation System — `src/defense/reputation.py`
- `BehaviorScore`: Exact weight vector from architecture (w1=0.1, w2=0.3, w3=0.4, w4=0.2) ✅
- `EnhancedReputationManager`: Quarantine/ban/rehabilitation state machine
- Status transitions: ACTIVE → QUARANTINED → BANNED (or REHABILITATED back to ACTIVE)
- Per-layer EWMA indicators for integrity, statistical, krum, historical

### 5. Layer 5 Galaxy Defense — `src/defense/layer5_galaxy.py`
- `GalaxyAnomalyDetector`: 3-check detection (norm, direction, cross-galaxy consistency)
- `GalaxyReputationManager`: EWMA reputation with consecutive low-rep streak tracking
- `AdaptiveReClusterer`: Galaxy dissolution with honest/malicious separation and round-robin redistribution
- `Layer5GalaxyDefense`: Full orchestration of detection → reputation → dissolution → isolation level
- 4-tier `IsolationLevel` enum matching architecture (NONE → CLIENT → PARTIAL → FULL → SYSTEM_WIDE)

### 6. Defense Coordinator — `src/defense/coordinator.py`
- Correctly sequences all 5 layers
- Full multi-indicator behavior scoring (Architecture Section 4.4)
- Forensic logging integration for quarantine/ban decisions
- Configurable aggregation method switching

### 7. ZKP Prover — `src/crypto/zkp_prover.py`
- Clean Rust bridge (`fl_zkp_bridge`) with graceful fallback
- `GradientSumCheckProver`: Per-layer sum folding (IVC approach)
- `GalaxyProofFolder`: Multi-client proof folding
- Fallback: SHA-256 commitment (not a real ZK proof, but documented and honest about it)
- No mocks or fake "always True" returns

### 8. Forensic Logger — `src/storage/forensic_logger.py`
- Evidence database with SHA-256 integrity hashing
- Query system with client/round/galaxy indices
- On-disk persistence with JSON evidence files
- `verify_evidence_integrity()` — cryptographic verification of stored evidence

### 9. Model Synchronizer — `src/orchestration/model_sync.py`
- Model versioning with hash tracking
- Client-side receiver with integrity verification
- `apply_update()`: Correct gradient descent application

### 10. Data Partitioning — `src/data/partition.py`
- `IIDPartitioner`: Random uniform split ✅
- `NonIIDPartitioner`: Label-based heterogeneous split ✅
- `DirichletPartitioner`: Dirichlet-α distribution ✅

---

## Summary of Required Actions

### Must Fix (Broken Code)

| # | Issue | File | Priority |
|---|-------|------|----------|
| BUG-1 | `Client.verify_merkle_root()` wrong parameter names | `src/client/client.py:81` | 🔴 Critical |
| BUG-2 | `CommitmentGenerator.get_commitment_proof()` references non-existent attribute | `src/client/commitment.py:37` | 🔴 Critical |
| BUG-3 | Global Merkle adapter passes unused metadata | `src/crypto/merkle_adapter.py:88` | ⚠️ Medium |

### Must Fix (Logic Errors)

| # | Issue | File | Priority |
|---|-------|------|----------|
| ISSUE-1 | Trust-weighted aggregation missing normalization `/ Σ(Rᵢ)` and applied too early | `src/orchestration/pipeline.py:380` | 🔴 Critical |
| ISSUE-2 | Reputation initializes to 1.0, architecture says 0.5 | `src/defense/reputation.py:87` | ⚠️ Medium |
| ISSUE-3 | Reputation EWMA not applied at top level (direct overwrite) | `src/defense/reputation.py:155` | 🔴 Critical |

### Should Implement (Missing From Architecture)

| # | Feature | Architecture Section | Priority |
|---|---------|---------------------|----------|
| MISSING-1 | Digital signatures / client key pairs | §2.2, §5.2 | 🔴 High |
| MISSING-2 | Explicit replay attack round-number validation | §4.1 | ⚠️ Medium |
| MISSING-3 | Client-side Merkle root verification in pipeline | §3.4 Phase 2 | ⚠️ Medium |
| MISSING-4 | Non-IID experiment scripts | §6.1 | ⚠️ Medium |
| MISSING-5 | CIFAR-10 experiments, FEMNIST, Shakespeare | §6.1 | 🟡 Low |
| MISSING-6 | ASR, TPR, FPR, convergence metrics | §6.4 | 🔴 High |
| MISSING-7 | Coordinate-Wise Median aggregator | §4.3 | 🟡 Low |
| MISSING-8 | FLTrust baseline | §6.3 | 🟡 Low |

### Should Consolidate (Design Debt)

| # | Issue | Recommendation |
|---|-------|---------------|
| CONCERN-1 | Three separate orchestrators | Merge ZKP integration from `ProtoGalaxyOrchestrator` into `ProtoGalaxyPipeline` |
| CONCERN-2 | Communication layer unused | Either integrate or clearly document as future work |
| CONCERN-3 | `ByzantineClientSimulator` trains on None | Require real data loader |
| CONCERN-6 | FLLogger `extra=` parameter silently ignored | Fix `_log()` to accept `extra` dict |

---

## File-by-File Audit Status

### `src/crypto/`

| File | Status | Notes |
|------|--------|-------|
| `merkle.py` | ✅ Correct | All tree operations verified. 481 lines of real crypto logic. |
| `merkle_adapter.py` | ⚠️ Minor issue | Unused metadata in global adapter (BUG-3) |
| `utils.py` | ✅ Correct | SHA-256, HMAC, nonce generation, constant-time compare |
| `zkp_prover.py` | ✅ Correct | Graceful Rust fallback, honest `is_real` flag, no mocks |

### `src/client/`

| File | Status | Notes |
|------|--------|-------|
| `client.py` | ⚠️ Bug | BUG-1: `verify_merkle_root()` crashes. All 5 attack types correctly implement architecture §6.2 |
| `commitment.py` | ⚠️ Bug | BUG-2: `get_commitment_proof()` crashes. `generate_commitment()` and `verify_commitment()` work correctly |
| `trainer.py` | ✅ Correct | Real PyTorch training with SGD, gradient extraction via weight diff, evaluation loop |
| `verifier.py` | ✅ Correct | Thin wrapper over `merkle.verify_proof` |

### `src/defense/`

| File | Status | Notes |
|------|--------|-------|
| `statistical.py` | ✅ Correct | All 4 metrics from architecture §4.2. Flagging threshold configurable. |
| `robust_agg.py` | ✅ Correct | Trimmed Mean (O(n log n · d)) and Multi-Krum (O(n² · d)) correctly implemented |
| `reputation.py` | ⚠️ Logic issue | ISSUE-2 (init value) and ISSUE-3 (EWMA not applied). BehaviorScore weights match spec. |
| `layer5_galaxy.py` | ✅ Correct | Full 3-check detection, reputation, re-clustering, 4-tier isolation |
| `coordinator.py` | ✅ Correct | Sequences all 5 layers, forensic logging integration |
| `coordinator_layer5_addon.py` | ✅ Correct | Layer 5 integration addon |

### `src/orchestration/`

| File | Status | Notes |
|------|--------|-------|
| `pipeline.py` | ⚠️ Issues | ISSUE-1 (trust weighting). Main pipeline - 937 lines, all 4 phases present |
| `protogalaxy_orchestrator.py` | ✅ Correct | ZKP-integrated alternative. 637 lines. Not used by main runner. |
| `model_sync.py` | ✅ Correct | Model versioning, hash verification, gradient application |
| `galaxy_manager.py` | ✅ Correct | Client assignments, reassignment, dissolution |
| `round_manager.py` | ✅ Correct | Phase lifecycle management (not used in main pipeline) |
| `coordinator.py` | ⚠️ Legacy | Doesn't use Merkle trees. Superseded by `pipeline.py`. |

### `src/aggregators/`

| File | Status | Notes |
|------|--------|-------|
| `galaxy.py` | ✅ Correct | Simple weighted averaging. Not used in main pipeline (ISSUE-4). |
| `global_agg.py` | ✅ Correct | Same. Not used in main pipeline. |

### `src/communication/`

| File | Status | Notes |
|------|--------|-------|
| `message.py` | ✅ Correct | Full message types, serialization, factory functions |
| `channel.py` | ✅ Correct | In-memory queues, thread-safe, message handler, buffer |
| `client_comm.py` | ✅ Correct | Client-side communication with callbacks |
| `server.py` | ✅ Correct | FL server with round management |
| `rest_api.py` | ✅ Correct | Flask-based API (optional dependency) |

**All communication modules are functional but unused in the pipeline (CONCERN-2).**

### `src/storage/`

| File | Status | Notes |
|------|--------|-------|
| `forensic_logger.py` | ✅ Correct | Evidence database with integrity verification, querying, export |
| `manager.py` | ✅ Correct | Model checkpointing, metrics persistence |

### `src/data/`

| File | Status | Notes |
|------|--------|-------|
| `datasets.py` | ✅ Correct | MNIST and CIFAR-10 loading with proper transforms |
| `partition.py` | ✅ Correct | IID, NonIID, Dirichlet partitioners |
| `loader.py` | ✅ Correct | DataLoader creation utilities |

### `src/models/`

| File | Status | Notes |
|------|--------|-------|
| `mnist.py` | ✅ Correct | Linear, MLP, CNN architectures for MNIST |
| `registry.py` | ✅ Correct | Model registry with factory pattern |

### `src/simulation/`

| File | Status | Notes |
|------|--------|-------|
| `runner.py` | ✅ Functional | Complete simulation framework |
| `clients.py` | ⚠️ Issue | `ByzantineClientSimulator` trains with `None` (CONCERN-3) |
| `metrics.py` | ✅ Correct | Metrics collection and CSV export |

### `src/logging/`

| File | Status | Notes |
|------|--------|-------|
| `__init__.py` | ⚠️ Minor issue | `extra=` parameter silently ignored. Otherwise comprehensive (JSON, CSV, round tracking) |

### `src/utils/`

| File | Status | Notes |
|------|--------|-------|
| `gradient_ops.py` | ✅ Correct | Flatten/unflatten, norm, similarity utilities |
| `stats.py` | ✅ Correct | Z-score, IQR outlier detection, distance matrix |
| `validation.py` | ✅ Correct | Input validation for gradients, weights, config |

---

## Conclusion

The ProtoGalaxy codebase is a **substantial and largely functional research prototype**. The core cryptographic infrastructure (Merkle trees, commitment-reveal protocol), the 5-layer defense framework, and the hierarchical FL pipeline are all present and fundamentally correct. There are no mocks, no `time.sleep` calls, and no simulation shortcuts in the critical path.

The most critical items to address are:

1. **BUG-1 and BUG-2** — Fix the two runtime crashes before any further testing
2. **ISSUE-1 and ISSUE-3** — Fix trust-weighted aggregation normalization and reputation EWMA to match the architecture's theoretical guarantees
3. **MISSING-1** — Add digital signatures (even a simple Ed25519 signing scheme) to close the attribution gap
4. **MISSING-6** — Implement evaluation metrics (TPR, FPR, ASR) to validate the claims

After these fixes, the system would be a credible research prototype suitable for generating experimental results consistent with the architecture's claims.
