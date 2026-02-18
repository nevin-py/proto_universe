# Architecture v2 Audit Report: Document vs. Codebase Validation

**Document Audited**: `protogalaxy_architecture(2).md` (1,272 lines, 10 sections + appendices)  
**Codebase**: ProtoGalaxy FL framework (Python) + Sonobe fl-zkp-bridge (Rust/PyO3)  
**Date**: 2025-07-17  
**Scope**: Read-only validation — no code modifications  

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Similarity** | **~72%** |
| Fully Matched Claims | 18 / 30 |
| Partial Matches (implemented but different) | 7 / 30 |
| Missing / Unimplemented | 5 / 30 |
| Critical Discrepancies | 5 |
| Moderate Discrepancies | 6 |
| Minor Discrepancies | 4 |

The architecture document describes an **aspirational design** that significantly overclaims what the ZKP subsystem actually proves. The FL pipeline, defense layers, Merkle commitment, and galaxy hierarchy are well-implemented (~90% match). The ZKP pipeline description is the largest gap — the document claims proofs of **computation correctness** and **dataset commitment**, but the code only proves **gradient sum correctness + norm bounds**. There is no global-level ProtoGalaxy folding; only galaxy-level IVC folding is implemented.

---

## Section-by-Section Validation

### 1. Abstract & Introduction (Sections 1–2)

| Claim | Status | Notes |
|-------|--------|-------|
| Hierarchical galaxy-based FL architecture | ✅ Match | `pipeline.py` implements 3-level hierarchy (Global → Galaxy → Client) |
| Clients organized into "galaxies" (trust domains) | ✅ Match | Round-robin assignment in `_assign_clients_to_galaxies()` |
| ZKP proving correctness of local training | ❌ **Overclaim** | ZKP proves gradient sum + norm bounds only, NOT SGD execution or dataset commitment |
| ProtoGalaxy/Nova folding-based proof aggregation | ⚠️ Partial | Galaxy-level IVC folding exists; **no global ProtoGalaxy folding** |
| O(1) final verification cost | ⚠️ Partial | Galaxy proof is O(1) verified; no global-level O(1) proof |
| Merkle tree-based commitment | ✅ Match | SHA-256, 2-level (client→galaxy, galaxy→global) |
| Multi-layer Byzantine defense | ✅ Match | 5 layers implemented |
| Groth16 or Plonky2 proof systems | ❌ **Wrong** | Actual system uses ProtoGalaxy IVC on BN254/Grumpkin curves |
| Up to α=30% Byzantine tolerance | ✅ Match | Tested at 20–50% in evaluations |

**Summary**: Core architecture matches. ZKP claims are significantly overspecified relative to implementation.

---

### 2. Architecture Overview (Section 3)

#### 3.1 Hierarchical Structure

| Component | Doc Claim | Code Reality | Status |
|-----------|-----------|-------------|--------|
| 3-level hierarchy | Global → Galaxy → Client | Identical | ✅ |
| Galaxy count | G galaxies | Configurable via `config.yaml` (`num_galaxies=3`) | ✅ |
| Client-to-galaxy assignment | Balanced partitioning | Round-robin in `pipeline.py` | ✅ |
| Galaxy aggregator | Per-galaxy aggregation + proof folding | `_galaxy_aggregate()` + `GalaxyProofFolder` | ✅ |

#### 3.2 Two-Level Merkle Tree

| Component | Doc Claim | Code Reality | Status |
|-----------|-----------|-------------|--------|
| Per-galaxy Merkle tree | Gradient commitments as leaves | SHA-256 tree in `src/crypto/merkle.py` | ✅ |
| Global Merkle tree | Galaxy roots as leaves | Implemented in `phase1_global_collect_galaxy_roots()` | ✅ |
| Leaf content | `H(gradient ∥ ZK_proof ∥ round ∥ timestamp ∥ nonce)` | `H(gradient ∥ metadata)` — **no ZK proof hash in leaf** | ⚠️ Partial |
| Merkle proof verification | O(log n) inclusion proof | Implemented in `verify_proof()` | ✅ |

**Discrepancy**: The document claims the Merkle leaf includes the ZK proof hash (`πᵢᶻᵏ`). In the code, the leaf commitment is computed from the gradient and metadata (round, timestamp, nonce) but does **not** include the ZK proof hash. The ZK proof is verified separately in Phase 2 (`phase2_verify_zk_proofs()`).

#### 3.3 Communication Protocol — Phase Structure

| Doc Phase | Doc Description | Code Implementation | Status |
|-----------|----------------|---------------------|--------|
| Phase 1 | Local Training + ZK Proof Generation | Phase 1: COMMITMENT — clients train, commit, build Merkle trees, **and** generate ZK proofs | ⚠️ Merged |
| Phase 2 | Commitment Collection + Merkle Construction | Merged into Phase 1 (Steps 1b, 1c) | ⚠️ Merged |
| Phase 3 | Revelation + ZK Proof Verification | Phase 2: REVELATION — submit gradients, verify Merkle+ZKP | ✅ Renamed |
| Phase 4 | IVC Folding at Galaxy Level | Phase 4: GLOBAL AGG — folding occurs as Step 4e | ⚠️ Reordered |
| Phase 5 | ProtoGalaxy Global Folding + Aggregation | **Not implemented** — no global-level folding | ❌ Missing |

**Actual Code Phases** (from `execute_round()` in `pipeline.py`):

| Code Phase | What Happens |
|------------|-------------|
| **Phase 1: COMMITMENT** | (1a) Clients train + commit, (1b) Galaxy Merkle trees, (1c) Global Merkle tree, (1d) ZK proof generation |
| **Phase 2: REVELATION** | (2a) Submit gradients, (2b) Merkle verify + quarantine + replay check, (2c) ZK proof verification |
| **Phase 3: DEFENSE** | (3) Per-galaxy defense pipeline (statistical + Krum + reputation) |
| **Phase 4: GLOBAL AGGREGATION** | (4a) Galaxy Merkle verify, (4b) Layer 5 galaxy defense, (4c) Global aggregation, (4d) Model update, (4e) Fold galaxy ZK proofs, (4f) Distribute model |

**Key difference**: Document describes **5 phases**; code implements **4 phases**. Training and commitment are combined into Phase 1. Folding happens at the end of Phase 4, not as a separate Phase 5. No global-level ProtoGalaxy folding exists.

---

### 3. ZKP Pipeline and Folding Scheme (Section 4)

This section has the **largest discrepancy** between document and code.

#### 4.1 Client-Side ZK Proof — What Is Actually Proved

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Proves: "I possess dataset Dᵢ committed to hash H(Dᵢ)" | **Not proved** — no dataset commitment circuit | ❌ **Major Gap** |
| Proves: "I ran T steps of SGD on w(r-1) with Dᵢ" | **Not proved** — no SGD computation circuit | ❌ **Major Gap** |
| Proves: "‖∇wᵢ(r)‖₂ lies within valid range [0, B]" | ✅ Proved — `gradient_sum² ≤ max_norm_squared` | ✅ |
| Proves gradient sum correctness | ✅ Proved — `z_{i+1} = z_i + gradient_sum` | ✅ |

**What the actual ZKP circuit does** (from `BoundedAdditionFCircuit` in `sonobe/fl-zkp-bridge/src/lib.rs`):

1. **Constraint 1**: `z_{i+1} = z_i + gradient_sum` (accumulation correctness)
2. **Constraint 2**: `gradient_sum² ≤ max_norm_squared` (norm bound via field arithmetic)

That's it. Two constraints. No forward/backward pass encoding, no dataset commitment, no SGD correctness proof.

**Impact**: The document's core narrative — "cryptographically prove that each client actually performed correct local training" — is **not what the implementation does**. The ZKP proves: *"I claim this gradient sum, and its squared norm doesn't exceed the server-set bound."* This is useful for norm enforcement but is categorically different from proving computation correctness.

#### 4.2 Proof System

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Groth16 SNARK (Option 1) | Not used | ❌ |
| Plonky2 STARK (Option 2) | Not used | ❌ |
| Actual system | ProtoGalaxy IVC folding on BN254/Grumpkin curves via Sonobe | — |
| R1CS constraint system | ✅ `BoundedAdditionFCircuit` is R1CS | ✅ |
| Trusted setup | IVC does not require per-circuit trusted setup | Different |

#### 4.3 Galaxy-Level IVC Folding

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Galaxy folds client proofs into single π_g^fold | ✅ `GalaxyProofFolder.fold_galaxy_proofs()` | ✅ |
| O(1) verification of galaxy proof | ✅ IVC proof is constant-size | ✅ |
| Folding is sequential within galaxy | ✅ Sequential proving of all layer sums | ✅ |
| Galaxies fold in parallel | ✅ Independent per-galaxy folding | ✅ |
| Folds *existing* proofs incrementally | ⚠️ **Re-proves from scratch** — creates new prover, feeds all layer sums | ⚠️ |

**Subtlety**: The code doesn't fold existing client proof instances. It creates a new `FLZKPBoundedProver`, sets z₀=0, and sequentially proves each client's layer sums as IVC steps. The output proof is constant-size (O(1)), but the process is O(N·L) proving time, not O(N) folding time.

#### 4.4 Global-Level ProtoGalaxy Folding

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Global aggregator folds galaxy proofs into single Π(r) | **NOT IMPLEMENTED** | ❌ **Missing** |
| Single O(1) global proof covering all n client computations | **NOT IMPLEMENTED** | ❌ **Missing** |
| Any third party can verify entire round in O(1) | **NOT IMPLEMENTED** | ❌ **Missing** |
| Proof chain Π(1) → Π(2) → ... → Π(r) | **NOT IMPLEMENTED** | ❌ **Missing** |

**Status**: The global-level ProtoGalaxy folding described in Section 4.3 of the document and Phase 5 of the protocol **does not exist in the codebase**. Galaxy proofs are folded independently but never combined into a global proof. The pipeline's `phase4_fold_galaxy_zk_proofs()` folds proofs per galaxy only.

#### 4.5 ZKP-Merkle Integration

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Merkle leaf includes ZK proof hash | ❌ No ZK proof hash in leaf | ❌ |
| Gradient accepted only if Merkle + ZKP pass | ✅ Both checked in Phase 2 | ✅ |
| Failed ZKP = hard rejection | ✅ ZK-rejected clients removed from verified set | ✅ |

---

### 4. Multi-Layer Byzantine Defense Framework (Section 5)

#### Layer 1: Cryptographic Integrity (Merkle + ZKP)

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Two-stage: Merkle verification + ZK proof verification | ✅ `phase2_galaxy_verify_and_collect()` + `phase2_verify_zk_proofs()` | ✅ |
| Catches: tampering, replay, false attribution | ✅ Replay detection via nonce + round tracking | ✅ |
| Catches: "gradients not derived from legitimate training" | ❌ ZKP only checks norm bounds, not training correctness | ❌ |
| O(1) global folded proof verifier | ❌ No global proof | ❌ |

**Note**: The `DefenseCoordinator`'s docstring claims it handles Layer 1 (crypto integrity), but **it doesn't** — Layer 1 is entirely handled in `pipeline.py` Phase 2. The coordinator's "Layer 1" is actually statistical norm detection, which is a misnomer in the code.

#### Layer 2: Statistical Anomaly Detection

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| 4 metrics: norm, direction, coordinate-wise, KL-divergence | ✅ All 4 in `StatisticalAnalyzer` (`src/defense/statistical.py`) | ✅ |
| Norm: flag if \|‖∇wᵢ‖₂ − μ\| > k·σ | ⚠️ Uses **MAD** (median absolute deviation), not σ | ⚠️ |
| Direction: cosine similarity < θ_min | ✅ Implemented | ✅ |
| Coordinate-wise: median + MAD based | ✅ Implemented | ✅ |
| KL-divergence for distribution shift | ✅ Implemented | ✅ |
| Flag if fails ≥2 of 4 metrics | ✅ `min_failures_to_flag = 2` | ✅ |

**Discrepancy**: Document says norm analysis uses "standard deviation σ" — code uses **MAD** (robust to outliers, better for Byzantine settings). This is arguably a code improvement over the document.

#### Layer 3: Byzantine-Robust Aggregation

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Multi-Krum (primary) | ✅ Implemented in `src/aggregators/` | ✅ |
| Coordinate-Wise Median (CWMed) | ✅ Implemented | ✅ |
| Trimmed Mean | ✅ Implemented | ✅ |
| FLTrust | ✅ Implemented (not in doc, bonus) | ✅+ |
| Operates on ZK-verified gradients only | ✅ ZK rejection happens before defense pipeline | ✅ |

#### Layer 4: Reputation-Based Adaptive Trust

| Aspect | Doc Claim | Code Reality | Status |
|--------|-----------|-------------|--------|
| Model | EWMA: R(t+1) = (1−λ)R(t) + λ·B(t) | ✅ `EnhancedReputationManager` | ✅ |
| Initial score | R(0) = 0.5 | ✅ | ✅ |
| Formula indicators | **5 indicators** (ZKP, integrity, statistical, krum, historical) | **4 indicators** (integrity, statistical, krum, historical) — **no ZKP indicator** | ❌ **Mismatch** |
| Weights | w₁=0.35, w₂=0.1, w₃=0.25, w₄=0.2, w₅=0.1 | w₁=0.1, w₂=0.3, w₃=0.4, w₄=0.2 | ❌ **Mismatch** |
| ZKP hard gate | If I_zkp=0, B=0 regardless of others | **Not implemented** in `BehaviorScore` | ❌ |
| Quarantine threshold | 0.2 | ✅ 0.2 | ✅ |
| Ban threshold | 0.1 | ✅ 0.1 | ✅ |
| Trust-weighted aggregation | R-weighted average | Not used — standard aggregation | ❌ |

**Weight Comparison Table**:

| Indicator | Architecture | Code | Delta |
|-----------|-------------|------|-------|
| ZKP (w₁) | 0.35 | **ABSENT** | Major |
| Integrity (w₂) | 0.10 | 0.10 | — |
| Statistical (w₃) | 0.25 | 0.30 | +0.05 |
| Krum (w₄) | 0.20 | 0.40 | +0.20 |
| Historical (w₅) | 0.10 | 0.20 | +0.10 |

The code has no ZKP indicator in `BehaviorScore`. The 4 implemented weights sum to 1.0 (0.1+0.3+0.4+0.2), redistributing the ZKP weight across other indicators with Krum receiving the largest share.

#### Layer 5: Galaxy-Level Defense

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Galaxy anomaly detection | ✅ `Layer5GalaxyDefense` (532 lines) | ✅ |
| IVC proof gate (galaxy quarantine if proof fails) | ⚠️ Galaxy proof folding exists but no explicit "gate" that quarantines galaxy on proof failure | ⚠️ |
| Norm-based galaxy detection | ✅ | ✅ |
| Direction-based galaxy detection | ✅ | ✅ |
| Cross-galaxy consistency | ✅ | ✅ |
| ZKP rejection rate tracking | ⚠️ Not directly checked at Layer 5 | ⚠️ |
| Galaxy reputation (EWMA) | ✅ | ✅ |
| Adaptive re-clustering | ✅ | ✅ |
| 4-tier isolation levels | ✅ | ✅ |

---

### 5. Norm Bound Computation

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Norm bound B as fixed parameter | **Adaptive**: `median(L2_norms) + k·MAD(L2_norms)` per layer | ⚠️ Better than doc |
| Server-side computation | ✅ `_compute_global_norm_bounds()` in `pipeline.py` | ✅ |
| k factor | Not specified in doc | k=3.0 (default `norm_scale_factor`) | — |
| Per-layer bounds | Not specified in doc | ✅ Each model layer gets own bound | ✅ |
| Uses L2 norms | ✅ | ✅ L2 norms explicitly, not scalar sums | ✅ |

**Note**: The actual norm bound computation is **more sophisticated** than the document describes. The doc treats B as a static parameter; the code dynamically computes per-layer bounds using robust statistics (median + k·MAD), which is strictly better.

---

### 6. Commit-Reveal Protocol

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Phase 1: commit only, Phase 3: reveal | ✅ Structurally separated in code | ✅ |
| Prevents adaptive attacks | ⚠️ Simulation artifact — server holds both commitment and gradient from Phase 1 | ⚠️ |
| Merkle root published before revelation | ✅ Merkle trees built in Phase 1, gradients submitted in Phase 2 | ✅ |

**Simulation Caveat**: In `execute_round()`, the server computes gradients and commitments in the same loop (Phase 1). The commit-reveal separation is architecturally present but not cryptographically meaningful in this single-process simulation. In a real distributed deployment, the server would not hold gradients until the reveal phase.

---

### 7. Forensic Logging

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| Evidence database for failed ZK proofs | ✅ `ForensicLogger` (374 lines) with SHA-256 integrity hashing | ✅ |
| Quarantine/ban logging | ✅ `log_quarantine()`, `log_ban()` | ✅ |
| Per-layer defense results | ✅ All 4 layer results stored per evidence record | ✅ |
| Query API | ✅ `query_evidence()` with `ForensicQuery` dataclass | ✅ |
| Evidence integrity verification | ✅ `verify_evidence_integrity()` re-hashes evidence | ✅ |
| Timeline export | ✅ `export_timeline()` | ✅ |

**Verdict**: Forensic logging is **fully implemented** and slightly exceeds the document's description.

---

### 8. Models and Datasets (Section 7)

| Doc Claim | Code Reality | Status |
|-----------|-------------|--------|
| ResNet-18 | ✅ `ResNet18` in `src/models/` | ✅ |
| CNN (MNIST/CIFAR) | ✅ `MNISTCNN`, `CIFAR10CNN` | ✅ |
| MLP | ✅ `MLP` | ✅ |
| Linear | ✅ `LinearRegression` | ✅ |
| LSTM | ❌ Not implemented | ❌ |
| MNIST | ✅ | ✅ |
| CIFAR-10 | ✅ | ✅ |
| FEMNIST | ❌ Not implemented | ❌ |
| Shakespeare | ❌ Not implemented | ❌ |
| IID partitioning | ✅ | ✅ |
| Non-IID (label-based) | ✅ | ✅ |
| Dirichlet partitioning | ✅ | ✅ |

---

### 9. Config System

| Doc Parameter | Config Default | Code Default | Status |
|---------------|---------------|-------------|--------|
| `defense.layer1_threshold` | — | 2.0 (config) vs 3.0 (code default) | ⚠️ |
| Reputation weights | w₁–w₅ in doc | Hardcoded in `BehaviorScore`, not configurable | ❌ |
| ZKP parameters | norm_scale_factor, use_bounds | Not exposed in config.yaml | ❌ |
| Quarantine/ban thresholds | 0.2, 0.1 | Hardcoded in `EnhancedReputationManager` | ⚠️ |

---

## Critical Discrepancies (Must Address in Paper)

### 1. ZKP Does NOT Prove Computation Correctness

**Document claims** (Section 4.1, repeated throughout):
> "Each client generates a succinct zero-knowledge proof... attesting that: (a) their gradient was computed on their committed local dataset, (b) the local training procedure was executed correctly"

**Reality**: The ZKP circuit (`BoundedAdditionFCircuit`) only has 2 constraints:
- `z_{i+1} = z_i + gradient_sum` (sum correctness)
- `gradient_sum² ≤ max_norm_squared` (norm bound)

There is **no dataset commitment**, **no SGD circuit**, and **no forward/backward pass encoding**. A malicious client could submit any gradient that satisfies the norm bound and produce a valid proof.

**Impact**: Sections 1, 3.4 Phase 1, 4.1, 5.1, and multiple theorems reference computation correctness guarantees that don't exist in the implementation.

### 2. No Global-Level ProtoGalaxy Folding

**Document claims** (Section 4.3, Phase 5):
> "Global aggregator applies the ProtoGalaxy folding scheme... producing a single global proof Π(r) of size O(1) covering the correctness of all n client computations"

**Reality**: Galaxy proofs are folded independently. There is no `Phase 5`, no global batch fold, no global proof Π(r), and no cross-round proof chain. The `phase4_fold_galaxy_zk_proofs()` function folds proofs within each galaxy only.

### 3. Proof System is Not Groth16/Plonky2

**Document claims** (Section 4.1): Groth16 or Plonky2 as candidate proof systems.

**Reality**: The system uses ProtoGalaxy IVC folding on BN254/Grumpkin curves via the Sonobe library. This is a fundamentally different approach — IVC folding avoids per-step SNARK generation entirely.

### 4. Reputation Scoring Formula Mismatch

**Document**: 5 indicators with ZKP as highest weight (0.35), hard gate on ZKP failure.  
**Code**: 4 indicators, no ZKP indicator at all, different weight distribution (Krum dominant at 0.40).

### 5. Phase Count Discrepancy

**Document**: 5 phases (Training, Commitment, Revelation, Galaxy Folding, Global Folding + Aggregation).  
**Code**: 4 phases (Commitment incl. training, Revelation, Defense, Global Aggregation incl. galaxy folding).

---

## Moderate Discrepancies

| # | Area | Document | Code |
|---|------|----------|------|
| 1 | Merkle leaf content | Includes ZK proof hash | Does not include ZK proof hash |
| 2 | Norm detection metric | Standard deviation σ | MAD (median absolute deviation) — code is better |
| 3 | Galaxy IVC folding | Folds existing proof instances | Re-proves from scratch using all layer sums |
| 4 | Trust-weighted aggregation | Reputation-weighted gradient average | Standard aggregation (reputation used for quarantine/ban only) |
| 5 | Layer 1 in coordinator | Crypto integrity (Merkle + ZKP) | Actually norm-based statistical (misnomer in code docstring) |
| 6 | Defense threshold defaults | Not specified | Config says 2.0, code defaults to 3.0 |

---

## Minor Discrepancies

| # | Area | Document | Code |
|---|------|----------|------|
| 1 | LSTM model | Listed | Not implemented |
| 2 | FEMNIST dataset | Listed | Not implemented |
| 3 | Shakespeare dataset | Listed | Not implemented |
| 4 | Config exposure | Reputation weights should be configurable | Hardcoded in `BehaviorScore` class |

---

## What Matches Well

1. **Hierarchical galaxy structure** — 3-level hierarchy with round-robin assignment
2. **2-level Merkle tree** — SHA-256, per-galaxy + global, with verification
3. **Commit-reveal protocol** — Structural separation of commit and reveal phases
4. **Statistical anomaly detection** — All 4 metrics (norm, cosine, coordinate, KL-divergence), min_failures=2
5. **Byzantine-robust aggregation** — Multi-Krum, CWMed, TrimmedMean + bonus FLTrust
6. **Reputation system** — EWMA model, quarantine/ban thresholds, gradual decay
7. **Galaxy-level IVC folding** — Per-galaxy proof folding via Rust/PyO3 bridge
8. **Layer 5 galaxy defense** — Anomaly detection, reputation, re-clustering, 4-tier isolation
9. **Forensic logging** — Complete implementation with integrity hashing and query API
10. **Adaptive norm bounds** — Server-side median + k·MAD (exceeds document specification)

---

## Recommendations for Paper Accuracy

1. **Rewrite Section 4.1** to accurately describe what the ZKP proves: gradient sum correctness and norm bounds — not computation correctness or dataset commitment. Frame these as **valuable but narrower guarantees** than full computation verification.
2. **Remove or qualify Section 4.3** (Global ProtoGalaxy Folding) — it's unimplemented. Either implement it or describe it as future work.
3. **Correct proof system references** from Groth16/Plonky2 to ProtoGalaxy IVC on BN254/Grumpkin.
4. **Update reputation formula** to match actual 4-indicator implementation with correct weights, or update the code to match the document.
5. **Adjust phase descriptions** to match the 4-phase implementation, or restructure code to match the 5-phase document.
6. **Update Merkle leaf** description to match actual content (no ZK proof hash).
7. **Change "standard deviation" to "MAD"** in Layer 2 statistical description (or note that MAD is used as a robust alternative).

---

## Appendix: File-to-Architecture Mapping

| Architecture Component | Primary Code File(s) |
|----------------------|---------------------|
| Pipeline / Phase Protocol | `src/orchestration/pipeline.py` |
| Merkle Tree | `src/crypto/merkle.py` |
| ZKP Prover (Python) | `src/crypto/zkp_prover.py` |
| ZKP Circuit (Rust) | `sonobe/fl-zkp-bridge/src/lib.rs` |
| Galaxy Proof Folder | `src/crypto/zkp_prover.py` (`GalaxyProofFolder`) |
| Defense Coordinator | `src/defense/coordinator.py` |
| Statistical Analyzer | `src/defense/statistical.py` |
| Reputation Manager | `src/defense/reputation.py` |
| Layer 5 Galaxy Defense | `src/defense/layer5_galaxy.py` |
| Multi-Krum Aggregator | `src/aggregators/multi_krum.py` |
| Trimmed Mean Aggregator | `src/aggregators/trimmed_mean.py` |
| CW Median Aggregator | `src/aggregators/coordinate_median.py` |
| FLTrust Aggregator | `src/aggregators/fltrust.py` |
| Forensic Logger | `src/storage/forensic_logger.py` |
| Models | `src/models/` |
| Data Partitioning | `src/data/` |
| Adaptive Attacker | `src/client/adaptive_attacker.py` |
| Rust PyO3 Bridge | `sonobe/fl-zkp-bridge/` + `sonobe-py-bindings/` |
| Config | `config.yaml` + `src/config/` |
