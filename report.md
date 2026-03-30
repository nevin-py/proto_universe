# FiZK Implementation Progress Report

## Current Status: Debugging PoT Circuit Failures

**Date:** 2026-03-30  
**Status:** 🔴 BLOCKED - Circuit constraint errors

---

## Problem Identified

### Error Summary
- **Symptom:** All clients fail PoT proof generation
- **Error:** `ProtoGalaxy(RemainderNotZero)` at IVC step 2/4
- **Scope:** 100% failure rate (honest + Byzantine clients)
- **Pattern:** Fails on 2nd sample regardless of label value

### Error Analysis
```
Client 0: IVC step 2/4 failed (y=2): ProtoGalaxy(RemainderNotZero)
Client 1: IVC step 2/4 failed (y=2): ProtoGalaxy(RemainderNotZero)
Client 2: IVC step 2/4 failed (y=8): ProtoGalaxy(RemainderNotZero)
Client 3: IVC step 2/4 failed (y=4): ProtoGalaxy(RemainderNotZero)
Client 4: IVC step 2/4 failed (y=1): ProtoGalaxy(RemainderNotZero)
```

**Root Cause Hypothesis:**
- `RemainderNotZero` indicates polynomial folding failure in ProtoGalaxy
- Suggests circuit constraints are inconsistent or external inputs malformed
- NOT a data poisoning issue (honest clients also fail)

### Root Cause Analysis

**Evidence from Tests:**
1. ✅ First IVC step succeeds → Circuit constraints are valid
2. ❌ Second IVC step fails → State transition issue
3. ✅ Fingerprint calculation works → Python/Rust match on init
4. ❌ Fails regardless of label value → Not one-hot encoding issue

**Hypothesis: Model Fingerprint Recomputation**

Looking at `lib.rs` lines 130-145 (circuit):
- Each step recomputes fingerprint from external inputs (w, b, r)
- Compares to state[0] (carried fingerprint)
- **Problem:** Floating-point precision in Python vs Field arithmetic in Rust

**Specific Issue:**
```python
# Python (zkp_prover.py:128-134)
fp = sum(int(r[k] * SCALE) * (int(b[k] * SCALE) + sum(int(w[k,j] * SCALE))))
```

```rust
// Rust (lib.rs:133-142)  
fp_computed = Σ r[k] * (b[k] + Σ w[k,j])  // Field arithmetic
```

**Mismatch:** Python computes as integers, Rust as field elements. After step 1:
- State carries original fingerprint (from initialize)
- Step 2 recomputes fingerprint → slight difference due to rounding
- Constraint `fp_computed == model_fingerprint` fails
- ProtoGalaxy folding fails with RemainderNotZero

### Attempted Fixes

1. ✅ **Relaxed fingerprint constraint** (i==0 only)
   - Implemented in lib.rs:162-176
   - RESULT: Did NOT fix the issue
   
2. ✅ **Switched to ProtoGalaxy**
   - User requested ProtoGalaxy-only
   - RESULT: Same error persists

### 🎯 ROOT CAUSE IDENTIFIED!

**Test 4: Minimal Circuit Tests** (test_minimal_circuit.py)
**Date:** 2026-03-30 13:45

| Test | Result | Details |
|------|--------|---------|
| All-zero inputs (2 steps) | ✅ **PASS** | ProtoGalaxy folding works! |
| Varying (x,y) inputs | ❌ FAIL | Step 2 fails when y changes |

**BREAKTHROUGH FINDING:**
- ✓ ProtoGalaxy folding mechanism works correctly
- ✓ Circuit constraints are satisfiable
- ✗ **Changing label `y` between IVC steps breaks one-hot encoding**

**Root Cause: One-Hot Encoding Cross-Step Inconsistency**

The circuit enforces one-hot encoding (lib.rs:213-244):
```rust
for k in 0..NUM_CLASSES {
    let target_k = witness(y == k ? 1 : 0);
    target_k * (1 - target_k) = 0;        // Binary constraint
    target_k * (y - k) = 0;                // Binding constraint
}
sum(target_k) = 1;                         // Exactly one
```

**Problem:**
- Step 1: y=5 → target[5]=1, constraints satisfied ✓
- Step 2: y=0 → target[0]=1, but ProtoGalaxy carries witness from step 1
- ProtoGalaxy folding tries to fold constraints with different y values
- `target_k * (y - k) = 0` becomes inconsistent across folding
- Result: RemainderNotZero error

**Solution Required:**
Remove cross-step witness dependencies in one-hot encoding, OR
use different constraint structure that doesn't carry witnesses

---

## Implementation Progress

### ✅ Completed (Before Circuit Debug)
- [x] FiZK-PoT pipeline architecture (`fizk_pot_pipeline.py`)
- [x] Merkle tree integration
- [x] Experiment scripts (comprehensive + ablation)
- [x] Analysis scripts (tables + figures)
- [x] Documentation (EXPERIMENTS_README.md, PAPER_REVISION_SUMMARY.md)
- [x] Multi-baseline support (Vanilla, Multi-Krum, Median, etc.)

### 🔴 Blocked (Circuit Issues)
- [ ] TrainingStepCircuit functional
- [ ] PoT proof generation working
- [ ] End-to-end validation test passing
- [ ] Experiment execution

### 📋 TODO (After Circuit Fix)
- [ ] Run ablation study
- [ ] Run comprehensive evaluation
- [ ] Generate paper tables/figures
- [ ] Update paper sections
- [ ] CIFAR-10 integration
- [ ] ResNet-18 experiments

---

## Debugging Plan

### Phase 1: Isolate Circuit Issue ✅ IN PROGRESS
- [x] Read test logs (test1.log)
- [ ] Create minimal 1-client test
- [ ] Test with simple AdditionFCircuit (verify IVC works)
- [ ] Add verbose logging to circuit
- [ ] Test with different model architectures

### Phase 2: Fix Circuit Constraints
- [ ] Fix label encoding (float vs field element)
- [ ] Verify fingerprint calculation matches circuit
- [ ] Check state initialization
- [ ] Test constraint satisfaction manually

### Phase 3: Validation
- [ ] Single-client test passes
- [ ] Multi-client test passes (honest only)
- [ ] Byzantine detection test passes
- [ ] Full pipeline test passes

### Phase 4: Resume Implementation
- [ ] Run experiments
- [ ] Generate results
- [ ] Update paper

---

## Test Results Log

### Test 1: Multi-client PoT Pipeline (FAILED)
**Date:** 2026-03-30 12:30  
**Config:** 5 clients, 2 Byzantine, 4 PoT samples, 3 rounds  
**Result:** ❌ FAILED - All clients fail at IVC step 2/4

**Observations:**
- Merkle tree builds successfully
- Phase 1 (commitment) works
- Phase 2 (PoT generation) fails for ALL clients
- Error appears at 2nd sample (step 2/4)
- No clients verified → no aggregation → no learning

### Test 2: Single-Client Debug Suite (Nova) 
**Date:** 2026-03-30 12:36  
**Results:** Single sample ✅ (800 bytes, 1426ms), Multi-sample ❌ at step 2

### Test 3: ProtoGalaxy Implementation
**Date:** 2026-03-30 13:30  
**Changes:** Switched from Nova to ProtoGalaxy folding
**Results:**

| Test | Result | Details |
|------|--------|---------|
| Single Sample | ✅ **PASS** | 1 sample (4.4MB proof, 1447ms) |
| Multi Sample (4) | ❌ FAIL | Fails at step 2/4 |
| Linear Architecture | ❌ FAIL | Fails at step 2/2 |

**CRITICAL FINDINGS:**
- ✅ Single IVC step works with ProtoGalaxy
- ❌ Second IVC step STILL fails with RemainderNotZero
- ⚠️ Proof size: 4.4MB (ProtoGalaxy) vs 800B (Nova) - concerning
- **Fingerprint check fix (i==0) did NOT solve the issue**
- Problem is deeper than fingerprint recomputation

---

## Architecture Tests Needed

### Test 1: Simple Addition Circuit (IVC Baseline)
**Purpose:** Verify ProtoGalaxy IVC works at all  
**Circuit:** `AdditionFCircuit` (z' = z + x)  
**Status:** ⏳ PENDING

### Test 2: MNIST Linear Model
**Purpose:** Test TrainingStepCircuit with linear model  
**Model:** 784→10 linear layer  
**Status:** 🔴 FAILED (current)

### Test 3: MNIST MLP
**Purpose:** Test with multi-layer model  
**Model:** 784→128→64→10  
**Status:** ⏳ PENDING

### Test 4: MNIST CNN (LeNet-5)
**Purpose:** Test with convolutional model  
**Model:** LeNet-5 architecture  
**Status:** ⏳ PENDING

---

## Code Changes Made

### 1. Fixed Merkle Tree API
**File:** `src/orchestration/fizk_pot_pipeline.py`
**Changes:**
- Replaced `GlobalMerkleTreeAdapter` with `MerkleTree`
- Fixed `build_tree()` → direct `MerkleTree(data=...)`
- Updated proof verification logic

**Status:** ✅ Working (Merkle tree builds successfully)

### 2. Fixed Test Script
**File:** `scripts/test_pot_pipeline.py`
**Changes:**
- Fixed unbound variable `accuracy`
- Initialized `byzantine_caught`

**Status:** ✅ Script runs (but circuit fails)

---

## Solution Implementation

### Root Cause
One-hot encoding witness variables create cross-step conflicts in IVC folding when labels change.

### Proposed Fix
Replace witness-based one-hot with **Lagrange indicator polynomials** (deterministic field arithmetic).

### Implementation Status
- [x] Problem identified: One-hot witness conflicts
- [x] Solution designed (Lagrange indicators)
- [x] ✅ Implemented Lagrange indicators in circuit
- [x] ⚠️ Tested with varying labels - **STILL FAILS**

### Test Results After Lagrange Fix

**Test A (test_diagnose.py):** Simple values (0.0, 1.0)
- ✓ Change x: PASS
- ✓ Change y: PASS  
- ✓ Both: PASS

**Test B (test_exact_fail.py):** Realistic values (w=0.1, x=1.0)
- ✗ Exact failing scenario: **STILL FAILS**
- ✗ Even with fingerprint=0: **STILL FAILS**

### New Finding: Constraint Complexity Issue

**Observation:** Simple values work, realistic values fail
- Lagrange indicators add 90 multiplications per step
- With W·x forward pass (~7840 mults), total is very high
- **ProtoGalaxy may hit constraint degree/size limits**

### Possible Root Causes

1. **Fingerprint still being checked** (need to verify i==0 works)
2. **ProtoGalaxy constraint limit exceeded** 
3. **Field arithmetic overflow** with large computations
4. **State accumulation issue** in grad_accum

### Next Actions
1. Verify fingerprint check only runs on i==0
2. Try simplified circuit (remove fingerprint check entirely)
3. Check ProtoGalaxy constraint limits
4. Consider alternative: Split into multiple smaller circuits

---

## Resources

### Key Files
- Circuit: `sonobe/fl-zkp-bridge/src/lib.rs` (lines 98-270)
- Prover: `src/crypto/zkp_prover.py` (lines 147-236)
- Test: `scripts/test_pot_pipeline.py`
- Logs: `test1.log`

### Error Documentation
- ProtoGalaxy RemainderNotZero: Polynomial folding failure
- Indicates: Constraint inconsistency or malformed inputs
- Source: `folding-schemes` crate error types

---

## Questions to Answer

1. ✅ Does Merkle tree work? → YES
2. ❓ Does basic IVC folding work? → TEST WITH AdditionFCircuit
3. ❓ Is label encoding correct? → CHECK float vs field
4. ❓ Is fingerprint calculation correct? → VERIFY MATCH
5. ❓ Are constraints satisfiable? → DEBUG CIRCUIT

---

*Last Updated: 2026-03-30 12:36*
