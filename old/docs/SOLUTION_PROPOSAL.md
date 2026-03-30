# Solution: Fix One-Hot Encoding for IVC Compatibility

## Problem Summary

**Root Cause:** One-hot encoding witness variables for label `y` create cross-step inconsistencies in ProtoGalaxy IVC folding.

**Evidence:**
- ✓ Constant inputs (y=0, y=0): 2 steps work
- ✗ Varying inputs (y=5, y=0): Step 2 fails

## Why Current Approach Fails

### Current One-Hot Encoding (lib.rs:213-244)
```rust
// For each class k, allocate witness target_k
let target_k = FpVar::new_witness(cs, || {
    Ok(if y_val == k { F::one() } else { F::zero() })
});

// Constraints:
target_k * (1 - target_k) = 0        // Binary
target_k * (y - k) = 0                // Binding  
sum(target_k) = 1                     // Uniqueness
```

**Issue:** `target_k` witnesses persist across IVC folding steps. When y changes:
- Step 1: target[5] = 1 (for y=5)
- Step 2: target[0] should = 1 (for y=0)
- But ProtoGalaxy accumulates constraints from both steps
- Witness values conflict → folding polynomial has non-zero remainder

## Solution Options

### Option 1: Indicator Function (Field Arithmetic)
**Replace witness-based one-hot with deterministic field arithmetic**

```rust
// Compute indicator without witness allocation
// For each k: is_k = Π_{m≠k} (y - m) / Π_{m≠k} (k - m)
// This is Lagrange basis polynomial

let mut targets = Vec::new();
for k in 0..NUM_CLASSES {
    let k_f = FpVar::constant(F::from(k as u64));
    
    // Numerator: (y-0)(y-1)...(y-(k-1))(y-(k+1))...(y-9)
    let mut num = FpVar::one();
    for m in 0..NUM_CLASSES {
        if m != k {
            let m_f = FpVar::constant(F::from(m as u64));
            num = &num * (y_label - &m_f);
        }
    }
    
    // Denominator: (k-0)(k-1)...(k-(k-1))(k-(k+1))...(k-9)
    let denom = precompute_lagrange_denom(k); // Constant
    let denom_inv = FpVar::constant(denom.inverse().unwrap());
    
    let target_k = &num * &denom_inv;
    targets.push(target_k);
}

// No additional constraints needed - target_k is deterministic
```

**Pros:**
- No witness allocation → no cross-step conflicts
- Purely arithmetic → IVC-compatible
- Mathematically correct

**Cons:**
- 90 multiplications per step (9 factors × 10 classes)
- Increases constraint count

### Option 2: Direct MSE Without One-Hot
**Simplify by computing MSE directly using y as index**

```rust
// Instead of one-hot, directly compute error for class y
let y_idx = y_label.value()?.to_integer() as usize;

// Compute errors: error[k] = logit[k] - (k == y ? 1 : 0)
let mut errors = Vec::new();
for k in 0..NUM_CLASSES {
    let expected = if k == y_idx { FpVar::one() } else { FpVar::zero() };
    let error_k = &logits[k] - &expected;
    errors.push(error_k);
}

// MSE loss
let mut loss = FpVar::zero();
for error in &errors {
    loss = &loss + (error * error);
}
```

**Pros:**
- Minimal constraints
- No witness issues
- Simple and efficient

**Cons:**
- Uses `.value()` → breaks full ZK property
- Label y is revealed to verifier

### Option 3: Remove Label Verification
**Only prove forward pass and gradient computation, trust label**

```rust
// Remove one-hot encoding entirely
// Assume prover provides correct one-hot encoding as external input

let targets = &ext[OFF_TARGETS..OFF_TARGETS+NUM_CLASSES]; // 10 more inputs

// Only verify: sum(targets) = 1 and targets are binary
let mut sum = FpVar::zero();
for t in targets {
    t.enforce_equal(&(t * t))?; // Binary: t^2 = t
    sum = &sum + t;
}
sum.enforce_equal(&FpVar::one())?;

// Use targets directly for gradient
for k in 0..NUM_CLASSES {
    let error_k = &logits[k] - &targets[k];
    // ... rest of gradient computation
}
```

**Pros:**
- Minimal circuit changes
- IVC-compatible (no witness conflicts)
- Still proves gradient correctness

**Cons:**
- Malicious prover can lie about label
- Weaker security (doesn't verify data authenticity)

## Recommended Solution

**Use Option 1: Lagrange Indicator Function**

**Rationale:**
- Maintains full ZK property
- Proves label correctness cryptographically
- IVC-compatible (no witnesses)
- Acceptable overhead (~90 extra multiplications)

**Implementation:**
1. Precompute Lagrange denominators (constants)
2. Replace witness-based one-hot with field arithmetic
3. Remove binary/binding constraints (implied by construction)
4. Test with varying labels

## Implementation Plan

### Step 1: Precompute Lagrange Denominators
```rust
// Outside circuit, compute once
fn lagrange_denom(k: usize, num_classes: usize) -> i64 {
    let mut denom = 1i64;
    for m in 0..num_classes {
        if m != k {
            denom *= (k as i64 - m as i64);
        }
    }
    denom
}

// For NUM_CLASSES=10:
const LAGRANGE_DENOMS: [i64; 10] = [
    -362880,  // 0!(-1)(-2)...(-9)
    362880,   // 1!(1)(-1)(-2)...(-8)
    -181440,  // 2!(2)(1)(-1)...(-7)
    ...
];
```

### Step 2: Update TrainingStepCircuit
```rust
// Replace lines 213-244 with Lagrange indicators
let mut targets = Vec::with_capacity(NUM_CLASSES);
for k in 0..NUM_CLASSES {
    let target_k = compute_lagrange_indicator(
        cs.clone(), y_label, k, NUM_CLASSES
    )?;
    targets.push(target_k);
}
```

### Step 3: Test
```bash
python scripts/test_minimal_circuit.py
# Should now pass varying inputs test
```

## Expected Results

After fix:
- ✓ Single sample: works
- ✓ Multi-sample (same y): works  
- ✓ **Multi-sample (varying y): works**
- ✓ Full 4-sample batch: works

Overhead:
- Constraint count: +900 per step (90 mults × 10 classes)
- Proving time: +10-15% (acceptable for correctness)

## Alternative If Overhead Too High

If Lagrange approach is too expensive, use **Option 3 with additional Merkle binding**:
- Prover commits to (x, y) pairs via Merkle tree
- Circuit receives pre-encoded one-hot as external input
- Verifier checks Merkle proof binds y to committed value
- Circuit only proves gradient correctness given targets

This splits verification: Merkle proves data authenticity, ZK proves computation.
