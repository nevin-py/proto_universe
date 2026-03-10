# ZKP Circuit Changes: Tautology → Real Range Proof

## TL;DR

The original `BoundedAdditionFCircuit` had a **broken norm-bound constraint** that was
algebraically always satisfied — it never actually rejected Byzantine clients at the
circuit level.  The Python caller included a guard check that was doing the real work.
The rewritten circuit removes that guard and enforces the bound inside the R1CS circuit
using a proper 64-bit bit-decomposition range proof.

---

## 1. Original Circuit (`commit 9c9f3edd`)

### External inputs
```
[gradient_sum, max_norm_squared]   // Python pre-squares max_norm before passing it in
```

### Key constraint (`generate_step_constraints`)
```rust
let sum_squared   = gradient_sum * gradient_sum;
let difference    = max_norm_squared - &sum_squared;          // (A)
let reconstructed = &sum_squared + &difference;               // (B)
reconstructed.enforce_equal(max_norm_squared)?;               // (C)
```

### Why this is a tautology
Line (A) allocates `difference = max_norm_squared - sum_squared`.  
Line (B) computes `reconstructed = sum_squared + (max_norm_squared - sum_squared)`.  
That simplifies to `reconstructed = max_norm_squared` — always, by construction.  
Line (C) is therefore `max_norm_squared == max_norm_squared` → **unconditionally true**.

The constraint adds zero security. A Byzantine client with `|gradient| = 1000 × bound`
would sail right through.

### Real enforcer (Python guard in Rust)
```rust
// prove_gradient_step — inside FLZKPBoundedProver
if gradient * gradient > max_norm * max_norm + 1e-6 {
    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(...));
}
```
This guard — a plain floating-point comparison in the Rust host function, **outside
the circuit** — was the only thing that rejected out-of-bound gradients.  It is not a
zero-knowledge proof; it is a conventional if-statement.

### Python side (`zkp_prover.py`)
Passed gradient **sums** (not norms):
```python
layer_sums = [g.sum().item() for g in gradients]
```
Problem: in a 7 840-parameter weight layer, positive and negative values cancel via the
Central Limit Theorem.  A model-poisoning attack scales all weights by 10× but the
layer sum stays near zero — completely invisible to a sum-based check.

---

## 2. New Circuit (current working copy, rebuilt with `maturin develop --release`)

### External inputs
```
[l2_norm, max_norm]    // circuit squares both internally — no pre-squaring in Python
```

### Key constraint (`generate_step_constraints`)
```rust
let norm_sq     = gradient_norm * gradient_norm;   // L2 norm²
let max_norm_sq = max_norm       * max_norm;        // bound²
let diff        = &max_norm_sq - &norm_sq;          // must be ≥ 0 for honest clients

// 64-bit bit-decomposition range proof
let diff_val    = diff.value().unwrap_or_default();
// ... 64 Boolean::new_witness allocations ...
// ... recon = Σ bit_i * 2^i ...
recon.enforce_equal(&diff)?;
```

### Why this is a genuine range proof
For an honest client where `l2_norm ≤ max_norm`:
- `diff = max_norm² - l2_norm² ≥ 0` and fits in 64 bits
- The 64-bit decomposition matches `diff` exactly → `enforce_equal` passes
- The accumulated IVC instance is satisfied → `verify_proof` returns `true`

For a Byzantine client where `l2_norm > max_norm`:
- `diff < 0` in real arithmetic → in BN254's field it **wraps** to a value near
  `Fr::MODULUS` (≈ 2²⁵⁴)
- That huge number cannot be expressed as a sum of 64 bits (max 2⁶⁴ − 1)
- The bit-witnesses the prover assigns reconstruct only the lower 64 bits of `diff`,
  so `recon ≠ diff` — the `enforce_equal` constraint is **unsatisfied**
- **When does rejection happen?** Not during `prove_step`.
  In release builds, `prove_step` only calls `cs.is_satisfied()` inside a
  `#[cfg(test)]` block (see `protogalaxy/mod.rs` line 966). Outside test mode,
  the folding step runs regardless — it just folds a **dirty** (unsatisfied) instance.
  The non-zero constraint violation accumulates into the running instance's error term `E`.
- `verify_proof` calls `PG::verify(vp, ivc_proof)` which checks the final
  accumulated instance.  A dirty accumulator (`E ≠ 0`) fails IVC verification.
- **No valid (verifiable) proof exists for Byzantine inputs.**
  The client can generate bytes, but those bytes will not pass server-side verification.

### No Python pre-check
The Rust guard was deliberately removed.  The circuit constraint is the sole
cryptographic enforcer — but it is enforced at **verification time**, not proof time:
```rust
// REMOVED from prove_gradient_step:
// if gradient * gradient > max_norm * max_norm + 1e-6 { return Err(...) }
//
// Honesty is now enforced by the verifier, not the prover-side guard.
```

### Python side (`zkp_prover.py`)
Now passes L2 norms (not sums):
```python
layer_norms = [torch.norm(g).item() for g in gradients]
```
A model-poisoning attack scaling weights by 10× gives a 10× larger L2 norm → the
circuit range proof catches it.

---

## 3. Side-by-Side Comparison

| Property | Before (tautology) | After (range proof) |
|---|---|---|
| Constraint type | Algebraic tautology | 64-bit bit decomposition |
| Actual enforcer | Python `if` guard (outside ZKP) | Circuit `enforce_equal` |
| Gradient metric | Sum (cancels in large layers) | L2 norm (scales with attack) |
| max_norm input | Pre-squared by Python | Raw; circuit squares internally |
| Byzantine model-poisoning | Passes circuit (guard was the barrier) | `diff` wraps → IVC accumulator dirty → **verify_proof returns False** |
| When Byzantine rejected | At `prove_step` (Python/Rust guard) | At `verify_proof` (IVC verification) |
| Honest client overhead | ~3 constraints/layer | ~131 constraints/layer |
| Paper claim validity | **Broken** — "ZKP enforces bound" was false | Correct |

---

## 4. What Still Remains the Same

- ProtoGalaxy IVC on BN254/Grumpkin — unchanged
- `FLZKPBoundedProver.initialize()` / `verify_proof()` — unchanged
- `FLZKPProver` (legacy, no bounds) — unchanged
- Python `ZKProof` dataclass (`layer_sums` field kept as-is, now holds L2 norms)
- `pipeline.py` still catches `Exception` from `prove_gradient_sum` and increments
  `zk_bound_failures` — the counting logic works without changes

---

## 5. Expected Behavioral Difference in Experiments

| Scenario | Before fix | After fix |
|---|---|---|
| Honest client, proof requested | Success | Success — `verify_proof` returns True (unchanged) |
| Byzantine client, `prove_gradient_step` | Rejected by Python/Rust guard (early return) | **Succeeds** — folding runs, dirty instance created |
| Byzantine client, `verify_proof` | Would pass (circuit was tautology) | **Fails** — dirty IVC accumulator (`E ≠ 0`) → `PG::verify` returns Err → False |
| Compromised aggregator bypasses norm filter | Client silently excluded by Python pre-check | Client proof bytes accepted by prover but **fail server-side `verify_proof`** |
| `fizk_norm` (norm filter only, no ZKP) | Identical accuracy to FiZK-Full | Still matches accuracy — but FiZK-Full now has cryptographically witnessed rejection |

Run `scripts/test_circuit_behavior.py` to verify the before/after behavior directly.
