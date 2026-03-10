"""
test_circuit_behavior.py
========================
Sanity-check for the ZKP circuit rewrite.

Three sections — each can be run independently:

  A. Pure-math proof that the OLD constraint was a tautology (instant, no Rust)
  B. Old Python guard behavior (honest vs Byzantine, instant)
  C. NEW circuit via fl_zkp_bridge — tests both directions:
       - honest client  → prove_gradient_step OK  → verify_proof returns True
       - Byzantine      → prove_gradient_step OK  → verify_proof returns False
                          (prove_step does NOT raise in release builds;
                           rejection is at IVC verification, not folding)

Important background (see CIRCUIT_CHANGES.md, Section 2):
  ProtoGalaxy's prove_step only calls cs.is_satisfied() inside #[cfg(test)]
  (mod.rs line 966).  In release builds the folding step always succeeds; the
  unsatisfied range-proof constraint corrupts the accumulated IVC instance (E!=0),
  which is caught by PG::verify inside verify_proof.

Run from workspace root:
    python scripts/test_circuit_behavior.py
"""

import math
import sys
import time

PASS  = "\033[92m[PASS]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
INFO  = "\033[94m[INFO]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A  —  Tautology proof (no Rust, instant)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("SECTION A  --  Old constraint was a tautology (pure math)")
print("=" * 62)

def old_constraint_always_passes(gradient_sum: float, max_norm: float) -> bool:
    """
    Mimics OLD generate_step_constraints:
        sum_sq        = gradient_sum^2
        difference    = max_norm^2 - sum_sq              # (A)
        reconstructed = sum_sq + difference              # (B) = max_norm^2 always
        enforce_equal(reconstructed, max_norm^2)         # (C) always True
    """
    max_norm_sq   = max_norm ** 2
    sum_sq        = gradient_sum ** 2
    difference    = max_norm_sq - sum_sq
    reconstructed = sum_sq + difference          # algebraically == max_norm_sq
    return math.isclose(reconstructed, max_norm_sq, rel_tol=1e-9)


cases = [
    ("Honest    -- norm = 0.5x bound", 0.5,   1.0),
    ("Honest    -- norm = bound",      1.0,   1.0),
    ("Byzantine -- norm = 2x bound",   2.0,   1.0),
    ("Byzantine -- norm = 100x bound", 100.0, 1.0),
]

all_a = True
for label, grad, bound in cases:
    result = old_constraint_always_passes(grad, bound)
    tag    = "tautology confirmed OK" if result else "UNEXPECTED rejection"
    print(f"  {PASS if result else FAIL}  {label:<44}  always_passes={result}  ({tag})")
    if not result:
        all_a = False

msg = "Old circuit accepted EVERY input including 100x Byzantine -- tautology confirmed."
print(f"\n  {PASS if all_a else FAIL}  {msg}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B  —  Old Python guard (the real enforcer before the fix)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("SECTION B  --  Old Python/Rust guard (host-level, not ZKP)")
print("=" * 62)

def old_guard_passes(gradient: float, max_norm: float) -> bool:
    """Mimics: if gradient^2 > max_norm^2 + 1e-6: return Err(...)"""
    return not (gradient * gradient > max_norm * max_norm + 1e-6)

for label, grad, bound in cases:
    passes   = old_guard_passes(grad, bound)
    expected = abs(grad) <= bound
    ok       = (passes == expected)
    print(f"  {PASS if ok else FAIL}  {label:<44}  guard_passes={passes}  (expected={expected})")

print(f"""
  {INFO}  Guard works but is NOT cryptographic -- a compromised aggregator
         or tampered pipeline.py can skip the check entirely. The old
         circuit constraint provided zero additional security.
""")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C  —  New circuit via fl_zkp_bridge (actual Rust calls)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("SECTION C  --  New circuit: prove then verify")
print("=" * 62)
print(f"""  {INFO}  Key design point:
         prove_gradient_step in release builds does NOT raise for Byzantine
         inputs -- cs.is_satisfied() is inside #[cfg(test)] in mod.rs:966.
         Byzantine rejection happens at verify_proof via IVC verification:
         the dirty accumulated instance (E != 0) fails PG::verify.
""")

try:
    import fl_zkp_bridge
    print(f"  {INFO}  fl_zkp_bridge imported OK\n")
except ImportError as e:
    print(f"  {FAIL}  Cannot import fl_zkp_bridge: {e}")
    print("         Run: cd sonobe/fl-zkp-bridge && maturin develop --release")
    sys.exit(1)


def run_circuit_test(
    label: str,
    gradient: float,
    max_norm: float,
    expect_prove_success: bool,  # True in release mode (even for Byzantine)
    expect_verify_valid: bool,   # True for honest, False for Byzantine
) -> bool:
    prover = fl_zkp_bridge.FLZKPBoundedProver()
    prover.initialize(0.0)

    # Step 1: prove
    t0 = time.time()
    prove_ok  = None
    prove_err = ""
    try:
        prover.prove_gradient_step(gradient, max_norm)
        prove_ok = True
    except Exception as e:
        prove_ok  = False
        prove_err = str(e)[:100]
    prove_t = time.time() - t0

    prove_correct = (prove_ok == expect_prove_success)
    prove_note    = f"  <- err: {prove_err}" if not prove_ok else ""
    print(f"  {PASS if prove_correct else FAIL}  {label}")
    print(f"         prove:  {'OK' if prove_ok else 'RAISED'} in {prove_t:.1f}s  "
          f"(expected {'OK' if expect_prove_success else 'RAISED'}){prove_note}")

    if not prove_ok and expect_prove_success:
        print(f"         {FAIL}  Unexpected prove failure -- cannot test verify")
        print()
        return False

    # Step 2: verify
    t1 = time.time()
    verify_result = None
    try:
        dummy_bytes = list(prover.generate_final_proof())
        verify_result = prover.verify_proof(dummy_bytes)
    except Exception as e:
        verify_result = False
        print(f"         verify raised: {str(e)[:100]}")
    verify_t = time.time() - t1

    verify_correct = (verify_result == expect_verify_valid)
    print(f"  {PASS if verify_correct else FAIL}  verify: {verify_result}  in {verify_t:.1f}s  "
          f"(expected {expect_verify_valid})")

    if not verify_correct:
        if expect_verify_valid and not verify_result:
            print(f"         {WARN}  Honest proof failed verify -- possible Pedersen/IVC setup issue")
        if not expect_verify_valid and verify_result:
            print(f"         {FAIL}  Byzantine proof PASSED verify -- range proof not working!")

    print()
    return prove_correct and verify_correct


print(f"  {INFO}  First call initializes Pedersen params (~10-30s per prover).\n")

results = []

results.append(run_circuit_test(
    "Honest    -- gradient=0.5, bound=2.0  (well within)",
    gradient=0.5, max_norm=2.0,
    expect_prove_success=True,
    expect_verify_valid=True,
))

results.append(run_circuit_test(
    "Honest    -- gradient=1.0, bound=1.0  (exactly at bound)",
    gradient=1.0, max_norm=1.0,
    expect_prove_success=True,
    expect_verify_valid=True,
))

results.append(run_circuit_test(
    "Byzantine -- gradient=1.5, bound=1.0  (1.5x over)",
    gradient=1.5, max_norm=1.0,
    expect_prove_success=True,   # release mode: prove_step folds silently
    expect_verify_valid=False,   # IVC verify catches dirty accumulator
))

results.append(run_circuit_test(
    "Byzantine -- gradient=10.0, bound=1.0  (10x, model-poisoning scale)",
    gradient=10.0, max_norm=1.0,
    expect_prove_success=True,
    expect_verify_valid=False,
))


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("Summary")
print("=" * 62)
passed = sum(results)
total  = len(results)
print(f"  Section A (tautology math):  {'CONFIRMED' if all_a else 'ERROR'}")
print(f"  Section C (circuit tests):   {passed}/{total} passed")
print()
if passed == total:
    print(f"  {PASS}  Circuit correctly provides IVC-level range proof.")
    print("         Honest proofs verify. Byzantine proofs fail verification.")
    print("         Rejection is at verify_proof, NOT at prove_gradient_step.")
else:
    print(f"  {FAIL}  Some tests failed. See CIRCUIT_CHANGES.md Section 2.")
    print()
    print("  If Byzantine verify=True: PG::verify is not catching the dirty")
    print("  accumulator. May need an explicit cs.is_satisfied() check outside")
    print("  #[cfg(test)] in prove_gradient_step.")
print()
