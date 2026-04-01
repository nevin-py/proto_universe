# ZKP Implementation Gap Report (Spec vs Code)

Date: 2026-04-01  
Scope: `architecture.md` vs current workspace implementation (`src/`, `sonobe/fl-zkp-bridge/`, runner scripts, tests).

---

## Executive Verdict

The project has a **real Sonobe-backed ProtoGalaxy proving and verification core** in Rust (`fl-zkp-bridge`), but the **end-to-end protocol described in `architecture.md` is not fully implemented** in the active Python orchestration path.

Most important gaps (updated after bridge work):

1. ~~Round orchestration imports~~ **`GradientSumCheckProver` / `GalaxyProofFolder` / `ZKProof` are defined** in `src/crypto/zkp_prover.py` and imported by orchestrators.
2. **Paper-style multi-instance ProtoGalaxy folding** (merging unrelated client proofs into one succinct IVC) is **still not available** in upstream Sonobe (`k = 1`; see [sonobe#82](https://github.com/privacy-scaling-explorations/sonobe/issues/82)). The **`PGFB` batch format** remains honest **O(N)** verification of **N** independent bundles.
3. **Decider (Groth16) finalization** is implemented as **`PGD1` bundles** (magic `PGD1`) when `prove_chunk` ran **at least twice**; verification is **O(1) in IVC length** for that chain. Single-step runs still emit the **legacy IVC bundle** because `DeciderEth::verify` requires `i > 1`.
4. Some **docs/tests** may still mention older wording; **`run_pipeline.py`** expects `FiZKPipeline` — now provided as an alias for `ProtoGalaxyPipeline`.

---

## What is correctly implemented

### 1) Real Sonobe ProtoGalaxy circuit + step proving

Implemented in `sonobe/fl-zkp-bridge/src/lib.rs`:

- `GradientFingerprintCircuit` implements `FCircuit`.
- State length is `7` via `state_len()`.
- Per-step constraints include all four products per element:
  - `r * g`
  - `g * g`
  - `g * ref`
  - `r * ref`
- Transition updates include `grad_fp_accum`, `norm_sq_accum`, `directional_fp_accum`, `ref_fp_accum`, and `step_count`.

This aligns with the intended 7-state accumulator model in your architecture document.

### 2) Real proof verification path in Rust

In `verify_proof_bundle_static`:

- **`PGD1`**: parses decider proof + verifier key + `i, z_0, z_i` + commitments, then **`ProtoGalaxyDeciderEth::verify`** (Groth16 + KZG).
- **Legacy** (first eight bytes are chunk marker + IVC layout): deserializes IVC proof, `vp_deserialize_with_mode`, **`PGGrad::verify`**.

So this is not a fake verification path.

### 3) Python strict no-fallback ZKP mode (current state)

In `src/crypto/zkp_prover.py`:

- Importing bridge is mandatory (throws on `ImportError`).
- `ModelAgnosticProver` hard-fails if bridge unavailable.
- Fallback verification/proving is disabled.

This is consistent with your requirement of “no fallbacks.”

---

## Critical gaps / missing technical pieces

### A) Orchestrator imports (resolved)

The symbols above **are present** in `src/crypto/zkp_prover.py` and wired from orchestrators.

### B) Folding logic vs architecture wording

`architecture.md` claims recursive folding
\[
\pi_g^{fold} = \mathcal{F}(\pi_1, \mathcal{F}(\pi_2, ...))
\]
and O(1) folded verification at server.

Current observable implementation reality:

- **`PGD1`**: one **succinct** verifier check per finished client chain (fixed-size Groth16 verify; cost does **not** scale with number of IVC folding steps).
- **`PGFB` batch**: packs N bundles; verifier runs **N** checks (**O(N)**). This is **not** a single merged proof; Sonobe ProtoGalaxy does not yet expose cryptographic aggregation of unrelated clients into one proof.
- `fold_proofs_bundle` / `verify_batch_bundle_static` in the bridge **accept both** legacy IVC sub-bundles and **`PGD1`** sub-bundles.

Impact:

- True **single-proof multi-client folding** as in a full recursive outer circuit is **not** implemented until upstream supports `k > 1` or a custom outer SNARK.

### C) Decider pipeline (implemented)

On `num_steps >= 2`, `generate_proof_bundle` runs **`ProtoGalaxyDeciderEth::prove`** and emits **`PGD1`**. `verify_proof_bundle_static` dispatches to **`PGDec::verify`** for that format.

### D) Round runner wiring (resolved)

`FiZKPipeline = ProtoGalaxyPipeline` is exported from `pipeline.py` for `run_pipeline.py`.

### E) Tests

`tests/test_zkp_prover.py` uses `GradientSumCheckProver` / `GalaxyProofFolder` / `ZKProof` and skips when the native bridge is missing. Run with `pytest` after `maturin develop` on `fl-zkp-bridge`.

---

## Spec-to-Code compliance matrix (high level)

| Architecture requirement                                 | Current status                                                                         |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 7-element IVC state                                      | ✅ Implemented in Rust circuit                                                         |
| Per-step constraints (r*g, g^2, g*ref, r\*ref)           | ✅ Implemented                                                                         |
| Strict reference anti-spoof tracking                     | ✅ Implemented (`ref_fp_accum` check vs `z_0[1]`)                                      |
| Real cryptographic verification in server path           | ✅ `PGGrad::verify` (legacy) + **`PGDec::verify` (`PGD1`)**                            |
| Decider-based final proof flow                           | ✅ **`PGD1`** when ≥2 steps; legacy IVC if 1 step                                      |
| Galaxy proof cryptographic folding (N clients → 1 proof) | ❌ Not in Sonobe ProtoGalaxy today; batch = O(N)                                       |
| O(1) verification w.r.t. IVC steps (per client)          | ✅ **`PGD1`** Groth16 path                                                             |
| O(1) verification of arbitrary multi-client batch        | ❌ **PGFB** remains O(N)                                                               |
| End-to-end round wiring to active ZKP classes            | ✅ Imports + `FiZKPipeline` alias                                                      |
| No-fallback operation                                    | ✅ Strict bridge path in `zkp_prover.py` (optional features may still exist elsewhere) |

---

## Remaining work for full “paper” multi-client folding

1. **Upstream or custom outer recursion** to fold unrelated ProtoGalaxy instances (`k > 1` or an aggregation SNARK), if the architecture requires a **single** verifier proof over many clients.
2. **Tests**: add an integration test that proves **≥2 chunks**, obtains **`PGD1`**, and verifies via `verify_proof_bundle_static`.
3. **Architecture wording**: distinguish **per-client succinct decider (PGD1)** from **cross-client proof aggregation**.

---

## Bottom line

- **Circuit, IVC, and Decider (Groth16) paths are implemented in the bridge** with real verification.
- **Batch `PGFB` is an honest container**, not paper-style folding of N clients into one proof.
- **Executable round wiring matches the ZKP module** via the `FiZKPipeline` alias and current `zkp_prover` API.
