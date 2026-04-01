# Top-Tier Conference Essential Test Plan (Minimal but Sufficient)

Date: 2026-04-01  
Project: FiZK / ProtoGalaxy-based FL integrity defense  
Goal: Run only the experiments required to survive top-tier review (ML + FL + security), with no unnecessary grid explosion.

---

## 1) Core Claims You Must Prove

Only claim what is directly supported by the matrix below:

1. **Cryptographic validity**: honest proofs verify; tampered/mismatched proofs fail.
2. **Byzantine robustness**: FiZK improves robustness under strong attacks.
3. **Utility retention**: robustness does not collapse clean performance.
4. **Scalability**: proof/verification overhead remains practical at realistic client counts.
5. **Causality**: gains come from FiZK components (ablation-backed).
6. **Realism**: results hold on non-toy dataset/model and Non-IID client distributions.

---

## 2) Primary Metrics (must report)

- **Robustness**
  - Attack Success Rate (ASR) for backdoor/targeted settings
  - Byzantine detection Recall / Precision / F1
  - False Positive Rate (honest rejected)
- **Utility**
  - Final clean test accuracy
  - Accuracy drop from no-attack control
  - Convergence rounds to target accuracy
- **Crypto/System**
  - Proof generation time/client (ms)
  - Verification time (single + batch)
  - Proof size (bytes)
  - End-to-end round time (s)

Always report **mean ± std**, 95% CI, and number of seeds.

---

## 3) Essential Experimental Matrix

## A. Correctness & Soundness (must-have)

1. Honest proof verification pass rate (~100%)
2. Tampered bundle rejection (~100%)
3. Wrong `model_fp` rejection (~100%)
4. Replay/cross-client mismatch rejection
5. `PGFB` batch verification consistency

Use:

- `tests/test_zkp_soundness.py`
- `tests/test_zkp_prover.py`
- one end-to-end integration run with logging enabled

---

## B. Main robustness benchmark (paper primary table)

### B1. Non-toy benchmark (CRITICAL)

You must include **CIFAR-10** with a standard architecture:

- Preferred: `ResNet-18`
- Acceptable: `ResNet-9` (if compute constrained)

### B2. Data heterogeneity (CRITICAL)

Run both data regimes:

- IID
- Non-IID Dirichlet partition (`beta=0.5` and `beta=0.1`)

### B3. Minimal configuration set (necessary only)

- Datasets: `FashionMNIST` (sanity), `CIFAR-10` (primary)
- Models: `cnn` for FashionMNIST; `resnet18` or `resnet9` for CIFAR-10
- Clients: `20` (primary), `10` (stress check)
- Byzantine fraction: `0.2`, `0.3`, `0.5`
- Attacks: `sign_flip`, `gaussian_noise`
- Aggregators: `fedavg`, `multi_krum`

Recommended minimal runs:

- **FashionMNIST**: 8 runs (sanity + trend)
- **CIFAR-10**: 12 runs (core paper evidence)
- For each run: **3 seeds minimum** (5 preferred camera-ready)

---

## C. Utility retention controls

For both FashionMNIST and CIFAR-10:

- No attack (`alpha=0.0`) under IID + Non-IID
- Same training budget and model as section B
- Report:
  - clean final accuracy
  - delta vs attacked settings

Minimum: 6 runs × 3 seeds.

---

## D. Scalability figure (must expose O(N) verification honestly)

To avoid reviewer concern about hidden linear cost, extend client counts:

- Clients: `10, 50, 100, 200`
- Fixed: attack=`sign_flip`, alpha=`0.3`, model=primary model (`resnet18` if possible), aggregator=`multi_krum`
- Gradient sample sizes: `500, 1000, 2000` (or one fixed sample if compute-limited)

Report:

- total verify time vs clients
- per-client verify time
- proof size and total batch payload
- round wall-clock time

Minimum recommended: 4 client points × 3 seeds.

---

## E. Ablation (causal evidence)

High-impact ablations only:

1. Full FiZK
2. Remove directional bound
3. Remove norm bound
4. Remove both bounds (crypto binding only)

Evaluate on hardest setting:

- dataset=`CIFAR-10`
- model=`resnet18`/`resnet9`
- clients=`20`
- alpha=`0.5`
- attack=`sign_flip`
- Non-IID `beta=0.1`

Minimum: 4 variants × 3 seeds.

---

## 4) Baselines policy (implementation + literature)

## 4.1 Implemented baselines (run these)

- `FedAvg`
- `Multi-Krum`
- `Coordinate-wise median` (if stable)

Do not claim FLTrust unless your FLTrust pipeline is fully implemented end-to-end.

## 4.2 Literature cryptographic baseline (must include in text)

Even if not runnable in your repo, include comparison section against prior robust/cryptographic FL work (e.g., RoFL-style papers):

- threat model coverage
- proof system assumptions
- asymptotic complexity (prover/verifier)
- reported empirical overhead

Provide a table: **FiZK vs prior work** (theoretical + reported practical costs), with clear caveat if results are from published numbers.

---

## 5) Statistical Protocol (required)

- Use identical seeds across methods (paired comparisons)
- Report mean ± std and 95% CI
- Significance tests:
  - paired t-test for near-normal differences
  - Wilcoxon signed-rank otherwise
- Correct multiple comparisons (Holm-Bonferroni)
- Pre-register primary endpoint: recommended `F1 at alpha=0.3 on CIFAR-10 Non-IID beta=0.5`

---

## 6) Reproducibility Requirements

For each run store:

- full config JSON
- git commit hash
- seed
- environment (Python, CUDA/CPU, Rust toolchain)
- per-round metrics and final summary

Directory convention:

- `outputs/paper_essential/<timestamp>/<experiment_family>/...`

---

## 7) Execution Order (fast → strong)

1. Soundness checks (A)
2. CIFAR-10 IID robustness core (B)
3. CIFAR-10 Non-IID (`beta=0.5`, `0.1`) (B)
4. Utility controls (C)
5. Scalability to 200 clients (D)
6. Ablation on hardest Non-IID setting (E)
7. Literature comparison table finalization (Section 4.2)

If compute-limited, never skip steps 2, 3, or 5.

---

## 8) Command Guidance

Current runner supports essential knobs but may need extension for CIFAR and explicit Dirichlet in the same script path.

Use:

- `scripts/run_zkp_research.py` for existing matrix and timing instrumentation
- extend config builder to include:
  - dataset switch: `FashionMNIST` + `CIFAR-10`
  - model switch: `cnn` + `resnet18/resnet9`
  - partition switch: `iid` + `dirichlet(beta=0.5,0.1)`
  - client count list including `50,100,200`

Keep one YAML/JSON manifest file for all paper runs.

---

## 9) What NOT to do

- Do not submit with only toy datasets/models.
- Do not omit Non-IID FL experiments.
- Do not stop scalability plot at 40 clients when claiming practical deployment.
- Do not compare only to statistical baselines without cryptographic literature context.
- Do not overclaim O(1) global verification for `PGFB` if it is O(N) by design.

---

## 10) Submission checklist

- [ ] Soundness appendix with tamper/replay/model-fingerprint tests
- [ ] Main robustness table on CIFAR-10 (IID + Non-IID)
- [ ] Utility retention table (clean controls)
- [ ] Scalability figure up to 200 clients
- [ ] Ablation table on hardest setting
- [ ] Baseline table: implemented methods
- [ ] Literature comparison table vs cryptographic/robust FL papers
- [ ] Reproducibility appendix (configs/seeds/env/commit)

If all items above are completed with proper statistics, your evaluation will satisfy the core reviewer expectations for top-tier FL/security venues without unnecessary experiment sprawl.
