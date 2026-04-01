FiZK: Master Architecture Blueprint (Technical Specification)

System: Federated Integrity with Zero Knowledge (FiZK)
Core Technologies: ProtoGalaxy IVC, PyTorch, Rust (Sonobe/Arkworks), Multi-Krum
Objective: Verifiable, Byzantine-robust Hierarchical Federated Learning

1. Formal System Philosophy & Threat Model

FiZK addresses the Proof-of-Training Gap. Let $\mathcal{D}_i$ be the private dataset of client $i$, and $\mathcal{L}(w, \mathcal{D}_i)$ be the loss function. Fully Verifiable Training requires proving $\nabla w_i = \nabla \mathcal{L}(w^{(t)}, \mathcal{D}_i)$ inside a ZK-SNARK, resulting in $O(|\mathcal{D}_i| \cdot |w|)$ constraints, which is computationally intractable for edge devices.

Instead, FiZK functions as a cryptographic Property Checker. It enforces a hard bounded space $\Omega^{(t)}$ in $\mathbb{R}^d$ for round $t$.
Let $g_i \in \mathbb{R}^d$ be the submitted gradient. FiZK guarantees:

Magnitude: $\|g_i\|_2^2 \leq B_l^2$

Direction: $\langle g_i, \nabla w_{ref} \rangle \ge 0$

By restricting the Byzantine attacker's action space to $\Omega^{(t)}$ cryptographically (Layer 1), the downstream statistical aggregators (Layer 2) can successfully operate under high Byzantine fractions ($\alpha \ge 0.50$) without succumbing to dimension-curse scale inflation or strict sign-inversion.

2. The Cryptographic Engine (Relaxed R1CS & ProtoGalaxy)

The integrity engine, GradientFingerprintCircuit<F>, is constructed over the BN254 elliptic curve scalar field $\mathbb{F}_p$, where $p \approx 2^{254}$. The circuit uses a uniform step function $F: z_i \times x_i \to z_{i+1}$ folded via ProtoGalaxy Incrementally Verifiable Computation (IVC).

2.1 The State Vector $\vec{z}_i \in \mathbb{F}_p^7$

The IVC step circuit maintains a strict 7-element state accumulator across every folding step $k$:

$z_i \leftarrow model\_fp$: Binds proof to the public global model hash.

$z_i \leftarrow ref\_grad\_fp$: Binds proof to the public reference gradient hash.

$z_i \leftarrow grad\_fp\_accum$: Accumulates $\sum (r_j \cdot g_j) \pmod p$.

$z_i \leftarrow norm\_sq\_accum$: Accumulates $\sum (g_j^2) \pmod p$.

$z_i \leftarrow dir\_fp\_accum$: Accumulates $\sum (g_j \cdot ref_j) \pmod p$.

$z_i \leftarrow ref\_fp\_accum$: Accumulates $\sum (r_j \cdot ref_j) \pmod p$.

$z_i \leftarrow step\_count$: Arithmetic counter $k \pmod p$.

2.2 R1CS Constraint Complexity ($C = 2048$)

In R1CS, linear combinations (addition) are free ($0$ constraints). Only multiplications $A \cdot B = C$ incur a constraint cost. For a chunk size of $C = 2048$, the per-step constraint generation is highly optimized:

$g_j^2$ (Norm): $1$ constraint per parameter.

$r_j \cdot g_j$ (Gradient Hash): $1$ constraint per parameter.

$g_j \cdot ref_j$ (Direction): $1$ constraint per parameter.

$r_j \cdot ref_j$ (Reference Hash): $1$ constraint per parameter.
Total Step Complexity: $4C \approx 8,192$ constraints per fold step.

For a $d=62,000$ parameter CNN, IVC reduces a monolithic $\sim 250,000$ constraint circuit into exactly $31$ sequential folding steps of $8,192$ constraints, heavily optimizing RAM usage and parallelization.

2.3 The Decider Constraints

After $K = \lceil d / C \rceil$ steps, the final state $z_{K}$ is evaluated in the Decider Circuit:

Magnitude Limit: cs.enforce_cmp(z_K, B_l_sq, LessThanOrEq)

Direction Limit: cs.enforce_cmp(z_K, F::ZERO, GreaterThanOrEq)

Spoof Protection: cs.enforce_eq(z_K, z_K)

2.4 Sonobe FCircuit Implementation (Rust Code Depth)

To utilize Sonobe's ProtoGalaxy implementation, the core logic is encapsulated in the FCircuit trait using ark_r1cs_std for finite field constraint generation.

use ark_r1cs_std::prelude::\*;
use folding_schemes::frontend::FCircuit;

#[derive(Clone, Debug)]
pub struct GradientFingerprintCircuit<F: PrimeField> {
pub chunk_size: usize,
pub \_f: std::marker::PhantomData<F>,
}

impl<F: PrimeField> FCircuit<F> for GradientFingerprintCircuit<F> {
type Params = ();

    fn state_len(&self) -> usize { 7 }
    fn external_inputs_len(&self) -> usize { self.chunk_size * 3 } // r, g, ref_g

    fn generate_step_constraints(
        &self,
        cs: ConstraintSystemRef<F>,
        _i: usize,
        z_i: Vec<FpVar<F>>,
        external_inputs: Vec<FpVar<F>>,
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {

        // 1. Unpack state z_i
        let model_fp = &z_i;
        let ref_grad_fp = &z_i;
        let mut grad_fp_accum = z_i.clone();
        let mut norm_sq_accum = z_i.clone();
        let mut dir_fp_accum = z_i.clone();
        let mut ref_fp_accum = z_i.clone();
        let step_count = &z_i;

        // 2. Unpack External Inputs (Witnesses)
        let r_chunk = &external_inputs[0..self.chunk_size];
        let g_chunk = &external_inputs[self.chunk_size..self.chunk_size*2];
        let ref_chunk = &external_inputs[self.chunk_size*2..self.chunk_size*3];

        // 3. Apply R1CS Constraints for the current chunk
        for j in 0..self.chunk_size {
            // grad_fp_accum += r_j * g_j
            let r_g = r_chunk[j].clone() * g_chunk[j].clone();
            grad_fp_accum = grad_fp_accum + r_g;

            // norm_sq_accum += g_j * g_j
            let g_sq = g_chunk[j].clone() * g_chunk[j].clone();
            norm_sq_accum = norm_sq_accum + g_sq;

            // dir_fp_accum += g_j * ref_j
            let g_ref = g_chunk[j].clone() * ref_chunk[j].clone();
            dir_fp_accum = dir_fp_accum + g_ref;

            // ref_fp_accum += r_j * ref_j
            let r_ref = r_chunk[j].clone() * ref_chunk[j].clone();
            ref_fp_accum = ref_fp_accum + r_ref;
        }

        // 4. Construct z_{i+1}
        let one = FpVar::Constant(F::one());
        let next_step = step_count.clone() + one;

        Ok(vec![
            model_fp.clone(),
            ref_grad_fp.clone(),
            grad_fp_accum,
            norm_sq_accum,
            dir_fp_accum,
            ref_fp_accum,
            next_step,
        ])
    }

}

2.5 ProtoGalaxy IVC Proving and Decider Pipeline

The Rust FFI bridge utilizes Sonobe's ProtoGalaxy and Decider traits to execute the folding scheme across the chunks provided by the Python runtime.

// Type aliases mapping the BN254 curve to ProtoGalaxy
type Projective = ark_bn254::G1Projective;
type PG = ProtoGalaxy<Projective, GradientFingerprintCircuit<Fr>>;
type Decider = DeciderEth<Projective, GradientFingerprintCircuit<Fr>>;

pub fn generate_ivc_proof(
z_0: Vec<Fr>,
chunks: Vec<Vec<Fr>> // Batched [r, g, ref] chunks
) -> Result<Vec<u8>, Error> {

    // 1. Setup & Circuit Initialization
    let circuit = GradientFingerprintCircuit { chunk_size: 2048, _f: PhantomData };
    let (prover_params, verifier_params) = PG::setup(..., &circuit)?;

    // 2. Initialize IVC State
    let mut ivc_state = PG::init(&prover_params, circuit, z_0.clone())?;

    // 3. Execute Folding Loop
    for external_inputs in chunks {
        // Absorbs the step into the folded accumulator
        ivc_state = PG::prove_step(ivc_state, external_inputs)?;
    }

    // 4. Finalize with Decider
    // The decider wraps the final folded state into a verifiable SNARK
    let decider_proof = Decider::prove(&prover_params, ivc_state)?;

    // Serialize to bytes for transmission to Aggregator
    Ok(serialize_bundle(decider_proof, ivc_state.z_i))

}

3. Python-to-Rust FFI Bridge & Finite Field Mapping

Neural network gradients are in FP32. To map these into the BN254 scalar field $\mathbb{F}_p$ while preserving negative values and avoiding float-truncation, FiZK implements a rigorous bijective mapping.

3.1 Strict Quantization & Field Wrapping

Let $\eta = 6$ be the quantization scale factor. For every parameter $g \in \mathbb{R}$:

$$\tilde{g} = \lfloor g \times 10^\eta \rceil$$

If $\tilde{g} < 0$, it is wrapped in the finite field as $\tilde{g}_{\mathbb{F}_p} = p - |\tilde{g}|$.

3.2 Dynamic Decider Scaling

Because the R1CS circuit computes the $L_2$ norm on quantized parameters, the scalar sum-of-squares operates in the scaled space. The Python server must scale the dynamic bound before passing it as a public input to the verifier:

$$B_{ZK}^2 = B_l^2 \times 10^{2\eta}$$

This $10^{12}$ scaling is safely contained within $\mathbb{F}_p$ capacity ($2^{254} \approx 10^{76}$), guaranteeing no cryptographic overflow.

3.3 Memory Layout & Padding

To satisfy $K \times C = d_{padded}$, Python zero-pads the flattened 1D gradient vector:

$$pad\_len = (2048 - (d \pmod{2048})) \pmod{2048}$$

Tensors are passed across the FFI boundary as contiguous C memory arrays (numpy.ndarray(dtype=np.int64)) natively converting to Rust Vec<i64>.

4. The Statistical Defense Engine

4.1 Sybil-Resistant Exponential Moving Average (EMA)

To calculate $B_l$, the global server measures the true un-rooted $L_2$ norm of accepted gradients. To prevent Sybil adversaries ($\alpha \ge 0.50$) from manipulating the threshold, the bound $B_l^{(t)}$ is anchored to historical data via momentum $\beta = 0.9$:

$$Q_{0.25}^{(t)} = \text{Percentile}_{25}\left(\{ \|g_i\|_2 \}_{i=1}^n\right)$$

$$B_l^{(t)} = \beta B_l^{(t-1)} + (1-\beta) \left[ Q_{0.25}^{(t)} \times (1 + \gamma) \right]$$

By anchoring at the 25th percentile, the system remains in the honest statistical regime as long as honest clients $\ge 25\%$.

4.2 Multi-Krum Scoring Function

For gradients surviving the ZK-Firewall, the Galaxy Aggregator computes the pairwise Euclidean distance matrix. For client $i$, its score $s_i$ is the sum of distances to its $n - f - 2$ closest neighbors (where $f$ is the estimated Byzantine limit):

$$s_i = \sum_{g_j \in \mathcal{N}_i} \| g_i - g_j \|_2^2$$

The aggregator selects the $m$ gradients with the lowest scores $s_i$ for final aggregation $\bar{w}_g = \frac{1}{m}\sum g_{selected}$.

5. Formal Protocol Flow

Phase 0: Context Initialization

Server broadcasts $w^{(t)}$, random Fiat-Shamir seed $r^{(t)}$, scaling threshold $B_{ZK}^2$, and reference vector $\nabla w_{ref} = \bar{w}_{global}^{(t-1)}$.

Phase 1: Commitment & Proof Generation

Client evaluates $\tilde{g}_i = \text{quantize}(\nabla \mathcal{L}(w^{(t)}, \mathcal{D}_i))$.

Client computes ZK-IVC proof $\pi_i^{zk}$ over $\tilde{g}_i$ asserting membership in $\Omega^{(t)}$ using the ProtoGalaxy::prove_step pipeline.

Client computes cryptographic binding: $h_i = \text{SHA256}(\tilde{g}_i \parallel r^{(t)} \parallel \text{nonce}_i)$.

Client transmits $\{ID_i, h_i, \pi_i^{zk}\}$ to Galaxy Aggregator.

Phase 2: Revelation & Firewall Verification

Aggregator locks round, building Merkle Tree $\mathcal{M}$ over $\{h_i\}_{i=1}^n$ with root $R_g$.

Client transmits plaintext $\{\tilde{g}_i, \text{nonce}_i\}$.

L1 Firewall: Aggregator asserts $\text{SHA256}(\dots) \equiv h_i$. Extracts public output state $\vec{z}_K$ of $\pi_i^{zk}$ and ensures it perfectly matches $B_{ZK}^2$ and public Fiat-Shamir hashes.

Phase 3: Selection & Compression

L2 Defense: Aggregator executes Multi-Krum on validated $\{\tilde{g}_i\}$.

Aggregator computes Galaxy mean $\bar{w}_g$.

IVC Fold: Aggregator applies the ProtoGalaxy non-interactive folding scheme $\mathcal{F}$ over all client bundles:

$$\pi_g^{\mathsf{fold}} = \mathcal{F}(\pi_1^{zk}, \mathcal{F}(\pi_2^{zk}, \dots \mathcal{F}(\pi_{m-1}^{zk}, \pi_m^{zk})\dots))$$

Transmits $\{\bar{w}_g, \pi_g^{\mathsf{fold}}, R_g\}$ to Server.

Phase 4: $O(1)$ Global State Update

Server verifies $\pi_g^{\mathsf{fold}}$ via the DeciderEth::verify method. Verification complexity is strictly $O(1)$ with respect to client count $m$.

Server incorporates $\bar{w}_g \to w^{(t+1)}$ and updates parameters.

6. Threat Model & Theoretical Boundaries

Attack Vector

Countermeasure

Mathematical Bound / Mechanism

Bait-and-Switch

Merkle Lock

Cryptographic Hash pre-image collision resistance.

Exploding Gradients

ZK Magnitude Check

Ensures $g_i \in \mathbb{R}^d$ bounded by hypersphere radius $B_l$.

Sign-Flipping

ZK Directional Check

Restricts to half-space: $\langle g_i, \nabla w_{ref} \rangle \ge 0$.

Reference Spoofing

In-Circuit Anti-Spoof

$\sum (r_j \cdot ref_j) \equiv Hash(ref\_public)$.

Sybil Inflation

EMA Bound Smoothing

Requires $\alpha > 0.75$ sustained over $T \to \infty$ rounds to diverge.

Semantic Data Poisoning

Multi-Krum / Median

$O(d)$ outlier pruning within the accepted ZK cryptographic hypersphere.

7. Known Theoretical Limitations

Orthogonal Backdoors ($\theta \to 90^\circ$): The directional constraint $\langle g_i, \nabla w_{ref} \rangle = \|g_i\|\|\nabla w_{ref}\| \cos(\theta) \ge 0$ restricts the attacker to the positive half-space. An attacker can submit a poisoned vector perfectly orthogonal to the reference ($\cos(89.9^\circ) \approx 0$). While ZK-approved, these are structurally distinct and are mitigated downstream by Multi-Krum.

Privacy vs. Integrity (DLG Vulnerability): The protocol mandates plaintext revelation to the Aggregator for Merkle verification, leaving clients susceptible to Deep Leakage from Gradients (DLG). Future iterations must compose the IVC proof with Secure Aggregation (SecAgg) secret-sharing primitives.
