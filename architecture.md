FiZK: System Architecture and Programming Specification

This document details the configuration, cryptographic circuit design, execution flow, and evaluation setup of the FiZK (Federated Integrity with Zero Knowledge) protocol. It incorporates the $L_2$ norm (Sum-of-Squares) R1CS patch, the EMA-based Sybil mitigation strategy, in-circuit directional filtering, and practical engineering constraints required for real-world deployment.

1. System Configuration & Hyperparameters

Cryptographic Stack

IVC Scheme: ProtoGalaxy (via Sonobe library).

Elliptic Curve Cycle: BN254 / Grumpkin (natively supports efficient folding).

Commitment Scheme: SHA-256 (Client-side Merkle Leaves), Pedersen Commitments (Inside ZK).

Circuit Chunk Size ($C$): 2048 parameters per fold step. (This drastically reduces IVC folding overhead by shifting the bottleneck from CPU-bound step verification to RAM-bound circuit size, bringing a 62K parameter LeNet-5 model from ~1,240 folding steps down to just 31 steps).

Zero-Knowledge Relation: Relaxed R1CS.

Practical Engineering Parameters

Quantization Scale Factor: $10^6$ (Maps FP32 floats to finite field integers $\mathbb{F}_p$). Note: Squared norm accumulations will scale by $10^{12}$, which is safely within the BN254 scalar field capacity ($\approx 2^{254}$).

Padding Strategy: Zero-padding (Pads gradient vectors until length % 2048 == 0).

Local Environment Optimizations (Rust/Maturin)

Because evaluation and testing are performed on a local computer, the heavy cryptographic Multi-Scalar Multiplications (MSMs) must be optimized for the host CPU to achieve realistic proving times:

Parallelism (Rayon): The ark-ff and ark-ec (Arkworks) dependencies are compiled with the parallel feature enabled. This allows the Sonobe prover to automatically distribute polynomial commitments and MSMs across all available CPU cores on the local machine.

Instruction Set Targeting: The Rust environment is configured with RUSTFLAGS="-C target-cpu=native". This allows the compiler to utilize advanced local hardware vector instructions (e.g., AVX2 or AVX-512) for finite field arithmetic.

Release Mode: The Python-Rust bridge is exclusively compiled using maturin develop --release. Debug builds drop performance by over 10x and must not be used for benchmarking.

Federated Learning Stack

Aggregation Algorithm: Multi-Krum + Cosine Similarity.

Norm Bound Baseline: 25th Percentile ($Q_{0.25}$) of client norms.

Sybil EMA Momentum ($\beta$): 0.9.

Tolerance Multiplier ($\gamma$): 5.0.

Reputation Decay ($\lambda$): 0.1.

Directional Threshold: 0 (Used to block negative inner products, mathematically guaranteeing an angle $\le 90^\circ$ between the client gradient and the global reference trajectory).

2. ZK Circuit Architecture (Rust / Sonobe)

The core integrity engine is the GradientFingerprintCircuit<F>, implemented in Rust. It utilizes a uniform step circuit that computes three critical cryptographic bounds: the gradient fingerprint, the $L_2$ norm squared (magnitude limit), and the reference inner product (directional limit).

2.1 State Vector (The Accumulators)

At step $i$, the circuit takes the state $z_i$ and outputs $z_{i+1}$:

pub struct CircuitState<F: PrimeField> {
    pub model_fp: F,               // Immutable: Binds to the global server's model
    pub ref_grad_fp: F,            // Immutable: Binds the public reference gradient to prevent spoofing
    pub grad_fp_accum: F,          // Accumulates: <r_chunk, g_chunk>
    pub norm_sq_accum: F,          // Accumulates: sum(g_j^2) for magnitude bounding
    pub directional_fp_accum: F,   // Accumulates: <g_chunk, ref_chunk> for directional bounding
    pub step_count: F,             // Accumulates: +1 per fold
}


2.2 R1CS Step Logic (prove_chunk)

For each chunk of 2048 gradients, the circuit enforces the following R1CS constraints. Providing the ref_chunk (the aggregated global gradient from round $t-1$) allows the circuit to mathematically block sign-flipping.

Cryptographic Note on Cosine Similarity: Calculating true cosine similarity ($\frac{A \cdot B}{||A|| ||B||}$) requires division and square roots, which are highly inefficient in R1CS. Because FiZK already bounds the magnitude ($L_2$ norm), the circuit only needs to compute the raw inner product. Bounding this inner product $\ge 0$ serves as a highly efficient, division-free proxy for Cosine Similarity.

// Pseudo-code for Sonobe R1CS constraint generation
fn synthesize_step(
    &self, 
    cs: &mut ConstraintSystem<F>, 
    z_in: CircuitState<F>, 
    r_chunk: Vec<F>, 
    g_chunk: Vec<F>,
    ref_chunk: Vec<F>
) -> Result<CircuitState<F>, Error> {
    
    let mut current_grad_fp = z_in.grad_fp_accum;
    let mut current_norm_sq = z_in.norm_sq_accum;
    let mut current_dir_fp = z_in.directional_fp_accum;

    for j in 0..CHUNK_SIZE { // CHUNK_SIZE = 2048
        // 1. Fingerprint Constraint: grad_fp += r_j * g_j
        let r_times_g = cs.mul(r_chunk[j], g_chunk[j]);
        current_grad_fp = cs.add(current_grad_fp, r_times_g);

        // 2. Magnitude Constraint: norm_sq += g_j * g_j
        let g_squared = cs.mul(g_chunk[j], g_chunk[j]); 
        current_norm_sq = cs.add(current_norm_sq, g_squared);

        // 3. Directional Constraint: dir_fp += g_j * ref_j (Blocks sign flips)
        let g_times_ref = cs.mul(g_chunk[j], ref_chunk[j]);
        current_dir_fp = cs.add(current_dir_fp, g_times_ref);
    }

    // Return z_{i+1}
    Ok(CircuitState {
        model_fp: z_in.model_fp, 
        ref_grad_fp: z_in.ref_grad_fp,
        grad_fp_accum: current_grad_fp,
        norm_sq_accum: current_norm_sq,
        directional_fp_accum: current_dir_fp,
        step_count: cs.add(z_in.step_count, F::ONE),
    })
}


2.3 The Decider Check (Post-Fold Verification)

Once the folding is complete, the final Decider circuit enforces both safety bounds (Magnitude and Direction).

// In the final Decider / Verification Phase

// 1. Prevent Exploding Gradients & Sum-to-Zero bypass
cs.enforce_less_than_or_equal(z_final.norm_sq_accum, B_l_squared);

// 2. Prevent Sign-Flipping & Directional Attacks (Requires angle <= 90 degrees)
// A threshold of F::ZERO ensures the vector is not pointing in the opposite direction.
cs.enforce_greater_than_or_equal(z_final.directional_fp_accum, F::ZERO);


3. Python-Side Pipeline & Algorithms

3.1 Dynamic Bound Calculation (EMA)

To prevent Sybil attacks from artificially inflating the bound, the Global Server calculates the bound using an Exponential Moving Average.

def calculate_dynamic_bound(client_norms, previous_bound, beta=0.9, gamma=5.0):
    """
    Calculates the $L_2$ norm threshold for the next round.
    """
    q_25 = np.percentile(client_norms, 25)
    current_target = q_25 * (1 + gamma)
    
    if previous_bound is None:
        return current_target
        
    # EMA Calculation
    new_bound = (beta * previous_bound) + ((1 - beta) * current_target)
    return new_bound


3.2 Galaxy Aggregator: Defense Pipeline

def process_galaxy_updates(client_payloads, R_g, B_l_squared, public_ref_hash):
    valid_clients = []
    
    for payload in client_payloads:
        client_id, plaintext_grad, zkp_bundle, nonce = payload
        
        # 1. Merkle Lock Check
        h_i = sha256(plaintext_grad, global_seed, nonce)
        if not verify_merkle_leaf(h_i, R_g):
            penalize_reputation(client_id)
            continue
            
        # 2. Public IO Check (Ensures client didn't spoof bounds or reference vectors)
        if not verify_public_inputs(zkp_bundle, plaintext_grad, B_l_squared, public_ref_hash):
            penalize_reputation(client_id)
            continue
            
        valid_clients.append((client_id, plaintext_grad, zkp_bundle))

    # 3. Statistical Filtering (Cosine -> Multi-Krum)
    filtered_clients = apply_cosine_similarity_filter(valid_clients)
    final_m_clients = apply_multi_krum(filtered_clients)
    
    # 4. Aggregation and Folding (Cryptographic Absorption)
    w_g = compute_mean([c for c in final_m_clients])
    pi_g_fold = fold_proofs_ivc([c for c in final_m_clients]) # Rust PG::fold
    
    return w_g, pi_g_fold


4. End-to-End Execution Flow (The 4 Phases)

Phase 0: Initialization & "Cold Start"

Cold Start: In Round $t=0$, there are no client norms to establish $Q_{0.25}$. $B_l^{(0)}$ is initialized via a pre-defined hyperparameter derived from a warm-up epoch on public data. The reference_grad is set to an array of ones (or initial weights) to permit initial exploration.

Phase 1: Commitment and ZKP Generation (Edge Client)

Receive Context: Client receives global model $w^{(t)}$, random challenge seed $r^{(t)}$, reference gradient $\nabla w_{ref}$, its public hash, and dynamic bounds.

Local Training: Execute local SGD to compute $\nabla w_i$.

Quantization & Padding:

Multiply $\nabla w_i$ and $\nabla w_{ref}$ by $10^6$ and map to $\mathbb{F}_p$.

Append zeroes until length is divisible by $C=2048$.

ZK Proving: Call ModelAgnosticProver.generate_proof().

Runs ProtoGalaxy IVC to generate $\pi_i^{zk}$. Fails natively if magnitude exceeds $(B_l^{(t)})^2$ or if the directional inner product falls below 0.

Commit: Compute $h_i = \text{SHA-256}(\text{Quantized}(\nabla w_i) \,\|\, r^{(t)} \,\|\, \text{nonce}_i)$.

Transmit 1: Send (client_id, h_i, pi_i_zk) to Galaxy Aggregator.

Phase 2: Gradient Revelation & ZK-Firewall (Galaxy Aggregator)

Construct Tree: Aggregator builds Merkle Tree from all $h_i$, publishes root $R_g^{(t)}$.

Transmit 2: Clients reveal (plaintext_grad, nonce_i).

Privacy Note: In a production deployment, this step uses Secure Aggregation (SecAgg). Clients transmit gradients combined with cryptographic masks. The ZK proof attests to the unmasked integrity, while the Aggregator only sees the masked sum.

Layer 1 Defense (Cryptographic IO Check):

Assert SHA-256(plaintext_grad...) == h_i.

Assert public variables in $\pi_i^{zk}$ match plaintext_grad fingerprint and public_ref_hash. Full cryptographic verification is deferred to the folding step.

Phase 3: Defense & Folding (Galaxy Aggregator)

Layer 2 & 3 Defenses: Execute server-side Cosine Similarity and Multi-Krum to prune minor statistical anomalies.

Aggregate: Compute $\bar{w}_g = \frac{1}{m} \sum_{k=1}^m \nabla w_k$.

Fold (The true Verification): Execute ProtoGalaxy fold() on the $m$ valid pi_i_zk bundles. This cryptographically absorbs and verifies the proofs, producing a constant-size $\pi_g^{\mathsf{fold}}$.

Transmit 3: Send (w_g, pi_g_fold, R_g) to Global Server.

Phase 4: Global Verification & Update (Global Server)

Verify Integrity:

Construct Global Merkle Tree.

Execute $O(1)$ IVC verification on each $\pi_g^{\mathsf{fold}}$.

Global Update: Apply valid, de-quantized galaxy gradients to $w^{(t)}$.

Update Contexts: Extract new $L_2$ norms, apply EMA to update $B_l^{(t+1)}$, and set the new aggregated gradient as $\nabla w_{ref}$ for Round $t+1$.

5. Experimental Setup & Evaluation Metrics

To thoroughly validate FiZK, the evaluation compares its performance against established baselines under diverse attack scenarios.

5.1 Defense Baselines

Vanilla FedAvg: No defense (Establishes the lower bound / attack effectiveness).

Multi-Krum: Standard robust aggregation based on Euclidean distance filtering.

FLTrust: A state-of-the-art defense relying on a trusted root dataset (to benchmark FiZK's root-free approach).

FiZK (Ours): Full proposed architecture (ZKP + Multi-Krum + Cosine).

5.2 Datasets & Models

Linear Task: MNIST dataset evaluated on a single-layer Linear Classifier.

Deep Task: Fashion-MNIST dataset evaluated on a LeNet-5 CNN (trained from scratch).

Distributions: Both IID and Dirichlet Non-IID ($\beta=0.5$).

5.3 Attack Threat Model

Simulate the following attacks varying Byzantine fractions ($\alpha \in \{0.30, 0.50, 0.60\}$):

Model Poisoning (Scaling): Gradients multiplied by large malicious factors. (Blocked natively by $L_2$ bound).

Sign-Flip Attack: $\nabla w_{malicious} = -1 \times \nabla w_{honest}$. (Blocked natively by in-circuit directional check).

Gaussian Noise Attack: Gradients replaced with vectors drawn from $\mathcal{N}(0, \sigma^2)$.

Label Flipping Attack: Local data labels inverted (e.g., $y_{malicious} = 9 - y_{honest}$).

5.4 Mandatory Data & Metrics to Gather

For each experiment, collect the following precise metrics (average across 3 random seeds):

1. Machine Learning Metrics (Effectiveness):

Global Test Accuracy (%): Final accuracy of the global model after $T$ rounds.

Byzantine Detection Rate:

TPR (True Positive Rate): % of malicious clients correctly dropped.

FPR (False Positive Rate): % of honest clients accidentally dropped.

F1-Score: Harmonic mean of Precision and Recall for detecting Byzantine clients.

2. Cryptographic/System Metrics (Efficiency):

Client Proving Time (ms): Time taken to run generate_proof(). Compare standard SGD time vs SGD + ZKP time to establish the computational $\Delta$.

Aggregator Folding Time (ms): Time taken to fold $m$ proofs.

Global Server Verification Time (ms): Time taken to verify the $O(1)$ folded instance.

Proof Size Overhead (KB/MB): Network bandwidth consumed by raw client proofs vs. the compressed $\pi_g^{\mathsf{fold}}$.

6. Security Limitations & Defense-in-Depth Context

This architecture explicitly acknowledges the boundaries of applied zero-knowledge proofs in modern Machine Learning via a defense-in-depth paradigm.

6.1 The "Proof-of-Training" Gap vs. Property Checking

The GradientFingerprintCircuit mathematically functions as a Property Checker. It irrefutably proves the mathematical properties of a submitted vector (its bounded magnitude and its directional alignment with the global trajectory). However, it does not provide Computation Provenance Checking. The circuit cannot mathematically prove that the client obtained gradient $g$ by running actual Stochastic Gradient Descent (SGD) on their private dataset.

Achieving fully Verifiable Training (compiling the forward pass, loss calculation, and backpropagation of a 62K parameter model over multiple epochs into an R1CS circuit) remains computationally prohibitive for edge devices. Consequently, a malicious client can submit an arbitrary gradient $g'$ provided they successfully engineer it to pass the magnitude and directional thresholds.

6.2 Defense-in-Depth Mitigation

To address this gap, FiZK pairs strict cryptographic property bounds with statistical aggregation layers (Multi-Krum).

The ZK-Firewall acts as a hard cryptographic limit, eliminating the most catastrophic attacks (exploding gradients, sign-inversions) that typically collapse statistical aggregators.

The Aggregator Algorithms act as semantic filters, discarding gradients that conform to the cryptographic rules but deviate maliciously in distribution (e.g., highly crafted label-flipping vectors). By restricting the search space available to an attacker with ZKPs, the statistical aggregators are heavily protected and can successfully tolerate Byzantine majorities ($\alpha \ge 50\%$).