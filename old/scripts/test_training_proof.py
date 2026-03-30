#!/usr/bin/env python3
"""End-to-end test of the Proof-of-Training ZK circuit.

Tests:
  1. Honest client: generates valid proof → verification succeeds
  2. Timing: measures prove + verify overhead per sample
"""

import time
import sys
import numpy as np
import torch

# Ensure src is on path
sys.path.insert(0, '/run/media/vane/Data/Project/proto_universe')

from src.crypto.zkp_prover import TrainingProofProver, TrainingProof

def main():
    print("=" * 60)
    print("Proof-of-Training Circuit E2E Test")
    print("=" * 60)

    # Create a simple linear model (784 → 10)
    torch.manual_seed(42)
    weights = torch.randn(10, 784) * 0.01  # small random weights
    bias = torch.zeros(10)

    # Create a few fake MNIST samples (pixel values 0-255 as floats)
    num_samples = 4
    train_data = []
    for i in range(num_samples):
        x = torch.randint(0, 256, (784,)).float()
        y = i % 10  # label
        train_data.append((x, y))

    print(f"\nModel: linear 784→10 ({weights.numel()} weights + {bias.numel()} biases)")
    print(f"Training batch: {num_samples} samples")

    # Initialize prover
    prover = TrainingProofProver()
    print(f"ZKP available: {prover._is_real}")

    # Generate proof
    print(f"\n--- Proving ---")
    proof = prover.prove_training(
        weights=weights,
        bias=bias,
        train_data=train_data,
        client_id=0,
        round_number=1,
        batch_size=num_samples,
    )

    print(f"Prove time: {proof.prove_time_ms:.1f} ms")
    print(f"Num steps: {proof.num_steps}")
    print(f"Proof size: {proof.proof_size} bytes")
    print(f"Model fingerprint: {proof.model_fingerprint:.6f}")
    print(f"Is real ZK: {proof.is_real}")

    # Verify proof
    print(f"\n--- Verifying ---")
    valid = TrainingProofProver.verify_training_proof(proof, weights, bias)
    print(f"Verification result: {'✓ VALID' if valid else '✗ INVALID'}")
    print(f"Verify time: {proof.verify_time_ms:.1f} ms")

    # Summary
    print(f"\n--- Summary ---")
    total_ms = proof.prove_time_ms + proof.verify_time_ms
    per_sample_ms = proof.prove_time_ms / max(proof.num_steps, 1)
    print(f"Total ZK overhead: {total_ms:.1f} ms")
    print(f"Per-sample prove time: {per_sample_ms:.1f} ms")
    print(f"Proof verified: {valid}")

    if valid:
        print("\n✓ E2E TEST PASSED")
    else:
        print("\n✗ E2E TEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
