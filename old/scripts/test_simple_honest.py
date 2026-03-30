#!/usr/bin/env python3
"""Simple test: honest client only, with detailed error reporting"""

import sys
sys.path.insert(0, 'src')

import torch
from crypto.zkp_prover import TrainingProofProver

# Create honest model
weights = torch.randn(10, 784) * 0.01
bias = torch.randn(10) * 0.01

# Create small batch
batch = [(torch.randn(784), i % 10) for i in range(4)]

# Create prover
prover = TrainingProofProver()

print("Testing honest client with 4 training samples...")
print(f"Model: W shape={weights.shape}, b shape={bias.shape}")
print(f"Batch size: {len(batch)}")

try:
    proof = prover.prove_training(
        weights=weights,
        bias=bias,
        batch=batch,
        client_id=1,
        model_type=0,
        num_samples=len(batch)
    )
    print("✅ SUCCESS: Honest client proof generated")
    print(f"   Proof size: {len(proof.proof_bytes)} bytes")
    print(f"   Steps: {proof.num_steps}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
