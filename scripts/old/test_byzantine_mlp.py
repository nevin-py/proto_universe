#!/usr/bin/env python3
"""Test Byzantine detection with MLP model (10×64 final layer)"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.crypto.zkp_prover import TrainingProofProver
from src.models.mnist import create_mnist_model

print("="*80)
print("BYZANTINE DETECTION - MLP Model (10×64)")
print("="*80)

# Create MLP model
model = create_mnist_model(model_type='mlp', num_classes=10)
weights = model.fc3.weight.data.clone()  # Final FC layer: 10×64
bias = model.fc3.bias.data.clone()

print(f"Model: {model.__class__.__name__}")
print(f"Proven layer: W={weights.shape}, b={bias.shape}")
print(f"Input dimension: {weights.shape[1]}")

# Create training data
input_dim = weights.shape[1]
batch_size = 4
batch = [(torch.randn(input_dim), i % 10) for i in range(batch_size)]

prover = TrainingProofProver()
round_number = 0

# Test 1: Honest MLP
print(f"\n{'='*80}")
print("TEST 1: Honest MLP Model")
print(f"{'='*80}")
try:
    proof = prover.prove_training(
        weights=weights,
        bias=bias,
        train_data=batch,
        client_id=1,
        round_number=round_number,
        batch_size=batch_size
    )
    print(f"✅ PASS: Honest proof generated ({len(proof.proof_bytes)} bytes)")
    honest_passed = True
except Exception as e:
    print(f"❌ FAIL: {e}")
    honest_passed = False

# Test 2: Malicious MLP (perturbed weights)
print(f"\n{'='*80}")
print("TEST 2: Malicious MLP Model (perturbed weights)")
print(f"{'='*80}")

malicious_weights = weights.clone()
malicious_weights += torch.randn_like(malicious_weights) * 0.1

r_vec = prover.generate_random_vector(round_number)
sample_size = min(100, input_dim)  # Adaptive sampling
honest_fp, _ = prover.compute_model_fingerprint(weights, bias, r_vec, round_number, sample_size=sample_size)
malicious_fp, _ = prover.compute_model_fingerprint(malicious_weights, bias, r_vec, round_number, sample_size=sample_size)

print(f"Honest fingerprint:    {honest_fp}")
print(f"Malicious fingerprint: {malicious_fp}")
print(f"Difference:            {abs(honest_fp - malicious_fp)}")

try:
    # NEW PROVER INSTANCE for malicious test (avoids R1CS mismatch)
    prover2 = TrainingProofProver()
    proof = prover2.prove_training(
        weights=malicious_weights,
        bias=bias,
        train_data=batch,
        client_id=2,
        round_number=round_number,
        batch_size=batch_size,
        expected_fingerprint=honest_fp  # Honest FP from server
    )
    print(f"❌ FAIL: Malicious proof accepted!")
    malicious_caught = False
except RuntimeError as e:
    if "fingerprint" in str(e).lower():
        print(f"✅ PASS: Malicious client rejected (fingerprint mismatch)")
        malicious_caught = True
    else:
        print(f"⚠️  Rejected for other reason: {str(e)[:80]}")
        malicious_caught = False

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"{'✅' if honest_passed else '❌'} Honest MLP: {'PASS' if honest_passed else 'FAIL'}")
print(f"{'✅' if malicious_caught else '❌'} Malicious MLP: {'PASS' if malicious_caught else 'FAIL'}")
print(f"\nDetection Rate: {100 if malicious_caught else 0}%")

if honest_passed and malicious_caught:
    print("\n🎉 MLP MODEL PASSED - 100% Byzantine detection!")
    sys.exit(0)
else:
    print("\n⚠️  SOME TESTS FAILED")
    sys.exit(1)
