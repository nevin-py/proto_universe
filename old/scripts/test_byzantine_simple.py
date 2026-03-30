#!/usr/bin/env python3
"""Simple Byzantine detection test with external fingerprint verification"""

import sys
sys.path.insert(0, 'src')

import torch
from crypto.zkp_prover import TrainingProofProver

print("="*80)
print("BYZANTINE DETECTION TEST - External Fingerprint Verification")
print("="*80)

# Setup
prover = TrainingProofProver()
round_number = 0
batch_size = 4

# Create honest model
honest_weights = torch.randn(10, 784) * 0.01
honest_bias = torch.randn(10) * 0.01

# Create training data
batch = [(torch.randn(784), i % 10) for i in range(batch_size)]

# Test 1: Honest client
print("\n" + "="*80)
print("TEST 1: Honest Client")
print("="*80)
try:
    proof = prover.prove_training(honest_weights, honest_bias, batch, 1, round_number, batch_size)
    print(f"✅ PASS: Honest proof generated ({len(proof.proof_bytes)} bytes)")
    honest_passed = True
except Exception as e:
    print(f"❌ FAIL: {e}")
    honest_passed = False

# Test 2: Malicious client (wrong weights)
print("\n" + "="*80)
print("TEST 2: Malicious Client (perturbed weights)")
print("="*80)

malicious_weights = honest_weights.clone()
malicious_weights += torch.randn_like(malicious_weights) * 0.1  # Significant perturbation

r_vec = prover.generate_random_vector(round_number)
honest_fp, _ = prover.compute_model_fingerprint(honest_weights, honest_bias, r_vec, round_number)
malicious_fp, _ = prover.compute_model_fingerprint(malicious_weights, honest_bias, r_vec, round_number)

print(f"   Honest fingerprint:    {honest_fp}")
print(f"   Malicious fingerprint: {malicious_fp}")
print(f"   Difference:            {abs(honest_fp - malicious_fp)}")

try:
    proof = prover.prove_training(malicious_weights, honest_bias, batch, 2, round_number, batch_size)
    print(f"❌ FAIL: Malicious proof accepted! Detection failed.")
    malicious_weights_caught = False
except RuntimeError as e:
    if "fingerprint" in str(e).lower():
        print(f"✅ PASS: Malicious client rejected (fingerprint mismatch)")
        malicious_weights_caught = True
    else:
        print(f"⚠️  Rejected but for different reason: {e}")
        malicious_weights_caught = False

# Test 3: Malicious client (wrong bias)
print("\n" + "="*80)
print("TEST 3: Malicious Client (perturbed bias)")
print("="*80)

malicious_bias = honest_bias.clone()
malicious_bias += torch.randn_like(malicious_bias) * 0.5

honest_fp2, _ = prover.compute_model_fingerprint(honest_weights, honest_bias, r_vec, round_number)
malicious_fp2, _ = prover.compute_model_fingerprint(honest_weights, malicious_bias, r_vec, round_number)

print(f"   Honest fingerprint:    {honest_fp2}")
print(f"   Malicious fingerprint: {malicious_fp2}")
print(f"   Difference:            {abs(honest_fp2 - malicious_fp2)}")

try:
    proof = prover.prove_training(honest_weights, malicious_bias, batch, 3, round_number, batch_size)
    print(f"❌ FAIL: Malicious proof accepted! Detection failed.")
    malicious_bias_caught = False
except RuntimeError as e:
    if "fingerprint" in str(e).lower():
        print(f"✅ PASS: Malicious client rejected (fingerprint mismatch)")
        malicious_bias_caught = True
    else:
        print(f"⚠️  Rejected but for different reason: {e}")
        malicious_bias_caught = False

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'✅' if honest_passed else '❌'} Honest client: {'PASS' if honest_passed else 'FAIL'}")
print(f"{'✅' if malicious_weights_caught else '❌'} Malicious (weights): {'PASS' if malicious_weights_caught else 'FAIL'}")
print(f"{'✅' if malicious_bias_caught else '❌'} Malicious (bias): {'PASS' if malicious_bias_caught else 'FAIL'}")

detection_rate = (malicious_weights_caught + malicious_bias_caught) / 2 * 100
print(f"\nDetection Rate: {detection_rate:.0f}%")

if honest_passed and malicious_weights_caught and malicious_bias_caught:
    print("\n🎉 ALL TESTS PASSED! 100% Byzantine detection with 0% false positives.")
    sys.exit(0)
else:
    print("\n⚠️  SOME TESTS FAILED")
    sys.exit(1)
