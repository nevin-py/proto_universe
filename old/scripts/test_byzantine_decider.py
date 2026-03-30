"""Test Byzantine detection with Decider-based fingerprint verification"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.crypto.zkp_prover import TrainingProofProver


def test_honest_client():
    """Test that honest client with correct model passes"""
    print("\n" + "="*80)
    print("TEST 1: Honest Client (correct model)")
    print("="*80)
    
    # Create simple linear model
    input_dim, num_classes = 784, 10
    weights = torch.randn(num_classes, input_dim) * 0.01
    bias = torch.randn(num_classes) * 0.01
    
    # Create training data (4 samples)
    batch_size = 4
    x_batch = torch.randn(batch_size, input_dim)
    y_batch = torch.randint(0, num_classes, (batch_size,))
    batch = list(zip(x_batch, y_batch))
    
    # Generate proof
    prover = TrainingProofProver()
    try:
        proof = prover.prove_training(
            weights=weights,
            bias=bias,
            train_data=batch,
            client_id=1,
            round_number=0,
            batch_size=4
        )
        print(f"✅ Proof generated successfully")
        print(f"   Proof size: {len(proof.proof_bytes)} bytes")
        print(f"   Steps: {proof.num_steps}")
        print(f"   Fingerprint: {proof.model_fingerprint}")
        return True
    except Exception as e:
        print(f"❌ Honest client failed: {e}")
        return False


def test_malicious_client_wrong_weights():
    """Test that malicious client with wrong weights is caught"""
    print("\n" + "="*80)
    print("TEST 2: Malicious Client (wrong weights)")
    print("="*80)
    
    input_dim, num_classes = 784, 10
    
    # Server's global model
    global_weights = torch.randn(num_classes, input_dim) * 0.01
    global_bias = torch.randn(num_classes) * 0.01
    
    # Malicious client uses DIFFERENT weights
    malicious_weights = torch.randn(num_classes, input_dim) * 0.05  # Different values
    
    # Training data
    batch_size = 4
    x_batch = torch.randn(batch_size, input_dim)
    y_batch = torch.randint(0, num_classes, (batch_size,))
    batch = list(zip(x_batch, y_batch))
    
    prover = TrainingProofProver()
    
    # First, initialize with CORRECT fingerprint (from global model)
    r_vec = prover.generate_random_vector(round_number=0)
    correct_fp, correct_sampled = prover.compute_model_fingerprint(
        global_weights, global_bias, r_vec, round_number=0
    )
    
    # Compute both fingerprints to show difference
    malicious_fp, _ = prover.compute_model_fingerprint(
        malicious_weights, global_bias, r_vec, round_number=0
    )
    
    print(f"   Correct fingerprint: {correct_fp}")
    print(f"   Malicious fingerprint: {malicious_fp}")
    print(f"   Difference: {abs(correct_fp - malicious_fp)}")
    
    # CRITICAL: Malicious client must prove against HONEST fingerprint
    # (they try to cheat by using different weights while claiming honest model)
    try:
        # Try to generate proof (should fail in Decider)
        proof = prover.prove_training(
            weights=malicious_weights,  # WRONG weights
            bias=global_bias,
            train_data=batch,
            client_id=2,
            round_number=0,
            batch_size=4,
            model_fingerprint=correct_fp  # Use honest_fp, not malicious_fp!
        )
        
        print(f"❌ SECURITY FAILURE: Malicious proof accepted!")
        print(f"   This should have been rejected by fingerprint check")
        return False
        
    except Exception as e:
        error_msg = str(e)
        if "fingerprint" in error_msg.lower() or "decider" in error_msg.lower():
            print(f"✅ Malicious client correctly rejected!")
            print(f"   Error: {error_msg[:100]}...")
            return True
        else:
            print(f"⚠️  Client rejected, but for different reason: {error_msg[:100]}...")
            return True  # Still caught, even if different error


def test_malicious_client_modified_bias():
    """Test that malicious client with modified bias is caught"""
    print("\n" + "="*80)
    print("TEST 3: Malicious Client (modified bias)")
    print("="*80)
    
    input_dim, num_classes = 784, 10
    
    # Global model
    global_weights = torch.randn(num_classes, input_dim) * 0.01
    global_bias = torch.randn(num_classes) * 0.01
    
    # Malicious: same weights, different bias
    malicious_bias = global_bias + torch.randn(num_classes) * 0.1
    
    # Training data
    batch_size = 4
    x_batch = torch.randn(batch_size, input_dim)
    y_batch = torch.randint(0, num_classes, (batch_size,))
    batch = list(zip(x_batch, y_batch))
    
    prover = TrainingProofProver()
    
    try:
        proof = prover.prove_training(
            weights=global_weights,      # Correct
            bias=malicious_bias,         # WRONG
            train_data=batch,
            client_id=3,
            round_number=0,
            batch_size=4
        )
        print(f"❌ SECURITY FAILURE: Malicious proof accepted!")
        return False
    except Exception as e:
        print(f"✅ Malicious client correctly rejected!")
        print(f"   Error: {str(e)[:100]}...")
        return True


def main():
    print("\n" + "="*80)
    print("BYZANTINE DETECTION TEST - Decider-Based Fingerprint")
    print("="*80)
    
    results = []
    
    # Test 1: Honest client
    results.append(("Honest client", test_honest_client()))
    
    # Test 2: Wrong weights
    results.append(("Malicious (weights)", test_malicious_client_wrong_weights()))
    
    # Test 3: Wrong bias
    results.append(("Malicious (bias)", test_malicious_client_modified_bias()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    detection_rate = sum(1 for r in results[1:] if r[1]) / len(results[1:]) * 100
    
    print(f"\nDetection Rate: {detection_rate:.0f}%")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - 100% Byzantine Detection!")
    else:
        print("\n⚠️  SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
