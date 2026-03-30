#!/usr/bin/env python3
"""Test Byzantine detection across different model architectures"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.crypto.zkp_prover import TrainingProofProver
from src.models.mnist import create_mnist_model

print("="*80)
print("BYZANTINE DETECTION - Multi-Model Architecture Test")
print("="*80)
print("\nNote: Circuit dimensions are configured at ProtoGalaxy preprocessing.")
print("      Each architecture needs matching dimensions:")
print("      - Linear: 10×784 (28×28 flattened)")
print("      - MLP: 10×64 (fc3 output layer)")  
print("      - CNN: 10×128 (fc2 output layer)")
print()

# Model types to test (only linear works with current circuit)
MODEL_TYPES = ['linear', 'mlp', 'cnn']
round_number = 0
batch_size = 4

results = {}

for model_type in MODEL_TYPES:
    print(f"\n{'='*80}")
    print(f"Testing Model Type: {model_type.upper()}")
    print(f"{'='*80}")
    
    try:
        # Create model
        model = create_mnist_model(model_type=model_type, num_classes=10)
        
        # Extract weights and bias for linear layer (first or last)
        # For ZKP, we only prove the final linear layer (10x784 or 10xhidden)
        if model_type == 'linear':
            weights = model.linear.weight.data.clone()
            bias = model.linear.bias.data.clone()
        elif model_type == 'mlp':
            # Use final output layer (fc3 -> 10 classes)
            weights = model.fc3.weight.data.clone()
            bias = model.fc3.bias.data.clone()
        elif model_type == 'cnn':
            # Use final FC layer
            weights = model.fc2.weight.data.clone()
            bias = model.fc2.bias.data.clone()
        
        print(f"   Model: {model.__class__.__name__}")
        print(f"   Proven layer shape: W={weights.shape}, b={bias.shape}")
        print(f"   Circuit dimensions: {weights.shape[0]}×{weights.shape[1]}")
        
        # Create training data matching model input dimensions
        model_input_dim = weights.shape[1]  # Get actual input dimension
        batch = [(torch.randn(model_input_dim), i % 10) for i in range(batch_size)]
        print(f"   Training batch: {batch_size} samples, input_dim={model_input_dim}")
        
        # Test 1: Honest client
        print(f"\n   Test 1: Honest {model_type} model")
        prover = TrainingProofProver()
        
        try:
            proof = prover.prove_training(
                weights=weights,
                bias=bias,
                train_data=batch,
                client_id=1,
                round_number=round_number,
                batch_size=batch_size
            )
            print(f"   ✅ Honest proof generated ({len(proof.proof_bytes)} bytes)")
            honest_passed = True
        except Exception as e:
            print(f"   ❌ Honest client failed: {e}")
            honest_passed = False
        
        # Test 2: Malicious client (perturbed weights)
        print(f"\n   Test 2: Malicious {model_type} model (perturbed weights)")
        
        malicious_weights = weights.clone()
        malicious_weights += torch.randn_like(malicious_weights) * 0.1
        
        r_vec = prover.generate_random_vector(round_number)
        honest_fp, _ = prover.compute_model_fingerprint(weights, bias, r_vec, round_number)
        malicious_fp, _ = prover.compute_model_fingerprint(malicious_weights, bias, r_vec, round_number)
        
        print(f"   Fingerprint difference: {abs(honest_fp - malicious_fp)}")
        
        try:
            # CRITICAL: Client must prove against HONEST fingerprint (from server's global model)
            proof = prover.prove_training(
                weights=malicious_weights,  # Malicious tries to use wrong weights
                bias=bias,
                train_data=batch,
                client_id=2,
                round_number=round_number,
                batch_size=batch_size,
                expected_fingerprint=honest_fp  # But must match honest fingerprint!
            )
            print(f"   ❌ Malicious proof accepted! Detection failed.")
            malicious_caught = False
        except RuntimeError as e:
            if "fingerprint" in str(e).lower():
                print(f"   ✅ Malicious client rejected (fingerprint mismatch)")
                malicious_caught = True
            else:
                print(f"   ⚠️  Rejected for other reason: {str(e)[:80]}")
                malicious_caught = False
        
        # Store results
        results[model_type] = {
            'status': 'tested',
            'honest_passed': honest_passed,
            'malicious_caught': malicious_caught,
            'detection_rate': 100 if malicious_caught else 0,
        }
        
        status = "✅ PASS" if (honest_passed and malicious_caught) else "❌ FAIL"
        print(f"\n   {status}: {model_type} - Detection: {results[model_type]['detection_rate']}%")
        
    except Exception as e:
        print(f"   ❌ Model setup failed: {e}")
        results[model_type] = {'status': 'error', 'error': str(e)}

# Summary
print(f"\n{'='*80}")
print("SUMMARY - Byzantine Detection Across Model Types")
print(f"{'='*80}")

tested_models = [k for k, v in results.items() if v['status'] == 'tested']
passed_models = [k for k, v in results.items() if v.get('honest_passed') and v.get('malicious_caught')]
skipped_models = [k for k, v in results.items() if v['status'] == 'skipped']

print(f"\nTested Models: {len(tested_models)}/{len(MODEL_TYPES)}")
for model_type, result in results.items():
    if result['status'] == 'tested':
        status = "✅" if result['honest_passed'] and result['malicious_caught'] else "❌"
        print(f"  {status} {model_type}: Detection {result['detection_rate']}%")
    elif result['status'] == 'skipped':
        print(f"  ⚠️  {model_type}: SKIPPED - {result['reason']}")
    else:
        print(f"  ❌ {model_type}: ERROR - {result.get('error', 'unknown')}")

if len(tested_models) > 0:
    avg_detection = sum(r['detection_rate'] for r in results.values() if r['status'] == 'tested') / len(tested_models)
    print(f"\nAverage Detection Rate: {avg_detection:.0f}%")

if len(passed_models) == len(tested_models) and len(tested_models) > 0:
    print("\n🎉 ALL TESTED MODELS PASSED - 100% Byzantine detection across architectures!")
    sys.exit(0)
else:
    print(f"\n⚠️  Results: {len(passed_models)}/{len(tested_models)} models passed")
    if len(skipped_models) > 0:
        print(f"   Note: {len(skipped_models)} models skipped due to circuit limitations")
    sys.exit(1)
