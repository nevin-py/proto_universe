"""Minimal single-client test to debug PoT circuit issues

Tests with different configurations to isolate the problem:
1. Simple AdditionFCircuit (verify IVC works)
2. TrainingStepCircuit with 1 sample
3. TrainingStepCircuit with multiple samples
4. Different model architectures
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crypto.zkp_prover import TrainingProofProver
from src.models.mnist import create_mnist_model


def test_addition_circuit():
    """Test 1: Verify basic IVC folding works with AdditionFCircuit"""
    print("\n" + "="*80)
    print("TEST 1: AdditionFCircuit (IVC Baseline)")
    print("="*80)
    
    try:
        from fl_zkp_bridge import FLZKPProver
        
        prover = FLZKPProver()
        prover.initialize(0)
        
        # Simple addition: z0=0, z1=z0+5, z2=z1+10, z3=z2+15
        values = [5, 10, 15, 20]
        
        print(f"Testing {len(values)} IVC steps...")
        for i, val in enumerate(values):
            print(f"  Step {i+1}: z' = z + {val}")
            prover.prove_step(val)
        
        print("  Generating final proof...")
        proof_bytes = prover.generate_final_proof()
        
        print(f"✓ SUCCESS: AdditionFCircuit works ({len(proof_bytes)} bytes)")
        return True
        
    except ImportError:
        print("✗ SKIPPED: fl_zkp_bridge not available")
        return None
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_circuit_single_sample():
    """Test 2: TrainingStepCircuit with 1 sample"""
    print("\n" + "="*80)
    print("TEST 2: TrainingStepCircuit (1 sample)")
    print("="*80)
    
    try:
        # Load single MNIST sample
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        
        # Get first sample
        x, y = dataset[0]
        print(f"Sample: label={y}, shape={x.shape}")
        
        # Create linear model
        model = create_mnist_model('linear')
        params = list(model.parameters())
        weights = params[0]  # 10×784
        bias = params[1]     # 10
        
        print(f"Model: weights={weights.shape}, bias={bias.shape}")
        
        # Create prover
        prover = TrainingProofProver()
        
        # Generate proof with 1 sample
        print("Generating PoT proof with 1 sample...")
        train_data = [(x, int(y))]
        
        proof = prover.prove_training(
            weights=weights,
            bias=bias,
            train_data=train_data,
            client_id=0,
            round_number=0,
            batch_size=1
        )
        
        print(f"✓ SUCCESS: Proof generated ({proof.proof_size} bytes, {proof.prove_time_ms:.1f}ms)")
        print(f"  Fingerprint: {proof.model_fingerprint}")
        print(f"  Steps: {proof.num_steps}")
        print(f"  Real proof: {proof.is_real}")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_circuit_multi_sample():
    """Test 3: TrainingStepCircuit with multiple samples"""
    print("\n" + "="*80)
    print("TEST 3: TrainingStepCircuit (4 samples)")
    print("="*80)
    
    try:
        # Load MNIST samples
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        
        # Get 4 samples with different labels
        train_data = []
        for i in range(4):
            x, y = dataset[i]
            train_data.append((x, int(y)))
            print(f"  Sample {i+1}: label={y}")
        
        # Create linear model
        model = create_mnist_model('linear')
        params = list(model.parameters())
        weights = params[0]
        bias = params[1]
        
        # Create prover
        prover = TrainingProofProver()
        
        # Generate proof
        print("Generating PoT proof with 4 samples...")
        proof = prover.prove_training(
            weights=weights,
            bias=bias,
            train_data=train_data,
            client_id=0,
            round_number=0,
            batch_size=4
        )
        
        print(f"✓ SUCCESS: Proof generated ({proof.proof_size} bytes, {proof.prove_time_ms:.1f}ms)")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_architectures():
    """Test 4: Different model architectures"""
    print("\n" + "="*80)
    print("TEST 4: Different Architectures")
    print("="*80)
    
    architectures = ['linear', 'mlp', 'cnn']
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    
    # Get 2 samples
    train_data = [(dataset[0][0], int(dataset[0][1])), 
                  (dataset[1][0], int(dataset[1][1]))]
    
    results = {}
    
    for arch in architectures:
        print(f"\n--- Testing {arch.upper()} ---")
        
        try:
            model = create_mnist_model(arch)
            params = list(model.parameters())
            
            # For now, only test linear (others need circuit support)
            if arch != 'linear':
                print(f"✗ SKIPPED: {arch} not yet supported in circuit")
                results[arch] = 'skipped'
                continue
            
            weights = params[0]
            bias = params[1]
            
            prover = TrainingProofProver()
            proof = prover.prove_training(
                weights=weights,
                bias=bias,
                train_data=train_data,
                client_id=0,
                round_number=0,
                batch_size=2
            )
            
            print(f"✓ SUCCESS: {arch} works")
            results[arch] = 'success'
            
        except Exception as e:
            print(f"✗ FAILED: {arch} - {e}")
            results[arch] = f'failed: {str(e)[:50]}'
    
    return results


def main():
    print("\n" + "="*80)
    print("FiZK Circuit Debug Suite")
    print("="*80)
    print("\nTesting PoT circuit with minimal configurations")
    print("Goal: Isolate RemainderNotZero error cause\n")
    
    results = {}
    
    # Test 1: Basic IVC
    results['addition_circuit'] = test_addition_circuit()
    
    # Test 2: Single sample
    results['single_sample'] = test_training_circuit_single_sample()
    
    # Test 3: Multiple samples (this is where it fails)
    results['multi_sample'] = test_training_circuit_multi_sample()
    
    # Test 4: Different architectures
    results['architectures'] = test_different_architectures()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    def status_str(result):
        if result is True:
            return "✓ PASS"
        elif result is False:
            return "✗ FAIL"
        elif result is None:
            return "○ SKIP"
        else:
            return str(result)
    
    print(f"1. AdditionFCircuit:        {status_str(results.get('addition_circuit'))}")
    print(f"2. Single Sample:           {status_str(results.get('single_sample'))}")
    print(f"3. Multiple Samples:        {status_str(results.get('multi_sample'))}")
    print(f"4. Architecture Tests:")
    if isinstance(results.get('architectures'), dict):
        for arch, res in results['architectures'].items():
            print(f"   - {arch:10s}:          {status_str(res)}")
    
    print("\n" + "="*80)
    
    # Determine next steps
    if results.get('addition_circuit') is False:
        print("\n🔴 CRITICAL: Basic IVC folding doesn't work")
        print("   → Check fl_zkp_bridge compilation")
        print("   → Verify ProtoGalaxy setup")
    elif results.get('single_sample') is False:
        print("\n🔴 ISSUE: Single sample fails")
        print("   → Check label encoding (float vs field)")
        print("   → Verify fingerprint calculation")
        print("   → Debug circuit constraints")
    elif results.get('multi_sample') is False:
        print("\n🔴 ISSUE: Multiple samples fail at step 2")
        print("   → State transition problem")
        print("   → External inputs mismatch between steps")
        print("   → Fingerprint not carried correctly")
    else:
        print("\n✓ All tests passed! Circuit is working correctly.")


if __name__ == "__main__":
    main()
