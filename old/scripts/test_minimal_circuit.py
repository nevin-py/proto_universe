"""Absolute minimal test - prove 2 steps without model parameters

Test if ProtoGalaxy folding works at all with our circuit structure
by removing as many variables as possible.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fl_zkp_bridge import FLTrainingProver
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    print("⚠️ fl_zkp_bridge not available")


def test_minimal_protogalaxy():
    """Test minimal ProtoGalaxy setup with simplest possible inputs"""
    if not BRIDGE_AVAILABLE:
        print("Cannot test - bridge not available")
        return False
    
    print("\n" + "="*80)
    print("MINIMAL PROTOGALAXY TEST")
    print("="*80)
    print("Testing if ProtoGalaxy folding works with our circuit at all\n")
    
    # Initialize with fingerprint = 0 (simplest case)
    prover = FLTrainingProver()
    prover.initialize(0)
    print("✓ Initialized with fingerprint=0")
    
    # Create minimal inputs (all zeros to minimize complexity)
    INPUT_DIM = 784
    NUM_CLASSES = 10
    WEIGHT_COUNT = INPUT_DIM * NUM_CLASSES
    
    print(f"\nStep 1: Proving with all-zero inputs...")
    x = [0.0] * INPUT_DIM
    y = 0.0  # Label 0
    w = [0.0] * WEIGHT_COUNT
    b = [0.0] * NUM_CLASSES  
    r = [0.0] * NUM_CLASSES
    
    try:
        result = prover.prove_training_step(x, y, w, b, r)
        print(f"✓ Step 1 SUCCESS: {result}")
    except Exception as e:
        print(f"✗ Step 1 FAILED: {e}")
        return False
    
    print(f"\nStep 2: Proving second sample (same inputs)...")
    try:
        result = prover.prove_training_step(x, y, w, b, r)
        print(f"✓ Step 2 SUCCESS: {result}")
        print(f"\n🎉 BOTH STEPS PASSED - ProtoGalaxy folding works!")
        return True
    except Exception as e:
        print(f"✗ Step 2 FAILED: {e}")
        print(f"\n💥 ProtoGalaxy folding fails even with all-zero inputs")
        print(f"   This indicates a fundamental issue with circuit/folding setup")
        return False


def test_varying_inputs():
    """Test if changing inputs between steps causes issues"""
    if not BRIDGE_AVAILABLE:
        return None
    
    print("\n" + "="*80)
    print("VARYING INPUTS TEST")
    print("="*80)
    
    prover = FLTrainingProver()
    prover.initialize(100)  # Non-zero fingerprint
    
    INPUT_DIM = 784
    NUM_CLASSES = 10
    WEIGHT_COUNT = INPUT_DIM * NUM_CLASSES
    
    # Step 1: All ones
    x1 = [1.0] * INPUT_DIM
    y1 = 1.0
    w = [0.1] * WEIGHT_COUNT  # Small non-zero weights
    b = [0.1] * NUM_CLASSES
    r = [1.0] * NUM_CLASSES
    
    print("Step 1: All inputs = 1.0")
    try:
        prover.prove_training_step(x1, y1, w, b, r)
        print("✓ Step 1 passed")
    except Exception as e:
        print(f"✗ Step 1 failed: {e}")
        return False
    
    # Step 2: Different sample (x=0, y=2)
    x2 = [0.0] * INPUT_DIM
    y2 = 2.0
    
    print("Step 2: x=0, y=2 (same w, b, r)")
    try:
        prover.prove_training_step(x2, y2, w, b, r)
        print("✓ Step 2 passed")
        print("\n✅ Varying inputs work!")
        return True
    except Exception as e:
        print(f"✗ Step 2 failed: {e}")
        print(f"\n❌ Varying (x,y) causes failure")
        return False


def main():
    print("\n" + "="*80)
    print("PROTOGALAXY CIRCUIT DEBUG - MINIMAL TESTS")
    print("="*80)
    print("\nGoal: Isolate if ProtoGalaxy folding itself works")
    print("or if there's an issue with our circuit/external inputs\n")
    
    if not BRIDGE_AVAILABLE:
        print("✗ Cannot run tests - fl_zkp_bridge not available")
        return
    
    # Test 1: Simplest possible (all zeros)
    result1 = test_minimal_protogalaxy()
    
    if result1:
        # Test 2: With varying inputs
        result2 = test_varying_inputs()
    else:
        result2 = None
        print("\n⏭️  Skipping varying inputs test (minimal failed)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Minimal (all zeros):  {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"Varying inputs:       {'✓ PASS' if result2 else '✗ FAIL' if result2 is False else '○ SKIP'}")
    
    if not result1:
        print("\n🔴 CRITICAL: ProtoGalaxy folding fails even with trivial inputs")
        print("   Possible causes:")
        print("   1. Circuit state_len mismatch with actual state")
        print("   2. External inputs structure incompatible with ProtoGalaxy")
        print("   3. Constraint system too large for ProtoGalaxy folding")
        print("   4. Bug in folding-schemes ProtoGalaxy implementation")
    elif result1 and not result2:
        print("\n⚠️  Minimal works but varying fails - input handling issue")
    elif result1 and result2:
        print("\n✅ ProtoGalaxy folding works - issue is in our model/data")


if __name__ == "__main__":
    main()
