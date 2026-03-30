"""Test the EXACT scenario that was failing"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fl_zkp_bridge import FLTrainingProver
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False


def main():
    if not BRIDGE_AVAILABLE:
        print("Bridge not available")
        return
    
    print("="*80)
    print("TEST: Exact failing scenario from test_minimal_circuit.py")
    print("="*80)
    
    prover = FLTrainingProver()
    prover.initialize(100)  # Non-zero fingerprint (was in failing test)
    
    INPUT_DIM = 784
    NUM_CLASSES = 10
    WEIGHT_COUNT = INPUT_DIM * NUM_CLASSES
    
    # Exact values from failing test
    x1 = [1.0] * INPUT_DIM
    y1 = 1.0
    w = [0.1] * WEIGHT_COUNT
    b = [0.1] * NUM_CLASSES
    r = [1.0] * NUM_CLASSES
    
    print("\nStep 1: x=[1.0...], y=1.0, w=[0.1...], b=[0.1...], r=[1.0...]")
    try:
        result = prover.prove_training_step(x1, y1, w, b, r)
        print(f"✓ Step 1: {result}")
    except Exception as e:
        print(f"✗ Step 1 failed: {e}")
        return
    
    # Step 2: Different x and y
    x2 = [0.0] * INPUT_DIM
    y2 = 2.0
    
    print("\nStep 2: x=[0.0...], y=2.0 (same w, b, r)")
    try:
        result = prover.prove_training_step(x2, y2, w, b, r)
        print(f"✓ Step 2: {result}")
        print("\n🎉 SUCCESS! Lagrange indicators fixed the issue!")
    except Exception as e:
        print(f"✗ Step 2 failed: {e}")
        print("\n❌ Still failing with exact scenario")
        
        # Try to narrow it down
        print("\n" + "-"*80)
        print("Trying simpler version...")
        print("-"*80)
        
        prover2 = FLTrainingProver()
        prover2.initialize(0)  # Zero fingerprint
        
        w_small = [0.01] * WEIGHT_COUNT
        b_small = [0.01] * NUM_CLASSES
        r_small = [0.1] * NUM_CLASSES
        
        try:
            prover2.prove_training_step(x1, y1, w_small, b_small, r_small)
            print("✓ Step 1 with smaller values")
            prover2.prove_training_step(x2, y2, w_small, b_small, r_small)
            print("✓ Step 2 with smaller values")
            print("\n→ Issue is with LARGE values causing field overflow")
        except Exception as e:
            print(f"✗ Still fails: {e}")
            print("\n→ Issue is more fundamental")


if __name__ == "__main__":
    main()
