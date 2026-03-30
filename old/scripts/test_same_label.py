"""Test if Lagrange indicators work with same label across steps"""

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
    
    print("\n" + "="*80)
    print("TEST: Same label across steps (y=5 both times)")
    print("="*80)
    
    prover = FLTrainingProver()
    prover.initialize(100)
    
    INPUT_DIM = 784
    NUM_CLASSES = 10
    WEIGHT_COUNT = INPUT_DIM * NUM_CLASSES
    
    # Step 1: y=5, different x
    x1 = [1.0] * INPUT_DIM
    y1 = 5.0
    w = [0.1] * WEIGHT_COUNT
    b = [0.1] * NUM_CLASSES
    r = [1.0] * NUM_CLASSES
    
    print("Step 1: x=1.0, y=5")
    try:
        prover.prove_training_step(x1, y1, w, b, r)
        print("✓ Step 1 passed")
    except Exception as e:
        print(f"✗ Step 1 failed: {e}")
        return
    
    # Step 2: y=5 (SAME), different x
    x2 = [2.0] * INPUT_DIM
    y2 = 5.0  # SAME label
    
    print("Step 2: x=2.0, y=5 (same label)")
    try:
        prover.prove_training_step(x2, y2, w, b, r)
        print("✓ Step 2 passed")
        print("\n✅ SAME LABEL works with Lagrange indicators!")
    except Exception as e:
        print(f"✗ Step 2 failed: {e}")
        print("\n❌ Even SAME label fails - Lagrange has issues")


if __name__ == "__main__":
    main()
