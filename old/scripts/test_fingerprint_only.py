"""Test if fingerprint verification alone works"""

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
    print("TEST: Fingerprint verification with varying inputs")
    print("="*80)
    
    INPUT_DIM = 784
    NUM_CLASSES = 10
    WEIGHT_COUNT = INPUT_DIM * NUM_CLASSES
    SAMPLED_W_COUNT = NUM_CLASSES * 100  # 1000 sampled weights
    
    # Create simple test data
    x1 = [0.0] * INPUT_DIM
    y1 = 0.0
    w_flat = [0.01] * WEIGHT_COUNT
    b = [0.01] * NUM_CLASSES
    r = [1.0] * NUM_CLASSES
    w_sampled = [0.01] * SAMPLED_W_COUNT  # Sampled weights
    
    prover = FLTrainingProver()
    prover.initialize(100)
    
    print("\nStep 1: y=0")
    try:
        result = prover.prove_training_step(x1, y1, w_flat, b, r, w_sampled)
        print(f"✓ {result}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return
    
    # Step 2: Different label
    x2 = [1.0] * INPUT_DIM
    y2 = 5.0
    
    print("Step 2: y=5 (different label)")
    try:
        result = prover.prove_training_step(x2, y2, w_flat, b, r, w_sampled)
        print(f"✓ {result}")
        print("\n🎉 Fingerprint verification works with varying labels!")
    except Exception as e:
        print(f"✗ Failed: {e}")
        print("\n❌ Fingerprint verification breaks ProtoGalaxy")


if __name__ == "__main__":
    main()
