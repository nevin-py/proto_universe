"""Diagnose what exactly breaks ProtoGalaxy folding"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fl_zkp_bridge import FLTrainingProver
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False


def test_scenario(name, x1, y1, x2, y2, w, b, r):
    """Test a specific scenario"""
    print(f"\n--- {name} ---")
    
    prover = FLTrainingProver()
    prover.initialize(0)
    
    try:
        prover.prove_training_step(x1, y1, w, b, r)
        print("✓ Step 1")
    except Exception as e:
        print(f"✗ Step 1: {e}")
        return False
    
    try:
        prover.prove_training_step(x2, y2, w, b, r)
        print("✓ Step 2")
        return True
    except Exception as e:
        print(f"✗ Step 2: {e}")
        return False


def main():
    if not BRIDGE_AVAILABLE:
        print("Bridge not available")
        return
    
    print("="*80)
    print("DIAGNOSTIC TESTS - What breaks ProtoGalaxy?")
    print("="*80)
    
    INPUT_DIM = 784
    NUM_CLASSES = 10
    WEIGHT_COUNT = INPUT_DIM * NUM_CLASSES
    
    # Test 1: Everything identical (baseline)
    x_zero = [0.0] * INPUT_DIM
    y_zero = 0.0
    w_zero = [0.0] * WEIGHT_COUNT
    b_zero = [0.0] * NUM_CLASSES
    r_zero = [0.0] * NUM_CLASSES
    
    test1 = test_scenario(
        "Test 1: All zeros, repeated",
        x_zero, y_zero, x_zero, y_zero, w_zero, b_zero, r_zero
    )
    
    # Test 2: Change only x (smallest change)
    x_one = [1.0] * INPUT_DIM
    test2 = test_scenario(
        "Test 2: Change x only (0→1)",
        x_zero, y_zero, x_one, y_zero, w_zero, b_zero, r_zero
    )
    
    # Test 3: Change only y
    test3 = test_scenario(
        "Test 3: Change y only (0→1)",
        x_zero, y_zero, x_zero, 1.0, w_zero, b_zero, r_zero
    )
    
    # Test 4: Change only first element of x
    x_single = [0.0] * INPUT_DIM
    x_single[0] = 1.0
    test4 = test_scenario(
        "Test 4: Change x[0] only",
        x_zero, y_zero, x_single, y_zero, w_zero, b_zero, r_zero
    )
    
    # Test 5: Non-zero w (does computation matter?)
    w_small = [0.001] * WEIGHT_COUNT
    test5 = test_scenario(
        "Test 5: With non-zero w, x changes",
        x_zero, y_zero, x_one, y_zero, w_small, b_zero, r_zero
    )
    
    # Test 6: Change w between steps (shouldn't matter, but test)
    w_one = [1.0] * WEIGHT_COUNT
    test6 = test_scenario(
        "Test 6: Change w between steps",
        x_zero, y_zero, x_zero, y_zero, w_zero, b_zero, r_zero
    )
    # Note: This won't work since we need to pass same w both times
    # Skip for now
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"All zeros repeated:       {'✓' if test1 else '✗'}")
    print(f"Change x only:            {'✓' if test2 else '✗'}")
    print(f"Change y only:            {'✓' if test3 else '✗'}")
    print(f"Change x[0] only:         {'✓' if test4 else '✗'}")
    print(f"With non-zero w:          {'✓' if test5 else '✗'}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if not test2 and not test3:
        print("❌ ANY change to inputs breaks folding")
        print("   → ProtoGalaxy incompatible with external inputs changing")
        print("   → May need to restructure circuit entirely")
    elif not test2 and test3:
        print("❌ Changing x breaks, changing y works")
        print("   → Issue in forward pass (W·x computation)")
    elif test2 and not test3:
        print("❌ Changing y breaks, changing x works")
        print("   → Issue in one-hot/label handling")
    else:
        print("✅ Both work individually - issue is combination")


if __name__ == "__main__":
    main()
