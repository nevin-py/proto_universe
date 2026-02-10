"""
Test suite for ProtoGalaxy variable k support via Python bindings.

This test suite verifies:
1. Standard single-instance folding (k=1)
2. Multi-instance folding (k=3, total 4 instances)
3. Invalid k constraint validation (k+1 must be power of two)
"""

import pytest

try:
    import sonobe_protogalaxy
    BINDINGS_AVAILABLE = True
except ImportError:
    BINDINGS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="sonobe_protogalaxy bindings not built")


class TestVariableK:
    """Test cases for variable k (multi-instance folding) support."""
    
    def test_k_equals_1_standard_folding(self):
        """Test Case 1: k=1 (standard single-instance folding)"""
        # Initialize with k=1
        pg = sonobe_protogalaxy.PyProtoGalaxy(k=1)
        
        # Verify k value
        assert pg.k == 1
        assert pg.total_instances() == 2
        assert pg.is_valid()
        
        # Test preprocess (simplified interface)
        result = pg.preprocess()
        assert "k=1" in result
        assert "2 instances" in result
        
        # Test prove_step (simplified interface)
        result = pg.prove_step()
        assert "k=1" in result
        
        print(f"✓ Test Case 1 passed: {pg}")
    
    def test_k_equals_3_multi_instance_folding(self):
        """Test Case 2: k=3 (multi-instance folding with 4 total instances)"""
        # Initialize with k=3
        pg = sonobe_protogalaxy.PyProtoGalaxy(k=3)
        
        # Verify k value
        assert pg.k == 3
        assert pg.total_instances() == 4
        assert pg.is_valid()
        
        # Test preprocess
        result = pg.preprocess()
        assert "k=3" in result
        assert "4 instances" in result
        
        # Test prove_step
        result = pg.prove_step()
        assert "k=3" in result
        
        print(f"✓ Test Case 2 passed: {pg}")
    
    def test_invalid_k_equals_2_constraint_failure(self):
        """Test Case 3: Invalid k=2 (k+1=3 is not a power of two)"""
        # Attempt to initialize with k=2 (invalid)
        with pytest.raises(ValueError) as exc_info:
            sonobe_protogalaxy.PyProtoGalaxy(k=2)
        
        # Verify error message
        error_msg = str(exc_info.value)
        assert "k+1" in error_msg or "power of two" in error_msg
        
        print(f"✓ Test Case 3 passed: Correctly rejected k=2 with error: {error_msg}")
    
    def test_invalid_k_equals_0(self):
        """Test Case 4: Invalid k=0 (k must be at least 1)"""
        with pytest.raises(ValueError) as exc_info:
            sonobe_protogalaxy.PyProtoGalaxy(k=0)
        
        error_msg = str(exc_info.value)
        assert "at least 1" in error_msg or "k must be" in error_msg
        
        print(f"✓ Test Case 4 passed: Correctly rejected k=0")
    
    def test_valid_k_values(self):
        """Test Case 5: Verify multiple valid k values"""
        # Valid k values: 1, 3, 7, 15, 31, ... (where k+1 is a power of 2)
        valid_k_values = [1, 3, 7, 15, 31]
        
        for k in valid_k_values:
            pg = sonobe_protogalaxy.PyProtoGalaxy(k=k)
            assert pg.k == k
            assert pg.total_instances() == k + 1
            assert pg.is_valid()
            assert (k + 1) & (k) == 0  # Verify k+1 is power of 2
        
        print(f"✓ Test Case 5 passed: All valid k values accepted: {valid_k_values}")
    
    def test_invalid_k_values(self):
        """Test Case 6: Verify multiple invalid k values are rejected"""
        # Invalid k values (k+1 is not a power of 2)
        invalid_k_values = [2, 4, 5, 6, 8, 9, 10]
        
        for k in invalid_k_values:
            with pytest.raises(ValueError):
                sonobe_protogalaxy.PyProtoGalaxy(k=k)
        
        print(f"✓ Test Case 6 passed: All invalid k values rejected: {invalid_k_values}")
    
    def test_repr(self):
        """Test Case 7: Verify string representation"""
        pg = sonobe_protogalaxy.PyProtoGalaxy(k=7)
        repr_str = repr(pg)
        
        assert "PyProtoGalaxy" in repr_str
        assert "k=7" in repr_str
        assert "total_instances=8" in repr_str
        
        print(f"✓ Test Case 7 passed: {repr_str}")


if __name__ == "__main__":
    if not BINDINGS_AVAILABLE:
        print("ERROR: sonobe_protogalaxy bindings not available.")
        print("Please build the bindings first:")
        print("  cd sonobe-py-bindings")
        print("  maturin develop")
        exit(1)
    
    # Run tests manually
    print("Running ProtoGalaxy Variable k Tests...")
    print("=" * 60)
    
    test = TestVariableK()
    
    try:
        test.test_k_equals_1_standard_folding()
        test.test_k_equals_3_multi_instance_folding()
        test.test_invalid_k_equals_2_constraint_failure()
        test.test_invalid_k_equals_0()
        test.test_valid_k_values()
        test.test_invalid_k_values()
        test.test_repr()
        
        print("=" * 60)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
