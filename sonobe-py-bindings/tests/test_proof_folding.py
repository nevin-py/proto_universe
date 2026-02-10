"""
Test suite for ProtoGalaxy full proof folding functionality.

This test suite verifies:
1. Proof creation and serialization
2. Multi-proof folding (k proofs -> 1 proof)
3. Proof verification
4. Error handling for invalid inputs
"""

import pytest
import json

try:
    import sonobe_protogalaxy
    BINDINGS_AVAILABLE = True
except ImportError:
    BINDINGS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="sonobe_protogalaxy bindings not built")


class TestProofFolding:
    """Test cases for full proof folding functionality."""
    
    def test_proof_creation(self):
        """Test Case 1: Create and serialize proofs"""
        # Create a proof
        proof = sonobe_protogalaxy.PyProof(
            instance_data=bytes([1, 2, 3, 4]),
            witness_data=bytes([5, 6, 7, 8]),
            public_inputs=["0x01", "0x02"]
        )
        
        assert len(proof.instance_data) == 4
        assert len(proof.witness_data) == 4
        assert len(proof.public_inputs) == 2
        
        # Test serialization
        json_str = proof.to_json()
        assert isinstance(json_str, str)
        
        # Test deserialization
        proof2 = sonobe_protogalaxy.PyProof.from_json(json_str)
        assert proof2.instance_data == proof.instance_data
        assert proof2.witness_data == proof.witness_data
        assert proof2.public_inputs == proof.public_inputs
        
        print(f"✓ Test Case 1 passed: {proof}")
    
    def test_witness_creation(self):
        """Test Case 2: Create and serialize witness"""
        witness = sonobe_protogalaxy.PyWitness(
            w=["0x01", "0x02", "0x03"]
        )
        
        assert len(witness.w) == 3
        
        # Test serialization
        json_str = witness.to_json()
        witness2 = sonobe_protogalaxy.PyWitness.from_json(json_str)
        assert witness2.w == witness.w
        
        print(f"✓ Test Case 2 passed: {witness}")
    
    def test_fold_single_proof_k1(self):
        """Test Case 3: Fold 1 proof into running proof (k=1)"""
        pg = sonobe_protogalaxy.PyProtoGalaxy(k=1)
        
        # Create running proof
        running_proof = pg.create_dummy_proof(witness_size=10, public_input_size=2)
        
        # Create 1 incoming proof
        incoming_proof = pg.create_dummy_proof(witness_size=10, public_input_size=2)
        
        # Fold proofs
        folded_proof = pg.fold_proofs(running_proof, [incoming_proof])
        
        assert isinstance(folded_proof, sonobe_protogalaxy.PyProof)
        assert len(folded_proof.instance_data) > 0
        assert len(folded_proof.witness_data) > 0
        
        print(f"✓ Test Case 3 passed: Folded 1 proof with k=1")
        print(f"  Running proof: {running_proof}")
        print(f"  Incoming proof: {incoming_proof}")
        print(f"  Folded proof: {folded_proof}")
    
    def test_fold_multiple_proofs_k3(self):
        """Test Case 4: Fold 3 proofs into running proof (k=3)"""
        pg = sonobe_protogalaxy.PyProtoGalaxy(k=3)
        
        # Create running proof
        running_proof = pg.create_dummy_proof(witness_size=10, public_input_size=2)
        
        # Create 3 incoming proofs
        incoming_proofs = [
            pg.create_dummy_proof(witness_size=10, public_input_size=2)
            for _ in range(3)
        ]
        
        # Fold proofs
        folded_proof = pg.fold_proofs(running_proof, incoming_proofs)
        
        assert isinstance(folded_proof, sonobe_protogalaxy.PyProof)
        
        print(f"✓ Test Case 4 passed: Folded 3 proofs with k=3")
        print(f"  Total proofs folded: {len(incoming_proofs) + 1}")
        print(f"  Folded proof: {folded_proof}")
    
    def test_fold_wrong_number_of_proofs(self):
        """Test Case 5: Error when wrong number of proofs provided"""
        pg = sonobe_protogalaxy.PyProtoGalaxy(k=3)
        
        running_proof = pg.create_dummy_proof(witness_size=10, public_input_size=2)
        
        # Try to fold with wrong number of incoming proofs (2 instead of 3)
        incoming_proofs = [
            pg.create_dummy_proof(witness_size=10, public_input_size=2)
            for _ in range(2)
        ]
        
        with pytest.raises(ValueError) as exc_info:
            pg.fold_proofs(running_proof, incoming_proofs)
        
        error_msg = str(exc_info.value)
        assert "Expected 3 incoming proofs" in error_msg
        
        print(f"✓ Test Case 5 passed: Correctly rejected wrong number of proofs")
        print(f"  Error: {error_msg}")
    
    def test_verify_proof(self):
        """Test Case 6: Verify a proof"""
        pg = sonobe_protogalaxy.PyProtoGalaxy(k=1)
        
        # Create a proof
        proof = pg.create_dummy_proof(witness_size=10, public_input_size=2)
        
        # Verify it
        is_valid = pg.verify_proof(proof)
        assert is_valid is True
        
        # Create an empty/invalid proof
        invalid_proof = sonobe_protogalaxy.PyProof(
            instance_data=bytes([]),
            witness_data=bytes([]),
            public_inputs=[]
        )
        
        is_valid = pg.verify_proof(invalid_proof)
        assert is_valid is False
        
        print(f"✓ Test Case 6 passed: Proof verification works")
    
    def test_iterative_folding(self):
        """Test Case 7: Iteratively fold multiple batches of proofs"""
        pg = sonobe_protogalaxy.PyProtoGalaxy(k=1)
        
        # Start with initial proof
        running_proof = pg.create_dummy_proof(witness_size=10, public_input_size=2)
        
        # Fold 5 proofs iteratively
        for i in range(5):
            incoming_proof = pg.create_dummy_proof(witness_size=10, public_input_size=2)
            running_proof = pg.fold_proofs(running_proof, [incoming_proof])
            assert pg.verify_proof(running_proof)
        
        print(f"✓ Test Case 7 passed: Iteratively folded 5 proofs")
        print(f"  Final proof: {running_proof}")
    
    def test_large_k_value(self):
        """Test Case 8: Test with larger k value (k=7, 8 total proofs)"""
        pg = sonobe_protogalaxy.PyProtoGalaxy(k=7)
        
        assert pg.k == 7
        assert pg.total_instances() == 8
        
        # Create running proof
        running_proof = pg.create_dummy_proof(witness_size=10, public_input_size=2)
        
        # Create 7 incoming proofs
        incoming_proofs = [
            pg.create_dummy_proof(witness_size=10, public_input_size=2)
            for _ in range(7)
        ]
        
        # Fold all proofs
        folded_proof = pg.fold_proofs(running_proof, incoming_proofs)
        
        assert isinstance(folded_proof, sonobe_protogalaxy.PyProof)
        
        print(f"✓ Test Case 8 passed: Folded 7 proofs with k=7 (8 total)")
        print(f"  Folded proof: {folded_proof}")


if __name__ == "__main__":
    if not BINDINGS_AVAILABLE:
        print("ERROR: sonobe_protogalaxy bindings not available.")
        print("Please build the bindings first:")
        print("  cd sonobe-py-bindings")
        print("  maturin develop")
        exit(1)
    
    # Run tests manually
    print("Running ProtoGalaxy Full Proof Folding Tests...")
    print("=" * 60)
    
    test = TestProofFolding()
    
    try:
        test.test_proof_creation()
        test.test_witness_creation()
        test.test_fold_single_proof_k1()
        test.test_fold_multiple_proofs_k3()
        test.test_fold_wrong_number_of_proofs()
        test.test_verify_proof()
        test.test_iterative_folding()
        test.test_large_k_value()
        
        print("=" * 60)
        print("✓ All proof folding tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
