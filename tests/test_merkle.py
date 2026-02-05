"""Unit tests for Merkle tree implementation.

Tests cover:
- Hash functions and serialization
- Merkle tree construction
- Proof generation and verification
- Galaxy and Global tree hierarchies
- Gradient commitment scheme
"""

import numpy as np
import pytest
import torch

from src.crypto.merkle import (
    compute_hash,
    verify_proof,
    combine_hashes,
    serialize_gradient,
    MerkleTree,
    GalaxyMerkleTree,
    GlobalMerkleTree,
    GradientCommitment,
)


class TestHashFunctions:
    """Tests for hash utility functions."""
    
    def test_compute_hash_bytes(self):
        """Hash of bytes is deterministic."""
        data = b"test data"
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex
    
    def test_compute_hash_string(self):
        """Hash of string works correctly."""
        data = "test string"
        h = compute_hash(data)
        assert len(h) == 64
    
    def test_compute_hash_numpy_array(self):
        """Hash of numpy array is deterministic."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        hash1 = compute_hash(arr)
        hash2 = compute_hash(arr)
        assert hash1 == hash2
    
    def test_compute_hash_torch_tensor(self):
        """Hash of torch tensor works correctly."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        h = compute_hash(tensor)
        assert len(h) == 64
    
    def test_compute_hash_with_metadata(self):
        """Hash with metadata differs from without."""
        data = b"test"
        h1 = compute_hash(data)
        h2 = compute_hash(data, metadata={'client_id': 1})
        assert h1 != h2
    
    def test_combine_hashes(self):
        """Combined hash is deterministic."""
        h1 = "a" * 64
        h2 = "b" * 64
        combined1 = combine_hashes(h1, h2)
        combined2 = combine_hashes(h1, h2)
        assert combined1 == combined2
        assert len(combined1) == 64
    
    def test_combine_hashes_order_matters(self):
        """Order of hashes affects result (non-commutative)."""
        h1 = "a" * 64
        h2 = "b" * 64
        assert combine_hashes(h1, h2) != combine_hashes(h2, h1)


class TestSerializeGradient:
    """Tests for gradient serialization."""
    
    def test_serialize_numpy(self):
        """Serialize numpy array to bytes."""
        arr = np.array([1.0, 2.0, 3.0])
        data = serialize_gradient(arr)
        assert isinstance(data, bytes)
    
    def test_serialize_torch(self):
        """Serialize torch tensor to bytes."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        data = serialize_gradient(tensor)
        assert isinstance(data, bytes)
    
    def test_serialize_list(self):
        """Serialize list to bytes."""
        lst = [1.0, 2.0, 3.0]
        data = serialize_gradient(lst)
        assert isinstance(data, bytes)


class TestMerkleTree:
    """Tests for base MerkleTree class."""
    
    def test_empty_tree(self):
        """Empty tree has no root."""
        tree = MerkleTree([])
        assert tree.get_root() is None
    
    def test_single_leaf(self):
        """Tree with single leaf."""
        tree = MerkleTree([b"data1"])
        assert tree.get_root() is not None
        assert len(tree) == 1
    
    def test_two_leaves(self):
        """Tree with two leaves."""
        tree = MerkleTree([b"data1", b"data2"])
        assert tree.get_root() is not None
        assert len(tree) == 2
    
    def test_power_of_two_leaves(self):
        """Tree with power of 2 leaves."""
        data = [f"item{i}".encode() for i in range(8)]
        tree = MerkleTree(data)
        assert tree.get_root() is not None
        assert len(tree) == 8
    
    def test_non_power_of_two_leaves(self):
        """Tree handles non-power-of-2 correctly."""
        data = [f"item{i}".encode() for i in range(5)]
        tree = MerkleTree(data)
        assert tree.get_root() is not None
        assert len(tree) == 5
    
    def test_root_determinism(self):
        """Same data produces same root."""
        data = [b"a", b"b", b"c"]
        tree1 = MerkleTree(data)
        tree2 = MerkleTree(data)
        assert tree1.get_root() == tree2.get_root()
    
    def test_different_data_different_root(self):
        """Different data produces different root."""
        tree1 = MerkleTree([b"a", b"b"])
        tree2 = MerkleTree([b"c", b"d"])
        assert tree1.get_root() != tree2.get_root()
    
    def test_proof_generation(self):
        """Proof can be generated for each leaf."""
        data = [f"item{i}".encode() for i in range(4)]
        tree = MerkleTree(data)
        
        for i in range(4):
            proof = tree.get_proof(i)
            assert len(proof) > 0
    
    def test_proof_verification(self):
        """Valid proofs verify correctly."""
        data = [f"item{i}".encode() for i in range(8)]
        tree = MerkleTree(data)
        
        for i in range(8):
            assert tree.verify(i), f"Leaf {i} should verify"
    
    def test_invalid_proof_rejection(self):
        """Wrong hash doesn't verify."""
        data = [b"a", b"b", b"c", b"d"]
        tree = MerkleTree(data)
        
        fake_hash = "f" * 64
        assert not tree.verify(0, data_hash=fake_hash)
    
    def test_proof_invalid_index(self):
        """Invalid index returns empty proof."""
        tree = MerkleTree([b"a", b"b"])
        assert tree.get_proof(-1) == []
        assert tree.get_proof(10) == []


class TestVerifyProof:
    """Tests for standalone proof verification."""
    
    def test_verify_valid_proof(self):
        """Valid proof verifies."""
        data = [b"a", b"b", b"c", b"d"]
        tree = MerkleTree(data)
        
        index = 1
        proof = tree.get_proof(index)
        leaf_hash = tree.leaf_hashes[index]
        root = tree.get_root()
        
        assert verify_proof(root, proof, leaf_hash, index)
    
    def test_verify_wrong_root_fails(self):
        """Wrong root hash fails verification."""
        data = [b"a", b"b", b"c", b"d"]
        tree = MerkleTree(data)
        
        proof = tree.get_proof(0)
        leaf_hash = tree.leaf_hashes[0]
        wrong_root = "0" * 64
        
        assert not verify_proof(wrong_root, proof, leaf_hash, 0)


class TestGalaxyMerkleTree:
    """Tests for GalaxyMerkleTree class."""
    
    def test_galaxy_tree_creation(self):
        """Galaxy tree stores ID and metadata."""
        gradients = [np.random.randn(10) for _ in range(4)]
        tree = GalaxyMerkleTree(gradients, galaxy_id=1)
        
        assert tree.galaxy_id == 1
        assert tree.get_root() is not None
    
    def test_galaxy_tree_with_client_ids(self):
        """Galaxy tree tracks client IDs."""
        gradients = [np.random.randn(10) for _ in range(3)]
        client_ids = [101, 102, 103]
        tree = GalaxyMerkleTree(gradients, galaxy_id=5, client_ids=client_ids)
        
        assert tree.client_ids == client_ids
    
    def test_get_client_proof(self):
        """Can get proof for specific client."""
        gradients = [np.random.randn(10) for _ in range(4)]
        client_ids = [10, 20, 30, 40]
        tree = GalaxyMerkleTree(gradients, galaxy_id=1, client_ids=client_ids)
        
        proof_data = tree.get_client_proof(client_id=20)
        
        assert proof_data is not None
        assert proof_data['client_id'] == 20
        assert proof_data['galaxy_id'] == 1
        assert proof_data['proof'] is not None
    
    def test_get_client_proof_not_found(self):
        """Returns None for unknown client."""
        gradients = [np.random.randn(10) for _ in range(2)]
        tree = GalaxyMerkleTree(gradients, galaxy_id=1, client_ids=[1, 2])
        
        assert tree.get_client_proof(client_id=999) is None


class TestGlobalMerkleTree:
    """Tests for GlobalMerkleTree class."""
    
    def test_global_tree_from_galaxies(self):
        """Global tree aggregates galaxy roots."""
        galaxies = []
        for gid in range(3):
            grads = [np.random.randn(10) for _ in range(4)]
            galaxies.append(GalaxyMerkleTree(grads, galaxy_id=gid))
        
        global_tree = GlobalMerkleTree(galaxies)
        
        assert global_tree.get_root() is not None
        assert len(global_tree) == 3
    
    def test_get_galaxy_proof(self):
        """Can get proof for galaxy inclusion."""
        galaxies = []
        for gid in [10, 20, 30]:
            grads = [np.random.randn(10) for _ in range(2)]
            galaxies.append(GalaxyMerkleTree(grads, galaxy_id=gid))
        
        global_tree = GlobalMerkleTree(galaxies)
        proof_data = global_tree.get_galaxy_proof(galaxy_id=20)
        
        assert proof_data is not None
        assert proof_data['galaxy_id'] == 20
    
    def test_verify_client_in_system(self):
        """Full verification from client to global root."""
        # Create galaxies
        galaxies = []
        for gid in range(2):
            grads = [np.random.randn(10) for _ in range(4)]
            cids = list(range(gid * 4, gid * 4 + 4))
            galaxies.append(GalaxyMerkleTree(grads, galaxy_id=gid, client_ids=cids))
        
        global_tree = GlobalMerkleTree(galaxies)
        
        # Verify client 2 in galaxy 0
        client_proof = galaxies[0].get_client_proof(client_id=2)
        assert global_tree.verify_client_in_system(
            client_id=2, galaxy_id=0, client_proof=client_proof
        )


class TestGradientCommitment:
    """Tests for GradientCommitment class."""
    
    def test_commitment_creation(self):
        """Create and commit gradients."""
        grads = [np.random.randn(10), np.random.randn(5)]
        commitment = GradientCommitment(
            gradients=grads,
            client_id=42,
            round_number=5
        )
        
        c = commitment.commit()
        assert c is not None
        assert len(c) == 64
    
    def test_commitment_verification_success(self):
        """Valid reveal verifies."""
        grads = [np.array([1.0, 2.0, 3.0])]
        commitment = GradientCommitment(
            gradients=grads,
            client_id=1,
            round_number=1
        )
        commitment.commit()
        
        # Reveal same gradients
        assert commitment.verify(grads)
    
    def test_commitment_verification_failure(self):
        """Modified reveal fails verification."""
        grads = [np.array([1.0, 2.0, 3.0])]
        commitment = GradientCommitment(
            gradients=grads,
            client_id=1,
            round_number=1
        )
        commitment.commit()
        
        # Try to reveal different gradients
        fake_grads = [np.array([9.0, 9.0, 9.0])]
        assert not commitment.verify(fake_grads)
    
    def test_get_metadata(self):
        """Metadata contains expected fields."""
        commitment = GradientCommitment(
            gradients=[np.array([1.0])],
            client_id=10,
            round_number=3
        )
        
        meta = commitment.get_metadata()
        assert meta['client_id'] == 10
        assert meta['round_number'] == 3
        assert 'timestamp' in meta
        assert 'nonce' in meta


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
