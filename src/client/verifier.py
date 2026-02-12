"""Merkle proof verification for gradient commitments"""

from src.crypto.merkle import verify_proof as _verify_proof


class ProofVerifier:
    """Verifies Merkle proofs for gradient commitments"""
    
    def __init__(self):
        """Initialize proof verifier"""
        self.verified_proofs = []
    
    def verify_proof(self, root: str, proof: list, leaf_hash: str, leaf_index: int = 0) -> bool:
        """Verify a Merkle proof.
        
        Args:
            root: Expected Merkle root hash
            proof: List of (sibling_hash, is_left) tuples
            leaf_hash: Hash of the leaf to verify
            leaf_index: Index of the leaf in the tree
            
        Returns:
            True if proof is valid
        """
        is_valid = _verify_proof(root, proof, leaf_hash, leaf_index)
        if is_valid:
            self.verified_proofs.append((root, leaf_hash))
        return is_valid
    
    def batch_verify(self, proofs: list) -> dict:
        """Verify multiple proofs in batch.
        
        Args:
            proofs: List of (root, proof, leaf_hash, leaf_index) tuples
            
        Returns:
            Dict mapping leaf_hash -> verification result
        """
        results = {}
        for proof_data in proofs:
            root, proof, leaf_hash, leaf_index = proof_data
            results[leaf_hash] = self.verify_proof(root, proof, leaf_hash, leaf_index)
        return results
    
    def get_verification_history(self):
        """Get history of verified proofs"""
        return self.verified_proofs
    
    def clear_history(self):
        """Clear verification history"""
        self.verified_proofs = []
