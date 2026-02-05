"""Merkle proof verification for gradient commitments"""

from src.crypto.merkle import verify_proof


class ProofVerifier:
    """Verifies Merkle proofs for gradient commitments"""
    
    def __init__(self):
        """Initialize proof verifier"""
        self.verified_proofs = []
    
    def verify_proof(self, root: str, proof: list, leaf_hash: str) -> bool:
        """Verify a Merkle proof"""
        is_valid = verify_proof(root, proof, leaf_hash)
        if is_valid:
            self.verified_proofs.append((root, leaf_hash))
        return is_valid
    
    def batch_verify(self, proofs: list) -> dict:
        """Verify multiple proofs in batch"""
        results = {}
        for proof_data in proofs:
            root, proof, leaf_hash = proof_data
            results[leaf_hash] = self.verify_proof(root, proof, leaf_hash)
        return results
    
    def get_verification_history(self):
        """Get history of verified proofs"""
        return self.verified_proofs
    
    def clear_history(self):
        """Clear verification history"""
        self.verified_proofs = []
