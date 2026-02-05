"""Gradient commitment generation for privacy-preserving verification"""

from src.crypto.merkle import GradientCommitment, MerkleTree


class CommitmentGenerator:
    """Generates cryptographic commitments for model gradients"""
    
    def __init__(self, client_id: int):
        """Initialize commitment generator"""
        self.client_id = client_id
        self.commitments = []
    
    def generate_commitment(self, gradients):
        """Generate commitment for gradient update"""
        commitment = GradientCommitment(gradients)
        commitment.commit()
        self.commitments.append(commitment)
        return commitment
    
    def get_commitment_proof(self, index: int):
        """Get proof for a specific commitment"""
        if index < len(self.commitments):
            return self.commitments[index].merkle_tree.get_proof(index)
        return None
    
    def verify_commitment(self, commitment, proof):
        """Verify a commitment with its proof"""
        return commitment.verify(proof, commitment.get_commitment())
    
    def get_all_commitments(self):
        """Get all generated commitments"""
        return [c.get_commitment() for c in self.commitments]
