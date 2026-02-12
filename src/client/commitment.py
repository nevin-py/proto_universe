"""Gradient commitment generation for privacy-preserving verification"""

from typing import List, Optional
from src.crypto.merkle import GradientCommitment, MerkleTree


class CommitmentGenerator:
    """Generates cryptographic commitments for model gradients"""
    
    def __init__(self, client_id: int):
        """Initialize commitment generator"""
        self.client_id = client_id
        self.commitments: List[GradientCommitment] = []
        self._merkle_tree: Optional[MerkleTree] = None
    
    def generate_commitment(self, gradients, round_number: int = 0):
        """Generate commitment for gradient update.
        
        Args:
            gradients: List of gradient arrays/tensors
            round_number: Current FL round number
            
        Returns:
            GradientCommitment with binding commitment hash
        """
        commitment = GradientCommitment(
            gradients=gradients,
            client_id=self.client_id,
            round_number=round_number
        )
        commitment.commit()
        self.commitments.append(commitment)
        
        # Rebuild Merkle tree over all commitment hashes so that
        # get_commitment_proof() can return a valid inclusion proof.
        commitment_hashes = [c.get_commitment() for c in self.commitments]
        self._merkle_tree = MerkleTree(data=commitment_hashes)
        
        return commitment
    
    def get_commitment_proof(self, index: int):
        """Get Merkle inclusion proof for a specific commitment.
        
        Args:
            index: Index of the commitment in the history list
            
        Returns:
            List of (sibling_hash, is_left) tuples, or None
        """
        if self._merkle_tree is not None and 0 <= index < len(self.commitments):
            return self._merkle_tree.get_proof(index)
        return None
    
    def verify_commitment(self, commitment: GradientCommitment, revealed_gradients: list) -> bool:
        """Verify revealed gradients match a commitment.
        
        Args:
            commitment: The original GradientCommitment object
            revealed_gradients: Gradients revealed by the client
            
        Returns:
            True if revealed gradients match the commitment
        """
        return commitment.verify(revealed_gradients)
    
    def get_all_commitments(self):
        """Get all generated commitment hashes"""
        return [c.get_commitment() for c in self.commitments]
