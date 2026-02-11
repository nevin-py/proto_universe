"""Adapter layer for Merkle trees in the ProtoGalaxy pipeline

This module provides simplified interfaces for the pipeline to interact with
Merkle trees without needing to know the implementation details.
"""

from typing import Dict, List, Optional
from src.crypto.merkle import MerkleTree, GalaxyMerkleTree, GlobalMerkleTree, compute_hash


class GalaxyMerkleTreeAdapter:
    """Simplified adapter for galaxy-level Merkle trees"""
    
    def __init__(self, galaxy_id: int):
        self.galaxy_id = galaxy_id
        self.tree: Optional[GalaxyMerkleTree] = None
        self.commitments: Dict[int, str] = {}  # client_id -> commitment_hash
        self.client_ids: List[int] = []
    
    def build_from_commitments(
        self,
        commitment_list: List[Dict],  # [{'client_id': int, 'commitment_hash': str}]
        round_number: int = 0
    ) -> str:
        """Build Merkle tree from commitment hashes"""
        self.commitments = {}
        self.client_ids = []
        hashes = []
        
        for item in commitment_list:
            client_id = item['client_id']
            commit_hash = item['commitment_hash']
            self.commitments[client_id] = commit_hash
            self.client_ids.append(client_id)
            hashes.append(commit_hash)  # Use hash directly
        
        # Build tree from hashes
        self.tree = GalaxyMerkleTree(
            gradients=hashes,  # Pass hashes as "gradients"
            galaxy_id=self.galaxy_id,
            client_ids=self.client_ids,
            round_number=round_number
        )
        
        root = self.tree.get_root()
        return root if root is not None else ""
    
    def get_root(self) -> Optional[str]:
        """Get the Merkle root"""
        if self.tree:
            return self.tree.get_root()
        return None
    
    def get_proof(self, client_id: int) -> Optional[List]:
        """Get Merkle proof for a client"""
        if not self.tree or client_id not in self.client_ids:
            return None
        
        proof_data = self.tree.get_client_proof(client_id)
        return proof_data['proof'] if proof_data else None
    
    def verify_proof(self, client_id: int, commitment_hash: str, proof: List) -> bool:
        """Verify a Merkle proof"""
        root = self.get_root()
        if not self.tree or not root:
            return False
        
        # Find index
        if client_id not in self.client_ids:
            return False
        
        index = self.client_ids.index(client_id)
        
        # Use the verify_proof function from merkle module
        from src.crypto.merkle import verify_proof as verify_proof_fn
        return verify_proof_fn(root, proof, commitment_hash, index)


class GlobalMerkleTreeAdapter:
    """Simplified adapter for global-level Merkle trees"""
    
    def __init__(self):
        self.tree: Optional[MerkleTree] = None  # Use base MerkleTree
        self.galaxy_roots: Dict[int, str] = {}  # galaxy_id -> root_hash
        self.galaxy_ids: List[int] = []
    
    def build_from_galaxy_roots(
        self,
        galaxy_root_list: List[Dict],  # [{'galaxy_id': int, 'merkle_root': str}]
        round_number: int = 0
    ) -> str:
        """Build global Merkle tree from galaxy roots"""
        self.galaxy_roots = {}
        self.galaxy_ids = []
        roots = []
        
        for item in galaxy_root_list:
            galaxy_id = item['galaxy_id']
            root = item['merkle_root']
            self.galaxy_roots[galaxy_id] = root
            self.galaxy_ids.append(galaxy_id)
            roots.append(root)
        
        # Build metadata
        metadata_list = []
        for galaxy_id in self.galaxy_ids:
            metadata_list.append({
                'galaxy_id': galaxy_id,
                'round_number': round_number
            })
        
        # Build tree from galaxy roots
        self.tree = MerkleTree(data=roots, metadata_list=metadata_list)
        
        root = self.tree.get_root()
        return root if root is not None else ""
    
    def get_root(self) -> Optional[str]:
        """Get the global Merkle root"""
        if self.tree:
            return self.tree.get_root()
        return None
    
    def get_galaxy_proof(self, galaxy_id: int) -> Optional[List]:
        """Get Merkle proof for a galaxy"""
        if not self.tree or galaxy_id not in self.galaxy_ids:
            return None
        
        index = self.galaxy_ids.index(galaxy_id)
        return self.tree.get_proof(index)
    
    def verify_galaxy_proof(self, galaxy_id: int, merkle_root: str, proof: List) -> bool:
        """Verify a galaxy's Merkle proof"""
        root = self.get_root()
        if not self.tree or not root:
            return False
        
        if galaxy_id not in self.galaxy_ids:
            return False
        
        index = self.galaxy_ids.index(galaxy_id)
        
        from src.crypto.merkle import verify_proof as verify_proof_fn
        return verify_proof_fn(root, proof, merkle_root, index)
