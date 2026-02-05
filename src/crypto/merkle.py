"""Merkle tree implementations for gradient commitments in Protogalaxy ZK-FL

This module provides Layer 1 (Cryptographic Integrity) verification for the
hierarchical federated learning architecture. It implements:
- Gradient hashing with metadata binding
- Efficient Merkle tree construction and proof generation
- Galaxy-level and Global-level tree hierarchies
- Commitment verification for tamper detection
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch


def serialize_gradient(gradient: Union[np.ndarray, torch.Tensor, list]) -> bytes:
    """Serialize gradient data to bytes for hashing.
    
    Args:
        gradient: Gradient as numpy array, torch tensor, or list
        
    Returns:
        Bytes representation of the gradient
    """
    if isinstance(gradient, torch.Tensor):
        arr = gradient.detach().cpu().numpy()
    elif isinstance(gradient, list):
        arr = np.array(gradient)
    else:
        arr = np.asarray(gradient)
    
    # Use tobytes for consistent serialization
    return arr.astype(np.float32).tobytes()


def compute_hash(data: Union[bytes, str, np.ndarray, torch.Tensor, list],
                 metadata: Optional[dict] = None) -> str:
    """Compute SHA-256 hash of data with optional metadata.
    
    Args:
        data: Data to hash (bytes, string, numpy array, or torch tensor)
        metadata: Optional dict with client_id, round_number, timestamp, nonce
        
    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.sha256()
    
    # Handle different data types
    if isinstance(data, bytes):
        hasher.update(data)
    elif isinstance(data, str):
        hasher.update(data.encode('utf-8'))
    elif isinstance(data, (np.ndarray, torch.Tensor, list)):
        hasher.update(serialize_gradient(data))
    else:
        hasher.update(str(data).encode('utf-8'))
    
    # Include metadata if provided
    if metadata:
        meta_bytes = json.dumps(metadata, sort_keys=True).encode('utf-8')
        hasher.update(meta_bytes)
    
    return hasher.hexdigest()


def combine_hashes(left: str, right: str) -> str:
    """Combine two hashes into a parent hash.
    
    Args:
        left: Left child hash
        right: Right child hash
        
    Returns:
        Combined parent hash
    """
    combined = (left + right).encode('utf-8')
    return hashlib.sha256(combined).hexdigest()


def verify_proof(root: str, proof: list, leaf_hash: str, leaf_index: int) -> bool:
    """Verify a Merkle inclusion proof.
    
    Args:
        root: Expected root hash
        proof: List of (sibling_hash, is_left) tuples
        leaf_hash: Hash of the leaf to verify
        leaf_index: Index of the leaf in the tree
        
    Returns:
        True if the proof is valid, False otherwise
    """
    current_hash = leaf_hash
    
    for sibling_hash, is_left in proof:
        if is_left:
            current_hash = combine_hashes(sibling_hash, current_hash)
        else:
            current_hash = combine_hashes(current_hash, sibling_hash)
    
    return current_hash == root


@dataclass
class MerkleNode:
    """Node in the Merkle tree."""
    hash: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False
    data_index: int = -1


class MerkleTree:
    """Base Merkle tree implementation for gradient verification.
    
    Provides O(log n) proof generation and verification for gradient
    integrity in federated learning.
    """
    
    def __init__(self, data: list, metadata_list: Optional[list] = None):
        """Initialize and build Merkle tree from data.
        
        Args:
            data: List of data items (gradients or hashes)
            metadata_list: Optional list of metadata dicts per item
        """
        self.data = data
        self.metadata_list = metadata_list or [None] * len(data)
        self.leaf_hashes: list[str] = []
        self.tree_levels: list[list[str]] = []
        self.root_node: Optional[MerkleNode] = None
        
        if data:
            self.build()
    
    def build(self):
        """Build the Merkle tree from leaf data."""
        if not self.data:
            return
        
        # Compute leaf hashes
        self.leaf_hashes = []
        for i, item in enumerate(self.data):
            metadata = self.metadata_list[i] if i < len(self.metadata_list) else None
            
            # If item is already a hash string, use it directly
            if isinstance(item, str) and len(item) == 64:
                leaf_hash = item
            else:
                leaf_hash = compute_hash(item, metadata)
            
            self.leaf_hashes.append(leaf_hash)
        
        # Build tree levels bottom-up
        self.tree_levels = [self.leaf_hashes.copy()]
        current_level = self.leaf_hashes.copy()
        
        while len(current_level) > 1:
            next_level = []
            
            # Pad with duplicate if odd number
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])
            
            for i in range(0, len(current_level), 2):
                parent_hash = combine_hashes(current_level[i], current_level[i + 1])
                next_level.append(parent_hash)
            
            self.tree_levels.append(next_level)
            current_level = next_level
    
    def get_root(self) -> Optional[str]:
        """Get the root hash of the tree.
        
        Returns:
            Root hash or None if tree is empty
        """
        if not self.tree_levels:
            return None
        return self.tree_levels[-1][0] if self.tree_levels[-1] else None
    
    def get_proof(self, index: int) -> list:
        """Generate Merkle proof for a leaf at given index.
        
        Args:
            index: Index of the leaf (0-indexed)
            
        Returns:
            List of (sibling_hash, is_left) tuples for proof verification
        """
        if not self.leaf_hashes or index < 0 or index >= len(self.leaf_hashes):
            return []
        
        proof = []
        current_index = index
        
        for level in self.tree_levels[:-1]:  # Exclude root level
            # Handle padding
            level_size = len(level)
            if level_size % 2 == 1:
                level = level + [level[-1]]
            
            # Determine sibling
            if current_index % 2 == 0:
                # Current is left child, sibling is right
                sibling_index = current_index + 1
                is_left = False
            else:
                # Current is right child, sibling is left
                sibling_index = current_index - 1
                is_left = True
            
            if sibling_index < len(level):
                proof.append((level[sibling_index], is_left))
            
            # Move to parent index
            current_index = current_index // 2
        
        return proof
    
    def verify(self, index: int, data_hash: Optional[str] = None) -> bool:
        """Verify that a leaf is correctly included in the tree.
        
        Args:
            index: Index of the leaf to verify
            data_hash: Optional hash to verify (uses stored hash if None)
            
        Returns:
            True if verification passes
        """
        if not self.leaf_hashes or index < 0 or index >= len(self.leaf_hashes):
            return False
        
        leaf_hash = data_hash if data_hash else self.leaf_hashes[index]
        proof = self.get_proof(index)
        root = self.get_root()
        
        return verify_proof(root, proof, leaf_hash, index)
    
    def __len__(self) -> int:
        """Return number of leaves in the tree."""
        return len(self.leaf_hashes)


class GalaxyMerkleTree(MerkleTree):
    """Galaxy-level Merkle tree for client gradient verification.
    
    Extends base MerkleTree with galaxy-specific metadata and
    client tracking capabilities.
    """
    
    def __init__(self, gradients: list, galaxy_id: int, 
                 client_ids: Optional[list] = None,
                 round_number: int = 0):
        """Initialize galaxy Merkle tree.
        
        Args:
            gradients: List of client gradient updates
            galaxy_id: Unique identifier for this galaxy
            client_ids: Optional list of client IDs
            round_number: Current training round number
        """
        self.galaxy_id = galaxy_id
        self.client_ids = client_ids or list(range(len(gradients)))
        self.round_number = round_number
        self.timestamp = time.time()
        
        # Build metadata for each gradient
        metadata_list = []
        for i, cid in enumerate(self.client_ids):
            metadata_list.append({
                'client_id': cid,
                'galaxy_id': galaxy_id,
                'round_number': round_number,
                'timestamp': self.timestamp,
                'index': i
            })
        
        super().__init__(gradients, metadata_list)
    
    def get_client_proof(self, client_id: int) -> Optional[dict]:
        """Get proof for a specific client's submission.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Dict with proof data or None if client not found
        """
        if client_id not in self.client_ids:
            return None
        
        index = self.client_ids.index(client_id)
        return {
            'client_id': client_id,
            'galaxy_id': self.galaxy_id,
            'round_number': self.round_number,
            'leaf_hash': self.leaf_hashes[index],
            'proof': self.get_proof(index),
            'galaxy_root': self.get_root(),
            'leaf_index': index
        }


class GlobalMerkleTree(MerkleTree):
    """Global-level Merkle tree for galaxy root aggregation.
    
    Aggregates galaxy roots into a single global verification tree,
    enabling hierarchical verification across the entire system.
    """
    
    def __init__(self, galaxy_trees: list, round_number: int = 0):
        """Initialize global Merkle tree from galaxy trees.
        
        Args:
            galaxy_trees: List of GalaxyMerkleTree instances
            round_number: Current training round
        """
        self.galaxy_trees = galaxy_trees
        self.round_number = round_number
        self.timestamp = time.time()
        
        # Extract galaxy roots as leaf data
        galaxy_roots = [tree.get_root() for tree in galaxy_trees]
        
        # Build metadata for each galaxy
        metadata_list = []
        for tree in galaxy_trees:
            metadata_list.append({
                'galaxy_id': tree.galaxy_id,
                'round_number': round_number,
                'timestamp': self.timestamp,
                'num_clients': len(tree)
            })
        
        super().__init__(galaxy_roots, metadata_list)
    
    def get_galaxy_proof(self, galaxy_id: int) -> Optional[dict]:
        """Get proof for a specific galaxy's inclusion.
        
        Args:
            galaxy_id: ID of the galaxy
            
        Returns:
            Dict with proof data or None if galaxy not found
        """
        for i, tree in enumerate(self.galaxy_trees):
            if tree.galaxy_id == galaxy_id:
                return {
                    'galaxy_id': galaxy_id,
                    'galaxy_root': tree.get_root(),
                    'proof': self.get_proof(i),
                    'global_root': self.get_root(),
                    'galaxy_index': i
                }
        return None
    
    def verify_client_in_system(self, client_id: int, galaxy_id: int,
                                 client_proof: dict) -> bool:
        """Verify a client's submission is included in global tree.
        
        Args:
            client_id: Client ID to verify
            galaxy_id: Galaxy containing the client
            client_proof: Proof from GalaxyMerkleTree.get_client_proof()
            
        Returns:
            True if client is verifiably included in the system
        """
        # First verify client in galaxy
        for tree in self.galaxy_trees:
            if tree.galaxy_id == galaxy_id:
                if not tree.verify(client_proof['leaf_index'], 
                                   client_proof['leaf_hash']):
                    return False
                break
        else:
            return False
        
        # Then verify galaxy in global tree
        galaxy_proof = self.get_galaxy_proof(galaxy_id)
        if not galaxy_proof:
            return False
        
        return verify_proof(
            self.get_root(),
            galaxy_proof['proof'],
            galaxy_proof['galaxy_root'],
            galaxy_proof['galaxy_index']
        )


@dataclass
class GradientCommitment:
    """Manages gradient commitments for the commit-reveal protocol.
    
    Provides binding commitments that prevent adaptive attacks by
    requiring clients to commit before revelation.
    """
    
    gradients: list
    client_id: int
    round_number: int
    nonce: str = field(default_factory=lambda: hashlib.sha256(
        str(time.time()).encode()).hexdigest()[:16])
    timestamp: float = field(default_factory=time.time)
    commitment: Optional[str] = None
    
    def commit(self) -> str:
        """Generate binding commitment for gradients.
        
        Returns:
            Commitment hash
        """
        # Serialize all gradients
        gradient_hashes = []
        for grad in self.gradients:
            gradient_hashes.append(compute_hash(grad))
        
        # Create commitment with metadata
        metadata = {
            'client_id': self.client_id,
            'round_number': self.round_number,
            'timestamp': self.timestamp,
            'nonce': self.nonce
        }
        
        combined = ''.join(gradient_hashes)
        self.commitment = compute_hash(combined, metadata)
        return self.commitment
    
    def get_commitment(self) -> Optional[str]:
        """Get the commitment value.
        
        Returns:
            Commitment hash or None if not yet committed
        """
        return self.commitment
    
    def get_metadata(self) -> dict:
        """Get commitment metadata for verification.
        
        Returns:
            Dict with client_id, round_number, timestamp, nonce
        """
        return {
            'client_id': self.client_id,
            'round_number': self.round_number,
            'timestamp': self.timestamp,
            'nonce': self.nonce
        }
    
    def verify(self, revealed_gradients: list) -> bool:
        """Verify revealed gradients match commitment.
        
        Args:
            revealed_gradients: Gradients revealed by client
            
        Returns:
            True if gradients match commitment
        """
        if not self.commitment:
            return False
        
        # Recompute commitment from revealed gradients
        gradient_hashes = []
        for grad in revealed_gradients:
            gradient_hashes.append(compute_hash(grad))
        
        metadata = self.get_metadata()
        combined = ''.join(gradient_hashes)
        revealed_commitment = compute_hash(combined, metadata)
        
        return revealed_commitment == self.commitment
