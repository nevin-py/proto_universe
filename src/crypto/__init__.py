"""Cryptographic utilities for ProtoGalaxy ZK-FL

This module provides cryptographic primitives for:
- Gradient commitment and verification (Merkle trees)
- Hash functions and secure random generation
- Proof verification utilities
"""

from .merkle import (
    MerkleTree,
    GalaxyMerkleTree,
    GlobalMerkleTree,
    GradientCommitment,
    compute_hash,
    verify_proof,
    combine_hashes,
    serialize_gradient,
)

from .utils import (
    hash_data,
    compute_hmac,
    secure_random_bytes,
    constant_time_compare,
)

__all__ = [
    # Merkle tree classes
    'MerkleTree',
    'GalaxyMerkleTree', 
    'GlobalMerkleTree',
    'GradientCommitment',
    # Hash functions
    'compute_hash',
    'verify_proof',
    'combine_hashes',
    'serialize_gradient',
    'hash_data',
    'compute_hmac',
    # Utilities
    'secure_random_bytes',
    'constant_time_compare',
]
