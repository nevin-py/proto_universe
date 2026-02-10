"""Cryptographic utilities for ProtoGalaxy ZK-FL

This module provides cryptographic primitives for:
- Gradient commitment and verification (Merkle trees)
- Hash functions and secure random generation (PROTO-1003)
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
    hash_sha256,
    compute_hmac,
    verify_hmac,
    secure_random_bytes,
    constant_time_compare,
    generate_nonce,
    generate_nonce_hex,
    generate_timestamp,
    generate_timestamp_unix,
    generate_timestamp_ms,
    hash_combine,
    generate_commitment_id,
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
    'hash_sha256',
    'compute_hmac',
    'verify_hmac',
    'hash_combine',
    # Nonce and timestamp generation (PROTO-1003)
    'generate_nonce',
    'generate_nonce_hex',
    'generate_timestamp',
    'generate_timestamp_unix',
    'generate_timestamp_ms',
    'generate_commitment_id',
    # Utilities
    'secure_random_bytes',
    'constant_time_compare',
]
