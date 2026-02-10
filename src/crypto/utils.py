"""Cryptographic utility functions (PROTO-1003)

Provides:
- SHA-256 hash function wrapper
- Nonce generation (random bytes)
- Timestamp generation
- HMAC computation
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime
from typing import Union


def hash_sha256(data: Union[bytes, str]) -> str:
    """
    SHA-256 hash function wrapper.
    
    Args:
        data: Data to hash (bytes or string)
        
    Returns:
        Hex string of SHA-256 hash
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def hash_data(data: bytes) -> str:
    """
    Hash data using SHA-256.
    
    Args:
        data: Bytes to hash
        
    Returns:
        Hex string of hash
    """
    return hashlib.sha256(data).hexdigest()


def generate_nonce(length: int = 32) -> bytes:
    """
    Generate cryptographically secure random bytes (nonce).
    
    Args:
        length: Number of random bytes to generate
        
    Returns:
        Random bytes
    """
    return secrets.token_bytes(length)


def generate_nonce_hex(length: int = 32) -> str:
    """
    Generate cryptographically secure random hex string.
    
    Args:
        length: Number of random bytes (output will be 2x length in hex chars)
        
    Returns:
        Hex string of random bytes
    """
    return secrets.token_hex(length)


def generate_timestamp() -> str:
    """
    Generate ISO format timestamp.
    
    Returns:
        ISO 8601 timestamp string
    """
    return datetime.now().isoformat()


def generate_timestamp_unix() -> float:
    """
    Generate Unix timestamp.
    
    Returns:
        Unix timestamp as float
    """
    return time.time()


def generate_timestamp_ms() -> int:
    """
    Generate Unix timestamp in milliseconds.
    
    Returns:
        Unix timestamp in milliseconds
    """
    return int(time.time() * 1000)


def compute_hmac(data: bytes, key: bytes) -> str:
    """
    Compute HMAC-SHA256.
    
    Args:
        data: Data to authenticate
        key: Secret key
        
    Returns:
        Hex string of HMAC
    """
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def verify_hmac(data: bytes, key: bytes, expected_hmac: str) -> bool:
    """
    Verify HMAC in constant time.
    
    Args:
        data: Data that was authenticated
        key: Secret key
        expected_hmac: Expected HMAC value
        
    Returns:
        True if HMAC matches
    """
    computed = compute_hmac(data, key)
    return constant_time_compare(computed, expected_hmac)


def secure_random_bytes(length: int) -> bytes:
    """
    Generate cryptographically secure random bytes.
    
    Args:
        length: Number of bytes to generate
        
    Returns:
        Random bytes
    """
    return secrets.token_bytes(length)


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time (prevents timing attacks).
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal
    """
    return hmac.compare_digest(a, b)


def hash_combine(*hashes: str) -> str:
    """
    Combine multiple hashes into a single hash.
    
    Args:
        *hashes: Variable number of hash strings
        
    Returns:
        Combined hash
    """
    combined = ''.join(hashes)
    return hash_sha256(combined)


def generate_commitment_id(client_id: str, round_number: int) -> str:
    """
    Generate unique commitment ID for a client's round submission.
    
    Args:
        client_id: Client identifier
        round_number: FL round number
        
    Returns:
        Unique commitment ID
    """
    data = f"{client_id}:{round_number}:{generate_timestamp()}"
    return hash_sha256(data)[:16]  # First 16 chars of hash
