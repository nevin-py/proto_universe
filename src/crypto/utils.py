"""Cryptographic utility functions"""

import hashlib
import hmac


def hash_data(data: bytes) -> str:
    """Hash data using SHA-256"""
    return hashlib.sha256(data).hexdigest()


def compute_hmac(data: bytes, key: bytes) -> str:
    """Compute HMAC-SHA256"""
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def secure_random_bytes(length: int) -> bytes:
    """Generate cryptographically secure random bytes"""
    import secrets
    return secrets.token_bytes(length)


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time"""
    return hmac.compare_digest(a, b)
