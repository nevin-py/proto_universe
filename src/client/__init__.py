"""Client package for federated learning.

Provides:
- Client: Main client class orchestrating training and communication
- Trainer: Local model training
- CommitmentGenerator: Cryptographic commitment generation
- ProofVerifier: Merkle proof verification
"""

from src.client.client import Client
from src.client.trainer import Trainer
from src.client.commitment import CommitmentGenerator
from src.client.verifier import ProofVerifier
from src.client.adaptive_attacker import AdaptiveAttacker
from src.client.sybil_coordinator import SybilCoordinator

__all__ = [
    'Client',
    'Trainer',
    'CommitmentGenerator',
    'ProofVerifier',
    'AdaptiveAttacker',
    'SybilCoordinator',
]
