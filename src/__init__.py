"""ProtoGalaxy: Hierarchical Federated Learning with Byzantine Defense.

This package provides a complete implementation of the ProtoGalaxy
federated learning framework with:

- Hierarchical aggregation (clients -> galaxies -> global)
- Multi-layer Byzantine defense
- Merkle tree-based verification
- Reputation-based trust management
- Comprehensive simulation framework

Main modules:
- client: Client-side training and communication
- aggregators: Galaxy and global aggregation
- defense: Byzantine defense layers
- communication: Message passing infrastructure
- orchestration: FL round coordination
- simulation: Complete FL simulation
- data: Dataset loading and partitioning
- models: ML model definitions
- logging: Structured FL logging
- storage: Model and metrics persistence
"""

__version__ = "0.1.0"
__author__ = "ProtoGalaxy Team"
