"""Data loading and partitioning for federated learning.

Provides utilities for:
- Loading datasets (MNIST, CIFAR-10, etc.)
- Partitioning data across clients (IID and non-IID)
- Creating data loaders for local training
"""

from src.data.datasets import load_mnist, load_cifar10
from src.data.partition import (
    DataPartitioner,
    IIDPartitioner,
    NonIIDPartitioner,
    DirichletPartitioner
)
from src.data.loader import create_client_loaders
from src.data.backdoor import BackdoorDataset

__all__ = [
    'load_mnist',
    'load_cifar10',
    'DataPartitioner',
    'IIDPartitioner',
    'NonIIDPartitioner',
    'DirichletPartitioner',
    'create_client_loaders',
    'BackdoorDataset',
]
