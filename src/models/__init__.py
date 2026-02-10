"""Models package for machine learning models.

Provides:
- MNIST CNN model
- CIFAR-10 CNN model
- Simple MLP
- Model registry for easy model creation
"""

from src.models.mnist import MNISTCnn, create_mnist_model
from src.models.registry import (
    get_model,
    list_models,
    create_model,
    count_parameters,
    get_model_info,
    SimpleMLP,
    CIFAR10CNN
)

__all__ = [
    'MNISTCnn',
    'create_mnist_model',
    'SimpleMLP',
    'CIFAR10CNN',
    'get_model',
    'list_models',
    'create_model',
    'count_parameters',
    'get_model_info'
]
