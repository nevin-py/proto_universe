"""Model registry for federated learning experiments.

Provides factory functions and registry for ML models.
"""

from typing import Dict, Type, Optional, Callable
import torch.nn as nn

from src.models.mnist import MNISTCnn


# Model registry
_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class.
    
    Args:
        name: Name to register the model under
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    """Get a model by name from the registry.
    
    Args:
        name: Model name
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model name not found
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found. Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list:
    """List all registered models.
    
    Returns:
        List of model names
    """
    return list(_MODEL_REGISTRY.keys())


# Register built-in models
_MODEL_REGISTRY['mnist_cnn'] = MNISTCnn
_MODEL_REGISTRY['mnist'] = MNISTCnn


# Simple MLP for basic experiments
class SimpleMLP(nn.Module):
    """Simple MLP for basic FL experiments"""
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 10
    ):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


_MODEL_REGISTRY['simple_mlp'] = SimpleMLP
_MODEL_REGISTRY['mlp'] = SimpleMLP


# CIFAR-10 CNN
class CIFAR10CNN(nn.Module):
    """CNN model for CIFAR-10 classification"""
    
    def __init__(self, num_classes: int = 10):
        super(CIFAR10CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


_MODEL_REGISTRY['cifar10_cnn'] = CIFAR10CNN
_MODEL_REGISTRY['cifar10'] = CIFAR10CNN

# ResNet18 for CIFAR-10 (imported lazily to avoid circular deps)
try:
    from src.models.resnet import CIFAR10ResNet18
    _MODEL_REGISTRY['resnet18'] = CIFAR10ResNet18
    _MODEL_REGISTRY['cifar10_resnet18'] = CIFAR10ResNet18
except ImportError:
    pass  # torchvision may not be installed


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Create a model by type.
    
    Convenience function that wraps get_model.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    return get_model(model_type, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> Dict:
    """Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model info
    """
    return {
        'type': type(model).__name__,
        'num_parameters': count_parameters(model),
        'num_layers': len(list(model.modules())),
        'trainable': any(p.requires_grad for p in model.parameters())
    }
