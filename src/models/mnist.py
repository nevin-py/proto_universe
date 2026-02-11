"""MNIST models for federated learning experiments"""

import torch
import torch.nn as nn


class MNISTLinearRegression(nn.Module):
    """Simple linear regression model for MNIST classification
    
    Flattens 28x28 images to 784 dimensions and applies a single linear layer.
    """
    
    def __init__(self, num_classes: int = 10):
        """Initialize MNIST linear regression model
        
        Args:
            num_classes: Number of output classes (default: 10 for MNIST)
        """
        super(MNISTLinearRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, num_classes)
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.flatten(x)
        x = self.linear(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST (2 hidden layers)"""
    
    def __init__(self, num_classes: int = 10):
        """Initialize simple MLP
        
        Args:
            num_classes: Number of output classes
        """
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """Forward pass"""
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MNISTCnn(nn.Module):
    """Simple CNN model for MNIST classification"""
    
    def __init__(self, num_classes: int = 10):
        """Initialize MNIST CNN"""
        super(MNISTCnn, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """Forward pass"""
        # Conv block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_mnist_model(model_type: str = 'linear', num_classes: int = 10) -> nn.Module:
    """Factory function to create MNIST models
    
    Args:
        model_type: Type of model ('linear', 'mlp', or 'cnn')
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    if model_type == 'linear':
        return MNISTLinearRegression(num_classes=num_classes)
    elif model_type == 'mlp':
        return SimpleMLP(num_classes=num_classes)
    elif model_type == 'cnn':
        return MNISTCnn(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'linear', 'mlp', or 'cnn'.")
