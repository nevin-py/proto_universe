"""MNIST CNN model for federated learning experiments"""

import torch
import torch.nn as nn


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


def create_mnist_model(num_classes: int = 10) -> nn.Module:
    """Factory function to create MNIST CNN model"""
    return MNISTCnn(num_classes=num_classes)
