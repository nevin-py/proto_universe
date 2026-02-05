"""Local model training for federated learning clients"""

import torch
import torch.nn as nn


class Trainer:
    """Handles local model training on client devices"""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.01):
        """Initialize trainer with model and hyperparameters"""
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, train_loader, num_epochs: int):
        """Train the model locally"""
        for epoch in range(num_epochs):
            total_loss = 0.0
            for data, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def get_gradients(self):
        """Extract gradients from model"""
        return [param.grad.clone() for param in self.model.parameters() if param.grad is not None]
    
    def set_weights(self, weights):
        """Update model with new weights"""
        for param, weight in zip(self.model.parameters(), weights):
            param.data = weight.clone()
