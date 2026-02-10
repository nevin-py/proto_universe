"""Local model training for federated learning clients.

Provides training functionality with:
- Local training on client data
- Gradient and weight-based updates
- Model evaluation
- Training metrics tracking
"""

import copy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Handles local model training on client devices.
    
    Supports both gradient-based and weight-based FL approaches.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        device: Optional[str] = None
    ):
        """Initialize trainer with model and hyperparameters.
        
        Args:
            model: PyTorch model to train
            learning_rate: Learning rate for optimizer
            momentum: Momentum for SGD optimizer
            weight_decay: L2 regularization weight
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Store initial weights for computing gradients
        self._initial_weights: Optional[List[torch.Tensor]] = None
        
        # Training history
        self.training_history: List[Dict] = []
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 1,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Train the model locally.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of local epochs
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training metrics
        """
        if train_loader is None or len(train_loader) == 0:
            return {'loss': 0.0, 'samples': 0}
        
        # Store initial weights
        self._initial_weights = self.get_weights()
        
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                batch_size = data.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
                
                _, predicted = torch.max(outputs.data, 1)
                epoch_correct += (predicted == labels).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            correct += epoch_correct
            
            if verbose:
                print(f"  Epoch {epoch + 1}/{num_epochs}: "
                      f"Loss={epoch_loss/epoch_samples:.4f}, "
                      f"Acc={epoch_correct/epoch_samples:.4f}")
        
        metrics = {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'accuracy': correct / total_samples if total_samples > 0 else 0.0,
            'samples': total_samples,
            'epochs': num_epochs
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        if test_loader is None:
            return {'accuracy': 0.0, 'loss': 0.0}
        
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * labels.size(0)
                
                if return_predictions:
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            'accuracy': correct / total if total > 0 else 0.0,
            'loss': total_loss / total if total > 0 else 0.0,
            'correct': correct,
            'total': total
        }
        
        if return_predictions:
            metrics['predictions'] = all_predictions
            metrics['labels'] = all_labels
        
        return metrics
    
    def get_gradients(self) -> List[torch.Tensor]:
        """Extract gradients from model (difference from initial weights).
        
        This computes pseudo-gradients as the difference between
        initial weights and current weights after training.
        
        Returns:
            List of gradient tensors
        """
        if self._initial_weights is None:
            # Return actual gradients if available
            return [
                param.grad.clone().detach() 
                for param in self.model.parameters() 
                if param.grad is not None
            ]
        
        # Compute weight difference (pseudo-gradient)
        current_weights = self.get_weights()
        gradients = [
            (init_w - curr_w).detach()
            for init_w, curr_w in zip(self._initial_weights, current_weights)
        ]
        
        return gradients
    
    def get_weights(self) -> List[torch.Tensor]:
        """Get current model weights.
        
        Returns:
            List of weight tensors
        """
        return [param.data.clone().detach() for param in self.model.parameters()]
    
    def set_weights(self, weights: List[torch.Tensor]) -> None:
        """Update model with new weights.
        
        Args:
            weights: List of weight tensors
        """
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.data.copy_(weight.to(self.device))
    
    def get_model_state(self) -> Dict:
        """Get model state dictionary.
        
        Returns:
            Model state dict
        """
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_state(self, state_dict: Dict) -> None:
        """Set model state from dictionary.
        
        Args:
            state_dict: Model state dictionary
        """
        self.model.load_state_dict(state_dict)
    
    def get_weight_update(self) -> Optional[List[torch.Tensor]]:
        """Get weight update (new weights - initial weights).
        
        Returns:
            List of weight update tensors, or None if no initial weights
        """
        if self._initial_weights is None:
            return None
        
        current = self.get_weights()
        return [
            (curr - init).detach()
            for curr, init in zip(current, self._initial_weights)
        ]
    
    def apply_gradient_update(
        self,
        gradients: List[torch.Tensor],
        learning_rate: Optional[float] = None
    ) -> None:
        """Apply gradient update to model weights.
        
        Args:
            gradients: Gradients to apply
            learning_rate: Learning rate (uses default if None)
        """
        lr = learning_rate or self.learning_rate
        
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), gradients):
                param.data -= lr * grad.to(self.device)
    
    def apply_weight_update(self, weight_update: List[torch.Tensor]) -> None:
        """Apply weight update directly to model.
        
        Args:
            weight_update: Weight changes to apply
        """
        with torch.no_grad():
            for param, update in zip(self.model.parameters(), weight_update):
                param.data += update.to(self.device)
    
    def reset_optimizer(self) -> None:
        """Reset optimizer state."""
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
    
    def clone_model(self) -> nn.Module:
        """Create a copy of the model.
        
        Returns:
            Deep copy of model
        """
        model_copy = copy.deepcopy(self.model)
        return model_copy
    
    def get_training_summary(self) -> Dict:
        """Get summary of training history.
        
        Returns:
            Summary dictionary
        """
        if not self.training_history:
            return {'rounds': 0}
        
        losses = [h['loss'] for h in self.training_history]
        accuracies = [h.get('accuracy', 0) for h in self.training_history]
        
        return {
            'rounds': len(self.training_history),
            'avg_loss': sum(losses) / len(losses),
            'final_loss': losses[-1],
            'avg_accuracy': sum(accuracies) / len(accuracies),
            'final_accuracy': accuracies[-1],
            'total_samples': sum(h['samples'] for h in self.training_history)
        }
