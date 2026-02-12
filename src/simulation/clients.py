"""Client simulators for honest and Byzantine clients"""

import torch
import numpy as np
from src.client.client import Client


class HonestClientSimulator:
    """Simulates honest clients following the protocol"""
    
    def __init__(self, client_id: int, model: torch.nn.Module, data_loader):
        """Initialize honest client simulator"""
        self.client = Client(client_id, model, is_byzantine=False)
        self.data_loader = data_loader
        self.training_history = []
    
    def simulate_round(self, num_epochs: int):
        """Simulate one training round"""
        loss = self.client.train_local(self.data_loader, num_epochs)
        accuracy = self.client.evaluate_local(self.data_loader)
        
        self.training_history.append({
            'loss': loss,
            'accuracy': accuracy
        })
        
        return {
            'client_id': self.client.client_id,
            'loss': loss,
            'accuracy': accuracy
        }
    
    def get_update(self):
        """Get client's model update"""
        return self.client.get_update()


class ByzantineClientSimulator:
    """Simulates Byzantine clients with various attack strategies"""
    
    def __init__(self, client_id: int, model: torch.nn.Module,
                 attack_type: str = "gradient_poison", data_loader=None):
        """Initialize Byzantine client simulator.
        
        Args:
            client_id: Client identifier
            model: Model to train / poison
            attack_type: Attack strategy name
            data_loader: Real data loader so that attacks operate on
                         genuine gradients (not zero vectors).
        """
        self.client = Client(client_id, model, is_byzantine=True)
        self.attack_type = attack_type
        self.data_loader = data_loader
        self.attack_history = []
    
    def simulate_round(self, num_epochs: int = 1):
        """Simulate one round with attack.
        
        The client trains on real data first so that attack perturbations
        are applied to genuine gradients rather than zero vectors.
        """
        # Train on real data so gradients are non-trivial
        if self.data_loader is not None:
            self.client.train_local(self.data_loader, num_epochs)
        else:
            self.client.train_local(None, 0)  # Legacy fallback
        
        # Apply attack
        self.client.attack(self.attack_type)
        
        self.attack_history.append({
            'attack_type': self.attack_type,
            'round': len(self.attack_history)
        })
        
        return {
            'client_id': self.client.client_id,
            'attack_type': self.attack_type,
            'is_byzantine': True
        }
    
    def get_update(self):
        """Get Byzantine client's poisoned update"""
        return self.client.get_update()


def create_client_simulators(num_clients: int, model: torch.nn.Module, 
                            data_loaders: list, num_byzantine: int = 0,
                            attack_type: str = "gradient_poison") -> list:
    """Factory function to create mixed honest and Byzantine client simulators"""
    simulators = []
    
    for i in range(num_clients):
        if i < num_byzantine:
            # Byzantine client — still give it a data loader so attacks
            # operate on real gradients
            data_loader = data_loaders[i] if i < len(data_loaders) else None
            sim = ByzantineClientSimulator(i, model, attack_type,
                                           data_loader=data_loader)
        else:
            # Honest client
            data_loader = data_loaders[i] if i < len(data_loaders) else None
            sim = HonestClientSimulator(i, model, data_loader)
        
        simulators.append(sim)
    
    return simulators
