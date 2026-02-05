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
    
    def __init__(self, client_id: int, model: torch.nn.Module, attack_type: str = "gradient_poison"):
        """Initialize Byzantine client simulator"""
        self.client = Client(client_id, model, is_byzantine=True)
        self.attack_type = attack_type
        self.attack_history = []
    
    def simulate_round(self, num_epochs: int = 1):
        """Simulate one round with attack"""
        # Simulate training (or skip)
        self.client.train_local(None, 0)  # No real training
        
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
            # Byzantine client
            sim = ByzantineClientSimulator(i, model, attack_type)
        else:
            # Honest client
            data_loader = data_loaders[i] if i < len(data_loaders) else None
            sim = HonestClientSimulator(i, model, data_loader)
        
        simulators.append(sim)
    
    return simulators
