"""Galaxy-level aggregator for hierarchical federated learning"""

import torch
import numpy as np


class GalaxyAggregator:
    """Aggregates model updates within a galaxy"""
    
    def __init__(self, galaxy_id: int, num_clients: int):
        """Initialize galaxy aggregator"""
        self.galaxy_id = galaxy_id
        self.num_clients = num_clients
        self.client_ids = []
        self.latest_aggregation = None
    
    def add_client(self, client_id: int):
        """Add a client to this galaxy"""
        self.client_ids.append(client_id)
    
    def aggregate(self, client_updates: list, weights: list = None) -> dict:
        """Aggregate client updates with optional weighting"""
        if not client_updates:
            return None
        
        # Default: uniform weights
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Aggregate gradients
        aggregated_update = None
        for update, weight in zip(client_updates, weights):
            if aggregated_update is None:
                aggregated_update = [g * weight for g in update['gradients']]
            else:
                for i, g in enumerate(update['gradients']):
                    aggregated_update[i] += g * weight
        
        self.latest_aggregation = {
            'galaxy_id': self.galaxy_id,
            'gradients': aggregated_update,
            'num_clients': len(client_updates),
            'timestamp': None
        }
        
        return self.latest_aggregation
    
    def get_aggregation(self):
        """Get latest aggregation result"""
        return self.latest_aggregation
    
    def clear_cache(self):
        """Clear cached aggregation"""
        self.latest_aggregation = None
