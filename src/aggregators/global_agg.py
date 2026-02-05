"""Global aggregator for federated learning"""

import torch
import numpy as np


class GlobalAggregator:
    """Aggregates galaxy-level updates into global model update"""
    
    def __init__(self, num_galaxies: int):
        """Initialize global aggregator"""
        self.num_galaxies = num_galaxies
        self.galaxy_updates = []
        self.global_update = None
        self.aggregation_history = []
    
    def aggregate(self, galaxy_updates: list, weights: list = None) -> dict:
        """Aggregate galaxy updates with optional weighting"""
        if not galaxy_updates:
            return None
        
        # Default: uniform weights
        if weights is None:
            weights = [1.0 / len(galaxy_updates)] * len(galaxy_updates)
        
        # Aggregate gradients
        aggregated_update = None
        for update, weight in zip(galaxy_updates, weights):
            if aggregated_update is None:
                aggregated_update = [g * weight for g in update['gradients']]
            else:
                for i, g in enumerate(update['gradients']):
                    aggregated_update[i] += g * weight
        
        self.global_update = {
            'round': len(self.aggregation_history),
            'gradients': aggregated_update,
            'num_galaxies': len(galaxy_updates),
            'timestamp': None
        }
        
        self.aggregation_history.append(self.global_update)
        return self.global_update
    
    def get_global_update(self):
        """Get the latest global update"""
        return self.global_update
    
    def get_history(self):
        """Get aggregation history"""
        return self.aggregation_history
    
    def compute_statistics(self) -> dict:
        """Compute statistics about aggregations"""
        if not self.aggregation_history:
            return {}
        
        return {
            'num_rounds': len(self.aggregation_history),
            'latest_round': self.global_update['round'] if self.global_update else -1
        }
