"""Robust aggregation using Multi-Krum algorithm"""

import numpy as np
import torch


class MultiKrumAggregator:
    """Layer 3: Multi-Krum robust aggregation"""
    
    def __init__(self, f: int = 1, m: int = 1):
        """
        Initialize Multi-Krum aggregator
        
        Args:
            f: Number of Byzantine clients to tolerate
            m: Number of Krum outputs to select
        """
        self.f = f
        self.m = m
        self.selected_indices = []
    
    def aggregate(self, updates: list) -> dict:
        """Perform Multi-Krum aggregation"""
        if not updates:
            return None
        
        n = len(updates)
        
        # Flatten gradients
        flattened = []
        for update in updates:
            flat = np.concatenate([g.flatten() if isinstance(g, torch.Tensor) 
                                   else np.array(g).flatten() 
                                   for g in update['gradients']])
            flattened.append(flat)
        
        flattened = np.array(flattened)
        
        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i][j] = np.linalg.norm(flattened[i] - flattened[j])
        
        # Select Krum candidates
        selected = []
        for i in range(n):
            # Sort distances for client i
            sorted_indices = np.argsort(distances[i])
            # Sum of k nearest neighbors (where k = n - f - 1)
            k = n - self.f - 1
            distance_sum = np.sum(sorted_indices[:k])
            selected.append((i, distance_sum))
        
        # Select m best candidates
        selected.sort(key=lambda x: x[1])
        self.selected_indices = [s[0] for s in selected[:min(self.m, n)]]
        
        # Average selected candidates
        aggregated = np.mean([flattened[i] for i in self.selected_indices], axis=0)
        
        return {
            'gradients': aggregated,
            'selected_indices': self.selected_indices
        }
    
    def get_selected_indices(self):
        """Get indices of selected updates"""
        return self.selected_indices
