"""Robust aggregation algorithms for Byzantine-resilient federated learning.

This module provides Layer 3 (Byzantine-Robust Aggregation) for the Protogalaxy
architecture. Implements:
- Trimmed Mean: Coordinate-wise trimming, O(n log n · d) complexity
- Multi-Krum: Distance-based selection (legacy, O(n² · d) complexity)
"""

from typing import Optional, Union

import numpy as np
import torch


class TrimmedMeanAggregator:
    """Layer 3: Trimmed Mean robust aggregation.
    
    Coordinate-wise trimmed mean removes outliers by trimming a fraction
    of the highest and lowest values at each dimension before averaging.
    
    This method is:
    - More efficient than Krum: O(n log n · d) vs O(n² · d)
    - Effective for symmetric attack distributions
    - Works well with near-Gaussian gradient distributions
    
    Per architecture Section 4.3:
        "Remove top and bottom β fraction of gradients per dimension.
         Average remaining gradients."
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        """Initialize Trimmed Mean aggregator.
        
        Args:
            trim_ratio: Fraction to trim from each end (0.0 to 0.5).
                        Default 0.1 trims 10% from each end (20% total).
        """
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0.0, 0.5)")
        
        self.trim_ratio = trim_ratio
        self.trimmed_indices: dict = {}  # Maps dimension -> set of trimmed indices
        self.original_shapes: list = []
        self._last_n: int = 0
    
    def aggregate(self, updates: list) -> Optional[dict]:
        """Perform coordinate-wise trimmed mean aggregation.
        
        Args:
            updates: List of update dicts with 'gradients' key containing
                     list of gradient tensors/arrays
        
        Returns:
            Dict with 'gradients' (aggregated) and 'trimmed_counts' (per-dim trim info)
            or None if no updates provided
        """
        if not updates:
            return None
        
        n = len(updates)
        self._last_n = n
        
        # Calculate how many to trim from each end
        trim_count = int(n * self.trim_ratio)
        
        # Handle edge case: not enough clients to trim
        if n - 2 * trim_count < 1:
            trim_count = max(0, (n - 1) // 2)
        
        # Flatten all gradients and record shapes
        flattened = []
        self.original_shapes = []
        
        for update in updates:
            grads = update['gradients']
            
            # Store shapes from first update for reconstruction
            if not self.original_shapes:
                for g in grads:
                    if isinstance(g, torch.Tensor):
                        self.original_shapes.append(g.shape)
                    else:
                        self.original_shapes.append(np.array(g).shape)
            
            # Flatten and concatenate all gradient components
            flat = np.concatenate([
                g.detach().cpu().numpy().flatten() if isinstance(g, torch.Tensor)
                else np.array(g).flatten()
                for g in grads
            ])
            flattened.append(flat)
        
        flattened = np.array(flattened)  # Shape: (n, d)
        d = flattened.shape[1]
        
        # Coordinate-wise trimmed mean
        aggregated = np.zeros(d, dtype=np.float32)
        self.trimmed_indices = {}
        
        for dim in range(d):
            values = flattened[:, dim]
            
            if trim_count > 0:
                # Sort and get indices
                sorted_indices = np.argsort(values)
                
                # Trim bottom and top
                keep_indices = sorted_indices[trim_count:n - trim_count]
                trimmed_low = set(sorted_indices[:trim_count])
                trimmed_high = set(sorted_indices[n - trim_count:])
                
                self.trimmed_indices[dim] = trimmed_low | trimmed_high
                
                # Average the kept values
                aggregated[dim] = np.mean(values[keep_indices])
            else:
                # No trimming, just average
                aggregated[dim] = np.mean(values)
        
        return {
            'gradients': aggregated,
            'trimmed_counts': {
                'per_end': trim_count,
                'total_per_dim': 2 * trim_count,
                'kept_per_dim': n - 2 * trim_count
            }
        }
    
    def get_trimmed_indices(self) -> dict:
        """Get indices of trimmed updates per dimension.
        
        Returns:
            Dict mapping dimension index to set of trimmed client indices
        """
        return self.trimmed_indices
    
    def get_frequently_trimmed_clients(self, threshold: float = 0.5) -> list:
        """Identify clients that were frequently trimmed across dimensions.
        
        This can help identify consistently malicious clients.
        
        Args:
            threshold: Fraction of dimensions where client must be trimmed
                      to be considered "frequently trimmed"
        
        Returns:
            List of (client_index, trim_fraction) tuples sorted by fraction
        """
        if not self.trimmed_indices or self._last_n == 0:
            return []
        
        # Count how often each client was trimmed
        trim_counts = {}
        total_dims = len(self.trimmed_indices)
        
        for dim, trimmed_set in self.trimmed_indices.items():
            for idx in trimmed_set:
                trim_counts[idx] = trim_counts.get(idx, 0) + 1
        
        # Calculate fraction and filter by threshold
        frequent = []
        for idx, count in trim_counts.items():
            fraction = count / total_dims
            if fraction >= threshold:
                frequent.append((idx, fraction))
        
        return sorted(frequent, key=lambda x: -x[1])


class MultiKrumAggregator:
    """Layer 3: Multi-Krum robust aggregation (legacy).
    
    Multi-Krum selects gradients based on pairwise distances, preferring
    gradients that are close to many others (assumed honest cluster).
    
    Complexity: O(n² · d) - more expensive than Trimmed Mean
    Use when: High Byzantine threat (f > 0.2n) as per architecture Section 4.3
    """
    
    def __init__(self, f: int = 1, m: int = 1):
        """Initialize Multi-Krum aggregator.
        
        Args:
            f: Number of Byzantine clients to tolerate
            m: Number of Krum outputs to select and average
        """
        self.f = f
        self.m = m
        self.selected_indices: list = []
    
    def aggregate(self, updates: list) -> Optional[dict]:
        """Perform Multi-Krum aggregation.
        
        Args:
            updates: List of update dicts with 'gradients' key
            
        Returns:
            Dict with 'gradients' and 'selected_indices' or None if empty
        """
        if not updates:
            return None
        
        n = len(updates)
        
        if n <= 2 * self.f + 2:
            # Not enough clients for Krum, fall back to simple average
            flattened = []
            for update in updates:
                flat = np.concatenate([
                    g.detach().cpu().numpy().flatten() if isinstance(g, torch.Tensor)
                    else np.array(g).flatten()
                    for g in update['gradients']
                ])
                flattened.append(flat)
            
            self.selected_indices = list(range(n))
            return {
                'gradients': np.mean(flattened, axis=0),
                'selected_indices': self.selected_indices
            }
        
        # Flatten gradients
        flattened = []
        for update in updates:
            flat = np.concatenate([
                g.detach().cpu().numpy().flatten() if isinstance(g, torch.Tensor)
                else np.array(g).flatten()
                for g in update['gradients']
            ])
            flattened.append(flat)
        
        flattened = np.array(flattened)
        
        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(flattened[i] - flattened[j])
                distances[i][j] = dist
                distances[j][i] = dist
        
        # Compute Krum scores
        k = max(1, n - self.f - 2)  # Number of nearest neighbors
        scores = []
        
        for i in range(n):
            # Get k smallest distances (excluding self)
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[1:k + 1])  # Skip distance to self (0)
            scores.append((i, score))
        
        # Select m best candidates (lowest scores)
        scores.sort(key=lambda x: x[1])
        self.selected_indices = [s[0] for s in scores[:min(self.m, n)]]
        
        # Average selected candidates
        aggregated = np.mean([flattened[i] for i in self.selected_indices], axis=0)
        
        return {
            'gradients': aggregated,
            'selected_indices': self.selected_indices
        }
    
    def get_selected_indices(self) -> list:
        """Get indices of selected updates.
        
        Returns:
            List of client indices selected by Krum
        """
        return self.selected_indices


class CoordinateWiseMedianAggregator:
    """Layer 3: Coordinate-Wise Median robust aggregation.
    
    For each parameter dimension, takes the median across all clients.
    This is optimal when up to half the clients can be Byzantine.
    
    Per architecture Section 4.3:
        "Coordinate-wise median offers computational efficiency
         with moderate threat tolerance."
    
    Complexity: O(n log n · d)  — same order as Trimmed Mean.
    Byzantine tolerance: up to 50% malicious clients (strongest).
    """
    
    def __init__(self):
        self.original_shapes: list = []
        self._last_n: int = 0
    
    def aggregate(self, updates: list) -> Optional[dict]:
        """Perform coordinate-wise median aggregation.
        
        Args:
            updates: List of update dicts with 'gradients' key
            
        Returns:
            Dict with 'gradients' (aggregated) or None if empty
        """
        if not updates:
            return None
        
        n = len(updates)
        self._last_n = n
        
        # Flatten all gradients and record shapes
        flattened = []
        self.original_shapes = []
        
        for update in updates:
            grads = update['gradients']
            if not self.original_shapes:
                for g in grads:
                    if isinstance(g, torch.Tensor):
                        self.original_shapes.append(g.shape)
                    else:
                        self.original_shapes.append(np.array(g).shape)
            
            flat = np.concatenate([
                g.detach().cpu().numpy().flatten() if isinstance(g, torch.Tensor)
                else np.array(g).flatten()
                for g in grads
            ])
            flattened.append(flat)
        
        flattened = np.array(flattened)  # (n, d)
        
        # Coordinate-wise median
        aggregated = np.median(flattened, axis=0).astype(np.float32)
        
        return {
            'gradients': aggregated,
            'method': 'coordinate_wise_median'
        }
