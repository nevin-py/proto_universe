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
        self.trimmed_indices: dict = {}  # Maps dimension -> set of trimmed indices (Deprecated)
        self.client_trim_counts: dict = {}  # Maps client_id -> count of trims
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
        
        # Coordinate-wise trimmed mean (Vectorized)
        if trim_count == 0:
            aggregated = np.mean(flattened, axis=0)
            self.client_trim_counts = {}
        else:
            # Sort along client dimension (axis 0)
            sorted_indices = np.argsort(flattened, axis=0)
            
            # Take middle values logic:
            # We can't simply take mean of middle values directly if we want to sort values
            # np.take_along_axis is useful here
            sorted_values = np.take_along_axis(flattened, sorted_indices, axis=0)
            
            # Keep only the middle rows
            kept_values = sorted_values[trim_count:n-trim_count, :]
            
            # Aggregate
            aggregated = np.mean(kept_values, axis=0)
            
            # Track frequent trimming (for reputation)
            # Flatten the trimmed indices (top and bottom)
            trimmed_low = sorted_indices[:trim_count, :].flatten()
            trimmed_high = sorted_indices[n-trim_count:, :].flatten()
            all_trimmed = np.concatenate([trimmed_low, trimmed_high])
            
            # Count occurrences of each client ID
            counts = np.bincount(all_trimmed, minlength=n)
            self.client_trim_counts = {i: int(c) for i, c in enumerate(counts) if c > 0}
            
            # Clear old memory-heavy dict
            self.trimmed_indices = {} 
        
        return {
            'gradients': aggregated,
            'trimmed_counts': {
                'per_end': trim_count,
                'total_per_dim': 2 * trim_count,
                'kept_per_dim': n - 2 * trim_count
            }
        }
    
    def get_trimmed_indices(self) -> dict:
        """Deprecated: Get indices of trimmed updates per dimension.
        
        Returns:
            Empty dict (memory optimization).
        """
        return {}
    
    def get_frequently_trimmed_clients(self, threshold: float = 0.5) -> list:
        """Identify clients that were frequently trimmed across dimensions.
        
        This can help identify consistently malicious clients.
        
        Args:
            threshold: Fraction of dimensions where client must be trimmed
                      to be considered "frequently trimmed"
        
        Returns:
            List of (client_index, trim_fraction) tuples sorted by fraction
        """
        if not hasattr(self, 'client_trim_counts') or not self.client_trim_counts or self.original_shapes == []:
             # Need total dims to compute fraction
             # We can compute total dims from original shapes or just assume it from last run
             # BUT aggregated shape is easier
             return []
        
        # Calculate total dimensions
        # self.client_trim_counts was computed on flattened array, so d is needed
        # We can approximate d by summing counts / (2 * trim_count) if we knew trim_count
        # Better: store d during aggregation
        
        # Let's check trimmed_counts from aggregate return, but here we don't have it.
        # We can infer total dimensions from summation of counts?
        # sum(counts) = d * 2 * trim_count
        # So d = sum(counts) / (2 * trim_count_per_dim)
        
        # Wait, simple way: max possible count is d.
        # But we don't stored 'd'.
        # Let's store 'd' in aggregate.
        # For now, let's fix this method after storing 'd' in aggregate properly.
        # Actually, let's look at how to get 'd'.
        
        total_dims = sum(np.prod(s) for s in self.original_shapes)
        if total_dims == 0:
            return []
            
        frequent = []
        for idx, count in self.client_trim_counts.items():
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
