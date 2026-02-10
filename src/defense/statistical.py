"""Statistical defense mechanisms for Byzantine-resilient aggregation

This module implements the 3-metric statistical analyzer as specified in PROTO-302:
1. Norm deviation (threshold: 3σ)
2. Direction similarity (cosine threshold: 0.5)
3. Coordinate-wise analysis (per-dimension outliers)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Set


def _flatten_gradients(gradients: list) -> np.ndarray:
    """Helper to flatten gradient list to numpy array"""
    flattened = []
    for g in gradients:
        if isinstance(g, torch.Tensor):
            flattened.append(g.detach().cpu().numpy().flatten())
        else:
            flattened.append(np.array(g).flatten())
    return np.concatenate(flattened)


class StatisticalAnalyzer:
    """
    Multi-metric anomaly detection for client gradients (PROTO-302).
    
    Implements 3 detection metrics:
    1. Norm deviation - flags gradients with unusual L2 norms (>3σ from mean)
    2. Direction similarity - flags gradients with low cosine similarity to mean (<0.5)
    3. Coordinate-wise analysis - flags gradients with outlier coordinates
    
    A gradient is flagged as anomalous if it fails ≥2 metrics.
    """
    
    def __init__(
        self,
        norm_threshold_sigma: float = 3.0,
        cosine_threshold: float = 0.5,
        coordinate_threshold_sigma: float = 3.0,
        min_failures_to_flag: int = 2
    ):
        """
        Initialize statistical analyzer.
        
        Args:
            norm_threshold_sigma: Number of standard deviations for norm outlier detection
            cosine_threshold: Minimum cosine similarity to mean gradient
            coordinate_threshold_sigma: Number of std devs for coordinate-wise outliers
            min_failures_to_flag: Minimum number of failed metrics to flag as anomaly
        """
        self.norm_threshold_sigma = norm_threshold_sigma
        self.cosine_threshold = cosine_threshold
        self.coordinate_threshold_sigma = coordinate_threshold_sigma
        self.min_failures_to_flag = min_failures_to_flag
        
        # Detection history
        self.detection_history = []
        self.metric_failures = {}  # client_id -> {metric_name: count}
    
    def analyze(self, updates: List[Dict]) -> Dict:
        """
        Analyze gradient updates using all 3 metrics.
        
        Args:
            updates: List of update dicts with 'client_id' and 'gradients' keys
            
        Returns:
            Dict with analysis results including flagged clients and per-metric results
        """
        if not updates or len(updates) < 3:
            return {
                'flagged_clients': [],
                'norm_outliers': [],
                'direction_outliers': [],
                'coordinate_outliers': [],
                'failure_counts': {}
            }
        
        # Extract client IDs and flatten gradients
        client_ids = [u.get('client_id', i) for i, u in enumerate(updates)]
        flattened = np.array([_flatten_gradients(u['gradients']) for u in updates])
        
        # Run all 3 metrics
        norm_outliers = self._detect_norm_deviation(flattened)
        direction_outliers = self._detect_direction_anomaly(flattened)
        coordinate_outliers = self._detect_coordinate_outliers(flattened)
        
        # Count failures per client
        failure_counts = {}
        for i in range(len(updates)):
            failures = 0
            if i in norm_outliers:
                failures += 1
            if i in direction_outliers:
                failures += 1
            if i in coordinate_outliers:
                failures += 1
            failure_counts[client_ids[i]] = failures
        
        # Flag clients with >= min_failures_to_flag failures
        flagged_indices = [i for i, cid in enumerate(client_ids) 
                         if failure_counts[cid] >= self.min_failures_to_flag]
        flagged_clients = [client_ids[i] for i in flagged_indices]
        
        # Record history
        self.detection_history.append({
            'flagged': flagged_clients,
            'norm_outliers': [client_ids[i] for i in norm_outliers],
            'direction_outliers': [client_ids[i] for i in direction_outliers],
            'coordinate_outliers': [client_ids[i] for i in coordinate_outliers]
        })
        
        return {
            'flagged_clients': flagged_clients,
            'flagged_indices': flagged_indices,
            'norm_outliers': [client_ids[i] for i in norm_outliers],
            'direction_outliers': [client_ids[i] for i in direction_outliers],
            'coordinate_outliers': [client_ids[i] for i in coordinate_outliers],
            'failure_counts': failure_counts
        }
    
    def _detect_norm_deviation(self, flattened: np.ndarray) -> List[int]:
        """
        Metric 1: Detect gradients with abnormal L2 norms.
        Flags gradients whose norm deviates > threshold_sigma from mean.
        """
        norms = np.linalg.norm(flattened, axis=1)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        if std_norm < 1e-8:
            return []
        
        z_scores = np.abs((norms - mean_norm) / std_norm)
        outliers = np.where(z_scores > self.norm_threshold_sigma)[0].tolist()
        return outliers
    
    def _detect_direction_anomaly(self, flattened: np.ndarray) -> List[int]:
        """
        Metric 2: Detect gradients with abnormal direction (low cosine similarity).
        Computes cosine similarity to mean gradient and flags those below threshold.
        """
        mean_gradient = np.mean(flattened, axis=0)
        mean_norm = np.linalg.norm(mean_gradient)
        
        if mean_norm < 1e-8:
            return []
        
        outliers = []
        for i, grad in enumerate(flattened):
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                outliers.append(i)
                continue
            
            cosine_sim = np.dot(grad, mean_gradient) / (grad_norm * mean_norm)
            if cosine_sim < self.cosine_threshold:
                outliers.append(i)
        
        return outliers
    
    def _detect_coordinate_outliers(self, flattened: np.ndarray) -> List[int]:
        """
        Metric 3: Coordinate-wise outlier detection.
        Checks each dimension independently and flags gradients with outlier coordinates.
        """
        mean = np.mean(flattened, axis=0)
        std = np.std(flattened, axis=0)
        
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        
        # Z-scores for each coordinate
        z_scores = np.abs((flattened - mean) / std)
        
        # Count outlier coordinates per gradient
        outlier_coords_count = np.sum(z_scores > self.coordinate_threshold_sigma, axis=1)
        
        # Flag if more than 10% of coordinates are outliers
        threshold_count = max(1, int(0.1 * flattened.shape[1]))
        outliers = np.where(outlier_coords_count > threshold_count)[0].tolist()
        
        return outliers
    
    def get_detection_history(self) -> List[Dict]:
        """Get history of all detections"""
        return self.detection_history
    
    def reset(self):
        """Reset detection history"""
        self.detection_history = []
        self.metric_failures = {}


class StatisticalDefenseLayer1:
    """
    Layer 1: Z-score based anomaly detection.
    Simple mean/std deviation check on flattened gradients.
    """
    
    def __init__(self, threshold: float = 3.0):
        """Initialize defense layer 1"""
        self.threshold = threshold
        self.detected_anomalies = []
    
    def detect_anomalies(self, updates: list) -> list:
        """Detect anomalous updates using z-score method"""
        if not updates:
            return []
        
        # Flatten and convert to numpy
        flattened = []
        for update in updates:
            flat = _flatten_gradients(update['gradients'])
            flattened.append(flat)
        
        flattened = np.array(flattened)
        
        # Compute mean and std
        mean = np.mean(flattened, axis=0)
        std = np.std(flattened, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        
        # Detect outliers using z-score
        anomalies = []
        for i, update in enumerate(flattened):
            z_scores = np.abs((update - mean) / std)
            if np.max(z_scores) > self.threshold:
                anomalies.append(i)
                self.detected_anomalies.append(i)
        
        return anomalies
    
    def reset(self):
        """Reset detection history"""
        self.detected_anomalies = []


class StatisticalDefenseLayer2:
    """
    Layer 2: Multi-dimensional norm-based anomaly detection.
    Checks gradient L2 norms and flags extreme values.
    """
    
    def __init__(self, norm_threshold_sigma: float = 3.0):
        """Initialize defense layer 2"""
        self.norm_threshold_sigma = norm_threshold_sigma
        self.detected_anomalies = []
    
    def detect_anomalies(self, updates: list) -> list:
        """Detect anomalies using norm analysis"""
        if not updates or len(updates) < 3:
            return []
        
        # Compute norms
        norms = []
        for update in updates:
            norm = np.linalg.norm(_flatten_gradients(update['gradients']))
            norms.append(norm)
        
        norms = np.array(norms)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        if std_norm < 1e-8:
            return []
        
        # Detect outliers
        anomalies = []
        for i, norm in enumerate(norms):
            z_score = abs((norm - mean_norm) / std_norm)
            if z_score > self.norm_threshold_sigma:
                anomalies.append(i)
                self.detected_anomalies.append(i)
        
        return anomalies
    
    def get_detection_history(self):
        """Get history of detected anomalies"""
        return self.detected_anomalies
    
    def reset(self):
        """Reset detection history"""
        self.detected_anomalies = []
