"""Statistical defense mechanisms for Byzantine-resilient aggregation

This module implements the 4-metric statistical analyzer (Architecture Section 4.2):
1. Norm deviation (threshold: 3σ)
2. Direction similarity (cosine threshold: 0.5)
3. Coordinate-wise analysis (per-dimension outliers)
4. Distribution shift detection (KL-divergence)

A gradient is flagged as anomalous if it fails ≥2 of the 4 metrics.
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
    Multi-metric anomaly detection for client gradients (Architecture Section 4.2).
    
    Implements 4 detection metrics:
    1. Norm deviation - flags gradients with unusual L2 norms (>3σ from mean)
    2. Direction similarity - flags gradients with low cosine similarity to mean (<0.5)
    3. Coordinate-wise analysis - flags gradients with outlier coordinates
    4. Distribution shift - flags gradients with high KL-divergence from aggregate
    
    A gradient is flagged as anomalous if it fails ≥2 of the 4 metrics.
    """
    
    def __init__(
        self,
        norm_threshold_sigma: float = 3.0,
        cosine_threshold: float = 0.5,
        coordinate_threshold_sigma: float = 3.0,
        kl_divergence_threshold: float = 2.0,
        min_failures_to_flag: int = 2
    ):
        """
        Initialize statistical analyzer.
        
        Args:
            norm_threshold_sigma: Number of standard deviations for norm outlier detection
            cosine_threshold: Minimum cosine similarity to mean gradient
            coordinate_threshold_sigma: Number of std devs for coordinate-wise outliers
            kl_divergence_threshold: Max KL-divergence from aggregate distribution
            min_failures_to_flag: Minimum number of failed metrics to flag as anomaly
        """
        self.norm_threshold_sigma = norm_threshold_sigma
        self.cosine_threshold = cosine_threshold
        self.coordinate_threshold_sigma = coordinate_threshold_sigma
        self.kl_divergence_threshold = kl_divergence_threshold
        self.min_failures_to_flag = min_failures_to_flag
        
        # Detection history
        self.detection_history = []
        self.metric_failures = {}  # client_id -> {metric_name: count}
    
    def analyze(self, updates: List[Dict]) -> Dict:
        """
        Analyze gradient updates using all 4 metrics (Architecture Section 4.2).
        
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
                'distribution_outliers': [],
                'failure_counts': {}
            }
        
        # Extract client IDs and flatten gradients
        client_ids = [u.get('client_id', i) for i, u in enumerate(updates)]
        flattened = np.array([_flatten_gradients(u['gradients']) for u in updates])
        
        # Run all 4 metrics
        norm_outliers = self._detect_norm_deviation(flattened)
        direction_outliers = self._detect_direction_anomaly(flattened)
        coordinate_outliers = self._detect_coordinate_outliers(flattened)
        distribution_outliers = self._detect_distribution_shift(flattened)
        
        # Count failures per client (out of 4 metrics)
        failure_counts = {}
        for i in range(len(updates)):
            failures = 0
            if i in norm_outliers:
                failures += 1
            if i in direction_outliers:
                failures += 1
            if i in coordinate_outliers:
                failures += 1
            if i in distribution_outliers:
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
            'coordinate_outliers': [client_ids[i] for i in coordinate_outliers],
            'distribution_outliers': [client_ids[i] for i in distribution_outliers]
        })
        
        return {
            'flagged_clients': flagged_clients,
            'flagged_indices': flagged_indices,
            'norm_outliers': [client_ids[i] for i in norm_outliers],
            'direction_outliers': [client_ids[i] for i in direction_outliers],
            'coordinate_outliers': [client_ids[i] for i in coordinate_outliers],
            'distribution_outliers': [client_ids[i] for i in distribution_outliers],
            'failure_counts': failure_counts
        }
    
    def _detect_norm_deviation(self, flattened: np.ndarray) -> List[int]:
        """
        Metric 1: Detect gradients with abnormal L2 norms.
        
        Uses **median + MAD** (Median Absolute Deviation) instead of
        mean + std so that the reference statistic is not corrupted
        when a significant fraction of clients are Byzantine.
        """
        norms = np.linalg.norm(flattened, axis=1)
        median_norm = np.median(norms)
        mad = np.median(np.abs(norms - median_norm))
        # Convert MAD to a std-like scale (for Gaussian: std ≈ 1.4826 * MAD)
        mad_std = 1.4826 * mad
        
        if mad_std < 1e-8:
            return []
        
        z_scores = np.abs((norms - median_norm) / mad_std)
        outliers = np.where(z_scores > self.norm_threshold_sigma)[0].tolist()
        return outliers
    
    def _detect_direction_anomaly(self, flattened: np.ndarray) -> List[int]:
        """
        Metric 2: Detect gradients with abnormal direction (low cosine similarity).
        
        Uses coordinate-wise **median** as the reference gradient instead
        of the mean, so that Byzantine clients (even at -10× scale)
        cannot flip the reference direction.
        """
        ref_gradient = np.median(flattened, axis=0)
        ref_norm = np.linalg.norm(ref_gradient)
        
        if ref_norm < 1e-8:
            return []
        
        outliers = []
        for i, grad in enumerate(flattened):
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                outliers.append(i)
                continue
            
            cosine_sim = np.dot(grad, ref_gradient) / (grad_norm * ref_norm)
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
    
    def _detect_distribution_shift(self, flattened: np.ndarray) -> List[int]:
        """
        Metric 4: Distribution shift detection via KL-divergence (Architecture Section 4.2).
        Bins each gradient into a histogram and computes KL-divergence against
        the aggregate distribution. Flags gradients with high divergence.
        """
        num_bins = min(50, max(10, flattened.shape[1] // 10))
        
        # Build aggregate distribution from all gradients combined
        all_values = flattened.flatten()
        bin_edges = np.histogram_bin_edges(all_values, bins=num_bins)
        
        aggregate_hist, _ = np.histogram(all_values, bins=bin_edges, density=True)
        # Add small epsilon to avoid div-by-zero in KL
        eps = 1e-10
        aggregate_hist = aggregate_hist + eps
        aggregate_hist = aggregate_hist / aggregate_hist.sum()
        
        outliers = []
        for i, grad in enumerate(flattened):
            grad_hist, _ = np.histogram(grad, bins=bin_edges, density=True)
            grad_hist = grad_hist + eps
            grad_hist = grad_hist / grad_hist.sum()
            
            # KL(grad || aggregate)
            kl = np.sum(grad_hist * np.log(grad_hist / aggregate_hist))
            
            if kl > self.kl_divergence_threshold:
                outliers.append(i)
        
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
