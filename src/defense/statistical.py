"""Statistical defense mechanisms for Byzantine-resilient aggregation"""

import numpy as np
import torch


class StatisticalDefenseLayer1:
    """Layer 1: Mean and standard deviation based anomaly detection"""
    
    def __init__(self, threshold: float = 2.0):
        """Initialize defense layer 1"""
        self.threshold = threshold
        self.detected_anomalies = []
    
    def detect_anomalies(self, updates: list) -> list:
        """Detect anomalous updates using statistical methods"""
        if not updates:
            return []
        
        # Flatten and convert to numpy
        flattened = []
        for update in updates:
            flat = np.concatenate([g.flatten() if isinstance(g, torch.Tensor) else np.array(g).flatten() 
                                   for g in update['gradients']])
            flattened.append(flat)
        
        flattened = np.array(flattened)
        
        # Compute mean and std
        mean = np.mean(flattened, axis=0)
        std = np.std(flattened, axis=0)
        
        # Detect outliers
        anomalies = []
        for i, update in enumerate(flattened):
            distances = np.abs((update - mean) / (std + 1e-8))
            if np.max(distances) > self.threshold:
                anomalies.append(i)
                self.detected_anomalies.append(i)
        
        return anomalies


class StatisticalDefenseLayer2:
    """Layer 2: Multi-dimensional anomaly detection"""
    
    def __init__(self, threshold: float = 3.0):
        """Initialize defense layer 2"""
        self.threshold = threshold
        self.detected_anomalies = []
    
    def detect_anomalies(self, updates: list) -> list:
        """Detect anomalies using multi-dimensional analysis"""
        if not updates:
            return []
        
        anomalies = []
        # Simplified multi-dimensional detection
        for i in range(len(updates)):
            # Check norm of gradient vectors
            norm = np.linalg.norm(np.concatenate([g.flatten() if isinstance(g, torch.Tensor) 
                                                   else np.array(g).flatten() 
                                                   for g in updates[i]['gradients']]))
            if norm > self.threshold:
                anomalies.append(i)
                self.detected_anomalies.append(i)
        
        return anomalies
    
    def get_detection_history(self):
        """Get history of detected anomalies"""
        return self.detected_anomalies
