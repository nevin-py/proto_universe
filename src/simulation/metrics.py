"""Metrics collection and reporting for federated learning"""

import numpy as np
import pandas as pd


class MetricsCollector:
    """Collects and aggregates training metrics"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = {
            'accuracy': [],
            'loss': [],
            'detections': [],
            'reputation': [],
            'communication_rounds': 0
        }
        self.per_client_metrics = {}
    
    def record_round_metrics(self, round_num: int, accuracy: float, loss: float):
        """Record metrics for a training round"""
        self.metrics['accuracy'].append(accuracy)
        self.metrics['loss'].append(loss)
    
    def record_detection(self, round_num: int, num_detected: int):
        """Record Byzantine detections"""
        self.metrics['detections'].append({
            'round': round_num,
            'count': num_detected
        })
    
    def record_reputation_scores(self, round_num: int, scores: dict):
        """Record reputation scores"""
        self.metrics['reputation'].append({
            'round': round_num,
            'scores': scores.copy()
        })
    
    def record_client_metrics(self, client_id: int, loss: float, accuracy: float):
        """Record per-client metrics"""
        if client_id not in self.per_client_metrics:
            self.per_client_metrics[client_id] = {'loss': [], 'accuracy': []}
        
        self.per_client_metrics[client_id]['loss'].append(loss)
        self.per_client_metrics[client_id]['accuracy'].append(accuracy)
    
    def get_summary(self) -> dict:
        """Get summary of all metrics"""
        return {
            'total_rounds': len(self.metrics['accuracy']),
            'final_accuracy': self.metrics['accuracy'][-1] if self.metrics['accuracy'] else 0.0,
            'final_loss': self.metrics['loss'][-1] if self.metrics['loss'] else 0.0,
            'avg_accuracy': np.mean(self.metrics['accuracy']) if self.metrics['accuracy'] else 0.0,
            'num_detections': len(self.metrics['detections']),
            'total_detected': sum(d['count'] for d in self.metrics['detections'])
        }
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get metrics as pandas DataFrame"""
        return pd.DataFrame({
            'round': range(len(self.metrics['accuracy'])),
            'accuracy': self.metrics['accuracy'],
            'loss': self.metrics['loss']
        })
    
    def export_csv(self, filepath: str):
        """Export metrics to CSV file"""
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'accuracy': [],
            'loss': [],
            'detections': [],
            'reputation': [],
            'communication_rounds': 0
        }
        self.per_client_metrics = {}
