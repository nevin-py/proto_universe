"""Storage manager for persisting models and metrics"""

import torch
import json
import os
from datetime import datetime


class StorageManager:
    """Manages persistent storage of models and data"""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize storage manager"""
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, "models")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        
        # Create directories
        for dir_path in [self.models_dir, self.logs_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_model(self, model: torch.nn.Module, round_number: int, metadata: dict = None):
        """Save model checkpoint"""
        model_path = os.path.join(self.models_dir, f"round_{round_number:02d}.pt")
        
        checkpoint = {
            'round': round_number,
            'model_state': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, model_path)
        return model_path
    
    def load_model(self, model: torch.nn.Module, round_number: int):
        """Load model checkpoint"""
        model_path = os.path.join(self.models_dir, f"round_{round_number:02d}.pt")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state'])
            return model
        return None
    
    def save_metrics(self, metrics: dict, filename: str):
        """Save metrics to JSON file"""
        metrics_path = os.path.join(self.metrics_dir, f"{filename}.json")
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics_path
    
    def load_metrics(self, filename: str):
        """Load metrics from JSON file"""
        metrics_path = os.path.join(self.metrics_dir, f"{filename}.json")
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return None
    
    def save_log(self, log_content: str, filename: str):
        """Save log to file"""
        log_path = os.path.join(self.logs_dir, f"{filename}.log")
        
        with open(log_path, 'a') as f:
            f.write(log_content + '\n')
        
        return log_path
    
    def get_available_models(self):
        """List all available model checkpoints"""
        models = []
        for f in os.listdir(self.models_dir):
            if f.endswith('.pt'):
                models.append(f)
        return sorted(models)
    
    def cleanup_old_models(self, keep_last_n: int = 5):
        """Remove old model checkpoints, keeping only last N"""
        models = self.get_available_models()
        if len(models) > keep_last_n:
            for model_file in models[:-keep_last_n]:
                os.remove(os.path.join(self.models_dir, model_file))
