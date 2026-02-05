"""Input validation utilities"""

import torch
import numpy as np


def validate_gradients(gradients: list) -> bool:
    """Validate gradient list format and values"""
    if not isinstance(gradients, list):
        return False
    
    for g in gradients:
        if isinstance(g, torch.Tensor):
            if torch.isnan(g).any() or torch.isinf(g).any():
                return False
        elif isinstance(g, np.ndarray):
            if np.isnan(g).any() or np.isinf(g).any():
                return False
        else:
            return False
    
    return True


def validate_model_weights(weights: list) -> bool:
    """Validate model weight list"""
    if not isinstance(weights, list):
        return False
    
    for w in weights:
        if isinstance(w, torch.Tensor):
            if w.dtype not in [torch.float32, torch.float64]:
                return False
        else:
            return False
    
    return True


def validate_config(config: dict) -> bool:
    """Validate configuration dictionary"""
    required_keys = ['fl', 'galaxy', 'model', 'defense']
    
    for key in required_keys:
        if key not in config:
            return False
    
    return True


def validate_learning_rate(lr: float) -> bool:
    """Validate learning rate value"""
    return 0 < lr <= 1.0


def validate_batch_size(batch_size: int) -> bool:
    """Validate batch size"""
    return batch_size > 0 and batch_size <= 32768


def validate_num_epochs(num_epochs: int) -> bool:
    """Validate number of epochs"""
    return num_epochs > 0 and num_epochs <= 1000
