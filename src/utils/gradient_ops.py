"""Gradient operation utilities for model update manipulation"""

import torch
import numpy as np


def flatten_gradients(gradients: list) -> np.ndarray:
    """Flatten list of gradient tensors into single vector"""
    flattened = []
    for g in gradients:
        if isinstance(g, torch.Tensor):
            flattened.append(g.detach().cpu().numpy().flatten())
        else:
            flattened.append(np.array(g).flatten())
    return np.concatenate(flattened)


def unflatten_gradients(flat_gradients: np.ndarray, shapes: list) -> list:
    """Reshape flat gradient vector back to original tensor shapes"""
    gradients = []
    offset = 0
    
    for shape in shapes:
        size = np.prod(shape)
        grad = flat_gradients[offset:offset + size].reshape(shape)
        gradients.append(torch.from_numpy(grad).float())
        offset += size
    
    return gradients


def compute_gradient_norm(gradients: list) -> float:
    """Compute L2 norm of gradients"""
    flat = flatten_gradients(gradients)
    return np.linalg.norm(flat)


def scale_gradients(gradients: list, scale_factor: float) -> list:
    """Scale gradients by a factor"""
    scaled = []
    for g in gradients:
        if isinstance(g, torch.Tensor):
            scaled.append(g * scale_factor)
        else:
            scaled.append(np.array(g) * scale_factor)
    return scaled


def average_gradients(gradient_list: list) -> list:
    """Average multiple gradient lists"""
    if not gradient_list:
        return []
    
    num_updates = len(gradient_list)
    averaged = None
    
    for gradients in gradient_list:
        if averaged is None:
            averaged = [g / num_updates if isinstance(g, torch.Tensor) else np.array(g) / num_updates 
                       for g in gradients]
        else:
            averaged = [avg + (g / num_updates if isinstance(g, torch.Tensor) else np.array(g) / num_updates)
                       for avg, g in zip(averaged, gradients)]
    
    return averaged


def gradient_cosine_similarity(grad1: list, grad2: list) -> float:
    """Compute cosine similarity between two gradient vectors"""
    flat1 = flatten_gradients(grad1)
    flat2 = flatten_gradients(grad2)
    
    dot_product = np.dot(flat1, flat2)
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    
    return dot_product / (norm1 * norm2 + 1e-8)
