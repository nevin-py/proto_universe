"""Statistical utilities for analysis and anomaly detection"""

import numpy as np
import scipy.stats as stats


def compute_mean_std(data: np.ndarray, axis=0):
    """Compute mean and standard deviation"""
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    return mean, std


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> list:
    """Detect outliers using Z-score method"""
    mean, std = compute_mean_std(data)
    z_scores = np.abs((data - mean) / (std + 1e-8))
    return np.where(np.max(z_scores, axis=1) > threshold)[0].tolist()


def detect_outliers_iqr(data: np.ndarray) -> list:
    """Detect outliers using Interquartile Range method"""
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = []
    for i, row in enumerate(data):
        if np.any(row < lower_bound) or np.any(row > upper_bound):
            outliers.append(i)
    
    return outliers


def compute_distance_matrix(data: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix"""
    n = len(data)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            distances[i][j] = np.linalg.norm(data[i] - data[j])
    
    return distances


def compute_entropy(data: np.ndarray) -> float:
    """Compute Shannon entropy"""
    value_counts = np.unique(data, return_counts=True)[1]
    probabilities = value_counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy


def compute_kurtosis(data: np.ndarray) -> float:
    """Compute kurtosis of data"""
    return stats.kurtosis(data)


def compute_skewness(data: np.ndarray) -> float:
    """Compute skewness of data"""
    return stats.skew(data)
