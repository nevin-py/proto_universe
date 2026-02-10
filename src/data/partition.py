"""Data partitioning strategies for federated learning.

Provides different partitioning methods:
- IID: Independent and identically distributed
- Non-IID: Heterogeneous label distribution
- Dirichlet: Dirichlet distribution-based partitioning
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class DataPartitioner(ABC):
    """Abstract base class for data partitioning strategies."""
    
    @abstractmethod
    def partition(
        self,
        dataset: Dataset,
        num_clients: int
    ) -> Dict[int, List[int]]:
        """Partition dataset into client subsets.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to list of data indices
        """
        pass
    
    def get_client_datasets(
        self,
        dataset: Dataset,
        num_clients: int
    ) -> Dict[int, Subset]:
        """Get Subset objects for each client.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to Subset
        """
        partitions = self.partition(dataset, num_clients)
        return {
            client_id: Subset(dataset, indices)
            for client_id, indices in partitions.items()
        }


class IIDPartitioner(DataPartitioner):
    """IID (Independent and Identically Distributed) partitioner.
    
    Randomly shuffles data and distributes equally across clients.
    Each client gets roughly the same amount of data from each class.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize IID partitioner.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
    
    def partition(
        self,
        dataset: Dataset,
        num_clients: int
    ) -> Dict[int, List[int]]:
        """Partition dataset using IID strategy.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to list of indices
        """
        np.random.seed(self.seed)
        
        n = len(dataset)
        indices = np.random.permutation(n)
        
        # Split indices into approximately equal chunks
        splits = np.array_split(indices, num_clients)
        
        return {i: split.tolist() for i, split in enumerate(splits)}


class NonIIDPartitioner(DataPartitioner):
    """Non-IID partitioner with label-based distribution.
    
    Creates heterogeneous data distribution where each client
    has data from only a subset of classes.
    """
    
    def __init__(
        self,
        num_classes_per_client: int = 2,
        seed: int = 42
    ):
        """Initialize Non-IID partitioner.
        
        Args:
            num_classes_per_client: Number of classes each client receives
            seed: Random seed
        """
        self.num_classes_per_client = num_classes_per_client
        self.seed = seed
    
    def partition(
        self,
        dataset: Dataset,
        num_clients: int
    ) -> Dict[int, List[int]]:
        """Partition dataset using Non-IID strategy.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to list of indices
        """
        np.random.seed(self.seed)
        
        # Get labels
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        else:
            # Fallback: iterate through dataset
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        num_classes = len(np.unique(labels))
        
        # Group indices by class
        class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
        
        # Shuffle indices within each class
        for c in class_indices:
            np.random.shuffle(class_indices[c])
        
        # Assign classes to clients
        partitions = {i: [] for i in range(num_clients)}
        
        # Round-robin class assignment
        classes_per_client = max(1, min(self.num_classes_per_client, num_classes))
        
        for client_id in range(num_clients):
            # Assign classes to this client
            assigned_classes = [
                (client_id + j) % num_classes 
                for j in range(classes_per_client)
            ]
            
            for cls in assigned_classes:
                # Get portion of this class for this client
                cls_idx = class_indices[cls]
                n_samples = len(cls_idx) // (num_clients // classes_per_client + 1)
                
                start = (client_id // classes_per_client) * n_samples
                end = start + n_samples
                
                partitions[client_id].extend(cls_idx[start:end].tolist())
        
        return partitions


class DirichletPartitioner(DataPartitioner):
    """Dirichlet distribution-based partitioner.
    
    Uses Dirichlet distribution to create naturally heterogeneous
    data distribution with controllable heterogeneity via alpha parameter.
    
    Lower alpha = more heterogeneous (clients have fewer classes)
    Higher alpha = more homogeneous (clients have more balanced classes)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        min_samples: int = 10,
        seed: int = 42
    ):
        """Initialize Dirichlet partitioner.
        
        Args:
            alpha: Dirichlet concentration parameter (lower = more heterogeneous)
            min_samples: Minimum samples per client
            seed: Random seed
        """
        self.alpha = alpha
        self.min_samples = min_samples
        self.seed = seed
    
    def partition(
        self,
        dataset: Dataset,
        num_clients: int
    ) -> Dict[int, List[int]]:
        """Partition dataset using Dirichlet distribution.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to list of indices
        """
        np.random.seed(self.seed)
        
        # Get labels
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        num_classes = len(np.unique(labels))
        n = len(dataset)
        
        # Initialize empty partitions
        partitions = {i: [] for i in range(num_clients)}
        
        # Get indices for each class
        class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
        
        # For each class, distribute samples according to Dirichlet
        for cls in range(num_classes):
            cls_idx = class_indices[cls]
            np.random.shuffle(cls_idx)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * num_clients)
            
            # Convert to actual counts
            counts = (proportions * len(cls_idx)).astype(int)
            
            # Distribute any remainder
            remainder = len(cls_idx) - counts.sum()
            for i in range(remainder):
                counts[i % num_clients] += 1
            
            # Assign indices to clients
            current = 0
            for client_id in range(num_clients):
                end = current + counts[client_id]
                partitions[client_id].extend(cls_idx[current:end].tolist())
                current = end
        
        # Ensure minimum samples
        self._redistribute_if_needed(partitions, n)
        
        return partitions
    
    def _redistribute_if_needed(
        self,
        partitions: Dict[int, List[int]],
        total: int
    ) -> None:
        """Redistribute samples to ensure minimum per client."""
        for client_id, indices in partitions.items():
            if len(indices) < self.min_samples:
                # Find clients with excess samples
                for other_id, other_indices in partitions.items():
                    if len(other_indices) > self.min_samples * 2:
                        # Transfer some samples
                        to_transfer = min(
                            self.min_samples - len(indices),
                            len(other_indices) - self.min_samples
                        )
                        transferred = other_indices[:to_transfer]
                        partitions[other_id] = other_indices[to_transfer:]
                        partitions[client_id].extend(transferred)
                        
                        if len(partitions[client_id]) >= self.min_samples:
                            break


class ShardPartitioner(DataPartitioner):
    """Shard-based Non-IID partitioner.
    
    Sorts data by label and divides into shards, then assigns
    a fixed number of shards to each client.
    This is the Non-IID strategy used in the original FedAvg paper.
    """
    
    def __init__(
        self,
        num_shards_per_client: int = 2,
        seed: int = 42
    ):
        """Initialize shard partitioner.
        
        Args:
            num_shards_per_client: Number of shards per client
            seed: Random seed
        """
        self.num_shards_per_client = num_shards_per_client
        self.seed = seed
    
    def partition(
        self,
        dataset: Dataset,
        num_clients: int
    ) -> Dict[int, List[int]]:
        """Partition dataset using shard strategy.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to list of indices
        """
        np.random.seed(self.seed)
        
        # Get labels
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        n = len(dataset)
        num_shards = num_clients * self.num_shards_per_client
        shard_size = n // num_shards
        
        # Sort indices by label
        sorted_indices = np.argsort(labels)
        
        # Create shards
        shards = [
            sorted_indices[i * shard_size:(i + 1) * shard_size].tolist()
            for i in range(num_shards)
        ]
        
        # Handle remaining samples
        remainder = sorted_indices[num_shards * shard_size:]
        if len(remainder) > 0:
            for i, idx in enumerate(remainder):
                shards[i % num_shards].append(idx)
        
        # Randomly assign shards to clients
        shard_ids = list(range(num_shards))
        np.random.shuffle(shard_ids)
        
        partitions = {}
        for client_id in range(num_clients):
            client_shards = shard_ids[
                client_id * self.num_shards_per_client:
                (client_id + 1) * self.num_shards_per_client
            ]
            partitions[client_id] = []
            for shard_id in client_shards:
                partitions[client_id].extend(shards[shard_id])
        
        return partitions


def get_partition_stats(
    dataset: Dataset,
    partitions: Dict[int, List[int]]
) -> Dict[str, any]:
    """Get statistics about a partition.
    
    Args:
        dataset: Original dataset
        partitions: Partition dictionary
        
    Returns:
        Statistics dictionary
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    stats = {
        'num_clients': len(partitions),
        'total_samples': sum(len(idx) for idx in partitions.values()),
        'samples_per_client': {},
        'classes_per_client': {},
        'class_distribution': {}
    }
    
    for client_id, indices in partitions.items():
        client_labels = labels[indices]
        unique_classes = np.unique(client_labels)
        
        stats['samples_per_client'][client_id] = len(indices)
        stats['classes_per_client'][client_id] = len(unique_classes)
        stats['class_distribution'][client_id] = {
            int(c): int((client_labels == c).sum())
            for c in unique_classes
        }
    
    # Summary stats
    samples_list = list(stats['samples_per_client'].values())
    classes_list = list(stats['classes_per_client'].values())
    
    stats['avg_samples_per_client'] = np.mean(samples_list)
    stats['std_samples_per_client'] = np.std(samples_list)
    stats['avg_classes_per_client'] = np.mean(classes_list)
    stats['std_classes_per_client'] = np.std(classes_list)
    
    return stats
