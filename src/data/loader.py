"""Data loader creation utilities for federated learning.

Provides functions to create DataLoaders for each client
based on partitioned data.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
import torch
from torch.utils.data import Dataset, DataLoader, Subset

if TYPE_CHECKING:
    from src.data.partition import DataPartitioner


def create_client_loaders(
    dataset: Dataset,
    partitions: Dict[int, List[int]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Dict[int, DataLoader]:
    """Create DataLoaders for each client.
    
    Args:
        dataset: Full dataset
        partitions: Dictionary mapping client_id to list of data indices
        batch_size: Batch size for each loader
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU
        
    Returns:
        Dictionary mapping client_id to DataLoader
    """
    client_loaders = {}
    
    for client_id, indices in partitions.items():
        client_subset = Subset(dataset, indices)
        
        loader = DataLoader(
            client_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        client_loaders[client_id] = loader
    
    return client_loaders


def create_galaxy_loaders(
    dataset: Dataset,
    partitions: Dict[int, List[int]],
    galaxy_assignments: Dict[int, int],
    batch_size: int = 32,
    shuffle: bool = True
) -> Dict[int, Dict[int, DataLoader]]:
    """Create DataLoaders organized by galaxy.
    
    Args:
        dataset: Full dataset
        partitions: Client partitions
        galaxy_assignments: Mapping of client_id to galaxy_id
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        Nested dict: galaxy_id -> client_id -> DataLoader
    """
    # First create all client loaders
    client_loaders = create_client_loaders(
        dataset, partitions, batch_size, shuffle
    )
    
    # Organize by galaxy
    galaxy_loaders: Dict[int, Dict[int, DataLoader]] = {}
    
    for client_id, loader in client_loaders.items():
        galaxy_id = galaxy_assignments.get(client_id, 0)
        
        if galaxy_id not in galaxy_loaders:
            galaxy_loaders[galaxy_id] = {}
        
        galaxy_loaders[galaxy_id][client_id] = loader
    
    return galaxy_loaders


def create_test_loader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for test data.
    
    Args:
        dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        DataLoader for evaluation
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


class FLDataManager:
    """Manager for federated learning data distribution.
    
    Handles dataset loading, partitioning, and loader creation.
    """
    
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        num_clients: int,
        num_galaxies: int = 1,
        batch_size: int = 32,
        partitioner: Optional['DataPartitioner'] = None
    ):
        """Initialize FL data manager.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            num_clients: Number of clients
            num_galaxies: Number of galaxies
            batch_size: Batch size for loaders
            partitioner: Data partitioning strategy
        """
        from src.data.partition import IIDPartitioner
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_clients = num_clients
        self.num_galaxies = num_galaxies
        self.batch_size = batch_size
        
        # Default to IID partitioning
        self.partitioner = partitioner or IIDPartitioner()
        
        # Create partitions
        self.partitions = self.partitioner.partition(train_dataset, num_clients)
        
        # Assign clients to galaxies
        self.galaxy_assignments = self._assign_galaxies()
        
        # Create loaders
        self.client_loaders = None
        self.test_loader = None
    
    def _assign_galaxies(self) -> Dict[int, int]:
        """Assign clients to galaxies (round-robin)."""
        assignments = {}
        for client_id in range(self.num_clients):
            assignments[client_id] = client_id % self.num_galaxies
        return assignments
    
    def get_client_loaders(self) -> Dict[int, DataLoader]:
        """Get DataLoaders for all clients."""
        if self.client_loaders is None:
            self.client_loaders = create_client_loaders(
                self.train_dataset,
                self.partitions,
                self.batch_size
            )
        return self.client_loaders
    
    def get_client_loader(self, client_id: int) -> Optional[DataLoader]:
        """Get DataLoader for a specific client."""
        loaders = self.get_client_loaders()
        return loaders.get(client_id)
    
    def get_galaxy_loaders(self) -> Dict[int, Dict[int, DataLoader]]:
        """Get loaders organized by galaxy."""
        return create_galaxy_loaders(
            self.train_dataset,
            self.partitions,
            self.galaxy_assignments,
            self.batch_size
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        if self.test_loader is None:
            self.test_loader = create_test_loader(
                self.test_dataset,
                batch_size=self.batch_size * 2  # Larger batch for eval
            )
        return self.test_loader
    
    def get_clients_in_galaxy(self, galaxy_id: int) -> List[int]:
        """Get list of client IDs in a galaxy."""
        return [
            client_id 
            for client_id, g_id in self.galaxy_assignments.items()
            if g_id == galaxy_id
        ]
    
    def get_partition_stats(self) -> Dict:
        """Get statistics about the data partition."""
        from src.data.partition import get_partition_stats
        return get_partition_stats(self.train_dataset, self.partitions)
    
    def get_client_data_size(self, client_id: int) -> int:
        """Get number of samples for a client."""
        if client_id in self.partitions:
            return len(self.partitions[client_id])
        return 0
