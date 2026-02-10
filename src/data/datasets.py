"""Dataset loading utilities for federated learning.

Provides functions to load and preprocess common datasets.
"""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms


def load_mnist(
    data_dir: str = "data/mnist",
    download: bool = True
) -> Tuple[Dataset, Dataset]:
    """Load MNIST dataset.
    
    Args:
        data_dir: Directory to store/load data
        download: Whether to download if not present
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )
    
    return train_dataset, test_dataset


def load_cifar10(
    data_dir: str = "data/cifar10",
    download: bool = True
) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load data
        download: Whether to download if not present
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform_test
    )
    
    return train_dataset, test_dataset


def get_dataset_info(dataset: Dataset) -> dict:
    """Get information about a dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        Dictionary with dataset info
    """
    info = {
        'size': len(dataset),
        'type': type(dataset).__name__
    }
    
    # Try to get label information
    try:
        if hasattr(dataset, 'targets'):
            targets = torch.tensor(dataset.targets)
            info['num_classes'] = len(targets.unique())
            info['class_distribution'] = {
                int(c): int((targets == c).sum()) 
                for c in targets.unique()
            }
    except:
        pass
    
    # Try to get sample shape
    try:
        sample = dataset[0]
        if isinstance(sample, tuple):
            info['sample_shape'] = list(sample[0].shape)
        else:
            info['sample_shape'] = list(sample.shape)
    except:
        pass
    
    return info


class LabelFlippedDataset(Dataset):
    """Dataset wrapper that flips labels (for Byzantine simulation).
    
    Used to simulate label flipping attacks by malicious clients.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        flip_ratio: float = 1.0,
        num_classes: int = 10,
        seed: int = 42
    ):
        """Initialize label flipped dataset.
        
        Args:
            dataset: Base dataset
            flip_ratio: Fraction of labels to flip (0-1)
            num_classes: Total number of classes
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.flip_ratio = flip_ratio
        self.num_classes = num_classes
        
        # Determine which indices to flip
        torch.manual_seed(seed)
        n = len(dataset)
        self.flip_mask = torch.rand(n) < flip_ratio
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data, label = self.dataset[idx]
        
        if self.flip_mask[idx]:
            # Flip to random different label
            new_label = (label + torch.randint(1, self.num_classes, (1,)).item()) % self.num_classes
            return data, new_label
        
        return data, label


class NoisyDataset(Dataset):
    """Dataset wrapper that adds noise to data (for Byzantine simulation).
    
    Used to simulate gradient poisoning via noisy data.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        noise_level: float = 0.1,
        seed: int = 42
    ):
        """Initialize noisy dataset.
        
        Args:
            dataset: Base dataset
            noise_level: Standard deviation of Gaussian noise
            seed: Random seed
        """
        self.dataset = dataset
        self.noise_level = noise_level
        torch.manual_seed(seed)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data, label = self.dataset[idx]
        
        # Add Gaussian noise
        noise = torch.randn_like(data) * self.noise_level
        noisy_data = data + noise
        
        return noisy_data, label
