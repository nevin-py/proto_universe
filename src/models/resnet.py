"""ResNet18 adapted for CIFAR-10 (32×32 images).

Standard torchvision ResNet18 with two modifications for the smaller
input resolution:

1. Replace the 7×7 / stride-2 first convolution with a 3×3 / stride-1
   convolution (padding 1).
2. Remove the initial max-pool layer (replaced by ``nn.Identity``).

These changes are standard practice in the CIFAR-10 research literature
(He et al., 2016; Kuangliu/pytorch-cifar) to avoid excessive spatial
down-sampling on 32×32 inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class CIFAR10ResNet18(nn.Module):
    """ResNet18 for CIFAR-10 with 10-class output.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 10 for CIFAR-10).
    pretrained : bool
        Whether to load ImageNet pretrained weights (not recommended
        for 32×32 images but available for transfer learning experiments).
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Modify conv1: 7×7/stride-2 → 3×3/stride-1/padding-1
        self.model.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3, stride=1, padding=1,
            bias=False,
        )

        # Remove max-pool (32×32 is already small)
        self.model.maxpool = nn.Identity()

        # Adjust final FC layer for num_classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
