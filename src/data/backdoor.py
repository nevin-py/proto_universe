"""Backdoor (data-level) poisoning dataset wrapper.

Injects a small pixel-pattern trigger into a configurable fraction of
training samples and reassigns their label to a chosen target class.
This enables authentic evaluation of data-level backdoor attacks in
federated learning settings.

Usage
-----
>>> from src.data.backdoor import BackdoorDataset
>>> from src.data.datasets import load_mnist
>>> train_ds, _ = load_mnist()
>>> trigger = torch.ones(1, 5, 5) * 2.5      # white 5×5 patch
>>> bd = BackdoorDataset(train_ds, trigger, trigger_position=(0, 0),
...                      target_class=0, poisoning_rate=0.1)
>>> img, label = bd[0]                        # may have trigger injected

The trigger is applied *after* the base dataset's transform pipeline,
so it operates on the normalised tensor representation.  If the base
transform includes normalisation, the trigger values should be chosen
in the normalised space (e.g. values > 2.0 are visually bright for
MNIST with mean 0.1307 / std 0.3081).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("protogalaxy.data.backdoor")


class BackdoorDataset(Dataset):
    """Wraps a base dataset and injects a pixel-level backdoor trigger.

    Parameters
    ----------
    base_dataset : Dataset
        The clean dataset to wrap.
    trigger_pattern : Tensor
        The trigger patch.  Shape ``(C, H_t, W_t)`` where ``C`` matches
        the image channels.
    trigger_position : tuple of int
        ``(row, col)`` top-left corner where the trigger is stamped.
    target_class : int
        The label assigned to all triggered samples.
    poisoning_rate : float
        Fraction of the dataset that will carry the trigger
        (0.0–1.0).  The poisoned indices are determined
        deterministically at construction for reproducibility.
    seed : int
        Random seed for selecting which samples are poisoned.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        trigger_pattern: torch.Tensor,
        trigger_position: Tuple[int, int] = (0, 0),
        target_class: int = 0,
        poisoning_rate: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.trigger_pattern = trigger_pattern.clone().detach()
        self.trigger_position = trigger_position
        self.target_class = target_class
        self.poisoning_rate = float(np.clip(poisoning_rate, 0.0, 1.0))
        self.seed = seed

        # Determine poisoned indices deterministically
        rng = np.random.RandomState(seed)
        total = len(base_dataset)
        num_poisoned = int(total * self.poisoning_rate)
        self._poisoned_indices = set(
            rng.choice(total, size=num_poisoned, replace=False).tolist()
        )

        logger.info(
            "BackdoorDataset: %d / %d samples poisoned (%.1f%%), "
            "target_class=%d, trigger_shape=%s, position=%s",
            num_poisoned,
            total,
            self.poisoning_rate * 100,
            target_class,
            tuple(self.trigger_pattern.shape),
            trigger_position,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        """Return ``(image, label)`` — possibly with trigger injected.

        If ``idx`` is in the poisoned set:
        - The trigger pattern is stamped onto the image at
          ``trigger_position``.
        - The label is changed to ``target_class``.

        Otherwise the sample is returned unchanged.
        """
        image, label = self.base_dataset[idx]

        if idx in self._poisoned_indices:
            image = self._apply_trigger(image)
            label = self.target_class

        return image, label

    # ------------------------------------------------------------------
    # Trigger injection
    # ------------------------------------------------------------------

    def _apply_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """Stamp the trigger onto a copy of the image tensor.

        Parameters
        ----------
        image : Tensor
            Shape ``(C, H, W)`` — the normalised image tensor.

        Returns
        -------
        Tensor
            The image with the trigger pattern overlaid.
        """
        triggered = image.clone()
        row, col = self.trigger_position
        c, h_t, w_t = self.trigger_pattern.shape

        # Clamp to image bounds
        h_img, w_img = triggered.shape[1], triggered.shape[2]
        r_end = min(row + h_t, h_img)
        c_end = min(col + w_t, w_img)
        h_actual = r_end - row
        w_actual = c_end - col

        if h_actual <= 0 or w_actual <= 0:
            logger.warning(
                "Trigger position (%d, %d) is outside the image (%d×%d); "
                "no trigger applied.",
                row, col, h_img, w_img,
            )
            return triggered

        # Overlay trigger (additive stamp)
        triggered[:c, row:r_end, col:c_end] = self.trigger_pattern[
            :c, :h_actual, :w_actual
        ]

        return triggered

    # ------------------------------------------------------------------
    # Helper: create standard triggers
    # ------------------------------------------------------------------

    @staticmethod
    def white_square_trigger(
        channels: int = 1,
        size: int = 5,
        intensity: float = 2.5,
    ) -> torch.Tensor:
        """Create a solid white square trigger.

        Parameters
        ----------
        channels : int
            Number of image channels (1 for MNIST, 3 for CIFAR-10).
        size : int
            Side length of the square trigger in pixels.
        intensity : float
            Pixel value to stamp (should be large in normalised space
            to be clearly visible).

        Returns
        -------
        Tensor of shape ``(channels, size, size)``
        """
        return torch.full((channels, size, size), intensity, dtype=torch.float32)

    @staticmethod
    def cross_trigger(
        channels: int = 1,
        size: int = 5,
        intensity: float = 2.5,
    ) -> torch.Tensor:
        """Create a cross (plus sign) pattern trigger.

        Parameters
        ----------
        channels : int
            Number of image channels.
        size : int
            Side length of the bounding box.
        intensity : float
            Pixel value for the cross lines.

        Returns
        -------
        Tensor of shape ``(channels, size, size)``
        """
        trigger = torch.zeros(channels, size, size, dtype=torch.float32)
        mid = size // 2
        trigger[:, mid, :] = intensity      # horizontal bar
        trigger[:, :, mid] = intensity      # vertical bar
        return trigger

    @staticmethod
    def checkerboard_trigger(
        channels: int = 1,
        size: int = 4,
        intensity: float = 2.5,
    ) -> torch.Tensor:
        """Create a checkerboard pattern trigger.

        Parameters
        ----------
        channels : int
            Number of image channels.
        size : int
            Side length of the trigger (should be even).
        intensity : float
            Pixel value for the "on" squares.

        Returns
        -------
        Tensor of shape ``(channels, size, size)``
        """
        trigger = torch.zeros(channels, size, size, dtype=torch.float32)
        for r in range(size):
            for c in range(size):
                if (r + c) % 2 == 0:
                    trigger[:, r, c] = intensity
        return trigger

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_poisoned(self) -> int:
        """Number of samples that carry the backdoor trigger."""
        return len(self._poisoned_indices)

    @property
    def poisoned_indices(self) -> set:
        """Set of indices that are poisoned."""
        return set(self._poisoned_indices)

    def get_info(self) -> dict:
        """Summary information about the backdoor configuration."""
        return {
            "base_size": len(self.base_dataset),
            "num_poisoned": self.num_poisoned,
            "poisoning_rate": self.poisoning_rate,
            "target_class": self.target_class,
            "trigger_shape": tuple(self.trigger_pattern.shape),
            "trigger_position": self.trigger_position,
        }
