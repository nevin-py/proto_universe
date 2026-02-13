"""Adaptive attacker that observes defense statistics and crafts evasive poisons.

The adversary maintains a sliding window of honest gradient statistics
(norms and cosine similarities) and generates malicious updates that
stay within the statistical acceptance envelope of Layer 1/2 detectors
while still biasing the global model toward the attacker's objective.

Strategy
--------
1. **Observation phase** (``observation_window`` rounds):  Record mean and
   standard deviation of honest gradient norms and mutual cosine
   similarities, plus the honest centroid direction.
2. **Poisoning phase** (subsequent rounds):  Craft a gradient that
   maximally moves the model toward the attack goal subject to:
   -  ``||g_poison|| ∈ [μ_norm - 2σ, μ_norm + 2σ]``
   -  ``cosine(g_poison, centroid) > cos_threshold``

This makes the poisoned gradient look statistically indistinguishable
from honest ones to norm- and direction-based anomaly detectors.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.gradient_ops import (
    flatten_gradients,
    unflatten_gradients,
    compute_gradient_norm,
    gradient_cosine_similarity,
    average_gradients,
)

logger = logging.getLogger("protogalaxy.attack.adaptive")


class AdaptiveAttacker:
    """An adversary that evades statistical anomaly detection.

    Parameters
    ----------
    observation_window : int
        Number of rounds of honest-gradient observation before the
        attacker begins adapting.  During observation the attacker
        submits honest gradients (stealth mode).
    norm_sigma_bound : float
        How many standard deviations from the mean honest norm the
        poisoned gradient norm may deviate (default 2.0).
    cos_threshold : float
        Minimum cosine similarity with the honest centroid that the
        poisoned gradient must maintain (default 0.3).
    attack_strength : float
        Scaling factor applied to the adversarial perturbation
        component before projection (higher → more aggressive but
        harder to stay within bounds).
    """

    def __init__(
        self,
        observation_window: int = 5,
        norm_sigma_bound: float = 2.0,
        cos_threshold: float = 0.3,
        attack_strength: float = 2.0,
    ):
        self.observation_window = observation_window
        self.norm_sigma_bound = norm_sigma_bound
        self.cos_threshold = cos_threshold
        self.attack_strength = attack_strength

        # Sliding-window statistics
        self._round_count: int = 0
        self._norm_history: deque[List[float]] = deque(maxlen=observation_window)
        self._cosine_history: deque[List[float]] = deque(maxlen=observation_window)
        self._centroid_history: deque[np.ndarray] = deque(maxlen=observation_window)

        # Cached statistics (updated after each observe call)
        self._mean_norm: float = 1.0
        self._std_norm: float = 0.5
        self._centroid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe_round(self, honest_gradients: List[List[torch.Tensor]]) -> None:
        """Record statistics of honest gradients for threshold estimation.

        Parameters
        ----------
        honest_gradients : list of list of Tensor
            Each inner list is one honest client's per-parameter gradients.
        """
        if not honest_gradients:
            return

        # Flatten each client's gradients
        flat_grads = [flatten_gradients(g) for g in honest_gradients]

        # Norms
        norms = [float(np.linalg.norm(fg)) for fg in flat_grads]
        self._norm_history.append(norms)

        # Honest centroid (mean direction)
        centroid = np.mean(flat_grads, axis=0)
        cent_norm = float(np.linalg.norm(centroid))
        if cent_norm > 1e-12:
            centroid = centroid / cent_norm
        self._centroid_history.append(centroid)

        # Pairwise cosine similarities against centroid
        cosines = []
        for fg in flat_grads:
            fg_norm = float(np.linalg.norm(fg))
            if fg_norm > 1e-12 and cent_norm > 1e-12:
                cos_val = float(np.dot(fg, centroid) / fg_norm)
            else:
                cos_val = 0.0
            cosines.append(cos_val)
        self._cosine_history.append(cosines)

        # Update cached statistics
        all_norms = [n for batch in self._norm_history for n in batch]
        self._mean_norm = float(np.mean(all_norms))
        self._std_norm = float(np.std(all_norms)) if len(all_norms) > 1 else 0.5
        self._centroid = centroid

        self._round_count += 1

    @property
    def is_ready(self) -> bool:
        """Whether enough rounds have been observed to start adapting."""
        return self._round_count >= self.observation_window

    # ------------------------------------------------------------------
    # Poison generation
    # ------------------------------------------------------------------

    def generate_adaptive_poison(
        self,
        honest_gradients: List[List[torch.Tensor]],
        attack_goal: str = "untargeted",
        target_class: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Craft a malicious gradient that evades detection.

        The algorithm:
        1. Compute the honest centroid ``c``.
        2. Compute the adversarial direction ``d``:
           - *untargeted*: ``d = -c``  (maximise loss / reverse honest direction)
           - *targeted*:   ``d = -c + bias toward target_class`` (if possible)
        3. Blend: ``g = α·c + (1 - α)·d``  choosing α so that
           ``cosine(g, c) ≥ cos_threshold``.
        4. Scale ``g`` so that ``||g|| ∈ [μ - kσ, μ + kσ]``.

        Parameters
        ----------
        honest_gradients : list of list of Tensor
            Current round's honest gradients (used for centroid even if
            the attacker has prior observations).
        attack_goal : str
            ``'untargeted'`` or ``'targeted'``.
        target_class : int or None
            Required when ``attack_goal == 'targeted'``.

        Returns
        -------
        list of Tensor
            Poisoned gradient tensors matching the shape of the first
            honest client's gradients.
        """
        if not honest_gradients:
            raise ValueError("Cannot generate adaptive poison without honest gradients.")

        # Always update observation
        self.observe_round(honest_gradients)

        # Reference shapes from first honest gradient
        reference_shapes = [
            g.shape if isinstance(g, torch.Tensor) else np.array(g).shape
            for g in honest_gradients[0]
        ]

        flat_grads = [flatten_gradients(g) for g in honest_gradients]
        centroid = np.mean(flat_grads, axis=0)
        cent_norm = float(np.linalg.norm(centroid))
        if cent_norm < 1e-12:
            # If centroid is zero, just return zeros
            return [torch.zeros(s, dtype=torch.float32) for s in reference_shapes]

        centroid_unit = centroid / cent_norm

        # ------------------------------------------------------------------
        # Adversarial direction
        # ------------------------------------------------------------------
        if attack_goal == "untargeted":
            # Maximise global loss by reversing the honest direction
            adv_direction = -centroid_unit
        elif attack_goal == "targeted":
            # Targeted: bias toward a random perturbation that is
            # orthogonal to centroid, plus a negation component.
            # This subtly moves the decision boundary for the target class.
            rand_component = np.random.randn(len(centroid_unit)).astype(np.float32)
            # Gram-Schmidt: orthogonalise w.r.t. centroid
            proj = np.dot(rand_component, centroid_unit) * centroid_unit
            orth = rand_component - proj
            orth_norm = float(np.linalg.norm(orth))
            if orth_norm > 1e-12:
                orth = orth / orth_norm
            # Mix negation and orthogonal
            adv_direction = -0.5 * centroid_unit + 0.5 * orth
            adv_norm = float(np.linalg.norm(adv_direction))
            if adv_norm > 1e-12:
                adv_direction = adv_direction / adv_norm
        else:
            raise ValueError(f"Unknown attack_goal: {attack_goal}")

        # ------------------------------------------------------------------
        # Blend centroid and adversarial to meet cosine threshold
        # ------------------------------------------------------------------
        # g = α·centroid_unit + (1-α)·attack_strength·adv_direction
        # We want cosine(g, centroid_unit) ≥ cos_threshold.
        # cosine ≈ α / ||g||  (if adv_direction ⊥ centroid_unit it's exact)
        # Binary search for the largest attack contribution that meets threshold.
        alpha_lo, alpha_hi = 0.0, 1.0
        best_alpha = 1.0  # default: pure centroid (safe)
        for _ in range(30):
            alpha = (alpha_lo + alpha_hi) / 2.0
            candidate = alpha * centroid_unit + (1.0 - alpha) * self.attack_strength * adv_direction
            cand_norm = float(np.linalg.norm(candidate))
            if cand_norm < 1e-12:
                alpha_lo = alpha
                continue
            cos_val = float(np.dot(candidate / cand_norm, centroid_unit))
            if cos_val >= self.cos_threshold:
                best_alpha = alpha
                alpha_hi = alpha  # try more aggressive (lower α)
            else:
                alpha_lo = alpha  # need more centroid

        # Construct the poisoned gradient with best alpha
        poisoned_flat = best_alpha * centroid_unit + (1.0 - best_alpha) * self.attack_strength * adv_direction

        # ------------------------------------------------------------------
        # Scale to match honest norm statistics
        # ------------------------------------------------------------------
        target_norm = self._mean_norm + 0.5 * self._std_norm  # slightly above mean
        # Clamp to [μ - kσ, μ + kσ]
        norm_lo = max(0.01, self._mean_norm - self.norm_sigma_bound * self._std_norm)
        norm_hi = self._mean_norm + self.norm_sigma_bound * self._std_norm
        target_norm = float(np.clip(target_norm, norm_lo, norm_hi))

        curr_norm = float(np.linalg.norm(poisoned_flat))
        if curr_norm > 1e-12:
            poisoned_flat = poisoned_flat * (target_norm / curr_norm)

        # Unflatten back to per-parameter tensors
        poisoned_tensors = unflatten_gradients(poisoned_flat, reference_shapes)

        # Convert to torch tensors
        result = []
        for t in poisoned_tensors:
            if isinstance(t, np.ndarray):
                result.append(torch.tensor(t, dtype=torch.float32))
            else:
                result.append(t)

        logger.debug(
            "Adaptive poison: α=%.3f  norm=%.4f (target=%.4f)  cos=%.4f",
            best_alpha,
            float(np.linalg.norm(flatten_gradients(result))),
            target_norm,
            float(gradient_cosine_similarity(result, [torch.tensor(centroid_unit)])),
        )

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, float]:
        """Return the currently estimated honest-gradient statistics."""
        return {
            "rounds_observed": self._round_count,
            "mean_norm": self._mean_norm,
            "std_norm": self._std_norm,
            "norm_lower_bound": max(0.01, self._mean_norm - self.norm_sigma_bound * self._std_norm),
            "norm_upper_bound": self._mean_norm + self.norm_sigma_bound * self._std_norm,
            "cos_threshold": self.cos_threshold,
            "is_ready": self.is_ready,
        }
