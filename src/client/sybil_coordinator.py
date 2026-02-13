"""Sybil attack coordinator for federated learning evaluation.

A single adversary controls multiple client IDs (Sybil identities) and
coordinates their poisoning to overwhelm majority-based defenses.

Three coordination strategies are supported:

1. **Synchronized** — All Sybil clients attack every round with identical
   poisoned gradients.  Maximises Byzantine ratio per round.
2. **Rotating** — A random half of Sybil clients attack each round while
   the others submit honest gradients, making detection harder.
3. **Sleeper** — Sybil clients behave honestly for a configurable number
   of rounds to build positive reputation, then launch a coordinated
   attack.

All strategies produce gradient-level updates that replace the Sybil
clients' honest gradients in the evaluation loop.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from src.utils.gradient_ops import (
    flatten_gradients,
    unflatten_gradients,
    compute_gradient_norm,
    average_gradients,
)

logger = logging.getLogger("protogalaxy.attack.sybil")


class SybilCoordinator:
    """Coordinates poisoned gradients across multiple Sybil identities.

    Parameters
    ----------
    sybil_ids : list of int
        Client IDs controlled by the Sybil adversary.
    strategy : str
        ``'synchronized'``, ``'rotating'``, or ``'sleeper'``.
    attack_scale : float
        Amplification factor for the poisoned direction.
    sleeper_rounds : int
        For the ``'sleeper'`` strategy, how many rounds the Sybils
        behave honestly before attacking.
    rotation_fraction : float
        For the ``'rotating'`` strategy, fraction of Sybils that
        attack each round (the rest submit honest gradients).
    seed : int
        RNG seed for reproducible rotation schedules.
    """

    STRATEGY_SYNCHRONIZED = "synchronized"
    STRATEGY_ROTATING = "rotating"
    STRATEGY_SLEEPER = "sleeper"
    VALID_STRATEGIES = (STRATEGY_SYNCHRONIZED, STRATEGY_ROTATING, STRATEGY_SLEEPER)

    def __init__(
        self,
        sybil_ids: List[int],
        strategy: str = "synchronized",
        attack_scale: float = 5.0,
        sleeper_rounds: int = 10,
        rotation_fraction: float = 0.5,
        seed: int = 42,
    ):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid Sybil strategy '{strategy}'. "
                f"Choose from {self.VALID_STRATEGIES}."
            )
        if not sybil_ids:
            raise ValueError("sybil_ids must be non-empty.")

        self.sybil_ids = list(sybil_ids)
        self.strategy = strategy
        self.attack_scale = attack_scale
        self.sleeper_rounds = sleeper_rounds
        self.rotation_fraction = rotation_fraction
        self._rng = np.random.RandomState(seed)
        self._round_count: int = 0
        self._attack_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def coordinate_attack(
        self,
        round_number: int,
        honest_gradients: Optional[Dict[int, List[torch.Tensor]]] = None,
        sybil_honest_gradients: Optional[Dict[int, List[torch.Tensor]]] = None,
    ) -> Dict[int, List[torch.Tensor]]:
        """Return (possibly poisoned) gradients for all Sybil clients.

        Parameters
        ----------
        round_number : int
            Current FL round (0-indexed).
        honest_gradients : dict, optional
            Mapping ``{client_id: [grad_tensors]}`` for honest (non-Sybil)
            clients.  Used to compute the honest centroid for crafting an
            effective counter-gradient.
        sybil_honest_gradients : dict, optional
            Mapping ``{sybil_id: [grad_tensors]}`` — the honest gradients
            that Sybil clients would have produced from their actual
            training data.  These are returned unchanged when a Sybil
            is in "stealth" mode.

        Returns
        -------
        dict
            ``{sybil_id: [poisoned_or_honest_gradient_tensors]}`` for
            every Sybil ID.
        """
        self._round_count = round_number

        # Determine which Sybils attack this round
        attacking_ids = self._select_attacking_ids(round_number)

        result: Dict[int, List[torch.Tensor]] = {}
        log_entry = {
            "round": round_number,
            "strategy": self.strategy,
            "attacking_ids": list(attacking_ids),
            "stealth_ids": [s for s in self.sybil_ids if s not in attacking_ids],
        }

        # Compute the poison direction (negated honest centroid, scaled)
        poison_template = self._compute_poison(honest_gradients)

        for sid in self.sybil_ids:
            if sid in attacking_ids and poison_template is not None:
                result[sid] = poison_template
            else:
                # Stealth: return the Sybil's honest gradient if available
                if sybil_honest_gradients and sid in sybil_honest_gradients:
                    result[sid] = sybil_honest_gradients[sid]
                elif honest_gradients:
                    # Fall back to random honest client's gradient
                    fallback_id = list(honest_gradients.keys())[
                        self._rng.randint(len(honest_gradients))
                    ]
                    result[sid] = honest_gradients[fallback_id]
                else:
                    # Last resort: zeros
                    result[sid] = poison_template or []

        self._attack_log.append(log_entry)
        return result

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_attacking_ids(self, round_number: int) -> Set[int]:
        """Choose which Sybil IDs attack this round."""
        if self.strategy == self.STRATEGY_SYNCHRONIZED:
            return set(self.sybil_ids)

        if self.strategy == self.STRATEGY_ROTATING:
            k = max(1, int(len(self.sybil_ids) * self.rotation_fraction))
            chosen = self._rng.choice(self.sybil_ids, size=k, replace=False)
            return set(chosen.tolist())

        if self.strategy == self.STRATEGY_SLEEPER:
            if round_number < self.sleeper_rounds:
                return set()  # All stealth — no attacks yet
            return set(self.sybil_ids)  # All attack after sleeper phase

        return set()

    # ------------------------------------------------------------------
    # Poison computation
    # ------------------------------------------------------------------

    def _compute_poison(
        self,
        honest_gradients: Optional[Dict[int, List[torch.Tensor]]],
    ) -> Optional[List[torch.Tensor]]:
        """Craft a single poisoned gradient from the honest centroid.

        The poison is ``-attack_scale * centroid``, which is the most
        effective direction for untargeted model degradation while
        keeping all Sybil gradients identical (maximising their
        collective weight in any averaging-based aggregation).

        Parameters
        ----------
        honest_gradients : dict or None
            Honest client gradients used to derive the centroid.

        Returns
        -------
        list of Tensor or None
        """
        if not honest_gradients:
            return None

        honest_list = list(honest_gradients.values())
        if not honest_list:
            return None

        # Compute centroid
        centroid = average_gradients(honest_list)

        # Negate and scale — the coordinated poison
        poisoned: List[torch.Tensor] = []
        for g in centroid:
            if isinstance(g, torch.Tensor):
                poisoned.append(-self.attack_scale * g)
            else:
                poisoned.append(
                    torch.tensor(
                        -self.attack_scale * np.array(g, dtype=np.float32),
                        dtype=torch.float32,
                    )
                )

        return poisoned

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_attack_log(self) -> List[Dict]:
        """Return history of per-round attack decisions."""
        return list(self._attack_log)

    def get_info(self) -> Dict:
        """Summary information about the Sybil setup."""
        return {
            "num_sybils": len(self.sybil_ids),
            "sybil_ids": list(self.sybil_ids),
            "strategy": self.strategy,
            "attack_scale": self.attack_scale,
            "sleeper_rounds": self.sleeper_rounds if self.strategy == self.STRATEGY_SLEEPER else None,
            "rotation_fraction": self.rotation_fraction if self.strategy == self.STRATEGY_ROTATING else None,
            "rounds_logged": len(self._attack_log),
        }
