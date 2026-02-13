"""FLTrust Byzantine-robust aggregation (Cao et al., NDSS 2021).

The server maintains a small clean root dataset and trains a reference
model each round.  Client updates are weighted by the ReLU-clipped cosine
similarity between their flattened gradient and the server reference
gradient, then normalised to the server update's norm before weighted
averaging.

Reference
---------
Cao, X., Fang, M., Liu, J., & Gong, N. Z. (2021).
FLTrust: Byzantine-robust federated learning via trust bootstrapping.
*NDSS 2021*.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.gradient_ops import (
    flatten_gradients,
    gradient_cosine_similarity,
    compute_gradient_norm,
)

logger = logging.getLogger("protogalaxy.defense.fltrust")


class FLTrustAggregator:
    """Trust-bootstrapped aggregation using a server root dataset.

    Algorithm (per round)
    ---------------------
    1. Server trains its own copy of the global model on the clean root
       dataset for ``server_epochs`` epochs → compute server gradient
       update ``g_s``.
    2. For each client update ``g_i``:
       a. ``cos_i = cosine_similarity(flatten(g_i), flatten(g_s))``
       b. ``ts_i  = max(0, cos_i)``          (ReLU clipping)
       c. ``g_i'  = ts_i * (||g_s|| / ||g_i||) * g_i``  (norm-scale to
          server magnitude and weight by trust score)
    3. ``g_global = Σ g_i' / Σ ts_i``

    Parameters
    ----------
    server_dataset : Dataset
        Small clean dataset (100–1000 samples).
    server_model : nn.Module
        Copy of the global model; retrained from global weights each round.
    learning_rate : float
        Server-side SGD learning rate.
    batch_size : int
        Server-side mini-batch size.
    server_epochs : int
        Number of server-side local epochs per round.
    device : str or None
        Device for server training (auto-detected if *None*).
    """

    def __init__(
        self,
        server_dataset: Dataset,
        server_model: nn.Module,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        server_epochs: int = 1,
        device: Optional[str] = None,
    ):
        self.server_dataset = server_dataset
        self.server_model = server_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.server_epochs = server_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute the server DataLoader once
        self._server_loader = DataLoader(
            server_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Most recent server gradient and trust scores (for diagnostics)
        self._server_gradient: Optional[List[torch.Tensor]] = None
        self._trust_scores: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_server_update(
        self,
        global_model: nn.Module,
        num_epochs: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Train server model on clean data and return gradient update.

        The server model is reset to ``global_model`` weights at the start
        of each round, then trained for ``num_epochs`` epochs.  The gradient
        is ``w_before - w_after`` (the same convention used by the rest of
        the pipeline).

        Parameters
        ----------
        global_model : nn.Module
            Current global model (read-only — weights are copied).
        num_epochs : int or None
            Override ``self.server_epochs`` if given.

        Returns
        -------
        list of Tensor
            Per-parameter gradient tensors (same order as
            ``global_model.parameters()``).
        """
        epochs = num_epochs if num_epochs is not None else self.server_epochs

        # Reset server model to current global weights
        src_state = global_model.state_dict()
        self.server_model.load_state_dict(copy.deepcopy(src_state))
        self.server_model.to(self.device)

        # Store pre-training weights
        w_before = [p.data.clone() for p in self.server_model.parameters()]

        # Train
        self.server_model.train()
        optimizer = torch.optim.SGD(
            self.server_model.parameters(),
            lr=self.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for data, labels in self._server_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.server_model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Gradient = w_before - w_after
        w_after = [p.data.clone() for p in self.server_model.parameters()]
        server_grad = [
            (wb - wa).detach().cpu() for wb, wa in zip(w_before, w_after)
        ]
        self._server_gradient = server_grad
        return server_grad

    def aggregate(self, updates: List[Dict]) -> Optional[Dict]:
        """Trust-weighted aggregation of client updates.

        Parameters
        ----------
        updates : list of dict
            Each dict must contain ``'gradients'`` (list of Tensor/ndarray).
            Optionally ``'client_id'``.

        Returns
        -------
        dict or None
            ``{'gradients': aggregated, 'trust_scores': {...}, 'method': 'fltrust'}``
            or *None* if aggregation cannot proceed.
        """
        if not updates:
            logger.warning("FLTrust: no updates to aggregate.")
            return None

        if self._server_gradient is None:
            logger.error(
                "FLTrust: compute_server_update() must be called before aggregate()."
            )
            return None

        server_grad = self._server_gradient

        # Flatten server gradient
        server_flat = flatten_gradients(server_grad)
        server_norm = float(np.linalg.norm(server_flat))
        if server_norm < 1e-12:
            logger.warning("FLTrust: server gradient has near-zero norm; using simple average.")
            # Fall back to simple average
            avg = [
                torch.zeros_like(torch.as_tensor(server_grad[j]))
                for j in range(len(server_grad))
            ]
            for upd in updates:
                grads = upd["gradients"]
                for j, g in enumerate(grads):
                    avg[j] = avg[j] + (
                        g if isinstance(g, torch.Tensor) else torch.tensor(g, dtype=torch.float32)
                    )
            avg = [a / len(updates) for a in avg]
            flat_avg = flatten_gradients(avg)
            return {
                "gradients": flat_avg if isinstance(flat_avg, np.ndarray) else flat_avg,
                "trust_scores": {i: 1.0 for i in range(len(updates))},
                "method": "fltrust",
            }

        # ------------------------------------------------------------------
        # Per-client trust scoring
        # ------------------------------------------------------------------
        trust_scores: Dict[int, float] = {}
        client_grads_flat: List[np.ndarray] = []

        for idx, upd in enumerate(updates):
            client_id = upd.get("client_id", idx)
            grads = upd["gradients"]
            c_flat = flatten_gradients(grads)
            client_grads_flat.append(c_flat)

            # Cosine similarity with server gradient
            cos_sim = float(gradient_cosine_similarity(grads, server_grad))

            # ReLU clipping
            ts = max(0.0, cos_sim)
            trust_scores[client_id] = ts

        # ------------------------------------------------------------------
        # Trust-weighted aggregation with norm normalisation
        # ------------------------------------------------------------------
        total_ts = sum(trust_scores.values())
        if total_ts < 1e-12:
            logger.warning(
                "FLTrust: all trust scores ≈ 0; all clients deemed adversarial."
            )
            return {
                "gradients": np.zeros_like(server_flat),
                "trust_scores": trust_scores,
                "method": "fltrust",
            }

        # Compute weighted sum: g_i' = ts_i * (server_norm / client_norm) * g_i
        weighted_sum = np.zeros_like(server_flat)
        for idx, (cid, ts) in enumerate(trust_scores.items()):
            if ts < 1e-12:
                continue  # skip zero-trust clients
            c_flat = client_grads_flat[idx]
            c_norm = float(np.linalg.norm(c_flat))
            if c_norm < 1e-12:
                continue
            # Normalise client to server norm, then weight by trust
            normalised = (server_norm / c_norm) * c_flat
            weighted_sum += ts * normalised

        aggregated = weighted_sum / total_ts

        self._trust_scores = trust_scores
        logger.debug(
            "FLTrust aggregated %d updates, trust scores: %s",
            len(updates),
            {k: f"{v:.3f}" for k, v in trust_scores.items()},
        )

        return {
            "gradients": aggregated,
            "trust_scores": trust_scores,
            "method": "fltrust",
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_trust_scores(self) -> Dict[int, float]:
        """Return trust scores from the most recent aggregation round."""
        return dict(self._trust_scores)

    def get_server_gradient(self) -> Optional[List[torch.Tensor]]:
        """Return the most recent server gradient (for inspection)."""
        return self._server_gradient
