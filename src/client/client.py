"""Main client orchestration and communication (Architecture Section 3, 6.2)"""

import torch
import numpy as np
from typing import Optional, List, Dict
from src.client.trainer import Trainer
from src.client.commitment import CommitmentGenerator
from src.client.verifier import ProofVerifier


class Client:
    """Main client class orchestrating local training and communication.
    
    Implements 5 attack types from Architecture Section 6.2:
    1. Label flipping: negate gradients (simulates class-swap confusion)
    2. Targeted label flipping: flip gradient direction for specific layers
    3. Backdoor injection: subtle scaled perturbation + trigger pattern
    4. Model poisoning: amplify gradients to maximize loss
    5. Gaussian noise: add calibrated noise to gradients
    """
    
    def __init__(self, client_id: int, model, is_byzantine: bool = False):
        """Initialize client"""
        self.client_id = client_id
        self.model = model
        self.is_byzantine = is_byzantine
        self.trainer = Trainer(model)
        self.commitment_gen = CommitmentGenerator(client_id)
        self.verifier = ProofVerifier()
        self.round_number = 0
        self.gradients = None
    
    def train_local(self, train_loader, num_epochs: int):
        """Perform local training"""
        loss = self.trainer.train(train_loader, num_epochs)
        self.gradients = self.trainer.get_gradients()
        return loss
    
    def evaluate_local(self, test_loader):
        """Evaluate model on local test data"""
        return self.trainer.evaluate(test_loader)
    
    def generate_commitment(self):
        """Generate commitment for current gradients"""
        if self.gradients is not None:
            return self.commitment_gen.generate_commitment(self.gradients)
        return None
    
    def receive_model_update(self, weights):
        """Receive and apply model update"""
        self.trainer.set_weights(weights)
    
    def get_update(self):
        """Prepare update for submission"""
        return {
            'client_id': self.client_id,
            'round': self.round_number,
            'gradients': self.gradients,
            'commitment': self.commitment_gen.get_all_commitments()[-1] if self.commitment_gen.get_all_commitments() else None
        }
    
    def verify_merkle_root(
        self,
        published_root: str,
        own_commitment_hash: str,
        proof: list,
        leaf_index: int = 0
    ) -> bool:
        """Verify published Merkle root is consistent with own commitment.
        
        Architecture Section 3.4 Phase 2 Step 1: Clients verify R_g(r) and R(r)
        are published and consistent with their committed values.
        
        Args:
            published_root: The Merkle root published by the galaxy/global
            own_commitment_hash: This client's commitment hash (leaf)
            proof: Merkle proof path from leaf to root
            leaf_index: Index of the leaf in the Merkle tree
            
        Returns:
            True if published root is consistent with commitment
        """
        return self.verifier.verify_proof(
            root=published_root,
            proof=proof,
            leaf_hash=own_commitment_hash,
            leaf_index=leaf_index
        )
    
    def attack(self, attack_type: str, **kwargs):
        """Apply Byzantine attack if configured (Architecture Section 6.2).
        
        Args:
            attack_type: One of 'label_flip', 'targeted_label_flip',
                        'backdoor', 'model_poisoning', 'gaussian_noise'
            **kwargs: Attack-specific parameters
        """
        if not self.is_byzantine or self.gradients is None:
            return
        
        if attack_type == "label_flip":
            self._attack_label_flip()
        elif attack_type == "targeted_label_flip":
            target_layers = kwargs.get('target_layers', None)
            self._attack_targeted_label_flip(target_layers)
        elif attack_type == "gradient_poison" or attack_type == "model_poisoning":
            scale = kwargs.get('scale', 10.0)
            self._attack_model_poisoning(scale)
        elif attack_type == "backdoor":
            scale = kwargs.get('scale', 0.1)
            trigger_seed = kwargs.get('trigger_seed', 42)
            self._attack_backdoor(scale, trigger_seed)
        elif attack_type == "gaussian_noise":
            noise_std = kwargs.get('noise_std', 1.0)
            self._attack_gaussian_noise(noise_std)
    
    def _attack_label_flip(self):
        """Attack 1: Label flipping — negate all gradients.
        Simulates clients that trained on swapped labels (0↔9, 1↔8, etc.).
        Effect: gradients point opposite to honest direction.
        """
        for i in range(len(self.gradients)):
            self.gradients[i] = -self.gradients[i]
    
    def _attack_targeted_label_flip(self, target_layers: Optional[List[int]] = None):
        """Attack 2: Targeted label flipping — negate specific layers only.
        More subtle than full label flip; harder to detect statistically.
        
        Args:
            target_layers: Indices of gradient layers to flip. If None, flips last 2 layers.
        """
        if target_layers is None:
            # Default: flip the last 2 layers (typically classifier head)
            target_layers = list(range(max(0, len(self.gradients) - 2), len(self.gradients)))
        
        for i in target_layers:
            if i < len(self.gradients):
                self.gradients[i] = -self.gradients[i]
    
    def _attack_backdoor(self, scale: float = 0.1, trigger_seed: int = 42):
        """Attack 3: Backdoor injection — subtle perturbation with trigger pattern.
        Adds a small, consistent trigger pattern to gradients so the global model
        learns a backdoor trigger while maintaining normal accuracy.
        
        Args:
            scale: Magnitude of the trigger perturbation (small = stealthy)
            trigger_seed: Fixed seed for reproducible trigger pattern
        """
        rng = np.random.RandomState(trigger_seed)
        for i in range(len(self.gradients)):
            g = self.gradients[i]
            if isinstance(g, torch.Tensor):
                trigger = torch.tensor(
                    rng.randn(*g.shape).astype(np.float32),
                    device=g.device, dtype=g.dtype
                )
                self.gradients[i] = g + scale * trigger
            else:
                trigger = rng.randn(*np.array(g).shape).astype(np.float32)
                self.gradients[i] = g + scale * trigger
    
    def _attack_model_poisoning(self, scale: float = 10.0):
        """Attack 4: Model poisoning — amplify gradients to maximize disruption.
        Scales all gradients by a large negative factor to push the model away
        from convergence.
        
        Args:
            scale: Amplification factor. Larger = more disruptive but easier to detect.
        """
        for i in range(len(self.gradients)):
            self.gradients[i] = self.gradients[i] * (-scale)
    
    def _attack_gaussian_noise(self, noise_std: float = 1.0):
        """Attack 5: Gaussian noise — add calibrated random noise.
        Adds zero-mean Gaussian noise to each gradient tensor. The noise standard
        deviation is calibrated relative to the gradient norm for stealth.
        
        Args:
            noise_std: Standard deviation multiplier for the noise.
        """
        for i in range(len(self.gradients)):
            g = self.gradients[i]
            if isinstance(g, torch.Tensor):
                noise = torch.randn_like(g) * noise_std
                self.gradients[i] = g + noise
            else:
                noise = np.random.randn(*np.array(g).shape).astype(np.float32) * noise_std
                self.gradients[i] = g + noise
    
    # Keep backward-compatible alias
    def poison_gradients(self):
        """Poison gradients for Byzantine attack (legacy alias)"""
        self._attack_model_poisoning(scale=10.0)
    
    def increment_round(self):
        """Move to next training round"""
        self.round_number += 1
