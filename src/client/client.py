"""Main client orchestration and communication"""

from src.client.trainer import Trainer
from src.client.commitment import CommitmentGenerator
from src.client.verifier import ProofVerifier


class Client:
    """Main client class orchestrating local training and communication"""
    
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
    
    def attack(self, attack_type: str):
        """Apply Byzantine attack if configured"""
        if not self.is_byzantine:
            return
        
        if attack_type == "label_flip":
            # Implement label flip attack
            pass
        elif attack_type == "gradient_poison":
            # Implement gradient poisoning
            self.poison_gradients()
        elif attack_type == "model_inversion":
            # Implement model inversion attack
            pass
    
    def poison_gradients(self):
        """Poison gradients for Byzantine attack"""
        if self.gradients is not None:
            for i in range(len(self.gradients)):
                self.gradients[i] *= -10  # Extreme opposite direction
    
    def increment_round(self):
        """Move to next training round"""
        self.round_number += 1
