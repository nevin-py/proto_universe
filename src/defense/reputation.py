"""Reputation-based Byzantine detection and client filtering"""

import numpy as np


class ReputationManager:
    """Layer 4: Reputation-based client filtering"""
    
    def __init__(self, num_clients: int, decay_factor: float = 0.9):
        """Initialize reputation manager"""
        self.num_clients = num_clients
        self.decay_factor = decay_factor
        self.reputation_scores = {i: 1.0 for i in range(num_clients)}
        self.history = []
    
    def update_reputation(self, client_id: int, reward: float):
        """Update reputation score for a client"""
        current = self.reputation_scores.get(client_id, 1.0)
        # Decay and update
        new_score = self.decay_factor * current + (1 - self.decay_factor) * reward
        self.reputation_scores[client_id] = np.clip(new_score, 0.0, 1.0)
        self.history.append({
            'client_id': client_id,
            'score': self.reputation_scores[client_id],
            'reward': reward
        })
    
    def get_reputation(self, client_id: int) -> float:
        """Get reputation score for a client"""
        return self.reputation_scores.get(client_id, 1.0)
    
    def get_all_reputations(self) -> dict:
        """Get all reputation scores"""
        return self.reputation_scores.copy()
    
    def filter_clients(self, threshold: float = 0.5) -> list:
        """Get list of clients above reputation threshold"""
        return [cid for cid, score in self.reputation_scores.items() if score >= threshold]
    
    def penalize_client(self, client_id: int, penalty: float = 0.1):
        """Penalize a client for Byzantine behavior"""
        self.update_reputation(client_id, 1.0 - penalty)
    
    def reward_client(self, client_id: int, reward: float = 0.1):
        """Reward a well-behaved client"""
        self.update_reputation(client_id, 1.0 + reward)
    
    def get_history(self):
        """Get reputation change history"""
        return self.history
