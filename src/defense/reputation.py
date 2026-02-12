"""Enhanced Reputation-based Byzantine Detection (Architecture Section 4.4)

Implements:
- Multi-indicator behavior scoring model
- Quarantine and ban thresholds
- Permanent ban with evidence tracking
- Rehabilitation mechanism
"""

import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


class ClientStatus(Enum):
    """Client status levels"""
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    BANNED = "banned"


@dataclass
class BehaviorScore:
    """Multi-indicator behavior scoring (Architecture Section 4.4)
    
    B(t) = w1·I_integrity + w2·I_statistical + w3·I_krum + w4·I_historical
    Weights: w1=0.1, w2=0.3, w3=0.4, w4=0.2
    """
    integrity_indicator: float  # Layer 1: Crypto verification pass rate
    statistical_indicator: float  # Layer 2: Statistical test pass rate
    krum_indicator: float  # Layer 3: Robust aggregation selection rate
    historical_indicator: float  # Layer 4: Historical reputation
    
    def compute_weighted_score(self) -> float:
        """Compute weighted behavior score"""
        w1, w2, w3, w4 = 0.1, 0.3, 0.4, 0.2
        score = (
            w1 * self.integrity_indicator +
            w2 * self.statistical_indicator +
            w3 * self.krum_indicator +
            w4 * self.historical_indicator
        )
        return np.clip(score, 0.0, 1.0)


class EnhancedReputationManager:
    """Enhanced Layer 4: Reputation with Quarantine System
    
    Features:
    - Multi-indicator behavior scoring
    - Quarantine threshold (θ_quarantine = 0.2)
    - Ban threshold (θ_ban = 0.1)
    - Permanent bans with evidence
    - Rehabilitation tracking
    """
    
    def __init__(
        self, 
        num_clients: int, 
        decay_factor: float = 0.9,
        quarantine_threshold: float = 0.2,
        ban_threshold: float = 0.1,
        rehabilitation_rounds: int = 10
    ):
        """Initialize enhanced reputation manager
        
        Args:
            num_clients: Total number of clients
            decay_factor: EWMA decay for historical reputation
            quarantine_threshold: Threshold below which clients quarantined
            ban_threshold: Threshold below which clients permanently banned
            rehabilitation_rounds: Rounds of good behavior to exit quarantine
        """
        self.num_clients = num_clients
        self.decay_factor = decay_factor
        self.quarantine_threshold = quarantine_threshold
        self.ban_threshold = ban_threshold
        self.rehabilitation_rounds = rehabilitation_rounds
        
        # Architecture Section 4.4: "Initialize: R_i(0) = 0.5 (neutral)"
        self.reputation_scores = {i: 0.5 for i in range(num_clients)}
        
        # Client statuses
        self.client_status: Dict[int, ClientStatus] = {
            i: ClientStatus.ACTIVE for i in range(num_clients)
        }
        
        # Banned clients (permanent)
        self.banned_clients: Set[int] = set()
        
        # Quarantine tracking
        self.quarantine_start_round: Dict[int, int] = {}
        self.good_behavior_streak: Dict[int, int] = {}
        
        # Per-layer indicators (for behavior scoring)
        self.layer_indicators: Dict[int, Dict[str, float]] = {
            i: {
                'integrity': 0.5,
                'statistical': 0.5,
                'krum': 0.5,
                'historical': 0.5
            } for i in range(num_clients)
        }
        
        self.history = []
    
    def update_behavior_score(
        self,
        client_id: int,
        layer_results: Dict[str, bool],
        round_number: int
    ):
        """Update client behavior score from defense layer results
        
        Args:
            client_id: Client ID
            layer_results: Dict with keys 'integrity', 'statistical', 'krum'
                          Values are True if layer passed, False if failed
            round_number: Current FL round
        """
        # Update per-layer indicators (EWMA)
        indicators = self.layer_indicators.get(client_id, {
            'integrity': 0.5,
            'statistical': 0.5,
            'krum': 0.5,
            'historical': 0.5
        })
        
        # Layer 1: Integrity
        integrity_pass = layer_results.get('integrity', True)
        indicators['integrity'] = (
            self.decay_factor * indicators['integrity'] +
            (1 - self.decay_factor) * (1.0 if integrity_pass else 0.0)
        )
        
        # Layer 2: Statistical
        statistical_pass = layer_results.get('statistical', True)
        indicators['statistical'] = (
            self.decay_factor * indicators['statistical'] +
            (1 - self.decay_factor) * (1.0 if statistical_pass else 0.0)
        )
        
        # Layer 3: Krum/Robust aggregation
        krum_selected = layer_results.get('krum', True)
        indicators['krum'] = (
            self.decay_factor * indicators['krum'] +
            (1 - self.decay_factor) * (1.0 if krum_selected else 0.0)
        )
        
        # Layer 4: Historical (current reputation)
        indicators['historical'] = self.reputation_scores.get(client_id, 0.5)
        
        self.layer_indicators[client_id] = indicators
        
        # Compute weighted behavior score
        behavior = BehaviorScore(
            integrity_indicator=indicators['integrity'],
            statistical_indicator=indicators['statistical'],
            krum_indicator=indicators['krum'],
            historical_indicator=indicators['historical']
        )
        new_score = behavior.compute_weighted_score()
        
        # Architecture Section 4.4: "R(t+1) = (1-λ)·R(t) + λ·B(t), λ = 0.1"
        # Apply EWMA at the top-level reputation score so that reputation
        # changes gradually rather than being directly overwritten.
        lambda_lr = 1.0 - self.decay_factor  # decay_factor = 0.9 → λ = 0.1
        old_score = self.reputation_scores.get(client_id, 0.5)
        smoothed_score = (1.0 - lambda_lr) * old_score + lambda_lr * new_score
        self.reputation_scores[client_id] = smoothed_score
        
        # Check for status changes
        self._update_client_status(client_id, smoothed_score, round_number)
        
        self.history.append({
            'round': round_number,
            'client_id': client_id,
            'old_score': old_score,
            'new_score': smoothed_score,
            'behavior_score': new_score,
            'status': self.client_status[client_id].value,
            'layer_results': layer_results.copy()
        })
    
    def _update_client_status(
        self, 
        client_id: int, 
        score: float, 
        round_number: int
    ):
        """Update client status based on reputation score
        
        Args:
            client_id: Client ID
            score: Current reputation score
            round_number: Current round
        """
        current_status = self.client_status.get(client_id, ClientStatus.ACTIVE)
        
        # Check for permanent ban
        if score <= self.ban_threshold:
            self.client_status[client_id] = ClientStatus.BANNED
            self.banned_clients.add(client_id)
            return
        
        # Check for quarantine
        if score <= self.quarantine_threshold:
            if current_status == ClientStatus.ACTIVE:
                # Enter quarantine
                self.client_status[client_id] = ClientStatus.QUARANTINED
                self.quarantine_start_round[client_id] = round_number
                self.good_behavior_streak[client_id] = 0
            elif current_status == ClientStatus.QUARANTINED:
                # Already quarantined - check for rehabilitation
                if score > self.quarantine_threshold:
                    # Improving
                    self.good_behavior_streak[client_id] = \
                        self.good_behavior_streak.get(client_id, 0) + 1
                    
                    if self.good_behavior_streak[client_id] >= self.rehabilitation_rounds:
                        # Rehabilitated!
                        self.client_status[client_id] = ClientStatus.ACTIVE
                        del self.quarantine_start_round[client_id]
                        del self.good_behavior_streak[client_id]
                else:
                    # Still low reputation
                    self.good_behavior_streak[client_id] = 0
        else:
            # Score is good
            if current_status == ClientStatus.QUARANTINED:
                # Track good behavior for rehabilitation
                self.good_behavior_streak[client_id] = \
                    self.good_behavior_streak.get(client_id, 0) + 1
                
                if self.good_behavior_streak[client_id] >= self.rehabilitation_rounds:
                    # Rehabilitated!
                    self.client_status[client_id] = ClientStatus.ACTIVE
                    del self.quarantine_start_round[client_id]
                    del self.good_behavior_streak[client_id]
            elif current_status == ClientStatus.ACTIVE:
                # Maintain active status
                pass
    
    def get_reputation(self, client_id: int) -> float:
        """Get reputation score for a client"""
        return self.reputation_scores.get(client_id, 0.5)
    
    def get_all_reputations(self) -> Dict[int, float]:
        """Get all reputation scores"""
        return self.reputation_scores.copy()
    
    def get_client_status(self, client_id: int) -> ClientStatus:
        """Get client status (active/quarantined/banned)"""
        return self.client_status.get(client_id, ClientStatus.ACTIVE)
    
    def is_banned(self, client_id: int) -> bool:
        """Check if client is permanently banned"""
        return client_id in self.banned_clients
    
    def is_quarantined(self, client_id: int) -> bool:
        """Check if client is quarantined"""
        return self.client_status.get(client_id) == ClientStatus.QUARANTINED
    
    def is_active(self, client_id: int) -> bool:
        """Check if client is active (not quarantined or banned)"""
        return self.client_status.get(client_id, ClientStatus.ACTIVE) == ClientStatus.ACTIVE
    
    def get_active_clients(self) -> List[int]:
        """Get list of active (non-banned, non-quarantined) clients"""
        return [
            cid for cid, status in self.client_status.items()
            if status == ClientStatus.ACTIVE
        ]
    
    def get_quarantined_clients(self) -> List[int]:
        """Get list of quarantined clients"""
        return [
            cid for cid, status in self.client_status.items()
            if status == ClientStatus.QUARANTINED
        ]
    
    def get_banned_clients(self) -> List[int]:
        """Get list of permanently banned clients"""
        return list(self.banned_clients)
    
    def filter_clients(self, threshold: float = None) -> List[int]:
        """Get list of clients above reputation threshold
        
        Args:
            threshold: Reputation threshold (uses quarantine_threshold if None)
        
        Returns:
            List of client IDs
        """
        if threshold is None:
            threshold = self.quarantine_threshold
        
        return [
            cid for cid, score in self.reputation_scores.items()
            if score >= threshold and not self.is_banned(cid)
        ]
    
    def penalize_client(self, client_id: int, penalty: float = 0.1):
        """Directly penalize a client
        
        Args:
            client_id: Client to penalize
            penalty: Amount to reduce reputation
        """
        current = self.reputation_scores.get(client_id, 0.5)
        new_score = max(0.0, current - penalty)
        self.reputation_scores[client_id] = new_score
    
    def reward_client(self, client_id: int, reward: float = 0.1):
        """Directly reward a client
        
        Args:
            client_id: Client to reward
            reward: Amount to increase reputation
        """
        current = self.reputation_scores.get(client_id, 0.5)
        new_score = min(1.0, current + reward)
        self.reputation_scores[client_id] = new_score
    
    def get_history(self) -> List[Dict]:
        """Get reputation change history"""
        return self.history
    
    def get_statistics(self) -> Dict:
        """Get reputation system statistics"""
        return {
            'total_clients': self.num_clients,
            'active_clients': len(self.get_active_clients()),
            'quarantined_clients': len(self.get_quarantined_clients()),
            'banned_clients': len(self.banned_clients),
            'avg_reputation': np.mean(list(self.reputation_scores.values())),
            'min_reputation': np.min(list(self.reputation_scores.values())),
            'max_reputation': np.max(list(self.reputation_scores.values())),
            'quarantine_threshold': self.quarantine_threshold,
            'ban_threshold': self.ban_threshold
        }


# Backward compatibility alias
class ReputationManager(EnhancedReputationManager):
    """Alias for backward compatibility"""
    pass
