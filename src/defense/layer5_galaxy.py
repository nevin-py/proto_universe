"""Layer 5: Galaxy-Level Defense (Architecture Section 4.5)

Implements galaxy-level anomaly detection and isolation:
- Treat each galaxy's aggregated update as a "super-client"
- Galaxy anomaly detection (norm-based, direction-based)
- Galaxy reputation system (EWMA at galaxy level)
- Adaptive re-clustering (dissolve compromised galaxies)
- Hierarchical isolation (4-tier system)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class IsolationLevel(Enum):
    """Hierarchical isolation levels (Architecture Section 4.5)"""
    NONE = "none"
    CLIENT_LEVEL = "client_level"  # Individual client quarantined
    PARTIAL_GALAXY = "partial_galaxy"  # Multiple clients quarantined
    FULL_GALAXY = "full_galaxy"  # Entire galaxy excluded
    SYSTEM_WIDE = "system_wide"  # Multiple galaxies compromised


@dataclass
class GalaxyAnomalyReport:
    """Report from galaxy anomaly detection"""
    galaxy_id: int
    is_anomalous: bool
    norm_deviation_score: float
    direction_similarity: float
    cross_galaxy_consistency: float
    failed_checks: List[str] = field(default_factory=list)
    isolation_recommendation: IsolationLevel = IsolationLevel.NONE


class GalaxyReputationManager:
    """Galaxy-level reputation tracking (Architecture Section 4.5)
    
    Galaxy reputation = weighted average of client reputations within galaxy
    Uses EWMA (Exponentially Weighted Moving Average) for temporal smoothing
    """
    
    def __init__(self, num_galaxies: int, decay_factor: float = 0.9):
        """Initialize galaxy reputation manager
        
        Args:
            num_galaxies: Total number of galaxies
            decay_factor: EWMA decay (0.9 means 90% history, 10% new)
        """
        self.num_galaxies = num_galaxies
        self.decay_factor = decay_factor
        self.galaxy_reputations = {g: 1.0 for g in range(num_galaxies)}
        self.history = []
        
        # Track consecutive low-reputation rounds per galaxy
        self.low_rep_streaks = {g: 0 for g in range(num_galaxies)}
        
    def update_from_client_reputations(
        self, 
        galaxy_id: int, 
        client_reputations: Dict[int, float]
    ):
        """Update galaxy reputation from constituent client reputations
        
        Args:
            galaxy_id: Galaxy identifier
            client_reputations: Dict mapping client_id to reputation score
        """
        if not client_reputations:
            return
        
        # Weighted average of client reputations
        avg_rep = np.mean(list(client_reputations.values()))
        
        # EWMA update
        current = self.galaxy_reputations.get(galaxy_id, 1.0)
        new_rep = self.decay_factor * current + (1 - self.decay_factor) * avg_rep
        self.galaxy_reputations[galaxy_id] = np.clip(new_rep, 0.0, 1.0)
        
        # Track streak
        if new_rep < 0.5:
            self.low_rep_streaks[galaxy_id] += 1
        else:
            self.low_rep_streaks[galaxy_id] = 0
        
        self.history.append({
            'galaxy_id': galaxy_id,
            'reputation': self.galaxy_reputations[galaxy_id],
            'avg_client_rep': avg_rep
        })
    
    def penalize_galaxy(self, galaxy_id: int, penalty: float = 0.2):
        """Directly penalize galaxy reputation
        
        Args:
            galaxy_id: Galaxy to penalize
            penalty: Amount to reduce reputation
        """
        current = self.galaxy_reputations.get(galaxy_id, 1.0)
        new_rep = max(0.0, current - penalty)
        self.galaxy_reputations[galaxy_id] = new_rep
        
        if new_rep < 0.5:
            self.low_rep_streaks[galaxy_id] += 1
    
    def get_reputation(self, galaxy_id: int) -> float:
        """Get galaxy reputation score"""
        return self.galaxy_reputations.get(galaxy_id, 1.0)
    
    def get_low_reputation_galaxies(self, threshold: float = 0.3) -> List[int]:
        """Get galaxies below reputation threshold
        
        Args:
            threshold: Reputation threshold
            
        Returns:
            List of galaxy IDs with low reputation
        """
        return [
            gid for gid, rep in self.galaxy_reputations.items()
            if rep < threshold
        ]
    
    def should_dissolve_galaxy(
        self, 
        galaxy_id: int, 
        consecutive_threshold: int = 3
    ) -> bool:
        """Check if galaxy should be dissolved (adaptive re-clustering)
        
        Args:
            galaxy_id: Galaxy to check
            consecutive_threshold: Number of consecutive low-rep rounds
            
        Returns:
            True if galaxy should be dissolved
        """
        return self.low_rep_streaks.get(galaxy_id, 0) >= consecutive_threshold


class GalaxyAnomalyDetector:
    """Detect anomalous galaxy updates (Architecture Section 4.5)
    
    Treats each galaxy's aggregated update as a "super-client" and applies
    similar statistical analysis as Layer 2 but at galaxy granularity.
    """
    
    def __init__(
        self,
        norm_threshold_sigma: float = 3.0,
        direction_threshold: float = 0.5,
        consistency_threshold: float = 0.7
    ):
        """Initialize galaxy anomaly detector
        
        Args:
            norm_threshold_sigma: Std deviations for norm outlier
            direction_threshold: Min cosine similarity to avg direction
            consistency_threshold: Min agreement with majority of galaxies
        """
        self.norm_threshold_sigma = norm_threshold_sigma
        self.direction_threshold = direction_threshold
        self.consistency_threshold = consistency_threshold
        
        self.detection_history = []
    
    def detect_galaxy_anomalies(
        self, 
        galaxy_updates: Dict[int, List[torch.Tensor]]
    ) -> Dict[int, GalaxyAnomalyReport]:
        """Detect anomalous galaxy updates
        
        Args:
            galaxy_updates: Dict mapping galaxy_id to aggregated gradients
            
        Returns:
            Dict mapping galaxy_id to anomaly report
        """
        if len(galaxy_updates) < 2:
            # Need at least 2 galaxies for comparison
            return {gid: GalaxyAnomalyReport(gid, False, 0.0, 1.0, 1.0) 
                    for gid in galaxy_updates.keys()}
        
        # Flatten galaxy gradients
        flattened = {}
        for gid, grads in galaxy_updates.items():
            flat = torch.cat([g.flatten() for g in grads]).cpu().numpy()
            flattened[gid] = flat
        
        reports = {}
        galaxy_ids = list(galaxy_updates.keys())
        
        # Compute statistics across all galaxies
        all_grads = np.array(list(flattened.values()))
        mean_grad = np.mean(all_grads, axis=0)
        
        for gid in galaxy_ids:
            grad = flattened[gid]
            
            # Check 1: Norm-based detection
            norm_score = self._check_norm_deviation(grad, all_grads)
            
            # Check 2: Direction-based detection
            direction_score = self._check_direction_similarity(grad, mean_grad)
            
            # Check 3: Cross-galaxy consistency
            consistency_score = self._check_cross_galaxy_consistency(
                grad, flattened, gid
            )
            
            # Determine if anomalous
            failed_checks = []
            if norm_score > self.norm_threshold_sigma:
                failed_checks.append("norm_deviation")
            if direction_score < self.direction_threshold:
                failed_checks.append("direction_anomaly")
            if consistency_score < self.consistency_threshold:
                failed_checks.append("low_consistency")
            
            is_anomalous = len(failed_checks) >= 2
            
            # Recommend isolation level
            isolation_rec = IsolationLevel.NONE
            if is_anomalous:
                if len(failed_checks) == 3:
                    isolation_rec = IsolationLevel.FULL_GALAXY
                else:
                    isolation_rec = IsolationLevel.PARTIAL_GALAXY
            
            reports[gid] = GalaxyAnomalyReport(
                galaxy_id=gid,
                is_anomalous=is_anomalous,
                norm_deviation_score=norm_score,
                direction_similarity=direction_score,
                cross_galaxy_consistency=consistency_score,
                failed_checks=failed_checks,
                isolation_recommendation=isolation_rec
            )
        
        self.detection_history.append(reports)
        return reports
    
    def _check_norm_deviation(self, grad: np.ndarray, all_grads: np.ndarray) -> float:
        """Check if gradient norm deviates from mean
        
        Returns:
            Number of standard deviations from mean
        """
        norms = np.linalg.norm(all_grads, axis=1)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        grad_norm = np.linalg.norm(grad)
        
        if std_norm < 1e-8:
            return 0.0
        
        return abs(grad_norm - mean_norm) / std_norm
    
    def _check_direction_similarity(
        self, 
        grad: np.ndarray, 
        mean_grad: np.ndarray
    ) -> float:
        """Check cosine similarity with mean gradient
        
        Returns:
            Cosine similarity score [0, 1]
        """
        norm_grad = np.linalg.norm(grad)
        norm_mean = np.linalg.norm(mean_grad)
        
        if norm_grad < 1e-8 or norm_mean < 1e-8:
            return 1.0
        
        cos_sim = np.dot(grad, mean_grad) / (norm_grad * norm_mean)
        return float(cos_sim)
    
    def _check_cross_galaxy_consistency(
        self,
        grad: np.ndarray,
        all_flattened: Dict[int, np.ndarray],
        current_gid: int
    ) -> float:
        """Check consistency with other galaxies
        
        Returns:
            Average cosine similarity with other galaxies
        """
        similarities = []
        for gid, other_grad in all_flattened.items():
            if gid == current_gid:
                continue
            
            norm_g = np.linalg.norm(grad)
            norm_o = np.linalg.norm(other_grad)
            
            if norm_g < 1e-8 or norm_o < 1e-8:
                continue
            
            cos_sim = np.dot(grad, other_grad) / (norm_g * norm_o)
            similarities.append(cos_sim)
        
        if not similarities:
            return 1.0
        
        return float(np.mean(similarities))


class AdaptiveReClusterer:
    """Adaptive re-clustering for compromised galaxies (Architecture Section 4.5)"""
    
    def __init__(self, num_galaxies: int):
        """Initialize re-clusterer
        
        Args:
            num_galaxies: Total number of galaxies
        """
        self.num_galaxies = num_galaxies
        self.dissolved_galaxies: Set[int] = set()
        self.client_reassignments: Dict[int, int] = {}  # client_id -> new_galaxy_id
    
    def dissolve_galaxy(
        self,
        galaxy_id: int,
        client_ids: List[int],
        client_reputations: Dict[int, float],
        malicious_threshold: float = 0.3
    ) -> Dict[str, any]:
        """Dissolve a compromised galaxy and redistribute clients
        
        Args:
            galaxy_id: Galaxy to dissolve
            client_ids: Clients in this galaxy
            client_reputations: Client reputation scores
            malicious_threshold: Threshold below which clients are quarantined
            
        Returns:
            Dict with reassignment info and quarantined clients
        """
        self.dissolved_galaxies.add(galaxy_id)
        
        # Separate honest from suspicious clients
        honest_clients = []
        quarantined_clients = []
        
        for cid in client_ids:
            rep = client_reputations.get(cid, 1.0)
            if rep >= malicious_threshold:
                honest_clients.append(cid)
            else:
                quarantined_clients.append(cid)
        
        # Reassign honest clients to other galaxies (round-robin)
        available_galaxies = [
            g for g in range(self.num_galaxies) 
            if g not in self.dissolved_galaxies
        ]
        
        if not available_galaxies:
            # All galaxies compromised - emergency scenario
            available_galaxies = [0]  # Reset to galaxy 0
        
        for i, cid in enumerate(honest_clients):
            new_galaxy = available_galaxies[i % len(available_galaxies)]
            self.client_reassignments[cid] = new_galaxy
        
        return {
            'dissolved_galaxy': galaxy_id,
            'honest_clients_reassigned': len(honest_clients),
            'clients_quarantined': len(quarantined_clients),
            'quarantined_client_ids': quarantined_clients,
            'reassignments': {
                cid: self.client_reassignments[cid] 
                for cid in honest_clients
            }
        }
    
    def get_client_galaxy(self, client_id: int, original_galaxy: int) -> int:
        """Get current galaxy assignment for client
        
        Args:
            client_id: Client ID
            original_galaxy: Original galaxy assignment
            
        Returns:
            Current galaxy assignment (may be reassigned)
        """
        return self.client_reassignments.get(client_id, original_galaxy)


class Layer5GalaxyDefense:
    """Complete Layer 5: Galaxy-Level Defense System
    
    Integrates:
    - Galaxy anomaly detection
    - Galaxy reputation management
    - Adaptive re-clustering
    - Hierarchical isolation
    """
    
    def __init__(
        self,
        num_galaxies: int,
        galaxy_rep_decay: float = 0.9,
        norm_threshold: float = 3.0,
        direction_threshold: float = 0.5,
        consistency_threshold: float = 0.7,
        dissolution_streak: int = 3
    ):
        """Initialize Layer 5 defense
        
        Args:
            num_galaxies: Total number of galaxies
            galaxy_rep_decay: EWMA decay for galaxy reputation
            norm_threshold: Sigma threshold for norm outliers
            direction_threshold: Min cosine similarity
            consistency_threshold: Min cross-galaxy agreement
            dissolution_streak: Consecutive low-rep rounds to trigger dissolution
        """
        self.reputation_manager = GalaxyReputationManager(
            num_galaxies, galaxy_rep_decay
        )
        self.anomaly_detector = GalaxyAnomalyDetector(
            norm_threshold, direction_threshold, consistency_threshold
        )
        self.re_clusterer = AdaptiveReClusterer(num_galaxies)
        self.dissolution_streak = dissolution_streak
        
        self.isolation_decisions = []
    
    def run_defense(
        self,
        galaxy_updates: Dict[int, List[torch.Tensor]],
        client_assignments: Dict[int, int],  # client_id -> galaxy_id
        client_reputations: Dict[int, float]
    ) -> Dict[str, any]:
        """Run complete Layer 5 defense pipeline
        
        Args:
            galaxy_updates: Dict mapping galaxy_id -> aggregated gradients
            client_assignments: Current client-to-galaxy mapping
            client_reputations: Client reputation scores
            
        Returns:
            Defense results including anomalies, isolations, reassignments
        """
        # Step 1: Detect galaxy anomalies
        anomaly_reports = self.anomaly_detector.detect_galaxy_anomalies(
            galaxy_updates
        )
        
        # Step 2: Update galaxy reputations
        for galaxy_id in galaxy_updates.keys():
            # Get clients in this galaxy
            galaxy_clients = {
                cid: rep for cid, rep in client_reputations.items()
                if client_assignments.get(cid) == galaxy_id
            }
            
            self.reputation_manager.update_from_client_reputations(
                galaxy_id, galaxy_clients
            )
            
            # Penalize if anomalous
            if anomaly_reports[galaxy_id].is_anomalous:
                self.reputation_manager.penalize_galaxy(galaxy_id, penalty=0.15)
        
        # Step 3: Check for galaxies to dissolve
        dissolve_actions = []
        for galaxy_id in galaxy_updates.keys():
            should_dissolve = self.reputation_manager.should_dissolve_galaxy(
                galaxy_id, self.dissolution_streak
            )
            
            if should_dissolve:
                galaxy_clients = [
                    cid for cid, gid in client_assignments.items()
                    if gid == galaxy_id
                ]
                
                action = self.re_clusterer.dissolve_galaxy(
                    galaxy_id, galaxy_clients, client_reputations
                )
                dissolve_actions.append(action)
        
        # Step 4: Determine system-wide isolation level
        anomalous_count = sum(1 for r in anomaly_reports.values() if r.is_anomalous)
        total_galaxies = len(galaxy_updates)
        
        if anomalous_count >= total_galaxies * 0.4:
            system_isolation = IsolationLevel.SYSTEM_WIDE
        elif dissolve_actions:
            system_isolation = IsolationLevel.FULL_GALAXY
        elif anomalous_count > 0:
            system_isolation = IsolationLevel.PARTIAL_GALAXY
        else:
            system_isolation = IsolationLevel.NONE
        
        # Compile results
        results = {
            'anomaly_reports': anomaly_reports,
            'galaxy_reputations': self.reputation_manager.galaxy_reputations.copy(),
            'flagged_galaxies': [
                gid for gid, report in anomaly_reports.items()
                if report.is_anomalous
            ],
            'dissolved_galaxies': dissolve_actions,
            'system_isolation_level': system_isolation,
            'verdicted_clean_galaxies': [
                gid for gid, report in anomaly_reports.items()
                if not report.is_anomalous
            ]
        }
        
        self.isolation_decisions.append(results)
        return results
    
    def get_active_galaxies(self) -> List[int]:
        """Get list of galaxies not dissolved
        
        Returns:
            List of active galaxy IDs
        """
        return [
            g for g in range(self.reputation_manager.num_galaxies)
            if g not in self.re_clusterer.dissolved_galaxies
        ]
