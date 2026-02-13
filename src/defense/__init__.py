"""Defense mechanisms package for Protogalaxy ZK-FL.

Provides multi-layer Byzantine defense:
- Layer 2: Statistical anomaly detection (3-metric analyzer per PROTO-302)
- Layer 3: Robust aggregation (Trimmed Mean / Multi-Krum)
- Layer 4: Reputation-based client filtering
- Layer 5: Galaxy-level defense (Architecture Section 4.5)
"""

from .statistical import (
    StatisticalDefenseLayer1, 
    StatisticalDefenseLayer2,
    StatisticalAnalyzer
)
from .robust_agg import TrimmedMeanAggregator, MultiKrumAggregator
from .fltrust import FLTrustAggregator
from .reputation import ReputationManager, EnhancedReputationManager, ClientStatus, BehaviorScore
from .layer5_galaxy import (
    Layer5GalaxyDefense,
    GalaxyAnomalyDetector,
    GalaxyReputationManager,
    AdaptiveReClusterer,
    IsolationLevel,
    GalaxyAnomalyReport
)
from .coordinator import DefenseCoordinator

__all__ = [
    # Statistical detection
    'StatisticalDefenseLayer1',
    'StatisticalDefenseLayer2',
    'StatisticalAnalyzer',  # 3-metric analyzer (PROTO-302)
    # Robust aggregation
    'TrimmedMeanAggregator',
    'MultiKrumAggregator',
    'FLTrustAggregator',
    # Reputation
    'ReputationManager',
    'EnhancedReputationManager',
    'ClientStatus',
    'BehaviorScore',
    # Layer 5: Galaxy-level defense
    'Layer5GalaxyDefense',
    'GalaxyAnomalyDetector',
    'GalaxyReputationManager',
    'AdaptiveReClusterer',
    'IsolationLevel',
    'GalaxyAnomalyReport',
    # Coordinator
    'DefenseCoordinator',
]

