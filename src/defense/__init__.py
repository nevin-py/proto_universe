"""Defense mechanisms package for Protogalaxy ZK-FL.

Provides multi-layer Byzantine defense:
- Layer 2: Statistical anomaly detection
- Layer 3: Robust aggregation (Trimmed Mean / Multi-Krum)
- Layer 4: Reputation-based client filtering
"""

from .statistical import StatisticalDefenseLayer1, StatisticalDefenseLayer2
from .robust_agg import TrimmedMeanAggregator, MultiKrumAggregator
from .reputation import ReputationManager
from .coordinator import DefenseCoordinator

__all__ = [
    # Statistical detection
    'StatisticalDefenseLayer1',
    'StatisticalDefenseLayer2',
    # Robust aggregation
    'TrimmedMeanAggregator',
    'MultiKrumAggregator',
    # Reputation
    'ReputationManager',
    # Coordinator
    'DefenseCoordinator',
]
