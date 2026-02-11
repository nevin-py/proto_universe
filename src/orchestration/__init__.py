"""Orchestration package for federated learning coordination.

Provides:
- FL Coordinator for round management
- Round Manager for phase tracking
- Model Synchronizer for model distribution (PROTO-1104)
- ProtoGalaxy Pipeline for end-to-end FL execution
"""

from src.orchestration.coordinator import FLCoordinator
from src.orchestration.round_manager import (
    RoundManager,
    RoundPhase,
    RoundContext,
    PhaseResult
)
from src.orchestration.model_sync import (
    ModelSynchronizer,
    ModelVersion,
    ClientModelReceiver,
    compute_model_hash,
    compute_state_dict_hash
)
from src.orchestration.pipeline import (
    ProtoGalaxyPipeline,
    ClientSubmission,
    GalaxySubmission
)

__all__ = [
    'FLCoordinator',
    'RoundManager',
    'RoundPhase',
    'RoundContext',
    'PhaseResult',
    # Model synchronization (PROTO-1104)
    'ModelSynchronizer',
    'ModelVersion',
    'ClientModelReceiver',
    'compute_model_hash',
    'compute_state_dict_hash',
    # Complete ProtoGalaxy pipeline
    'ProtoGalaxyPipeline',
    'ClientSubmission',
    'GalaxySubmission'
]
