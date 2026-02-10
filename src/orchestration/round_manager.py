"""Round management for federated learning training.

Provides comprehensive round lifecycle management including:
- Round state tracking
- Phase coordination
- Timeout handling
- Metrics collection per round
"""

import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging


class RoundPhase(Enum):
    """Phases within a single FL round"""
    INITIALIZATION = "initialization"
    COMMITMENT = "commitment"  # Clients submit commitments
    TRAINING = "training"  # Clients train locally
    SUBMISSION = "submission"  # Clients submit gradients
    VERIFICATION = "verification"  # Verify commitments
    DEFENSE = "defense"  # Run defense layers
    AGGREGATION = "aggregation"  # Aggregate gradients
    DISTRIBUTION = "distribution"  # Distribute new model
    EVALUATION = "evaluation"  # Evaluate global model
    COMPLETE = "complete"


@dataclass
class PhaseResult:
    """Result of a round phase"""
    phase: RoundPhase
    success: bool
    duration: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class RoundContext:
    """Context for a training round"""
    round_num: int
    start_time: float = 0.0
    current_phase: RoundPhase = RoundPhase.INITIALIZATION
    phase_results: Dict[RoundPhase, PhaseResult] = field(default_factory=dict)
    
    # Client tracking
    expected_clients: List[str] = field(default_factory=list)
    responded_clients: List[str] = field(default_factory=list)
    failed_clients: List[str] = field(default_factory=list)
    
    # Data collected during round
    commitments: Dict[str, str] = field(default_factory=dict)
    gradients: Dict[str, Any] = field(default_factory=dict)
    
    # Defense results
    detected_malicious: List[str] = field(default_factory=list)
    reputation_updates: Dict[str, float] = field(default_factory=dict)
    
    # Aggregation result
    aggregated_gradients: Optional[List] = None
    
    # Round metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def participation_rate(self) -> float:
        """Calculate client participation rate"""
        if not self.expected_clients:
            return 0.0
        return len(self.responded_clients) / len(self.expected_clients)
    
    @property
    def duration(self) -> float:
        """Get round duration so far"""
        return time.time() - self.start_time


class RoundManager:
    """Manages FL training round lifecycle.
    
    Coordinates the phases of each training round:
    1. Initialization - Setup round parameters
    2. Commitment - Collect client commitments (optional)
    3. Training - Clients train locally
    4. Submission - Collect gradient updates
    5. Verification - Verify against commitments
    6. Defense - Run Byzantine defense layers
    7. Aggregation - Aggregate verified gradients
    8. Distribution - Send updated model
    9. Evaluation - Evaluate new model
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize round manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger("round_manager")
        
        # Round tracking
        self.current_round = 0
        self.round_context: Optional[RoundContext] = None
        self.round_history: List[RoundContext] = []
        
        # Configuration
        self.phase_timeout = self.config.get('phase_timeout', 60.0)
        self.min_participation = self.config.get('min_participation', 0.8)
        self.enable_commitments = self.config.get('enable_commitments', True)
        
        # Phase handlers
        self._phase_handlers: Dict[RoundPhase, Callable] = {}
    
    def register_phase_handler(
        self,
        phase: RoundPhase,
        handler: Callable[[RoundContext], PhaseResult]
    ) -> None:
        """Register a handler for a round phase.
        
        Args:
            phase: Phase to handle
            handler: Handler function
        """
        self._phase_handlers[phase] = handler
    
    def start_round(self, expected_clients: List[str]) -> RoundContext:
        """Start a new training round.
        
        Args:
            expected_clients: List of client IDs expected to participate
            
        Returns:
            New RoundContext
        """
        self.current_round += 1
        
        self.round_context = RoundContext(
            round_num=self.current_round,
            start_time=time.time(),
            current_phase=RoundPhase.INITIALIZATION,
            expected_clients=expected_clients.copy()
        )
        
        self.logger.info(
            f"Starting round {self.current_round} with "
            f"{len(expected_clients)} expected clients"
        )
        
        return self.round_context
    
    def advance_phase(self, next_phase: RoundPhase) -> PhaseResult:
        """Advance to the next phase of the round.
        
        Args:
            next_phase: Phase to advance to
            
        Returns:
            Result of the previous phase
        """
        if self.round_context is None:
            raise RuntimeError("No active round")
        
        prev_phase = self.round_context.current_phase
        phase_start = time.time()
        
        # Execute phase handler if registered
        if prev_phase in self._phase_handlers:
            try:
                result = self._phase_handlers[prev_phase](self.round_context)
            except Exception as e:
                result = PhaseResult(
                    phase=prev_phase,
                    success=False,
                    duration=time.time() - phase_start,
                    errors=[str(e)]
                )
        else:
            # Default: success without custom handling
            result = PhaseResult(
                phase=prev_phase,
                success=True,
                duration=time.time() - phase_start
            )
        
        # Store result
        self.round_context.phase_results[prev_phase] = result
        
        # Move to next phase
        self.round_context.current_phase = next_phase
        
        self.logger.debug(
            f"Phase {prev_phase.value} -> {next_phase.value} "
            f"(duration: {result.duration:.2f}s)"
        )
        
        return result
    
    def record_commitment(self, client_id: str, commitment: str) -> bool:
        """Record a client's gradient commitment.
        
        Args:
            client_id: Client identifier
            commitment: Commitment hash
            
        Returns:
            True if recorded successfully
        """
        if self.round_context is None:
            return False
        
        if client_id not in self.round_context.expected_clients:
            self.logger.warning(f"Unexpected client {client_id}")
            return False
        
        self.round_context.commitments[client_id] = commitment
        return True
    
    def record_gradient(self, client_id: str, gradient: Any) -> bool:
        """Record a client's gradient submission.
        
        Args:
            client_id: Client identifier
            gradient: Gradient data
            
        Returns:
            True if recorded successfully
        """
        if self.round_context is None:
            return False
        
        if client_id not in self.round_context.expected_clients:
            self.logger.warning(f"Unexpected client {client_id}")
            return False
        
        self.round_context.gradients[client_id] = gradient
        
        if client_id not in self.round_context.responded_clients:
            self.round_context.responded_clients.append(client_id)
        
        return True
    
    def record_detection(self, client_id: str) -> None:
        """Record a client as potentially malicious.
        
        Args:
            client_id: Detected client ID
        """
        if self.round_context is None:
            return
        
        if client_id not in self.round_context.detected_malicious:
            self.round_context.detected_malicious.append(client_id)
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a metric for the current round.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if self.round_context is None:
            return
        
        self.round_context.metrics[name] = value
    
    def check_sufficient_participation(self) -> bool:
        """Check if enough clients have participated.
        
        Returns:
            True if minimum participation threshold met
        """
        if self.round_context is None:
            return False
        
        return self.round_context.participation_rate >= self.min_participation
    
    def get_verified_gradients(self) -> Dict[str, Any]:
        """Get gradients from verified (non-malicious) clients.
        
        Returns:
            Dictionary of client_id -> gradient for clean clients
        """
        if self.round_context is None:
            return {}
        
        return {
            client_id: grad
            for client_id, grad in self.round_context.gradients.items()
            if client_id not in self.round_context.detected_malicious
        }
    
    def end_round(self) -> RoundContext:
        """End the current round and archive it.
        
        Returns:
            Completed RoundContext
        """
        if self.round_context is None:
            raise RuntimeError("No active round")
        
        # Final phase transition
        self.advance_phase(RoundPhase.COMPLETE)
        
        # Calculate final metrics
        self.round_context.metrics['duration'] = self.round_context.duration
        self.round_context.metrics['participation_rate'] = self.round_context.participation_rate
        self.round_context.metrics['num_detections'] = len(self.round_context.detected_malicious)
        
        # Archive
        completed = self.round_context
        self.round_history.append(completed)
        self.round_context = None
        
        self.logger.info(
            f"Round {completed.round_num} complete. "
            f"Duration: {completed.duration:.2f}s, "
            f"Participation: {completed.participation_rate:.1%}, "
            f"Detections: {len(completed.detected_malicious)}"
        )
        
        return completed
    
    def get_round_summary(self, round_num: Optional[int] = None) -> Dict:
        """Get summary of a round.
        
        Args:
            round_num: Round number (None = current/last)
            
        Returns:
            Summary dictionary
        """
        if round_num is None:
            context = self.round_context or (
                self.round_history[-1] if self.round_history else None
            )
        else:
            context = next(
                (r for r in self.round_history if r.round_num == round_num),
                None
            )
        
        if context is None:
            return {}
        
        return {
            'round_num': context.round_num,
            'duration': context.duration,
            'current_phase': context.current_phase.value,
            'expected_clients': len(context.expected_clients),
            'responded_clients': len(context.responded_clients),
            'participation_rate': context.participation_rate,
            'detected_malicious': context.detected_malicious,
            'metrics': context.metrics
        }
    
    def get_history_metrics(self) -> List[Dict]:
        """Get metrics from all completed rounds.
        
        Returns:
            List of metric dictionaries per round
        """
        return [
            {
                'round': r.round_num,
                **r.metrics
            }
            for r in self.round_history
        ]
