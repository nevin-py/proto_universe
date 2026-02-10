"""FL Server for coordinating federated learning rounds.

Central server component that manages FL training rounds,
collects gradients, and distributes model updates.
"""

import time
import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn

from src.communication.message import (
    Message, MessageType, 
    create_model_update_msg, create_round_start_msg, create_round_end_msg
)
from src.communication.channel import InMemoryChannel, MessageBuffer


class RoundState(Enum):
    """State of a training round"""
    IDLE = "idle"
    WAITING_FOR_GRADIENTS = "waiting_for_gradients"
    AGGREGATING = "aggregating"
    DISTRIBUTING = "distributing"
    COMPLETE = "complete"


@dataclass
class RoundInfo:
    """Information about a training round"""
    round_num: int
    state: RoundState = RoundState.IDLE
    start_time: float = 0.0
    end_time: float = 0.0
    num_participants: int = 0
    num_received: int = 0
    gradients_received: Dict[str, Any] = field(default_factory=dict)
    aggregated_gradients: Optional[List] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class FLServer:
    """Federated Learning Server.
    
    Manages FL training rounds including:
    - Client registration and tracking
    - Round coordination (start, wait, aggregate, distribute)
    - Model management and distribution
    - Progress tracking and metrics collection
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_expected_clients: int,
        config: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize FL Server.
        
        Args:
            model: Global model to train
            num_expected_clients: Expected number of clients
            config: Server configuration
            logger: Logger instance
        """
        self.model = model
        self.num_expected_clients = num_expected_clients
        self.config = config or {}
        self.logger = logger or logging.getLogger("fl_server")
        
        # Communication
        self.channel = InMemoryChannel("global", self.logger)
        
        # Client management
        self.registered_clients: Dict[str, Dict] = {}
        self.galaxy_mapping: Dict[str, str] = {}  # client_id -> galaxy_id
        
        # Round management
        self.current_round = 0
        self.round_info: Optional[RoundInfo] = None
        self.round_history: List[RoundInfo] = []
        
        # Configuration
        self.round_timeout = self.config.get('round_timeout', 60.0)
        self.min_clients_ratio = self.config.get('min_clients_ratio', 0.8)
        
        # Aggregation function (can be customized)
        self._aggregation_fn: Optional[Callable] = None
    
    def register_client(
        self, 
        client_id: str, 
        galaxy_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Register a client with the server.
        
        Args:
            client_id: Unique client identifier
            galaxy_id: Galaxy this client belongs to
            metadata: Additional client metadata
            
        Returns:
            True if registration successful
        """
        if client_id in self.registered_clients:
            self.logger.warning(f"Client {client_id} already registered")
            return False
        
        self.registered_clients[client_id] = {
            'id': client_id,
            'galaxy_id': galaxy_id,
            'registered_at': time.time(),
            'metadata': metadata or {},
            'rounds_participated': 0
        }
        
        if galaxy_id:
            self.galaxy_mapping[client_id] = galaxy_id
        
        self.logger.info(f"Client {client_id} registered (galaxy: {galaxy_id})")
        return True
    
    def set_aggregation_function(
        self, 
        fn: Callable[[List[List[torch.Tensor]]], List[torch.Tensor]]
    ) -> None:
        """Set custom aggregation function.
        
        Args:
            fn: Function that takes list of gradient lists and returns aggregated gradients
        """
        self._aggregation_fn = fn
    
    def start_round(self, selected_clients: Optional[List[str]] = None) -> RoundInfo:
        """Start a new training round.
        
        Args:
            selected_clients: Specific clients to participate (None = all)
            
        Returns:
            RoundInfo for this round
        """
        self.current_round += 1
        
        # Select participating clients
        if selected_clients is None:
            participants = list(self.registered_clients.keys())
        else:
            participants = [c for c in selected_clients if c in self.registered_clients]
        
        # Create round info
        self.round_info = RoundInfo(
            round_num=self.current_round,
            state=RoundState.WAITING_FOR_GRADIENTS,
            start_time=time.time(),
            num_participants=len(participants)
        )
        
        self.logger.info(
            f"Starting round {self.current_round} with {len(participants)} clients"
        )
        
        # Send round start notifications
        weights = self.get_model_weights()
        for client_id in participants:
            msg = create_round_start_msg(
                round_num=self.current_round,
                receiver_id=client_id,
                config={'local_epochs': self.config.get('local_epochs', 1)}
            )
            msg.payload['weights'] = weights
            self.channel.send(msg)
        
        return self.round_info
    
    def receive_gradient(self, client_id: str, gradients: List[torch.Tensor]) -> bool:
        """Receive gradient update from a client.
        
        Args:
            client_id: ID of sending client
            gradients: List of gradient tensors
            
        Returns:
            True if accepted
        """
        if self.round_info is None:
            self.logger.warning("No active round")
            return False
        
        if self.round_info.state != RoundState.WAITING_FOR_GRADIENTS:
            self.logger.warning(f"Round not accepting gradients (state: {self.round_info.state})")
            return False
        
        if client_id in self.round_info.gradients_received:
            self.logger.warning(f"Already received gradient from {client_id}")
            return False
        
        self.round_info.gradients_received[client_id] = gradients
        self.round_info.num_received += 1
        
        # Update client stats
        if client_id in self.registered_clients:
            self.registered_clients[client_id]['rounds_participated'] += 1
        
        self.logger.debug(
            f"Received gradient from {client_id} "
            f"({self.round_info.num_received}/{self.round_info.num_participants})"
        )
        
        return True
    
    def check_round_complete(self) -> bool:
        """Check if round has enough gradients to aggregate.
        
        Returns:
            True if ready to aggregate
        """
        if self.round_info is None:
            return False
        
        min_clients = int(self.round_info.num_participants * self.min_clients_ratio)
        return self.round_info.num_received >= min_clients
    
    def aggregate_gradients(self) -> List[torch.Tensor]:
        """Aggregate received gradients.
        
        Returns:
            Aggregated gradients
        """
        if self.round_info is None:
            raise RuntimeError("No active round")
        
        self.round_info.state = RoundState.AGGREGATING
        
        # Collect all gradients
        gradient_lists = list(self.round_info.gradients_received.values())
        
        if not gradient_lists:
            raise RuntimeError("No gradients to aggregate")
        
        # Use custom aggregation function if set
        if self._aggregation_fn:
            aggregated = self._aggregation_fn(gradient_lists)
        else:
            # Default: FedAvg (simple averaging)
            aggregated = self._fedavg_aggregate(gradient_lists)
        
        self.round_info.aggregated_gradients = aggregated
        self.logger.info(f"Aggregated {len(gradient_lists)} gradient updates")
        
        return aggregated
    
    def _fedavg_aggregate(
        self, 
        gradient_lists: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Default FedAvg aggregation.
        
        Args:
            gradient_lists: List of gradient lists from clients
            
        Returns:
            Averaged gradients
        """
        num_clients = len(gradient_lists)
        
        # Initialize with first client's gradients
        aggregated = [g.clone().float() / num_clients for g in gradient_lists[0]]
        
        # Add remaining clients
        for gradients in gradient_lists[1:]:
            for i, g in enumerate(gradients):
                aggregated[i] += g.float() / num_clients
        
        return aggregated
    
    def apply_gradients(self, gradients: List[torch.Tensor], lr: float = 1.0) -> None:
        """Apply aggregated gradients to global model.
        
        Args:
            gradients: Aggregated gradients
            lr: Learning rate for gradient application
        """
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), gradients):
                param.data -= lr * grad
    
    def get_model_weights(self) -> List[torch.Tensor]:
        """Get current model weights.
        
        Returns:
            List of model parameter tensors
        """
        return [p.data.clone() for p in self.model.parameters()]
    
    def set_model_weights(self, weights: List[torch.Tensor]) -> None:
        """Set model weights.
        
        Args:
            weights: New model weights
        """
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.data.copy_(weight)
    
    def end_round(self, metrics: Optional[Dict] = None) -> RoundInfo:
        """End the current round.
        
        Args:
            metrics: Optional round metrics
            
        Returns:
            Completed RoundInfo
        """
        if self.round_info is None:
            raise RuntimeError("No active round")
        
        self.round_info.state = RoundState.DISTRIBUTING
        self.round_info.end_time = time.time()
        self.round_info.metrics = metrics or {}
        
        # Send model updates to clients
        weights = self.get_model_weights()
        for client_id in self.registered_clients:
            msg = create_model_update_msg(
                sender_id="global",
                receiver_id=client_id,
                round_num=self.current_round,
                weights=weights,
                metadata={'round_metrics': self.round_info.metrics}
            )
            self.channel.send(msg)
        
        # Send round end notifications
        for client_id in self.registered_clients:
            msg = create_round_end_msg(
                round_num=self.current_round,
                receiver_id=client_id,
                metrics=self.round_info.metrics
            )
            self.channel.send(msg)
        
        self.round_info.state = RoundState.COMPLETE
        
        # Archive round
        completed = self.round_info
        self.round_history.append(completed)
        self.round_info = None
        
        duration = completed.end_time - completed.start_time
        self.logger.info(
            f"Round {completed.round_num} complete. "
            f"Duration: {duration:.2f}s, Participants: {completed.num_received}"
        )
        
        return completed
    
    def execute_round(
        self,
        client_loaders: Dict[str, Any],
        selected_clients: Optional[List[str]] = None,
        evaluate_fn: Optional[Callable] = None
    ) -> RoundInfo:
        """Execute a complete training round (for simulation).
        
        This is a convenience method that runs all round phases.
        In production, phases would be driven by actual message exchanges.
        
        Args:
            client_loaders: Dict mapping client_id to data loader
            selected_clients: Clients to participate
            evaluate_fn: Optional evaluation function
            
        Returns:
            Completed RoundInfo
        """
        from src.client.trainer import Trainer
        
        # Start round
        round_info = self.start_round(selected_clients)
        participants = selected_clients or list(self.registered_clients.keys())
        
        # Simulate client training
        weights = self.get_model_weights()
        for client_id in participants:
            if client_id not in client_loaders:
                continue
            
            # Create temporary model copy for client
            client_model = type(self.model)()
            client_model.load_state_dict(self.model.state_dict())
            
            # Train locally
            trainer = Trainer(client_model, self.config.get('learning_rate', 0.01))
            trainer.train(client_loaders[client_id], self.config.get('local_epochs', 1))
            
            # Get gradients (difference from original weights)
            gradients = []
            for orig, new in zip(weights, trainer.model.parameters()):
                gradients.append(orig - new.data)
            
            self.receive_gradient(client_id, gradients)
        
        # Aggregate
        if self.check_round_complete():
            aggregated = self.aggregate_gradients()
            self.apply_gradients(aggregated)
        
        # Evaluate if function provided
        metrics = {}
        if evaluate_fn:
            metrics = evaluate_fn(self.model)
        
        return self.end_round(metrics)
    
    def get_round_history(self) -> List[RoundInfo]:
        """Get history of all completed rounds.
        
        Returns:
            List of RoundInfo objects
        """
        return self.round_history.copy()
    
    def get_client_stats(self) -> Dict[str, Dict]:
        """Get statistics for all registered clients.
        
        Returns:
            Dict mapping client_id to stats
        """
        return {
            client_id: {
                'galaxy_id': info.get('galaxy_id'),
                'rounds_participated': info.get('rounds_participated', 0),
                'registered_at': info.get('registered_at')
            }
            for client_id, info in self.registered_clients.items()
        }
    
    def shutdown(self) -> None:
        """Shutdown the server"""
        self.channel.close()
        self.logger.info("FL Server shutdown")
