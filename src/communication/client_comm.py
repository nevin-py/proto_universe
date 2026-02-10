"""Client-side communication handler for FL.

Manages client communication with galaxy aggregator and global server.
"""

import time
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.communication.message import (
    Message, MessageType,
    create_gradient_submit_msg, create_ack_msg
)
from src.communication.channel import InMemoryChannel, MessageHandler


@dataclass
class ClientState:
    """State of a federated learning client"""
    client_id: str
    galaxy_id: str
    current_round: int = 0
    is_registered: bool = False
    last_update_time: float = 0.0
    model_version: int = 0


class ClientCommunicator:
    """Handles client-side communication for federated learning.
    
    Manages:
    - Registration with server/galaxy
    - Receiving model updates
    - Submitting gradient updates
    - Handling round notifications
    """
    
    def __init__(
        self,
        client_id: str,
        galaxy_id: str,
        model: nn.Module,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize client communicator.
        
        Args:
            client_id: Unique client identifier
            galaxy_id: Galaxy this client belongs to
            model: Client's local model
            logger: Logger instance
        """
        self.client_id = client_id
        self.galaxy_id = galaxy_id
        self.model = model
        self.logger = logger or logging.getLogger(f"client_{client_id}")
        
        # State management
        self.state = ClientState(client_id=client_id, galaxy_id=galaxy_id)
        
        # Communication channel
        self.channel = InMemoryChannel(client_id, self.logger)
        self.message_handler = MessageHandler(self.channel)
        
        # Setup message handlers
        self._setup_handlers()
        
        # Callbacks for round events
        self._on_round_start_callback = None
        self._on_model_update_callback = None
        self._on_round_end_callback = None
    
    def _setup_handlers(self) -> None:
        """Setup message handlers for different message types"""
        self.message_handler.register_handler(
            MessageType.ROUND_START, self._handle_round_start
        )
        self.message_handler.register_handler(
            MessageType.MODEL_UPDATE, self._handle_model_update
        )
        self.message_handler.register_handler(
            MessageType.ROUND_END, self._handle_round_end
        )
        self.message_handler.register_handler(
            MessageType.GRADIENT_REQUEST, self._handle_gradient_request
        )
        self.message_handler.register_handler(
            MessageType.PROOF_RESPONSE, self._handle_proof_response
        )
    
    def set_round_start_callback(self, callback) -> None:
        """Set callback for round start events"""
        self._on_round_start_callback = callback
    
    def set_model_update_callback(self, callback) -> None:
        """Set callback for model update events"""
        self._on_model_update_callback = callback
    
    def set_round_end_callback(self, callback) -> None:
        """Set callback for round end events"""
        self._on_round_end_callback = callback
    
    def _handle_round_start(self, message: Message) -> None:
        """Handle round start notification"""
        self.state.current_round = message.round_num
        self.logger.info(f"Round {message.round_num} started")
        
        # Update model if weights provided
        if 'weights' in message.payload:
            self._update_model_weights(message.payload['weights'])
        
        # Trigger callback
        if self._on_round_start_callback:
            self._on_round_start_callback(message.round_num, message.payload)
    
    def _handle_model_update(self, message: Message) -> None:
        """Handle model update from server"""
        if 'weights' in message.payload:
            self._update_model_weights(message.payload['weights'])
            self.state.model_version += 1
            self.state.last_update_time = time.time()
            self.logger.info(
                f"Model updated (version {self.state.model_version})"
            )
        
        if self._on_model_update_callback:
            self._on_model_update_callback(message.payload)
    
    def _handle_round_end(self, message: Message) -> None:
        """Handle round end notification"""
        metrics = message.payload.get('metrics', {})
        self.logger.info(f"Round {message.round_num} ended. Metrics: {metrics}")
        
        if self._on_round_end_callback:
            self._on_round_end_callback(message.round_num, metrics)
    
    def _handle_gradient_request(self, message: Message) -> None:
        """Handle gradient request from aggregator"""
        # This would trigger local training if not already done
        self.logger.debug("Received gradient request")
    
    def _handle_proof_response(self, message: Message) -> None:
        """Handle Merkle proof response"""
        proof = message.payload.get('proof')
        is_valid = message.payload.get('is_valid', False)
        self.logger.debug(f"Received proof response: valid={is_valid}")
    
    def _update_model_weights(self, weights: List[torch.Tensor]) -> None:
        """Update local model with new weights"""
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.data.copy_(weight)
    
    def register(self, metadata: Optional[Dict] = None) -> bool:
        """Register with the server/galaxy.
        
        Args:
            metadata: Additional registration metadata
            
        Returns:
            True if registration message sent
        """
        msg = Message(
            msg_type=MessageType.CLIENT_REGISTER,
            sender_id=self.client_id,
            receiver_id=self.galaxy_id,
            payload={
                'galaxy_id': self.galaxy_id,
                'metadata': metadata or {}
            }
        )
        
        success = self.channel.send(msg)
        if success:
            self.state.is_registered = True
            self.logger.info(f"Registered with galaxy {self.galaxy_id}")
        
        return success
    
    def submit_gradients(
        self,
        gradients: List[torch.Tensor],
        commitment: Optional[str] = None
    ) -> bool:
        """Submit gradient update to galaxy aggregator.
        
        Args:
            gradients: List of gradient tensors
            commitment: Optional Merkle commitment hash
            
        Returns:
            True if submission successful
        """
        # Serialize gradients for transmission
        serialized_gradients = [g.cpu().detach() for g in gradients]
        
        msg = create_gradient_submit_msg(
            client_id=self.client_id,
            galaxy_id=self.galaxy_id,
            round_num=self.state.current_round,
            gradients=serialized_gradients,
            commitment=commitment
        )
        
        success = self.channel.send(msg)
        if success:
            self.logger.info(
                f"Submitted gradients for round {self.state.current_round}"
            )
        else:
            self.logger.error("Failed to submit gradients")
        
        return success
    
    def submit_commitment(self, commitment: str) -> bool:
        """Submit gradient commitment (before revealing gradients).
        
        Args:
            commitment: Merkle commitment hash
            
        Returns:
            True if submission successful
        """
        msg = Message(
            msg_type=MessageType.COMMITMENT_SUBMIT,
            sender_id=self.client_id,
            receiver_id=self.galaxy_id,
            round_num=self.state.current_round,
            payload={'commitment': commitment}
        )
        
        success = self.channel.send(msg)
        if success:
            self.logger.debug(f"Submitted commitment: {commitment[:16]}...")
        
        return success
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to maintain connection.
        
        Returns:
            True if sent successfully
        """
        msg = Message(
            msg_type=MessageType.HEARTBEAT,
            sender_id=self.client_id,
            receiver_id=self.galaxy_id,
            payload={'timestamp': time.time()}
        )
        return self.channel.send(msg)
    
    def get_model_weights(self) -> List[torch.Tensor]:
        """Get current local model weights.
        
        Returns:
            List of model parameter tensors
        """
        return [p.data.clone() for p in self.model.parameters()]
    
    def process_pending_messages(self) -> int:
        """Process all pending messages.
        
        Returns:
            Number of messages processed
        """
        messages = self.channel.receive_all()
        for msg in messages:
            self.message_handler.handle_message(msg)
        return len(messages)
    
    def wait_for_round_start(self, timeout: float = 60.0) -> Optional[int]:
        """Wait for round start notification.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Round number if received, None if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.channel.receive(timeout=1.0)
            if msg and msg.msg_type == MessageType.ROUND_START:
                self._handle_round_start(msg)
                return msg.round_num
        return None
    
    def start_listening(self) -> None:
        """Start background message listener"""
        self.message_handler.start_listening()
    
    def stop_listening(self) -> None:
        """Stop background message listener"""
        self.message_handler.stop_listening()
    
    def close(self) -> None:
        """Close the client communicator"""
        self.stop_listening()
        self.channel.close()
        self.logger.info("Client communicator closed")
