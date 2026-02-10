"""Model Synchronization Module (PROTO-1104)

This module handles global model distribution to all clients with integrity verification.
"""

import torch
import hashlib
import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


def compute_model_hash(model: torch.nn.Module) -> str:
    """
    Compute SHA-256 hash of model parameters for integrity verification.
    
    Args:
        model: PyTorch model
        
    Returns:
        Hex string of SHA-256 hash
    """
    hasher = hashlib.sha256()
    
    for name, param in sorted(model.named_parameters()):
        # Add parameter name
        hasher.update(name.encode('utf-8'))
        # Add parameter data
        param_bytes = param.detach().cpu().numpy().tobytes()
        hasher.update(param_bytes)
    
    return hasher.hexdigest()


def compute_state_dict_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Compute SHA-256 hash of state dict for integrity verification.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Hex string of SHA-256 hash
    """
    hasher = hashlib.sha256()
    
    for name in sorted(state_dict.keys()):
        param = state_dict[name]
        hasher.update(name.encode('utf-8'))
        if isinstance(param, torch.Tensor):
            param_bytes = param.detach().cpu().numpy().tobytes()
        else:
            param_bytes = np.array(param).tobytes()
        hasher.update(param_bytes)
    
    return hasher.hexdigest()


@dataclass
class ModelVersion:
    """Represents a versioned model snapshot"""
    round_number: int
    model_hash: str
    state_dict: Dict[str, torch.Tensor]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelSynchronizer:
    """
    Manages model synchronization across all clients (PROTO-1104).
    
    Features:
    - Broadcast model parameters to all clients
    - Verify model integrity via hash checking
    - Track model versions
    - Handle sync failures gracefully
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize model synchronizer.
        
        Args:
            model: The global PyTorch model to synchronize
        """
        self.model = model
        self.current_version: Optional[ModelVersion] = None
        self.version_history: List[ModelVersion] = []
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'verification_failures': 0
        }
    
    def create_version(self, round_number: int, metadata: Dict = None) -> ModelVersion:
        """
        Create a new model version snapshot.
        
        Args:
            round_number: Current FL round number
            metadata: Optional metadata to attach
            
        Returns:
            ModelVersion object
        """
        state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        model_hash = compute_state_dict_hash(state_dict)
        
        version = ModelVersion(
            round_number=round_number,
            model_hash=model_hash,
            state_dict=state_dict,
            metadata=metadata or {}
        )
        
        self.current_version = version
        self.version_history.append(version)
        
        return version
    
    def get_sync_package(self, round_number: int = None) -> Dict:
        """
        Get model sync package for distribution.
        
        Args:
            round_number: Optional round number (creates new version if provided)
            
        Returns:
            Dict containing state_dict, hash, and metadata
        """
        if round_number is not None:
            self.create_version(round_number)
        
        if self.current_version is None:
            self.create_version(0)
        
        # Convert tensors to lists for JSON serialization
        serializable_state = {}
        for key, tensor in self.current_version.state_dict.items():
            serializable_state[key] = tensor.cpu().numpy().tolist()
        
        return {
            'round_number': self.current_version.round_number,
            'model_hash': self.current_version.model_hash,
            'state_dict': serializable_state,
            'timestamp': self.current_version.timestamp,
            'metadata': self.current_version.metadata
        }
    
    def distribute_to_clients(
        self, 
        clients: List[Any], 
        round_number: int
    ) -> Dict[str, bool]:
        """
        Distribute model to all clients (sequential).
        
        Args:
            clients: List of client objects with load_model method
            round_number: Current round number
            
        Returns:
            Dict mapping client_id to sync success status
        """
        sync_package = self.get_sync_package(round_number)
        results = {}
        
        for client in clients:
            client_id = getattr(client, 'client_id', str(id(client)))
            try:
                # Client receives and loads model
                if hasattr(client, 'receive_model'):
                    success = client.receive_model(sync_package)
                elif hasattr(client, 'load_model'):
                    # Convert back to tensors
                    state_dict = {}
                    for key, value in sync_package['state_dict'].items():
                        state_dict[key] = torch.tensor(value)
                    client.load_model(state_dict)
                    success = True
                else:
                    # Direct state dict assignment
                    state_dict = {}
                    for key, value in sync_package['state_dict'].items():
                        state_dict[key] = torch.tensor(value)
                    if hasattr(client, 'model'):
                        client.model.load_state_dict(state_dict)
                    success = True
                
                results[client_id] = success
                self.sync_stats['total_syncs'] += 1
                if success:
                    self.sync_stats['successful_syncs'] += 1
                else:
                    self.sync_stats['failed_syncs'] += 1
                    
            except Exception as e:
                results[client_id] = False
                self.sync_stats['total_syncs'] += 1
                self.sync_stats['failed_syncs'] += 1
        
        return results
    
    def verify_client_model(
        self, 
        client_state_dict: Dict[str, torch.Tensor],
        expected_hash: str = None
    ) -> bool:
        """
        Verify client model matches expected hash.
        
        Args:
            client_state_dict: Client's model state dict
            expected_hash: Expected hash (uses current version if None)
            
        Returns:
            True if verification passes
        """
        if expected_hash is None:
            if self.current_version is None:
                return False
            expected_hash = self.current_version.model_hash
        
        client_hash = compute_state_dict_hash(client_state_dict)
        
        if client_hash != expected_hash:
            self.sync_stats['verification_failures'] += 1
            return False
        
        return True
    
    def apply_update(
        self, 
        gradients: List[torch.Tensor], 
        learning_rate: float = 1.0
    ):
        """
        Apply gradient update to the global model.
        
        Args:
            gradients: List of gradient tensors
            learning_rate: Learning rate for update
        """
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), gradients):
                if isinstance(grad, np.ndarray):
                    grad = torch.from_numpy(grad)
                param.data -= learning_rate * grad.to(param.device)
    
    def get_version_history(self) -> List[Dict]:
        """Get simplified version history"""
        return [
            {
                'round': v.round_number,
                'hash': v.model_hash,
                'timestamp': v.timestamp
            }
            for v in self.version_history
        ]
    
    def get_stats(self) -> Dict:
        """Get synchronization statistics"""
        return self.sync_stats.copy()
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'verification_failures': 0
        }


class ClientModelReceiver:
    """
    Client-side model receiver with integrity verification.
    
    Use this mixin or standalone class to handle model synchronization on clients.
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.last_received_hash: Optional[str] = None
        self.last_received_round: int = -1
    
    def receive_model(self, sync_package: Dict) -> bool:
        """
        Receive and verify model from server.
        
        Args:
            sync_package: Dict with state_dict, model_hash, round_number
            
        Returns:
            True if model loaded and verified successfully
        """
        try:
            # Convert lists back to tensors
            state_dict = {}
            for key, value in sync_package['state_dict'].items():
                state_dict[key] = torch.tensor(value)
            
            # Load into model
            self.model.load_state_dict(state_dict)
            
            # Verify hash
            computed_hash = compute_state_dict_hash(state_dict)
            expected_hash = sync_package.get('model_hash')
            
            if expected_hash and computed_hash != expected_hash:
                raise ValueError(
                    f"Model hash mismatch: expected {expected_hash}, got {computed_hash}"
                )
            
            self.last_received_hash = computed_hash
            self.last_received_round = sync_package.get('round_number', -1)
            
            return True
            
        except Exception as e:
            print(f"Model receive failed: {e}")
            return False
    
    def get_model_hash(self) -> Optional[str]:
        """Get hash of current model"""
        return compute_model_hash(self.model)
    
    def verify_model(self, expected_hash: str) -> bool:
        """Verify current model matches expected hash"""
        return self.get_model_hash() == expected_hash
