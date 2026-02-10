"""REST API Communication Module (PROTO-306, PROTO-407)

This module provides REST API server and client implementations for:
- Galaxy-level communication (PROTO-306)
- Global aggregator communication hub (PROTO-407)

For MVP, these can be used for testing. The simulation uses in-memory channels.
"""

import json
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

# Note: Flask/FastAPI imports are optional - use try/except for flexibility
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


# ==============================================================================
# Data Classes for API Messages
# ==============================================================================

@dataclass
class GradientSubmission:
    """Gradient submission from client to galaxy"""
    client_id: str
    galaxy_id: str
    round_number: int
    gradients: List[List[float]]  # Serialized as nested lists
    commitment_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'client_id': self.client_id,
            'galaxy_id': self.galaxy_id,
            'round_number': self.round_number,
            'gradients': self.gradients,
            'commitment_hash': self.commitment_hash,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GradientSubmission':
        return cls(**data)


@dataclass  
class GalaxyAggregate:
    """Aggregated result from galaxy to global"""
    galaxy_id: str
    round_number: int
    aggregated_gradients: List[List[float]]
    merkle_root: str
    num_clients: int
    client_ids: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'galaxy_id': self.galaxy_id,
            'round_number': self.round_number,
            'aggregated_gradients': self.aggregated_gradients,
            'merkle_root': self.merkle_root,
            'num_clients': self.num_clients,
            'client_ids': self.client_ids,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GalaxyAggregate':
        return cls(**data)


@dataclass
class MerkleProofResponse:
    """Merkle proof response to client"""
    client_id: str
    round_number: int
    proof: List[Dict]  # List of {hash, direction}
    root: str
    verified: bool
    
    def to_dict(self) -> Dict:
        return {
            'client_id': self.client_id,
            'round_number': self.round_number,
            'proof': self.proof,
            'root': self.root,
            'verified': self.verified
        }


@dataclass
class ModelBroadcast:
    """Global model broadcast message"""
    round_number: int
    model_state: Dict[str, List[float]]  # Serialized state dict
    model_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'round_number': self.round_number,
            'model_state': self.model_state,
            'model_hash': self.model_hash,
            'timestamp': self.timestamp
        }


# ==============================================================================
# Abstract API Interface
# ==============================================================================

class CommunicationAPI(ABC):
    """Abstract base class for communication APIs"""
    
    @abstractmethod
    def start(self):
        """Start the API server"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the API server"""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if server is running"""
        pass


# ==============================================================================
# Galaxy Communication Module (PROTO-306)
# ==============================================================================

class GalaxyAPIServer(CommunicationAPI):
    """
    Galaxy-level REST API server (PROTO-306).
    
    Endpoints:
    - POST /galaxy/submit_gradient - Receive gradients from clients
    - GET /galaxy/proof/<client_id> - Return Merkle proof to client
    - GET /galaxy/status - Get galaxy status
    """
    
    def __init__(
        self,
        galaxy_id: str,
        host: str = '0.0.0.0',
        port: int = 5000
    ):
        self.galaxy_id = galaxy_id
        self.host = host
        self.port = port
        
        # Storage
        self.gradient_queue: queue.Queue = queue.Queue()
        self.collected_gradients: Dict[int, Dict[str, GradientSubmission]] = {}
        self.merkle_proofs: Dict[int, Dict[str, MerkleProofResponse]] = {}
        self.current_round = 0
        
        # Callbacks
        self.on_gradient_received: Optional[Callable] = None
        self.on_round_complete: Optional[Callable] = None
        
        # Server state
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        
        if FLASK_AVAILABLE:
            self.app = self._create_flask_app()
        else:
            self.app = None
    
    def _create_flask_app(self) -> 'Flask':
        """Create Flask application with routes"""
        app = Flask(f'galaxy_{self.galaxy_id}')
        
        @app.route('/galaxy/submit_gradient', methods=['POST'])
        def submit_gradient():
            data = request.get_json()
            submission = GradientSubmission.from_dict(data)
            
            # Store gradient
            round_num = submission.round_number
            if round_num not in self.collected_gradients:
                self.collected_gradients[round_num] = {}
            
            self.collected_gradients[round_num][submission.client_id] = submission
            
            # Notify callback
            if self.on_gradient_received:
                self.on_gradient_received(submission)
            
            return jsonify({
                'status': 'received',
                'client_id': submission.client_id,
                'round': round_num
            })
        
        @app.route('/galaxy/proof/<client_id>', methods=['GET'])
        def get_proof(client_id: str):
            round_num = request.args.get('round', self.current_round, type=int)
            
            if round_num in self.merkle_proofs:
                if client_id in self.merkle_proofs[round_num]:
                    proof = self.merkle_proofs[round_num][client_id]
                    return jsonify(proof.to_dict())
            
            return jsonify({'error': 'Proof not found'}), 404
        
        @app.route('/galaxy/status', methods=['GET'])
        def get_status():
            return jsonify({
                'galaxy_id': self.galaxy_id,
                'current_round': self.current_round,
                'clients_this_round': len(self.collected_gradients.get(self.current_round, {}))
            })
        
        return app
    
    def start(self):
        """Start the API server in a background thread"""
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask not available. Install with: pip install flask")
        
        def run_server():
            self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self._running = True
    
    def stop(self):
        """Stop the API server"""
        self._running = False
        # Note: Flask dev server doesn't have clean shutdown; use production server in real deployment
    
    def is_running(self) -> bool:
        return self._running
    
    def set_merkle_proof(self, round_number: int, client_id: str, proof: MerkleProofResponse):
        """Store Merkle proof for client retrieval"""
        if round_number not in self.merkle_proofs:
            self.merkle_proofs[round_number] = {}
        self.merkle_proofs[round_number][client_id] = proof
    
    def get_round_gradients(self, round_number: int) -> Dict[str, GradientSubmission]:
        """Get all collected gradients for a round"""
        return self.collected_gradients.get(round_number, {})
    
    def advance_round(self, new_round: int):
        """Advance to new round"""
        self.current_round = new_round


class GalaxyAPIClient:
    """
    Client for communicating with Galaxy API server.
    
    Used by clients to submit gradients and retrieve proofs.
    """
    
    def __init__(self, galaxy_host: str, galaxy_port: int, timeout: float = 30.0):
        self.base_url = f"http://{galaxy_host}:{galaxy_port}"
        self.timeout = timeout
        
        try:
            import requests
            self.requests = requests
            self._available = True
        except ImportError:
            self._available = False
    
    def submit_gradient(self, submission: GradientSubmission) -> Dict:
        """Submit gradient to galaxy server"""
        if not self._available:
            raise RuntimeError("requests library not available")
        
        response = self.requests.post(
            f"{self.base_url}/galaxy/submit_gradient",
            json=submission.to_dict(),
            timeout=self.timeout
        )
        return response.json()
    
    def get_proof(self, client_id: str, round_number: int) -> Optional[MerkleProofResponse]:
        """Get Merkle proof from galaxy server"""
        if not self._available:
            raise RuntimeError("requests library not available")
        
        response = self.requests.get(
            f"{self.base_url}/galaxy/proof/{client_id}",
            params={'round': round_number},
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            return MerkleProofResponse(**data)
        return None
    
    def get_status(self) -> Dict:
        """Get galaxy status"""
        if not self._available:
            raise RuntimeError("requests library not available")
        
        response = self.requests.get(
            f"{self.base_url}/galaxy/status",
            timeout=self.timeout
        )
        return response.json()


# ==============================================================================
# Global Communication Hub (PROTO-407)
# ==============================================================================

class GlobalAPIServer(CommunicationAPI):
    """
    Global aggregator communication hub (PROTO-407).
    
    Endpoints:
    - POST /global/submit - Receive galaxy aggregates
    - GET /global/model - Get current global model
    - GET /global/status - Get global aggregator status
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 6000
    ):
        self.host = host
        self.port = port
        
        # Storage
        self.galaxy_aggregates: Dict[int, Dict[str, GalaxyAggregate]] = {}
        self.global_model: Optional[ModelBroadcast] = None
        self.current_round = 0
        self.expected_galaxies: int = 0
        
        # Callbacks
        self.on_aggregate_received: Optional[Callable] = None
        self.on_all_galaxies_received: Optional[Callable] = None
        
        # Server state
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        
        if FLASK_AVAILABLE:
            self.app = self._create_flask_app()
        else:
            self.app = None
    
    def _create_flask_app(self) -> 'Flask':
        """Create Flask application with routes"""
        app = Flask('global_aggregator')
        
        @app.route('/global/submit', methods=['POST'])
        def submit_aggregate():
            data = request.get_json()
            aggregate = GalaxyAggregate.from_dict(data)
            
            round_num = aggregate.round_number
            if round_num not in self.galaxy_aggregates:
                self.galaxy_aggregates[round_num] = {}
            
            self.galaxy_aggregates[round_num][aggregate.galaxy_id] = aggregate
            
            # Notify callback
            if self.on_aggregate_received:
                self.on_aggregate_received(aggregate)
            
            # Check if all galaxies received
            if len(self.galaxy_aggregates[round_num]) >= self.expected_galaxies:
                if self.on_all_galaxies_received:
                    self.on_all_galaxies_received(round_num)
            
            return jsonify({
                'status': 'received',
                'galaxy_id': aggregate.galaxy_id,
                'round': round_num
            })
        
        @app.route('/global/model', methods=['GET'])
        def get_model():
            if self.global_model:
                return jsonify(self.global_model.to_dict())
            return jsonify({'error': 'No model available'}), 404
        
        @app.route('/global/status', methods=['GET'])
        def get_status():
            return jsonify({
                'current_round': self.current_round,
                'expected_galaxies': self.expected_galaxies,
                'received_this_round': len(self.galaxy_aggregates.get(self.current_round, {})),
                'has_model': self.global_model is not None
            })
        
        return app
    
    def start(self):
        """Start the API server"""
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask not available")
        
        def run_server():
            self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self._running = True
    
    def stop(self):
        self._running = False
    
    def is_running(self) -> bool:
        return self._running
    
    def set_expected_galaxies(self, num_galaxies: int):
        """Set expected number of galaxies"""
        self.expected_galaxies = num_galaxies
    
    def set_global_model(self, model: ModelBroadcast):
        """Update global model for distribution"""
        self.global_model = model
    
    def get_round_aggregates(self, round_number: int) -> Dict[str, GalaxyAggregate]:
        """Get all galaxy aggregates for a round"""
        return self.galaxy_aggregates.get(round_number, {})
    
    def advance_round(self, new_round: int):
        """Advance to new round"""
        self.current_round = new_round


class GlobalAPIClient:
    """
    Client for communicating with Global aggregator API.
    
    Used by galaxy aggregators to submit results and retrieve global model.
    """
    
    def __init__(self, global_host: str, global_port: int, timeout: float = 30.0):
        self.base_url = f"http://{global_host}:{global_port}"
        self.timeout = timeout
        
        try:
            import requests
            self.requests = requests
            self._available = True
        except ImportError:
            self._available = False
    
    def submit_aggregate(self, aggregate: GalaxyAggregate) -> Dict:
        """Submit galaxy aggregate to global server"""
        if not self._available:
            raise RuntimeError("requests library not available")
        
        response = self.requests.post(
            f"{self.base_url}/global/submit",
            json=aggregate.to_dict(),
            timeout=self.timeout
        )
        return response.json()
    
    def get_model(self) -> Optional[ModelBroadcast]:
        """Get global model from server"""
        if not self._available:
            raise RuntimeError("requests library not available")
        
        response = self.requests.get(
            f"{self.base_url}/global/model",
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            return ModelBroadcast(**data)
        return None
    
    def get_status(self) -> Dict:
        """Get global aggregator status"""
        if not self._available:
            raise RuntimeError("requests library not available")
        
        response = self.requests.get(
            f"{self.base_url}/global/status",
            timeout=self.timeout
        )
        return response.json()


# ==============================================================================
# Utility Functions
# ==============================================================================

def serialize_gradients(gradients: List) -> List[List[float]]:
    """Serialize gradient tensors to nested lists for JSON"""
    import torch
    import numpy as np
    
    serialized = []
    for g in gradients:
        if isinstance(g, torch.Tensor):
            serialized.append(g.detach().cpu().numpy().tolist())
        elif isinstance(g, np.ndarray):
            serialized.append(g.tolist())
        else:
            serialized.append(list(g))
    return serialized


def deserialize_gradients(serialized: List[List[float]]) -> List:
    """Deserialize gradients from nested lists"""
    import torch
    return [torch.tensor(g, dtype=torch.float32) for g in serialized]
