"""Communication module for federated learning message passing

Provides:
- In-memory channels for simulation
- REST API for production (PROTO-306, PROTO-407)
"""

from src.communication.message import Message, MessageType
from src.communication.channel import CommunicationChannel
from src.communication.server import FLServer
from src.communication.client_comm import ClientCommunicator
from src.communication.rest_api import (
    GradientSubmission,
    GalaxyAggregate,
    MerkleProofResponse,
    ModelBroadcast,
    GalaxyAPIServer,
    GalaxyAPIClient,
    GlobalAPIServer,
    GlobalAPIClient,
    serialize_gradients,
    deserialize_gradients
)

__all__ = [
    # Message types
    'Message',
    'MessageType',
    # In-memory channels (simulation)
    'CommunicationChannel',
    'FLServer',
    'ClientCommunicator',
    # REST API (production)
    'GradientSubmission',
    'GalaxyAggregate',
    'MerkleProofResponse',
    'ModelBroadcast',
    'GalaxyAPIServer',
    'GalaxyAPIClient',
    'GlobalAPIServer',
    'GlobalAPIClient',
    'serialize_gradients',
    'deserialize_gradients'
]
