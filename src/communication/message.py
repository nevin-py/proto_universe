"""Message types and structures for FL communication.

Defines all message types exchanged between clients, galaxy aggregators, 
and the global aggregator in the ProtoGalaxy system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import json
import hashlib


class MessageType(Enum):
    """Types of messages in federated learning communication"""
    
    # Client -> Galaxy messages
    CLIENT_REGISTER = "client_register"
    GRADIENT_SUBMIT = "gradient_submit"
    COMMITMENT_SUBMIT = "commitment_submit"
    
    # Galaxy -> Client messages
    REGISTER_ACK = "register_ack"
    MODEL_UPDATE = "model_update"
    GRADIENT_REQUEST = "gradient_request"
    PROOF_RESPONSE = "proof_response"
    
    # Galaxy -> Global messages
    GALAXY_REGISTER = "galaxy_register"
    GALAXY_UPDATE = "galaxy_update"
    
    # Global -> Galaxy messages
    GLOBAL_MODEL = "global_model"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    
    # Control messages
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    ACK = "ack"


@dataclass
class Message:
    """Base message class for FL communication.
    
    Attributes:
        msg_type: Type of the message
        sender_id: ID of the sender (client_id, galaxy_id, or 'global')
        receiver_id: ID of the receiver
        round_num: Current FL round number
        payload: Message payload data
        timestamp: Message creation timestamp
        msg_id: Unique message identifier
    """
    
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    round_num: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default="")
    
    def __post_init__(self):
        """Generate message ID if not provided"""
        if not self.msg_id:
            self.msg_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique message ID based on content"""
        content = f"{self.sender_id}:{self.receiver_id}:{self.timestamp}:{self.msg_type.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'msg_type': self.msg_type.value,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'round_num': self.round_num,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'msg_id': self.msg_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            msg_type=MessageType(data['msg_type']),
            sender_id=data['sender_id'],
            receiver_id=data['receiver_id'],
            round_num=data.get('round_num', 0),
            payload=data.get('payload', {}),
            timestamp=data.get('timestamp', time.time()),
            msg_id=data.get('msg_id', '')
        )
    
    def to_json(self) -> str:
        """Serialize message to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Deserialize message from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


# Convenience factory functions for common message types

def create_gradient_submit_msg(
    client_id: str,
    galaxy_id: str,
    round_num: int,
    gradients: List[Any],
    commitment: Optional[str] = None
) -> Message:
    """Create a gradient submission message from client to galaxy"""
    return Message(
        msg_type=MessageType.GRADIENT_SUBMIT,
        sender_id=client_id,
        receiver_id=galaxy_id,
        round_num=round_num,
        payload={
            'gradients': gradients,
            'commitment': commitment
        }
    )


def create_model_update_msg(
    sender_id: str,
    receiver_id: str,
    round_num: int,
    weights: List[Any],
    metadata: Optional[Dict] = None
) -> Message:
    """Create a model update message"""
    return Message(
        msg_type=MessageType.MODEL_UPDATE,
        sender_id=sender_id,
        receiver_id=receiver_id,
        round_num=round_num,
        payload={
            'weights': weights,
            'metadata': metadata or {}
        }
    )


def create_round_start_msg(
    round_num: int,
    receiver_id: str,
    config: Optional[Dict] = None
) -> Message:
    """Create a round start notification message"""
    return Message(
        msg_type=MessageType.ROUND_START,
        sender_id='global',
        receiver_id=receiver_id,
        round_num=round_num,
        payload={
            'config': config or {}
        }
    )


def create_round_end_msg(
    round_num: int,
    receiver_id: str,
    metrics: Optional[Dict] = None
) -> Message:
    """Create a round end notification message"""
    return Message(
        msg_type=MessageType.ROUND_END,
        sender_id='global',
        receiver_id=receiver_id,
        round_num=round_num,
        payload={
            'metrics': metrics or {}
        }
    )


def create_error_msg(
    sender_id: str,
    receiver_id: str,
    error_code: str,
    error_message: str
) -> Message:
    """Create an error message"""
    return Message(
        msg_type=MessageType.ERROR,
        sender_id=sender_id,
        receiver_id=receiver_id,
        payload={
            'error_code': error_code,
            'error_message': error_message
        }
    )


def create_ack_msg(
    sender_id: str,
    receiver_id: str,
    original_msg_id: str,
    status: str = 'ok'
) -> Message:
    """Create an acknowledgment message"""
    return Message(
        msg_type=MessageType.ACK,
        sender_id=sender_id,
        receiver_id=receiver_id,
        payload={
            'original_msg_id': original_msg_id,
            'status': status
        }
    )
