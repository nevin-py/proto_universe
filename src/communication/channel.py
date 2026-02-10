"""Communication channel abstraction for FL message passing.

Provides abstract interface for communication channels between FL components.
Supports both synchronous and asynchronous message passing patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
from queue import Queue, Empty
import threading
import time
import logging

from src.communication.message import Message, MessageType


class CommunicationChannel(ABC):
    """Abstract base class for communication channels.
    
    Defines the interface for sending and receiving messages between
    FL components (clients, galaxy aggregators, global aggregator).
    """
    
    @abstractmethod
    def send(self, message: Message) -> bool:
        """Send a message to the receiver.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive a message from the channel.
        
        Args:
            timeout: Maximum time to wait for a message (None = blocking)
            
        Returns:
            Received message or None if timeout
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the communication channel"""
        pass


class InMemoryChannel(CommunicationChannel):
    """In-memory communication channel for simulation.
    
    Uses queues for message passing between components in the same process.
    Useful for local simulation and testing.
    """
    
    # Shared message queues across all channels (keyed by receiver_id)
    _message_queues: Dict[str, Queue] = defaultdict(Queue)
    _lock = threading.Lock()
    
    def __init__(self, node_id: str, logger: Optional[logging.Logger] = None):
        """Initialize in-memory channel.
        
        Args:
            node_id: ID of this node (client_id, galaxy_id, or 'global')
            logger: Optional logger for message tracking
        """
        self.node_id = node_id
        self.logger = logger or logging.getLogger(f"channel_{node_id}")
        self._running = True
    
    def send(self, message: Message) -> bool:
        """Send message to receiver's queue.
        
        Args:
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        if not self._running:
            return False
        
        try:
            with self._lock:
                receiver_queue = self._message_queues[message.receiver_id]
                receiver_queue.put(message)
            
            self.logger.debug(
                f"Sent {message.msg_type.value} from {message.sender_id} "
                f"to {message.receiver_id}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive message from this node's queue.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Received message or None
        """
        if not self._running:
            return None
        
        try:
            my_queue = self._message_queues[self.node_id]
            message = my_queue.get(timeout=timeout)
            
            self.logger.debug(
                f"Received {message.msg_type.value} from {message.sender_id}"
            )
            return message
            
        except Empty:
            return None
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None
    
    def receive_all(self) -> List[Message]:
        """Receive all pending messages.
        
        Returns:
            List of all pending messages
        """
        messages = []
        while True:
            msg = self.receive(timeout=0.01)
            if msg is None:
                break
            messages.append(msg)
        return messages
    
    def peek(self) -> Optional[Message]:
        """Peek at next message without removing it.
        
        Returns:
            Next message or None if queue is empty
        """
        my_queue = self._message_queues[self.node_id]
        try:
            # Use Queue's internal list for peeking
            with self._lock:
                if not my_queue.empty():
                    return my_queue.queue[0]
        except:
            pass
        return None
    
    def pending_count(self) -> int:
        """Get number of pending messages.
        
        Returns:
            Number of messages in queue
        """
        return self._message_queues[self.node_id].qsize()
    
    def close(self) -> None:
        """Close the channel"""
        self._running = False
        self.logger.debug(f"Channel {self.node_id} closed")
    
    @classmethod
    def clear_all_queues(cls) -> None:
        """Clear all message queues (for testing/reset)"""
        with cls._lock:
            cls._message_queues.clear()


class MessageHandler:
    """Handler for processing received messages.
    
    Provides callback-based message handling with support for
    different message types.
    """
    
    def __init__(self, channel: CommunicationChannel):
        """Initialize message handler.
        
        Args:
            channel: Communication channel to handle messages for
        """
        self.channel = channel
        self._handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._default_handler: Optional[Callable] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def register_handler(
        self, 
        msg_type: MessageType, 
        handler: Callable[[Message], None]
    ) -> None:
        """Register a handler for a specific message type.
        
        Args:
            msg_type: Message type to handle
            handler: Callback function that takes a Message
        """
        self._handlers[msg_type].append(handler)
    
    def set_default_handler(self, handler: Callable[[Message], None]) -> None:
        """Set default handler for unhandled message types.
        
        Args:
            handler: Default callback function
        """
        self._default_handler = handler
    
    def handle_message(self, message: Message) -> None:
        """Process a received message.
        
        Args:
            message: Message to process
        """
        handlers = self._handlers.get(message.msg_type, [])
        
        if handlers:
            for handler in handlers:
                try:
                    handler(message)
                except Exception as e:
                    logging.error(f"Handler error for {message.msg_type}: {e}")
        elif self._default_handler:
            self._default_handler(message)
    
    def start_listening(self, poll_interval: float = 0.1) -> None:
        """Start listening for messages in background thread.
        
        Args:
            poll_interval: Interval between receive attempts
        """
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            args=(poll_interval,),
            daemon=True
        )
        self._thread.start()
    
    def stop_listening(self) -> None:
        """Stop the listening thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _listen_loop(self, poll_interval: float) -> None:
        """Background loop for receiving messages"""
        while self._running:
            message = self.channel.receive(timeout=poll_interval)
            if message:
                self.handle_message(message)


class MessageBuffer:
    """Buffer for collecting messages before batch processing.
    
    Useful for collecting gradients from multiple clients before aggregation.
    """
    
    def __init__(self, expected_count: int, timeout: float = 60.0):
        """Initialize message buffer.
        
        Args:
            expected_count: Number of messages to collect
            timeout: Maximum time to wait for all messages
        """
        self.expected_count = expected_count
        self.timeout = timeout
        self._messages: List[Message] = []
        self._lock = threading.Lock()
        self._complete_event = threading.Event()
    
    def add(self, message: Message) -> bool:
        """Add a message to the buffer.
        
        Args:
            message: Message to add
            
        Returns:
            True if buffer is now complete
        """
        with self._lock:
            self._messages.append(message)
            if len(self._messages) >= self.expected_count:
                self._complete_event.set()
                return True
        return False
    
    def wait_for_complete(self) -> bool:
        """Wait for buffer to be complete.
        
        Returns:
            True if complete, False if timeout
        """
        return self._complete_event.wait(timeout=self.timeout)
    
    def get_messages(self) -> List[Message]:
        """Get all buffered messages.
        
        Returns:
            List of buffered messages
        """
        with self._lock:
            return self._messages.copy()
    
    def is_complete(self) -> bool:
        """Check if buffer is complete.
        
        Returns:
            True if expected count reached
        """
        return len(self._messages) >= self.expected_count
    
    def clear(self) -> None:
        """Clear the buffer"""
        with self._lock:
            self._messages.clear()
            self._complete_event.clear()
    
    @property
    def count(self) -> int:
        """Get current message count"""
        return len(self._messages)
