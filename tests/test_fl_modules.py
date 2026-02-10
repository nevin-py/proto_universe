"""Tests for basic FL modules.

Tests communication, training, data loading, and simulation components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


def create_dummy_loader(num_samples=100, batch_size=10):
    """Create a dummy data loader for testing"""
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestTrainer:
    """Tests for Trainer class"""
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized"""
        from src.client.trainer import Trainer
        
        model = SimpleModel()
        trainer = Trainer(model, learning_rate=0.01)
        
        assert trainer.model is not None
        assert trainer.learning_rate == 0.01
    
    def test_trainer_train(self):
        """Test trainer can train"""
        from src.client.trainer import Trainer
        
        model = SimpleModel()
        trainer = Trainer(model, learning_rate=0.01)
        loader = create_dummy_loader()
        
        metrics = trainer.train(loader, num_epochs=1)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['samples'] > 0
    
    def test_trainer_evaluate(self):
        """Test trainer can evaluate"""
        from src.client.trainer import Trainer
        
        model = SimpleModel()
        trainer = Trainer(model)
        loader = create_dummy_loader()
        
        metrics = trainer.evaluate(loader)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_trainer_get_set_weights(self):
        """Test getting and setting weights"""
        from src.client.trainer import Trainer
        
        model = SimpleModel()
        trainer = Trainer(model)
        
        weights = trainer.get_weights()
        assert len(weights) > 0
        
        # Modify and set back
        new_weights = [w * 2 for w in weights]
        trainer.set_weights(new_weights)
        
        updated = trainer.get_weights()
        assert torch.allclose(updated[0], new_weights[0])
    
    def test_trainer_get_gradients(self):
        """Test getting gradients after training"""
        from src.client.trainer import Trainer
        
        model = SimpleModel()
        trainer = Trainer(model)
        loader = create_dummy_loader()
        
        trainer.train(loader, num_epochs=1)
        gradients = trainer.get_gradients()
        
        assert len(gradients) > 0


class TestDataPartition:
    """Tests for data partitioning"""
    
    def test_iid_partitioner(self):
        """Test IID partitioning"""
        from src.data.partition import IIDPartitioner
        
        # Create dummy dataset
        dataset = TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 10, (100,))
        )
        
        partitioner = IIDPartitioner(seed=42)
        partitions = partitioner.partition(dataset, num_clients=5)
        
        assert len(partitions) == 5
        
        # Check all indices are assigned
        all_indices = []
        for indices in partitions.values():
            all_indices.extend(indices)
        
        assert len(set(all_indices)) == 100
    
    def test_dirichlet_partitioner(self):
        """Test Dirichlet partitioning"""
        from src.data.partition import DirichletPartitioner
        
        # Create dataset with labels
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        dataset.targets = y.numpy()
        
        partitioner = DirichletPartitioner(alpha=0.5, seed=42)
        partitions = partitioner.partition(dataset, num_clients=5)
        
        assert len(partitions) == 5
        
        # Each client should have some data
        for indices in partitions.values():
            assert len(indices) > 0


class TestCommunication:
    """Tests for communication modules"""
    
    def test_message_creation(self):
        """Test message creation"""
        from src.communication.message import Message, MessageType
        
        msg = Message(
            msg_type=MessageType.GRADIENT_SUBMIT,
            sender_id="client_1",
            receiver_id="galaxy_0",
            round_num=1,
            payload={'gradients': [1, 2, 3]}
        )
        
        assert msg.msg_type == MessageType.GRADIENT_SUBMIT
        assert msg.sender_id == "client_1"
        assert msg.msg_id  # Should have auto-generated ID
    
    def test_message_serialization(self):
        """Test message JSON serialization"""
        from src.communication.message import Message, MessageType
        
        msg = Message(
            msg_type=MessageType.ACK,
            sender_id="server",
            receiver_id="client_1"
        )
        
        json_str = msg.to_json()
        restored = Message.from_json(json_str)
        
        assert restored.msg_type == msg.msg_type
        assert restored.sender_id == msg.sender_id
    
    def test_inmemory_channel(self):
        """Test in-memory communication channel"""
        from src.communication.channel import InMemoryChannel
        
        # Clear any existing queues
        InMemoryChannel.clear_all_queues()
        
        sender = InMemoryChannel("sender")
        receiver = InMemoryChannel("receiver")
        
        from src.communication.message import Message, MessageType
        
        msg = Message(
            msg_type=MessageType.HEARTBEAT,
            sender_id="sender",
            receiver_id="receiver"
        )
        
        sender.send(msg)
        received = receiver.receive(timeout=1.0)
        
        assert received is not None
        assert received.msg_type == MessageType.HEARTBEAT


class TestRoundManager:
    """Tests for round management"""
    
    def test_round_manager_start(self):
        """Test starting a round"""
        from src.orchestration.round_manager import RoundManager
        
        manager = RoundManager()
        context = manager.start_round(['client_0', 'client_1', 'client_2'])
        
        assert context.round_num == 1
        assert len(context.expected_clients) == 3
    
    def test_round_manager_record_gradient(self):
        """Test recording gradients"""
        from src.orchestration.round_manager import RoundManager
        
        manager = RoundManager()
        manager.start_round(['client_0', 'client_1'])
        
        manager.record_gradient('client_0', [torch.randn(10)])
        manager.record_gradient('client_1', [torch.randn(10)])
        
        assert len(manager.round_context.gradients) == 2
        assert manager.round_context.participation_rate == 1.0


class TestLogging:
    """Tests for FL logging"""
    
    def test_fl_logger_creation(self):
        """Test FL logger creation"""
        from src.logging import FLLogger
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FLLogger("test", log_dir=tmpdir)
            logger.info("Test message")
            logger.close()
    
    def test_fl_logger_metrics(self):
        """Test metrics logging"""
        from src.logging import FLLogger
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FLLogger("test", log_dir=tmpdir)
            logger.set_round(1)
            logger.log_metrics({'accuracy': 0.95, 'loss': 0.1})
            
            csv_path = logger.export_metrics_csv()
            assert os.path.exists(csv_path)
            
            logger.close()


class TestSimulation:
    """Tests for FL simulation"""
    
    def test_simulation_config(self):
        """Test simulation configuration"""
        from src.simulation.runner import SimulationConfig
        
        config = SimulationConfig(
            num_clients=5,
            num_rounds=3,
            num_byzantine=1
        )
        
        assert config.num_clients == 5
        assert config.num_rounds == 3
        assert config.num_byzantine == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
