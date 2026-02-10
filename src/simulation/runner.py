"""Complete FL Simulation Runner for ProtoGalaxy.

Provides a comprehensive simulation framework that:
- Sets up clients and galaxies
- Runs FL training rounds
- Applies Byzantine attacks
- Collects and exports metrics
"""

import time
import copy
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.client.trainer import Trainer
from src.aggregators.galaxy import GalaxyAggregator
from src.aggregators.global_agg import GlobalAggregator
from src.defense.coordinator import DefenseCoordinator
from src.orchestration.round_manager import RoundManager, RoundPhase
from src.logging import FLLogger, FLLoggerFactory
from src.simulation.metrics import MetricsCollector


@dataclass
class SimulationConfig:
    """Configuration for FL simulation"""
    # Basic settings
    num_clients: int = 10
    num_galaxies: int = 2
    num_rounds: int = 10
    
    # Training settings
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Byzantine settings
    num_byzantine: int = 0
    attack_type: str = "gradient_poison"  # gradient_poison, label_flip, none
    attack_strength: float = 1.0
    
    # Defense settings
    enable_defense: bool = True
    defense_method: str = "trimmed_mean"  # trimmed_mean, multi_krum
    trim_ratio: float = 0.1
    
    # Evaluation settings
    eval_every: int = 1  # Evaluate every N rounds
    
    # Logging
    log_dir: str = "outputs/logs"
    verbose: bool = True


@dataclass
class ClientState:
    """State of a simulated client"""
    client_id: int
    galaxy_id: int
    is_byzantine: bool = False
    trainer: Optional[Trainer] = None
    data_loader: Optional[DataLoader] = None
    reputation: float = 1.0
    rounds_participated: int = 0


class FLSimulation:
    """Federated Learning Simulation Runner.
    
    Complete simulation framework for testing FL algorithms
    with Byzantine clients and defense mechanisms.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SimulationConfig,
        train_loaders: Dict[int, DataLoader],
        test_loader: DataLoader,
        logger: Optional[FLLogger] = None
    ):
        """Initialize FL simulation.
        
        Args:
            model: Global model to train
            config: Simulation configuration
            train_loaders: Dict mapping client_id to training DataLoader
            test_loader: Test data loader for evaluation
            logger: FL logger instance
        """
        self.model = model
        self.config = config
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        
        # Setup logging
        FLLoggerFactory.configure(log_dir=config.log_dir)
        self.logger = logger or FLLoggerFactory.get_global_logger()
        
        # Initialize clients
        self.clients: Dict[int, ClientState] = {}
        self._setup_clients()
        
        # Initialize aggregators
        self.galaxy_aggregators: Dict[int, GalaxyAggregator] = {}
        self.global_aggregator: Optional[GlobalAggregator] = None
        self._setup_aggregators()
        
        # Defense coordinator
        self.defense = DefenseCoordinator(
            num_clients=config.num_clients,
            num_galaxies=config.num_galaxies,
            config={
                'layer3_method': config.defense_method,
                'layer3_trim_ratio': config.trim_ratio
            }
        ) if config.enable_defense else None
        
        # Round management
        self.round_manager = RoundManager(
            config={'min_participation': 0.8},
            logger=self.logger.logger
        )
        
        # Metrics collection
        self.metrics = MetricsCollector()
        
        # Training state
        self.current_round = 0
        self.global_weights: List[torch.Tensor] = self._get_model_weights()
    
    def _setup_clients(self) -> None:
        """Setup simulated clients"""
        clients_per_galaxy = self.config.num_clients // self.config.num_galaxies
        
        for i in range(self.config.num_clients):
            galaxy_id = i // clients_per_galaxy
            is_byzantine = i < self.config.num_byzantine
            
            # Create trainer with model copy
            model_copy = copy.deepcopy(self.model)
            trainer = Trainer(
                model_copy,
                learning_rate=self.config.learning_rate
            )
            
            self.clients[i] = ClientState(
                client_id=i,
                galaxy_id=galaxy_id,
                is_byzantine=is_byzantine,
                trainer=trainer,
                data_loader=self.train_loaders.get(i)
            )
        
        byzantine_ids = [c.client_id for c in self.clients.values() if c.is_byzantine]
        self.logger.info(
            f"Setup {self.config.num_clients} clients "
            f"({self.config.num_byzantine} Byzantine: {byzantine_ids})"
        )
    
    def _setup_aggregators(self) -> None:
        """Setup galaxy and global aggregators"""
        clients_per_galaxy = self.config.num_clients // self.config.num_galaxies
        
        for g in range(self.config.num_galaxies):
            self.galaxy_aggregators[g] = GalaxyAggregator(
                galaxy_id=g,
                num_clients=clients_per_galaxy
            )
        
        self.global_aggregator = GlobalAggregator(
            num_galaxies=self.config.num_galaxies
        )
    
    def _get_model_weights(self) -> List[torch.Tensor]:
        """Get current global model weights"""
        return [p.data.clone() for p in self.model.parameters()]
    
    def _set_model_weights(self, weights: List[torch.Tensor]) -> None:
        """Set global model weights"""
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.data.copy_(weight)
    
    def _distribute_model(self) -> None:
        """Distribute global model to all clients"""
        weights = self._get_model_weights()
        
        for client in self.clients.values():
            if client.trainer:
                client.trainer.set_weights(weights)
    
    def _apply_byzantine_attack(
        self,
        client: ClientState,
        gradients: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply Byzantine attack to gradients.
        
        Args:
            client: Client state
            gradients: Original gradients
            
        Returns:
            Poisoned gradients
        """
        if not client.is_byzantine:
            return gradients
        
        attack_type = self.config.attack_type
        strength = self.config.attack_strength
        
        if attack_type == "gradient_poison":
            # Scale and flip gradients
            return [g * -strength * 10 for g in gradients]
        
        elif attack_type == "noise":
            # Add random noise
            return [
                g + torch.randn_like(g) * strength * g.std()
                for g in gradients
            ]
        
        elif attack_type == "zero":
            # Send zero gradients
            return [torch.zeros_like(g) for g in gradients]
        
        elif attack_type == "label_flip":
            # Label flip attack is applied during training
            # Gradients are already affected
            return gradients
        
        return gradients
    
    def _train_client(self, client: ClientState) -> Optional[List[torch.Tensor]]:
        """Train a single client and get gradients.
        
        Args:
            client: Client state
            
        Returns:
            Client gradients or None if training failed
        """
        if client.trainer is None or client.data_loader is None:
            return None
        
        # Train locally
        metrics = client.trainer.train(
            client.data_loader,
            num_epochs=self.config.local_epochs
        )
        
        # Get gradients (weight difference)
        gradients = client.trainer.get_gradients()
        
        if not gradients:
            return None
        
        # Apply Byzantine attack if applicable
        gradients = self._apply_byzantine_attack(client, gradients)
        
        # Log training
        self.logger.log_training(
            client_id=str(client.client_id),
            loss=metrics.get('loss', 0),
            accuracy=metrics.get('accuracy'),
            epochs=self.config.local_epochs
        )
        
        client.rounds_participated += 1
        
        return gradients
    
    def _aggregate_gradients(
        self,
        gradient_dict: Dict[int, List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Aggregate gradients from all clients.
        
        Args:
            gradient_dict: Dict mapping client_id to gradients
            
        Returns:
            Aggregated gradients
        """
        if not gradient_dict:
            return []
        
        # Apply defense if enabled
        if self.defense:
            # Convert to format expected by defense
            updates = [
                {'client_id': cid, 'gradients': grads}
                for cid, grads in gradient_dict.items()
            ]
            
            defense_results = self.defense.run_defense_pipeline(updates)
            
            # Log detections
            detected = (
                defense_results.get('layer1_detections', []) +
                defense_results.get('layer2_detections', [])
            )
            if detected:
                self.logger.log_detection(
                    detected_clients=[str(d) for d in detected],
                    detection_type=self.config.defense_method,
                    layer="defense"
                )
            
            # Use cleaned gradients if available
            if defense_results.get('layer3_aggregation') is not None:
                return defense_results['layer3_aggregation']
        
        # Simple averaging (FedAvg)
        gradient_lists = list(gradient_dict.values())
        num_clients = len(gradient_lists)
        
        aggregated = [g.clone() / num_clients for g in gradient_lists[0]]
        for grads in gradient_lists[1:]:
            for i, g in enumerate(grads):
                aggregated[i] += g / num_clients
        
        return aggregated
    
    def run_round(self) -> Dict:
        """Execute one federated learning round.
        
        Returns:
            Dictionary with round results
        """
        self.current_round += 1
        round_start = time.time()
        
        self.logger.set_round(self.current_round)
        self.logger.log_round_start(
            self.current_round,
            self.config.num_clients,
            {'local_epochs': self.config.local_epochs}
        )
        
        # Phase 1: Distribute model
        self._distribute_model()
        
        # Phase 2: Client training
        gradient_dict: Dict[int, List[torch.Tensor]] = {}
        
        for client_id, client in self.clients.items():
            gradients = self._train_client(client)
            if gradients:
                gradient_dict[client_id] = gradients
        
        # Phase 3: Aggregation
        if gradient_dict:
            aggregated = self._aggregate_gradients(gradient_dict)
            
            if aggregated:
                # Apply aggregated gradient to global model
                self._apply_global_update(aggregated)
                
                self.logger.log_aggregation(
                    num_gradients=len(gradient_dict),
                    method="fedavg" if not self.defense else self.config.defense_method
                )
        
        # Phase 4: Evaluation (if scheduled)
        eval_metrics = {}
        if self.current_round % self.config.eval_every == 0:
            eval_metrics = self._evaluate_global_model()
        
        round_duration = time.time() - round_start
        
        # Record metrics
        self.metrics.record_round_metrics(
            self.current_round,
            eval_metrics.get('accuracy', 0),
            eval_metrics.get('loss', 0)
        )
        
        self.logger.log_round_end(
            self.current_round,
            round_duration,
            eval_metrics
        )
        
        return {
            'round': self.current_round,
            'duration': round_duration,
            'num_participants': len(gradient_dict),
            'metrics': eval_metrics
        }
    
    def _apply_global_update(self, gradients: List[torch.Tensor]) -> None:
        """Apply aggregated gradients to global model."""
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), gradients):
                param.data -= self.config.learning_rate * grad
        
        self.global_weights = self._get_model_weights()
    
    def _evaluate_global_model(self) -> Dict:
        """Evaluate the global model on test data."""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        
        self.logger.log_metrics({
            'accuracy': accuracy,
            'loss': avg_loss
        })
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def run(self) -> Dict:
        """Run complete FL simulation.
        
        Returns:
            Dictionary with final results
        """
        self.logger.info(f"Starting simulation with {self.config.num_rounds} rounds")
        sim_start = time.time()
        
        round_results = []
        for _ in range(self.config.num_rounds):
            result = self.run_round()
            round_results.append(result)
            
            if self.config.verbose and result['metrics']:
                print(f"Round {result['round']}: "
                      f"Acc={result['metrics'].get('accuracy', 0):.4f}, "
                      f"Loss={result['metrics'].get('loss', 0):.4f}")
        
        sim_duration = time.time() - sim_start
        
        # Final evaluation
        final_metrics = self._evaluate_global_model()
        
        # Summary
        summary = {
            'total_rounds': self.config.num_rounds,
            'total_duration': sim_duration,
            'final_accuracy': final_metrics['accuracy'],
            'final_loss': final_metrics['loss'],
            'num_byzantine': self.config.num_byzantine,
            'defense_enabled': self.config.enable_defense,
            'round_results': round_results,
            'metrics_summary': self.metrics.get_summary()
        }
        
        self.logger.info(
            f"Simulation complete. Final accuracy: {final_metrics['accuracy']:.4f}"
        )
        
        # Export metrics
        self.logger.export_metrics_csv()
        
        return summary
    
    def get_client_stats(self) -> Dict:
        """Get statistics for all clients."""
        return {
            client_id: {
                'galaxy_id': client.galaxy_id,
                'is_byzantine': client.is_byzantine,
                'reputation': client.reputation,
                'rounds_participated': client.rounds_participated
            }
            for client_id, client in self.clients.items()
        }


def run_simulation(
    model: nn.Module,
    train_dataset,
    test_dataset,
    config: Optional[SimulationConfig] = None
) -> Dict:
    """Convenience function to run a simulation.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        test_dataset: Test dataset
        config: Simulation configuration
        
    Returns:
        Simulation results
    """
    from src.data.partition import IIDPartitioner
    from src.data.loader import create_client_loaders, create_test_loader
    
    config = config or SimulationConfig()
    
    # Partition data
    partitioner = IIDPartitioner()
    partitions = partitioner.partition(train_dataset, config.num_clients)
    
    # Create loaders
    train_loaders = create_client_loaders(
        train_dataset, partitions, config.batch_size
    )
    test_loader = create_test_loader(test_dataset, config.batch_size)
    
    # Run simulation
    sim = FLSimulation(
        model=model,
        config=config,
        train_loaders=train_loaders,
        test_loader=test_loader
    )
    
    return sim.run()
