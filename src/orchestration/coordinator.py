"""Federated learning coordinator for round management and orchestration"""

import torch
from src.client.client import Client
from src.aggregators.galaxy import GalaxyAggregator
from src.aggregators.global_agg import GlobalAggregator
from src.defense.coordinator import DefenseCoordinator


class FLCoordinator:
    """Coordinates federated learning training rounds"""
    
    def __init__(self, num_clients: int, num_galaxies: int, model, config: dict):
        """Initialize FL coordinator"""
        self.num_clients = num_clients
        self.num_galaxies = num_galaxies
        self.model = model
        self.config = config
        
        self.clients = []
        self.galaxy_aggregators = []
        self.global_aggregator = GlobalAggregator(num_galaxies)
        self.defense_coordinator = DefenseCoordinator(num_clients, num_galaxies, config.get('defense', {}))
        
        self.current_round = 0
        self.round_history = []
    
    def initialize_clients(self, num_byzantine: int = 0):
        """Initialize clients with model copies"""
        for i in range(self.num_clients):
            is_byzantine = i < num_byzantine
            client = Client(i, self.model, is_byzantine=is_byzantine)
            self.clients.append(client)
    
    def initialize_galaxies(self):
        """Initialize galaxy aggregators"""
        clients_per_galaxy = self.num_clients // self.num_galaxies
        for g in range(self.num_galaxies):
            galaxy = GalaxyAggregator(g, clients_per_galaxy)
            start_idx = g * clients_per_galaxy
            for i in range(start_idx, start_idx + clients_per_galaxy):
                if i < self.num_clients:
                    galaxy.add_client(i)
            self.galaxy_aggregators.append(galaxy)
    
    def execute_round(self, client_data_loaders: list, byzantine_attack: str = None):
        """Execute one federated learning round"""
        round_data = {
            'round': self.current_round,
            'client_updates': [],
            'galaxy_updates': [],
            'defense_results': None,
            'global_update': None
        }
        
        # Client training
        for i, client in enumerate(self.clients):
            if i < len(client_data_loaders):
                client.train_local(client_data_loaders[i], self.config['fl']['local_epochs'])
                if client.is_byzantine and byzantine_attack:
                    client.attack(byzantine_attack)
                client.generate_commitment()
                round_data['client_updates'].append(client.get_update())
        
        # Galaxy aggregation
        clients_per_galaxy = len(self.clients) // len(self.galaxy_aggregators)
        for g_idx, galaxy in enumerate(self.galaxy_aggregators):
            start = g_idx * clients_per_galaxy
            end = start + clients_per_galaxy
            galaxy_updates = round_data['client_updates'][start:end]
            agg = galaxy.aggregate(galaxy_updates)
            if agg:
                round_data['galaxy_updates'].append(agg)
        
        # Defense pipeline
        defense_results = self.defense_coordinator.run_defense_pipeline(round_data['galaxy_updates'])
        round_data['defense_results'] = defense_results
        
        # Global aggregation
        global_update = self.global_aggregator.aggregate(round_data['galaxy_updates'])
        round_data['global_update'] = global_update
        
        self.current_round += 1
        self.round_history.append(round_data)
        
        return round_data
    
    def get_current_model(self):
        """Get the current global model"""
        return self.model
    
    def get_round_history(self):
        """Get history of all training rounds"""
        return self.round_history
