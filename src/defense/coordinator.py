"""Multi-layer defense coordination"""

from src.defense.statistical import StatisticalDefenseLayer1, StatisticalDefenseLayer2
from src.defense.robust_agg import MultiKrumAggregator
from src.defense.reputation import ReputationManager


class DefenseCoordinator:
    """Coordinates all four defense layers"""
    
    def __init__(self, num_clients: int, num_galaxies: int, config: dict = None):
        """Initialize defense coordinator"""
        self.num_clients = num_clients
        self.num_galaxies = num_galaxies
        
        # Default configuration
        if config is None:
            config = {
                'layer1_threshold': 2.0,
                'layer2_threshold': 3.0,
                'layer3_krum_f': 1,
                'layer4_decay': 0.9
            }
        
        self.layer1 = StatisticalDefenseLayer1(threshold=config.get('layer1_threshold', 2.0))
        self.layer2 = StatisticalDefenseLayer2(threshold=config.get('layer2_threshold', 3.0))
        self.layer3 = MultiKrumAggregator(f=config.get('layer3_krum_f', 1))
        self.layer4 = ReputationManager(num_clients, decay_factor=config.get('layer4_decay', 0.9))
        
        self.detection_results = []
    
    def run_defense_pipeline(self, updates: list) -> dict:
        """Run all defense layers in sequence"""
        results = {
            'layer1_detections': [],
            'layer2_detections': [],
            'layer3_aggregation': None,
            'reputation_scores': {},
            'cleaned_updates': updates
        }
        
        # Layer 1: Statistical detection
        layer1_anomalies = self.layer1.detect_anomalies(updates)
        results['layer1_detections'] = layer1_anomalies
        
        # Layer 2: Multi-dimensional detection
        layer2_anomalies = self.layer2.detect_anomalies(updates)
        results['layer2_detections'] = layer2_anomalies
        
        # Combine detections
        all_anomalies = set(layer1_anomalies + layer2_anomalies)
        
        # Penalize detected clients in Layer 4
        for idx in all_anomalies:
            if idx < self.num_clients:
                self.layer4.penalize_client(idx)
        
        # Layer 3: Multi-Krum aggregation
        krum_result = self.layer3.aggregate(updates)
        results['layer3_aggregation'] = krum_result
        
        # Layer 4: Get reputation scores
        results['reputation_scores'] = self.layer4.get_all_reputations()
        
        self.detection_results.append(results)
        return results
    
    def get_detection_summary(self) -> dict:
        """Get summary of all detections"""
        if not self.detection_results:
            return {}
        
        latest = self.detection_results[-1]
        return {
            'layer1_count': len(latest['layer1_detections']),
            'layer2_count': len(latest['layer2_detections']),
            'total_detections': len(set(latest['layer1_detections'] + latest['layer2_detections']))
        }
