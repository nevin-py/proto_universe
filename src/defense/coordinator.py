"""Multi-layer defense coordination for Protogalaxy.

Coordinates all defense layers in sequence:
- Layer 1: Cryptographic integrity (Merkle verification)
- Layer 2: Statistical anomaly detection
- Layer 3: Byzantine-robust aggregation (Trimmed Mean or Multi-Krum)
- Layer 4: Reputation-based filtering
"""

from typing import Optional

from src.defense.statistical import StatisticalDefenseLayer1, StatisticalDefenseLayer2
from src.defense.robust_agg import TrimmedMeanAggregator, MultiKrumAggregator
from src.defense.reputation import ReputationManager


class DefenseCoordinator:
    """Coordinates all defense layers for Byzantine-resilient FL.
    
    Runs defense pipeline in sequence and tracks detection results.
    """
    
    AGGREGATOR_TRIMMED_MEAN = 'trimmed_mean'
    AGGREGATOR_MULTI_KRUM = 'multi_krum'
    
    def __init__(self, num_clients: int, num_galaxies: int, config: Optional[dict] = None):
        """Initialize defense coordinator.
        
        Args:
            num_clients: Total number of clients in the system
            num_galaxies: Number of galaxies
            config: Configuration dict with keys:
                - layer1_threshold: Statistical threshold for z-score detection
                - layer2_threshold: Norm threshold for multi-dim detection
                - layer3_method: 'trimmed_mean' or 'multi_krum'
                - layer3_trim_ratio: Trim ratio for Trimmed Mean (default 0.1)
                - layer3_krum_f: Byzantine tolerance for Krum
                - layer4_decay: Reputation decay factor
        """
        self.num_clients = num_clients
        self.num_galaxies = num_galaxies
        
        # Default configuration
        if config is None:
            config = {}
        
        self.config = {
            'layer1_threshold': config.get('layer1_threshold', 2.0),
            'layer2_threshold': config.get('layer2_threshold', 3.0),
            'layer3_method': config.get('layer3_method', self.AGGREGATOR_TRIMMED_MEAN),
            'layer3_trim_ratio': config.get('layer3_trim_ratio', 0.1),
            'layer3_krum_f': config.get('layer3_krum_f', 1),
            'layer3_krum_m': config.get('layer3_krum_m', 1),
            'layer4_decay': config.get('layer4_decay', 0.9),
        }
        
        # Initialize layers
        self.layer1 = StatisticalDefenseLayer1(
            threshold=self.config['layer1_threshold']
        )
        self.layer2 = StatisticalDefenseLayer2(
            threshold=self.config['layer2_threshold']
        )
        
        # Layer 3: Select aggregation method
        if self.config['layer3_method'] == self.AGGREGATOR_MULTI_KRUM:
            self.layer3 = MultiKrumAggregator(
                f=self.config['layer3_krum_f'],
                m=self.config['layer3_krum_m']
            )
        else:  # Default to Trimmed Mean
            self.layer3 = TrimmedMeanAggregator(
                trim_ratio=self.config['layer3_trim_ratio']
            )
        
        self.layer4 = ReputationManager(
            num_clients, 
            decay_factor=self.config['layer4_decay']
        )
        
        self.detection_results = []
    
    def run_defense_pipeline(self, updates: list) -> dict:
        """Run all defense layers in sequence.
        
        Args:
            updates: List of update dicts with 'gradients' key
            
        Returns:
            Dict with detection results and aggregated gradient
        """
        results = {
            'layer1_detections': [],
            'layer2_detections': [],
            'layer3_aggregation': None,
            'layer3_info': {},
            'reputation_scores': {},
            'cleaned_updates': updates
        }
        
        # Layer 1: Statistical z-score detection
        layer1_anomalies = self.layer1.detect_anomalies(updates)
        results['layer1_detections'] = layer1_anomalies
        
        # Layer 2: Multi-dimensional norm detection
        layer2_anomalies = self.layer2.detect_anomalies(updates)
        results['layer2_detections'] = layer2_anomalies
        
        # Combine detections from statistical layers
        all_anomalies = set(layer1_anomalies + layer2_anomalies)
        
        # Penalize detected clients in Layer 4 (reputation)
        for idx in all_anomalies:
            if idx < self.num_clients:
                self.layer4.penalize_client(idx)
        
        # Layer 3: Robust aggregation
        agg_result = self.layer3.aggregate(updates)
        results['layer3_aggregation'] = agg_result
        
        # Store layer 3 specific info
        if isinstance(self.layer3, TrimmedMeanAggregator):
            results['layer3_info'] = {
                'method': 'trimmed_mean',
                'frequently_trimmed': self.layer3.get_frequently_trimmed_clients()
            }
            # Penalize frequently trimmed clients
            for client_idx, trim_fraction in results['layer3_info']['frequently_trimmed']:
                if client_idx < self.num_clients:
                    self.layer4.penalize_client(client_idx, penalty=trim_fraction * 0.2)
        else:
            results['layer3_info'] = {
                'method': 'multi_krum',
                'selected_indices': self.layer3.get_selected_indices()
            }
        
        # Layer 4: Get reputation scores
        results['reputation_scores'] = self.layer4.get_all_reputations()
        
        self.detection_results.append(results)
        return results
    
    def get_detection_summary(self) -> dict:
        """Get summary of all detections.
        
        Returns:
            Dict with detection counts and stats
        """
        if not self.detection_results:
            return {}
        
        latest = self.detection_results[-1]
        return {
            'layer1_count': len(latest['layer1_detections']),
            'layer2_count': len(latest['layer2_detections']),
            'total_detections': len(set(latest['layer1_detections'] + latest['layer2_detections'])),
            'aggregation_method': latest['layer3_info'].get('method', 'unknown'),
        }
    
    def get_suspicious_clients(self, threshold: float = 0.5) -> list:
        """Get list of clients with low reputation.
        
        Args:
            threshold: Reputation threshold below which clients are suspicious
            
        Returns:
            List of (client_id, reputation) tuples
        """
        suspicious = []
        for cid, rep in self.layer4.get_all_reputations().items():
            if rep < threshold:
                suspicious.append((cid, rep))
        return sorted(suspicious, key=lambda x: x[1])
    
    def set_aggregation_method(self, method: str, **kwargs):
        """Switch aggregation method.
        
        Args:
            method: 'trimmed_mean' or 'multi_krum'
            **kwargs: Method-specific parameters
        """
        if method == self.AGGREGATOR_TRIMMED_MEAN:
            trim_ratio = kwargs.get('trim_ratio', self.config['layer3_trim_ratio'])
            self.layer3 = TrimmedMeanAggregator(trim_ratio=trim_ratio)
            self.config['layer3_method'] = method
        elif method == self.AGGREGATOR_MULTI_KRUM:
            f = kwargs.get('f', self.config['layer3_krum_f'])
            m = kwargs.get('m', self.config['layer3_krum_m'])
            self.layer3 = MultiKrumAggregator(f=f, m=m)
            self.config['layer3_method'] = method
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
