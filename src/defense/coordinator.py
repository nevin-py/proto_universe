"""Multi-layer defense coordination for Protogalaxy.

Coordinates all defense layers in sequence:
- Layer 1: Cryptographic integrity (Merkle verification)
- Layer 2: Statistical anomaly detection (3-metric analyzer)
- Layer 3: Byzantine-robust aggregation (Trimmed Mean or Multi-Krum)
- Layer 4: Reputation-based filtering
- Layer 5: Galaxy-level defense (Architecture Section 4.5)
"""

from typing import Optional, Dict, List
import torch

from src.defense.statistical import (
    StatisticalDefenseLayer1, 
    StatisticalDefenseLayer2,
    StatisticalAnalyzer
)
from src.defense.robust_agg import TrimmedMeanAggregator, MultiKrumAggregator, CoordinateWiseMedianAggregator
from src.defense.reputation import ReputationManager
from src.defense.layer5_galaxy import Layer5GalaxyDefense
from src.storage.forensic_logger import ForensicLogger


class DefenseCoordinator:
    """Coordinates all defense layers for Byzantine-resilient FL.
    
    Runs defense pipeline in sequence and tracks detection results.
    """
    
    AGGREGATOR_TRIMMED_MEAN = 'trimmed_mean'
    AGGREGATOR_MULTI_KRUM = 'multi_krum'
    AGGREGATOR_COORDINATE_MEDIAN = 'coordinate_wise_median'
    
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
                - use_full_analyzer: Use 3-metric StatisticalAnalyzer (PROTO-302)
        """
        self.num_clients = num_clients
        self.num_galaxies = num_galaxies
        
        # Default configuration
        if config is None:
            config = {}
        
        self.config = {
            'layer1_threshold': config.get('layer1_threshold', 3.0),
            'layer2_threshold': config.get('layer2_threshold', 3.0),
            'layer3_method': config.get('layer3_method', self.AGGREGATOR_TRIMMED_MEAN),
            'layer3_trim_ratio': config.get('layer3_trim_ratio', 0.1),
            'layer3_krum_f': config.get('layer3_krum_f', 1),
            'layer3_krum_m': config.get('layer3_krum_m', 1),
            'layer4_decay': config.get('layer4_decay', 0.9),
            'use_full_analyzer': config.get('use_full_analyzer', True),
            'cosine_threshold': config.get('cosine_threshold', 0.5),
        }
        
        # Initialize layers
        # Use the full 3-metric analyzer if configured (PROTO-302 compliant)
        self.use_full_analyzer = self.config['use_full_analyzer']
        
        if self.use_full_analyzer:
            self.statistical_analyzer = StatisticalAnalyzer(
                norm_threshold_sigma=self.config['layer1_threshold'],
                cosine_threshold=self.config['cosine_threshold'],
                coordinate_threshold_sigma=self.config['layer2_threshold']
            )
        else:
            # Legacy: separate Layer 1 and Layer 2
            self.layer1 = StatisticalDefenseLayer1(
                threshold=self.config['layer1_threshold']
            )
            self.layer2 = StatisticalDefenseLayer2(
                norm_threshold_sigma=self.config['layer2_threshold']
            )
        
        # Layer 3: Select aggregation method
        if self.config['layer3_method'] == self.AGGREGATOR_MULTI_KRUM:
            self.layer3 = MultiKrumAggregator(
                f=self.config['layer3_krum_f'],
                m=self.config['layer3_krum_m']
            )
        elif self.config['layer3_method'] == self.AGGREGATOR_COORDINATE_MEDIAN:
            self.layer3 = CoordinateWiseMedianAggregator()
        else:  # Default to Trimmed Mean
            self.layer3 = TrimmedMeanAggregator(
                trim_ratio=self.config['layer3_trim_ratio']
            )
        
        self.layer4 = ReputationManager(
            num_clients, 
            decay_factor=self.config['layer4_decay']
        )
        
        # Layer 5: Galaxy-level defense (Architecture Section 4.5)
        self.layer5 = Layer5GalaxyDefense(
            num_galaxies=num_galaxies,
            galaxy_rep_decay=self.config.get('layer5_galaxy_decay', 0.9),
            norm_threshold=self.config.get('layer5_norm_threshold', 3.0),
            direction_threshold=self.config.get('layer5_direction_threshold', 0.5),
            consistency_threshold=self.config.get('layer5_consistency_threshold', 0.7),
            dissolution_streak=self.config.get('layer5_dissolution_streak', 3)
        )
        
        # Forensic logger for quarantine/ban evidence (Architecture Section 3.3, 4.4)
        self.forensic_logger = ForensicLogger(
            storage_dir=config.get('forensic_storage_dir', './forensic_evidence')
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
        
        # Layer 1 & 2: Statistical detection
        if self.use_full_analyzer:
            # Use 4-metric analyzer (Architecture Section 4.2)
            analysis = self.statistical_analyzer.analyze(updates)
            results['layer1_detections'] = analysis.get('norm_outliers', [])
            results['layer2_detections'] = analysis.get('direction_outliers', [])
            results['coordinate_outliers'] = analysis.get('coordinate_outliers', [])
            results['distribution_outliers'] = analysis.get('distribution_outliers', [])
            results['statistical_flagged'] = analysis.get('flagged_clients', [])
            results['failure_counts'] = analysis.get('failure_counts', {})
            all_anomalies = set(analysis.get('flagged_indices', []))
        else:
            # Legacy: separate Layer 1 and Layer 2
            layer1_anomalies = self.layer1.detect_anomalies(updates)
            results['layer1_detections'] = layer1_anomalies
            
            layer2_anomalies = self.layer2.detect_anomalies(updates)
            results['layer2_detections'] = layer2_anomalies
            
            all_anomalies = set(layer1_anomalies + layer2_anomalies)
        
        # ── Filter flagged updates before L3 aggregation ──────────────
        # Architecture requires defense layers to feed forward: clients
        # flagged by L1/L2 statistical analysis should be excluded from
        # the robust aggregation in L3 so that poisoned gradients do not
        # contaminate the trimmed-mean / Krum computation.
        if all_anomalies:
            cleaned_updates = [
                u for i, u in enumerate(updates) if i not in all_anomalies
            ]
            # Safety: keep at least 1 update (never empty-aggregate)
            if len(cleaned_updates) < 1:
                cleaned_updates = updates  # fallback to full set
        else:
            cleaned_updates = updates
        results['cleaned_updates'] = cleaned_updates
        
        # Layer 3: Robust aggregation (on cleaned set)
        agg_result = self.layer3.aggregate(cleaned_updates)
        results['layer3_aggregation'] = agg_result
        
        # Store layer 3 specific info
        trimmed_or_rejected = set()
        if isinstance(self.layer3, TrimmedMeanAggregator):
            results['layer3_info'] = {
                'method': 'trimmed_mean',
                'frequently_trimmed': self.layer3.get_frequently_trimmed_clients()
            }
            trimmed_or_rejected = {idx for idx, _ in results['layer3_info']['frequently_trimmed']}
        elif isinstance(self.layer3, MultiKrumAggregator):
            results['layer3_info'] = {
                'method': 'multi_krum',
                'selected_indices': self.layer3.get_selected_indices()
            }
            # Clients NOT selected by Krum are implicitly rejected
            selected = set(self.layer3.get_selected_indices())
            trimmed_or_rejected = {i for i in range(len(updates))} - selected
        elif isinstance(self.layer3, CoordinateWiseMedianAggregator):
            results['layer3_info'] = {
                'method': 'coordinate_wise_median'
            }
            # CWMed uses all updates (median is inherently robust)
            trimmed_or_rejected = set()
        else:
            results['layer3_info'] = {'method': 'unknown'}
            trimmed_or_rejected = set()
        
        # =====================================================================
        # Layer 4: Full behavior score update (Architecture Section 4.4)
        # B(t) = w1*I_integrity + w2*I_statistical + w3*I_krum + w4*I_historical
        # Uses update_behavior_score() instead of simple penalize_client()
        # =====================================================================
        statistical_flagged = set(results.get('statistical_flagged', []))
        # Build a set of original-update indices that survived to cleaned_updates
        survived_indices = set(
            i for i in range(len(updates)) if i not in all_anomalies
        ) if all_anomalies else set(range(len(updates)))
        for i, update in enumerate(updates):
            client_id = update.get('client_id', i)
            if client_id >= self.num_clients:
                continue
            
            # Determine per-layer pass/fail for this client
            layer_results = {
                'integrity': True,  # Layer 1 integrity always passes if we reach here
                'statistical': client_id not in statistical_flagged and i not in all_anomalies,
                'krum': i in survived_indices and i not in trimmed_or_rejected,
            }
            
            # Use the full multi-indicator behavior scoring
            self.layer4.update_behavior_score(
                client_id=client_id,
                layer_results=layer_results,
                round_number=len(self.detection_results)
            )
        
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
        elif method == self.AGGREGATOR_COORDINATE_MEDIAN:
            self.layer3 = CoordinateWiseMedianAggregator()
            self.config['layer3_method'] = method
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def run_galaxy_defense(
        self,
        galaxy_updates: Dict[int, List[torch.Tensor]],
        client_assignments: Dict[int, int],
        round_number: int
    ) -> Dict:
        """Run Layer 5: Galaxy-level defense (Architecture Section 4.5)
        
        Args:
            galaxy_updates: Dict mapping galaxy_id -> aggregated gradients
            client_assignments: Client-to-galaxy mapping
            round_number: Current FL round
            
        Returns:
            Layer 5 defense results with anomaly reports and isolation decisions
        """
        client_reputations = self.layer4.get_all_reputations()
        
        # Run Layer 5 defense
        layer5_results = self.layer5.run_defense(
            galaxy_updates=galaxy_updates,
            client_assignments=client_assignments,
            client_reputations=client_reputations
        )
        
        # Log quarantine/ban decisions for flagged galaxies
        for galaxy_id in layer5_results['flagged_galaxies']:
            anomaly_report = layer5_results['anomaly_reports'][galaxy_id]
            
            # Get clients in this galaxy
            galaxy_clients = [
                cid for cid, gid in client_assignments.items()
                if gid == galaxy_id
            ]
            
            # Log evidence for each low-reputation client
            for client_id in galaxy_clients:
                rep = client_reputations.get(client_id, 1.0)
                
                if self.layer4.is_quarantined(client_id):
                    # Log quarantine with galaxy-level evidence
                    self.forensic_logger.log_quarantine(
                        client_id=client_id,
                        round_number=round_number,
                        commitment_hash="",  # To be filled by pipeline
                        merkle_proof=[],  # To be filled by pipeline
                        merkle_root="",  # To be filled by pipeline
                        layer_results={
                            'layer1_failed': False,
                            'layer2_flags': [],
                            'layer3_rejected': False,
                            'layer4_reputation': rep,
                            'layer5_galaxy_flagged': True,
                            'layer5_failed_checks': anomaly_report.failed_checks
                        },
                        metadata={
                            'galaxy_id': galaxy_id,
                            'galaxy_norm_deviation': anomaly_report.norm_deviation_score,
                            'galaxy_direction_similarity': anomaly_report.direction_similarity,
                            'galaxy_consistency': anomaly_report.cross_galaxy_consistency
                        }
                    )
                elif self.layer4.is_banned(client_id):
                    # Log permanent ban
                    self.forensic_logger.log_ban(
                        client_id=client_id,
                        round_number=round_number,
                        commitment_hash="",
                        merkle_proof=[],
                        merkle_root="",
                        layer_results={
                            'layer1_failed': False,
                            'layer2_flags': [],
                            'layer3_rejected': False,
                            'layer4_reputation': rep,
                            'layer5_galaxy_flagged': True,
                            'layer5_failed_checks': anomaly_report.failed_checks
                        },
                        metadata={'galaxy_id': galaxy_id}
                    )
        
        return layer5_results

