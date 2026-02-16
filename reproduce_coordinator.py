
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from src.defense.coordinator import DefenseCoordinator
from src.defense.reputation import EnhancedReputationManager

# Mock classes to avoid dependencies
class MockAnalyzer:
    def analyze(self, updates):
        # Flag client 3 (index 1 in this list if we pass [0, 3] etc? No, assume 10 clients)
        # We will pass 10 updates. Flag index 3.
        return {
            'flagged_clients': [3],
            'flagged_indices': [3],
            'norm_outliers': [],
            'direction_outliers': [],
            'coordinate_outliers': [],
            'distribution_outliers': [],
            'failure_counts': {3: 2}
        }

class MockAggregator:
    def aggregate(self, updates):
        return {'gradients': [torch.zeros(1)]} # Minimal return

def reproduce():
    # Setup
    num_clients = 10
    config = {
        'layer4_decay': 0.9,
        'layer3_trim_ratio': 0.1,
        'layer5_galaxy_decay': 0.9,
        'use_full_analyzer': True
    }
    
    coordinator = DefenseCoordinator(
        num_clients=num_clients,
        num_galaxies=1,
        config=config
    )
    
    # Mock components
    coordinator.statistical_analyzer = MockAnalyzer()
    coordinator.layer3 = MockAggregator()
    
    # Create updates for 10 clients
    updates = []
    for i in range(num_clients):
        updates.append({
            'client_id': i,
            'gradients': [torch.zeros(10)], # Dummy grads
            'reputation': 0.5
        })
        
    print(f"Initial Reputation Client 3: {coordinator.layer4.get_reputation(3)}")
    
    # Run pipeline
    results = coordinator.run_defense_pipeline(updates)
    
    print("Results:", results.keys())
    print("Statistical Flagged:", results.get('statistical_flagged'))
    
    # Check reputation of Client 3
    rep_3 = coordinator.layer4.get_reputation(3)
    print(f"Final Reputation Client 3: {rep_3}")
    
    # Calculated expected:
    # 0.9*0.5 + 0.1*0.47 = 0.497 (if failed)
    # 0.9*0.5 + 0.1*0.54 = 0.504 (if passed)
    
    if abs(rep_3 - 0.504) < 0.001:
        print("FAIL: Client 3 got 0.504 (PASSED) despite being flagged!")
    elif abs(rep_3 - 0.497) < 0.001:
        print("SUCCESS: Client 3 got 0.497 (FAILED) as expected.")
    else:
        print(f"Unknown result: {rep_3}")

if __name__ == "__main__":
    reproduce()
