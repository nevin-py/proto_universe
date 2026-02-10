# ProtoGalaxy MVP - File Structure

## Project Directory Structure

```
protogalaxy/
├── README.md
├── requirements.txt
├── setup.py
├── config.yaml
├── .gitignore
│
├── docs/
│   ├── architecture.md
│   ├── api_specification.md
│   ├── setup_guide.md
│   └── user_guide.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── crypto/                          # Epic 1: Core Cryptographic Infrastructure
│   │   ├── __init__.py
│   │   ├── merkle_tree.py              # PROTO-101, 102, 103
│   │   ├── commitment.py                # PROTO-104
│   │   ├── galaxy_merkle.py            # PROTO-105
│   │   ├── global_merkle.py            # PROTO-106
│   │   └── crypto_utils.py             # PROTO-1003
│   │
│   ├── client/                          # Epic 2: Client Components
│   │   ├── __init__.py
│   │   ├── local_trainer.py            # PROTO-201
│   │   ├── commitment_generator.py     # PROTO-202
│   │   ├── proof_verifier.py           # PROTO-203
│   │   ├── client_communication.py     # PROTO-205
│   │   └── client.py                   # Main client class
│   │
│   ├── galaxy/                          # Epic 3: Galaxy Aggregator Components
│   │   ├── __init__.py
│   │   ├── merkle_constructor.py       # PROTO-301
│   │   ├── statistical_analyzer.py     # PROTO-302
│   │   ├── robust_aggregator.py        # PROTO-303
│   │   ├── reputation_manager.py       # PROTO-304
│   │   ├── galaxy_communication.py     # PROTO-306
│   │   └── galaxy_aggregator.py        # Main galaxy aggregator class
│   │
│   ├── global_agg/                      # Epic 4: Global Aggregator Components
│   │   ├── __init__.py
│   │   ├── global_merkle_constructor.py # PROTO-401
│   │   ├── final_aggregator.py         # PROTO-403
│   │   ├── model_manager.py            # PROTO-404
│   │   ├── global_communication.py     # PROTO-407
│   │   └── global_aggregator.py        # Main global aggregator class
│   │
│   ├── defense/                         # Epic 5: Multi-Layer Defense System
│   │   ├── __init__.py
│   │   ├── layer1_verification.py      # PROTO-501
│   │   ├── layer2_statistical.py       # PROTO-502
│   │   ├── layer3_robust_agg.py        # PROTO-503
│   │   ├── layer4_reputation.py        # PROTO-504
│   │   └── defense_coordinator.py      # PROTO-505
│   │
│   ├── config/                          # Epic 6: Configuration
│   │   ├── __init__.py
│   │   ├── config_manager.py           # PROTO-701
│   │   └── logger.py                   # PROTO-702
│   │
│   ├── orchestration/                   # Epic 7: Orchestration
│   │   ├── __init__.py
│   │   ├── fl_coordinator.py           # PROTO-1101
│   │   ├── round_manager.py            # PROTO-1102
│   │   └── model_sync.py               # PROTO-1104
│   │
│   ├── utils/                           # Epic 8: Utilities
│   │   ├── __init__.py
│   │   ├── gradient_utils.py           # PROTO-1001
│   │   ├── statistical_utils.py        # PROTO-1002
│   │   └── data_structures.py          # Helper data structures
│   │
│   ├── galaxy_management/               # Epic 9: Galaxy Management
│   │   ├── __init__.py
│   │   └── galaxy_assignment.py        # PROTO-801
│   │
│   ├── storage/                         # Epic 10: Data Management
│   │   ├── __init__.py
│   │   ├── gradient_store.py           # PROTO-1201
│   │   ├── reputation_store.py         # PROTO-1203
│   │   └── checkpoint_manager.py       # PROTO-1205
│   │
│   ├── simulation/                      # Epic 11: Simulation
│   │   ├── __init__.py
│   │   ├── honest_client_simulator.py  # PROTO-1501
│   │   ├── byzantine_simulator.py      # PROTO-1502
│   │   ├── metrics_collector.py        # PROTO-1503
│   │   └── dataset_manager.py          # Helper for data distribution
│   │
│   ├── models/                          # Neural network models
│   │   ├── __init__.py
│   │   ├── mnist_cnn.py                # Simple CNN for MNIST
│   │   └── model_registry.py           # Model factory
│   │
│   └── api/                             # REST API definitions
│       ├── __init__.py
│       ├── client_api.py               # Client endpoints
│       ├── galaxy_api.py               # Galaxy aggregator endpoints
│       ├── global_api.py               # Global aggregator endpoints
│       └── schemas.py                  # API request/response schemas
│
├── tests/                               # Epic 12: Testing
│   ├── __init__.py
│   │
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_merkle_tree.py
│   │   ├── test_commitment.py
│   │   ├── test_statistical_analyzer.py
│   │   ├── test_robust_aggregator.py
│   │   ├── test_reputation_manager.py
│   │   ├── test_gradient_utils.py
│   │   └── test_crypto_utils.py
│   │
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_client_galaxy.py
│   │   ├── test_galaxy_global.py
│   │   ├── test_defense_layers.py
│   │   └── test_fl_round.py
│   │
│   └── e2e/
│       ├── __init__.py
│       └── test_end_to_end.py          # PROTO-1105
│
├── scripts/
│   ├── run_client.py                   # Start a single client
│   ├── run_galaxy_aggregator.py        # Start a galaxy aggregator
│   ├── run_global_aggregator.py        # Start global aggregator
│   ├── run_simulation.py               # Run full simulation
│   ├── run_experiment.py               # Run experiment with different configs
│   ├── analyze_results.py              # Analyze experiment results
│   └── visualize_metrics.py            # Plot training metrics
│
├── experiments/                         # Experiment configurations and results
│   ├── configs/
│   │   ├── baseline.yaml
│   │   ├── label_flip_attack.yaml
│   │   ├── gradient_poison_attack.yaml
│   │   └── adaptive_attack.yaml
│   │
│   └── results/
│       ├── baseline/
│       ├── label_flip/
│       └── gradient_poison/
│
├── data/                                # Data directory (gitignored)
│   ├── mnist/
│   │   ├── train/
│   │   └── test/
│   └── partitions/                     # Client data partitions
│       ├── client_0/
│       ├── client_1/
│       └── ...
│
├── outputs/                             # Runtime outputs (gitignored)
│   ├── logs/
│   │   ├── client_0.log
│   │   ├── galaxy_0.log
│   │   └── global.log
│   │
│   ├── models/                         # Saved model checkpoints
│   │   ├── round_0.pt
│   │   ├── round_5.pt
│   │   └── round_10.pt
│   │
│   ├── metrics/                        # Training metrics
│   │   ├── accuracy.csv
│   │   ├── detections.csv
│   │   └── reputation.csv
│   │
│   ├── reputation/                     # Reputation data
│   │   └── reputation.json
│   │
│   └── visualizations/                 # Generated plots
│       ├── accuracy_plot.png
│       ├── detection_rate.png
│       └── reputation_heatmap.png
│
└── notebooks/                          # Jupyter notebooks for analysis
    ├── 01_data_exploration.ipynb
    ├── 02_merkle_tree_demo.ipynb
    ├── 03_defense_analysis.ipynb
    └── 04_results_visualization.ipynb
```

---

## Detailed File Descriptions

### Root Level Files

**README.md**
```markdown
# ProtoGalaxy MVP

Hierarchical Federated Learning with Multi-Layer Byzantine Defense

## Quick Start
pip install -r requirements.txt
python scripts/run_simulation.py

## Documentation
See docs/ directory for detailed documentation.
```

**requirements.txt**
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
flask>=2.3.0
requests>=2.31.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
pytest>=7.3.0
```

**setup.py**
```python
from setuptools import setup, find_packages

setup(
    name="protogalaxy",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[...],
)
```

**config.yaml**
```yaml
# System Configuration
system:
  num_clients: 30
  num_galaxies: 3
  byzantine_threshold: 0.3
  
# Training Configuration
training:
  num_rounds: 10
  local_epochs: 1
  batch_size: 32
  learning_rate: 0.01
  
# Defense Configuration
defense:
  reputation_threshold: 0.3
  ewma_decay: 0.1
  statistical_threshold_sigma: 3.0
  cosine_threshold: 0.5
  
# Network Configuration
network:
  client_timeout: 30
  galaxy_timeout: 60
  global_timeout: 120
```

**.gitignore**
```
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/

# Data
data/
outputs/
*.pt
*.pth

# Logs
*.log

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

---

### Core Module Files

#### src/crypto/merkle_tree.py
```python
"""
PROTO-101, 102, 103: Base Merkle Tree Implementation

This module implements the fundamental Merkle tree data structure
with proof generation and verification capabilities.
"""

import hashlib
from typing import List, Tuple, Optional

class MerkleTree:
    """Merkle tree implementation for gradient verification."""
    
    def __init__(self):
        self.tree = []
        self.root = None
        
    def build(self, leaves: List[bytes]) -> bytes:
        """
        Build Merkle tree from leaf hashes.
        
        Args:
            leaves: List of leaf hashes
            
        Returns:
            Root hash of the tree
        """
        pass
        
    def get_proof(self, index: int) -> List[Tuple[bytes, str]]:
        """
        Generate Merkle proof for leaf at index.
        
        Args:
            index: Index of leaf to generate proof for
            
        Returns:
            List of (hash, direction) tuples
        """
        pass
        
    @staticmethod
    def verify_proof(leaf: bytes, proof: List[Tuple[bytes, str]], 
                    root: bytes) -> bool:
        """
        Verify Merkle proof.
        
        Args:
            leaf: Leaf hash to verify
            proof: Merkle proof path
            root: Expected root hash
            
        Returns:
            True if proof is valid, False otherwise
        """
        pass
```

#### src/crypto/commitment.py
```python
"""
PROTO-104: Gradient Commitment Hash

Implements cryptographic commitment for gradients with metadata.
"""

import hashlib
import json
import time
from typing import Dict, Any
import torch

def create_commitment(gradient: torch.Tensor, 
                     client_id: str, 
                     round_num: int) -> bytes:
    """
    Create cryptographic commitment for gradient.
    
    Args:
        gradient: Gradient tensor
        client_id: Client identifier
        round_num: Current round number
        
    Returns:
        SHA-256 hash commitment
    """
    pass
    
def serialize_gradient(gradient: torch.Tensor) -> bytes:
    """Serialize gradient tensor to bytes."""
    pass
```

#### src/client/client.py
```python
"""
Main Client Class

Integrates all client components: training, commitment, verification.
"""

from typing import Optional
import torch
import torch.nn as nn

class FederatedClient:
    """Federated learning client."""
    
    def __init__(self, client_id: str, galaxy_id: int, 
                 model: nn.Module, dataset):
        self.client_id = client_id
        self.galaxy_id = galaxy_id
        self.model = model
        self.dataset = dataset
        self.local_trainer = None
        self.commitment_generator = None
        self.proof_verifier = None
        
    def train_local_model(self, epochs: int = 1) -> torch.Tensor:
        """Train model on local data and return gradients."""
        pass
        
    def submit_update(self) -> Dict[str, Any]:
        """Generate and submit gradient update with commitment."""
        pass
        
    def verify_inclusion(self, proof: Dict) -> bool:
        """Verify gradient was included in Merkle tree."""
        pass
        
    def update_global_model(self, model_state: Dict):
        """Update local model with global parameters."""
        pass
```

#### src/galaxy/galaxy_aggregator.py
```python
"""
Main Galaxy Aggregator Class

Coordinates galaxy-level aggregation with defense mechanisms.
"""

from typing import List, Dict, Any
import torch

class GalaxyAggregator:
    """Galaxy-level aggregator with Byzantine defense."""
    
    def __init__(self, galaxy_id: int, config: Dict):
        self.galaxy_id = galaxy_id
        self.config = config
        self.merkle_constructor = None
        self.statistical_analyzer = None
        self.robust_aggregator = None
        self.reputation_manager = None
        self.defense_coordinator = None
        
    def collect_gradients(self, gradients: List[Dict]) -> None:
        """Collect gradients from clients in this galaxy."""
        pass
        
    def build_merkle_tree(self) -> bytes:
        """Build Merkle tree from client commitments."""
        pass
        
    def detect_anomalies(self) -> List[int]:
        """Run statistical analysis to detect anomalies."""
        pass
        
    def aggregate_gradients(self) -> torch.Tensor:
        """Apply robust aggregation with defense layers."""
        pass
        
    def generate_proofs(self) -> Dict[str, Any]:
        """Generate Merkle proofs for all clients."""
        pass
        
    def update_reputations(self, flagged_clients: List[int]) -> None:
        """Update client reputation scores."""
        pass
```

#### src/global_agg/global_aggregator.py
```python
"""
Main Global Aggregator Class

Coordinates global-level aggregation across galaxies.
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn

class GlobalAggregator:
    """Global aggregator coordinating all galaxies."""
    
    def __init__(self, model: nn.Module, num_galaxies: int, config: Dict):
        self.model = model
        self.num_galaxies = num_galaxies
        self.config = config
        self.global_merkle_constructor = None
        self.model_manager = None
        
    def collect_galaxy_updates(self, updates: List[Dict]) -> None:
        """Collect aggregated updates from all galaxies."""
        pass
        
    def build_global_merkle_tree(self) -> bytes:
        """Build global Merkle tree from galaxy roots."""
        pass
        
    def aggregate_global_model(self) -> None:
        """Perform final global aggregation."""
        pass
        
    def distribute_model(self) -> Dict[str, Any]:
        """Distribute updated global model."""
        pass
        
    def save_checkpoint(self, round_num: int) -> None:
        """Save model checkpoint."""
        pass
```

#### src/orchestration/fl_coordinator.py
```python
"""
PROTO-1101: Federated Learning Coordinator

Main orchestrator for FL training rounds.
"""

from typing import List
import torch.nn as nn

class FLCoordinator:
    """Coordinates multi-round federated learning."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.clients = []
        self.galaxy_aggregators = []
        self.global_aggregator = None
        self.round_manager = None
        self.metrics_collector = None
        
    def initialize_system(self) -> None:
        """Initialize all system components."""
        pass
        
    def run_training(self, num_rounds: int) -> Dict:
        """Execute federated learning for specified rounds."""
        pass
        
    def execute_round(self, round_num: int) -> Dict:
        """Execute a single FL round."""
        pass
        
    def collect_metrics(self) -> Dict:
        """Collect and return training metrics."""
        pass
```

---

### Script Files

#### scripts/run_simulation.py
```python
"""
Main simulation script for ProtoGalaxy MVP.

Runs complete federated learning simulation with Byzantine clients.
"""

import argparse
import yaml
import logging
from pathlib import Path

from src.orchestration.fl_coordinator import FLCoordinator
from src.simulation.honest_client_simulator import HonestClientSimulator
from src.simulation.byzantine_simulator import ByzantineClientSimulator
from src.simulation.metrics_collector import MetricsCollector

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--attack', type=str, default='label_flip',
                       choices=['none', 'label_flip', 'gradient_poison'])
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize coordinator
    coordinator = FLCoordinator(config)
    
    # Create clients (honest + Byzantine)
    num_honest = int(config['system']['num_clients'] * 0.7)
    num_byzantine = config['system']['num_clients'] - num_honest
    
    # Run training
    results = coordinator.run_training(config['training']['num_rounds'])
    
    # Save results
    print(f"Final accuracy: {results['accuracy']:.2f}%")
    print(f"Detection rate: {results['detection_rate']:.2f}%")

if __name__ == '__main__':
    main()
```

#### scripts/run_experiment.py
```python
"""
Run experiments with different configurations and attacks.
"""

import yaml
import json
from pathlib import Path
from run_simulation import main as run_simulation

def run_experiment(experiment_config: str):
    """Run single experiment."""
    # Load experiment config
    # Run simulation
    # Save results
    pass

def run_all_experiments():
    """Run all predefined experiments."""
    experiments = [
        'baseline.yaml',
        'label_flip_attack.yaml',
        'gradient_poison_attack.yaml',
    ]
    
    for exp in experiments:
        run_experiment(exp)

if __name__ == '__main__':
    run_all_experiments()
```

---

### Test Files

#### tests/unit/test_merkle_tree.py
```python
"""
Unit tests for Merkle tree implementation.
"""

import pytest
import hashlib
from src.crypto.merkle_tree import MerkleTree

class TestMerkleTree:
    
    def test_build_tree(self):
        """Test Merkle tree construction."""
        leaves = [hashlib.sha256(str(i).encode()).digest() 
                 for i in range(8)]
        tree = MerkleTree()
        root = tree.build(leaves)
        assert root is not None
        assert len(root) == 32  # SHA-256 hash size
        
    def test_proof_generation(self):
        """Test proof generation for leaf."""
        leaves = [hashlib.sha256(str(i).encode()).digest() 
                 for i in range(8)]
        tree = MerkleTree()
        root = tree.build(leaves)
        proof = tree.get_proof(0)
        assert len(proof) == 3  # log2(8) = 3
        
    def test_proof_verification(self):
        """Test proof verification."""
        leaves = [hashlib.sha256(str(i).encode()).digest() 
                 for i in range(8)]
        tree = MerkleTree()
        root = tree.build(leaves)
        proof = tree.get_proof(0)
        
        # Valid proof
        assert MerkleTree.verify_proof(leaves[0], proof, root)
        
        # Invalid proof
        wrong_leaf = hashlib.sha256(b'wrong').digest()
        assert not MerkleTree.verify_proof(wrong_leaf, proof, root)
```

#### tests/e2e/test_end_to_end.py
```python
"""
PROTO-1105: End-to-end integration test.

Tests complete FL training with Byzantine clients.
"""

import pytest
import yaml
from src.orchestration.fl_coordinator import FLCoordinator

class TestEndToEnd:
    
    def test_baseline_training(self):
        """Test FL training with no attacks."""
        config = {
            'system': {'num_clients': 10, 'num_galaxies': 2},
            'training': {'num_rounds': 5, 'local_epochs': 1},
        }
        coordinator = FLCoordinator(config)
        results = coordinator.run_training(5)
        
        # Should achieve >90% accuracy
        assert results['final_accuracy'] > 0.90
        
    def test_byzantine_defense(self):
        """Test defense against label flipping attack."""
        config = {
            'system': {'num_clients': 30, 'num_galaxies': 3,
                      'byzantine_threshold': 0.3},
            'training': {'num_rounds': 10, 'local_epochs': 1},
        }
        coordinator = FLCoordinator(config)
        results = coordinator.run_training(10)
        
        # Should maintain accuracy despite 30% Byzantine
        assert results['final_accuracy'] > 0.90
        
        # Should detect >80% of Byzantine clients
        assert results['detection_rate'] > 0.80
        
    def test_quarantine_system(self):
        """Test that Byzantine clients are quarantined."""
        # Run training with Byzantine clients
        # Verify quarantine list is populated
        # Verify quarantined clients excluded from aggregation
        pass
```

---

## File Dependencies Map

### Critical Path Dependencies

```
Merkle Tree Module:
merkle_tree.py (PROTO-101,102,103)
    → commitment.py (PROTO-104)
    → galaxy_merkle.py (PROTO-105)
    → global_merkle.py (PROTO-106)

Client Module:
local_trainer.py (PROTO-201)
commitment_generator.py (PROTO-202) [depends on commitment.py]
proof_verifier.py (PROTO-203) [depends on merkle_tree.py]
    → client.py (integrates all)

Galaxy Module:
merkle_constructor.py (PROTO-301) [depends on galaxy_merkle.py]
statistical_analyzer.py (PROTO-302)
robust_aggregator.py (PROTO-303)
reputation_manager.py (PROTO-304)
    → galaxy_aggregator.py (integrates all)
    → defense_coordinator.py (PROTO-505)

Global Module:
global_merkle_constructor.py (PROTO-401) [depends on global_merkle.py]
final_aggregator.py (PROTO-403)
model_manager.py (PROTO-404)
    → global_aggregator.py (integrates all)

Orchestration:
round_manager.py (PROTO-1102)
model_sync.py (PROTO-1104)
    → fl_coordinator.py (PROTO-1101) [integrates all components]
```

---

## Module Size Estimates

```
Lines of Code Estimates:

src/crypto/merkle_tree.py           ~200 lines
src/crypto/commitment.py             ~100 lines
src/crypto/galaxy_merkle.py          ~150 lines
src/crypto/global_merkle.py          ~150 lines

src/client/local_trainer.py         ~150 lines
src/client/commitment_generator.py   ~80 lines
src/client/proof_verifier.py         ~100 lines
src/client/client.py                 ~200 lines

src/galaxy/statistical_analyzer.py   ~300 lines
src/galaxy/robust_aggregator.py      ~250 lines
src/galaxy/reputation_manager.py     ~150 lines
src/galaxy/galaxy_aggregator.py      ~300 lines

src/global_agg/global_aggregator.py  ~250 lines

src/defense/defense_coordinator.py   ~200 lines

src/orchestration/fl_coordinator.py  ~400 lines
src/orchestration/round_manager.py   ~250 lines

src/utils/gradient_utils.py          ~200 lines
src/utils/statistical_utils.py       ~150 lines

src/simulation/honest_client_simulator.py    ~200 lines
src/simulation/byzantine_simulator.py        ~250 lines
src/simulation/metrics_collector.py          ~150 lines

Total: ~4,500 lines of production code (approx)
Tests: ~2,000 lines (approx)
Scripts: ~500 lines (approx)

Grand Total: ~7,000 lines
```

---

## Build and Development Instructions

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/protogalaxy.git
cd protogalaxy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### Running the MVP

```bash
# Run full simulation
python scripts/run_simulation.py

# Run with specific attack
python scripts/run_simulation.py --attack label_flip

# Run experiments
python scripts/run_experiment.py

# Analyze results
python scripts/analyze_results.py --experiment baseline

# Visualize metrics
python scripts/visualize_metrics.py --output outputs/visualizations/
```

This file structure provides a clean, modular organization that maps directly to the MVP Jira tasks while remaining scalable for future enhancements.
