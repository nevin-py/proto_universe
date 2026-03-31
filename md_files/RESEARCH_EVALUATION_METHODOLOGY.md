# ProtoGalaxy Research Evaluation & Benchmarking Methodology

**Purpose:** Comprehensive evaluation framework for academic research and benchmarking  
**Based on:** Architecture Section 6 (Experimental Methodology)  
**Target:** Research publication quality results with statistical rigor

---

## Table of Contents

1. [Overview](#1-overview)
2. [Baseline Performance Evaluation](#2-baseline-performance-evaluation)
3. [Attack Scenario Evaluation](#3-attack-scenario-evaluation)
4. [Comparative Benchmarking](#4-comparative-benchmarking)
5. [Scalability Testing](#5-scalability-testing)
6. [Ablation Studies](#6-ablation-studies)
7. [Test Scripts Implementation](#7-test-scripts-implementation)
8. [Statistical Methodology](#8-statistical-methodology)
9. [Data Collection & Analysis](#9-data-collection--analysis)
10. [Publication-Ready Visualization](#10-publication-ready-visualization)

---

## 1. Overview

### 1.1 Research Questions

**RQ1: Effectiveness**
- Does ProtoGalaxy maintain model accuracy under Byzantine attacks?
- What is the detection rate for different attack types?

**RQ2: Efficiency**
- What is the computational and communication overhead vs. baseline FL?
- Does the hierarchical structure provide scalability benefits?

**RQ3: Accountability**
- Can ProtoGalaxy accurately attribute malicious behavior to specific clients?
- What is the false positive rate for honest clients?

**RQ4: Comparative Performance**
- How does ProtoGalaxy compare to state-of-the-art defenses (Krum, FLTrust, etc.)?
- What are the trade-offs between different defense layers?

### 1.2 Evaluation Dimensions

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION MATRIX                         │
├─────────────────┬───────────────────────────────────────────┤
│ DIMENSION       │ METRICS                                    │
├─────────────────┼───────────────────────────────────────────┤
│ Effectiveness   │ • Model Accuracy                           │
│                 │ • Attack Success Rate (ASR)                │
│                 │ • Byzantine Detection Rate (TPR)           │
│                 │ • False Positive Rate (FPR)                │
├─────────────────┼───────────────────────────────────────────┤
│ Efficiency      │ • Communication Overhead (bytes)           │
│                 │ • Computation Time (wall-clock)            │
│                 │ • Storage Requirements (MB)                │
│                 │ • Proof Verification Time (ms)             │
├─────────────────┼───────────────────────────────────────────┤
│ Accountability  │ • Attribution Accuracy (%)                 │
│                 │ • Evidence Quality Score                   │
│                 │ • Forensic Analysis Time (s)               │
├─────────────────┼───────────────────────────────────────────┤
│ Scalability     │ • Clients: 100, 500, 1000, 5000            │
│                 │ • Galaxies: 5, 10, 20, 50                  │
│                 │ • Throughput (rounds/hour)                 │
└─────────────────┴───────────────────────────────────────────┘
```

---

## 2. Baseline Performance Evaluation

### 2.1 Normal Operation Benchmarks

**Test 1.1: IID Data Distribution (Baseline)**

**Objective:** Establish baseline performance with ideal conditions

```yaml
Configuration:
  dataset: MNIST
  num_clients: 100
  num_galaxies: 10
  clients_per_galaxy: 10
  num_rounds: 100
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.01
  partition: IID
  byzantine_fraction: 0.0
  aggregation_method: trimmed_mean
  trim_ratio: 0.1
```

**Expected Results:**
- Target accuracy: >98% (MNIST standard)
- Convergence: <50 rounds to 95% accuracy
- Communication: ~2MB per client per round

**Test Script:** `scripts/eval_baseline_iid.py`

---

**Test 1.2: Non-IID Label Skew**

**Objective:** Evaluate with realistic heterogeneous data

```yaml
Configuration:
  dataset: MNIST
  num_clients: 100
  num_galaxies: 10
  partition: noniid
  classes_per_client: 2
  byzantine_fraction: 0.0
  # Other params same as 1.1
```

**Expected Results:**
- Target accuracy: >96% (slight degradation acceptable)
- Convergence: 60-80 rounds
- Communication: Same as baseline

**Test Script:** `scripts/eval_baseline_noniid_label.py`

---

**Test 1.3: Non-IID Quantity Skew (Dirichlet)**

**Objective:** Evaluate with unbalanced data distribution

```yaml
Configuration:
  dataset: MNIST
  num_clients: 100
  partition: dirichlet
  dirichlet_alpha: 0.5  # Lower = more heterogeneous
  byzantine_fraction: 0.0
```

**Expected Results:**
- Target accuracy: >95%
- Convergence: 70-100 rounds
- Per-client variance: Measure via Gini coefficient

**Test Script:** `scripts/eval_baseline_noniid_quantity.py`

---

**Test 1.4: CIFAR-10 Benchmark**

**Objective:** Evaluate on complex dataset

```yaml
Configuration:
  dataset: CIFAR10
  model: ResNet18
  num_clients: 500
  num_galaxies: 20
  num_rounds: 200
  local_epochs: 5
  learning_rate: 0.01
  partition: IID
```

**Expected Results:**
- Target accuracy: >85%
- Convergence: 120-150 rounds
- Training time: ~2-3 hours (GPU cluster)

**Test Script:** `scripts/eval_baseline_cifar10.py`

---

### 2.2 Overhead Measurement

**Test 2.1: Communication Overhead Analysis**

**Objective:** Quantify Merkle + ZK proof overhead

**Methodology:**
1. Run baseline FL (no defense) - measure bytes
2. Run ProtoGalaxy Full - measure bytes
3. Breakdown:
   - Gradient transmission: baseline
   - Merkle proofs: O(log n) per client
   - ZK proofs: constant size (~1KB)
   - Metadata: timestamps, nonces

**Metrics to Collect:**
```python
{
    'baseline_bytes': total_gradient_bytes,
    'merkle_proof_bytes': sum(proof_sizes),
    'zk_proof_bytes': sum(zk_proof_sizes),
    'metadata_bytes': sum(metadata_sizes),
    'total_overhead_percent': (protogalaxy - baseline) / baseline * 100
}
```

**Test Script:** `scripts/eval_overhead_communication.py`

---

**Test 2.2: Computational Time Breakdown**

**Objective:** Profile time spent in each phase

**Methodology:**
```python
timings = {
    'phase1_commitment': {
        'client_hash': [],
        'galaxy_merkle_build': [],
        'global_merkle_build': [],
        'zk_proof_generation': []
    },
    'phase2_revelation': {
        'merkle_verification': [],
        'zk_proof_verification': []
    },
    'phase3_defense': {
        'statistical_analysis': [],
        'robust_aggregation': [],
        'reputation_update': []
    },
    'phase4_aggregation': {
        'galaxy_verification': [],
        'layer5_defense': [],
        'global_aggregation': [],
        'model_update': [],
        'zk_folding': []
    }
}
```

**Test Script:** `scripts/eval_overhead_computation.py`

---

**Test 2.3: Storage Requirements**

**Objective:** Measure storage footprint

**Components to Measure:**
- Merkle trees (per round): O(n) hashes
- Gradient history (forensics): O(n × d × T)
- Reputation scores: O(n) floats
- Evidence database: O(k × log n) for k quarantined clients
- ZK proofs: O(n) constant-size proofs

**Test Script:** `scripts/eval_overhead_storage.py`

---

## 3. Attack Scenario Evaluation

### 3.1 Gradient Poisoning Attacks

**Test 3.1: Label Flipping Attack**

**Objective:** Evaluate defense against random label flipping

**Attack Configuration:**
```yaml
attack_type: label_flip
byzantine_fractions: [0.1, 0.2, 0.3, 0.4]
flip_probability: 1.0  # Flip all labels
target_labels:
  - source: ALL
    target: RANDOM
```

**Experimental Setup:**
```python
configs = [
    {'byzantine': 0.0, 'attack': None},  # Control
    {'byzantine': 0.1, 'attack': 'label_flip'},
    {'byzantine': 0.2, 'attack': 'label_flip'},
    {'byzantine': 0.3, 'attack': 'label_flip'},
    {'byzantine': 0.4, 'attack': 'label_flip'},
]

for config in configs:
    run_experiment(
        dataset='MNIST',
        num_clients=100,
        num_rounds=100,
        **config
    )
```

**Metrics to Collect:**
- Final test accuracy
- Accuracy per round (convergence curve)
- Byzantine detection rate (TPR)
- False positive rate (FPR)
- Time to detection (first round Byzantine client flagged)

**Expected Results:**
| Byzantine % | Baseline Acc | ProtoGalaxy Acc | Degradation |
|-------------|--------------|-----------------|-------------|
| 0%          | 98.5%        | 98.3%           | 0.2%        |
| 10%         | 92.0%        | 97.8%           | 0.7%        |
| 20%         | 75.0%        | 96.5%           | 2.0%        |
| 30%         | 45.0%        | 94.0%           | 4.5%        |
| 40%         | 20.0%        | 88.0%           | 10.5%       |

**Test Script:** `scripts/eval_attack_label_flip.py`

---

**Test 3.2: Targeted Label Flipping**

**Objective:** Evaluate against targeted misclassification

**Attack Configuration:**
```yaml
attack_type: targeted_label_flip
byzantine_fraction: 0.2
target_pairs:
  - [3, 7]  # Flip 3 → 7
  - [7, 3]  # Flip 7 → 3
```

**Additional Metrics:**
- Attack Success Rate (ASR) for target pairs
- Per-class accuracy degradation
- Detection latency (rounds to detect)

**Test Script:** `scripts/eval_attack_targeted_flip.py`

---

**Test 3.3: Backdoor Injection Attack**

**Objective:** Evaluate against backdoor/trojan attacks

**Attack Configuration:**
```yaml
attack_type: backdoor
byzantine_fraction: 0.1
backdoor_config:
  trigger_pattern: "bottom_right_square"  # 3×3 white pixels
  trigger_size: [3, 3]
  trigger_position: [25, 25]
  target_class: 7
  poisoning_rate: 0.1  # 10% of Byzantine client data
```

**Metrics:**
- Main task accuracy (clean test set)
- Attack Success Rate (ASR) on triggered samples
- Backdoor persistence (ASR over rounds)
- Detection: Can defense identify backdoor clients?

**Test Procedure:**
1. Train with backdoor clients for 100 rounds
2. Every 10 rounds: evaluate ASR on triggered test set
3. Track which clients are flagged/quarantined
4. Measure backdoor "durability" (rounds before ASR < 10%)

**Expected Results:**
- ASR starts at 60-80% initially
- ProtoGalaxy reduces ASR to <10% within 20-30 rounds
- Main task accuracy maintained at >95%

**Test Script:** `scripts/eval_attack_backdoor.py`

---

**Test 3.4: Model Poisoning Attack**

**Objective:** Evaluate against gradient-based model poisoning

**Attack Configuration:**
```yaml
attack_type: model_poisoning
byzantine_fraction: 0.2
poisoning_strategy: "maximize_loss"
scaling_factor: 10.0  # Amplify malicious gradients
```

**Poisoning Strategies to Test:**
1. **Max Loss:** ∇w_malicious = -η × ∇w_honest
2. **Random Noise:** ∇w_malicious = Gaussian(0, σ²)
3. **Sign Flip:** ∇w_malicious = -∇w_honest
4. **Little Is Enough:** Carefully crafted to evade detection

**Test Script:** `scripts/eval_attack_model_poison.py`

---

### 3.2 Adaptive Attacks

**Test 3.5: Adaptive Byzantine Attack**

**Objective:** Evaluate against intelligent adversary

**Attack Configuration:**
```yaml
attack_type: adaptive_byzantine
byzantine_fraction: 0.3
adaptation_strategy:
  - Observe statistical thresholds for 10 rounds
  - Craft gradients just below detection threshold
  - Intermittent attacks (50% of rounds)
  - Gradient mimicry (close to honest centroid)
```

**Implementation:**
```python
class AdaptiveAttacker:
    def __init__(self):
        self.observed_thresholds = []
        self.attack_probability = 0.5
        
    def generate_malicious_gradient(self, honest_gradients):
        # Estimate honest centroid
        centroid = np.mean(honest_gradients, axis=0)
        std = np.std(honest_gradients, axis=0)
        
        # Generate gradient within 2σ of centroid (evade statistical detection)
        malicious = centroid + np.random.randn(*centroid.shape) * 2 * std
        
        # Add subtle bias towards attack goal
        malicious -= 0.1 * centroid  # Small perturbation
        
        return malicious
```

**Test Script:** `scripts/eval_attack_adaptive.py`

---

**Test 3.6: Sybil Attack Simulation**

**Objective:** Test resistance to identity-based attacks

**Attack Configuration:**
```yaml
attack_type: sybil
num_sybil_identities: 30
colluding_groups:
  - group_1: [client_10, client_20, ..., client_100]
  - group_2: [client_101, client_110, ..., client_130]
attack_coordination: "synchronized"  # All Sybils attack together
```

**Test Script:** `scripts/eval_attack_sybil.py`

---

### 3.3 Aggregator-Level Attacks

**Test 3.7: Malicious Aggregator Simulation**

**Objective:** Verify Merkle verification catches aggregator tampering

**Attack Scenarios:**
1. **Gradient Modification:** Aggregator changes 20% of client gradients
2. **Selective Exclusion:** Aggregator drops gradients from honest clients
3. **False Attribution:** Aggregator claims honest client sent malicious gradient

**Verification:**
- Merkle proof verification should detect 100% of tampering
- Clients should reject modified aggregations
- Cryptographic evidence enables dispute resolution

**Test Script:** `scripts/eval_attack_aggregator.py`

---

## 4. Comparative Benchmarking

### 4.1 Baseline Comparisons

**Test 4.1: Vanilla FL (No Defense)**

```yaml
method: vanilla_fedavg
aggregation: simple_average
defense_layers: []
```

**Test 4.2: Krum**

```yaml
method: krum
aggregation: multi_krum
krum_f: computed_from_byzantine_fraction
krum_m: 1  # Single-Krum
```

**Test 4.3: Multi-Krum**

```yaml
method: multi_krum
krum_m: num_clients - byzantine_count
```

**Test 4.4: Trimmed Mean**

```yaml
method: trimmed_mean
trim_ratio: 0.1
```

**Test 4.5: Coordinate-Wise Median**

```yaml
method: median
aggregation: coordinate_wise_median
```

**Test 4.6: FLTrust**

```yaml
method: fltrust
server_dataset_size: 1000  # Clean root dataset
trust_scoring: cosine_similarity
```

**Test 4.7: ProtoGalaxy (Lite)**

```yaml
method: protogalaxy_lite
layers: [Layer1_Merkle, Layer3_Krum]
```

**Test 4.8: ProtoGalaxy (Full)**

```yaml
method: protogalaxy_full
layers: [Layer1_Merkle, Layer2_Statistical, Layer3_Robust, Layer4_Reputation, Layer5_Galaxy]
```

---

### 4.2 Comparative Evaluation Matrix

**Experimental Design:**

For each method × attack combination, run 5 independent trials:

```python
methods = ['vanilla', 'krum', 'multi_krum', 'trimmed_mean', 
           'median', 'fltrust', 'protogalaxy_lite', 'protogalaxy_full']

attacks = ['label_flip_10', 'label_flip_20', 'label_flip_30',
           'backdoor', 'model_poison', 'adaptive']

for method in methods:
    for attack in attacks:
        for trial in range(5):
            results = run_experiment(
                method=method,
                attack=attack,
                trial_id=trial,
                seed=42 + trial
            )
            save_results(results, f'{method}_{attack}_trial{trial}.json')
```

**Test Script:** `scripts/eval_comparative_benchmark.py`

---

## 5. Scalability Testing

### 5.1 Client Scaling

**Test 5.1: Varying Number of Clients**

```python
client_counts = [50, 100, 200, 500, 1000, 2000, 5000]

for n_clients in client_counts:
    num_galaxies = int(np.sqrt(n_clients))  # Heuristic: √n galaxies
    
    run_experiment(
        num_clients=n_clients,
        num_galaxies=num_galaxies,
        num_rounds=50,  # Shorter for large-scale
        measure_metrics=['throughput', 'latency', 'memory']
    )
```

**Metrics:**
- Throughput: Rounds per hour
- Latency: Seconds per round
- Memory: Peak RAM usage
- Scalability coefficient: Empirical growth rate O(n^α)

**Test Script:** `scripts/eval_scalability_clients.py`

---

### 5.2 Galaxy Scaling

**Test 5.2: Optimal Galaxy Partitioning**

**Objective:** Find optimal clients-per-galaxy ratio

```python
configs = [
    {'clients': 1000, 'galaxies': 5},    # 200 clients/galaxy
    {'clients': 1000, 'galaxies': 10},   # 100 clients/galaxy
    {'clients': 1000, 'galaxies': 20},   # 50 clients/galaxy
    {'clients': 1000, 'galaxies': 50},   # 20 clients/galaxy
    {'clients': 1000, 'galaxies': 100},  # 10 clients/galaxy
]
```

**Hypothesis:** There exists an optimal ratio balancing:
- Defense effectiveness (smaller galaxies = better isolation)
- Efficiency (larger galaxies = fewer aggregation levels)

**Test Script:** `scripts/eval_scalability_galaxies.py`

---

### 5.3 Round Scaling

**Test 5.3: Long-Term Training Performance**

```yaml
Configuration:
  num_rounds: 500
  checkpoints: [50, 100, 200, 300, 400, 500]
  byzantine_fraction: 0.2
  attack_type: "intermittent_label_flip"  # Attacks in 50% of rounds
```

**Metrics:**
- Convergence stability over time
- Reputation score evolution
- Cumulative detection rate
- Model accuracy variance

**Test Script:** `scripts/eval_scalability_rounds.py`

---

## 6. Ablation Studies

### 6.1 Defense Layer Ablation

**Test 6.1: Individual Layer Contribution**

**Objective:** Quantify contribution of each defense layer

```python
layer_configs = [
    {'layers': [],                          'name': 'Baseline'},
    {'layers': ['Layer1'],                  'name': 'Merkle Only'},
    {'layers': ['Layer2'],                  'name': 'Statistical Only'},
    {'layers': ['Layer3'],                  'name': 'Krum Only'},
    {'layers': ['Layer4'],                  'name': 'Reputation Only'},
    {'layers': ['Layer1', 'Layer3'],        'name': 'Merkle + Krum'},
    {'layers': ['Layer2', 'Layer3'],        'name': 'Statistical + Krum'},
    {'layers': ['Layer1', 'Layer2', 'Layer3'], 'name': 'Layers 1-3'},
    {'layers': ['Layer1', 'Layer2', 'Layer3', 'Layer4'], 'name': 'Layers 1-4'},
    {'layers': ['All'],                     'name': 'Full Stack'},
]
```

**Test Script:** `scripts/eval_ablation_layers.py`

---

### 6.2 Hyperparameter Sensitivity

**Test 6.2: Statistical Threshold Sensitivity**

```python
threshold_configs = [
    {'norm_threshold': 2.0, 'cosine_threshold': 0.3},
    {'norm_threshold': 2.5, 'cosine_threshold': 0.4},
    {'norm_threshold': 3.0, 'cosine_threshold': 0.5},  # Default
    {'norm_threshold': 3.5, 'cosine_threshold': 0.6},
    {'norm_threshold': 4.0, 'cosine_threshold': 0.7},
]
```

**Expected Trade-off:**
- Lower thresholds → Higher TPR, Higher FPR
- Higher thresholds → Lower FPR, Lower TPR

**Test Script:** `scripts/eval_ablation_thresholds.py`

---

**Test 6.3: Trimmed Mean Ratio Sensitivity**

```python
trim_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
```

**Test Script:** `scripts/eval_ablation_trim_ratio.py`

---

**Test 6.4: Reputation Learning Rate**

```python
reputation_configs = [
    {'lambda': 0.05, 'quarantine_threshold': 0.2},
    {'lambda': 0.10, 'quarantine_threshold': 0.2},  # Default
    {'lambda': 0.15, 'quarantine_threshold': 0.2},
    {'lambda': 0.20, 'quarantine_threshold': 0.2},
]
```

**Test Script:** `scripts/eval_ablation_reputation.py`

---

### 6.3 ZK Proof Ablation

**Test 6.5: Merkle Only vs. Merkle + ZK**

**Configurations:**
```python
configs = [
    {'verification': 'merkle_only', 'zk_enabled': False},
    {'verification': 'merkle_zk', 'zk_enabled': True},
]
```

**Metrics:**
- Detection accuracy (do ZK proofs catch additional attacks?)
- Overhead (computational, communication)
- Proof generation/verification time

**Test Script:** `scripts/eval_ablation_zkp.py`

---

## 7. Test Scripts Implementation

### 7.1 Core Test Framework

**File: `scripts/eval_framework.py`**

```python
"""
Core evaluation framework for ProtoGalaxy research benchmarking.
Provides base classes and utilities for all evaluation scripts.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.pipeline import ProtoGalaxyPipeline
from src.models.mnist import create_mnist_model
from src.client.trainer import Trainer
from run_pipeline import load_mnist, iid_partition, noniid_partition, evaluate_model


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    # Dataset
    dataset: str = 'mnist'
    model_type: str = 'linear'
    partition: str = 'iid'
    
    # FL Setup
    num_clients: int = 100
    num_galaxies: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Attack Configuration
    byzantine_fraction: float = 0.0
    attack_type: Optional[str] = None
    attack_params: Dict = None
    
    # Defense Configuration
    defense_layers: List[str] = None
    aggregation_method: str = 'trimmed_mean'
    trim_ratio: float = 0.1
    
    # Experiment Metadata
    experiment_name: str = 'baseline'
    trial_id: int = 0
    random_seed: int = 42
    
    def __post_init__(self):
        if self.defense_layers is None:
            self.defense_layers = ['all']
        if self.attack_params is None:
            self.attack_params = {}


@dataclass
class ExperimentResults:
    """Results from a single experiment run."""
    
    config: ExperimentConfig
    
    # Performance Metrics
    final_accuracy: float
    accuracies_per_round: List[float]
    convergence_round: int  # Round reaching 95% of final accuracy
    
    # Attack Metrics (if applicable)
    byzantine_detection_rate: float  # TPR
    false_positive_rate: float  # FPR
    attack_success_rate: float  # For backdoor attacks
    
    # Efficiency Metrics
    round_times: List[float]
    total_training_time: float
    communication_bytes: int
    
    # Overhead Breakdown
    merkle_proof_time: List[float]
    zk_proof_time: List[float]
    defense_time: List[float]
    
    # Attribution
    flagged_clients: List[List[int]]  # Per round
    quarantined_clients: List[int]
    
    # Metadata
    timestamp: str
    experiment_id: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['config'] = asdict(self.config)
        return result
    
    def save(self, filepath: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        config = ExperimentConfig(**data['config'])
        data['config'] = config
        return cls(**data)


class ProtoGalaxyExperiment:
    """Base class for running ProtoGalaxy experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = None
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
    
    def setup_data(self):
        """Load and partition dataset."""
        print(f"📦 Loading {self.config.dataset.upper()} dataset...")
        
        if self.config.dataset == 'mnist':
            train_dataset, test_dataset = load_mnist()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        
        # Partition data
        if self.config.partition == 'iid':
            client_data = iid_partition(train_dataset, self.config.num_clients)
        elif self.config.partition == 'noniid':
            client_data = noniid_partition(train_dataset, self.config.num_clients)
        else:
            raise ValueError(f"Unknown partition: {self.config.partition}")
        
        return train_dataset, test_dataset, client_data
    
    def inject_attack(self, client_id: int, trainer: Trainer, gradients: List[torch.Tensor]):
        """Inject attack into client gradients if client is Byzantine."""
        num_byzantine = int(self.config.num_clients * self.config.byzantine_fraction)
        
        if client_id >= num_byzantine:
            return gradients  # Honest client
        
        # Apply attack based on type
        if self.config.attack_type == 'label_flip':
            return self._attack_label_flip(gradients)
        elif self.config.attack_type == 'backdoor':
            return self._attack_backdoor(trainer, gradients)
        elif self.config.attack_type == 'model_poison':
            return self._attack_model_poison(gradients)
        else:
            return gradients
    
    def _attack_label_flip(self, gradients):
        """Label flipping attack: negate gradients."""
        return [-g for g in gradients]
    
    def _attack_backdoor(self, trainer, gradients):
        """Backdoor attack: subtle perturbation."""
        # Implementation depends on specific backdoor strategy
        return gradients
    
    def _attack_model_poison(self, gradients):
        """Model poisoning: amplified malicious gradients."""
        scaling = self.config.attack_params.get('scaling_factor', 10.0)
        return [-g * scaling for g in gradients]
    
    def run(self) -> ExperimentResults:
        """Execute the experiment."""
        print(f"\n{'='*70}")
        print(f"  Experiment: {self.config.experiment_name}")
        print(f"  Trial: {self.config.trial_id}")
        print(f"{'='*70}")
        
        # Setup
        train_dataset, test_dataset, client_data = self.setup_data()
        global_model = create_mnist_model(self.config.model_type)
        
        # Create pipeline
        defense_config = {
            'layer3_method': self.config.aggregation_method,
            'layer3_trim_ratio': self.config.trim_ratio,
        }
        
        pipeline = ProtoGalaxyPipeline(
            global_model=global_model,
            num_clients=self.config.num_clients,
            num_galaxies=self.config.num_galaxies,
            defense_config=defense_config,
            logger=None
        )
        
        # Tracking
        accuracies = []
        round_times = []
        flagged_per_round = []
        
        # Training loop
        start_time = time.time()
        
        for round_num in range(self.config.num_rounds):
            round_start = time.time()
            
            # Local training
            client_trainers = {}
            for cid in range(self.config.num_clients):
                client_model = copy.deepcopy(global_model)
                trainer = Trainer(model=client_model, learning_rate=self.config.learning_rate)
                loader = DataLoader(client_data[cid], batch_size=self.config.batch_size, shuffle=True)
                trainer.train(loader, num_epochs=self.config.local_epochs)
                client_trainers[cid] = trainer
            
            # Execute round (with potential attacks)
            round_stats = pipeline.execute_round(client_trainers, round_num)
            
            # Evaluate
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
            accuracy = evaluate_model(global_model, test_loader)
            accuracies.append(accuracy)
            
            round_time = time.time() - round_start
            round_times.append(round_time)
            
            flagged_per_round.append(round_stats.get('flagged_galaxies', []))
            
            if round_num % 10 == 0:
                print(f"  Round {round_num}: Acc={accuracy:.2%}, Time={round_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Compile results
        self.results = ExperimentResults(
            config=self.config,
            final_accuracy=accuracies[-1],
            accuracies_per_round=accuracies,
            convergence_round=self._find_convergence(accuracies),
            byzantine_detection_rate=0.0,  # Compute if applicable
            false_positive_rate=0.0,
            attack_success_rate=0.0,
            round_times=round_times,
            total_training_time=total_time,
            communication_bytes=0,  # Compute from pipeline stats
            merkle_proof_time=[],
            zk_proof_time=[],
            defense_time=[],
            flagged_clients=flagged_per_round,
            quarantined_clients=[],
            timestamp=datetime.now().isoformat(),
            experiment_id=f"{self.config.experiment_name}_trial{self.config.trial_id}"
        )
        
        return self.results
    
    def _find_convergence(self, accuracies: List[float]) -> int:
        """Find round where model reaches 95% of final accuracy."""
        target = 0.95 * accuracies[-1]
        for i, acc in enumerate(accuracies):
            if acc >= target:
                return i
        return len(accuracies) - 1


def run_experiment_suite(
    experiment_name: str,
    configs: List[ExperimentConfig],
    output_dir: str = './results'
):
    """Run a suite of experiments with multiple configurations."""
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'#'*70}")
        print(f"  Running experiment {i+1}/{len(configs)}")
        print(f"{'#'*70}")
        
        experiment = ProtoGalaxyExperiment(config)
        results = experiment.run()
        
        # Save results
        output_file = os.path.join(
            output_dir,
            f"{experiment_name}_{config.experiment_name}_trial{config.trial_id}.json"
        )
        results.save(output_file)
        all_results.append(results)
        
        print(f"  ✓ Results saved to {output_file}")
    
    return all_results


if __name__ == '__main__':
    # Example usage
    config = ExperimentConfig(
        experiment_name='test',
        num_clients=10,
        num_rounds=5,
        trial_id=0
    )
    
    exp = ProtoGalaxyExperiment(config)
    results = exp.run()
    results.save('./results/test.json')
    print(f"\n✓ Test completed: Final accuracy = {results.final_accuracy:.2%}")
```

---

### 7.2 Attack Evaluation Scripts

**File: `scripts/eval_attack_label_flip.py`**

```python
"""
Evaluation: Label Flipping Attack
Tests Byzantine resilience against random label flipping at various intensities.
"""

from eval_framework import (
    ExperimentConfig, 
    run_experiment_suite
)

def main():
    # Test label flipping at different Byzantine fractions
    byzantine_fractions = [0.0, 0.1, 0.2, 0.3, 0.4]
    num_trials = 5
    
    configs = []
    
    for byzantine_frac in byzantine_fractions:
        for trial in range(num_trials):
            config = ExperimentConfig(
                experiment_name=f'label_flip_{int(byzantine_frac*100)}pct',
                dataset='mnist',
                num_clients=100,
                num_galaxies=10,
                num_rounds=100,
                byzantine_fraction=byzantine_frac,
                attack_type='label_flip' if byzantine_frac > 0 else None,
                trial_id=trial,
                random_seed=42 + trial
            )
            configs.append(config)
    
    # Run experiments
    results = run_experiment_suite(
        experiment_name='label_flip_attack',
        configs=configs,
        output_dir='./results/attacks/label_flip'
    )
    
    print(f"\n{'='*70}")
    print(f"  Label Flip Attack Evaluation Complete")
    print(f"  Total experiments: {len(results)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
```

---

### 7.3 Comparative Benchmark Script

**File: `scripts/eval_comparative_benchmark.py`**

```python
"""
Comparative Benchmark: ProtoGalaxy vs. State-of-the-Art
Compares against Krum, Trimmed Mean, Median, FLTrust, etc.
"""

from eval_framework import ExperimentConfig, run_experiment_suite

def main():
    # Test configuration
    num_trials = 5
    byzantine_fraction = 0.3
    
    # Defense methods to compare
    methods = [
        {'name': 'vanilla', 'layers': [], 'aggregation': 'average'},
        {'name': 'krum', 'layers': ['Layer3'], 'aggregation': 'multi_krum'},
        {'name': 'trimmed_mean', 'layers': ['Layer3'], 'aggregation': 'trimmed_mean'},
        {'name': 'median', 'layers': ['Layer3'], 'aggregation': 'coordinate_wise_median'},
        {'name': 'protogalaxy_lite', 'layers': ['Layer1', 'Layer3'], 'aggregation': 'trimmed_mean'},
        {'name': 'protogalaxy_full', 'layers': ['all'], 'aggregation': 'trimmed_mean'},
    ]
    
    # Attack scenarios
    attacks = [
        {'name': 'label_flip', 'type': 'label_flip'},
        {'name': 'model_poison', 'type': 'model_poison'},
    ]
    
    configs = []
    
    for method in methods:
        for attack in attacks:
            for trial in range(num_trials):
                config = ExperimentConfig(
                    experiment_name=f"{method['name']}_{attack['name']}",
                    num_clients=100,
                    num_galaxies=10,
                    num_rounds=100,
                    byzantine_fraction=byzantine_fraction,
                    attack_type=attack['type'],
                    defense_layers=method['layers'],
                    aggregation_method=method['aggregation'],
                    trial_id=trial,
                    random_seed=42 + trial
                )
                configs.append(config)
    
    # Run experiments
    results = run_experiment_suite(
        experiment_name='comparative_benchmark',
        configs=configs,
        output_dir='./results/comparative'
    )
    
    print(f"\n✓ Comparative benchmark complete: {len(results)} experiments")


if __name__ == '__main__':
    main()
```

---

## 8. Statistical Methodology

### 8.1 Experimental Design Principles

**1. Replication**
- Run each experiment 5 times with different random seeds
- Report mean ± standard deviation
- Use seeds: 42, 43, 44, 45, 46

**2. Randomization**
- Randomize client data partition for each trial
- Randomize Byzantine client selection (if applicable)
- Randomize attack timing (for intermittent attacks)

**3. Blocking**
- Group experiments by attack type (blocking factor)
- Ensures fair comparison across methods

**4. Factorial Design**
- Vary multiple factors: Byzantine %, defense method, dataset
- Enable interaction effects analysis

---

### 8.2 Statistical Significance Testing

**T-Test for Pairwise Comparison**

```python
from scipy import stats

def compare_methods(results_A, results_B, metric='final_accuracy'):
    """
    Compare two methods using paired t-test.
    
    H0: μ_A = μ_B (no difference)
    H1: μ_A ≠ μ_B (significant difference)
    """
    values_A = [r.__dict__[metric] for r in results_A]
    values_B = [r.__dict__[metric] for r in results_B]
    
    t_stat, p_value = stats.ttest_ind(values_A, values_B)
    
    return {
        'mean_A': np.mean(values_A),
        'mean_B': np.mean(values_B),
        'std_A': np.std(values_A),
        'std_B': np.std(values_B),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': (np.mean(values_A) - np.mean(values_B)) / np.sqrt((np.std(values_A)**2 + np.std(values_B)**2) / 2)
    }
```

**ANOVA for Multi-Method Comparison**

```python
from scipy.stats import f_oneway

def compare_multiple_methods(results_dict, metric='final_accuracy'):
    """
    Compare multiple methods using one-way ANOVA.
    
    H0: All methods have equal means
    H1: At least one method differs
    """
    groups = []
    for method_name, results in results_dict.items():
        values = [r.__dict__[metric] for r in results]
        groups.append(values)
    
    f_stat, p_value = f_oneway(*groups)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

---

### 8.3 Confidence Intervals

```python
def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for mean."""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return {
        'mean': mean,
        'lower_bound': mean - margin,
        'upper_bound': mean + margin,
        'margin_of_error': margin
    }
```

---

### 8.4 Effect Size Calculation

**Cohen's d:**

```python
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((len(group1)-1)*std1**2 + (len(group2)-1)*std2**2) / (len(group1) + len(group2) - 2))
    
    d = (mean1 - mean2) / pooled_std
    
    # Interpretation
    if abs(d) < 0.2:
        magnitude = "negligible"
    elif abs(d) < 0.5:
        magnitude = "small"
    elif abs(d) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    return {'d': d, 'magnitude': magnitude}
```

---

## 9. Data Collection & Analysis

### 9.1 Results Aggregation Script

**File: `scripts/analyze_results.py`** (Enhanced version)

```python
"""
Results Analysis and Aggregation
Processes experiment results and generates summary statistics.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict
from scipy import stats

def load_experiment_results(results_dir: str, experiment_pattern: str = '*.json'):
    """Load all result JSON files from directory."""
    files = glob.glob(os.path.join(results_dir, experiment_pattern))
    results = []
    
    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results

def aggregate_by_configuration(results: List[Dict]) -> pd.DataFrame:
    """Group results by configuration and compute statistics."""
    
    df = pd.DataFrame(results)
    
    # Group by experiment name (which encodes configuration)
    grouped = df.groupby('config')['final_accuracy']
    
    summary = grouped.agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    # Add confidence intervals
    summary['ci_95_lower'] = summary.apply(
        lambda row: row['mean'] - 1.96 * row['std'] / np.sqrt(row['count']),
        axis=1
    )
    summary['ci_95_upper'] = summary.apply(
        lambda row: row['mean'] + 1.96 * row['std'] / np.sqrt(row['count']),
        axis=1
    )
    
    return summary

def create_comparison_table(results_dict: Dict[str, List[Dict]], metric: str = 'final_accuracy'):
    """Create comparison table for multiple methods."""
    
    rows = []
    
    for method_name, results in results_dict.items():
        values = [r[metric] for r in results]
        
        row = {
            'Method': method_name,
            'Mean': np.mean(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'N': len(values)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Mean', ascending=False)
    
    return df


if __name__ == '__main__':
    # Example: Analyze label flip attack results
    results = load_experiment_results('./results/attacks/label_flip')
    
    summary = aggregate_by_configuration(results)
    print(summary)
    
    # Save summary
    summary.to_csv('./results/label_flip_summary.csv', index=False)
```

---

### 9.2 Metrics Extraction

**Key Metrics to Extract from Results:**

```python
metrics_to_extract = {
    # Effectiveness
    'final_accuracy': 'Final test accuracy',
    'convergence_round': 'Round reaching 95% of final accuracy',
    'accuracy_improvement': 'Final - Initial accuracy',
    
    # Attack Metrics
    'byzantine_detection_rate': 'True Positive Rate (TPR)',
    'false_positive_rate': 'False Positive Rate (FPR)',
    'attack_success_rate': 'Backdoor attack success rate',
    'precision': 'TP / (TP + FP)',
    'recall': 'TP / (TP + FN)',
    'f1_score': '2 * (precision * recall) / (precision + recall)',
    
    # Efficiency
    'avg_round_time': 'Average time per round (seconds)',
    'total_training_time': 'Total training time (seconds)',
    'communication_overhead': 'Extra bytes vs. baseline (%)',
    'throughput': 'Rounds per hour',
    
    # Overhead Breakdown
    'merkle_time_pct': '% time spent on Merkle operations',
    'zk_time_pct': '% time spent on ZK operations',
    'defense_time_pct': '% time spent on defense',
    
    # Scalability
    'time_complexity_alpha': 'Empirical exponent (O(n^α))',
    'memory_usage_mb': 'Peak memory usage (MB)',
}
```

---

## 10. Publication-Ready Visualization

### 10.1 Visualization Scripts

**File: `scripts/visualize_results.py`** (Enhanced)

```python
"""
Publication-Quality Visualization
Generates plots for research papers.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
})


def plot_accuracy_vs_byzantine_fraction(results_df: pd.DataFrame, output_file: str):
    """
    Plot: Final accuracy vs. Byzantine fraction for different methods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = results_df['method'].unique()
    
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        
        # Group by Byzantine fraction
        grouped = method_data.groupby('byzantine_fraction')['final_accuracy']
        means = grouped.mean()
        stds = grouped.std()
        
        ax.plot(means.index * 100, means.values * 100, 
                marker='o', label=method, linewidth=2)
        ax.fill_between(means.index * 100, 
                        (means - stds).values * 100,
                        (means + stds).values * 100,
                        alpha=0.2)
    
    ax.set_xlabel('Byzantine Clients (%)')
    ax.set_ylabel('Final Accuracy (%)')
    ax.set_title('Model Accuracy vs. Byzantine Fraction')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_convergence_curves(results_list: List[Dict], output_file: str):
    """
    Plot: Accuracy over training rounds (convergence curves).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for result in results_list:
        config_name = result['config']['experiment_name']
        accuracies = result['accuracies_per_round']
        rounds = range(len(accuracies))
        
        ax.plot(rounds, [acc * 100 for acc in accuracies], 
                label=config_name, linewidth=1.5)
    
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Convergence Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_roc_curve(results_df: pd.DataFrame, output_file: str):
    """
    Plot: ROC curve (TPR vs. FPR) for Byzantine detection.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    methods = results_df['method'].unique()
    
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        
        fpr = method_data['false_positive_rate'].values
        tpr = method_data['byzantine_detection_rate'].values
        
        ax.plot(fpr, tpr, marker='o', label=method, linewidth=2)
    
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve: Byzantine Detection')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_overhead_comparison(results_df: pd.DataFrame, output_file: str):
    """
    Plot: Communication and computation overhead comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = results_df['method'].unique()
    
    # Communication overhead
    comm_overheads = [results_df[results_df['method'] == m]['communication_overhead'].mean() 
                      for m in methods]
    ax1.bar(methods, comm_overheads)
    ax1.set_ylabel('Communication Overhead (%)')
    ax1.set_title('Communication Overhead vs. Baseline')
    ax1.tick_params(axis='x', rotation=45)
    
    # Computation time
    comp_times = [results_df[results_df['method'] == m]['avg_round_time'].mean() 
                  for m in methods]
    ax2.bar(methods, comp_times)
    ax2.set_ylabel('Avg. Round Time (s)')
    ax2.set_title('Computation Time per Round')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_scalability(results_df: pd.DataFrame, output_file: str):
    """
    Plot: Scalability (time vs. number of clients).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by number of clients
    grouped = results_df.groupby('num_clients')['avg_round_time']
    means = grouped.mean()
    stds = grouped.std()
    
    ax.plot(means.index, means.values, marker='o', linewidth=2, label='Empirical')
    ax.fill_between(means.index, 
                    means - stds,
                    means + stds,
                    alpha=0.2)
    
    # Fit polynomial to estimate complexity
    z = np.polyfit(np.log(means.index), np.log(means.values), 1)
    alpha = z[0]  # Exponent in O(n^α)
    
    # Plot theoretical curve
    theoretical = means.values[0] * (means.index / means.index[0]) ** alpha
    ax.plot(means.index, theoretical, '--', linewidth=2, 
            label=f'O(n^{alpha:.2f})')
    
    ax.set_xlabel('Number of Clients')
    ax.set_ylabel('Avg. Round Time (s)')
    ax.set_title('Scalability: Time Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_ablation_heatmap(ablation_results: pd.DataFrame, output_file: str):
    """
    Plot: Heatmap showing contribution of each defense layer.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Pivot data for heatmap
    pivot = ablation_results.pivot(
        index='attack_type',
        columns='defense_config',
        values='final_accuracy'
    )
    
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, ax=ax)
    
    ax.set_title('Ablation Study: Defense Layer Contribution')
    ax.set_xlabel('Defense Configuration')
    ax.set_ylabel('Attack Type')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


if __name__ == '__main__':
    # Example usage
    print("Generating publication-quality plots...")
    
    # Load results
    results = load_experiment_results('./results')
    df = pd.DataFrame(results)
    
    # Generate plots
    plot_accuracy_vs_byzantine_fraction(df, './figures/accuracy_vs_byzantine.png')
    plot_convergence_curves(results[:5], './figures/convergence.png')
    plot_roc_curve(df, './figures/roc_curve.png')
    plot_overhead_comparison(df, './figures/overhead.png')
    plot_scalability(df, './figures/scalability.png')
    
    print("\n✓ All plots generated!")
```

---

### 10.2 LaTeX Tables for Papers

**File: `scripts/generate_latex_tables.py`**

```python
"""
Generate LaTeX tables for research papers.
"""

import pandas as pd

def generate_comparison_table(results_df: pd.DataFrame, output_file: str):
    """Generate LaTeX table comparing methods."""
    
    # Group and compute stats
    summary = results_df.groupby('method').agg({
        'final_accuracy': ['mean', 'std'],
        'byzantine_detection_rate': ['mean', 'std'],
        'false_positive_rate': ['mean', 'std'],
        'avg_round_time': ['mean', 'std']
    }).round(4)
    
    # Format as LaTeX
    latex_table = r"""\begin{table}[htbp]
\centering
\caption{Comparative Performance of Byzantine Defense Methods}
\label{tab:comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Accuracy (\%)} & \textbf{Detection (TPR)} & \textbf{False Pos. (FPR)} & \textbf{Time (s)} \\
\midrule
"""
    
    for method, row in summary.iterrows():
        acc_mean = row[('final_accuracy', 'mean')] * 100
        acc_std = row[('final_accuracy', 'std')] * 100
        tpr_mean = row[('byzantine_detection_rate', 'mean')]
        tpr_std = row[('byzantine_detection_rate', 'std')]
        fpr_mean = row[('false_positive_rate', 'mean')]
        fpr_std = row[('false_positive_rate', 'std')]
        time_mean = row[('avg_round_time', 'mean')]
        time_std = row[('avg_round_time', 'std')]
        
        latex_table += f"{method} & ${acc_mean:.1f} \\pm {acc_std:.1f}$ & "
        latex_table += f"${tpr_mean:.2f} \\pm {tpr_std:.2f}$ & "
        latex_table += f"${fpr_mean:.2f} \\pm {fpr_std:.2f}$ & "
        latex_table += f"${time_mean:.2f} \\pm {time_std:.2f}$ \\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"✓ LaTeX table saved: {output_file}")
```

---

## Summary: Complete Evaluation Workflow

### Step-by-Step Research Methodology

**Week 1-2: Baseline Benchmarking**
1. Run `scripts/eval_baseline_iid.py`
2. Run `scripts/eval_baseline_noniid_label.py`
3. Run `scripts/eval_baseline_noniid_quantity.py`
4. Run `scripts/eval_baseline_cifar10.py`
5. Measure overheads: `scripts/eval_overhead_*.py`

**Week 3-4: Attack Evaluation**
1. Run `scripts/eval_attack_label_flip.py`
2. Run `scripts/eval_attack_backdoor.py`
3. Run `scripts/eval_attack_model_poison.py`
4. Run `scripts/eval_attack_adaptive.py`

**Week 5-6: Comparative Benchmarking**
1. Run `scripts/eval_comparative_benchmark.py`
2. Compare against all baseline methods

**Week 7: Scalability Testing**
1. Run `scripts/eval_scalability_clients.py`
2. Run `scripts/eval_scalability_galaxies.py`
3. Run `scripts/eval_scalability_rounds.py`

**Week 8: Ablation Studies**
1. Run `scripts/eval_ablation_layers.py`
2. Run `scripts/eval_ablation_thresholds.py`
3. Run `scripts/eval_ablation_zkp.py`

**Week 9: Analysis & Visualization**
1. Run `scripts/analyze_results.py`
2. Run `scripts/visualize_results.py`
3. Generate LaTeX tables

**Week 10: Paper Writing**
1. Incorporate results into paper
2. Create publication-ready figures
3. Write analysis sections

---

**Total Estimated Experiments:** ~500-800 individual runs  
**Estimated Compute Time:** ~200-400 GPU hours  
**Expected Publication Impact:** Top-tier ML/Security conference (NeurIPS, ICLR, IEEE S&P)

---

**Document End**
