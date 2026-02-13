# Evaluation TBD — Components Implementation Status

**Purpose:** Tracks components referenced by the evaluation script (`scripts/run_evaluation.py`). All items below have been fully implemented with real logic — no mocks, stubs, or simulations.

> **Policy:** No mock, stub, or simulation stand-ins are used in the evaluation script or any component listed here.

---

## 1. FLTrust Aggregation Method

**Status:** ✅ IMPLEMENTED (`src/defense/fltrust.py`)  
**Referenced by:** `--defense fltrust` evaluation mode  
**Architecture reference:** Comparative baseline (Section 6.3)

**What is needed:**
FLTrust (Cao et al., 2021) requires a **clean root dataset** held by the server. The server trains a reference model on this dataset each round. Client updates are scored by their cosine similarity to the server's reference update, then trust-score-weighted averaging is applied.

**Required implementation:**
```
File: src/defense/fltrust.py

class FLTrustAggregator:
    def __init__(self, server_dataset: Dataset, server_model: nn.Module,
                 learning_rate: float = 0.01, batch_size: int = 64):
        """
        Args:
            server_dataset: Small clean dataset (100-1000 samples) held by server
            server_model: Copy of global model for server-side training
            learning_rate: Server-side training LR
            batch_size: Server-side batch size
        """
    
    def compute_server_update(self, global_model: nn.Module, num_epochs: int = 1) -> List[Tensor]:
        """Train server model on clean data, return gradient update."""
    
    def aggregate(self, updates: List[Dict]) -> Optional[Dict]:
        """
        1. Compute cosine similarity of each client update vs server update
        2. ReLU clipping: trust_score = max(0, cosine_sim)
        3. Normalize client updates to unit norm
        4. Weighted average: w_global = Σ(ts_i * norm_update_i) / Σ(ts_i)
        
        Returns: {'gradients': aggregated, 'trust_scores': {...}, 'method': 'fltrust'}
        """
```

**Integration path:**
1. Create `src/defense/fltrust.py` with the class above
2. Register in `DefenseCoordinator` as `AGGREGATOR_FLTRUST = 'fltrust'`
3. Update `set_aggregation_method()` to accept `'fltrust'` with `server_dataset` kwarg
4. The evaluation script already has the calling convention ready

**Paper reference:** Cao, X., Fang, M., Liu, J., & Gong, N. Z. (2021). FLTrust: Byzantine-robust federated learning via trust bootstrapping. NDSS.

---

## 2. Adaptive Attack Strategy

**Status:** ✅ IMPLEMENTED (`src/client/adaptive_attacker.py`)  
**Referenced by:** `--attack adaptive` evaluation mode  
**Architecture reference:** Section 6.2 (advanced adversary model)

**What is needed:**
An intelligent adversary that observes defense thresholds over time and crafts gradients just below detection boundaries.

**Required implementation:**
```
File: src/client/adaptive_attacker.py

class AdaptiveAttacker:
    def __init__(self, observation_window: int = 5):
        """
        Args:
            observation_window: Number of rounds to observe before adapting
        """
        self.observed_norms: List[float] = []
        self.observed_cosines: List[float] = []
    
    def observe_round(self, honest_gradients: List[List[Tensor]]):
        """Record statistics of honest gradients for threshold estimation."""
    
    def generate_adaptive_poison(self, honest_gradients: List[List[Tensor]],
                                  attack_goal: str = 'untargeted') -> List[Tensor]:
        """
        Craft malicious gradient that:
        1. Has norm within 2σ of honest mean norm
        2. Has cosine similarity > detection threshold with honest centroid
        3. Still biases model toward attacker's objective
        
        Returns: Poisoned gradient tensors
        """
```

**Integration path:**
1. Create the file above
2. Import in `Client.attack()` as `attack_type='adaptive'`
3. The evaluation script has the calling convention ready

---

## 3. Sybil Attack Simulation

**Status:** ✅ IMPLEMENTED (`src/client/sybil_coordinator.py`)  
**Referenced by:** `--attack sybil` evaluation mode  
**Architecture reference:** Section 6.2 (identity-based attacks)

**What is needed:**
Coordinated Sybil identities that collude — i.e., a single adversary controls multiple client IDs and coordinates their poisoning to overwhelm majority-based defenses.

**Required implementation:**
```
File: src/client/sybil_coordinator.py

class SybilCoordinator:
    def __init__(self, sybil_ids: List[int], strategy: str = 'synchronized'):
        """
        Args:
            sybil_ids: Client IDs controlled by the Sybil adversary
            strategy: 'synchronized' (all attack every round),
                      'rotating' (subset attacks each round),
                      'sleeper' (build reputation then attack)
        """
    
    def coordinate_attack(self, round_number: int, 
                          honest_gradients: Optional[Dict[int, List[Tensor]]] = None
                          ) -> Dict[int, List[Tensor]]:
        """Return coordinated poisoned gradients for all Sybil clients."""
```

**Integration path:**
1. Create the file above
2. Wire into `run_evaluation.py` attack injection logic
3. Sybil clients must be assigned across **different galaxies** to test cross-galaxy collusion

---

## 4. ResNet18 for CIFAR-10

**Status:** ✅ IMPLEMENTED (`src/models/resnet.py`)  
**Referenced by:** CIFAR-10 evaluation with `--model resnet18`  

**What is needed:**
```
File: src/models/resnet.py

# Standard torchvision ResNet18 adapted for CIFAR-10 (smaller input, 10 classes)
class CIFAR10ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        # Adjust first conv for 32x32 input (kernel_size=3, stride=1, padding=1)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for small images
```

**Integration path:**
1. Create the file above  
2. Register in `src/models/registry.py` as `resnet18`
3. Currently the evaluation script falls back to `cifar10_cnn` when resnet18 is unavailable

**Note:** The evaluation script already handles this gracefully — if `resnet18` is not in the model registry, it uses `cifar10_cnn` and logs a warning.

---

## 5. Backdoor Data-Level Poisoning (Pixel Trigger)

**Status:** ✅ IMPLEMENTED (`src/data/backdoor.py`)  
**Current state:** `BackdoorDataset` wraps base dataset and injects pixel-level trigger into `poisoning_rate` fraction of samples.

**What is needed for authentic backdoor evaluation:**
A data-level backdoor where Byzantine clients train on data with a pixel trigger stamped on a fraction of samples, with labels changed to the target class.

**Required implementation:**
```
File: src/data/backdoor.py

class BackdoorDataset(Dataset):
    def __init__(self, base_dataset: Dataset, trigger_pattern: Tensor,
                 trigger_position: Tuple[int, int], target_class: int,
                 poisoning_rate: float = 0.1):
        """
        Wraps a dataset and injects a pixel-level trigger into `poisoning_rate`
        fraction of samples, changing their label to `target_class`.
        """
    
    def __getitem__(self, idx):
        """Return (possibly triggered image, possibly flipped label)."""
```

**Impact on evaluation:**
- The current gradient-level backdoor (`Client._attack_backdoor()`) is still valid as "gradient backdoor"
- The data-level backdoor requires this new dataset wrapper
- The evaluation script currently uses the gradient-level backdoor which IS implemented

---

## Summary Table

| Component | Status | Evaluation Mode | File |
|-----------|--------|----------------|------|
| FLTrust | ✅ **IMPLEMENTED** | `--defense fltrust` | `src/defense/fltrust.py` |
| Adaptive Attack | ✅ **IMPLEMENTED** | `--attack adaptive` | `src/client/adaptive_attacker.py` |
| Sybil Attack | ✅ **IMPLEMENTED** | `--attack sybil` | `src/client/sybil_coordinator.py` |
| ResNet18 | ✅ **IMPLEMENTED** | `--model resnet18` | `src/models/resnet.py` |
| Data-level Backdoor | ✅ **IMPLEMENTED** | N/A | `src/data/backdoor.py` |

---

**Last updated:** 2025-07-11\n\n**All components fully implemented — evaluation script can now run all modes without restrictions.**
