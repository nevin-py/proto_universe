#!/usr/bin/env python3
"""
ProtoGalaxy Comprehensive Evaluation Script
=============================================

Runs reproducible FL experiments across every combination of:
  - Defense methods:  vanilla | multi_krum | fltrust | protogalaxy_full
  - Attack scenarios: none | label_flip | targeted_label_flip | model_poisoning |
                      backdoor | gaussian_noise | adaptive | sybil
  - Data partitions:  iid | noniid | dirichlet
  - Ablation modes:   merkle_only | zk_merkle
  - Scale configs:    varying num_clients × num_galaxies

Every evaluation uses ONLY real, implemented architecture functions — no mock,
stub, or placeholder logic.  Components that are not yet implemented will raise
a clear error pointing to ``eval_tbd.md``.

Usage
-----
# 1. Baseline vanilla FL, IID, no attack, 5 trials
python scripts/run_evaluation.py --mode baseline --trials 5

# 2. All attack scenarios against all defenses (default scale)
python scripts/run_evaluation.py --mode attacks --trials 3

# 3. Ablation study: merkle-only vs zk-merkle
python scripts/run_evaluation.py --mode ablation --trials 3

# 4. Scalability sweep
python scripts/run_evaluation.py --mode scalability --trials 2

# 5. Full evaluation (all of the above — WARNING: ~500+ runs)
python scripts/run_evaluation.py --mode full --trials 3

# 6. Single custom experiment
python scripts/run_evaluation.py --mode custom \\
    --defense protogalaxy_full --attack label_flip \\
    --partition iid --num-clients 50 --num-galaxies 5 \\
    --byzantine-fraction 0.2 --num-rounds 30 --trials 3

# 7. Dry-run: print experiment matrix without executing
python scripts/run_evaluation.py --mode full --dry-run

# 8. Resume from a previous run (skip completed experiments)
python scripts/run_evaluation.py --mode full --trials 3 --resume

Assumptions
-----------
1. MNIST dataset will be auto-downloaded to ``./data`` on first run.
2. CIFAR-10 experiments require separate download (``--dataset cifar10``).
3. The Rust ZKP bridge (``fl_zkp_bridge``) must be compiled for REAL mode:
       cd sonobe/fl-zkp-bridge && maturin develop --release
   If unavailable, ZKP falls back to SHA-256 commitments and ablation will
   note this in the results JSON.
4. ``fltrust`` defense, ``adaptive`` attack, and ``sybil`` attack are now
   fully implemented in ``src/defense/fltrust.py``,
   ``src/client/adaptive_attacker.py``, and ``src/client/sybil_coordinator.py``.
5. Galaxy assignment is round-robin (client_id % num_galaxies) as in the
   main pipeline.
6. Byzantine clients are always the first ``int(num_clients * byzantine_fraction)``
   client IDs for reproducibility.
7. Each trial uses seed = ``base_seed + trial_id`` (default base_seed=42).
8. All timing measurements use ``time.perf_counter`` (wall-clock).
9. The ``protogalaxy_full`` defense uses all 5 layers including ZKP prove/verify/fold.
10. The ``vanilla`` defense bypasses all defense layers and uses simple averaging.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    import GPUtil
    _GPUTIL_AVAILABLE = True
except ImportError:
    _GPUTIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.client.trainer import Trainer
from src.crypto.merkle import compute_hash, verify_proof as merkle_verify_proof
from src.crypto.zkp_prover import (
    GalaxyProofFolder,
    GradientSumCheckProver,
    ZKProof,
)
from src.data.datasets import load_cifar10 as _load_cifar10
from src.data.datasets import load_mnist as _load_mnist
from src.data.loader import create_client_loaders, create_test_loader
from src.data.partition import (
    DirichletPartitioner,
    IIDPartitioner,
    NonIIDPartitioner,
)
from src.defense.coordinator import DefenseCoordinator
from src.defense.robust_agg import (
    CoordinateWiseMedianAggregator,
    MultiKrumAggregator,
    TrimmedMeanAggregator,
)
from src.defense.reputation import EnhancedReputationManager, ClientStatus
from src.defense.statistical import StatisticalAnalyzer
from src.logging import FLLogger, FLLoggerFactory, LogLevel
from src.models.mnist import create_mnist_model
from src.models.registry import create_model, list_models, count_parameters
from src.orchestration.pipeline import ProtoGalaxyPipeline
from src.storage.forensic_logger import ForensicLogger
from src.storage.manager import StorageManager
from src.utils.gradient_ops import (
    flatten_gradients,
    compute_gradient_norm,
    gradient_cosine_similarity,
    average_gradients,
)
from src.defense.fltrust import FLTrustAggregator
from src.client.adaptive_attacker import AdaptiveAttacker
from src.client.sybil_coordinator import SybilCoordinator
from src.data.backdoor import BackdoorDataset

logger = logging.getLogger("protogalaxy.eval")

# Import specialized evaluators
try:
    import sys
    eval_scripts_path = Path(__file__).parent
    sys.path.insert(0, str(eval_scripts_path))
    from evaluate_zkp_performance import ZKPPerformanceEvaluator
    from evaluate_attack_rejection import ByzantineAttackEvaluator
    ZKP_PERF_AVAILABLE = True
    ATTACK_REJECTION_AVAILABLE = True
except ImportError as e:
    # Logger may not be configured yet, just set availability flags
    ZKP_PERF_AVAILABLE = False
    ATTACK_REJECTION_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_DEFENSES = ("vanilla", "multi_krum", "fltrust", "protogalaxy_full")
SUPPORTED_ATTACKS = (
    "none",
    "label_flip",
    "targeted_label_flip",
    "model_poisoning",
    "backdoor",
    "gaussian_noise",
    "adaptive",
    "sybil",
)
SUPPORTED_PARTITIONS = ("iid", "noniid", "dirichlet")
ABLATION_MODES = ("merkle_only", "zk_merkle")
EVAL_MODES = (
    "baseline", "attacks", "ablation", "scalability",
    "zkp_performance", "attack_rejection", "full", "custom"
)
EVAL_TBD_PATH = PROJECT_ROOT / "eval_tbd.md"


# ============================================================================
# System Resource Monitor
# ============================================================================

class SystemResourceMonitor:
    """Tracks CPU, GPU, and RAM usage during experiment execution.
    
    Samples system metrics in a background thread at configurable intervals.
    Produces per-experiment and aggregate statistics suitable for paper tables.
    """

    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, float]] = []
        self._process = psutil.Process(os.getpid())

    def start(self):
        """Begin background sampling."""
        self._samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return aggregated stats."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        return self._aggregate()

    def _sample_loop(self):
        while self._running:
            sample: Dict[str, float] = {
                'timestamp': time.time(),
                'cpu_percent': self._process.cpu_percent(interval=0),
                'ram_mb': self._process.memory_info().rss / (1024 * 1024),
                'ram_percent': self._process.memory_percent(),
            }
            # GPU metrics
            if torch.cuda.is_available():
                sample['gpu_mem_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                sample['gpu_mem_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                if _GPUTIL_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        sample['gpu_util_percent'] = gpus[0].load * 100
                        sample['gpu_mem_util_percent'] = gpus[0].memoryUtil * 100
                        sample['gpu_temp_c'] = gpus[0].temperature
            self._samples.append(sample)
            time.sleep(self.sample_interval)

    def _aggregate(self) -> Dict[str, Any]:
        if not self._samples:
            return {'num_samples': 0}

        keys_to_agg = [
            'cpu_percent', 'ram_mb', 'ram_percent',
            'gpu_mem_allocated_mb', 'gpu_mem_reserved_mb',
            'gpu_util_percent', 'gpu_mem_util_percent', 'gpu_temp_c',
        ]
        stats: Dict[str, Any] = {'num_samples': len(self._samples)}
        for key in keys_to_agg:
            values = [s[key] for s in self._samples if key in s]
            if values:
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'max': float(np.max(values)),
                    'min': float(np.min(values)),
                    'std': float(np.std(values)),
                }
        return stats


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class ExperimentConfig:
    """Fully describes one evaluation experiment."""

    # Identifiers
    experiment_id: str = ""
    mode: str = "custom"
    trial_id: int = 0
    seed: int = 42

    # Dataset / model
    dataset: str = "mnist"
    model_type: str = "linear"
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    classes_per_client: int = 2

    # FL setup
    num_clients: int = 20
    num_galaxies: int = 4
    num_rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.01

    # Defense
    defense: str = "protogalaxy_full"
    aggregation_method: str = "trimmed_mean"
    trim_ratio: float = 0.1
    ablation: str = ""  # "" | "merkle_only" | "zk_merkle"

    # Attack
    attack: str = "none"
    byzantine_fraction: float = 0.0
    attack_scale: float = 10.0
    backdoor_scale: float = 0.1

    # Optimization parameters
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    use_amp: bool = True
    
    # Evaluation
    eval_every: int = 1  # Evaluate every N rounds

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = (
                f"{self.defense}__{self.attack}__{self.partition}"
                f"__c{self.num_clients}_g{self.num_galaxies}"
                f"__byz{int(self.byzantine_fraction * 100)}"
                f"__t{self.trial_id}"
            )
            if self.ablation:
                self.experiment_id += f"__{self.ablation}"


@dataclass
class RoundMetrics:
    """Metrics collected per FL round."""

    round_num: int = 0
    accuracy: float = 0.0
    loss: float = 0.0

    # Detection
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    tpr: float = 0.0
    fpr: float = 0.0
    precision: float = 0.0
    f1: float = 0.0

    # Flagged sets
    flagged_client_ids: List[int] = field(default_factory=list)
    quarantined_client_ids: List[int] = field(default_factory=list)
    banned_client_ids: List[int] = field(default_factory=list)

    # Timing (seconds)
    round_time: float = 0.0
    phase1_time: float = 0.0
    phase2_time: float = 0.0
    phase3_time: float = 0.0
    phase4_time: float = 0.0
    zk_prove_time: float = 0.0
    zk_verify_time: float = 0.0
    zk_fold_time: float = 0.0
    merkle_build_time: float = 0.0
    merkle_verify_time: float = 0.0

    # ZKP
    zk_mode: str = "NONE"
    zk_proofs_generated: int = 0
    zk_proofs_verified: int = 0
    zk_proofs_failed: int = 0

    # Communication
    bytes_sent: int = 0


@dataclass
class ExperimentResult:
    """Complete result of one experiment run."""

    config: ExperimentConfig
    rounds: List[RoundMetrics] = field(default_factory=list)

    # Aggregate metrics
    final_accuracy: float = 0.0
    best_accuracy: float = 0.0
    convergence_round_85: int = -1
    convergence_round_90: int = -1
    convergence_round_95: int = -1
    total_time: float = 0.0

    # Detection aggregates
    avg_tpr: float = 0.0
    avg_fpr: float = 0.0
    avg_precision: float = 0.0
    avg_f1: float = 0.0

    # Overhead
    avg_round_time: float = 0.0
    avg_zk_prove_time: float = 0.0
    avg_zk_verify_time: float = 0.0
    avg_merkle_time: float = 0.0
    total_bytes: int = 0

    # Reputation snapshot
    final_reputations: Dict[int, float] = field(default_factory=dict)

    # System resource usage
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def compute_aggregates(self):
        """Derive aggregate metrics from per-round data."""
        if not self.rounds:
            return
        accs = [r.accuracy for r in self.rounds]
        self.final_accuracy = accs[-1]
        self.best_accuracy = max(accs)
        self.convergence_round_85 = _convergence_round(accs, 0.85)
        self.convergence_round_90 = _convergence_round(accs, 0.90)
        self.convergence_round_95 = _convergence_round(accs, 0.95)
        self.total_time = sum(r.round_time for r in self.rounds)
        self.avg_round_time = float(np.mean([r.round_time for r in self.rounds]))

        # Detection metrics (only meaningful when byzantines exist)
        tprs = [r.tpr for r in self.rounds]
        fprs = [r.fpr for r in self.rounds]
        precs = [r.precision for r in self.rounds]
        f1s = [r.f1 for r in self.rounds]
        self.avg_tpr = float(np.mean(tprs)) if tprs else 0.0
        self.avg_fpr = float(np.mean(fprs)) if fprs else 0.0
        self.avg_precision = float(np.mean(precs)) if precs else 0.0
        self.avg_f1 = float(np.mean(f1s)) if f1s else 0.0

        # ZKP / Merkle timings
        self.avg_zk_prove_time = float(np.mean([r.zk_prove_time for r in self.rounds]))
        self.avg_zk_verify_time = float(np.mean([r.zk_verify_time for r in self.rounds]))
        self.avg_merkle_time = float(
            np.mean([r.merkle_build_time + r.merkle_verify_time for r in self.rounds])
        )
        self.total_bytes = sum(r.bytes_sent for r in self.rounds)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "config": asdict(self.config),
            "final_accuracy": self.final_accuracy,
            "best_accuracy": self.best_accuracy,
            "convergence_round_85": self.convergence_round_85,
            "convergence_round_90": self.convergence_round_90,
            "convergence_round_95": self.convergence_round_95,
            "total_time": self.total_time,
            "avg_round_time": self.avg_round_time,
            "avg_tpr": self.avg_tpr,
            "avg_fpr": self.avg_fpr,
            "avg_precision": self.avg_precision,
            "avg_f1": self.avg_f1,
            "avg_zk_prove_time": self.avg_zk_prove_time,
            "avg_zk_verify_time": self.avg_zk_verify_time,
            "avg_merkle_time": self.avg_merkle_time,
            "total_bytes": self.total_bytes,
            "final_reputations": {str(k): v for k, v in self.final_reputations.items()},
            "resource_usage": self.resource_usage,
            "timestamp": self.timestamp,
            "rounds": [asdict(r) for r in self.rounds],
        }
        return d


# ============================================================================
# Helpers
# ============================================================================

def _convergence_round(accuracies: List[float], target: float) -> int:
    for i, acc in enumerate(accuracies):
        if acc >= target:
            return i
    return -1


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataset(name: str, data_dir: str = "./data"):
    """Load train/test datasets using src.data.datasets loaders."""
    logger.info(f"Loading dataset '{name}' from {data_dir}")
    load_start = time.perf_counter()
    if name == "mnist":
        train_ds, test_ds = _load_mnist(data_dir=os.path.join(data_dir, "mnist"))
    elif name == "cifar10":
        train_ds, test_ds = _load_cifar10(data_dir=os.path.join(data_dir, "cifar10"))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    elapsed = time.perf_counter() - load_start
    logger.info(
        f"Dataset '{name}' loaded in {elapsed:.2f}s — "
        f"train={len(train_ds)} samples, test={len(test_ds)} samples"
    )
    return train_ds, test_ds


def _partition_data(
    dataset, num_clients: int, method: str, seed: int,
    alpha: float = 0.5, classes_per_client: int = 2,
) -> Dict[int, List[int]]:
    """Partition dataset using src.data.partition partitioners.

    Returns dict of client_id -> list of sample indices.
    """
    if method == "iid":
        partitioner = IIDPartitioner(seed=seed)
    elif method == "noniid":
        partitioner = NonIIDPartitioner(
            num_classes_per_client=classes_per_client, seed=seed,
        )
    elif method == "dirichlet":
        partitioner = DirichletPartitioner(alpha=alpha, seed=seed)
    else:
        raise ValueError(f"Unknown partition method: {method}")

    part_start = time.perf_counter()
    partitions = partitioner.partition(dataset, num_clients)
    elapsed = time.perf_counter() - part_start
    sizes = [len(v) for v in partitions.values()]
    logger.info(
        f"Partitioned {len(dataset)} samples across {num_clients} clients "
        f"(method={method}, {elapsed:.2f}s) — "
        f"min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.0f} samples/client"
    )
    return partitions


def _create_model(dataset: str, model_type: str) -> nn.Module:
    """Create model using src.models factories."""
    if dataset == "mnist":
        return create_mnist_model(model_type)
    elif dataset == "cifar10":
        available = list_models()
        if model_type in available:
            return create_model(model_type)
        elif "cifar10_cnn" in available:
            logger.warning(
                f"Model '{model_type}' not in registry; falling back to 'cifar10_cnn'. "
                f"See eval_tbd.md §4 (ResNet18)."
            )
            return create_model("cifar10_cnn")
        else:
            raise ValueError(
                f"No CIFAR-10 model available. Available: {available}"
            )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def _evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = "cpu") -> float:
    """Evaluate global model accuracy on test set.

    Uses the same logic as ``run_pipeline.evaluate_model``.
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0


def _inject_attack(
    gradients: List[torch.Tensor],
    attack: str,
    client_id: int,
    attack_scale: float = 10.0,
    backdoor_scale: float = 0.1,
) -> List[torch.Tensor]:
    """Apply attack to a Byzantine client's gradients.

    Uses the **same** attack logic as ``src.client.client.Client`` methods
    (``_attack_label_flip``, ``_attack_model_poisoning``, etc.) but applied
    inline so we don't depend on the ``Client`` class orchestration path.

    Parameters
    ----------
    gradients : list of Tensor
        Clean gradients from local training.
    attack : str
        One of the SUPPORTED_ATTACKS (excluding 'none').
    client_id : int
        Used as seed offset for reproducible backdoor triggers.
    attack_scale : float
        Amplification factor for model_poisoning.
    backdoor_scale : float
        Trigger magnitude for backdoor.

    Returns
    -------
    list of Tensor
        Poisoned gradients.
    """
    if attack == "none":
        return gradients

    if attack == "label_flip":
        # Client._attack_label_flip: negate all gradients
        return [-g for g in gradients]

    if attack == "targeted_label_flip":
        # Client._attack_targeted_label_flip: flip last 2 layers
        out = list(gradients)
        for li in range(max(0, len(out) - 2), len(out)):
            out[li] = -out[li]
        return out

    if attack == "model_poisoning":
        # Client._attack_model_poisoning: scale * negate
        return [g * (-attack_scale) for g in gradients]

    if attack == "backdoor":
        # Client._attack_backdoor: add reproducible trigger
        rng = np.random.RandomState(42 + client_id)
        out = []
        for g in gradients:
            if isinstance(g, torch.Tensor):
                trigger = torch.tensor(
                    rng.randn(*g.shape).astype(np.float32),
                    device=g.device,
                    dtype=g.dtype,
                )
                out.append(g + backdoor_scale * trigger)
            else:
                trigger = rng.randn(*np.array(g).shape).astype(np.float32)
                out.append(g + backdoor_scale * trigger)
        return out

    if attack == "gaussian_noise":
        # Client._attack_gaussian_noise: additive Gaussian
        out = []
        for g in gradients:
            if isinstance(g, torch.Tensor):
                out.append(g + torch.randn_like(g))
            else:
                out.append(g + np.random.randn(*np.array(g).shape).astype(np.float32))
        return out

    if attack == "adaptive":
        # Adaptive attacks are handled post-training-loop, not here.
        # If called here directly, return gradients unchanged (stealth mode).
        return gradients

    if attack == "sybil":
        # Sybil attacks are coordinated post-training-loop by SybilCoordinator.
        # If called here directly, return gradients unchanged (stealth mode).
        return gradients

    raise ValueError(f"Unknown attack: {attack}")


def _check_fltrust_available():
    """Check if FLTrust aggregator is available."""
    try:
        from src.defense.fltrust import FLTrustAggregator  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================
# Core experiment runner
# ============================================================================

def run_single_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Execute one complete FL experiment end-to-end.

    This function mirrors ``run_pipeline.main()`` but is structured for
    automated evaluation with full metrics extraction.

    Parameters
    ----------
    cfg : ExperimentConfig
        Complete experiment specification.

    Returns
    -------
    ExperimentResult
        All collected metrics.
    """
    # ── Pre-flight checks ──────────────────────────────────────────────
    _set_seed(cfg.seed)
    result = ExperimentResult(config=cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Start resource monitoring
    resource_monitor = SystemResourceMonitor(sample_interval=2.0)
    resource_monitor.start()
    logger.debug("SystemResourceMonitor started (interval=2.0s)")

    logger.info("")
    logger.info("=" * 72)
    logger.info(f"EXPERIMENT: {cfg.experiment_id}")
    logger.info("=" * 72)
    logger.info(f"  Defense:      {cfg.defense} (agg={cfg.aggregation_method})")
    logger.info(f"  Attack:       {cfg.attack} (byzantine={cfg.byzantine_fraction:.0%}, scale={cfg.attack_scale})")
    logger.info(f"  FL Setup:     {cfg.num_clients} clients, {cfg.num_galaxies} galaxies, {cfg.num_rounds} rounds")
    logger.info(f"  Training:     lr={cfg.learning_rate}, batch={cfg.batch_size}, local_epochs={cfg.local_epochs}")
    logger.info(f"  Data:         {cfg.dataset} / {cfg.model_type} / {cfg.partition}")
    logger.info(f"  Seed:         {cfg.seed}, trial={cfg.trial_id}")
    logger.info(f"  Device:       {device}")
    if cfg.ablation:
        logger.info(f"  Ablation:     {cfg.ablation}")
    logger.info("-" * 72)

    # ── 1. Load data ───────────────────────────────────────────────────
    logger.info("[Phase: DATA] Loading dataset and partitioning...")
    train_dataset, test_dataset = _load_dataset(cfg.dataset)
    partitions = _partition_data(
        train_dataset, cfg.num_clients, cfg.partition, cfg.seed,
        alpha=cfg.dirichlet_alpha, classes_per_client=cfg.classes_per_client,
    )
    client_loaders = create_client_loaders(
        train_dataset, partitions, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, prefetch_factor=cfg.prefetch_factor,
    )
    test_loader = create_test_loader(
        test_dataset, batch_size=256,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, prefetch_factor=cfg.prefetch_factor,
    )

    # ── 2. Create model ───────────────────────────────────────────────
    logger.info("[Phase: MODEL] Creating and initializing model...")
    global_model = _create_model(cfg.dataset, cfg.model_type).to(device)
    num_params = count_parameters(global_model)
    initial_accuracy = _evaluate_model(global_model, test_loader, device)
    logger.info(
        f"  Model:            {type(global_model).__name__}\n"
        f"  Parameters:       {num_params:,}\n"
        f"  Device:           {device}\n"
        f"  Initial accuracy: {initial_accuracy:.4f} ({initial_accuracy:.2%})"
    )

    # ── 3. Byzantine setup ─────────────────────────────────────────────
    num_byzantine = int(cfg.num_clients * cfg.byzantine_fraction)
    byzantine_ids: Set[int] = set(range(num_byzantine))
    if num_byzantine > 0:
        logger.info(
            f"[Phase: BYZANTINE] {num_byzantine}/{cfg.num_clients} Byzantine clients "
            f"({cfg.byzantine_fraction:.0%}) — IDs: {sorted(byzantine_ids)}"
        )
        logger.info(f"  Attack type: {cfg.attack}, scale={cfg.attack_scale}")
    else:
        logger.info("[Phase: BYZANTINE] No Byzantine clients (clean run)")

    # ── 4. Build pipeline or vanilla path ──────────────────────────────
    logger.info(f"[Phase: DEFENSE] Configuring defense: {cfg.defense}")
    # Defense config passed to DefenseCoordinator inside ProtoGalaxyPipeline
    if cfg.defense == "vanilla":
        logger.info("  Strategy: Vanilla FedAvg — no defense, simple averaging")
        pipeline = None  # We drive vanilla manually
    elif cfg.defense == "multi_krum":
        # Use ProtoGalaxy pipeline but set aggregation to multi_krum
        # Cap f so 2f + 2 < clients_per_galaxy (user fix for Bug #6)
        clients_per_galaxy = cfg.num_clients // cfg.num_galaxies
        max_f = (clients_per_galaxy - 3) // 2
        krum_f = min(num_byzantine, max_f)
        if krum_f < 1:
            logger.warning(
                f"Galaxy size {clients_per_galaxy} too small for Multi-Krum (f={num_byzantine}). "
                "Setting f=1 (unsafe) or consider increasing clients/galaxy."
            )
            krum_f = 1
            
        defense_config = {
            "layer3_method": "multi_krum",
            "layer3_krum_f": krum_f,
            "layer3_krum_m": max(1, clients_per_galaxy - krum_f - 2),
        }
        pipeline = ProtoGalaxyPipeline(
            global_model=global_model,
            num_clients=cfg.num_clients,
            num_galaxies=cfg.num_galaxies,
            defense_config=defense_config,
        )
        logger.info(
            f"  Strategy: Multi-Krum\n"
            f"  krum_f={defense_config['layer3_krum_f']} (capped for n_gal={clients_per_galaxy}), "
            f"krum_m={defense_config['layer3_krum_m']} (selection count)"
        )
    elif cfg.defense == "fltrust":
        # FLTrust: use a small clean root dataset held by the server
        # Take a random 500-sample subset of training data as server root
        rng = np.random.RandomState(cfg.seed)
        root_indices = rng.choice(len(train_dataset), size=min(500, len(train_dataset)), replace=False)
        from torch.utils.data import Subset as _Subset
        server_root_dataset = _Subset(train_dataset, root_indices.tolist())
        server_model = copy.deepcopy(global_model)
        fltrust_agg = FLTrustAggregator(
            server_dataset=server_root_dataset,
            server_model=server_model,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
        )
        # Build pipeline with trimmed_mean as L3 (FLTrust replaces aggregation in the round)
        defense_config = {
            "layer3_method": "trimmed_mean",
            "layer3_trim_ratio": cfg.trim_ratio,
        }
        pipeline = ProtoGalaxyPipeline(
            global_model=global_model,
            num_clients=cfg.num_clients,
            num_galaxies=cfg.num_galaxies,
            defense_config=defense_config,
        )
        logger.info(
            f"  Strategy: FLTrust\n"
            f"  Server root dataset: {len(server_root_dataset)} samples\n"
            f"  Trust scores computed per-round via cosine similarity"
        )
    elif cfg.defense == "protogalaxy_full":
        defense_config = {
            "layer3_method": cfg.aggregation_method,
            "layer3_trim_ratio": cfg.trim_ratio,
        }
        pipeline = ProtoGalaxyPipeline(
            global_model=global_model,
            num_clients=cfg.num_clients,
            num_galaxies=cfg.num_galaxies,
            defense_config=defense_config,
        )
        logger.info(
            f"  Strategy: ProtoGalaxy Full 5-Layer Defense\n"
            f"  L1: Commitment + Merkle Tree\n"
            f"  L2: Statistical Analysis (z-score, MAD)\n"
            f"  L3: Robust Aggregation ({cfg.aggregation_method}, trim={cfg.trim_ratio})\n"
            f"  L4: Reputation Management\n"
            f"  L5: ZKP Norm-Bounded Proofs (ProtoGalaxy IVC)"
        )
    else:
        raise ValueError(f"Unknown defense: {cfg.defense}")

    # ── Adaptive / Sybil attack state ──────────────────────────────────
    adaptive_attacker = None
    sybil_coordinator = None
    if cfg.attack == "adaptive":
        adaptive_attacker = AdaptiveAttacker(
            observation_window=5,
            norm_sigma_bound=2.0,
            cos_threshold=0.3,
            attack_strength=2.0,
        )
        logger.info(
            "  Adaptive attacker initialized: window=5, norm_sigma=2.0, "
            "cos_thresh=0.3, strength=2.0"
        )
    elif cfg.attack == "sybil":
        sybil_coordinator = SybilCoordinator(
            sybil_ids=list(byzantine_ids),
            strategy="synchronized",
            attack_scale=cfg.attack_scale,
            seed=cfg.seed,
        )
        logger.info(
            f"  Sybil coordinator initialized: strategy=synchronized, "
            f"sybil_ids={sorted(byzantine_ids)}, scale={cfg.attack_scale}"
        )

    # ── 5. Training loop ───────────────────────────────────────────────
    logger.info("")
    logger.info(f"[Phase: TRAINING] Starting FL training — {cfg.num_rounds} rounds")
    logger.info("-" * 72)
    experiment_start = time.perf_counter()

    for round_num in range(cfg.num_rounds):
        rmetrics = RoundMetrics(round_num=round_num)
        round_start = time.perf_counter()
        logger.debug(f"Round {round_num}/{cfg.num_rounds} — starting local training")

        # ── 5a. Local training ─────────────────────────────────────────
        client_trainers: Dict[int, Trainer] = {}
        client_grads: Dict[int, List[torch.Tensor]] = {}

        for cid in range(cfg.num_clients):
            client_model = copy.deepcopy(global_model)
            trainer = Trainer(
                model=client_model, learning_rate=cfg.learning_rate,
                use_amp=cfg.use_amp,
            )
            loader = client_loaders[cid]
            trainer.train(loader, num_epochs=cfg.local_epochs)
            grads = trainer.get_gradients()

            # Optimization: Move gradients to CPU immediately to save GPU memory
            # This is critical for large-scale experiments (100+ clients)
            cpu_grads = [g.cpu() for g in grads]

            # Inject attack for Byzantine clients (skip adaptive/sybil here —
            # they are handled after all clients train so they can see honest grads)
            if cid in byzantine_ids and cfg.attack not in ("none", "adaptive", "sybil"):
                cpu_grads = _inject_attack(
                    cpu_grads, cfg.attack, cid,
                    attack_scale=cfg.attack_scale,
                    backdoor_scale=cfg.backdoor_scale,
                )
                logger.debug(f"    Client {cid}: injected {cfg.attack} attack (scale={cfg.attack_scale})")

            client_trainers[cid] = trainer
            client_grads[cid] = cpu_grads

            # Move model to CPU to save GPU memory
            trainer.model.cpu()
            # Clear optimizer state if it takes memory (optional but good practice)
            # trainer.optimizer = None 

            # Clear GPU cache after each client to prevent fragmentation
            # Clear GPU cache periodically to prevent fragmentation (every 10 clients)
            if torch.cuda.is_available() and (cid + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # ── 5a-post. Adaptive / Sybil attack (needs all honest grads) ─
        if cfg.attack == "adaptive" and adaptive_attacker is not None:
            honest_grads_list = [
                client_grads[cid] for cid in range(cfg.num_clients)
                if cid not in byzantine_ids
            ]
            for cid in byzantine_ids:
                client_grads[cid] = adaptive_attacker.generate_adaptive_poison(
                    honest_grads_list, attack_goal="untargeted",
                )
        elif cfg.attack == "sybil" and sybil_coordinator is not None:
            honest_grads_dict = {
                cid: client_grads[cid] for cid in range(cfg.num_clients)
                if cid not in byzantine_ids
            }
            sybil_honest_dict = {
                cid: client_grads[cid] for cid in byzantine_ids
            }
            sybil_result = sybil_coordinator.coordinate_attack(
                round_number=round_num,
                honest_gradients=honest_grads_dict,
                sybil_honest_gradients=sybil_honest_dict,
            )
            for cid, poisoned_grads in sybil_result.items():
                client_grads[cid] = poisoned_grads

        # ── 5a-fltrust. Compute server update before aggregation ───────
        if cfg.defense == "fltrust":
            fltrust_agg.compute_server_update(global_model)

        # ── 5b. Execute round (defense-dependent) ─────────────────────
        if cfg.defense == "vanilla":
            rmetrics = _run_vanilla_round(
                global_model, client_grads, rmetrics, device,
            )
        elif cfg.defense == "fltrust":
            rmetrics = _run_fltrust_round(
                fltrust_agg, global_model, client_grads, rmetrics, device,
            )
        else:
            rmetrics = _run_protogalaxy_round(
                pipeline, global_model, client_trainers, client_grads,
                round_num, cfg, rmetrics, byzantine_ids,
            )

        # ── 5c. Evaluate ──────────────────────────────────────────────
        # Only evaluate every `eval_every` rounds (and the last round)
        if (round_num + 1) % cfg.eval_every == 0 or (round_num + 1) == cfg.num_rounds:
            accuracy = _evaluate_model(global_model, test_loader, device)
            rmetrics.accuracy = accuracy
            logger.info(
                f"Round {round_num}/{cfg.num_rounds} — Eval accuracy: {accuracy:.4f}"
            )
        else:
            # Carry forward previous accuracy if not evaluating
            # (or 0.0 if not yet evaluated, though R0 is usually evaluated if eval_every=1)
            # If we skip, we just leave it as 0.0 or the default
            pass

        # ── 5d. Detection metrics ─────────────────────────────────────
        flagged_set = set(rmetrics.flagged_client_ids)
        honest_ids = set(range(cfg.num_clients)) - byzantine_ids
        tp = len(flagged_set & byzantine_ids)
        fp = len(flagged_set - byzantine_ids)
        fn = len(byzantine_ids - flagged_set)
        tn = len(honest_ids - flagged_set)
        rmetrics.true_positives = tp
        rmetrics.false_positives = fp
        rmetrics.false_negatives = fn
        rmetrics.true_negatives = tn
        rmetrics.tpr = tp / max(1, tp + fn)
        rmetrics.fpr = fp / max(1, fp + tn)
        rmetrics.precision = tp / max(1, tp + fp)
        rmetrics.f1 = (
            2 * rmetrics.precision * rmetrics.tpr /
            max(1e-9, rmetrics.precision + rmetrics.tpr)
        )

        # Communication size estimate
        param_bytes = sum(p.numel() * 4 for p in global_model.parameters())
        rmetrics.bytes_sent = param_bytes * cfg.num_clients * 2

        rmetrics.round_time = time.perf_counter() - round_start
        result.rounds.append(rmetrics)

        # Detailed per-round log (every round to DEBUG, periodic to INFO)
        round_log = (
            f"Round {round_num:>3d}/{cfg.num_rounds}  "
            f"acc={accuracy:.4f}  loss={rmetrics.loss:.4f}  "
            f"TPR={rmetrics.tpr:.3f}  FPR={rmetrics.fpr:.3f}  "
            f"prec={rmetrics.precision:.3f}  F1={rmetrics.f1:.3f}  "
            f"flagged={len(rmetrics.flagged_client_ids)}  "
            f"quar={len(rmetrics.quarantined_client_ids)}  "
            f"banned={len(rmetrics.banned_client_ids)}"
        )
        timing_log = (
            f"  timing: total={rmetrics.round_time:.2f}s  "
            f"P1={rmetrics.phase1_time:.3f}s  P2={rmetrics.phase2_time:.3f}s  "
            f"P3={rmetrics.phase3_time:.3f}s  P4={rmetrics.phase4_time:.3f}s"
        )
        zkp_log = (
            f"  zkp: mode={rmetrics.zk_mode}  "
            f"prove={rmetrics.zk_prove_time:.3f}s  verify={rmetrics.zk_verify_time:.3f}s  "
            f"fold={rmetrics.zk_fold_time:.3f}s  "
            f"gen={rmetrics.zk_proofs_generated}  ok={rmetrics.zk_proofs_verified}  "
            f"fail={rmetrics.zk_proofs_failed}"
        )
        merkle_log = (
            f"  merkle: build={rmetrics.merkle_build_time:.4f}s  "
            f"verify={rmetrics.merkle_verify_time:.4f}s"
        )

        # Always log full detail to DEBUG
        logger.debug(round_log)
        logger.debug(timing_log)
        logger.debug(zkp_log)
        logger.debug(merkle_log)

        # Periodic summary to INFO
        if round_num % max(1, cfg.num_rounds // 10) == 0 or round_num == cfg.num_rounds - 1:
            logger.info(round_log)

    # ── 6. Finalize results ────────────────────────────────────────────
    logger.info("")
    logger.info("[Phase: FINALIZE] Aggregating results and collecting metrics...")

    # Stop resource monitoring and collect stats
    result.resource_usage = resource_monitor.stop()

    # Collect final reputation scores if pipeline exists
    if pipeline is not None:
        for gid, coord in pipeline.galaxy_defense_coordinators.items():
            reps = coord.layer4.get_all_reputations()
            result.final_reputations.update(reps)
        # Log reputation summary
        if result.final_reputations:
            rep_values = list(result.final_reputations.values())
            logger.info(
                f"  Reputation scores: min={min(rep_values):.3f}, "
                f"max={max(rep_values):.3f}, avg={np.mean(rep_values):.3f}"
            )

    result.compute_aggregates()

    # Detailed experiment summary
    logger.info("")
    logger.info("=" * 72)
    logger.info(f"EXPERIMENT COMPLETE: {cfg.experiment_id}")
    logger.info("=" * 72)
    logger.info(f"  Accuracy:")
    logger.info(f"    Final:   {result.final_accuracy:.4f} ({result.final_accuracy:.2%})")
    logger.info(f"    Best:    {result.best_accuracy:.4f} ({result.best_accuracy:.2%})")
    logger.info(f"    Initial: {initial_accuracy:.4f} ({initial_accuracy:.2%})")
    logger.info(f"    Gain:    {result.final_accuracy - initial_accuracy:+.4f}")
    logger.info(f"  Convergence:")
    logger.info(f"    85%: round {result.convergence_round_85}")
    logger.info(f"    90%: round {result.convergence_round_90}")
    logger.info(f"    95%: round {result.convergence_round_95}")
    if num_byzantine > 0:
        logger.info(f"  Detection:")
        logger.info(f"    Avg TPR (recall):  {result.avg_tpr:.4f}")
        logger.info(f"    Avg FPR:           {result.avg_fpr:.4f}")
        logger.info(f"    Avg Precision:     {result.avg_precision:.4f}")
        logger.info(f"    Avg F1:            {result.avg_f1:.4f}")
    logger.info(f"  Timing:")
    logger.info(f"    Total:             {result.total_time:.2f}s ({result.total_time/60:.1f} min)")
    logger.info(f"    Avg round:         {result.avg_round_time:.3f}s")
    logger.info(f"    Avg ZK prove:      {result.avg_zk_prove_time:.4f}s")
    logger.info(f"    Avg ZK verify:     {result.avg_zk_verify_time:.4f}s")
    logger.info(f"    Avg Merkle:        {result.avg_merkle_time:.4f}s")
    logger.info(f"  Communication:")
    logger.info(f"    Total bytes:       {result.total_bytes:,} ({result.total_bytes/(1024**2):.1f} MB)")
    # Resource summary
    ru = result.resource_usage
    if ru.get('num_samples', 0) > 0:
        logger.info(f"  Resources ({ru['num_samples']} samples):")
        if 'cpu_percent' in ru:
            logger.info(f"    CPU:   avg={ru['cpu_percent']['mean']:.1f}%, max={ru['cpu_percent']['max']:.1f}%")
        if 'ram_mb' in ru:
            logger.info(f"    RAM:   avg={ru['ram_mb']['mean']:.0f} MB, peak={ru['ram_mb']['max']:.0f} MB")
        if 'gpu_util_percent' in ru:
            logger.info(f"    GPU:   avg={ru['gpu_util_percent']['mean']:.1f}%, max={ru['gpu_util_percent']['max']:.1f}%")
        if 'gpu_mem_allocated_mb' in ru:
            logger.info(f"    VRAM:  avg={ru['gpu_mem_allocated_mb']['mean']:.0f} MB, peak={ru['gpu_mem_allocated_mb']['max']:.0f} MB")
        if 'gpu_temp_c' in ru:
            logger.info(f"    Temp:  avg={ru['gpu_temp_c']['mean']:.0f}°C, max={ru['gpu_temp_c']['max']:.0f}°C")
    logger.info("=" * 72)

    # Cleanup
    del global_model, client_trainers, client_grads
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================================
# Round execution paths
# ============================================================================

def _run_vanilla_round(
    global_model: nn.Module,
    client_grads: Dict[int, List[torch.Tensor]],
    rmetrics: RoundMetrics,
    device: str,
) -> RoundMetrics:
    """Vanilla FedAvg: simple averaging, no defense.

    Uses ``src.utils.gradient_ops.average_gradients`` for arithmetic averaging.
    Applies update with learning_rate=1.0 (standard FedAvg).
    """
    all_grads_list = list(client_grads.values())

    # average_gradients expects list of gradient-lists
    avg_grads = average_gradients(all_grads_list)

    # Apply to global model: w_new = w - lr * avg_grad (lr=1.0 for FedAvg)
    with torch.no_grad():
        for param, grad in zip(global_model.parameters(), avg_grads):
            if isinstance(grad, torch.Tensor):
                param.data.sub_(grad.to(param.device))
            else:
                param.data.sub_(torch.tensor(grad, dtype=param.dtype, device=param.device))

    rmetrics.zk_mode = "NONE"
    rmetrics.flagged_client_ids = []
    logger.debug("  Vanilla round: averaged %d client gradient sets", len(client_grads))
    return rmetrics


def _run_fltrust_round(
    fltrust_agg: "FLTrustAggregator",
    global_model: nn.Module,
    client_grads: Dict[int, List[torch.Tensor]],
    rmetrics: RoundMetrics,
    device: str,
) -> RoundMetrics:
    """FLTrust round: trust-score-weighted aggregation.

    Uses the ``FLTrustAggregator`` which has already computed the server
    gradient via ``compute_server_update()`` earlier in the round.
    Client updates are scored by cosine similarity to the server reference
    update, then normalised and weighted-averaged.
    """
    # Build updates list in the format FLTrust expects
    updates = [
        {"client_id": cid, "gradients": grads}
        for cid, grads in client_grads.items()
    ]

    agg_result = fltrust_agg.aggregate(updates)

    if agg_result is not None:
        aggregated = agg_result["gradients"]
        trust_scores = agg_result.get("trust_scores", {})

        # Apply global update
        from src.utils.gradient_ops import unflatten_gradients
        param_shapes = [p.shape for p in global_model.parameters()]
        unflat = unflatten_gradients(aggregated, param_shapes)

        with torch.no_grad():
            for param, grad in zip(global_model.parameters(), unflat):
                if isinstance(grad, torch.Tensor):
                    param.data.sub_(grad.to(param.device))
                else:
                    param.data.sub_(torch.tensor(grad, dtype=param.dtype, device=param.device))

        # Flag clients with zero trust scores
        rmetrics.flagged_client_ids = [
            cid for cid, ts in trust_scores.items() if ts < 1e-6
        ]
    else:
        rmetrics.flagged_client_ids = []

    rmetrics.zk_mode = "NONE"
    logger.debug(
        "  FLTrust round: %d clients, flagged=%s",
        len(client_grads), rmetrics.flagged_client_ids
    )
    return rmetrics


def _run_protogalaxy_round(
    pipeline: ProtoGalaxyPipeline,
    global_model: nn.Module,
    client_trainers: Dict[int, Trainer],
    client_grads: Dict[int, List[torch.Tensor]],
    round_num: int,
    cfg: ExperimentConfig,
    rmetrics: RoundMetrics,
    byzantine_ids: Set[int],
) -> RoundMetrics:
    """Full ProtoGalaxy round with all 4 phases.

    Uses the actual pipeline methods: ``phase1_*``, ``phase2_*``,
    ``phase3_*``, ``phase4_*``.

    If ``cfg.ablation == 'merkle_only'``, ZKP prove/verify/fold steps are
    skipped.  If ``cfg.ablation == 'zk_merkle'``, all ZKP steps are included.
    """
    pipeline.current_round = round_num
    enable_zkp = cfg.ablation != "merkle_only"
    logger.debug(
        f"  ProtoGalaxy round {round_num}: zkp={'ON' if enable_zkp else 'OFF'}, "
        f"ablation={cfg.ablation or 'none'}"
    )

    # ==========================
    # PHASE 1: COMMITMENT
    # ==========================
    phase1_start = time.perf_counter()

    commitments_by_galaxy: Dict[int, Dict[int, str]] = {}
    client_metadata: Dict[int, Dict] = {}

    for cid, grads in client_grads.items():
        # Overwrite trainer gradients with (possibly poisoned) gradients
        commit_hash, metadata = pipeline.phase1_client_commitment(
            cid, grads, round_num,
        )
        # Move metadata tensors to CPU if they exist
        for k, v in metadata.items():
            if isinstance(v, torch.Tensor):
                metadata[k] = v.cpu()

        client_metadata[cid] = metadata
        galaxy_id = cid % cfg.num_galaxies
        commitments_by_galaxy.setdefault(galaxy_id, {})[cid] = commit_hash

    # Galaxy Merkle trees
    merkle_build_start = time.perf_counter()
    galaxy_roots = {}
    for galaxy_id, commits in commitments_by_galaxy.items():
        root = pipeline.phase1_galaxy_collect_commitments(galaxy_id, commits, round_num)
        galaxy_roots[galaxy_id] = root

    # Global Merkle tree
    global_root = pipeline.phase1_global_collect_galaxy_roots(galaxy_roots, round_num)
    rmetrics.merkle_build_time = time.perf_counter() - merkle_build_start

    # ZK proof generation
    if enable_zkp:
        zk_start = time.perf_counter()
        zk_metrics = pipeline.phase1_generate_zk_proofs(client_grads, round_num)
        rmetrics.zk_prove_time = time.perf_counter() - zk_start
        rmetrics.zk_proofs_generated = zk_metrics.get("proofs_generated", 0)
        rmetrics.zk_mode = zk_metrics.get("mode", "NONE")
    else:
        rmetrics.zk_mode = "DISABLED"
        
    # Free memory after ZKP generation (if any GPU tensors were created)
    # Free memory after ZKP generation (if any GPU tensors were created)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Client-side Merkle verification
    merkle_verify_start = time.perf_counter()
    for galaxy_id, commits in commitments_by_galaxy.items():
        adapter = pipeline.galaxy_merkle_trees[galaxy_id]
        gal_root = adapter.get_root()
        for cid, commit_hash in commits.items():
            proof = adapter.get_proof(cid)
            if gal_root and proof is not None:
                idx = adapter.client_ids.index(cid) if cid in adapter.client_ids else 0
                merkle_verify_proof(gal_root, proof, commit_hash, idx)
    rmetrics.merkle_verify_time = time.perf_counter() - merkle_verify_start

    rmetrics.phase1_time = time.perf_counter() - phase1_start
    logger.debug(
        f"  Phase 1 COMMIT: {rmetrics.phase1_time:.3f}s — "
        f"merkle_build={rmetrics.merkle_build_time:.4f}s, "
        f"merkle_verify={rmetrics.merkle_verify_time:.4f}s, "
        f"zk_prove={rmetrics.zk_prove_time:.3f}s ({rmetrics.zk_proofs_generated} proofs)"
    )

    # ==========================
    # PHASE 2: REVELATION
    # ==========================
    phase2_start = time.perf_counter()

    galaxy_verified: Dict[int, List[Dict]] = {}
    total_verified = 0

    for galaxy_id in range(cfg.num_galaxies):
        subs = {}
        for cid in commitments_by_galaxy.get(galaxy_id, {}):
            sub = pipeline.phase2_client_submit_gradients(
                cid, galaxy_id, client_grads[cid],
                commitments_by_galaxy[galaxy_id][cid],
                client_metadata[cid], round_num,
            )
            subs[cid] = sub
        verified, rejected = pipeline.phase2_galaxy_verify_and_collect(galaxy_id, subs)
        galaxy_verified[galaxy_id] = verified
        total_verified += len(verified)

    # ZK verification
    if enable_zkp:
        all_verified_ids = [
            u["client_id"] for updates in galaxy_verified.values() for u in updates
        ]
        zk_verify_start = time.perf_counter()
        zk_verify_metrics = pipeline.phase2_verify_zk_proofs(all_verified_ids)
        rmetrics.zk_verify_time = time.perf_counter() - zk_verify_start
        rmetrics.zk_proofs_verified = zk_verify_metrics.get("zk_verified", 0)
        rmetrics.zk_proofs_failed = zk_verify_metrics.get("zk_failed", 0)

        # Remove ZK-rejected clients
        if zk_verify_metrics.get("zk_rejected_ids"):
            zk_reject_set = set(zk_verify_metrics["zk_rejected_ids"])
            for gid in galaxy_verified:
                galaxy_verified[gid] = [
                    u for u in galaxy_verified[gid]
                    if u["client_id"] not in zk_reject_set
                ]

    rmetrics.phase2_time = time.perf_counter() - phase2_start
    logger.debug(
        f"  Phase 2 REVEAL: {rmetrics.phase2_time:.3f}s — "
        f"verified={total_verified}, zk_ok={rmetrics.zk_proofs_verified}, "
        f"zk_fail={rmetrics.zk_proofs_failed}"
    )

    # ==========================
    # PHASE 3: DEFENSE
    # ==========================
    phase3_start = time.perf_counter()

    galaxy_agg_grads = {}
    galaxy_defense_reports = {}
    round_flagged: Set[int] = set()

    for galaxy_id, verified_updates in galaxy_verified.items():
        if not verified_updates:
            continue
        agg_grads, report = pipeline.phase3_galaxy_defense_pipeline(galaxy_id, verified_updates)
        galaxy_agg_grads[galaxy_id] = agg_grads
        galaxy_defense_reports[galaxy_id] = report

        # Collect flagged client IDs
        for idx in report.get("flagged_clients", []):
            if idx < len(verified_updates):
                round_flagged.add(verified_updates[idx]["client_id"])
        for idx in report.get("statistical_flagged", []):
            if idx < len(verified_updates):
                round_flagged.add(verified_updates[idx]["client_id"])

    # Galaxy submissions
    galaxy_submissions = {}
    for galaxy_id in galaxy_agg_grads:
        client_ids = [u["client_id"] for u in galaxy_verified[galaxy_id]]
        galaxy_sub = pipeline.phase3_galaxy_submit_to_global(
            galaxy_id, galaxy_agg_grads[galaxy_id],
            galaxy_defense_reports[galaxy_id], client_ids,
        )
        galaxy_submissions[galaxy_id] = galaxy_sub

    rmetrics.phase3_time = time.perf_counter() - phase3_start
    logger.debug(
        f"  Phase 3 DEFENSE: {rmetrics.phase3_time:.3f}s — "
        f"{len(galaxy_agg_grads)} galaxies aggregated, "
        f"flagged={sorted(round_flagged)}"
    )

    # ==========================
    # PHASE 4: GLOBAL AGGREGATION
    # ==========================
    phase4_start = time.perf_counter()

    verified_galaxies, rejected_galaxies = pipeline.phase4_global_verify_galaxies(
        galaxy_submissions
    )

    layer5_result = pipeline.phase4_layer5_galaxy_defense(verified_galaxies)

    global_grads, global_defense_report = pipeline.phase4_global_defense_and_aggregate(
        verified_galaxies, layer5_result=layer5_result,
    )

    pipeline.phase4_update_global_model(global_grads)

    # ZK folding
    if enable_zkp:
        clean_galaxy_ids = [
            u["galaxy_id"] for u in verified_galaxies
            if u["galaxy_id"] not in set(
                global_defense_report.get("flagged_galaxies", [])
                + global_defense_report.get("layer5_flagged_galaxies", [])
            )
        ]
        zk_fold_start = time.perf_counter()
        zk_fold_metrics = pipeline.phase4_fold_galaxy_zk_proofs(clean_galaxy_ids)
        rmetrics.zk_fold_time = time.perf_counter() - zk_fold_start

    pipeline.phase4_distribute_model()

    rmetrics.phase4_time = time.perf_counter() - phase4_start
    logger.debug(
        f"  Phase 4 GLOBAL: {rmetrics.phase4_time:.3f}s — "
        f"verified_galaxies={len(verified_galaxies)}, "
        f"rejected_galaxies={len(rejected_galaxies)}, "
        f"zk_fold={rmetrics.zk_fold_time:.3f}s"
    )

    # Collect reputation data for quarantine/ban
    rmetrics.flagged_client_ids = sorted(round_flagged)
    quarantined = []
    banned = []
    for gid, coord in pipeline.galaxy_defense_coordinators.items():
        quarantined.extend(coord.layer4.get_quarantined_clients())
        banned.extend(coord.layer4.get_banned_clients())
    rmetrics.quarantined_client_ids = sorted(set(quarantined))
    rmetrics.banned_client_ids = sorted(set(banned))

    return rmetrics


# ============================================================================
# Experiment matrix generators
# ============================================================================

def _gen_baseline_configs(
    trials: int, base_seed: int, num_rounds: int,
) -> List[ExperimentConfig]:
    """Baseline: vanilla FL, no attacks, across partitions."""
    configs = []
    for partition in SUPPORTED_PARTITIONS:
        for trial in range(trials):
            configs.append(ExperimentConfig(
                mode="baseline",
                trial_id=trial,
                seed=base_seed + trial,
                partition=partition,
                defense="vanilla",
                attack="none",
                byzantine_fraction=0.0,
                num_rounds=num_rounds,
            ))
    return configs


def _gen_attack_configs(
    trials: int, base_seed: int, num_rounds: int,
) -> List[ExperimentConfig]:
    """All defense × attack × byzantine fraction combinations."""
    defenses = ["vanilla", "multi_krum", "protogalaxy_full"]
    # Exclude not-yet-implemented attacks and fltrust
    attacks = ["label_flip", "targeted_label_flip", "model_poisoning",
               "backdoor", "gaussian_noise"]
    byz_fractions = [0.1, 0.2, 0.3]

    configs = []
    for defense, attack, byz_frac in itertools.product(defenses, attacks, byz_fractions):
        for trial in range(trials):
            agg = "trimmed_mean"
            if defense == "multi_krum":
                agg = "multi_krum"
            configs.append(ExperimentConfig(
                mode="attacks",
                trial_id=trial,
                seed=base_seed + trial,
                defense=defense,
                attack=attack,
                aggregation_method=agg,
                byzantine_fraction=byz_frac,
                num_rounds=num_rounds,
            ))
    return configs


def _gen_ablation_configs(
    trials: int, base_seed: int, num_rounds: int,
) -> List[ExperimentConfig]:
    """Ablation: merkle_only vs zk_merkle under attack."""
    attacks = ["label_flip", "model_poisoning", "backdoor"]
    byz_fractions = [0.2, 0.3]

    configs = []
    for ablation in ABLATION_MODES:
        for attack, byz_frac in itertools.product(attacks, byz_fractions):
            for trial in range(trials):
                configs.append(ExperimentConfig(
                    mode="ablation",
                    trial_id=trial,
                    seed=base_seed + trial,
                    defense="protogalaxy_full",
                    attack=attack,
                    byzantine_fraction=byz_frac,
                    ablation=ablation,
                    num_rounds=num_rounds,
                ))
    return configs


def _gen_scalability_configs(
    trials: int, base_seed: int, num_rounds: int,
    dataset: str = "mnist", model_type: str = "linear",
) -> List[ExperimentConfig]:
    """Scalability: vary clients × galaxies across defenses and attacks."""
    scale_points = [
        {"num_clients": 10, "num_galaxies": 2},
        {"num_clients": 20, "num_galaxies": 4},
        {"num_clients": 50, "num_galaxies": 5},
        {"num_clients": 100, "num_galaxies": 10},
        {"num_clients": 200, "num_galaxies": 10},
    ]
    defenses = ["vanilla", "multi_krum", "protogalaxy_full"]
    attacks_byz = [
        ("none", 0.0),
        ("label_flip", 0.2),
        ("model_poisoning", 0.2),
    ]

    configs = []
    for sp in scale_points:
        for defense in defenses:
            for attack, byz_frac in attacks_byz:
                for trial in range(trials):
                    agg = "trimmed_mean"
                    if defense == "multi_krum":
                        agg = "multi_krum"
                    configs.append(ExperimentConfig(
                        mode="scalability",
                        trial_id=trial,
                        seed=base_seed + trial,
                        num_clients=sp["num_clients"],
                        num_galaxies=sp["num_galaxies"],
                        defense=defense,
                        attack=attack,
                        aggregation_method=agg,
                        byzantine_fraction=byz_frac,
                        num_rounds=min(num_rounds, 15),  # shorter for large scale
                        dataset=dataset,
                        model_type=model_type,
                    ))
    return configs


def generate_experiment_matrix(
    mode: str,
    trials: int = 3,
    base_seed: int = 42,
    num_rounds: int = 20,
    dataset: str = "mnist",
    model_type: str = "linear",
    custom_cfg: Optional[ExperimentConfig] = None,
) -> List[ExperimentConfig]:
    """Generate the experiment matrix for a given mode.

    Parameters
    ----------
    mode : str
        One of 'baseline', 'attacks', 'ablation', 'scalability',
        'zkp_performance', 'attack_rejection', 'full', 'custom'.
    trials : int
        Number of independent trials per configuration.
    base_seed : int
        Base random seed (trial_i uses base_seed + i).
    num_rounds : int
        FL rounds per experiment.
    custom_cfg : ExperimentConfig, optional
        If mode='custom', this config is used directly.
    """
    if mode == "baseline":
        return _gen_baseline_configs(trials, base_seed, num_rounds)
    if mode == "attacks":
        return _gen_attack_configs(trials, base_seed, num_rounds)
    if mode == "ablation":
        return _gen_ablation_configs(trials, base_seed, num_rounds)
    if mode == "scalability":
        return _gen_scalability_configs(trials, base_seed, num_rounds, dataset, model_type)
    if mode == "zkp_performance":
        # ZKP performance mode uses specialized evaluator, return empty list
        return []
    if mode == "attack_rejection":
        # Attack rejection mode uses specialized evaluator, return empty list
        return []
    if mode == "full":
        return (
            _gen_baseline_configs(trials, base_seed, num_rounds)
            + _gen_attack_configs(trials, base_seed, num_rounds)
            + _gen_ablation_configs(trials, base_seed, num_rounds)
            + _gen_scalability_configs(trials, base_seed, num_rounds, dataset, model_type)
        )
    if mode == "custom":
        if custom_cfg is None:
            raise ValueError("custom mode requires --defense, --attack, etc.")
        return [custom_cfg]
    raise ValueError(f"Unknown mode: {mode}")


# ============================================================================
# Result I/O
# ============================================================================

def _results_dir(output_dir: str, mode: str) -> Path:
    p = Path(output_dir) / mode
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_result(result: ExperimentResult, output_dir: str):
    rdir = _results_dir(output_dir, result.config.mode)
    filename = f"{result.config.experiment_id}.json"
    filepath = rdir / filename
    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    logger.info(f"  Saved → {filepath}")


def _is_completed(cfg: ExperimentConfig, output_dir: str) -> bool:
    rdir = _results_dir(output_dir, cfg.mode)
    filepath = rdir / f"{cfg.experiment_id}.json"
    return filepath.exists()


# ============================================================================
# Summary / reporting
# ============================================================================

def run_zkp_performance_evaluation(trials: int, output_dir: str) -> bool:
    """Run ZKP performance benchmarking.
    
    Returns True if successful, False otherwise.
    """
    if not ZKP_PERF_AVAILABLE:
        logger.error("ZKP Performance Evaluator not available")
        return False
    
    try:
        from fl_zkp_bridge import FLZKPBoundedProver
        logger.info("✓ Rust ZKP module loaded\n")
    except ImportError:
        logger.error("✗ Rust ZKP module not available")
        logger.error("  Build with: cd sonobe/fl-zkp-bridge && maturin develop --release")
        return False
    
    print("\n" + "="*70)
    print("  ZKP PERFORMANCE BENCHMARKING")
    print("="*70)
    print(f"  Trials per configuration: {trials}")
    print(f"  Output directory: {output_dir}")
    print("="*70 + "\n")
    
    evaluator = ZKPPerformanceEvaluator(num_trials=trials)
    evaluator.run_full_evaluation()
    evaluator.print_summary_table()
    
    # Save results
    output_path = Path(output_dir) / "zkp_performance"
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "zkp_performance_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(evaluator.results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_file}")
    return True


def run_attack_rejection_evaluation(trials: int, output_dir: str) -> bool:
    """Run Byzantine attack rejection evaluation.
    
    Returns True if successful, False otherwise.
    """
    if not ATTACK_REJECTION_AVAILABLE:
        logger.error("Attack Rejection Evaluator not available")
        return False
    
    try:
        from fl_zkp_bridge import FLZKPBoundedProver
        logger.info("✓ Rust ZKP module loaded\n")
    except ImportError:
        logger.error("✗ Rust ZKP module not available")
        logger.error("  Build with: cd sonobe/fl-zkp-bridge && maturin develop --release")
        return False
    
    print("\n" + "="*70)
    print("  BYZANTINE ATTACK REJECTION EVALUATION")
    print("="*70)
    print(f"  Trials per attack: {trials}")
    print(f"  Output directory: {output_dir}")
    print("="*70 + "\n")
    
    evaluator = ByzantineAttackEvaluator(num_trials=trials)
    evaluator.run_attack_suite()
    evaluator.print_summary_table()
    
    # Save results
    output_path = Path(output_dir) / "attack_rejection"
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "byzantine_attack_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(evaluator.results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_file}")
    return True


def print_summary(results: List[ExperimentResult]):
    """Print a tabular summary of all experiment results."""
    if not results:
        print("No results to summarize.")
        return

    header = (
        f"{'Defense':<20} {'Attack':<22} {'Byz%':>4} {'Partition':<10} "
        f"{'Clients':>7} {'Galaxies':>8} {'Acc':>6} {'TPR':>5} {'FPR':>5} "
        f"{'F1':>5} {'Conv85':>6} {'Time':>7} {'ZK Mode':<10}"
    )
    print("\n" + "=" * len(header))
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        c = r.config
        print(
            f"{c.defense:<20} {c.attack:<22} {c.byzantine_fraction*100:>3.0f}% "
            f"{c.partition:<10} {c.num_clients:>7} {c.num_galaxies:>8} "
            f"{r.final_accuracy:>5.3f} {r.avg_tpr:>5.3f} {r.avg_fpr:>5.3f} "
            f"{r.avg_f1:>5.3f} {r.convergence_round_85:>6} "
            f"{r.total_time:>6.1f}s "
            f"{r.rounds[-1].zk_mode if r.rounds else 'N/A':<10}"
        )

    print("=" * len(header))
    print(f"Total experiments: {len(results)}")


def save_resource_report(results: List[ExperimentResult], output_dir: str):
    """Save detailed resource usage report for all experiments.
    
    Writes a human-readable text file and a JSON summary with:
    - Per-experiment wall-clock time, CPU, RAM, GPU usage
    - Aggregate statistics across all experiments
    - Suitable for inclusion in paper methodology section
    """
    report_path = Path(output_dir) / "resource_usage_report.txt"
    json_path = Path(output_dir) / "resource_usage_report.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    lines = [
        "=" * 90,
        "  SYSTEM RESOURCE USAGE REPORT",
        "=" * 90,
        f"  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Device:     {device_name}",
        f"  CPU Cores:  {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical",
        f"  Total RAM:  {psutil.virtual_memory().total / (1024**3):.1f} GB",
    ]
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        lines.append(f"  GPU VRAM:   {gpu_mem:.1f} GB")
    lines.append("=" * 90)
    lines.append("")

    # Per-experiment header
    lines.append(
        f"{'Experiment':<55} {'Time(s)':>8} {'CPU%':>7} {'RAM(MB)':>9} "
        f"{'GPU%':>7} {'VRAM(MB)':>9}"
    )
    lines.append("-" * 100)

    json_records = []
    all_times = []
    all_cpus = []
    all_rams = []
    all_gpus = []
    all_vrams = []

    for r in results:
        ru = r.resource_usage
        wall = r.total_time
        cpu_avg = ru.get('cpu_percent', {}).get('mean', 0)
        ram_peak = ru.get('ram_mb', {}).get('max', 0)
        gpu_avg = ru.get('gpu_util_percent', {}).get('mean', 0)
        vram_peak = ru.get('gpu_mem_allocated_mb', {}).get('max', 0)

        all_times.append(wall)
        all_cpus.append(cpu_avg)
        all_rams.append(ram_peak)
        all_gpus.append(gpu_avg)
        all_vrams.append(vram_peak)

        eid = r.config.experiment_id
        lines.append(
            f"{eid:<55} {wall:>7.1f}s {cpu_avg:>6.1f}% {ram_peak:>8.0f} "
            f"{gpu_avg:>6.1f}% {vram_peak:>8.0f}"
        )

        json_records.append({
            'experiment_id': eid,
            'defense': r.config.defense,
            'attack': r.config.attack,
            'num_clients': r.config.num_clients,
            'final_accuracy': r.final_accuracy,
            'wall_time_s': round(wall, 2),
            'cpu_percent_avg': round(cpu_avg, 1),
            'ram_peak_mb': round(ram_peak, 1),
            'gpu_util_percent_avg': round(gpu_avg, 1),
            'vram_peak_mb': round(vram_peak, 1),
            'resource_detail': ru,
        })

    lines.append("-" * 100)

    # Aggregate
    if all_times:
        lines.append("")
        lines.append("AGGREGATE STATISTICS")
        lines.append(f"  Total wall-clock time:    {sum(all_times):.1f}s ({sum(all_times)/60:.1f} min)")
        lines.append(f"  Avg wall-clock per exp:   {np.mean(all_times):.1f}s")
        lines.append(f"  CPU usage (avg/max):      {np.mean(all_cpus):.1f}% / {max(all_cpus):.1f}%")
        lines.append(f"  RAM peak (avg/max):       {np.mean(all_rams):.0f} MB / {max(all_rams):.0f} MB")
        if any(g > 0 for g in all_gpus):
            lines.append(f"  GPU util (avg/max):       {np.mean(all_gpus):.1f}% / {max(all_gpus):.1f}%")
            lines.append(f"  VRAM peak (avg/max):      {np.mean(all_vrams):.0f} MB / {max(all_vrams):.0f} MB")

    lines.append("")
    lines.append("=" * 90)

    report_text = "\n".join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)

    json_report = {
        'system': {
            'device': device_name,
            'cpu_cores_physical': psutil.cpu_count(logical=False),
            'cpu_cores_logical': psutil.cpu_count(logical=True),
            'total_ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'gpu_vram_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if torch.cuda.is_available() else 0,
        },
        'aggregate': {
            'total_time_s': round(sum(all_times), 1) if all_times else 0,
            'avg_time_s': round(float(np.mean(all_times)), 1) if all_times else 0,
            'avg_cpu_percent': round(float(np.mean(all_cpus)), 1) if all_cpus else 0,
            'max_ram_mb': round(float(max(all_rams)), 1) if all_rams else 0,
            'avg_gpu_percent': round(float(np.mean(all_gpus)), 1) if all_gpus else 0,
            'max_vram_mb': round(float(max(all_vrams)), 1) if all_vrams else 0,
        },
        'experiments': json_records,
    }
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)

    print(f"\n  Resource report saved to:")
    print(f"    {report_path}")
    print(f"    {json_path}")
    return report_text


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ProtoGalaxy Comprehensive Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", type=str, default="baseline",
        choices=list(EVAL_MODES),
        help="Evaluation mode (default: baseline)",
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per config")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Output directory for JSON results")
    parser.add_argument("--num-rounds", type=int, default=20, help="FL rounds per experiment")
    parser.add_argument("--dry-run", action="store_true", help="Print matrix, don't run")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments whose result file already exists")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")

    # Custom mode args
    parser.add_argument("--defense", type=str, default="protogalaxy_full",
                        choices=list(SUPPORTED_DEFENSES))
    parser.add_argument("--attack", type=str, default="none",
                        choices=list(SUPPORTED_ATTACKS))
    parser.add_argument("--partition", type=str, default="iid",
                        choices=list(SUPPORTED_PARTITIONS))
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-galaxies", type=int, default=4)
    parser.add_argument("--byzantine-fraction", type=float, default=0.0)
    parser.add_argument("--aggregation-method", type=str, default="trimmed_mean",
                        choices=["trimmed_mean", "multi_krum", "coordinate_wise_median"])
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--model-type", type=str, default="linear",
                        choices=["linear", "mlp", "cnn", "cifar10_cnn", "resnet18"])
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--ablation", type=str, default="",
                        choices=["", "merkle_only", "zk_merkle"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)

    args = parser.parse_args()

    # Logging — dual output: console + file
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Console handler — concise
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    root_logger.addHandler(console_handler)
    # File handler — detailed with timestamps
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG to file
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root_logger.addHandler(file_handler)
    logger.info(f"Log file: {log_file}")

    # Build experiment matrix
    custom_cfg = None
    if args.mode == "custom":
        custom_cfg = ExperimentConfig(
            mode="custom",
            defense=args.defense,
            attack=args.attack,
            partition=args.partition,
            num_clients=args.num_clients,
            num_galaxies=args.num_galaxies,
            byzantine_fraction=args.byzantine_fraction,
            aggregation_method=args.aggregation_method,
            trim_ratio=args.trim_ratio,
            dataset=args.dataset,
            model_type=args.model_type,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_rounds=args.num_rounds,
            ablation=args.ablation,
            dirichlet_alpha=args.dirichlet_alpha,
        )

    configs = generate_experiment_matrix(
        mode=args.mode,
        trials=args.trials,
        base_seed=args.base_seed,
        num_rounds=args.num_rounds,
        dataset=args.dataset,
        model_type=args.model_type,
        custom_cfg=custom_cfg,
    )

    # Handle specialized evaluation modes
    if args.mode == "zkp_performance":
        success = run_zkp_performance_evaluation(args.trials, args.output_dir)
        sys.exit(0 if success else 1)
    
    if args.mode == "attack_rejection":
        success = run_attack_rejection_evaluation(args.trials, args.output_dir)
        sys.exit(0 if success else 1)
    
    print("\n" + "=" * 70)
    print("  ProtoGalaxy Evaluation")
    print("=" * 70)
    print(f"  Mode:        {args.mode}")
    print(f"  Experiments:  {len(configs)}")
    print(f"  Trials/cfg:  {args.trials}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Device:      {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")
    print(f"  CPU Cores:   {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical")
    print(f"  Total RAM:   {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU VRAM:    {gpu_mem:.1f} GB")
    print("=" * 70)

    if args.dry_run:
        print("\n  DRY RUN — experiment matrix:\n")
        for i, c in enumerate(configs):
            print(
                f"  {i+1:>4}. {c.defense:<20s} {c.attack:<22s} "
                f"byz={c.byzantine_fraction:.0%} {c.partition:<10s} "
                f"c={c.num_clients} g={c.num_galaxies} "
                f"abl={c.ablation or '-':<12s} trial={c.trial_id}"
            )
        print(f"\n  Total: {len(configs)} experiments")
        return

    # Run experiments
    all_results: List[ExperimentResult] = []
    failed: List[Tuple[str, str]] = []

    for i, cfg in enumerate(configs):
        print(f"\n{'─' * 70}")
        print(f"  Experiment {i+1}/{len(configs)}: {cfg.experiment_id}")
        print(f"{'─' * 70}")

        if args.resume and _is_completed(cfg, args.output_dir):
            logger.info(f"  ✓ Already completed — skipping")
            continue

        try:
            result = run_single_experiment(cfg)
            _save_result(result, args.output_dir)
            all_results.append(result)
        except NotImplementedError as e:
            logger.warning(f"  ⚠ Skipped (not implemented): {e}")
            failed.append((cfg.experiment_id, str(e)))
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            logger.debug(traceback.format_exc())
            failed.append((cfg.experiment_id, str(e)))

    # Summary
    print_summary(all_results)

    # Save resource usage report
    if all_results:
        save_resource_report(all_results, args.output_dir)

    if failed:
        print(f"\n  ⚠ {len(failed)} experiments failed or skipped:")
        for eid, reason in failed:
            print(f"    - {eid}: {reason}")

    print(f"\n  Results saved to: {args.output_dir}/")
    print(f"  See eval_tbd.md for components not yet implemented.")


if __name__ == "__main__":
    main()
