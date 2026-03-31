#!/usr/bin/env python3
"""
ZK Proof-of-Gradient Research Experiments
==========================================
Evaluates the model-agnostic ZK proof-of-gradient system across:
  - Model architectures: linear, mlp, cnn
  - Byzantine fractions: 20%, 30%, 50%
  - Client counts: 10, 20
  - Attack types: scale_10x, sign_flip, gaussian_noise, random_update
  - FL rounds: 5 per experiment

For each experiment, clients train locally on real MNIST data, generate
ZK proof bundles from actual computed gradients, and the server accepts
only clients whose proofs are valid.

Key distinction from baseline:
  - ZK proof is generated on actual gradient deltas (not claimed values)
  - Server calls verify_gradient_proof() which calls PG::verify (real ZKP)
  - Detection = no valid proof OR statistical norm outlier
  - All training metrics (loss, accuracy) are from real training

Usage:
  python scripts/run_zkp_research.py --quick         # 4 experiments (one per attack)
  python scripts/run_zkp_research.py --output outputs/zkp_run
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from src.models.mnist import create_mnist_model
from src.crypto.zkp_prover import (
    ModelAgnosticProver, FingerprintHelper, verify_gradient_proof,
    GradientProofBundle, _ZKP_AVAILABLE, EMABoundManager,
    cosine_similarity_filter
)
from src.defense.robust_agg import MultiKrumAggregator, CoordinateWiseMedianAggregator

# ── Logging ──────────────────────────────────────────────────────────────────

def make_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("ZKPResearch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


log: logging.Logger = None  # set in main


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_mnist(num_clients: int, data_dir: str = "./data"):
    """Load and IID-partition MNIST across clients."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_ds = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_ds  = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    per_client = len(train_ds) // num_clients
    indices = list(range(len(train_ds)))
    np.random.shuffle(indices)
    client_datasets = [
        Subset(train_ds, indices[i * per_client:(i + 1) * per_client])
        for i in range(num_clients)
    ]
    return client_datasets, test_ds


# ── Training helpers ──────────────────────────────────────────────────────────

def train_local(model: nn.Module, dataset, epochs: int, lr: float = 0.01) -> Dict:
    """Real local training. Returns {'loss': float, 'accuracy': float}."""
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return {"loss": total_loss / total, "accuracy": correct / total}


def evaluate(model: nn.Module, dataset) -> Dict:
    """Evaluate model on dataset. Returns {'loss', 'accuracy'}."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return {"loss": total_loss / total, "accuracy": correct / total}


def get_gradient_deltas(local_model: nn.Module, global_model: nn.Module) -> List[torch.Tensor]:
    """Compute gradient update deltas = local_params - global_params."""
    deltas = []
    for lp, gp in zip(local_model.parameters(), global_model.parameters()):
        deltas.append((lp.detach() - gp.detach()).clone())
    return deltas


def apply_attack(deltas: List[torch.Tensor], attack: str) -> List[torch.Tensor]:
    """Apply Byzantine attack to gradient deltas."""
    if attack == "scale_10x":
        return [d * 10.0 for d in deltas]
    elif attack == "sign_flip":
        return [-d for d in deltas]
    elif attack == "gaussian_noise":
        return [d + torch.randn_like(d) * d.std() * 5.0 for d in deltas]
    elif attack == "random_update":
        return [torch.randn_like(d) * d.abs().mean() * 10.0 for d in deltas]
    return deltas


def federated_average(accepted_updates: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Average gradient deltas from accepted clients."""
    if not accepted_updates:
        return []
    avg = [torch.stack([u[i] for u in accepted_updates]).mean(0)
           for i in range(len(accepted_updates[0]))]
    return avg


# ── Statistical norm filter ───────────────────────────────────────────────────

def compute_norm_bounds(all_norms: List[float], scale: float = 5.0) -> float:
    """Legacy bound fallback. Replaced by EMABoundManager in architecture."""
    arr = np.array(all_norms)
    q25 = float(np.quantile(arr, 0.25))
    return q25 + scale * max(abs(q25), 1e-6)


# ── Single Round ──────────────────────────────────────────────────────────────

def run_round(
    global_model: nn.Module,
    client_datasets: List,
    byzantine_ids: List[int],
    attack: str,
    round_num: int,
    local_epochs: int,
    prover: ModelAgnosticProver,
    bound_manager: EMABoundManager,
    model_fp: int,
    ref_gradients: List[torch.Tensor],
    ref_grad_fp: int,
    test_ds,
    model_type: str = "cnn",
    aggregator_name: str = "fedavg",
) -> Dict:
    """Run one FL round. Returns per-round metrics."""
    client_results = []
    proof_times = []
    global_params = list(global_model.parameters())

    # ── Phase 1: each client trains locally and generates a ZK proof ──────────
    for cid, dataset in enumerate(client_datasets):
        is_byz = cid in byzantine_ids

        # Real local training
        local_model = create_mnist_model(model_type, num_classes=10)
        local_model.load_state_dict(global_model.state_dict())

        t0 = time.time()
        train_metrics = train_local(local_model, dataset, local_epochs)
        train_time_ms = (time.time() - t0) * 1000

        # Compute actual gradient deltas
        deltas = get_gradient_deltas(local_model, global_model)

        # Apply attack to Byzantine clients
        if is_byz:
            deltas = apply_attack(deltas, attack)

        # Compute gradient norm (for statistical check)
        flat_grad = torch.cat([d.flatten() for d in deltas])
        grad_norm = float(flat_grad.norm())

        # Generate ZK proof on actual (possibly poisoned) gradient
        proof_t0 = time.time()
        proof: GradientProofBundle = prover.generate_proof(
            gradients=deltas,
            ref_gradients=ref_gradients,
            model_fp=model_fp,
            ref_grad_fp=ref_grad_fp,
            client_id=cid,
            round_number=round_num,
        )
        proof_time_ms = (time.time() - proof_t0) * 1000
        proof_times.append(proof_time_ms)

        client_results.append({
            "cid": cid,
            "is_byz": is_byz,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_time_ms": train_time_ms,
            "deltas": deltas,
            "proof": proof,
            "proof_time_ms": proof_time_ms,
            "grad_norm": grad_norm,
        })

    # ── Phase 2: server verifies ZK proofs ───────────────────────────────────
    all_norms = [r["grad_norm"] for r in client_results]
    
    # EMABoundManager threshold check
    bound_manager.update(all_norms)

    accepted_deltas = []
    accepted_ids = []
    rejected_ids = []
    zk_pass = 0
    zk_fail = 0

    for r in client_results:
        cid = r["cid"]
        proof = r["proof"]

        # ZK proof verification (calls PG::verify — real check)
        valid, reason = verify_gradient_proof(proof, expected_model_fp=model_fp, ema_bound=bound_manager)

        # Statistical norm check (handled by verify_gradient_proof now, but fallback if not used)
        norm_ok = bound_manager.check(r["grad_norm"])

        if valid and norm_ok:
            accepted_deltas.append(r["deltas"])
            accepted_ids.append(cid)
            zk_pass += 1
        else:
            rejected_ids.append(cid)
            zk_fail += 1
            log.info(
                f"    Client {cid} {'[BYZ]' if r['is_byz'] else '[honest]'}: "
                f"REJECTED — zkp={valid}({reason}), norm_ok={norm_ok} "
                f"(norm={r['grad_norm']:.3f})"
            )

    # ── Phase 3: aggregate accepted updates ───────────────────────────────────
    if accepted_deltas:
        if aggregator_name == "multi_krum":
            # Cosine Sim Filter -> Multi-Krum
            accepted_dicts = [{"client_id": cid, "gradients": d} for cid, d in zip(accepted_ids, accepted_deltas)]
            filtered = cosine_similarity_filter(accepted_dicts, threshold=0.1)
            
            if filtered:
                filtered_deltas = [f["gradients"] for f in filtered]
                agg = MultiKrumAggregator(f=max(1, int(len(filtered) * 0.3)))
                agg_dicts = [{"client_id": i, "gradients": d} for i, d in enumerate(filtered_deltas)]
                avg_raw = agg.aggregate(agg_dicts)["gradients"]
                
                avg_deltas = []
                offset = 0
                for shape_t in filtered_deltas[0]:
                    numel = shape_t.numel()
                    avg_deltas.append(torch.tensor(avg_raw[offset:offset+numel].reshape(shape_t.shape), dtype=torch.float32))
                    offset += numel
            else:
                avg_deltas = []
        elif aggregator_name == "coord_median":
            agg = CoordinateWiseMedianAggregator()
            agg_dicts = [{"client_id": i, "gradients": d} for i, d in enumerate(accepted_deltas)]
            avg_raw = agg.aggregate(agg_dicts)["gradients"]
            
            avg_deltas = []
            offset = 0
            for shape_t in accepted_deltas[0]:
                numel = shape_t.numel()
                avg_deltas.append(torch.tensor(avg_raw[offset:offset+numel].reshape(shape_t.shape), dtype=torch.float32))
                offset += numel
        else:
            avg_deltas = federated_average(accepted_deltas)
            
        if avg_deltas:
            lr_server = 1.0  # simple server-side learning rate
            with torch.no_grad():
                for param, delta in zip(global_model.parameters(), avg_deltas):
                    param.data += lr_server * delta

    # ── Metrics ───────────────────────────────────────────────────────────────
    test_metrics = evaluate(global_model, test_ds)

    true_positives  = len([r for r in client_results if r["is_byz"] and r["cid"] in rejected_ids])
    false_positives = len([r for r in client_results if not r["is_byz"] and r["cid"] in rejected_ids])
    false_negatives = len([r for r in client_results if r["is_byz"] and r["cid"] not in rejected_ids])

    return {
        "round": round_num,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "accepted": len(accepted_deltas),
        "rejected": len(rejected_ids),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "avg_proof_time_ms": np.mean(proof_times) if proof_times else 0.0,
        "avg_proof_size_bytes": int(np.mean([r["proof"].proof_size for r in client_results])),
        "zkp_real": _ZKP_AVAILABLE,
        "avg_deltas": avg_deltas if accepted_deltas else [],
    }


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(
    model_type: str,
    num_clients: int,
    byz_fraction: float,
    attack: str,
    num_rounds: int,
    local_epochs: int,
    seed: int,
    aggregator: str,
    data_dir: str,
    output_dir: Path,
    grad_sample_size: int = 1000,
) -> Dict:
    exp_id = f"{model_type}_c{num_clients}_byz{int(byz_fraction*100)}__{attack}__{aggregator}_r{num_rounds}"
    log.info("=" * 90)
    log.info(f"EXPERIMENT: {exp_id}")
    log.info("=" * 90)
    log.info(f"  Model: {model_type}  |  Clients: {num_clients}  |  "
             f"Byzantine: {byz_fraction*100:.0f}%  |  Attack: {attack}  |  Agg: {aggregator}  |  Rounds: {num_rounds}")
    log.info(f"  Local epochs: {local_epochs}  |  Seed: {seed}  |  ZKP real: {_ZKP_AVAILABLE}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    t_exp_start = time.time()
    client_datasets, test_ds = load_mnist(num_clients, data_dir)

    num_byzantine = max(1, int(num_clients * byz_fraction))
    byzantine_ids = list(range(num_byzantine))
    log.info(f"  Byzantine clients: {byzantine_ids}")

    # Initialize global model
    global_model = create_mnist_model(model_type, num_classes=10)

    # Compute model fingerprint from initial weights (round 0)
    model_fp, _ = FingerprintHelper.compute_model_fingerprint(
        list(global_model.parameters()), round_number=0
    )
    log.info(f"  Initial model fingerprint: {model_fp}")

    # Initialize reference gradients
    ref_gradients = [torch.zeros_like(p) for p in global_model.parameters()]

    # Initialize ZK prover (model-agnostic, shared across all clients per round)
    prover = ModelAgnosticProver(grad_sample_size=grad_sample_size)
    bound_manager = EMABoundManager()
    log.info(f"  ZK prover ready. grad_sample_size={grad_sample_size}, is_real={prover.is_real}")

    round_results = []
    for rnd in range(1, num_rounds + 1):
        log.info(f"\n  --- Round {rnd}/{num_rounds} ---")
        # Update fingerprints each round
        if rnd > 1:
            model_fp, _ = FingerprintHelper.compute_model_fingerprint(
                list(global_model.parameters()), round_number=rnd
            )
        ref_grad_fp, _ = FingerprintHelper.compute_model_fingerprint(
            ref_gradients, round_number=rnd
        )

        result = run_round(
            global_model=global_model,
            client_datasets=client_datasets,
            byzantine_ids=byzantine_ids,
            attack=attack,
            round_num=rnd,
            local_epochs=local_epochs,
            prover=prover,
            bound_manager=bound_manager,
            model_fp=model_fp,
            ref_gradients=ref_gradients,
            ref_grad_fp=ref_grad_fp,
            test_ds=test_ds,
            model_type=model_type,
            aggregator_name=aggregator,
        )
        round_results.append(result)
        if result["avg_deltas"]:
            ref_gradients = [d.clone().detach() for d in result["avg_deltas"]]
            
        log.info(
            f"  Round {rnd}: test_acc={result['test_accuracy']:.4f}  "
            f"test_loss={result['test_loss']:.4f}  "
            f"accepted={result['accepted']}/{num_clients}  "
            f"TP={result['true_positives']} FP={result['false_positives']} "
            f"FN={result['false_negatives']}  "
            f"avg_proof={result['avg_proof_time_ms']:.0f}ms"
        )

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_tp = sum(r["true_positives"] for r in round_results)
    total_fp = sum(r["false_positives"] for r in round_results)
    total_fn = sum(r["false_negatives"] for r in round_results)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    final_acc = round_results[-1]["test_accuracy"]
    avg_proof_ms = np.mean([r["avg_proof_time_ms"] for r in round_results])
    avg_proof_bytes = int(np.mean([r["avg_proof_size_bytes"] for r in round_results]))

    exp_time = time.time() - t_exp_start
    log.info(f"\n  EXPERIMENT COMPLETE ({exp_time:.1f}s)")
    log.info(f"  Detection: precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
    log.info(f"  Final test accuracy: {final_acc:.4f}")
    log.info(f"  Avg proof time: {avg_proof_ms:.0f}ms  |  Avg proof size: {avg_proof_bytes}B")
    log.info(f"  ZKP mode: {'REAL ProtoGalaxy IVC' if _ZKP_AVAILABLE else 'SHA-256 FALLBACK'}")

    result = {
        "experiment_id": exp_id,
        "config": {
            "model_type": model_type,
            "num_clients": num_clients,
            "byz_fraction": byz_fraction,
            "attack": attack,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "seed": seed,
            "grad_sample_size": grad_sample_size,
            "aggregator": aggregator,
        },
        "byzantine_ids": byzantine_ids,
        "zkp_mode": "real_protogalaxy" if _ZKP_AVAILABLE else "sha256_fallback",
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "final_test_accuracy": final_acc,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "avg_proof_time_ms": avg_proof_ms,
            "avg_proof_size_bytes": avg_proof_bytes,
        },
        "round_results": round_results,
        "total_time_s": exp_time,
    }

    result_file = output_dir / f"{exp_id}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Experiment matrix ─────────────────────────────────────────────────────────

def build_configs(quick: bool, fixed_aggregator: str = None) -> List[Tuple]:
    """Build experiment configuration matrix."""
    attacks = ["scale_10x", "sign_flip", "gaussian_noise", "random_update"]
    models = ["cnn"] # Focusing on LeNet-5 equivalent CNN
    aggregators = [fixed_aggregator] if fixed_aggregator else ["fedavg", "multi_krum", "coord_median"]

    if quick:
        # One attack per aggregator, smallest config (sanity check)
        return [
            ("cnn", 10, 0.3, "scale_10x",   3, 2, 42, aggregators[0]),
            ("cnn", 10, 0.3, "sign_flip",    3, 2, 43, aggregators[1 % len(aggregators)]),
            ("cnn", 10, 0.3, "gaussian_noise", 3, 2, 44, aggregators[2 % len(aggregators)]),
        ]

    # Full matrix:
    # 1 model × 2 client counts × 3 byz fractions × 4 attacks × 3 aggregators = 72 experiments
    configs = []
    base_seed = 100
    for model in models:
        for n_clients in [10, 20]:
            for byz_frac in [0.2, 0.3, 0.5]:
                for attack in attacks:
                    for agg in aggregators:
                        seed = base_seed
                        base_seed += 1
                        configs.append((model, n_clients, byz_frac, attack, 5, 2, seed, agg))
    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZK Proof-of-Gradient Research Experiments")
    parser.add_argument("--quick", action="store_true", help="Run 4 quick experiments")
    parser.add_argument("--output", default="outputs/zkp_research", help="Output directory")
    parser.add_argument("--data-dir", default="./data", help="MNIST data directory")
    parser.add_argument("--grad-sample", type=int, default=1000,
                        help="Gradient elements sampled per proof (default 1000)")
    parser.add_argument("--aggregator", type=str, choices=["fedavg", "multi_krum", "coord_median"],
                        help="Fix a single aggregator for all tests instead of the full matrix")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    global log
    log = make_logger(out_dir / "main.log")

    log.info("=" * 90)
    log.info("ZK PROOF-OF-GRADIENT RESEARCH EXPERIMENTS")
    log.info("=" * 90)
    log.info(f"Timestamp: {ts}")
    log.info(f"Output: {out_dir}")
    log.info(f"Mode: {'QUICK (3 experiments)' if args.quick else 'FULL (72 experiments)'}")
    log.info(f"ZKP backend: {'ProtoGalaxy IVC (real)' if _ZKP_AVAILABLE else 'SHA-256 fallback'}")
    log.info(f"Gradient sample size: {args.grad_sample}")
    log.info(f"Fixed Aggregator: {args.aggregator if args.aggregator else 'All'}")

    configs = build_configs(args.quick, args.aggregator)
    log.info(f"Total experiments: {len(configs)}")

    all_results = []
    for i, config in enumerate(configs, 1):
        log.info(f"\n[{i}/{len(configs)}] Starting experiment...")
        try:
            result = run_experiment(
                *config,
                data_dir=args.data_dir,
                output_dir=out_dir,
                grad_sample_size=args.grad_sample,
            )
            all_results.append(result)
        except Exception as e:
            log.error(f"Experiment {config} FAILED: {e}")
            import traceback
            log.error(traceback.format_exc())

    # ── Aggregate summary ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 90)
    log.info("SUITE COMPLETE — AGGREGATE RESULTS")
    log.info("=" * 90)

    if all_results:
        avg_recall = np.mean([r["metrics"]["recall"] for r in all_results])
        avg_f1 = np.mean([r["metrics"]["f1"] for r in all_results])
        avg_acc = np.mean([r["metrics"]["final_test_accuracy"] for r in all_results])
        avg_proof_ms = np.mean([r["metrics"]["avg_proof_time_ms"] for r in all_results])
        log.info(f"  Avg recall:     {avg_recall:.3f}")
        log.info(f"  Avg F1:         {avg_f1:.3f}")
        log.info(f"  Avg final acc:  {avg_acc:.4f}")
        log.info(f"  Avg proof time: {avg_proof_ms:.0f}ms")

        # Per-attack breakdown
        attack_names = set(r["config"]["attack"] for r in all_results)
        for attack in sorted(attack_names):
            subset = [r for r in all_results if r["config"]["attack"] == attack]
            a_recall = np.mean([r["metrics"]["recall"] for r in subset])
            log.info(f"  {attack:20s}: recall={a_recall:.3f} ({len(subset)} experiments)")

        # Per-model breakdown
        for model in ["linear", "mlp", "cnn"]:
            subset = [r for r in all_results if r["config"]["model_type"] == model]
            if subset:
                m_acc = np.mean([r["metrics"]["final_test_accuracy"] for r in subset])
                m_f1 = np.mean([r["metrics"]["f1"] for r in subset])
                log.info(f"  {model:8s}: final_acc={m_acc:.4f}  F1={m_f1:.3f} ({len(subset)} exps)")

    summary = {
        "timestamp": ts,
        "zkp_mode": "real_protogalaxy" if _ZKP_AVAILABLE else "sha256_fallback",
        "total_experiments": len(all_results),
        "experiments": all_results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\nResults saved to: {out_dir}")
    log.info(f"Summary JSON: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
