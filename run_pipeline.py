"""
ProtoGalaxy Full Pipeline Runner
=================================
Runs the complete 4-phase federated learning pipeline end-to-end:
  Phase 1: Commitment  — clients commit gradients via CommitmentGenerator
  Phase 2: Revelation  — clients reveal gradients, verified via ProofVerifier
  Phase 3: Defense     — multi-layer Byzantine detection + robust aggregation
  Phase 4: Aggregation — global model update + model distribution

Uses real MNIST/CIFAR-10 data with IID or Non-IID partitioning across clients.
Computes all evaluation metrics from Architecture Section 6.4:
  - Model Accuracy
  - Byzantine Detection Rate (True Positive Rate)
  - False Positive Rate
  - Convergence Speed
  - Communication Overhead
"""

import sys
import os
import time
import copy
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.mnist import create_mnist_model
from src.client.trainer import Trainer
from src.orchestration.pipeline import ProtoGalaxyPipeline
from src.crypto.merkle import verify_proof as merkle_verify_proof


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_mnist(data_dir="./data"):
    """Download and load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def load_cifar10(data_dir="./data"):
    """Download and load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    return train_dataset, test_dataset


def iid_partition(dataset, num_clients):
    """IID partition: randomly split dataset equally among clients."""
    n = len(dataset)
    indices = np.random.permutation(n).tolist()
    shard_size = n // num_clients
    client_data = {}
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size
        client_data[i] = Subset(dataset, indices[start:end])
    return client_data


def noniid_partition(dataset, num_clients, num_classes=10, classes_per_client=2):
    """Non-IID (label-skew) partition: each client gets a limited set of classes.

    Architecture Section 6.1 — Non-IID (Label Skew).
    """
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    class_indices = {c: np.where(targets == c)[0].tolist()
                     for c in range(num_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    client_data = {i: [] for i in range(num_clients)}
    all_classes = list(range(num_classes))
    np.random.shuffle(all_classes)
    for cid in range(num_clients):
        chosen = [all_classes[(cid * classes_per_client + j) % num_classes]
                  for j in range(classes_per_client)]
        for c in chosen:
            share = len(class_indices[c]) // num_clients
            client_data[cid].extend(class_indices[c][:share])
            class_indices[c] = class_indices[c][share:]

    return {cid: Subset(dataset, idxs) for cid, idxs in client_data.items()}


def dirichlet_partition(dataset, num_clients, alpha=0.5, num_classes=10):
    """Dirichlet Non-IID partition (Architecture Section 6.1)."""
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    class_indices = {c: np.where(targets == c)[0].tolist()
                     for c in range(num_classes)}
    client_data = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        idxs = class_indices[c]
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)
        splits = np.split(np.array(idxs), proportions[:-1])
        for cid in range(num_clients):
            client_data[cid].extend(splits[cid].tolist())

    return {cid: Subset(dataset, idxs) for cid, idxs in client_data.items()}


def evaluate_model(model, test_loader, device="cpu"):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0


# ─── Evaluation Metrics (Architecture Section 6.4) ───────────────────────────

class MetricsTracker:
    """Tracks all evaluation metrics from Architecture Section 6.4."""

    def __init__(self, byzantine_ids: set, num_clients: int):
        self.byzantine_ids = byzantine_ids
        self.num_clients = num_clients
        self.round_metrics: list = []
        self.accuracies: list = []
        self.total_bytes_sent: int = 0

    def record_round(self, round_num, accuracy, flagged_clients,
                     round_time, bytes_sent=0):
        self.accuracies.append(accuracy)
        self.total_bytes_sent += bytes_sent

        tp = len(flagged_clients & self.byzantine_ids)
        fp = len(flagged_clients - self.byzantine_ids)
        fn = len(self.byzantine_ids - flagged_clients)
        honest = set(range(self.num_clients)) - self.byzantine_ids
        tn = len(honest - flagged_clients)

        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)

        self.round_metrics.append({
            'round': round_num, 'accuracy': accuracy,
            'tpr': tpr, 'fpr': fpr,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'flagged': sorted(flagged_clients),
            'round_time': round_time,
            'cumulative_bytes': self.total_bytes_sent,
        })

    def convergence_round(self, target_acc=0.85):
        for i, acc in enumerate(self.accuracies):
            if acc >= target_acc:
                return i
        return -1

    def summary(self):
        if not self.round_metrics:
            return {}
        avg_tpr = float(np.mean([m['tpr'] for m in self.round_metrics]))
        avg_fpr = float(np.mean([m['fpr'] for m in self.round_metrics]))
        return {
            'final_accuracy': self.accuracies[-1],
            'avg_byzantine_detection_rate_TPR': avg_tpr,
            'avg_false_positive_rate_FPR': avg_fpr,
            'convergence_round_85pct': self.convergence_round(0.85),
            'total_bytes_sent': self.total_bytes_sent,
            'per_round': self.round_metrics,
        }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ProtoGalaxy Full Pipeline")
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--num-galaxies", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar10"])
    parser.add_argument("--model-type", type=str, default="linear",
                        choices=["linear", "mlp", "cnn"])
    parser.add_argument("--partition", type=str, default="iid",
                        choices=["iid", "noniid", "dirichlet"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--byzantine-fraction", type=float, default=0.0)
    parser.add_argument("--attack-type", type=str, default="label_flip",
                        choices=["label_flip", "targeted_label_flip",
                                 "backdoor", "model_poisoning",
                                 "gaussian_noise"])
    parser.add_argument("--trim-ratio", type=float, default=0.3)
    parser.add_argument("--aggregation-method", type=str, default="trimmed_mean",
                        choices=["trimmed_mean", "multi_krum",
                                 "coordinate_wise_median"])
    parser.add_argument("--save-metrics", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("   ProtoGalaxy Full Pipeline — End-to-End Run")
    print("=" * 70)
    print(f"  Clients: {args.num_clients}  |  Galaxies: {args.num_galaxies}  |  "
          f"Rounds: {args.num_rounds}  |  Model: {args.model_type}")
    print(f"  Dataset: {args.dataset}  |  Partition: {args.partition}  |  "
          f"Aggregation: {args.aggregation_method}")

    num_byzantine = int(args.num_clients * args.byzantine_fraction)
    byzantine_ids = set(range(num_byzantine))
    if num_byzantine > 0:
        print(f"  ⚠️  Byzantine: {num_byzantine} clients ({args.attack_type})  "
              f"IDs: {sorted(byzantine_ids)}")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n📦 Loading {args.dataset.upper()} dataset...")
    if args.dataset == "cifar10":
        train_dataset, test_dataset = load_cifar10()
    else:
        train_dataset, test_dataset = load_mnist()

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    if args.partition == "noniid":
        client_data = noniid_partition(train_dataset, args.num_clients)
        print(f"   Non-IID (label-skew) partition")
    elif args.partition == "dirichlet":
        client_data = dirichlet_partition(train_dataset, args.num_clients,
                                          alpha=args.dirichlet_alpha)
        print(f"   Dirichlet partition (α={args.dirichlet_alpha})")
    else:
        client_data = iid_partition(train_dataset, args.num_clients)
        print(f"   IID partition")
    print(f"   {len(train_dataset)} train samples → ~{len(client_data[0])} per client")

    # ── 2. Create global model ────────────────────────────────────────────────
    global_model = create_mnist_model(args.model_type)
    initial_acc = evaluate_model(global_model, test_loader)
    print(f"\n🧠 Global model ({args.model_type}): "
          f"{sum(p.numel() for p in global_model.parameters())} params")
    print(f"   Initial accuracy: {initial_acc:.2%}")

    # ── 3. Create pipeline ────────────────────────────────────────────────────
    defense_config = {
        'layer3_trim_ratio': args.trim_ratio,
        'layer3_method': args.aggregation_method,
    }
    pipeline = ProtoGalaxyPipeline(
        global_model=global_model,
        num_clients=args.num_clients,
        num_galaxies=args.num_galaxies,
        defense_config=defense_config,
        logger=None,
    )
    print(f"\n🚀 Pipeline initialized")
    print(f"   Galaxy assignments: "
          f"{ {g: len(c) for g, c in pipeline.galaxy_assignments.items()} }")

    metrics = MetricsTracker(byzantine_ids=byzantine_ids,
                             num_clients=args.num_clients)

    # ── 4. Run rounds ─────────────────────────────────────────────────────────
    for round_num in range(args.num_rounds):
        round_start = time.time()
        pipeline.current_round = round_num

        print(f"\n{'─' * 70}")
        print(f"  ROUND {round_num}")
        print(f"{'─' * 70}")

        # LOCAL TRAINING
        print(f"\n  ⚙️  Local training ({args.local_epochs} epoch(s))...")
        client_trainers = {}
        for cid in range(args.num_clients):
            client_model = copy.deepcopy(global_model)
            trainer = Trainer(model=client_model, learning_rate=args.lr)
            loader = DataLoader(client_data[cid], batch_size=args.batch_size,
                                shuffle=True)
            trainer.train(loader, num_epochs=args.local_epochs)
            client_trainers[cid] = trainer

        # PHASE 1: COMMITMENT
        print(f"  📝 Phase 1: Commitment...")
        client_grads = {}
        commitments_by_galaxy = {}
        client_metadata = {}

        for cid, trainer in client_trainers.items():
            gradients = trainer.get_gradients()

            if cid in byzantine_ids:
                if args.attack_type == "label_flip":
                    gradients = [-g for g in gradients]
                elif args.attack_type == "targeted_label_flip":
                    for li in range(max(0, len(gradients) - 2), len(gradients)):
                        gradients[li] = -gradients[li]
                elif args.attack_type == "model_poisoning":
                    gradients = [g * (-10.0) for g in gradients]
                elif args.attack_type == "backdoor":
                    rng = np.random.RandomState(42 + cid)
                    for li in range(len(gradients)):
                        g = gradients[li]
                        if isinstance(g, torch.Tensor):
                            trigger = torch.tensor(
                                rng.randn(*g.shape).astype(np.float32),
                                device=g.device, dtype=g.dtype)
                            gradients[li] = g + 0.1 * trigger
                elif args.attack_type == "gaussian_noise":
                    for li in range(len(gradients)):
                        g = gradients[li]
                        if isinstance(g, torch.Tensor):
                            gradients[li] = g + torch.randn_like(g)

            client_grads[cid] = gradients

            commit_hash, metadata = pipeline.phase1_client_commitment(
                cid, gradients, round_num)
            client_metadata[cid] = metadata

            galaxy_id = cid % args.num_galaxies
            if galaxy_id not in commitments_by_galaxy:
                commitments_by_galaxy[galaxy_id] = {}
            commitments_by_galaxy[galaxy_id][cid] = commit_hash

        # Galaxy Merkle trees
        galaxy_roots = {}
        for galaxy_id, commits in commitments_by_galaxy.items():
            root = pipeline.phase1_galaxy_collect_commitments(
                galaxy_id, commits, round_num)
            galaxy_roots[galaxy_id] = root

        # Global Merkle tree
        global_root = pipeline.phase1_global_collect_galaxy_roots(
            galaxy_roots, round_num)
        print(f"     Global Merkle root: {global_root[:16]}...")

        # ZK sum-check proof generation (Architecture §3.3 + §4.1)
        zk_prove_metrics = pipeline.phase1_generate_zk_proofs(
            client_grads, round_num)
        print(f"     🔐 ZK proofs: {zk_prove_metrics['proofs_generated']} generated "
              f"[{zk_prove_metrics['mode']}] "
              f"({zk_prove_metrics['avg_prove_time_ms']:.0f}ms/client)")

        # ── Client-side Merkle root verification (§3.4 / §5.4) ────
        merkle_ok = 0
        merkle_fail = 0
        for galaxy_id, commits in commitments_by_galaxy.items():
            adapter = pipeline.galaxy_merkle_trees[galaxy_id]
            gal_root = adapter.get_root()
            for cid, commit_hash in commits.items():
                proof = adapter.get_proof(cid)
                if gal_root and proof is not None:
                    idx = (adapter.client_ids.index(cid)
                           if cid in adapter.client_ids else 0)
                    ok = merkle_verify_proof(gal_root, proof, commit_hash, idx)
                    if ok:
                        merkle_ok += 1
                    else:
                        merkle_fail += 1
                        print(f"     ⚠️  Client {cid} Merkle verification FAILED")
                else:
                    merkle_fail += 1
        print(f"     🔒 Client Merkle verification: "
              f"{merkle_ok} ok, {merkle_fail} fail")

        # PHASE 2: REVELATION & VERIFICATION
        print(f"  🔍 Phase 2: Revelation & verification...")
        galaxy_verified = {}
        total_verified = 0
        total_rejected = 0

        for galaxy_id in range(args.num_galaxies):
            subs = {}
            for cid in commitments_by_galaxy.get(galaxy_id, {}):
                sub = pipeline.phase2_client_submit_gradients(
                    cid, galaxy_id, client_grads[cid],
                    commitments_by_galaxy[galaxy_id][cid],
                    client_metadata[cid], round_num)
                subs[cid] = sub

            verified, rejected = pipeline.phase2_galaxy_verify_and_collect(
                galaxy_id, subs)
            galaxy_verified[galaxy_id] = verified
            total_verified += len(verified)
            total_rejected += len(rejected)

        print(f"     ✅ Verified: {total_verified}  ❌ Rejected: {total_rejected}")

        # ZK sum-check proof verification (Architecture §4.1)
        all_verified_ids = [
            u['client_id']
            for updates in galaxy_verified.values()
            for u in updates
        ]
        zk_verify_metrics = pipeline.phase2_verify_zk_proofs(all_verified_ids)
        if zk_verify_metrics['zk_verified'] > 0 or zk_verify_metrics['zk_failed'] > 0:
            print(f"     🔐 ZK verify [{zk_verify_metrics['mode']}]: "
                  f"{zk_verify_metrics['zk_verified']} valid, "
                  f"{zk_verify_metrics['zk_failed']} invalid "
                  f"({zk_verify_metrics['verify_time_ms']:.0f}ms)")

        # Remove ZK-rejected clients from verified updates
        if zk_verify_metrics['zk_rejected_ids']:
            zk_reject_set = set(zk_verify_metrics['zk_rejected_ids'])
            for galaxy_id in galaxy_verified:
                original = galaxy_verified[galaxy_id]
                galaxy_verified[galaxy_id] = [
                    u for u in original if u['client_id'] not in zk_reject_set
                ]
            total_verified -= len(zk_reject_set)
            total_rejected += len(zk_reject_set)
            print(f"     ⚠️  ZK rejected: {sorted(zk_reject_set)}")

        # PHASE 3: DEFENSE
        print(f"  🛡️  Phase 3: Multi-layer defense...")
        galaxy_agg_grads = {}
        galaxy_defense_reports = {}
        round_flagged_clients: set = set()

        for galaxy_id, verified_updates in galaxy_verified.items():
            if not verified_updates:
                continue

            agg_grads, report = pipeline.phase3_galaxy_defense_pipeline(
                galaxy_id, verified_updates)
            galaxy_agg_grads[galaxy_id] = agg_grads
            galaxy_defense_reports[galaxy_id] = report

            flagged = report.get('flagged_clients', [])
            for idx in flagged:
                if idx < len(verified_updates):
                    round_flagged_clients.add(
                        verified_updates[idx]['client_id'])
            for idx in report.get('statistical_flagged', []):
                if idx < len(verified_updates):
                    round_flagged_clients.add(
                        verified_updates[idx]['client_id'])

            method = report.get('aggregation_method', 'unknown')
            print(f"     Galaxy {galaxy_id}: method={method}, "
                  f"flagged={len(flagged)}")

        # Galaxy submissions to global
        galaxy_submissions = {}
        for galaxy_id in galaxy_agg_grads:
            client_ids = [u['client_id'] for u in galaxy_verified[galaxy_id]]
            galaxy_sub = pipeline.phase3_galaxy_submit_to_global(
                galaxy_id, galaxy_agg_grads[galaxy_id],
                galaxy_defense_reports[galaxy_id], client_ids)
            galaxy_submissions[galaxy_id] = galaxy_sub

        # PHASE 4: GLOBAL AGGREGATION
        print(f"  🌐 Phase 4: Global aggregation...")

        verified_galaxies, rejected_galaxies = \
            pipeline.phase4_global_verify_galaxies(galaxy_submissions)
        print(f"     Verified galaxies: {len(verified_galaxies)}  "
              f"Rejected: {len(rejected_galaxies)}")

        layer5_result = pipeline.phase4_layer5_galaxy_defense(
            verified_galaxies)

        global_grads, global_defense_report = \
            pipeline.phase4_global_defense_and_aggregate(
                verified_galaxies, layer5_result=layer5_result)

        if num_byzantine > 0:
            l5_flagged = global_defense_report.get(
                'layer5_flagged_galaxies', [])
            if l5_flagged:
                print(f"     🚨 Layer 5 flagged galaxies: {l5_flagged}")
            reps = global_defense_report.get('galaxy_reputations', {})
            if reps:
                rep_str = ', '.join(
                    f'G{k}={v:.3f}' for k, v in sorted(reps.items()))
                print(f"     📈 Galaxy reputations: {rep_str}")

        pipeline.phase4_update_global_model(global_grads)

        # Galaxy ZK proof folding (Architecture §3.2 — multi-level ZK)
        clean_galaxy_ids = [
            u['galaxy_id'] for u in verified_galaxies
            if u['galaxy_id'] not in set(
                global_defense_report.get('flagged_galaxies', [])
                + global_defense_report.get('layer5_flagged_galaxies', [])
            )
        ]
        zk_fold_metrics = pipeline.phase4_fold_galaxy_zk_proofs(
            clean_galaxy_ids)
        if zk_fold_metrics['galaxies_folded'] > 0:
            print(f"     🔐 Galaxy proofs folded [{zk_fold_metrics['mode']}]: "
                  f"{zk_fold_metrics['galaxies_folded']} galaxies "
                  f"({zk_fold_metrics['folding_time_ms']:.0f}ms)")

        sync_package = pipeline.phase4_distribute_model()

        # ROUND SUMMARY + METRICS
        round_time = time.time() - round_start
        accuracy = evaluate_model(global_model, test_loader)

        param_bytes = sum(p.numel() * 4 for p in global_model.parameters())
        bytes_this_round = param_bytes * args.num_clients * 2

        metrics.record_round(
            round_num=round_num,
            accuracy=accuracy,
            flagged_clients=round_flagged_clients,
            round_time=round_time,
            bytes_sent=bytes_this_round,
        )

        rm = metrics.round_metrics[-1]
        print(f"\n  📊 Round {round_num} summary:")
        print(f"     Accuracy: {accuracy:.2%}")
        print(f"     Model hash: {sync_package['model_hash'][:16]}...")
        print(f"     Duration: {round_time:.2f}s")
        if num_byzantine > 0:
            print(f"     Detection TPR: {rm['tpr']:.2%}  |  "
                  f"FPR: {rm['fpr']:.2%}")
            print(f"     Flagged clients: {rm['flagged']}")

    # ── Final ─────────────────────────────────────────────────────────────────
    final_acc = evaluate_model(global_model, test_loader)
    summary = metrics.summary()
    print(f"\n{'=' * 70}")
    print(f"   PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Initial accuracy:  {initial_acc:.2%}")
    print(f"  Final accuracy:    {final_acc:.2%}")
    print(f"  Improvement:       {(final_acc - initial_acc):.2%}")
    print(f"  Rounds completed:  {args.num_rounds}")
    print(f"  Clients/round:     {args.num_clients}")
    print(f"  Galaxies:          {args.num_galaxies}")
    if num_byzantine > 0:
        print(f"  Avg Detection TPR: "
              f"{summary['avg_byzantine_detection_rate_TPR']:.2%}")
        print(f"  Avg False Pos FPR: "
              f"{summary['avg_false_positive_rate_FPR']:.2%}")
    conv = summary.get('convergence_round_85pct', -1)
    if conv >= 0:
        print(f"  Convergence (85%): Round {conv}")
    else:
        print(f"  Convergence (85%): Not reached in {args.num_rounds} rounds")
    print(f"  Comm overhead:     "
          f"{summary.get('total_bytes_sent', 0) / 1e6:.2f} MB")
    print(f"{'=' * 70}")

    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics) or '.', exist_ok=True)
        with open(args.save_metrics, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  📁 Metrics saved to {args.save_metrics}")


if __name__ == "__main__":
    main()
