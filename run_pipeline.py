"""
ProtoGalaxy Full Pipeline Runner
=================================
Runs the complete 4-phase federated learning pipeline end-to-end:
  Phase 1: Commitment  — clients commit gradients via CommitmentGenerator
  Phase 2: Revelation  — clients reveal gradients, verified via ProofVerifier
  Phase 3: Defense     — multi-layer Byzantine detection + robust aggregation
  Phase 4: Aggregation — global model update + model distribution

Uses real MNIST data with IID partitioning across clients.
"""

import sys
import os
import time
import copy
import argparse

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ProtoGalaxy Full Pipeline")
    parser.add_argument("--num-clients", type=int, default=8, help="Number of clients")
    parser.add_argument("--num-galaxies", type=int, default=2, help="Number of galaxies")
    parser.add_argument("--num-rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--model-type", type=str, default="linear",
                        choices=["linear", "mlp", "cnn"], help="Model architecture")
    parser.add_argument("--byzantine-fraction", type=float, default=0.0,
                        help="Fraction of clients that are Byzantine (0.0-0.5)")
    parser.add_argument("--attack-type", type=str, default="label_flip",
                        choices=["label_flip", "targeted_label_flip", "backdoor",
                                 "model_poisoning", "gaussian_noise"],
                        help="Attack type for Byzantine clients")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("   ProtoGalaxy Full Pipeline — End-to-End Run")
    print("=" * 70)
    print(f"  Clients: {args.num_clients}  |  Galaxies: {args.num_galaxies}  |  "
          f"Rounds: {args.num_rounds}  |  Model: {args.model_type}")
    
    # Determine Byzantine clients
    num_byzantine = int(args.num_clients * args.byzantine_fraction)
    byzantine_ids = set(range(num_byzantine))  # First N clients are Byzantine
    if num_byzantine > 0:
        print(f"  ⚠️  Byzantine: {num_byzantine} clients ({args.attack_type})  "
              f"IDs: {sorted(byzantine_ids)}")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n📦 Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data = iid_partition(train_dataset, args.num_clients)
    print(f"   {len(train_dataset)} train samples → {len(client_data[0])} per client (IID)")

    # ── 2. Create global model ────────────────────────────────────────────────
    global_model = create_mnist_model(args.model_type)
    initial_acc = evaluate_model(global_model, test_loader)
    print(f"\n🧠 Global model ({args.model_type}): {sum(p.numel() for p in global_model.parameters())} params")
    print(f"   Initial accuracy: {initial_acc:.2%}")

    # ── 3. Create pipeline ────────────────────────────────────────────────────
    pipeline = ProtoGalaxyPipeline(
        global_model=global_model,
        num_clients=args.num_clients,
        num_galaxies=args.num_galaxies,
        logger=None  # Skip logging (FLLogger has signature mismatches)
    )
    print(f"\n🚀 Pipeline initialized")
    print(f"   Galaxy assignments: { {g: len(c) for g, c in pipeline.galaxy_assignments.items()} }")

    # ── 4. Run rounds ──────────────────────────────────────────────────────────
    for round_num in range(args.num_rounds):
        round_start = time.time()
        pipeline.current_round = round_num

        print(f"\n{'─' * 70}")
        print(f"  ROUND {round_num}")
        print(f"{'─' * 70}")

        # ============================================================
        # LOCAL TRAINING
        # ============================================================
        print(f"\n  ⚙️  Local training ({args.local_epochs} epoch(s))...")
        client_trainers = {}
        for cid in range(args.num_clients):
            # Each client gets a copy of the global model
            client_model = copy.deepcopy(global_model)
            trainer = Trainer(model=client_model, learning_rate=args.lr)
            loader = DataLoader(client_data[cid], batch_size=args.batch_size, shuffle=True)
            trainer.train(loader, num_epochs=args.local_epochs)
            client_trainers[cid] = trainer

        # ============================================================
        # PHASE 1: COMMITMENT
        # ============================================================
        print(f"  📝 Phase 1: Commitment...")
        client_grads = {}
        commitments_by_galaxy = {}
        client_metadata = {}

        for cid, trainer in client_trainers.items():
            gradients = trainer.get_gradients()
            
            # ── Apply Byzantine attack if this client is malicious ──
            if cid in byzantine_ids:
                if args.attack_type == "label_flip":
                    gradients = [-g for g in gradients]
                elif args.attack_type == "targeted_label_flip":
                    # Flip last 2 layers only
                    for li in range(max(0, len(gradients)-2), len(gradients)):
                        gradients[li] = -gradients[li]
                elif args.attack_type == "model_poisoning":
                    gradients = [g * (-10.0) for g in gradients]
                elif args.attack_type == "backdoor":
                    rng = np.random.RandomState(42 + cid)
                    for li in range(len(gradients)):
                        g = gradients[li]
                        if isinstance(g, torch.Tensor):
                            trigger = torch.tensor(rng.randn(*g.shape).astype(np.float32),
                                                   device=g.device, dtype=g.dtype)
                            gradients[li] = g + 0.1 * trigger
                        else:
                            gradients[li] = g + 0.1 * rng.randn(*np.array(g).shape).astype(np.float32)
                elif args.attack_type == "gaussian_noise":
                    for li in range(len(gradients)):
                        g = gradients[li]
                        if isinstance(g, torch.Tensor):
                            gradients[li] = g + torch.randn_like(g)
                        else:
                            gradients[li] = g + np.random.randn(*np.array(g).shape).astype(np.float32)
            
            client_grads[cid] = gradients

            commit_hash, metadata = pipeline.phase1_client_commitment(
                cid, gradients, round_num
            )
            client_metadata[cid] = metadata

            galaxy_id = cid % args.num_galaxies
            if galaxy_id not in commitments_by_galaxy:
                commitments_by_galaxy[galaxy_id] = {}
            commitments_by_galaxy[galaxy_id][cid] = commit_hash

        # Galaxy Merkle trees
        galaxy_roots = {}
        for galaxy_id, commits in commitments_by_galaxy.items():
            root = pipeline.phase1_galaxy_collect_commitments(galaxy_id, commits, round_num)
            galaxy_roots[galaxy_id] = root

        # Global Merkle tree
        global_root = pipeline.phase1_global_collect_galaxy_roots(galaxy_roots, round_num)
        print(f"     Global Merkle root: {global_root[:16]}...")

        # ============================================================
        # PHASE 2: REVELATION & VERIFICATION
        # ============================================================
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
                    client_metadata[cid], round_num
                )
                subs[cid] = sub

            verified, rejected = pipeline.phase2_galaxy_verify_and_collect(galaxy_id, subs)
            galaxy_verified[galaxy_id] = verified
            total_verified += len(verified)
            total_rejected += len(rejected)

        print(f"     ✅ Verified: {total_verified}  ❌ Rejected: {total_rejected}")

        # ============================================================
        # PHASE 3: DEFENSE
        # ============================================================
        print(f"  🛡️  Phase 3: Multi-layer defense...")
        galaxy_agg_grads = {}
        galaxy_defense_reports = {}

        for galaxy_id, verified_updates in galaxy_verified.items():
            if not verified_updates:
                continue

            agg_grads, report = pipeline.phase3_galaxy_defense_pipeline(
                galaxy_id, verified_updates
            )
            galaxy_agg_grads[galaxy_id] = agg_grads
            galaxy_defense_reports[galaxy_id] = report

            flagged = report.get('flagged_clients', [])
            method = report.get('aggregation_method', 'unknown')
            print(f"     Galaxy {galaxy_id}: method={method}, flagged={len(flagged)}")

        # Galaxy submissions to global
        galaxy_submissions = {}
        for galaxy_id in galaxy_agg_grads:
            client_ids = [u['client_id'] for u in galaxy_verified[galaxy_id]]
            galaxy_sub = pipeline.phase3_galaxy_submit_to_global(
                galaxy_id, galaxy_agg_grads[galaxy_id],
                galaxy_defense_reports[galaxy_id], client_ids
            )
            galaxy_submissions[galaxy_id] = galaxy_sub

        # ============================================================
        # PHASE 4: GLOBAL AGGREGATION
        # ============================================================
        print(f"  🌐 Phase 4: Global aggregation...")

        # Verify galaxy submissions
        verified_galaxies, rejected_galaxies = pipeline.phase4_global_verify_galaxies(
            galaxy_submissions
        )
        print(f"     Verified galaxies: {len(verified_galaxies)}  "
              f"Rejected: {len(rejected_galaxies)}")

        # Layer 5 galaxy defense
        layer5_result = pipeline.phase4_layer5_galaxy_defense(verified_galaxies)

        # Global defense and aggregation (with Layer 5 filtering)
        global_grads, global_defense_report = pipeline.phase4_global_defense_and_aggregate(
            verified_galaxies, layer5_result=layer5_result
        )
        
        if num_byzantine > 0:
            l5_flagged = global_defense_report.get('layer5_flagged_galaxies', [])
            if l5_flagged:
                print(f"     🚨 Layer 5 flagged galaxies: {l5_flagged}")
            reps = global_defense_report.get('galaxy_reputations', {})
            if reps:
                rep_str = ', '.join(f'G{k}={v:.3f}' for k, v in sorted(reps.items()))
                print(f"     📈 Galaxy reputations: {rep_str}")

        # Update global model
        pipeline.phase4_update_global_model(global_grads)

        # Distribute (creates sync package)
        sync_package = pipeline.phase4_distribute_model()

        # ============================================================
        # ROUND SUMMARY
        # ============================================================
        round_time = time.time() - round_start
        accuracy = evaluate_model(global_model, test_loader)

        print(f"\n  📊 Round {round_num} summary:")
        print(f"     Accuracy: {accuracy:.2%}")
        print(f"     Model hash: {sync_package['model_hash'][:16]}...")
        print(f"     Duration: {round_time:.2f}s")

    # ── Final ──────────────────────────────────────────────────────────────────
    final_acc = evaluate_model(global_model, test_loader)
    print(f"\n{'=' * 70}")
    print(f"   PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Initial accuracy:  {initial_acc:.2%}")
    print(f"  Final accuracy:    {final_acc:.2%}")
    print(f"  Improvement:       {(final_acc - initial_acc):.2%}")
    print(f"  Rounds completed:  {args.num_rounds}")
    print(f"  Clients/round:     {args.num_clients}")
    print(f"  Galaxies:          {args.num_galaxies}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
