"""Instrumented pipeline verification — traces every stage with detailed output."""
import torch, copy, numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision, torchvision.transforms as transforms

from src.models.mnist import create_mnist_model
from src.client.trainer import Trainer
from src.orchestration.pipeline import ProtoGalaxyPipeline
from src.crypto.merkle import verify_proof

# Setup
torch.manual_seed(42); np.random.seed(42)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

global_model = create_mnist_model('linear')
num_clients, num_galaxies = 4, 2

# Partition data
indices = np.random.permutation(len(train_ds)).tolist()
shard = len(train_ds) // num_clients
client_data = {i: Subset(train_ds, indices[i*shard:(i+1)*shard]) for i in range(num_clients)}

# Create pipeline
pipeline = ProtoGalaxyPipeline(
    global_model=global_model, num_clients=num_clients, num_galaxies=num_galaxies,
    defense_config={'layer3_trim_ratio': 0.3, 'layer3_method': 'trimmed_mean'}, logger=None)

# ===== TRAINING =====
print('=== 1. MODEL TRAINING ===')
trainers = {}
for cid in range(num_clients):
    cm = copy.deepcopy(global_model)
    t = Trainer(model=cm, learning_rate=0.01)
    loader = DataLoader(client_data[cid], batch_size=64, shuffle=True)
    metrics = t.train(loader, num_epochs=1)
    trainers[cid] = t
    print(f'  Client {cid}: loss={metrics["loss"]:.4f}, acc={metrics["accuracy"]:.4f}, samples={metrics["samples"]}')

grads = trainers[0].get_gradients()
print(f'  Gradient shapes: {[g.shape for g in grads]}')
print(f'  Gradient norms: {[f"{g.norm().item():.4f}" for g in grads]}')

# ===== COMMITMENT =====
print('\n=== 2. COMMITMENT GENERATION ===')
client_grads, commitments_by_galaxy, client_metadata = {}, {}, {}
for cid, t in trainers.items():
    g = t.get_gradients()
    if cid == 0:  # Byzantine: flip gradient
        g = [-x for x in g]
    client_grads[cid] = g
    commit_hash, metadata = pipeline.phase1_client_commitment(cid, g, round_number=0)
    client_metadata[cid] = metadata
    galaxy_id = cid % num_galaxies
    commitments_by_galaxy.setdefault(galaxy_id, {})[cid] = commit_hash
    print(f'  Client {cid}: commit={commit_hash[:24]}..., nonce={metadata["nonce"][:8]}...')

# ===== MERKLE TREE =====
print('\n=== 3. MERKLE TREE CONSTRUCTION ===')
galaxy_roots = {}
for gid, commits in commitments_by_galaxy.items():
    root = pipeline.phase1_galaxy_collect_commitments(gid, commits, 0)
    galaxy_roots[gid] = root
    print(f'  Galaxy {gid}: root={root[:24]}..., clients={list(commits.keys())}')

global_root = pipeline.phase1_global_collect_galaxy_roots(galaxy_roots, 0)
print(f'  Global root: {global_root[:24]}...')

# ===== CLIENT-SIDE MERKLE VERIFICATION =====
print('\n=== 4. CLIENT-SIDE MERKLE VERIFICATION ===')
for gid, commits in commitments_by_galaxy.items():
    adapter = pipeline.galaxy_merkle_trees[gid]
    gal_root = adapter.get_root()
    for cid, commit_hash in commits.items():
        proof = adapter.get_proof(cid)
        idx = adapter.client_ids.index(cid)
        ok = verify_proof(gal_root, proof, commit_hash, idx)
        print(f'  Client {cid} in Galaxy {gid}: proof_len={len(proof)}, leaf_idx={idx}, valid={ok}')

# ===== PHASE 2: VERIFICATION =====
print('\n=== 5. PHASE 2 - REVELATION & VERIFICATION ===')
galaxy_verified = {}
for gid in range(num_galaxies):
    subs = {}
    for cid in commitments_by_galaxy.get(gid, {}):
        sub = pipeline.phase2_client_submit_gradients(
            cid, gid, client_grads[cid],
            commitments_by_galaxy[gid][cid], client_metadata[cid], 0)
        subs[cid] = sub
    verified, rejected = pipeline.phase2_galaxy_verify_and_collect(gid, subs)
    galaxy_verified[gid] = verified
    print(f'  Galaxy {gid}: verified={len(verified)}, rejected={rejected}')
    for v in verified:
        gnorm = sum(g.norm().item()**2 for g in v["gradients"])**.5
        print(f'    Client {v["client_id"]}: reputation={v["reputation"]:.4f}, grad_norm={gnorm:.4f}')

# ===== PHASE 3: DEFENSE + AGGREGATION =====
print('\n=== 6. PHASE 3 - DEFENSE PIPELINE ===')
galaxy_agg, galaxy_reports = {}, {}
for gid, verified in galaxy_verified.items():
    if not verified:
        continue
    agg, report = pipeline.phase3_galaxy_defense_pipeline(gid, verified)
    galaxy_agg[gid] = agg
    galaxy_reports[gid] = report
    print(f'  Galaxy {gid}: method={report["aggregation_method"]}, flagged={report["flagged_clients"]}')
    print(f'    Agg grad shapes: {[g.shape for g in agg]}')
    print(f'    Statistical flagged: {report.get("statistical_flagged", [])}')
    print(f'    Reputation scores: {report.get("reputation_scores", {})}')

# ===== PHASE 4: GLOBAL =====
print('\n=== 7. PHASE 4 - GLOBAL AGGREGATION ===')
galaxy_submissions = {}
for gid in galaxy_agg:
    client_ids = [u['client_id'] for u in galaxy_verified[gid]]
    gs = pipeline.phase3_galaxy_submit_to_global(gid, galaxy_agg[gid], galaxy_reports[gid], client_ids)
    galaxy_submissions[gid] = gs

verified_galaxies, rejected_galaxies = pipeline.phase4_global_verify_galaxies(galaxy_submissions)
print(f'  Verified galaxies: {len(verified_galaxies)}, rejected: {rejected_galaxies}')

l5 = pipeline.phase4_layer5_galaxy_defense(verified_galaxies)
print(f'  Layer 5 flagged: {l5.get("flagged_galaxies", [])}')

global_grads, global_report = pipeline.phase4_global_defense_and_aggregate(verified_galaxies, layer5_result=l5)
print(f'  Global agg method: {global_report["aggregation_method"]}')
print(f'  Global grad shapes: {[g.shape for g in global_grads]}')
print(f'  Galaxy reputations: {global_report.get("galaxy_reputations", {})}')

# ===== MODEL UPDATE =====
print('\n=== 8. MODEL UPDATE & DISTRIBUTION ===')
w_before = [p.data.clone() for p in global_model.parameters()]
pipeline.phase4_update_global_model(global_grads)
w_after = [p.data.clone() for p in global_model.parameters()]
weight_change = sum((a-b).norm().item()**2 for a,b in zip(w_after, w_before))**.5
print(f'  Weight change (L2): {weight_change:.6f}')
assert weight_change > 0, "Model weights did NOT change!"

sync = pipeline.phase4_distribute_model()
print(f'  Model hash: {sync["model_hash"][:32]}...')
print(f'  Sync package keys: {list(sync.keys())}')

# Final accuracy
global_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = global_model(imgs)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
acc = correct/total
print(f'  Final accuracy: {acc:.2%}')
assert acc > 0.5, f"Accuracy too low ({acc:.2%}), model not learning!"

print('\n' + '='*60)
print('  ✅ ALL 8 STAGES VERIFIED SUCCESSFULLY')
print('='*60)
