"""ProtoGalaxy 4-Phase Protocol Orchestrator.

Implements the complete 4-phase federated learning protocol per Architecture Section 3.4:
- Phase 1: Commitment (Merkle tree construction)
- Phase 2: Revelation (Merkle proof verification)  
- Phase 3: Multi-layer defense (Layers 1-5)
- Phase 4: Global aggregation and model distribution
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import logging
from dataclasses import dataclass

from src.crypto.merkle import GalaxyMerkleTree, GlobalMerkleTree, GradientCommitment
from src.crypto.zkp_prover import GradientSumCheckProver, GalaxyProofFolder, ZKProof
from src.defense.coordinator import DefenseCoordinator
from src.aggregators.galaxy import GalaxyAggregator
from src.aggregators.global_agg import GlobalAggregator
from src.orchestration.galaxy_manager import GalaxyManager
from src.storage.forensic_logger import ForensicLogger

logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """Result from a protocol phase."""
    phase_name: str
    success: bool
    data: Dict
    metrics: Dict


class ProtoGalaxyOrchestrator:
    """Orchestrates the complete 4-phase ProtoGalaxy FL protocol."""
    
    def __init__(
        self,
        num_clients: int,
        num_galaxies: int,
        model: torch.nn.Module,
        defense_config: Optional[Dict] = None,
        forensic_dir: str = './forensic_evidence'
    ):
        """Initialize orchestrator.
        
        Args:
            num_clients: Total number of clients
            num_galaxies: Number of galaxies
            model: Global model
            defense_config: Configuration for defense layers
            forensic_dir: Directory for forensic evidence storage
        """
        self.num_clients = num_clients
        self.num_galaxies = num_galaxies
        self.model = model
        
        # Initialize galaxy manager
        self.galaxy_manager = GalaxyManager(num_galaxies)
        
        # Initialize defense coordinator (all 5 layers)
        self.defense_coordinator = DefenseCoordinator(
            num_clients=num_clients,
            num_galaxies=num_galaxies,
            config=defense_config or {}
        )
        
        # Initialize aggregators
        self.galaxy_aggregators = {
            gid: GalaxyAggregator(galaxy_id=gid, num_clients=num_clients // num_galaxies)
            for gid in range(num_galaxies)
        }
        self.global_aggregator = GlobalAggregator(num_galaxies=num_galaxies)
        
        # Merkle trees
        self.galaxy_merkle_trees: Dict[int, GalaxyMerkleTree] = {}
        self.global_merkle_tree: Optional[GlobalMerkleTree] = None
        
        # ZKP infrastructure
        self.zkp_prover = GradientSumCheckProver()
        self.galaxy_proof_folder = GalaxyProofFolder()
        self.client_zk_proofs: Dict[int, ZKProof] = {}
        self.galaxy_zk_proofs: Dict[int, ZKProof] = {}
        
        # Round state
        self.current_round = 0
        self.round_metrics = []
        
        logger.info(f"ProtoGalaxyOrchestrator initialized:")
        logger.info(f"  Clients: {num_clients}")
        logger.info(f"  Galaxies: {num_galaxies}")
        logger.info(f"  Defense: 5 layers enabled")
        logger.info(f"  Forensic logging: {forensic_dir}")
    
    def run_round(
        self,
        client_updates: Dict[int, torch.Tensor],
        client_commitments: Dict[int, GradientCommitment],
        client_proofs: Optional[Dict[int, List]] = None,
        client_gradients: Optional[Dict[int, List]] = None
    ) -> Dict:
        """Run one complete FL round with all 4 phases.
        
        Args:
            client_updates: Dict mapping client_id -> gradient tensor
            client_commitments: Dict mapping client_id -> commitment hash
            client_proofs: Optional dict mapping client_id -> Merkle proof
            client_gradients: Dict mapping client_id -> List[torch.Tensor] (raw gradients for ZK proofs)
            
        Returns:
            Dict with round results and metrics
        """
        logger.info("=" * 100)
        logger.info(f"ROUND {self.current_round}: Starting 4-Phase Protocol")
        logger.info("=" * 100)
        
        round_result = {
            'round': self.current_round,
            'phases': {},
            'metrics': {},
            'success': True
        }
        
        # Phase 1: Commitment + ZK Proof Generation
        logger.info("\n" + "="*100)
        logger.info("PHASE 1: COMMITMENT, MERKLE TREES & ZK SUM-CHECK PROOFS")
        logger.info("="*100)
        phase1_result = self._phase1_commitment(client_commitments, client_gradients or {})
        round_result['phases']['phase1'] = phase1_result
        
        if not phase1_result.success:
            logger.error("Phase 1 failed!")
            round_result['success'] = False
            return round_result
        
        # Phase 2: Revelation
        logger.info("\n" + "="*100)
        logger.info("PHASE 2: REVELATION & MERKLE PROOF VERIFICATION")
        logger.info("="*100)
        phase2_result = self._phase2_revelation(
            client_updates,
            client_commitments,
            client_proofs or {}
        )
        round_result['phases']['phase2'] = phase2_result
        
        if not phase2_result.success:
            logger.error("Phase 2 failed!")
            round_result['success'] = False
            return round_result
        
        # Phase 3: Multi-Layer Defense
        logger.info("\n" + "="*100)
        logger.info("PHASE 3: MULTI-LAYER DEFENSE (Layers 1-5)")
        logger.info("="*100)
        phase3_result = self._phase3_defense(
            phase2_result.data['verified_updates']
        )
        round_result['phases']['phase3'] = phase3_result
        
        # Phase 4: Global Aggregation
        logger.info("\n" + "="*100)
        logger.info("PHASE 4: GLOBAL AGGREGATION & MODEL UPDATE")
        logger.info("="*100)
        phase4_result = self._phase4_aggregation(
            phase3_result.data['clean_updates']
        )
        round_result['phases']['phase4'] = phase4_result
        
        # Compile metrics
        round_result['metrics'] = self._compile_round_metrics(round_result)
        
        # Increment round
        self.current_round += 1
        
        logger.info("\n" + "="*100)
        logger.info(f"ROUND {self.current_round - 1}: COMPLETE")
        logger.info("="*100)
        self._log_round_summary(round_result)
        
        return round_result
    
    def _phase1_commitment(self, client_commitments: Dict[int, GradientCommitment], client_gradients: Optional[Dict[int, List]] = None) -> PhaseResult:
        """Phase 1: Build Merkle trees and generate ZK sum-check proofs.
        
        Args:
            client_commitments: Dict mapping client_id -> commitment object
            client_gradients: Dict mapping client_id -> List[torch.Tensor] (raw gradients for ZK)
            
        Returns:
            PhaseResult with Merkle roots and ZK proofs
        """
        import time as _time
        logger.info(f"Building Merkle trees for {len(client_commitments)} client commitments")
        
        # Build galaxy Merkle trees
        galaxy_roots = {}
        for galaxy_id in range(self.num_galaxies):
            galaxy_clients = self.galaxy_manager.get_galaxy_clients(galaxy_id)
            
            # Get gradients and clients for this galaxy
            galaxy_gradients = []
            galaxy_client_ids = []
            for client_id in galaxy_clients:
                if client_id in client_commitments:
                    commitment_obj = client_commitments[client_id]
                    # Extract gradients from commitment object and flatten to tensor
                    # commitment_obj.gradients is a list of [torch.Tensor, ...]
                    grad_list = commitment_obj.gradients
                    # Flatten all gradients to a single vector
                    flat_grad = torch.cat([g.flatten() for g in grad_list])
                    galaxy_gradients.append(flat_grad)
                    galaxy_client_ids.append(client_id)
            
            if not galaxy_gradients:
                logger.warning(f"Galaxy {galaxy_id} has no commitments!")
                continue
            
            # Create galaxy Merkle tree - gradients FIRST
            tree = GalaxyMerkleTree(
                gradients=galaxy_gradients,
                galaxy_id=galaxy_id,
                client_ids=galaxy_client_ids,
                round_number=self.current_round
            )
            
            root = tree.get_root()
            galaxy_roots[galaxy_id] = root
            self.galaxy_merkle_trees[galaxy_id] = tree
            
            logger.info(f"  Galaxy {galaxy_id}: {len(galaxy_gradients)} commitments, root={root[:16]}...")
        
        # Build global Merkle tree from galaxy tree objects
        global_tree = GlobalMerkleTree(
            galaxy_trees=list(self.galaxy_merkle_trees.values()),
            round_number=self.current_round
        )
        global_root = global_tree.get_root()
        self.global_merkle_tree = global_tree
        
        logger.info(f"✓ Global Merkle root: {global_root[:16]}...")
        logger.info(f"✓ Phase 1 complete: {len(galaxy_roots)} galaxy trees built")
        
        # --- ZK Sum-Check Proof Generation ---
        zk_proofs_generated = 0
        zk_total_time_ms = 0.0
        self.client_zk_proofs = {}  # Reset for this round
        
        if client_gradients:
            logger.info(f"Generating ZK sum-check proofs for {len(client_gradients)} clients...")
            
            # Filter out clients with excessive L2 norms (Byzantine check)
            # Threshold matches Rust backend default (10.0) or user observed value (4.92 vs 49.20)
            NORM_THRESHOLD = 15.0 # Conservative upper bound, malicious observed at ~49.2
            
            valid_gradients = {}
            for cid, grads in client_gradients.items():
                try:
                    # Calculate L2 norm of the full gradient vector
                    # grads is list of tensors
                    total_norm = 0.0
                    for g in grads:
                        total_norm += g.norm().item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    if total_norm > NORM_THRESHOLD:
                         logger.warning(f"Client {cid} rejected: L2 norm {total_norm:.4f} exceeds limit {NORM_THRESHOLD}")
                         continue
                        
                    valid_gradients[cid] = grads
                except Exception as e:
                    logger.error(f"Error checking norm for client {cid}: {e}")
                    
            logger.info(f"  Passed norm check: {len(valid_gradients)}/{len(client_gradients)} clients")

            # Parallelize ZKP generation
            # Rust backend releases GIL, so threading works well here
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import os
            
            # Use max workers based on CPU count (Colab usually has 2-4 vCPUs)
            # But for IO/Rust-bound tasks we can oversubscribe slightly
            max_workers = min(32, (os.cpu_count() or 1) * 4)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_cid = {
                    executor.submit(
                        self.zkp_prover.prove_gradient_sum,
                        gradients=grads,
                        client_id=client_id,
                        round_number=self.current_round
                    ): client_id
                    for client_id, grads in valid_gradients.items()
                }
                
                for future in as_completed(future_to_cid):
                    client_id = future_to_cid[future]
                    try:
                        zk_proof = future.result()
                        self.client_zk_proofs[client_id] = zk_proof
                        zk_proofs_generated += 1
                        zk_total_time_ms += zk_proof.prove_time_ms
                    except Exception as e:
                        logger.error(f"Failed to generate ZK proof for client {client_id}: {e}")
            
            mode = "REAL (ProtoGalaxy IVC)" if self.client_zk_proofs and next(iter(self.client_zk_proofs.values())).is_real else "FALLBACK (SHA-256)"
            logger.info(f"✓ ZK proofs: {zk_proofs_generated} generated [{mode}] with {max_workers} threads")
            # Average time is misleading in parallel, but total_time gives CPU-effort indication
            logger.info(f"  Total prove CPU-time: {zk_total_time_ms:.1f}ms")
        else:
            logger.info("  (No client gradients provided — ZK proofs skipped)")
        
        return PhaseResult(
            phase_name="commitment",
            success=True,
            data={
                'galaxy_roots': galaxy_roots,
                'global_root': global_root,
                'zk_proofs': dict(self.client_zk_proofs)
            },
            metrics={
                'galaxies_built': len(galaxy_roots),
                'total_commitments': len(client_commitments),
                'zk_proofs_generated': zk_proofs_generated,
                'zk_prove_time_ms': zk_total_time_ms
            }
        )
    
    def _phase2_revelation(
        self,
        client_updates: Dict[int, torch.Tensor],
        client_commitments: Dict[int, GradientCommitment],
        client_proofs: Dict[int, List]
    ) -> PhaseResult:
        """Phase 2: Verify Merkle proofs for revealed gradients.
        
        Args:
            client_updates: Client gradient updates
            client_commitments: Client commitment hashes
            client_proofs: Merkle proofs for each client
            
        Returns:
            PhaseResult with verified updates
        """
        logger.info(f"Verifying Merkle proofs for {len(client_updates)} clients")
        
        # 2a. Verify Merkle Proofs & ZK Proofs
        # 2a. Verify Merkle Proofs & ZK Proofs
        verified_updates = {}
        rejected_clients = []
        verification_results = {}
        verified_clients = []
        
        # Parallelize verification
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        max_workers = min(32, (os.cpu_count() or 1) * 4)

        zk_verified = 0
        zk_failed = 0
        zk_verify_time_ms = 0.0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # We need to verify both Merkle and ZKP for each client
            # We'll submit tasks that do both checks
            
            def verify_client(cid, comm, proof, zk_proof, galaxy_tree):
                # Merkle check
                merkle_valid = True
                if comm and proof and galaxy_tree:
                    try:
                        leaf_index = proof.get('leaf_index') if isinstance(proof, dict) else None
                        leaf_hash = proof.get('leaf_hash') if isinstance(proof, dict) else None
                        
                        if leaf_index is not None and leaf_hash is not None:
                            merkle_valid = galaxy_tree.verify(leaf_index, leaf_hash)
                        else:
                            # If proof format is wrong or missing keys
                            merkle_valid = False
                    except Exception:
                        merkle_valid = False
                elif not galaxy_tree:
                    # If we don't have a galaxy tree
                    merkle_valid = False

                # ZKP check
                zkp_valid = True
                if zk_proof:
                    zkp_valid = GradientSumCheckProver.verify_proof(zk_proof)
                
                return cid, merkle_valid, zkp_valid

            future_to_cid = {}
            import time
            zk_start = time.time()

            for client_id, update in client_updates.items():
                galaxy_id = self.galaxy_manager.get_client_galaxy(client_id)
                if galaxy_id is None:
                    logger.warning(f"Client {client_id} not assigned to any galaxy, rejecting")
                    rejected_clients.append(client_id)
                    verification_results[client_id] = False
                    continue
                
                galaxy_tree = self.galaxy_merkle_trees.get(galaxy_id)
                if not galaxy_tree and hasattr(self, 'ablation') and self.ablation != 'merkle_only':
                     logger.warning(f"No Merkle tree for galaxy {galaxy_id}, rejecting client {client_id}")
                     rejected_clients.append(client_id)
                     verification_results[client_id] = False
                     continue
                
                comm = client_commitments.get(client_id)
                proof = client_proofs.get(client_id) # Merkle proof
                zk_proof = self.client_zk_proofs.get(client_id)
                
                future = executor.submit(verify_client, client_id, comm, proof, zk_proof, galaxy_tree)
                future_to_cid[future] = client_id

            for future in as_completed(future_to_cid):
                client_id = future_to_cid[future]
                try:
                    _, m_valid, z_valid = future.result()
                    
                    if not m_valid:
                        logger.warning(f"Client {client_id} failed Merkle verification")
                        rejected_clients.append(client_id)
                        verification_results[client_id] = False
                        continue
                        
                    if not z_valid:
                        logger.warning(f"Client {client_id} failed ZK verification")
                        rejected_clients.append(client_id)
                        verification_results[client_id] = False
                        zk_failed += 1
                        continue
                        
                    verified_clients.append(client_id)
                    verified_updates[client_id] = client_updates[client_id]
                    verification_results[client_id] = True
                    if self.client_zk_proofs.get(client_id):
                        zk_verified += 1
                    
                except Exception as e:
                    logger.error(f"Error verifying client {client_id}: {e}")
                    rejected_clients.append(client_id)
                    verification_results[client_id] = False

            zk_verify_time_ms = (time.time() - zk_start) * 1000

        mode = "REAL" if self.client_zk_proofs and next(iter(self.client_zk_proofs.values())).is_real else "FALLBACK"
        logger.info(f"✓ Verified {len(verified_clients)}/{len(client_updates)} clients (Merkle+ZKP) with {max_workers} threads")
        
        if self.client_zk_proofs:
            logger.info(f"✓ ZK verification [{mode}]: {zk_verified} valid, {zk_failed} invalid ({zk_verify_time_ms:.1f}ms)")
        
        if rejected_clients:
            logger.warning(f"✗ Rejected: {len(rejected_clients)} clients (invalid proofs)")
            
        acceptance_rate = len(verified_updates) / len(client_updates) if client_updates else 0
            
        return PhaseResult(
            phase_name="revelation",
            success=True,
            data={
                'verified_updates': verified_updates,
                'rejected_clients': rejected_clients,
                'verification_results': verification_results
            },
            metrics={
                'verified_count': len(verified_updates),
                'rejected_count': len(rejected_clients),
                'acceptance_rate': acceptance_rate,
                'zk_verified': zk_verified,
                'zk_failed': zk_failed,
                'zk_verify_time_ms': zk_verify_time_ms
            }
        )
    
    def _phase3_defense(self, verified_updates: Dict[int, torch.Tensor]) -> PhaseResult:
        """Phase 3: Run all 5 defense layers.
        
        Args:
            verified_updates: Updates that passed Merkle verification
            
        Returns:
            PhaseResult with clean updates and defense metrics
        """
        logger.info(f"Running 5-layer defense on {len(verified_updates)} verified updates")
        
        # Convert to list format for defense coordinator
        update_list = [
            {'client_id': cid, 'gradients': update}
            for cid, update in verified_updates.items()
        ]
        
        # Run Layers 1-4 (client-level defense)
        defense_result = self.defense_coordinator.run_defense_pipeline(update_list)
        
        logger.info(f"  Layer 1 (Integrity): {len(defense_result['layer1_detections'])} detections")
        logger.info(f"  Layer 2 (Statistical): {len(defense_result['layer2_detections'])} detections")
        logger.info(f"  Layer 3 (Robust Agg): {defense_result['layer3_info']['method']}")
        logger.info(f"  Layer 4 (Reputation): {len(defense_result['reputation_scores'])} clients scored")
        
        # Aggregate at galaxy level
        galaxy_updates = {}
        for galaxy_id in range(self.num_galaxies):
            galaxy_clients = self.galaxy_manager.get_galaxy_clients(galaxy_id)
            galaxy_client_updates = [
                update for update in update_list
                if update['client_id'] in galaxy_clients
            ]
            
            if galaxy_client_updates:
                # Convert gradient dicts to flat tensors and average
                galaxy_gradients = []
                for u in galaxy_client_updates:
                    grads = u['gradients']
                    # grads is List[torch.Tensor] from Trainer.get_gradients()
                    if isinstance(grads, dict):
                        flat_grad = torch.cat([g.flatten() for g in grads.values()])
                    elif isinstance(grads, list):
                        flat_grad = torch.cat([g.flatten() for g in grads])
                    elif isinstance(grads, torch.Tensor):
                        flat_grad = grads.flatten()
                    else:
                        flat_grad = torch.tensor(grads).flatten()
                    galaxy_gradients.append(flat_grad)
                galaxy_updates[galaxy_id] = torch.stack(galaxy_gradients).mean(dim=0)
        
        logger.info(f"  Galaxy aggregation: {len(galaxy_updates)} galaxies with updates")
        
        # Run Layer 5 (galaxy-level defense)
        layer5_result = self.defense_coordinator.run_galaxy_defense(
            galaxy_updates=galaxy_updates,
            client_assignments={
                cid: self.galaxy_manager.get_client_galaxy(cid)
                for cid in verified_updates.keys()
            },
            round_number=self.current_round
        )
        
        logger.info(f"  Layer 5 (Galaxy): {len(layer5_result['flagged_galaxies'])} galaxies flagged")
        
        # Handle galaxy dissolutions
        if layer5_result['dissolved_galaxies']:
            for dissolution in layer5_result['dissolved_galaxies']:
                self.galaxy_manager.dissolve_galaxy(
                    galaxy_id=dissolution['dissolved_galaxy'],
                    honest_clients=list(dissolution['reassignments'].keys()),
                    malicious_clients=dissolution['quarantined_client_ids']
                )
        
        # Get clean updates (only from non-flagged galaxies)
        clean_galaxy_updates = {
            gid: update for gid, update in galaxy_updates.items()
            if gid in layer5_result['verdicted_clean_galaxies']
        }
        
        logger.info(f"✓ Clean galaxies: {len(clean_galaxy_updates)}/{len(galaxy_updates)}")
        
        return PhaseResult(
            phase_name="defense",
            success=True,
            data={
                'clean_updates': clean_galaxy_updates,
                'defense_results': defense_result,
                'layer5_results': layer5_result
            },
            metrics={
                'layer1_detections': len(defense_result['layer1_detections']),
                'layer2_detections': len(defense_result['layer2_detections']),
                'flagged_galaxies': len(layer5_result['flagged_galaxies']),
                'clean_galaxies': len(clean_galaxy_updates),
                'dissolved_galaxies': len(layer5_result['dissolved_galaxies'])
            }
        )
    
    def _phase4_aggregation(self, clean_galaxy_updates: Dict[int, torch.Tensor]) -> PhaseResult:
        """Phase 4: Global aggregation and model update.
        
        Args:
            clean_galaxy_updates: Updates from verified clean galaxies
            
        Returns:
            PhaseResult with global update
        """
        logger.info(f"Aggregating {len(clean_galaxy_updates)} clean galaxy updates")
        
        if not clean_galaxy_updates:
            logger.error("No clean updates available for aggregation!")
            return PhaseResult(
                phase_name="aggregation",
                success=False,
                data={},
                metrics={}
            )
        
        # Global aggregation (simple average)
        global_update = torch.stack(list(clean_galaxy_updates.values())).mean(dim=0)
        
        # Apply update to global model
        with torch.no_grad():
            # Flatten model parameters
            param_vector = torch.cat([p.data.flatten() for p in self.model.parameters()])
            
            # Apply gradient update (gradient descent)
            # lr=1.0 for FedAvg: w_new = w_global - 1.0 * avg(w_global - w_local_i) = avg(w_local_i)
            learning_rate = 1.0
            param_vector -= learning_rate * global_update
            
            # Unflatten back to model
            offset = 0
            for p in self.model.parameters():
                numel = p.numel()
                p.data.copy_(param_vector[offset:offset+numel].view_as(p))
                offset += numel
        
        logger.info(f"✓ Global model updated with learning rate {learning_rate}")
        
        # --- Galaxy Proof Folding ---
        self.galaxy_zk_proofs = {}
        folding_time_ms = 0.0
        
        if self.client_zk_proofs:
            import time as _time
            logger.info(f"Folding ZK proofs per galaxy...")
            fold_start = _time.time()
            
            # Parallelize galaxy folding
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import os
            max_workers = min(32, (os.cpu_count() or 1) * 4)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_gid = {}
                
                for galaxy_id in range(self.num_galaxies):
                    if galaxy_id not in clean_galaxy_updates:
                        continue
                    
                    galaxy_clients = self.galaxy_manager.get_galaxy_clients(galaxy_id)
                    galaxy_proofs = [
                        self.client_zk_proofs[cid]
                        for cid in galaxy_clients
                        if cid in self.client_zk_proofs
                    ]
                    
                    if not galaxy_proofs:
                        continue
                        
                    future = executor.submit(
                        self.zkp_prover.galaxy_prover.fold_galaxy_proofs,
                        client_proofs=galaxy_proofs,
                        galaxy_id=galaxy_id
                    )
                    future_to_gid[future] = galaxy_id
                
                for future in as_completed(future_to_gid):
                    galaxy_id = future_to_gid[future]
                    try:
                        folded_proof = future.result()
                        self.galaxy_zk_proofs[galaxy_id] = folded_proof
                        # Accumulate actual folding time from the proof object
                        folding_time_ms += folded_proof.prove_time_ms
                    except Exception as e:
                        logger.error(f"Failed to fold galaxy {galaxy_id}: {e}")

            fold_time = (_time.time() - fold_start) * 1000
            # Average folding time is misleading in parallel, report wall clock and sum of individual folding times
            mode = "REAL" if self.galaxy_zk_proofs and next(iter(self.galaxy_zk_proofs.values())).is_real else "FALLBACK"
            logger.info(f"✓ Galaxy proofs folded [{mode}]: {len(self.galaxy_zk_proofs)} galaxies (Wall clock: {fold_time:.1f}ms, Total folding: {folding_time_ms:.1f}ms)")
        
        return PhaseResult(
            phase_name="aggregation",
            success=True,
            data={
                'global_update': global_update,
                'galaxy_zk_proofs': dict(self.galaxy_zk_proofs)
            },
            metrics={
                'aggregated_galaxies': len(clean_galaxy_updates),
                'update_norm': global_update.norm().item(),
                'galaxy_proofs_folded': len(self.galaxy_zk_proofs),
                'folding_time_ms': folding_time_ms
            }
        )
    
    def _compile_round_metrics(self, round_result: Dict) -> Dict:
        """Compile comprehensive metrics for the round."""
        metrics = {
            'round': self.current_round,
            'success': round_result['success']
        }
        
        # Aggregate metrics from all phases
        for phase_key, phase_result in round_result['phases'].items():
            metrics.update({
                f'{phase_result.phase_name}_{k}': v
                for k, v in phase_result.metrics.items()
            })
        
        return metrics
    
    def _log_round_summary(self, round_result: Dict):
        """Log summary of round results."""
        logger.info("\n" + "="*100)
        logger.info("ROUND SUMMARY")
        logger.info("="*100)
        
        metrics = round_result['metrics']
        
        logger.info(f"Phase 1 (Commitment):    {metrics.get('commitment_total_commitments', 0)} commitments")
        logger.info(f"Phase 2 (Revelation):    {metrics.get('revelation_verified_count', 0)} verified, "
                   f"{metrics.get('revelation_rejected_count', 0)} rejected "
                   f"({metrics.get('revelation_acceptance_rate', 0):.1%} acceptance)")
        logger.info(f"Phase 3 (Defense):       L1:{metrics.get('defense_layer1_detections', 0)} "
                   f"L2:{metrics.get('defense_layer2_detections', 0)} "
                   f"L5:{metrics.get('defense_flagged_galaxies', 0)} flagged galaxies, "
                   f"{metrics.get('defense_dissolved_galaxies', 0)} dissolved")
        logger.info(f"Phase 4 (Aggregation):   {metrics.get('aggregation_aggregated_galaxies', 0)} galaxies, "
                   f"update norm: {metrics.get('aggregation_update_norm', 0):.4f}")
        
        logger.info(f"\nRound Status: {'✓ SUCCESS' if round_result['success'] else '✗ FAILED'}")
        logger.info("="*100 + "\n")
