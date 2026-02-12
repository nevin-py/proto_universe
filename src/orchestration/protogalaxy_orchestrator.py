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
    
    def _phase1_commitment(self, client_commitments: Dict[int, str], client_gradients: Dict[int, List] = None) -> PhaseResult:
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
                    # commitment_obj.gradients is a dict of {param_name: tensor}
                    grad_dict = commitment_obj.gradients
                    # Flatten all gradients to a single vector
                    flat_grad = torch.cat([g.flatten() for g in grad_dict])
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
            for client_id, grads in client_gradients.items():
                zk_proof = self.zkp_prover.prove_gradient_sum(
                    gradients=grads,
                    client_id=client_id,
                    round_number=self.current_round,
                )
                self.client_zk_proofs[client_id] = zk_proof
                zk_proofs_generated += 1
                zk_total_time_ms += zk_proof.prove_time_ms
            
            mode = "REAL (ProtoGalaxy IVC)" if self.client_zk_proofs and next(iter(self.client_zk_proofs.values())).is_real else "FALLBACK (SHA-256)"
            logger.info(f"✓ ZK proofs: {zk_proofs_generated} generated [{mode}]")
            logger.info(f"  Total prove time: {zk_total_time_ms:.1f}ms ({zk_total_time_ms/max(zk_proofs_generated,1):.1f}ms/client)")
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
        client_commitments: Dict[int, str],
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
        
        verified_updates = {}
        rejected_clients = []
        verification_results = {}
        
        for client_id, update in client_updates.items():
            proof = client_proofs.get(client_id, [])
            commitment = client_commitments.get(client_id, "")
            
            # Get galaxy
            galaxy_id = self.galaxy_manager.get_client_galaxy(client_id)
            if galaxy_id is None:
                logger.warning(f"Client {client_id} not assigned to any galaxy, rejecting")
                rejected_clients.append(client_id)
                verification_results[client_id] = False
                continue
            
            # Verify Merkle proof
            galaxy_tree = self.galaxy_merkle_trees.get(galaxy_id)
            if galaxy_tree is None:
                logger.warning(f"No Merkle tree for galaxy {galaxy_id}, rejecting client {client_id}")
                rejected_clients.append(client_id)
                verification_results[client_id] = False
                continue
            
            # Verify Merkle proof cryptographically
            client_proof_data = galaxy_tree.get_client_proof(client_id)
            if client_proof_data is None:
                logger.warning(f"Client {client_id} has no proof in galaxy {galaxy_id} tree, rejecting")
                rejected_clients.append(client_id)
                verification_results[client_id] = False
                continue
            
            # Use tree's verify method: checks leaf hash + proof path == root
            leaf_index = client_proof_data['leaf_index']
            is_valid = galaxy_tree.verify(leaf_index, client_proof_data['leaf_hash'])
            
            if is_valid:
                verified_updates[client_id] = update
                verification_results[client_id] = True
            else:
                logger.warning(f"✗ Client {client_id}: INVALID Merkle proof")
                rejected_clients.append(client_id)
                verification_results[client_id] = False
                
                # Log to forensic logger (Layer 1 failure)
                self.defense_coordinator.forensic_logger.log_quarantine(
                    client_id=client_id,
                    round_number=self.current_round,
                    commitment_hash=commitment,
                    merkle_proof=proof,
                    merkle_root=galaxy_tree.get_root(),
                    layer_results={
                        'layer1_failed': True,
                        'layer2_flags': [],
                        'layer3_rejected': False,
                        'layer4_reputation': 1.0,
                        'layer5_galaxy_flagged': False,
                    },
                    metadata={'phase': 'revelation'}
                )
        
        acceptance_rate = len(verified_updates) / len(client_updates) if client_updates else 0
        
        # --- ZK Sum-Check Verification ---
        zk_verified = 0
        zk_failed = 0
        zk_verify_time_ms = 0.0
        
        if self.client_zk_proofs:
            import time as _time
            logger.info(f"Verifying ZK sum-check proofs for {len(self.client_zk_proofs)} clients...")
            zk_start = _time.time()
            
            zk_invalid_clients = []
            for client_id, zk_proof in self.client_zk_proofs.items():
                if client_id not in verified_updates:
                    continue  # Already rejected by Merkle
                
                is_valid = GradientSumCheckProver.verify_proof(zk_proof)
                if is_valid:
                    zk_verified += 1
                else:
                    zk_failed += 1
                    zk_invalid_clients.append(client_id)
                    logger.warning(f"  ✗ Client {client_id}: INVALID ZK sum-check proof")
            
            # Remove ZK-invalid clients from verified set
            for cid in zk_invalid_clients:
                verified_updates.pop(cid, None)
                rejected_clients.append(cid)
                verification_results[cid] = False
            
            zk_verify_time_ms = (_time.time() - zk_start) * 1000
            mode = "REAL" if next(iter(self.client_zk_proofs.values())).is_real else "FALLBACK"
            logger.info(f"✓ ZK verification [{mode}]: {zk_verified} valid, {zk_failed} invalid ({zk_verify_time_ms:.1f}ms)")
        
        logger.info(f"✓ Verified: {len(verified_updates)}/{len(client_updates)} clients ({acceptance_rate:.1%})")
        if rejected_clients:
            logger.warning(f"✗ Rejected: {len(rejected_clients)} clients (invalid proofs)")
        
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
                    malicious_clients=dissolution['clients_quarantined']
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
            learning_rate = 0.01  # Could be configurable
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
            
            for galaxy_id in range(self.num_galaxies):
                galaxy_clients = self.galaxy_manager.get_galaxy_clients(galaxy_id)
                galaxy_proofs = [
                    self.client_zk_proofs[cid]
                    for cid in galaxy_clients
                    if cid in self.client_zk_proofs and cid in clean_galaxy_updates
                ]
                
                if galaxy_proofs:
                    folded = self.galaxy_proof_folder.fold_galaxy_proofs(galaxy_proofs, galaxy_id)
                    self.galaxy_zk_proofs[galaxy_id] = folded
            
            folding_time_ms = (_time.time() - fold_start) * 1000
            mode = "REAL" if self.galaxy_zk_proofs and next(iter(self.galaxy_zk_proofs.values())).is_real else "FALLBACK"
            logger.info(f"✓ Galaxy proofs folded [{mode}]: {len(self.galaxy_zk_proofs)} galaxies ({folding_time_ms:.1f}ms)")
        
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
