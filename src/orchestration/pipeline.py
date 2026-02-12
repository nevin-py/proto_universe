"""End-to-End ProtoGalaxy FL Pipeline

Implements the complete architecture flow:
1. Clients compute gradients and generate commitments
2. Clients send gradients + hashes to Galaxy Aggregator
3. Galaxy builds Merkle tree and generates proofs
4. Galaxy runs defense pipeline (statistical + Krum)
5. Galaxy sends aggregated update + Merkle root to Global
6. Global builds galaxy-level Merkle tree
7. Global runs final defense and aggregation
8. Global distributes new model to all clients

This implements the architecture from protogalaxy_architecture.md Section 3.4.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

from src.client.trainer import Trainer
from src.client.commitment import CommitmentGenerator
from src.client.verifier import ProofVerifier
from src.crypto.merkle_adapter import GalaxyMerkleTreeAdapter, GlobalMerkleTreeAdapter
from src.crypto.utils import generate_timestamp, generate_nonce_hex
from src.crypto.merkle import compute_hash
from src.defense.coordinator import DefenseCoordinator
from src.defense.statistical import StatisticalAnalyzer
from src.aggregators.galaxy import GalaxyAggregator
from src.aggregators.global_agg import GlobalAggregator
from src.orchestration.model_sync import ModelSynchronizer
from src.storage.manager import StorageManager
from src.logging import FLLogger


@dataclass
class ClientSubmission:
    """Client's gradient submission with cryptographic commitment"""
    client_id: int
    galaxy_id: int
    round_number: int
    gradients: List[torch.Tensor]
    commitment_hash: str
    metadata: Dict[str, Any]
    timestamp: str = field(default_factory=generate_timestamp)
    nonce: str = field(default_factory=lambda: generate_nonce_hex(16))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'client_id': self.client_id,
            'galaxy_id': self.galaxy_id,
            'round_number': self.round_number,
            'gradients': self.gradients,
            'commitment_hash': self.commitment_hash,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'nonce': self.nonce
        }


@dataclass
class GalaxySubmission:
    """Galaxy's aggregated submission to global aggregator"""
    galaxy_id: int
    round_number: int
    aggregated_gradients: List[torch.Tensor]
    merkle_root: str
    num_clients: int
    client_ids: List[int]
    flagged_clients: List[int]
    defense_report: Dict[str, Any]
    timestamp: str = field(default_factory=generate_timestamp)


class ProtoGalaxyPipeline:
    """
    Complete end-to-end ProtoGalaxy FL pipeline.
    
    Implements the architecture flow from Section 3:
    - Phase 1: Commitment (clients → galaxy → global)
    - Phase 2: Revelation (clients send gradients with proofs)
    - Phase 3: Multi-Layer Defense (4 defense layers)
    - Phase 4: Global Aggregation (final model update)
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        num_galaxies: int,
        defense_config: Optional[Dict] = None,
        logger: Optional[FLLogger] = None,
        storage_manager: Optional[StorageManager] = None
    ):
        """
        Initialize ProtoGalaxy pipeline.
        
        Args:
            global_model: The global PyTorch model
            num_clients: Total number of clients
            num_galaxies: Number of galaxies
            defense_config: Defense coordinator configuration
            logger: FL logger instance
            storage_manager: Storage manager for checkpoints
        """
        self.global_model = global_model
        self.num_clients = num_clients
        self.num_galaxies = num_galaxies
        self.current_round = 0
        
        # Initialize components
        self.logger = logger
        self.storage = storage_manager or StorageManager()
        
        # Galaxy assignment (round-robin for simplicity)
        self.galaxy_assignments = self._assign_clients_to_galaxies()
        
        # Initialize galaxy aggregators with Merkle trees
        self.galaxy_aggregators: Dict[int, GalaxyAggregator] = {}
        self.galaxy_merkle_trees: Dict[int, GalaxyMerkleTreeAdapter] = {}
        self.galaxy_defense_coordinators: Dict[int, DefenseCoordinator] = {}
        
        for galaxy_id in range(num_galaxies):
            clients_in_galaxy = self.galaxy_assignments[galaxy_id]
            self.galaxy_aggregators[galaxy_id] = GalaxyAggregator(
                galaxy_id=galaxy_id,
                num_clients=len(clients_in_galaxy)
            )
            self.galaxy_merkle_trees[galaxy_id] = GalaxyMerkleTreeAdapter(galaxy_id)
            self.galaxy_defense_coordinators[galaxy_id] = DefenseCoordinator(
                num_clients=len(clients_in_galaxy),
                num_galaxies=num_galaxies,
                config=defense_config
            )
        
        # Initialize global aggregator
        self.global_aggregator = GlobalAggregator(num_galaxies=num_galaxies)
        self.global_merkle_tree = GlobalMerkleTreeAdapter()
        
        # Global defense coordinator
        self.global_defense = DefenseCoordinator(
            num_clients=num_galaxies,  # Treat galaxies as "super-clients"
            num_galaxies=1,
            config=defense_config
        )
        
        # Model synchronizer
        self.model_sync = ModelSynchronizer(global_model)
        
        # Client commitment generators and proof verifier
        self.commitment_generators: Dict[int, CommitmentGenerator] = {
            cid: CommitmentGenerator(client_id=cid)
            for cid in range(num_clients)
        }
        self.proof_verifier = ProofVerifier()
        
        # Storage for current round
        self.round_commitments: Dict[int, Dict[int, str]] = {}  # galaxy_id -> {client_id -> hash}
        self.round_submissions: Dict[int, Dict[int, ClientSubmission]] = {}  # galaxy_id -> {client_id -> submission}
        self.round_galaxy_submissions: Dict[int, GalaxySubmission] = {}  # galaxy_id -> submission
        self.round_commitment_objects: Dict[int, Any] = {}  # client_id -> GradientCommitment
        
        # Statistics
        self.round_stats = []
        
        if self.logger:
            self.logger.info("ProtoGalaxy pipeline initialized", extra={
                'num_clients': num_clients,
                'num_galaxies': num_galaxies
            })
    
    def _assign_clients_to_galaxies(self) -> Dict[int, List[int]]:
        """Assign clients to galaxies using round-robin (PROTO-801)"""
        assignments = {i: [] for i in range(self.num_galaxies)}
        for client_id in range(self.num_clients):
            galaxy_id = client_id % self.num_galaxies
            assignments[galaxy_id].append(client_id)
        return assignments
    
    # =========================================================================
    # Phase 1: Commitment Phase
    # =========================================================================
    
    def phase1_client_commitment(
        self,
        client_id: int,
        gradients: List[torch.Tensor],
        round_number: int
    ) -> Tuple[str, Dict]:
        """
        Phase 1: Client generates commitment for gradients.
        
        Args:
            client_id: Client identifier
            gradients: List of gradient tensors
            round_number: Current FL round
            
        Returns:
            Tuple of (commitment_hash, metadata)
        """
        # Use CommitmentGenerator to produce a proper GradientCommitment
        gen = self.commitment_generators.get(client_id)
        if gen is None:
            gen = CommitmentGenerator(client_id=client_id)
            self.commitment_generators[client_id] = gen
        
        commitment_obj = gen.generate_commitment(gradients, round_number=round_number)
        commitment_hash = commitment_obj.commitment
        
        # Store commitment object for later verification in Phase 2
        self.round_commitment_objects[client_id] = commitment_obj
        
        metadata = commitment_obj.get_metadata()
        
        if self.logger:
            self.logger.debug(f"Client {client_id} generated commitment", extra={
                'commitment_hash': commitment_hash[:16],
                'round': round_number
            })
        
        return commitment_hash, metadata
    
    def phase1_galaxy_collect_commitments(
        self,
        galaxy_id: int,
        commitments: Dict[int, str],
        round_number: int = 0
    ) -> str:
        """
        Phase 1: Galaxy collects commitments and builds Merkle tree.
        
        Args:
            galaxy_id: Galaxy identifier
            commitments: Dict mapping client_id to commitment_hash
            round_number: Current round number
            
        Returns:
            Galaxy Merkle root hash
        """
        # Store commitments
        self.round_commitments[galaxy_id] = commitments
        
        # Build Merkle tree from commitments
        commitment_list = []
        for client_id in sorted(commitments.keys()):
            commitment_list.append({
                'client_id': client_id,
                'commitment_hash': commitments[client_id]
            })
        
        galaxy_root = self.galaxy_merkle_trees[galaxy_id].build_from_commitments(
            commitment_list,
            round_number
        )
        
        if self.logger:
            self.logger.info(f"Galaxy {galaxy_id} built Merkle tree", extra={
                'galaxy_id': galaxy_id,
                'merkle_root': galaxy_root[:16],
                'num_clients': len(commitments)
            })
        
        return galaxy_root
    
    def phase1_global_collect_galaxy_roots(
        self,
        galaxy_roots: Dict[int, str],
        round_number: int = 0
    ) -> str:
        """
        Phase 1: Global aggregator collects galaxy roots and builds global tree.
        
        Args:
            galaxy_roots: Dict mapping galaxy_id to merkle_root
            round_number: Current round number
            
        Returns:
            Global Merkle root hash
        """
        # Build global Merkle tree from galaxy roots
        galaxy_root_list = []
        for galaxy_id in sorted(galaxy_roots.keys()):
            galaxy_root_list.append({
                'galaxy_id': galaxy_id,
                'merkle_root': galaxy_roots[galaxy_id]
            })
        
        global_root = self.global_merkle_tree.build_from_galaxy_roots(
            galaxy_root_list,
            round_number
        )
        
        if self.logger:
            self.logger.info("Global Merkle tree built", extra={
                'global_root': global_root[:16],
                'num_galaxies': len(galaxy_roots)
            })
        
        return global_root
    
    # =========================================================================
    # Phase 2: Revelation Phase
    # =========================================================================
    
    def phase2_client_submit_gradients(
        self,
        client_id: int,
        galaxy_id: int,
        gradients: List[torch.Tensor],
        commitment_hash: str,
        metadata: Dict,
        round_number: int
    ) -> ClientSubmission:
        """
        Phase 2: Client submits actual gradients with Merkle proof.
        
        Args:
            client_id: Client identifier
            galaxy_id: Galaxy this client belongs to
            gradients: Actual gradient tensors
            commitment_hash: Previously generated commitment
            metadata: Commitment metadata
            round_number: Current FL round
            
        Returns:
            ClientSubmission object
        """
        submission = ClientSubmission(
            client_id=client_id,
            galaxy_id=galaxy_id,
            round_number=round_number,
            gradients=gradients,
            commitment_hash=commitment_hash,
            metadata=metadata
        )
        
        return submission
    
    def phase2_galaxy_verify_and_collect(
        self,
        galaxy_id: int,
        submissions: Dict[int, ClientSubmission]
    ) -> Tuple[List[Dict], List[int]]:
        """
        Phase 2: Galaxy verifies submissions against commitments.
        
        Args:
            galaxy_id: Galaxy identifier
            submissions: Dict mapping client_id to ClientSubmission
            
        Returns:
            Tuple of (verified_updates, rejected_client_ids)
        """
        verified_updates = []
        rejected_clients = []
        
        merkle_tree = self.galaxy_merkle_trees[galaxy_id]
        defense_coordinator = self.galaxy_defense_coordinators[galaxy_id]
        
        for client_id, submission in submissions.items():
            # ============================================================
            # Gap 3: Quarantine/Ban enforcement (Architecture Section 4.4)
            # Reject quarantined or banned clients BEFORE accepting them.
            # ============================================================
            if not defense_coordinator.layer4.is_active(client_id):
                rejected_clients.append(client_id)
                status = defense_coordinator.layer4.get_client_status(client_id)
                if self.logger:
                    self.logger.warning(
                        f"Client {client_id} rejected: status={status}"
                    )
                continue
            
            # Verify against commitment
            expected_hash = self.round_commitments[galaxy_id].get(client_id)
            
            if expected_hash is None:
                rejected_clients.append(client_id)
                if self.logger:
                    self.logger.warning(f"No commitment found for client {client_id}")
                continue
            
            # Verify Merkle proof using ProofVerifier
            proof = merkle_tree.get_proof(client_id)
            root = merkle_tree.get_root()
            if root and proof is not None:
                index = merkle_tree.client_ids.index(client_id) if client_id in merkle_tree.client_ids else -1
                leaf_hash = submission.commitment_hash
                is_valid = self.proof_verifier.verify_proof(
                    root, proof, leaf_hash, leaf_index=index
                )
            else:
                is_valid = False
            
            if not is_valid:
                rejected_clients.append(client_id)
                if self.logger:
                    self.logger.warning(f"Invalid Merkle proof for client {client_id}")
                continue
            
            # ============================================================
            # Gap 2: Trust-weighted aggregation (Architecture Section 4.4)
            # Weight gradients by reputation: w_bar = Sum(R_i * grad_i) / Sum(R_i)
            # Apply reputation weight before passing to aggregation.
            # ============================================================
            reputation = defense_coordinator.layer4.get_reputation(client_id)
            weighted_gradients = []
            for g in submission.gradients:
                weighted_gradients.append(g * reputation)
            
            # Store verified submission with weighted gradients
            verified_updates.append({
                'client_id': client_id,
                'gradients': weighted_gradients,
                'metadata': submission.metadata,
                'reputation': reputation
            })
        
        # Store for this round
        self.round_submissions[galaxy_id] = submissions
        
        if self.logger:
            self.logger.info(f"Galaxy {galaxy_id} verified submissions", extra={
                'verified': len(verified_updates),
                'rejected': len(rejected_clients)
            })
        
        return verified_updates, rejected_clients
    
    # =========================================================================
    # Phase 3: Multi-Layer Defense
    # =========================================================================
    
    def phase3_galaxy_defense_pipeline(
        self,
        galaxy_id: int,
        verified_updates: List[Dict]
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Phase 3: Galaxy runs 4-layer defense pipeline.
        
        Layers:
        1. Cryptographic integrity (already done in Phase 2)
        2. Statistical anomaly detection
        3. Byzantine-robust aggregation (Krum/Trimmed Mean)
        4. Reputation-based filtering
        
        Args:
            galaxy_id: Galaxy identifier
            verified_updates: List of verified client updates
            
        Returns:
            Tuple of (aggregated_gradients, defense_report)
        """
        defense_coordinator = self.galaxy_defense_coordinators[galaxy_id]
        
        # Run defense pipeline
        defense_result = defense_coordinator.run_defense_pipeline(verified_updates)
        
        # Extract aggregated gradients from TrimmedMeanAggregator output.
        # The aggregator returns {'gradients': flat_np_array, 'trimmed_counts': ...}
        # We need to reconstruct a list of tensors with proper shapes.
        agg_raw = defense_result['layer3_aggregation']
        if agg_raw is not None and isinstance(agg_raw, dict) and 'gradients' in agg_raw:
            flat_grads = agg_raw['gradients']
            if isinstance(flat_grads, np.ndarray):
                # Reconstruct tensor list from flattened array using first client's shapes
                first_update_grads = verified_updates[0]['gradients']
                reconstructed = []
                offset = 0
                for g in first_update_grads:
                    if isinstance(g, torch.Tensor):
                        shape = g.shape
                    else:
                        shape = np.array(g).shape
                    numel = int(np.prod(shape))
                    chunk = flat_grads[offset:offset + numel]
                    reconstructed.append(torch.tensor(chunk.reshape(shape), dtype=torch.float32))
                    offset += numel
                aggregated_gradients = reconstructed
            else:
                aggregated_gradients = flat_grads
        else:
            aggregated_gradients = agg_raw
        
        # Get flagged clients
        flagged_clients = list(set(
            defense_result.get('layer1_detections', []) +
            defense_result.get('layer2_detections', [])
        ))
        
        defense_report = {
            'layer1_detections': defense_result.get('layer1_detections', []),
            'layer2_detections': defense_result.get('layer2_detections', []),
            'statistical_flagged': defense_result.get('statistical_flagged', []),
            'flagged_clients': flagged_clients,
            'aggregation_method': defense_result.get('layer3_info', {}).get('method', 'unknown'),
            'reputation_scores': defense_result.get('reputation_scores', {})
        }
        
        if self.logger:
            self.logger.log_detection(
                detected_clients=[str(c) for c in flagged_clients],
                detection_type='multi_layer',
                layer=f'galaxy_{galaxy_id}'
            )
        
        return aggregated_gradients, defense_report
    
    def phase3_galaxy_submit_to_global(
        self,
        galaxy_id: int,
        aggregated_gradients: List[torch.Tensor],
        defense_report: Dict,
        client_ids: List[int]
    ) -> GalaxySubmission:
        """
        Phase 3: Galaxy submits aggregated update to global aggregator.
        
        Args:
            galaxy_id: Galaxy identifier
            aggregated_gradients: Defense-filtered aggregated gradients
            defense_report: Defense pipeline results
            client_ids: List of client IDs that participated
            
        Returns:
            GalaxySubmission object
        """
        # Get Merkle root for this galaxy
        merkle_root = self.galaxy_merkle_trees[galaxy_id].get_root()
        
        submission = GalaxySubmission(
            galaxy_id=galaxy_id,
            round_number=self.current_round,
            aggregated_gradients=aggregated_gradients,
            merkle_root=merkle_root,
            num_clients=len(client_ids),
            client_ids=client_ids,
            flagged_clients=defense_report.get('flagged_clients', []),
            defense_report=defense_report
        )
        
        self.round_galaxy_submissions[galaxy_id] = submission
        
        return submission
    
    # =========================================================================
    # Phase 4: Global Aggregation
    # =========================================================================
    
    def phase4_global_verify_galaxies(
        self,
        galaxy_submissions: Dict[int, GalaxySubmission]
    ) -> Tuple[List[Dict], List[int]]:
        """
        Phase 4: Global verifies galaxy submissions against Merkle tree.
        
        Args:
            galaxy_submissions: Dict mapping galaxy_id to GalaxySubmission
            
        Returns:
            Tuple of (verified_galaxy_updates, rejected_galaxy_ids)
        """
        verified_updates = []
        rejected_galaxies = []
        
        for galaxy_id, submission in galaxy_submissions.items():
            # Verify Merkle root matches
            proof = self.global_merkle_tree.get_galaxy_proof(galaxy_id)
            is_valid = self.global_merkle_tree.verify_galaxy_proof(
                galaxy_id,
                submission.merkle_root,
                proof
            )
            
            if not is_valid:
                rejected_galaxies.append(galaxy_id)
                if self.logger:
                    self.logger.warning(f"Invalid Merkle proof for galaxy {galaxy_id}")
                continue
            
            verified_updates.append({
                'galaxy_id': galaxy_id,
                'gradients': submission.aggregated_gradients,
                'num_clients': submission.num_clients,
                'defense_report': submission.defense_report
            })
        
        if self.logger:
            self.logger.info("Global verified galaxy submissions", extra={
                'verified': len(verified_updates),
                'rejected': len(rejected_galaxies)
            })
        
        return verified_updates, rejected_galaxies
    
    def phase4_layer5_galaxy_defense(
        self,
        verified_galaxy_updates: List[Dict]
    ) -> Dict:
        """
        Phase 4: Run Layer 5 galaxy-level defense (Architecture Section 4.5).
        
        Treats each galaxy's aggregated update as a "super-client" and applies:
        - Galaxy anomaly detection (norm, direction, cross-galaxy consistency)
        - Galaxy reputation updates
        - Adaptive re-clustering for compromised galaxies
        - Forensic logging for quarantine/ban decisions
        
        Args:
            verified_galaxy_updates: List of verified galaxy updates
            
        Returns:
            Layer 5 defense results
        """
        # Build galaxy_updates dict for Layer 5
        galaxy_updates = {}
        for update in verified_galaxy_updates:
            galaxy_updates[update['galaxy_id']] = update['gradients']
        
        # Build client assignments reverse map
        client_assignments = {}
        for galaxy_id, client_ids in self.galaxy_assignments.items():
            for cid in client_ids:
                client_assignments[cid] = galaxy_id
        
        # Run Layer 5: Galaxy-level defense with forensic logging
        layer5_result = self.global_defense.run_galaxy_defense(
            galaxy_updates=galaxy_updates,
            client_assignments=client_assignments,
            round_number=self.current_round
        )
        
        return layer5_result
    
    def phase4_global_defense_and_aggregate(
        self,
        verified_galaxy_updates: List[Dict],
        layer5_result: Optional[Dict] = None
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Phase 4: Global runs defense pipeline and final aggregation.
        
        Args:
            verified_galaxy_updates: List of verified galaxy updates
            layer5_result: Optional Layer 5 defense results for filtering
            
        Returns:
            Tuple of (global_gradients, defense_report)
        """
        # Filter out galaxies flagged by Layer 5
        if layer5_result and layer5_result.get('flagged_galaxies'):
            flagged_gids = set(layer5_result['flagged_galaxies'])
            filtered_updates = [
                u for u in verified_galaxy_updates
                if u['galaxy_id'] not in flagged_gids
            ]
            if filtered_updates:
                verified_galaxy_updates = filtered_updates
        
        # Run global defense pipeline (treat galaxies as super-clients)
        defense_result = self.global_defense.run_defense_pipeline(verified_galaxy_updates)
        
        # Extract final aggregated gradients (same unwrapping as Phase 3)
        agg_raw = defense_result['layer3_aggregation']
        if agg_raw is not None and isinstance(agg_raw, dict) and 'gradients' in agg_raw:
            flat_grads = agg_raw['gradients']
            if isinstance(flat_grads, np.ndarray) and verified_galaxy_updates:
                first_grads = verified_galaxy_updates[0]['gradients']
                global_gradients = []
                offset = 0
                for g in first_grads:
                    if isinstance(g, torch.Tensor):
                        shape = g.shape
                    else:
                        shape = np.array(g).shape
                    numel = int(np.prod(shape))
                    chunk = flat_grads[offset:offset + numel]
                    global_gradients.append(torch.tensor(chunk.reshape(shape), dtype=torch.float32))
                    offset += numel
            else:
                global_gradients = flat_grads
        else:
            global_gradients = agg_raw
        
        defense_report = {
            'flagged_galaxies': list(set(
                defense_result.get('layer1_detections', []) +
                defense_result.get('layer2_detections', [])
            )),
            'aggregation_method': defense_result.get('layer3_info', {}).get('method', 'unknown'),
            'galaxy_reputations': defense_result.get('reputation_scores', {})
        }
        
        # Merge Layer 5 results if available
        if layer5_result:
            defense_report['layer5_flagged_galaxies'] = layer5_result.get('flagged_galaxies', [])
            defense_report['layer5_anomaly_reports'] = {
                gid: {
                    'is_anomalous': report.is_anomalous,
                    'norm_deviation': report.norm_deviation_score,
                    'direction_similarity': report.direction_similarity,
                    'consistency': report.cross_galaxy_consistency
                }
                for gid, report in layer5_result.get('anomaly_reports', {}).items()
            }
            defense_report['layer5_dissolutions'] = layer5_result.get('dissolutions', [])
        
        if self.logger:
            self.logger.log_aggregation(
                num_gradients=len(verified_galaxy_updates),
                method=defense_report['aggregation_method'],
                galaxy_id=None
            )
        
        return global_gradients, defense_report
    
    def phase4_update_global_model(
        self,
        global_gradients: List[torch.Tensor],
        learning_rate: float = 1.0
    ):
        """
        Phase 4: Apply global gradients to update global model.
        
        Args:
            global_gradients: Aggregated gradients from all galaxies
            learning_rate: Learning rate for update
        """
        self.model_sync.apply_update(global_gradients, learning_rate)
        
        if self.logger:
            self.logger.info(f"Global model updated for round {self.current_round}")
    
    def phase4_distribute_model(self) -> Dict:
        """
        Phase 4: Distribute updated global model to all clients.
        
        Returns:
            Sync package for distribution
        """
        sync_package = self.model_sync.get_sync_package(self.current_round)
        
        if self.logger:
            self.logger.info("Model distributed to clients", extra={
                'round': self.current_round,
                'model_hash': sync_package['model_hash'][:16]
            })
        
        return sync_package
    
    # =========================================================================
    # Complete Round Execution
    # =========================================================================
    
    def execute_round(
        self,
        client_trainers: Dict[int, Trainer],
        round_number: int
    ) -> Dict:
        """
        Execute a complete FL round through all 4 phases.
        
        Args:
            client_trainers: Dict mapping client_id to Trainer instance
            round_number: Current round number
            
        Returns:
            Round statistics and results
        """
        self.current_round = round_number
        round_start_time = time.time()
        
        if self.logger:
            self.logger.log_round_start(round_number, len(client_trainers))
        
        # =========================
        # PHASE 1: COMMITMENT
        # =========================
        
        # Step 1a: Clients generate commitments
        client_commitments = {}  # {galaxy_id: {client_id: hash}}
        client_gradients = {}  # {client_id: gradients}
        client_metadata = {}  # {client_id: metadata}
        
        for client_id, trainer in client_trainers.items():
            # Train locally and get gradients
            gradients = trainer.get_gradients()
            client_gradients[client_id] = gradients
            
            # Generate commitment
            commit_hash, metadata = self.phase1_client_commitment(
                client_id, gradients, round_number
            )
            client_metadata[client_id] = metadata
            
            # Group by galaxy
            galaxy_id = client_id % self.num_galaxies
            if galaxy_id not in client_commitments:
                client_commitments[galaxy_id] = {}
            client_commitments[galaxy_id][client_id] = commit_hash
        
        # Step 1b: Galaxies build Merkle trees
        galaxy_roots = {}
        for galaxy_id, commitments in client_commitments.items():
            root = self.phase1_galaxy_collect_commitments(galaxy_id, commitments)
            galaxy_roots[galaxy_id] = root
        
        # Step 1c: Global builds Merkle tree
        global_root = self.phase1_global_collect_galaxy_roots(galaxy_roots)
        
        # =========================
        # PHASE 2: REVELATION
        # =========================
        
        # Step 2a: Clients submit gradients
        galaxy_submissions_phase2 = {}  # {galaxy_id: {client_id: submission}}
        
        for client_id, gradients in client_gradients.items():
            galaxy_id = client_id % self.num_galaxies
            commit_hash = client_commitments[galaxy_id][client_id]
            metadata = client_metadata[client_id]
            
            submission = self.phase2_client_submit_gradients(
                client_id, galaxy_id, gradients, commit_hash, metadata, round_number
            )
            
            if galaxy_id not in galaxy_submissions_phase2:
                galaxy_submissions_phase2[galaxy_id] = {}
            galaxy_submissions_phase2[galaxy_id][client_id] = submission
        
        # Step 2b: Galaxies verify and collect
        galaxy_verified_updates = {}
        galaxy_rejected_clients = {}
        
        for galaxy_id, submissions in galaxy_submissions_phase2.items():
            verified, rejected = self.phase2_galaxy_verify_and_collect(
                galaxy_id, submissions
            )
            galaxy_verified_updates[galaxy_id] = verified
            galaxy_rejected_clients[galaxy_id] = rejected
        
        # =========================
        # PHASE 3: DEFENSE
        # =========================
        
        # Step 3: Galaxies run defense pipelines
        galaxy_final_submissions = {}
        
        for galaxy_id, verified_updates in galaxy_verified_updates.items():
            if not verified_updates:
                continue
                
            aggregated_grads, defense_report = self.phase3_galaxy_defense_pipeline(
                galaxy_id, verified_updates
            )
            
            client_ids = [u['client_id'] for u in verified_updates]
            
            galaxy_submission = self.phase3_galaxy_submit_to_global(
                galaxy_id, aggregated_grads, defense_report, client_ids
            )
            
            galaxy_final_submissions[galaxy_id] = galaxy_submission
        
        # =========================
        # PHASE 4: GLOBAL AGGREGATION
        # =========================
        
        # Step 4a: Global verifies galaxies
        verified_galaxies, rejected_galaxies = self.phase4_global_verify_galaxies(
            galaxy_final_submissions
        )
        
        # Step 4b: Layer 5 galaxy-level defense (Architecture Section 4.5)
        layer5_result = self.phase4_layer5_galaxy_defense(verified_galaxies)
        
        # Step 4c: Global defense and aggregation (with Layer 5 filtering)
        global_gradients, global_defense_report = self.phase4_global_defense_and_aggregate(
            verified_galaxies, layer5_result=layer5_result
        )
        
        # Step 4d: Update global model
        self.phase4_update_global_model(global_gradients)
        
        # Step 4e: Distribute model
        sync_package = self.phase4_distribute_model()
        
        # =========================
        # ROUND COMPLETION
        # =========================
        
        round_time = time.time() - round_start_time
        
        round_stats = {
            'round': round_number,
            'global_root': global_root,
            'num_galaxies': len(galaxy_final_submissions),
            'total_clients': len(client_trainers),
            'verified_clients': sum(len(g.client_ids) for g in galaxy_final_submissions.values()),
            'rejected_clients': sum(len(r) for r in galaxy_rejected_clients.values()),
            'flagged_galaxies': global_defense_report.get('flagged_galaxies', []),
            'layer5_flagged': global_defense_report.get('layer5_flagged_galaxies', []),
            'layer5_dissolutions': global_defense_report.get('layer5_dissolutions', []),
            'model_hash': sync_package['model_hash'],
            'round_time': round_time
        }
        
        self.round_stats.append(round_stats)
        
        if self.logger:
            self.logger.log_round_end(
                round_num=round_number,
                duration=round_time,
                metrics=round_stats
            )
        
        return round_stats
    
    def get_round_statistics(self) -> List[Dict]:
        """Get statistics for all completed rounds"""
        return self.round_stats
    
    def save_checkpoint(self, filepath: str):
        """Save pipeline checkpoint"""
        if self.storage:
            self.storage.save_model(
                self.global_model,
                self.current_round,
                metadata={'round_stats': self.round_stats}
            )
