"""Pipeline integration tests for ProtoGalaxy.

Verifies that the critical fixes to CommitmentGenerator, ProofVerifier,
and the type annotations work correctly through the actual pipeline code
in src/orchestration/pipeline.py.

Tests cover:
- CommitmentGenerator creates GradientCommitments with correct fields (Finding 1)
- ProofVerifier handles leaf_index correctly (Finding 2)  
- Orchestrator Phase 1 type contract with GradientCommitment objects (Finding 3)
- Full pipeline commit→build_tree→verify→defense flow
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
# Unit Tests for Fixed Modules
# =========================================================================

class TestCommitmentGeneratorFix:
    """Finding 1: CommitmentGenerator must pass client_id and round_number."""

    def test_commitment_has_client_id(self):
        """Commitment carries the correct client_id."""
        from src.client.commitment import CommitmentGenerator

        gen = CommitmentGenerator(client_id=42)
        grads = [np.random.randn(10), np.random.randn(5)]
        commitment = gen.generate_commitment(grads, round_number=7)

        assert commitment.client_id == 42
        assert commitment.round_number == 7

    def test_commitment_hash_is_generated(self):
        """Commitment hash is non-None after generate_commitment."""
        from src.client.commitment import CommitmentGenerator

        gen = CommitmentGenerator(client_id=1)
        grads = [np.array([1.0, 2.0, 3.0])]
        commitment = gen.generate_commitment(grads, round_number=0)

        assert commitment.commitment is not None
        assert len(commitment.commitment) == 64  # SHA-256 hex

    def test_verify_commitment_valid(self):
        """verify_commitment succeeds with matching gradients."""
        from src.client.commitment import CommitmentGenerator

        gen = CommitmentGenerator(client_id=5)
        grads = [np.array([1.0, 2.0, 3.0])]
        commitment = gen.generate_commitment(grads, round_number=1)

        assert gen.verify_commitment(commitment, grads)

    def test_verify_commitment_tampered(self):
        """verify_commitment rejects tampered gradients."""
        from src.client.commitment import CommitmentGenerator

        gen = CommitmentGenerator(client_id=5)
        grads = [np.array([1.0, 2.0, 3.0])]
        commitment = gen.generate_commitment(grads, round_number=1)

        fake_grads = [np.array([9.0, 9.0, 9.0])]
        assert not gen.verify_commitment(commitment, fake_grads)

    def test_commitments_differ_across_rounds(self):
        """Different round numbers produce different commitment hashes."""
        from src.client.commitment import CommitmentGenerator

        gen = CommitmentGenerator(client_id=10)
        grads = [np.array([1.0, 2.0])]
        c1 = gen.generate_commitment(grads, round_number=0)
        c2 = gen.generate_commitment(grads, round_number=1)

        assert c1.commitment != c2.commitment


class TestProofVerifierFix:
    """Finding 2: ProofVerifier must pass leaf_index to verify_proof."""

    def test_verify_proof_with_leaf_index(self):
        """Single proof verification works with leaf_index."""
        from src.client.verifier import ProofVerifier
        from src.crypto.merkle import MerkleTree

        tree = MerkleTree([b"a", b"b", b"c", b"d"])
        verifier = ProofVerifier()

        for i in range(4):
            proof = tree.get_proof(i)
            leaf_hash = tree.leaf_hashes[i]
            root = tree.get_root()
            assert verifier.verify_proof(root, proof, leaf_hash, leaf_index=i), \
                f"Leaf {i} should verify"

    def test_batch_verify_with_4_tuples(self):
        """Batch verification works with (root, proof, leaf_hash, leaf_index) tuples."""
        from src.client.verifier import ProofVerifier
        from src.crypto.merkle import MerkleTree

        tree = MerkleTree([b"x", b"y", b"z", b"w"])
        root = tree.get_root()
        verifier = ProofVerifier()

        proofs = []
        for i in range(4):
            proof = tree.get_proof(i)
            leaf_hash = tree.leaf_hashes[i]
            proofs.append((root, proof, leaf_hash, i))

        results = verifier.batch_verify(proofs)
        assert all(results.values())
        assert len(results) == 4


class TestOrchestratorTypeContract:
    """Finding 3: Orchestrator type annotation must use GradientCommitment."""

    def test_gradient_commitment_has_gradients_attr(self):
        """GradientCommitment objects have .gradients for Phase 1 access."""
        from src.crypto.merkle import GradientCommitment

        grads = [np.random.randn(10)]
        gc = GradientCommitment(gradients=grads, client_id=1, round_number=1)
        gc.commit()

        assert hasattr(gc, 'gradients')
        assert gc.gradients is grads

    def test_commitment_objects_work_with_galaxy_tree(self):
        """Commitments can be used to extract gradients for GalaxyMerkleTree."""
        from src.crypto.merkle import GradientCommitment, GalaxyMerkleTree

        commitments = {}
        for cid in range(4):
            grads = [torch.randn(10), torch.randn(5)]
            gc = GradientCommitment(gradients=grads, client_id=cid, round_number=0)
            gc.commit()
            commitments[cid] = gc

        # Extract gradients and flatten (mimics orchestrator Phase 1)
        galaxy_gradients = []
        galaxy_client_ids = []
        for cid, commitment_obj in commitments.items():
            flat_grad = torch.cat([g.flatten() for g in commitment_obj.gradients])
            galaxy_gradients.append(flat_grad)
            galaxy_client_ids.append(cid)

        tree = GalaxyMerkleTree(
            gradients=galaxy_gradients,
            galaxy_id=0,
            client_ids=galaxy_client_ids,
            round_number=0
        )

        assert tree.get_root() is not None
        assert len(tree) == 4
        for cid in galaxy_client_ids:
            proof_data = tree.get_client_proof(cid)
            assert proof_data is not None
            assert tree.verify(proof_data['leaf_index'])


# =========================================================================
# Pipeline Integration Tests
# =========================================================================

class TestPipelinePhase1:
    """Phase 1 commitment flow through pipeline.py using CommitmentGenerator."""

    def test_phase1_uses_commitment_generator(self):
        """pipeline.phase1_client_commitment uses CommitmentGenerator internally."""
        from src.orchestration.pipeline import ProtoGalaxyPipeline

        model = nn.Linear(10, 2)
        pipeline = ProtoGalaxyPipeline(
            global_model=model,
            num_clients=4,
            num_galaxies=2
        )

        grads = [torch.randn(10, 2), torch.randn(2)]
        commit_hash, metadata = pipeline.phase1_client_commitment(
            client_id=0, gradients=grads, round_number=5
        )

        assert commit_hash is not None
        assert len(commit_hash) == 64
        assert 'client_id' in metadata
        assert metadata['client_id'] == 0

        # The commitment object should also be stored
        assert 0 in pipeline.round_commitment_objects
        commitment_obj = pipeline.round_commitment_objects[0]
        assert commitment_obj.client_id == 0
        assert commitment_obj.round_number == 5

    def test_phase1_galaxy_merkle_tree(self):
        """Full Phase 1: clients commit → galaxy builds Merkle tree → global tree."""
        from src.orchestration.pipeline import ProtoGalaxyPipeline

        model = nn.Linear(10, 2)
        pipeline = ProtoGalaxyPipeline(
            global_model=model,
            num_clients=4,
            num_galaxies=2
        )

        # Clients generate commitments
        commitments_by_galaxy = {}
        for cid in range(4):
            grads = [torch.randn(10, 2), torch.randn(2)]
            commit_hash, _ = pipeline.phase1_client_commitment(cid, grads, round_number=1)

            galaxy_id = cid % 2
            if galaxy_id not in commitments_by_galaxy:
                commitments_by_galaxy[galaxy_id] = {}
            commitments_by_galaxy[galaxy_id][cid] = commit_hash

        # Galaxy builds Merkle tree
        galaxy_roots = {}
        for galaxy_id, commits in commitments_by_galaxy.items():
            root = pipeline.phase1_galaxy_collect_commitments(galaxy_id, commits, round_number=1)
            assert root is not None
            assert len(root) == 64
            galaxy_roots[galaxy_id] = root

        # Global builds Merkle tree
        global_root = pipeline.phase1_global_collect_galaxy_roots(galaxy_roots, round_number=1)
        assert global_root is not None
        assert len(global_root) == 64


class TestPipelinePhase2:
    """Phase 2 revelation flow through pipeline.py using ProofVerifier."""

    def _setup_phase1(self, pipeline, num_clients=4):
        """Run Phase 1 and return client data."""
        client_grads = {}
        commitments_by_galaxy = {}
        client_metadata = {}

        for cid in range(num_clients):
            grads = [torch.randn(10, 2), torch.randn(2)]
            client_grads[cid] = grads
            commit_hash, metadata = pipeline.phase1_client_commitment(cid, grads, round_number=1)
            client_metadata[cid] = metadata

            galaxy_id = cid % pipeline.num_galaxies
            if galaxy_id not in commitments_by_galaxy:
                commitments_by_galaxy[galaxy_id] = {}
            commitments_by_galaxy[galaxy_id][cid] = commit_hash

        galaxy_roots = {}
        for galaxy_id, commits in commitments_by_galaxy.items():
            root = pipeline.phase1_galaxy_collect_commitments(galaxy_id, commits, round_number=1)
            galaxy_roots[galaxy_id] = root

        pipeline.phase1_global_collect_galaxy_roots(galaxy_roots, round_number=1)

        return client_grads, commitments_by_galaxy, client_metadata

    def test_phase2_verify_uses_proof_verifier(self):
        """Phase 2 verification goes through ProofVerifier with leaf_index."""
        from src.orchestration.pipeline import ProtoGalaxyPipeline

        model = nn.Linear(10, 2)
        pipeline = ProtoGalaxyPipeline(
            global_model=model,
            num_clients=4,
            num_galaxies=2
        )

        client_grads, commitments_by_galaxy, client_metadata = self._setup_phase1(pipeline)

        # Phase 2: Clients submit gradients
        for galaxy_id in range(pipeline.num_galaxies):
            galaxy_submissions = {}
            galaxy_clients = commitments_by_galaxy.get(galaxy_id, {})

            for cid in galaxy_clients:
                submission = pipeline.phase2_client_submit_gradients(
                    client_id=cid,
                    galaxy_id=galaxy_id,
                    gradients=client_grads[cid],
                    commitment_hash=commitments_by_galaxy[galaxy_id][cid],
                    metadata=client_metadata[cid],
                    round_number=1
                )
                galaxy_submissions[cid] = submission

            verified, rejected = pipeline.phase2_galaxy_verify_and_collect(
                galaxy_id, galaxy_submissions
            )

            # All honest clients should verify
            assert len(verified) == len(galaxy_clients), \
                f"Galaxy {galaxy_id}: expected {len(galaxy_clients)} verified, got {len(verified)}"
            assert len(rejected) == 0

        # ProofVerifier should have recorded successful verifications
        assert len(pipeline.proof_verifier.get_verification_history()) > 0


class TestPipelinePhase3:
    """Phase 3 defense flow through pipeline.py."""

    def test_phase3_defense_accepts_verified_updates(self):
        """Phase 3 defense pipeline accepts list-of-dicts from Phase 2."""
        from src.orchestration.pipeline import ProtoGalaxyPipeline

        model = nn.Linear(10, 2)
        pipeline = ProtoGalaxyPipeline(
            global_model=model,
            num_clients=4,
            num_galaxies=2
        )

        # Simulate verified updates in the format Phase 2 produces
        verified_updates = [
            {'client_id': 0, 'gradients': [torch.randn(10, 2), torch.randn(2)], 'metadata': {}},
            {'client_id': 2, 'gradients': [torch.randn(10, 2), torch.randn(2)], 'metadata': {}},
        ]

        aggregated_grads, defense_report = pipeline.phase3_galaxy_defense_pipeline(
            galaxy_id=0, verified_updates=verified_updates
        )

        assert aggregated_grads is not None
        assert 'layer1_detections' in defense_report
        assert 'aggregation_method' in defense_report


class TestFullPipelineRound:
    """End-to-end pipeline round (Phases 1-4)."""

    def test_full_round_phases_1_through_3(self):
        """Complete pipeline flow: Phase 1 → Phase 2 → Phase 3."""
        from src.orchestration.pipeline import ProtoGalaxyPipeline

        model = nn.Linear(10, 2)
        num_clients = 4
        num_galaxies = 2
        pipeline = ProtoGalaxyPipeline(
            global_model=model,
            num_clients=num_clients,
            num_galaxies=num_galaxies
        )

        # --- Phase 1: Commitment ---
        client_grads = {}
        commitments_by_galaxy = {}
        client_metadata = {}

        for cid in range(num_clients):
            grads = [torch.randn(10, 2), torch.randn(2)]
            client_grads[cid] = grads
            commit_hash, metadata = pipeline.phase1_client_commitment(cid, grads, round_number=0)
            client_metadata[cid] = metadata

            galaxy_id = cid % num_galaxies
            if galaxy_id not in commitments_by_galaxy:
                commitments_by_galaxy[galaxy_id] = {}
            commitments_by_galaxy[galaxy_id][cid] = commit_hash

        galaxy_roots = {}
        for galaxy_id, commits in commitments_by_galaxy.items():
            root = pipeline.phase1_galaxy_collect_commitments(galaxy_id, commits, round_number=0)
            galaxy_roots[galaxy_id] = root

        global_root = pipeline.phase1_global_collect_galaxy_roots(galaxy_roots, round_number=0)
        assert global_root is not None

        # --- Phase 2: Revelation ---
        galaxy_verified = {}
        for galaxy_id in range(num_galaxies):
            subs = {}
            for cid in commitments_by_galaxy.get(galaxy_id, {}):
                sub = pipeline.phase2_client_submit_gradients(
                    cid, galaxy_id, client_grads[cid],
                    commitments_by_galaxy[galaxy_id][cid],
                    client_metadata[cid], round_number=0
                )
                subs[cid] = sub

            verified, rejected = pipeline.phase2_galaxy_verify_and_collect(galaxy_id, subs)
            assert len(rejected) == 0, f"Galaxy {galaxy_id} rejected clients unexpectedly"
            galaxy_verified[galaxy_id] = verified

        # --- Phase 3: Defense ---
        for galaxy_id, verified_updates in galaxy_verified.items():
            if verified_updates:
                agg_grads, report = pipeline.phase3_galaxy_defense_pipeline(
                    galaxy_id, verified_updates
                )
                assert agg_grads is not None
                assert 'aggregation_method' in report

        # Verify all commitment objects are GradientCommitment instances
        from src.crypto.merkle import GradientCommitment
        for cid, obj in pipeline.round_commitment_objects.items():
            assert isinstance(obj, GradientCommitment), \
                f"Client {cid}: commitment should be GradientCommitment, got {type(obj)}"
            assert obj.client_id == cid

        print("\n✓ Full pipeline Phase 1→2→3 completed successfully")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
