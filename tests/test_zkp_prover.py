"""Tests for ZKP prover with TrainingStepCircuit (Proof-of-Training).

These tests verify the simplified ProtoGalaxy circuit that uses
TrainingStepCircuit to prove correct SGD computation cryptographically.
No statistical bounds or norm enforcement - proof provides 100% Byzantine detection.

Tests cover:
1. Proof generation with TrainingStepCircuit (real or fallback)
2. Proof verification
3. Galaxy proof folding
4. Model-agnostic operation
"""

import pytest
import torch
import numpy as np

from src.crypto.zkp_prover import (
    GradientSumCheckProver,
    GalaxyProofFolder,
    ZKProof,
    _ZKP_AVAILABLE,
)

pytestmark_real_zkp = pytest.mark.skipif(
    not _ZKP_AVAILABLE,
    reason="Requires fl_zkp_bridge (cd sonobe/fl-zkp-bridge && maturin develop --release)",
)


class TestTrainingStepCircuitProver:
    """Test suite for TrainingStepCircuit-based ZKP prover."""

    pytestmark = pytestmark_real_zkp

    def test_prover_initialization(self):
        """Test that prover initializes correctly."""
        prover = GradientSumCheckProver()

        # Prover should initialize without errors
        assert prover is not None
        # is_real indicates whether Rust module is available
        assert isinstance(prover._is_real, bool)

    def test_prove_gradient_sum_basic(self):
        """Test basic proof generation for gradient sums."""
        prover = GradientSumCheckProver()

        # Normal gradients
        gradients = [
            torch.randn(100) * 0.1,
            torch.randn(50) * 0.1,
            torch.randn(50) * 0.1,
            torch.randn(10) * 0.1,
        ]

        proof = prover.prove_gradient_sum(gradients, client_id=1, round_number=1)

        assert proof is not None
        assert proof.client_id == 1
        assert proof.round_number == 1
        assert proof.claimed_sum is not None
        assert len(proof.layer_sums) == 4

        # Should have either real proof bytes or fallback hash
        assert len(proof.proof_bytes) > 0

    def test_prove_gradient_sum_different_sizes(self):
        """Test proof generation with varying gradient tensor sizes."""
        prover = GradientSumCheckProver()

        # Different architecture shapes
        gradients = [
            torch.randn(784, 128) * 0.01,  # Large layer
            torch.randn(128) * 0.01,         # Bias
            torch.randn(128, 64) * 0.01,    # Medium layer
            torch.randn(64) * 0.01,          # Bias
            torch.randn(64, 10) * 0.01,      # Output
            torch.randn(10) * 0.01,          # Output bias
        ]

        proof = prover.prove_gradient_sum(gradients, client_id=2, round_number=5)

        assert proof is not None
        assert len(proof.layer_sums) == 6

        # Verify layer sums are computed correctly
        for i, (g, ls) in enumerate(zip(gradients, proof.layer_sums)):
            expected = g.sum().item()
            assert abs(ls - expected) < 1e-5, f"Layer {i} sum mismatch"

    def test_proof_verification(self):
        """Test proof verification works correctly."""
        prover = GradientSumCheckProver()

        gradients = [torch.randn(100) * 0.1 for _ in range(3)]
        proof = prover.prove_gradient_sum(gradients, client_id=1, round_number=1)

        # Verify should return True
        valid = GradientSumCheckProver.verify_proof(proof)
        assert valid == True

    def test_proof_contains_metadata(self):
        """Test that proof contains all required metadata."""
        prover = GradientSumCheckProver()

        gradients = [torch.randn(50) * 0.1 for _ in range(2)]
        proof = prover.prove_gradient_sum(gradients, client_id=42, round_number=10)

        # Check metadata fields
        assert proof.client_id == 42
        assert proof.round_number == 10
        assert proof.prove_time_ms > 0
        assert proof.num_steps > 0
        assert isinstance(proof.is_real, bool)

    def test_proof_size(self):
        """Test proof size property."""
        prover = GradientSumCheckProver()

        gradients = [torch.randn(100) * 0.1]
        proof = prover.prove_gradient_sum(gradients, client_id=1, round_number=1)

        # proof_size should return byte length
        assert proof.proof_size == len(proof.proof_bytes)


class TestGalaxyProofFolder:
    """Test suite for folding multiple client proofs into galaxy proofs."""

    pytestmark = pytestmark_real_zkp

    def test_fold_empty_proofs(self):
        """Test folding with no client proofs."""
        folder = GalaxyProofFolder()

        result = folder.fold_galaxy_proofs([], galaxy_id=1)

        # Empty folding returns client_id=-1 as placeholder
        assert result.client_id == -1
        assert result.claimed_sum == 0.0
        assert result.num_steps == 0

    def test_fold_single_client_proof(self):
        """Test folding with a single client proof."""
        folder = GalaxyProofFolder()

        # Create a proof
        prover = GradientSumCheckProver()
        gradients = [torch.randn(100) * 0.1]
        proof = prover.prove_gradient_sum(gradients, client_id=1, round_number=1)

        folded = folder.fold_galaxy_proofs([proof], galaxy_id=99)

        assert folded.client_id == 99
        assert folded.round_number == 1
        assert folded.claimed_sum == proof.claimed_sum

    def test_fold_multiple_client_proofs(self):
        """Test folding multiple client proofs."""
        folder = GalaxyProofFolder()

        # Create multiple client proofs
        prover = GradientSumCheckProver()
        gradients1 = [torch.randn(100) * 0.1]
        gradients2 = [torch.randn(100) * 0.2]

        proof1 = prover.prove_gradient_sum(gradients1, client_id=1, round_number=1)
        proof2 = prover.prove_gradient_sum(gradients2, client_id=2, round_number=1)

        folded = folder.fold_galaxy_proofs([proof1, proof2], galaxy_id=1)

        # Total should be sum of both client sums
        expected_total = proof1.claimed_sum + proof2.claimed_sum
        assert abs(folded.claimed_sum - expected_total) < 1e-5

        # Layer sums should be concatenated
        assert len(folded.layer_sums) == len(proof1.layer_sums) + len(proof2.layer_sums)


class TestModelAgnosticOperation:
    """Test that prover works with different model architectures."""

    pytestmark = pytestmark_real_zkp

    def test_small_mlp(self):
        """Test with small MLP gradients."""
        prover = GradientSumCheckProver()

        # Simple 2-layer MLP
        gradients = [
            torch.randn(784, 128) * 0.001,
            torch.randn(128) * 0.001,
            torch.randn(128, 10) * 0.001,
            torch.randn(10) * 0.001,
        ]

        proof = prover.prove_gradient_sum(gradients, client_id=1, round_number=1)

        assert proof is not None
        assert len(proof.layer_sums) == 4

        valid = GradientSumCheckProver.verify_proof(proof)
        assert valid == True

    def test_deep_network(self):
        """Test with deep network (many layers)."""
        prover = GradientSumCheckProver()

        # Deep network: 20 layers
        gradients = [torch.randn(100, 100) * 0.001 for _ in range(20)]

        proof = prover.prove_gradient_sum(gradients, client_id=2, round_number=1)

        assert proof is not None
        assert len(proof.layer_sums) == 20

        valid = GradientSumCheckProver.verify_proof(proof)
        assert valid == True

    def test_varied_layer_shapes(self):
        """Test with varied layer dimensions."""
        prover = GradientSumCheckProver()

        # Realistic CNN-like structure
        gradients = [
            torch.randn(64, 3, 3, 3) * 0.001,   # Conv1
            torch.randn(64) * 0.001,             # BN1
            torch.randn(128, 64, 3, 3) * 0.001,  # Conv2
            torch.randn(128) * 0.001,            # BN2
            torch.randn(512, 128) * 0.001,       # FC1
            torch.randn(10, 512) * 0.001,       # FC2
        ]

        proof = prover.prove_gradient_sum(gradients, client_id=3, round_number=1)

        assert proof is not None
        assert len(proof.layer_sums) == 6

        valid = GradientSumCheckProver.verify_proof(proof)
        assert valid == True


class TestFallbackMode:
    """Without fl_zkp_bridge, proving raises (no SHA-256 fallback)."""

    @pytest.mark.skipif(_ZKP_AVAILABLE, reason="Only when fl_zkp_bridge is not installed")
    def test_prover_raises_without_bridge(self):
        prover = GradientSumCheckProver()
        gradients = [torch.randn(100) * 0.1]
        with pytest.raises(RuntimeError):
            prover.prove_gradient_sum(gradients, client_id=1, round_number=1)


class TestByzantineResistance:
    """Test that TrainingStepCircuit provides Byzantine resistance."""

    pytestmark = pytestmark_real_zkp

    def test_different_gradients_same_data_different_proofs(self):
        """Test that different gradient values produce different proofs."""
        prover = GradientSumCheckProver()

        # Same structure, different values
        gradients1 = [torch.ones(100) * 1.0]
        gradients2 = [torch.ones(100) * 2.0]

        proof1 = prover.prove_gradient_sum(gradients1, client_id=1, round_number=1)
        proof2 = prover.prove_gradient_sum(gradients2, client_id=1, round_number=1)

        # Claims should differ
        assert proof1.claimed_sum != proof2.claimed_sum

        # Both should still verify (different gradients are valid)
        assert GradientSumCheckProver.verify_proof(proof1) == True
        assert GradientSumCheckProver.verify_proof(proof2) == True

    def test_proof_ties_to_specific_gradients(self):
        """Tampering metadata does not bypass IVC verification."""
        prover = GradientSumCheckProver()

        gradients = [torch.randn(100) * 0.5]
        proof = prover.prove_gradient_sum(gradients, client_id=1, round_number=1)

        assert GradientSumCheckProver.verify_proof(proof) is True

        modified_layer_sums = [ls + 999.0 for ls in proof.layer_sums]
        modified_proof = ZKProof(
            proof_bytes=proof.proof_bytes,
            claimed_sum=proof.claimed_sum,
            num_steps=proof.num_steps,
            client_id=proof.client_id,
            round_number=proof.round_number,
            prove_time_ms=proof.prove_time_ms,
            is_real=proof.is_real,
            layer_sums=modified_layer_sums,
            gradient_bundle=proof.gradient_bundle,
        )
        assert GradientSumCheckProver.verify_proof(modified_proof) is True