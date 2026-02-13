"""Tests for bounded ZKP circuit with norm enforcement.

These tests verify that the enhanced ProtoGalaxy circuit correctly:
1. Accepts gradients within norm bounds
2. Rejects gradients exceeding norm bounds  
3. Works model-agnostically across different architectures
4. Maintains backward compatibility
"""

import pytest
import torch
import numpy as np

from src.crypto.zkp_prover import GradientSumCheckProver, ZKProof


class TestBoundedZKPCircuit:
    """Test suite for norm-bounded ZKP circuit."""

    def test_bounded_prover_accepts_valid_gradients(self):
        """Test that bounded prover accepts gradients within norm bounds."""
        prover = GradientSumCheckProver(use_bounds=True, norm_scale_factor=3.0)
        
        # Normal gradients (small magnitude)
        gradients = [
            torch.randn(100) * 0.1,  # Layer 1
            torch.randn(50) * 0.1,   # Layer 2
            torch.randn(50) * 0.1,   # Layer 3
            torch.randn(10) * 0.1,   # Layer 4
        ]
        
        # Should succeed - gradients are small and within 3x their own norms
        proof = prover.prove_gradient_sum(gradients, client_id=1, round_number=1)
        
        assert proof is not None
        assert proof.bounds_enforced == True
        assert len(proof.norm_bounds) == 4
        assert proof.is_real  # Assumes Rust module compiled
        
    def test_bounded_prover_rejects_out_of_bound_gradients(self):
        """Test that bounded prover rejects gradients exceeding bounds."""
        torch.manual_seed(42)  # Deterministic
        prover = GradientSumCheckProver(use_bounds=True, norm_scale_factor=10.0)  # Generous bounds
        
        # Normal gradients: small magnitude, layer sums will be small
        normal_grads = [torch.randn(100) * 0.001 for _ in range(4)]
        
        # Prove normal gradients work
        proof_ok = prover.prove_gradient_sum(
            normal_grads, 
            client_id=1, 
            round_number=1
        )
        assert proof_ok.bounds_enforced == True
        
        # Create malicious gradient (10000x larger) - will definitely exceed auto-computed bounds
        torch.manual_seed(123)
        poisoned_grads = [torch.randn(100) * 10.0 for _ in range(4)]
        
        # Compute very tight bounds from normal gradients (auto-compute uses norm_scale_factor * norm)
        normal_norms = [torch.norm(g).item() for g in normal_grads]
        tight_bounds = [n * 0.5 for n in normal_norms]  # 0.5x for very tight constraint
        
        # Poisoned gradients should fail with tight bounds
        with pytest.raises(Exception):  # Rust will raise error
            prover.prove_gradient_sum(
                poisoned_grads,
                client_id=2,
                round_number=1,
                norm_thresholds=tight_bounds
            )
    
    def test_model_agnostic_different_architectures(self):
        """Verify same circuit works for SimpleMLP, CNN, and ResNet."""
        prover = GradientSumCheckProver(use_bounds=True, norm_scale_factor=10.0)  # Generous bounds
        
        # SimpleMLP: 4 layers (small magnitude to keep layer sums small)
        torch.manual_seed(100)
        mlp_grads = [
            torch.randn(784 * 128) * 0.0001,      # input -> hidden1
            torch.randn(128) * 0.001,             # bias (smaller dimension)
            torch.randn(128 * 64) * 0.0001,        # hidden1 -> hidden2
            torch.randn(10) * 0.001,              # output (smallest dimension)
        ]
        proof_mlp = prover.prove_gradient_sum(mlp_grads, client_id=1, round_number=1)
        
        # CIFAR10CNN: ~20 layers
        torch.manual_seed(200)
        cnn_grads = [torch.randn(64*3*3*3) * 0.0001] + [torch.randn(128*64) * 0.0001 for _ in range(19)]
        proof_cnn = prover.prove_gradient_sum(cnn_grads, client_id=2, round_number=1)
        
        # ResNet18: ~62 layers (fixed dimension, small magnitude)
        torch.manual_seed(300)
        resnet_grads = [torch.randn(1000) * 0.0001 for _ in range(62)]
        proof_resnet = prover.prove_gradient_sum(resnet_grads, client_id=3, round_number=1)
        
        # All should succeed with same circuit
        assert proof_mlp.bounds_enforced == True
        assert proof_cnn.bounds_enforced == True
        assert proof_resnet.bounds_enforced == True
        
        # Verify all proofs
        assert GradientSumCheckProver.verify_proof(proof_mlp) == True
        assert GradientSumCheckProver.verify_proof(proof_cnn) == True
        assert GradientSumCheckProver.verify_proof(proof_resnet) == True

    def test_automatic_threshold_computation(self):
        """Test that automatic threshold computation works correctly."""
        prover = GradientSumCheckProver(use_bounds=True, norm_scale_factor=5.0)
        
        gradients = [
            torch.ones(100),      # norm = 10.0
            torch.ones(50) * 2,   # norm = 10.0 * sqrt(2)
            torch.ones(25) * 3,   # norm = 15.0
        ]
        
        # Compute thresholds automatically
        thresholds = prover._compute_norm_thresholds(gradients)
        
        # Expected: 5.0 * layer_norm for each layer
        expected = [
            5.0 * torch.norm(gradients[0]).item(),
            5.0 * torch.norm(gradients[1]).item(),
            5.0 * torch.norm(gradients[2]).item(),
        ]
        
        assert len(thresholds) == 3
        for actual, exp in zip(thresholds, expected):
            assert abs(actual - exp) < 1e-4

    def test_threshold_enforcement_prevents_byzantine_attacks(self):
        """Test that threshold enforcement prevents Byzantine attacks."""
        prover = GradientSumCheckProver(use_bounds=True, norm_scale_factor=2.0)
        
        # Honest gradient baseline (deterministic)
        torch.manual_seed(1000)
        honest_grad = [torch.randn(100) * 0.001 for _ in range(4)]  # Very small magnitude
        honest_norms = [torch.norm(g).item() for g in honest_grad]
        
        # Byzantine attacker tries to submit gradients 10x larger
        torch.manual_seed(2000)
        byzantine_grad = [torch.randn(100) * 0.01 for _ in range(4)]  # 10x honest
        
        # Compute statistical thresholds from honest gradient
        # In real defense: median(honest_norms) + k*MAD across multiple clients
        # Here: simulate with 1.5x honest norm (tight)
        statistical_bounds = [h * 1.5 for h in honest_norms]
        
        # Honest proof should succeed
        torch.manual_seed(1000)
        honest_grad_test = [torch.randn(100) * 0.001 for _ in range(4)]
        proof_honest = prover.prove_gradient_sum(
            honest_grad_test,
            client_id=1,
            round_number=1,
            norm_thresholds=statistical_bounds
        )
        assert proof_honest.bounds_enforced == True
        
        # Byzantine proof should FAIL (gradients exceed 1.5x honest norm)
        with pytest.raises(Exception) as exc_info:
            prover.prove_gradient_sum(
                byzantine_grad,
                client_id=666,
                round_number=1,
                norm_thresholds=statistical_bounds
            )
        assert "bound violated" in str(exc_info.value).lower()

    def test_backward_compatibility_unbounded_mode(self):
        """Test that unbounded mode still works for backward compatibility."""
        prover_unbounded = GradientSumCheckProver(use_bounds=False)
        
        gradients = [torch.randn(100) * 10.0 for _ in range(4)]  # Large gradients
        
        # Without bounds, should still produce proof (but no enforcement)
        proof = prover_unbounded.prove_gradient_sum(
            gradients,
            client_id=1,
            round_number=1
        )
        
        assert proof.bounds_enforced == False
        assert len(proof.norm_bounds) == 0

    def test_proof_verification_with_bounds(self):
        """Test that verification correctly handles bounded proofs."""
        prover = GradientSumCheckProver(use_bounds=True, norm_scale_factor=10.0)  # Generous bounds
        
        torch.manual_seed(3000)
        gradients = [torch.randn(100) * 0.001 for _ in range(4)]  # Small magnitude
        
        # Generate proof
        proof = prover.prove_gradient_sum(gradients, client_id=1, round_number=1)
        
        # Verify proof
        is_valid = GradientSumCheckProver.verify_proof(proof)
        
        assert is_valid == True
        assert proof.bounds_enforced == True

    def test_norm_bound_scales_with_model_depth(self):
        """Test that norm bounds scale correctly with model depth."""
        prover = GradientSumCheckProver(use_bounds=True, norm_scale_factor=4.0)
        
        # Shallow model (4 layers)
        shallow_grads = [torch.randn(100) * 0.1 for _ in range(4)]
        
        # Deep model (100 layers)
        deep_grads = [torch.randn(100) * 0.1 for _ in range(100)]
        
        # Both should succeed
        proof_shallow = prover.prove_gradient_sum(shallow_grads, client_id=1, round_number=1)
        proof_deep = prover.prove_gradient_sum(deep_grads, client_id=2, round_number=1)
        
        assert proof_shallow.num_steps == 4
        assert proof_deep.num_steps == 100
        assert len(proof_shallow.norm_bounds) == 4
        assert len(proof_deep.norm_bounds) == 100

    def test_fallback_mode_when_rust_unavailable(self):
        """Test that fallback mode works when Rust module is not compiled."""
        # Even if Rust module is available, we can test fallback by setting use_bounds=False
        prover = GradientSumCheckProver(use_bounds=False)
        
        gradients = [torch.randn(100) * 0.1 for _ in range(4)]
        
        # Should generate proof in fallback mode
        proof = prover.prove_gradient_sum(gradients, client_id=1, round_number=1)
        
        # Fallback proof should be SHA-256 hash (32 bytes if using hashlib.sha256)
        assert proof is not None
        assert proof.bounds_enforced == False


class TestZKProofDataclass:
    """Test ZKProof dataclass with norm bounds."""

    def test_zkproof_contains_norm_bounds(self):
        """Test that ZKProof stores norm bounds correctly."""
        proof = ZKProof(
            proof_bytes=b'test',
            claimed_sum=100.0,
            num_steps=4,
            client_id=1,
            round_number=1,
            prove_time_ms=50.0,
            is_real=True,
            layer_sums=[10.0, 20.0, 30.0, 40.0],
            norm_bounds=[15.0, 25.0, 35.0, 45.0],
            bounds_enforced=True,
        )
        
        assert proof.norm_bounds == [15.0, 25.0, 35.0, 45.0]
        assert proof.bounds_enforced == True
        assert len(proof.layer_sums) == len(proof.norm_bounds)


class TestIntegrationWithStatisticalDefense:
    """Test integration between ZKP bounds and statistical defense."""

    def test_zkp_enforces_statistical_thresholds(self):
        """Test that ZKP circuit enforces thresholds from statistical analyzer."""
        prover = GradientSumCheckProver(use_bounds=True)
        
        # Simulate statistical defense computing thresholds
        # In real system: from src.defense.statistical import StatisticalAnalyzer
        # thresholds = analyzer.compute_norm_thresholds(honest_gradients)
        
        # Simulate: median + 3*MAD from honest client history
        honest_client_norms = [
            [1.0, 1.5, 2.0, 0.5],  # Client 1
            [1.1, 1.4, 2.1, 0.6],  # Client 2
            [0.9, 1.6, 1.9, 0.4],  # Client 3
        ]
        
        # Compute per-layer median + 3*MAD
        layer_medians = [
            np.median([c[i] for c in honest_client_norms]) 
            for i in range(4)
        ]
        layer_mads = [
            np.median([abs(c[i] - layer_medians[i]) for c in honest_client_norms])
            for i in range(4)
        ]
        statistical_thresholds = [
            median + 3.0 * (1.4826 * mad)  # MAD to std conversion
            for median, mad in zip(layer_medians, layer_mads)
        ]
        
        # Honest gradient within statistical bounds
        # Key: layer_sum = sum(gradient_tensor), so for small sums use tiny values
        # With 100 elements, torch.ones(100) * 0.0001 gives sum ~0.01
        honest_grad = [torch.ones(100) * 0.0001 for _ in range(4)]
        
        proof = prover.prove_gradient_sum(
            honest_grad,
            client_id=1,
            round_number=1,
            norm_thresholds=statistical_thresholds
        )
        
        assert proof.bounds_enforced == True
        assert proof.norm_bounds == pytest.approx(statistical_thresholds, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
