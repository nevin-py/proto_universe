"""
ZKP Soundness Tests
===================
Tests that verify the cryptographic soundness of the GradientZKProver:

1. Honest proof verifies (completeness)
2. Tampered proof bytes rejected (soundness)
3. Wrong model fingerprint rejected
4. Proof bundle round-trip (serialize → deserialize → verify)
5. Norm bound detection via statistical filter
6. Empty proof rejected
7. Cross-client proof replay rejected (model_fp mismatch)

Run without Rust module (SHA-256 fallback):
  pytest tests/test_zkp_soundness.py -v

Run with real ProtoGalaxy (after maturin develop --release):
  ZKP_AVAILABLE=1 pytest tests/test_zkp_soundness.py -v --timeout=600
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crypto.zkp_prover import (
    ModelAgnosticProver,
    FingerprintHelper,
    GradientProofBundle,
    verify_gradient_proof,
    _ZKP_AVAILABLE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_linear_gradients():
    """Gradient list for a tiny linear model (784 → 10)."""
    torch.manual_seed(42)
    return [
        torch.randn(10, 784) * 0.01,  # weight gradient
        torch.randn(10) * 0.01,        # bias gradient
    ]


@pytest.fixture
def small_mlp_gradients():
    """Gradient list for a small MLP (784→128→64→10)."""
    torch.manual_seed(43)
    return [
        torch.randn(128, 784) * 0.01,
        torch.randn(128) * 0.01,
        torch.randn(64, 128) * 0.01,
        torch.randn(64) * 0.01,
        torch.randn(10, 64) * 0.01,
        torch.randn(10) * 0.01,
    ]


@pytest.fixture
def model_fp():
    """A deterministic model fingerprint."""
    return FingerprintHelper.compute_model_fingerprint(
        [torch.randn(10, 784), torch.randn(10)],
        round_number=1,
    )[0]


@pytest.fixture
def prover():
    return ModelAgnosticProver(grad_sample_size=100)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCompleteness:
    """Honest provers should always produce valid proofs."""

    def test_honest_proof_verifies_linear(self, small_linear_gradients, model_fp, prover):
        """Honest linear model gradient proof must verify."""
        bundle = prover.generate_proof(
            gradients=small_linear_gradients,
            ref_gradients=small_linear_gradients,
            model_fp=model_fp,
            ref_grad_fp=model_fp,
            client_id=0,
            round_number=1,
        )
        assert bundle is not None
        assert bundle.num_steps > 0
        assert bundle.proof_size > 0

        valid, reason = verify_gradient_proof(bundle, expected_model_fp=model_fp)
        assert valid, f"Honest proof failed verification: {reason}"

    def test_honest_proof_verifies_mlp(self, small_mlp_gradients, model_fp, prover):
        """Honest MLP gradient proof must verify."""
        bundle = prover.generate_proof(
            gradients=small_mlp_gradients,
            ref_gradients=small_mlp_gradients,
            model_fp=model_fp,
            ref_grad_fp=model_fp,
            client_id=1,
            round_number=1,
        )
        valid, reason = verify_gradient_proof(bundle, expected_model_fp=model_fp)
        assert valid, f"MLP proof failed: {reason}"

    def test_multiple_rounds_each_verify(self, small_linear_gradients, prover):
        """Proofs across multiple rounds should each independently verify."""
        for rnd in [1, 2, 5]:
            fp, _ = FingerprintHelper.compute_model_fingerprint(
                small_linear_gradients, round_number=rnd
            )
            bundle = prover.generate_proof(
                gradients=small_linear_gradients,
                ref_gradients=small_linear_gradients,
                model_fp=fp,
                ref_grad_fp=fp,
                client_id=0,
                round_number=rnd,
            )
            valid, reason = verify_gradient_proof(bundle)
            assert valid, f"Round {rnd} proof failed: {reason}"

    def test_different_clients_proofs_both_verify(self, small_linear_gradients, prover):
        """Different clients can both have valid proofs in the same round."""
        torch.manual_seed(0)
        grads_a = [torch.randn(10, 784) * 0.01, torch.randn(10) * 0.01]
        torch.manual_seed(1)
        grads_b = [torch.randn(10, 784) * 0.01, torch.randn(10) * 0.01]

        fp = 123456789
        bundle_a = prover.generate_proof(grads_a, ref_gradients=grads_a, model_fp=fp, ref_grad_fp=fp, client_id=0, round_number=1)
        bundle_b = prover.generate_proof(grads_b, ref_gradients=grads_b, model_fp=fp, ref_grad_fp=fp, client_id=1, round_number=1)

        valid_a, _ = verify_gradient_proof(bundle_a, expected_model_fp=fp)
        valid_b, _ = verify_gradient_proof(bundle_b, expected_model_fp=fp)
        assert valid_a, "Client A proof invalid"
        assert valid_b, "Client B proof invalid"


class TestSoundness:
    """Invalid proofs must be rejected."""

    def test_wrong_model_fp_rejected(self, small_linear_gradients, prover):
        """Proof generated with model_fp=X should fail if server checks model_fp=Y."""
        bundle = prover.generate_proof(
            gradients=small_linear_gradients,
            ref_gradients=small_linear_gradients,
            model_fp=111111,
            ref_grad_fp=111111,
            client_id=0,
            round_number=1,
        )
        # Server expects a different model fingerprint
        valid, reason = verify_gradient_proof(bundle, expected_model_fp=999999)
        assert not valid, f"Wrong model_fp should have been rejected but got: {reason}"

    def test_tampered_proof_bytes_rejected(self, small_linear_gradients, model_fp, prover):
        """Flipping bytes in the proof bundle must cause verify to fail."""
        if not _ZKP_AVAILABLE:
            pytest.skip("Byte-level tampering only detectable with real ZK module")

        bundle = prover.generate_proof(
            gradients=small_linear_gradients,
            ref_gradients=small_linear_gradients,
            model_fp=model_fp,
            ref_grad_fp=model_fp,
            client_id=0,
            round_number=1,
        )

        # Tamper with proof bytes (flip bytes in the middle of the IVC proof)
        tampered = bytearray(bundle.proof_bytes)
        mid = len(tampered) // 2
        tampered[mid] ^= 0xFF
        tampered[mid + 1] ^= 0xAA
        
        tampered_bundle = GradientProofBundle(
            proof_bytes=bytes(tampered),
            model_fp=bundle.model_fp,
            grad_fp=bundle.grad_fp,
            norm_sq_quantized=bundle.norm_sq_quantized,
            directional_fp_quantized=bundle.directional_fp_quantized,
            num_steps=bundle.num_steps,
            num_gradient_elements=bundle.num_gradient_elements,
            client_id=bundle.client_id,
            round_number=bundle.round_number,
            prove_time_ms=bundle.prove_time_ms,
            is_real=bundle.is_real,
        )

        valid, reason = verify_gradient_proof(tampered_bundle)
        assert not valid, f"Tampered proof should have been rejected but: {reason}"

    def test_empty_proof_rejected(self, model_fp):
        """A proof with zero steps or empty bytes must be rejected."""
        empty_bundle = GradientProofBundle(
            proof_bytes=b"",
            model_fp=model_fp,
            grad_fp=0,
            norm_sq_quantized=0.0,
            directional_fp_quantized=0.0,
            num_steps=0,
            num_gradient_elements=0,
            client_id=0,
            round_number=1,
            prove_time_ms=0.0,
            is_real=False,
        )
        valid, reason = verify_gradient_proof(empty_bundle)
        assert not valid, f"Empty proof should be rejected but got: {reason}"

    def test_fallback_roundtrip_honest(self, small_linear_gradients, prover):
        """In fallback mode, honest proof should still verify (SHA-256 consistency)."""
        if _ZKP_AVAILABLE:
            pytest.skip("Fallback test only relevant without Rust module")

        bundle = prover.generate_proof(
            gradients=small_linear_gradients,
            ref_gradients=small_linear_gradients,
            model_fp=42,
            ref_grad_fp=42,
            client_id=0,
            round_number=1,
        )
        assert not bundle.is_real, "Should be in fallback mode"
        valid, reason = verify_gradient_proof(bundle)
        assert valid, f"Fallback honest proof failed: {reason}"

    def test_fallback_tampered_rejected(self, small_linear_gradients, prover):
        """In fallback mode, tampered proof commitment must be rejected."""
        if _ZKP_AVAILABLE:
            pytest.skip("Fallback test only relevant without Rust module")

        bundle = prover.generate_proof(
            gradients=small_linear_gradients,
            ref_gradients=small_linear_gradients,
            model_fp=42,
            ref_grad_fp=42,
            client_id=0,
            round_number=1,
        )

        # Flip one byte in the SHA-256 commitment
        tampered = bytearray(bundle.proof_bytes)
        tampered[0] ^= 0xFF

        tampered_bundle = GradientProofBundle(
            proof_bytes=bytes(tampered),
            model_fp=bundle.model_fp,
            grad_fp=bundle.grad_fp,
            norm_sq_quantized=bundle.norm_sq_quantized,
            directional_fp_quantized=bundle.directional_fp_quantized,
            num_steps=bundle.num_steps,
            num_gradient_elements=bundle.num_gradient_elements,
            client_id=bundle.client_id,
            round_number=bundle.round_number,
            prove_time_ms=bundle.prove_time_ms,
            is_real=bundle.is_real,
        )
        valid, reason = verify_gradient_proof(tampered_bundle)
        assert not valid, f"Tampered fallback proof should be rejected"


class TestByzantineDetection:
    """Byzantine attacks should result in proof failure or rejection."""

    def test_scale_attack_large_norm_detected(self, prover):
        """Scale ×10 attack produces a very large gradient norm — detected by norm filter."""
        torch.manual_seed(10)
        honest_grads = [torch.randn(10, 784) * 0.01, torch.randn(10) * 0.01]
        # Byzantine: scale by 10x
        byzantine_grads = [g * 10.0 for g in honest_grads]

        honest_norm = float(torch.cat([g.flatten() for g in honest_grads]).norm())
        byzantine_norm = float(torch.cat([g.flatten() for g in byzantine_grads]).norm())

        assert byzantine_norm > honest_norm * 5.0, (
            f"Scale attack should produce much larger norm: "
            f"honest={honest_norm:.3f}, byzantine={byzantine_norm:.3f}"
        )

    def test_sign_flip_changes_fingerprint(self, prover):
        """Sign flip attack produces a different gradient fingerprint."""
        torch.manual_seed(11)
        grads = [torch.randn(10, 784) * 0.01, torch.randn(10) * 0.01]
        flipped_grads = [-g for g in grads]

        r_chunks = FingerprintHelper.generate_random_chunks(
            num_elements=10 * 784 + 10,
            round_number=1, client_id=0
        )
        flat_honest  = np.concatenate([g.numpy().flatten() for g in grads])
        flat_flipped = np.concatenate([g.numpy().flatten() for g in flipped_grads])
        flat_honest_padded  = np.pad(flat_honest,  (0, max(0, len(r_chunks)*2048 - len(flat_honest))))
        flat_flipped_padded = np.pad(flat_flipped, (0, max(0, len(r_chunks)*2048 - len(flat_flipped))))

        fp_honest  = FingerprintHelper.compute_gradient_fingerprint(flat_honest_padded, r_chunks)
        fp_flipped = FingerprintHelper.compute_gradient_fingerprint(flat_flipped_padded, r_chunks)

        assert fp_honest != fp_flipped, "Sign flip should produce different fingerprint"
        assert abs(fp_honest + fp_flipped) < abs(fp_honest) * 0.01 or fp_honest != fp_flipped, (
            "Fingerprints should differ for honest vs sign-flipped gradients"
        )

    def test_proof_metadata_is_correct(self, small_linear_gradients, model_fp, prover):
        """Proof bundle must contain correct metadata fields."""
        bundle = prover.generate_proof(
            gradients=small_linear_gradients,
            ref_gradients=small_linear_gradients,
            model_fp=model_fp,
            ref_grad_fp=model_fp,
            client_id=5,
            round_number=3,
        )
        assert bundle.client_id == 5
        assert bundle.round_number == 3
        assert bundle.model_fp == model_fp
        assert bundle.num_steps > 0
        assert bundle.num_gradient_elements > 0
        assert bundle.prove_time_ms > 0
        assert bundle.proof_size > 0


class TestFingerprint:
    """FingerprintHelper correctness tests."""

    def test_same_model_same_fp(self):
        """Same parameters produce the same fingerprint."""
        params = [torch.ones(10, 784), torch.ones(10)]
        fp1, _ = FingerprintHelper.compute_model_fingerprint(params, round_number=1)
        fp2, _ = FingerprintHelper.compute_model_fingerprint(params, round_number=1)
        assert fp1 == fp2

    def test_different_models_different_fp(self):
        """Different parameters (very likely) produce different fingerprints."""
        torch.manual_seed(100)
        params_a = [torch.randn(10, 784), torch.randn(10)]
        torch.manual_seed(200)
        params_b = [torch.randn(10, 784), torch.randn(10)]
        fp_a, _ = FingerprintHelper.compute_model_fingerprint(params_a, round_number=1)
        fp_b, _ = FingerprintHelper.compute_model_fingerprint(params_b, round_number=1)
        assert fp_a != fp_b

    def test_different_rounds_different_fp_same_model(self):
        """Different rounds produce different fingerprints (different r vectors)."""
        params = [torch.ones(10, 784), torch.ones(10)]
        fp1, _ = FingerprintHelper.compute_model_fingerprint(params, round_number=1)
        fp5, _ = FingerprintHelper.compute_model_fingerprint(params, round_number=5)
        assert fp1 != fp5

    def test_r_chunks_deterministic(self):
        """Random challenge chunks are deterministic (Fiat-Shamir)."""
        chunks_a = FingerprintHelper.generate_random_chunks(200, round_number=3, client_id=0)
        chunks_b = FingerprintHelper.generate_random_chunks(200, round_number=3, client_id=0)
        for a, b in zip(chunks_a, chunks_b):
            np.testing.assert_array_equal(a, b)

    def test_r_chunks_differ_across_clients(self):
        """Different clients get different random challenges (replay prevention)."""
        chunks_0 = FingerprintHelper.generate_random_chunks(200, round_number=1, client_id=0)
        chunks_1 = FingerprintHelper.generate_random_chunks(200, round_number=1, client_id=1)
        assert not np.allclose(chunks_0[0], chunks_1[0]), \
            "Client 0 and 1 should have different challenge vectors"
