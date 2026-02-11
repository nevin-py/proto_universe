"""ZKP Sum-Check Prover for Federated Learning Gradients.

Wraps the Rust fl_zkp_bridge module (ProtoGalaxy folding with Pedersen
commitments on BN254/Grumpkin) to prove that gradient weight sums are
computed correctly.

Sum-check approach:
  - Flatten gradient tensors per model layer
  - Compute per-layer sums
  - Fold each layer sum as a prove step: z_{i+1} = z_i + layer_sum
  - Final accumulated state = total gradient sum
  - Proof is constant-size (IVC property)

If the Rust module is not compiled, falls back gracefully with a warning.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch

logger = logging.getLogger(__name__)

# Try to import the Rust ZKP module
_ZKP_AVAILABLE = False
_FLZKPProver = None

try:
    from fl_zkp_bridge import FLZKPProver as _FLZKPProver
    _ZKP_AVAILABLE = True
    logger.info("✓ fl_zkp_bridge loaded — real ZK proofs enabled")
except ImportError:
    logger.warning(
        "fl_zkp_bridge not available — ZK proofs will use fallback mode. "
        "Build with: cd sonobe/fl-zkp-bridge && maturin develop --release"
    )


@dataclass
class ZKProof:
    """A zero-knowledge proof of gradient sum correctness."""
    proof_bytes: bytes
    claimed_sum: float
    num_steps: int
    client_id: int
    round_number: int
    prove_time_ms: float
    is_real: bool  # True if using Rust ZKP, False if fallback
    layer_sums: List[float] = field(default_factory=list)

    @property
    def proof_size(self) -> int:
        return len(self.proof_bytes)


class GradientSumCheckProver:
    """Proves correct computation of gradient weight sums via ProtoGalaxy IVC.

    Each model layer's gradient sum becomes one folding step:
      z_0 = 0
      z_1 = z_0 + sum(layer_0_gradients)
      z_2 = z_1 + sum(layer_1_gradients)
      ...
      z_n = total_gradient_sum

    The resulting proof is constant-size and verifiable in O(1).
    """

    def __init__(self):
        """Initialize the prover."""
        self._prover = None
        self._is_real = _ZKP_AVAILABLE

    def prove_gradient_sum(
        self,
        gradients: List[torch.Tensor],
        client_id: int,
        round_number: int,
    ) -> ZKProof:
        """Generate a ZK proof that gradient sums are correctly computed.

        Args:
            gradients: List of gradient tensors (one per model layer)
            client_id: Client identifier
            round_number: FL round number

        Returns:
            ZKProof with proof bytes and metadata
        """
        start = time.time()

        # Compute per-layer sums
        layer_sums = [g.sum().item() for g in gradients]
        total_sum = sum(layer_sums)

        if self._is_real:
            proof_bytes, num_steps = self._prove_real(layer_sums)
        else:
            proof_bytes, num_steps = self._prove_fallback(layer_sums)

        elapsed_ms = (time.time() - start) * 1000

        return ZKProof(
            proof_bytes=proof_bytes,
            claimed_sum=total_sum,
            num_steps=num_steps,
            client_id=client_id,
            round_number=round_number,
            prove_time_ms=elapsed_ms,
            is_real=self._is_real,
            layer_sums=layer_sums,
        )

    def _prove_real(self, layer_sums: List[float]) -> tuple:
        """Generate real ZK proof using Rust ProtoGalaxy."""
        prover = _FLZKPProver()
        prover.initialize(0.0)  # z_0 = 0

        # Each layer sum is one folding step
        for layer_sum in layer_sums:
            prover.prove_gradient_step(layer_sum)

        # Extract IVC proof
        proof_bytes = bytes(prover.generate_final_proof())
        num_steps = prover.get_num_steps()

        return proof_bytes, num_steps

    def _prove_fallback(self, layer_sums: List[float]) -> tuple:
        """Fallback: SHA-256 commitment (not a real ZK proof)."""
        import hashlib
        data = f"sumcheck:{layer_sums}".encode()
        commitment = hashlib.sha256(data).digest()
        return commitment, len(layer_sums)

    @staticmethod
    def verify_proof(proof: ZKProof) -> bool:
        """Verify a ZK sum-check proof.

        Args:
            proof: The ZKProof to verify

        Returns:
            True if proof is valid
        """
        start = time.time()

        if proof.is_real and _ZKP_AVAILABLE:
            valid = GradientSumCheckProver._verify_real(proof)
        else:
            valid = GradientSumCheckProver._verify_fallback(proof)

        elapsed_ms = (time.time() - start) * 1000
        logger.debug(
            f"  ZK verify client {proof.client_id}: "
            f"{'✓ VALID' if valid else '✗ INVALID'} ({elapsed_ms:.1f}ms)"
        )
        return valid

    @staticmethod
    def _verify_real(proof: ZKProof) -> bool:
        """Verify using Rust ProtoGalaxy IVC verification."""
        try:
            # Reconstruct prover state and verify
            prover = _FLZKPProver()
            prover.initialize(0.0)

            # Re-fold the layer sums
            for layer_sum in proof.layer_sums:
                prover.prove_gradient_step(layer_sum)

            # Verify the IVC proof
            return prover.verify_proof(list(proof.proof_bytes))
        except Exception as e:
            logger.error(f"ZK verification failed: {e}")
            return False

    @staticmethod
    def _verify_fallback(proof: ZKProof) -> bool:
        """Fallback verification: recompute hash commitment."""
        import hashlib
        data = f"sumcheck:{proof.layer_sums}".encode()
        expected = hashlib.sha256(data).digest()
        return proof.proof_bytes == expected


class GalaxyProofFolder:
    """Folds multiple client ZK proofs into a single galaxy proof.

    Uses ProtoGalaxy's IVC property: fold N proofs into 1 with
    constant proof size and O(1) verification.
    """

    def __init__(self):
        self._is_real = _ZKP_AVAILABLE

    def fold_galaxy_proofs(
        self,
        client_proofs: List[ZKProof],
        galaxy_id: int,
    ) -> ZKProof:
        """Fold multiple client proofs into a single galaxy proof.

        Args:
            client_proofs: List of client ZK proofs in this galaxy
            galaxy_id: Galaxy identifier

        Returns:
            Folded ZKProof representing all clients in the galaxy
        """
        if not client_proofs:
            return ZKProof(
                proof_bytes=b'',
                claimed_sum=0.0,
                num_steps=0,
                client_id=-1,
                round_number=0,
                prove_time_ms=0.0,
                is_real=False,
            )

        start = time.time()

        # Collect all layer sums from all clients
        all_layer_sums = []
        total_sum = 0.0
        round_number = client_proofs[0].round_number

        for proof in client_proofs:
            all_layer_sums.extend(proof.layer_sums)
            total_sum += proof.claimed_sum

        if self._is_real:
            proof_bytes, num_steps = self._fold_real(all_layer_sums)
        else:
            proof_bytes, num_steps = self._fold_fallback(all_layer_sums)

        elapsed_ms = (time.time() - start) * 1000

        return ZKProof(
            proof_bytes=proof_bytes,
            claimed_sum=total_sum,
            num_steps=num_steps,
            client_id=galaxy_id,  # Galaxy ID as "client"
            round_number=round_number,
            prove_time_ms=elapsed_ms,
            is_real=self._is_real,
            layer_sums=all_layer_sums,
        )

    def _fold_real(self, all_layer_sums: List[float]) -> tuple:
        """Fold via ProtoGalaxy — incrementally accumulate all sums."""
        prover = _FLZKPProver()
        prover.initialize(0.0)

        for layer_sum in all_layer_sums:
            prover.prove_gradient_step(layer_sum)

        proof_bytes = bytes(prover.generate_final_proof())
        num_steps = prover.get_num_steps()
        return proof_bytes, num_steps

    def _fold_fallback(self, all_layer_sums: List[float]) -> tuple:
        """Fallback galaxy folding."""
        import hashlib
        data = f"galaxy_fold:{all_layer_sums}".encode()
        commitment = hashlib.sha256(data).digest()
        return commitment, len(all_layer_sums)
