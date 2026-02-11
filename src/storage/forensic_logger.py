"""Forensic Logger and Evidence Database (Architecture Section 3.3, 4.4, 5.4)

Provides cryptographic audit trail for:
- Quarantine decisions with Merkle proof evidence
- Post-hoc attack tracing
- Dispute resolution

Architecture references:
- Section 3.3: "Forensic Logger: Maintains evidence database"
- Section 4.4: "Store Merkle proof evidence on ban"
- Section 5.4: "Forensic Analysis Time metric"
"""

import json
import os
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class QuarantineEvidence:
    """Evidence for a quarantine/ban decision"""
    client_id: int
    round_number: int
    timestamp: str
    decision_type: str  # "quarantine" or "ban"
    
    # Cryptographic evidence
    commitment_hash: str
    merkle_proof: List[Dict]
    merkle_root: str
    
    # Defense layer results
    layer1_crypto_violation: bool
    layer2_statistical_flags: List[str]
    layer3_robust_agg_rejected: bool
    layer4_reputation_score: float
    
    # Supporting metadata
    gradient_norm: Optional[float] = None
    cosine_similarity: Optional[float] = None
    galaxy_id: Optional[int] = None
    
    # Forensic metadata
    evidence_hash: str = ""  # SHA-256 of entire evidence
    
    def compute_evidence_hash(self) -> str:
        """Compute cryptographic hash of all evidence"""
        # Serialize evidence (excluding the hash field itself)
        evidence_dict = asdict(self)
        evidence_dict.pop('evidence_hash', None)
        
        evidence_str = json.dumps(evidence_dict, sort_keys=True)
        hash_obj = hashlib.sha256(evidence_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def finalize(self):
        """Compute and set evidence hash"""
        self.evidence_hash = self.compute_evidence_hash()


@dataclass
class ForensicQuery:
    """Query parameters for forensic analysis"""
    client_ids: Optional[List[int]] = None
    round_range: Optional[tuple] = None  # (min_round, max_round)
    decision_type: Optional[str] = None
    galaxy_id: Optional[int] = None
    layer_flags: Optional[List[str]] = None  # e.g., ["layer2_norm", "layer3_krum"]


class ForensicLogger:
    """Forensic evidence logger with cryptographic audit trail
    
    Maintains an immutable evidence database for:
    - All quarantine and ban decisions
    - Merkle proofs for cryptographic verification
    - Defense layer detection results
    - Timeline reconstruction for post-hoc analysis
    """
    
    def __init__(self, storage_dir: str = "./forensic_evidence"):
        """Initialize forensic logger
        
        Args:
            storage_dir: Directory to store evidence files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of recent evidence
        self.evidence_cache: List[QuarantineEvidence] = []
        
        # Indices for fast lookup
        self.client_index: Dict[int, List[int]] = {}  # client_id -> evidence indices
        self.round_index: Dict[int, List[int]] = {}   # round -> evidence indices
        self.galaxy_index: Dict[int, List[int]] = {}  # galaxy_id -> evidence indices
        
        # Load existing evidence
        self._load_existing_evidence()
    
    def log_quarantine(
        self,
        client_id: int,
        round_number: int,
        commitment_hash: str,
        merkle_proof: List[Dict],
        merkle_root: str,
        layer_results: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> QuarantineEvidence:
        """Log a quarantine decision with cryptographic evidence
        
        Args:
            client_id: Client being quarantined
            round_number: FL round number
            commitment_hash: Client's gradient commitment hash
            merkle_proof: Merkle proof for verification
            merkle_root: Galaxy or global Merkle root
            layer_results: Results from each defense layer
            metadata: Additional metadata
            
        Returns:
            Created QuarantineEvidence object
        """
        evidence = QuarantineEvidence(
            client_id=client_id,
            round_number=round_number,
            timestamp=datetime.now().isoformat(),
            decision_type="quarantine",
            commitment_hash=commitment_hash,
            merkle_proof=merkle_proof,
            merkle_root=merkle_root,
            layer1_crypto_violation=layer_results.get('layer1_failed', False),
            layer2_statistical_flags=layer_results.get('layer2_flags', []),
            layer3_robust_agg_rejected=layer_results.get('layer3_rejected', False),
            layer4_reputation_score=layer_results.get('layer4_reputation', 1.0),
            gradient_norm=metadata.get('gradient_norm') if metadata else None,
            cosine_similarity=metadata.get('cosine_similarity') if metadata else None,
            galaxy_id=metadata.get('galaxy_id') if metadata else None
        )
        
        evidence.finalize()
        self._store_evidence(evidence)
        return evidence
    
    def log_ban(
        self,
        client_id: int,
        round_number: int,
        commitment_hash: str,
        merkle_proof: List[Dict],
        merkle_root: str,
        layer_results: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> QuarantineEvidence:
        """Log a permanent ban decision with cryptographic evidence
        
        Same as log_quarantine but marks decision_type as "ban"
        """
        evidence = QuarantineEvidence(
            client_id=client_id,
            round_number=round_number,
            timestamp=datetime.now().isoformat(),
            decision_type="ban",
            commitment_hash=commitment_hash,
            merkle_proof=merkle_proof,
            merkle_root=merkle_root,
            layer1_crypto_violation=layer_results.get('layer1_failed', False),
            layer2_statistical_flags=layer_results.get('layer2_flags', []),
            layer3_robust_agg_rejected=layer_results.get('layer3_rejected', False),
            layer4_reputation_score=layer_results.get('layer4_reputation', 0.0),
            gradient_norm=metadata.get('gradient_norm') if metadata else None,
            cosine_similarity=metadata.get('cosine_similarity') if metadata else None,
            galaxy_id=metadata.get('galaxy_id') if metadata else None
        )
        
        evidence.finalize()
        self._store_evidence(evidence)
        return evidence
    
    def _store_evidence(self, evidence: QuarantineEvidence):
        """Store evidence to disk and update indices"""
        # Add to cache
        idx = len(self.evidence_cache)
        self.evidence_cache.append(evidence)
        
        # Update indices
        if evidence.client_id not in self.client_index:
            self.client_index[evidence.client_id] = []
        self.client_index[evidence.client_id].append(idx)
        
        if evidence.round_number not in self.round_index:
            self.round_index[evidence.round_number] = []
        self.round_index[evidence.round_number].append(idx)
        
        if evidence.galaxy_id is not None:
            if evidence.galaxy_id not in self.galaxy_index:
                self.galaxy_index[evidence.galaxy_id] = []
            self.galaxy_index[evidence.galaxy_id].append(idx)
        
        # Write to disk (one file per evidence for immutability)
        filename = f"evidence_{evidence.client_id}_r{evidence.round_number}_{evidence.timestamp.replace(':', '-')}.json"
        filepath = self.storage_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(evidence), f, indent=2)
    
    def _load_existing_evidence(self):
        """Load existing evidence files from disk"""
        if not self.storage_dir.exists():
            return
        
        for filepath in sorted(self.storage_dir.glob("evidence_*.json")):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    evidence = QuarantineEvidence(**data)
                    
                    idx = len(self.evidence_cache)
                    self.evidence_cache.append(evidence)
                    
                    # Rebuild indices
                    if evidence.client_id not in self.client_index:
                        self.client_index[evidence.client_id] = []
                    self.client_index[evidence.client_id].append(idx)
                    
                    if evidence.round_number not in self.round_index:
                        self.round_index[evidence.round_number] = []
                    self.round_index[evidence.round_number].append(idx)
                    
                    if evidence.galaxy_id is not None:
                        if evidence.galaxy_id not in self.galaxy_index:
                            self.galaxy_index[evidence.galaxy_id] = []
                        self.galaxy_index[evidence.galaxy_id].append(idx)
            except Exception as e:
                print(f"Warning: Could not load evidence file {filepath}: {e}")
    
    def query_evidence(self, query: ForensicQuery) -> List[QuarantineEvidence]:
        """Query evidence database
        
        Args:
            query: Query parameters
            
        Returns:
            List of matching evidence records
        """
        # Start with all evidence
        candidates = set(range(len(self.evidence_cache)))
        
        # Filter by client_ids
        if query.client_ids:
            client_matches = set()
            for cid in query.client_ids:
                client_matches.update(self.client_index.get(cid, []))
            candidates &= client_matches
        
        # Filter by round range
        if query.round_range:
            min_round, max_round = query.round_range
            round_matches = set()
            for r in range(min_round, max_round + 1):
                round_matches.update(self.round_index.get(r, []))
            candidates &= round_matches
        
        # Filter by galaxy
        if query.galaxy_id is not None:
            galaxy_matches = set(self.galaxy_index.get(query.galaxy_id, []))
            candidates &= galaxy_matches
        
        # Get matching evidence
        results = [self.evidence_cache[i] for i in sorted(candidates)]
        
        # Filter by decision type
        if query.decision_type:
            results = [e for e in results if e.decision_type == query.decision_type]
        
        # Filter by layer flags
        if query.layer_flags:
            filtered = []
            for e in results:
                if any(flag in e.layer2_statistical_flags for flag in query.layer_flags):
                    filtered.append(e)
            results = filtered
        
        return results
    
    def get_client_history(self, client_id: int) -> List[QuarantineEvidence]:
        """Get complete quarantine/ban history for a client
        
        Args:
            client_id: Client ID
            
        Returns:
            List of evidence records for this client
        """
        indices = self.client_index.get(client_id, [])
        return [self.evidence_cache[i] for i in indices]
    
    def get_round_decisions(self, round_number: int) -> List[QuarantineEvidence]:
        """Get all quarantine/ban decisions for a round
        
        Args:
            round_number: FL round number
            
        Returns:
            List of evidence records for this round
        """
        indices = self.round_index.get(round_number, [])
        return [self.evidence_cache[i] for i in indices]
    
    def verify_evidence_integrity(self, evidence: QuarantineEvidence) -> bool:
        """Verify cryptographic integrity of evidence
        
        Args:
            evidence: Evidence to verify
            
        Returns:
            True if evidence hash is valid
        """
        expected_hash = evidence.evidence_hash
        actual_hash = evidence.compute_evidence_hash()
        return expected_hash == actual_hash
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get forensic database statistics
        
        Returns:
            Statistics dictionary
        """
        total = len(self.evidence_cache)
        quarantines = sum(1 for e in self.evidence_cache if e.decision_type == "quarantine")
        bans = sum(1 for e in self.evidence_cache if e.decision_type == "ban")
        
        layer_breakdown = {
            'layer1_crypto_violations': sum(1 for e in self.evidence_cache if e.layer1_crypto_violation),
            'layer2_statistical_flags': sum(1 for e in self.evidence_cache if e.layer2_statistical_flags),
            'layer3_rejections': sum(1 for e in self.evidence_cache if e.layer3_robust_agg_rejected),
        }
        
        return {
            'total_records': total,
            'quarantines': quarantines,
            'bans': bans,
            'unique_clients': len(self.client_index),
            'unique_rounds': len(self.round_index),
            'layer_breakdown': layer_breakdown,
            'storage_path': str(self.storage_dir)
        }
    
    def export_timeline(self, output_file: str):
        """Export chronological timeline of all decisions
        
        Args:
            output_file: Path to output JSON file
        """
        timeline = []
        for evidence in self.evidence_cache:
            timeline.append({
                'timestamp': evidence.timestamp,
                'round': evidence.round_number,
                'client_id': evidence.client_id,
                'decision': evidence.decision_type,
                'galaxy_id': evidence.galaxy_id,
                'reputation': evidence.layer4_reputation_score,
                'evidence_hash': evidence.evidence_hash
            })
        
        with open(output_file, 'w') as f:
            json.dump(timeline, f, indent=2)
