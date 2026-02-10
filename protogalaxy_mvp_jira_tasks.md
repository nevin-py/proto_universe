# ProtoGalaxy - MVP Jira Tasks (Scrum)

## MVP Scope Definition

This MVP focuses on demonstrating the core ProtoGalaxy architecture with essential Byzantine defense capabilities. The goal is to build a working system that can:
1. Train a federated model across multiple clients organized in galaxies
2. Verify gradient integrity using Merkle trees
3. Detect and filter Byzantine attacks using statistical analysis and robust aggregation
4. Track client reputation and quarantine malicious clients

**Excluded from MVP**: Public audit interfaces, authentication systems, re-admission protocols, forensic tools, dynamic rebalancing, advanced optimizations.

---

## MVP Phase 1: Foundation (Sprints 1-2)

### Epic 1: Core Cryptographic Infrastructure

**PROTO-101: Implement Base Merkle Tree Structure**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 1
- **Description**: Implement the fundamental Merkle tree data structure with SHA-256 hashing
- **Acceptance Criteria**:
  - Create MerkleTree class with build, get_root, get_proof methods
  - Support dynamic tree construction from arbitrary number of leaves
  - Handle odd number of nodes (duplicate last node)
  - O(n) construction complexity
- **Technical Details**:
  - Use SHA-256 for hashing
  - Store tree as list/array structure
  - Support serialization/deserialization

**PROTO-102: Implement Merkle Proof Generation**
- **Story Points**: 5
- **Priority**: Critical
- **Sprint**: 1
- **Description**: Create proof generation mechanism for any leaf in the tree
- **Acceptance Criteria**:
  - Generate proof path from leaf to root
  - Include sibling hashes and direction flags
  - Proof size O(log n)
  - Support batch proof generation
- **Dependencies**: PROTO-101

**PROTO-103: Implement Merkle Proof Verification**
- **Story Points**: 3
- **Priority**: Critical
- **Sprint**: 1
- **Description**: Create standalone proof verification function
- **Acceptance Criteria**:
  - Verify proof independently without full tree
  - O(log n) verification complexity
  - Return boolean valid/invalid
  - Handle edge cases (empty proofs, invalid formats)
- **Dependencies**: PROTO-101

**PROTO-104: Implement Gradient Commitment Hash**
- **Story Points**: 3
- **Priority**: Critical
- **Sprint**: 1
- **Description**: Create commitment hashing function for gradients with metadata
- **Acceptance Criteria**:
  - Hash gradient tensor + metadata (client_id, round, timestamp)
  - Consistent serialization of gradient tensors
  - Support for different tensor shapes/types
  - Deterministic output
- **Technical Details**:
  - Serialize gradient using numpy/torch standard methods
  - Concatenate metadata fields in fixed order

**PROTO-105: Implement Galaxy-Level Merkle Tree**
- **Story Points**: 5
- **Priority**: Critical
- **Sprint**: 2
- **Description**: Create specialized Merkle tree for galaxy aggregation
- **Acceptance Criteria**:
  - Build tree from client gradient commitments
  - Store client metadata in leaf nodes
  - Support querying by client_id
  - Generate proofs for specific clients
- **Dependencies**: PROTO-101, PROTO-104

**PROTO-106: Implement Global Merkle Tree**
- **Story Points**: 5
- **Priority**: Critical
- **Sprint**: 2
- **Description**: Create two-level Merkle tree from galaxy roots
- **Acceptance Criteria**:
  - Build tree from galaxy Merkle roots
  - Support cross-galaxy proof generation
  - Store galaxy metadata
  - Enable global verification
- **Dependencies**: PROTO-105

---

## MVP Phase 2: Client and Aggregator Core (Sprints 3-4)

### Epic 2: Client Components

**PROTO-201: Implement Local Trainer**
- **Story Points**: 13
- **Priority**: Critical
- **Sprint**: 3
- **Description**: Create client-side model training module
- **Acceptance Criteria**:
  - Train on local dataset for specified epochs
  - Compute gradients (∇w) from local data
  - Support PyTorch models (start with simple CNN)
  - Handle MNIST dataset with data loading
  - Return computed gradients
- **Technical Details**:
  - Use SGD optimizer
  - Fixed batch size (32), learning rate (0.01)
  - Simple model: 2 conv layers + 2 FC layers

**PROTO-202: Implement Commitment Generator**
- **Story Points**: 3
- **Priority**: Critical
- **Sprint**: 3
- **Description**: Generate cryptographic commitments for gradients
- **Acceptance Criteria**:
  - Generate metadata (client_id, round, timestamp)
  - Compute commitment hash using PROTO-104
  - Return commitment with gradient
- **Dependencies**: PROTO-104

**PROTO-203: Implement Proof Verifier Module**
- **Story Points**: 5
- **Priority**: Critical
- **Sprint**: 3
- **Description**: Client-side verification of received Merkle proofs
- **Acceptance Criteria**:
  - Verify inclusion proofs from galaxy aggregator
  - Log verification results to console
  - Raise exception on verification failures
- **Dependencies**: PROTO-103

**PROTO-205: Implement Client Communication Layer**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 4
- **Description**: Handle client-server communication (simplified)
- **Acceptance Criteria**:
  - Send gradients + commitments to galaxy aggregator
  - Receive Merkle proofs and global model
  - Use simple REST API (Flask/FastAPI)
  - Basic timeout handling (30s)
- **Technical Details**:
  - REST endpoints: POST /submit_gradient, GET /get_proof, GET /get_model
  - JSON serialization for gradients (convert to list)

### Epic 3: Galaxy Aggregator Components

**PROTO-301: Implement Merkle Tree Constructor for Galaxy**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 3
- **Description**: Build galaxy-level Merkle tree from client submissions
- **Acceptance Criteria**:
  - Collect client gradients and commitments
  - Construct Merkle tree using PROTO-105
  - Publish galaxy root to global aggregator
  - Store tree for proof generation
- **Dependencies**: PROTO-105

**PROTO-302: Implement Statistical Analyzer**
- **Story Points**: 13
- **Priority**: Critical
- **Sprint**: 4
- **Description**: Multi-metric anomaly detection for client gradients
- **Acceptance Criteria**:
  - Compute gradient norms and statistics
  - Implement 3 detection metrics (MVP scope):
    1. Norm deviation (threshold: 3σ)
    2. Direction similarity (cosine threshold: 0.5)
    3. Coordinate-wise analysis (per-dimension outliers)
  - Flag gradients failing ≥2 metrics
  - Return flagged client indices
- **Technical Details**:
  - Use numpy for statistical computations
  - Complexity: O(n·d) where d = model dimension

**PROTO-303: Implement Byzantine-Robust Aggregator**
- **Story Points**: 13
- **Priority**: Critical
- **Sprint**: 4
- **Description**: Multi-Krum robust aggregation algorithm (simplified)
- **Acceptance Criteria**:
  - Implement basic Multi-Krum with f = 0.3·n Byzantine threshold
  - Compute pairwise gradient distances
  - Select m = n-f top gradients by Krum score
  - Aggregate selected gradients (simple average)
  - Return robust aggregate
- **Technical Details**:
  - Set k = n-f-2 for distance computation
  - Use L2 distance metric
  - Complexity: O(n²·d)

**PROTO-304: Implement Local Reputation Manager**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 5
- **Description**: Track client behavior and reputation scores
- **Acceptance Criteria**:
  - Maintain per-client reputation scores (dict)
  - Update based on statistical analysis results
  - Apply EWMA with decay λ = 0.1
  - Check quarantine thresholds (< 0.3)
  - Store flagged clients in quarantine list
- **Technical Details**:
  - Behavior score: 1.0 if pass all checks, 0.0 if flagged
  - Persist to JSON file for simplicity

**PROTO-306: Implement Galaxy Communication Module**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 4
- **Description**: Handle communication with clients and global aggregator
- **Acceptance Criteria**:
  - Receive gradients from clients via REST API
  - Send Merkle proofs back to clients
  - Submit galaxy aggregate + root to global aggregator
  - Receive global model updates
- **Technical Details**:
  - Simple synchronous API (Flask/FastAPI)
  - Endpoints: POST /galaxy/aggregate, GET /galaxy/proof

### Epic 4: Global Aggregator Components

**PROTO-401: Implement Global Merkle Constructor**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 5
- **Description**: Build global Merkle tree from galaxy roots
- **Acceptance Criteria**:
  - Collect galaxy aggregates and roots
  - Build global Merkle tree using PROTO-106
  - Store global root
  - Generate cross-galaxy proofs
- **Dependencies**: PROTO-106

**PROTO-403: Implement Final Robust Aggregation**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 5
- **Description**: Final aggregation of galaxy updates (simplified)
- **Acceptance Criteria**:
  - Apply weighted average on galaxy aggregates
  - Weight by number of clients in galaxy
  - Compute final global model update
  - Apply update to global model
- **Technical Details**:
  - Simple weighted average for MVP
  - No additional Krum at global level

**PROTO-404: Implement Global Model Manager**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 5
- **Description**: Manage global model state and versioning
- **Acceptance Criteria**:
  - Store global model parameters
  - Track model version by round number
  - Apply gradient updates to model
  - Save model checkpoints to disk (every 10 rounds)
- **Technical Details**:
  - Use PyTorch model serialization
  - Store in models/ directory

**PROTO-407: Implement Global Communication Hub**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 5
- **Description**: Coordinate communication with all galaxy aggregators
- **Acceptance Criteria**:
  - Receive galaxy aggregates via REST API
  - Broadcast global model to all galaxies
  - Simple sequential processing (no parallelism in MVP)
- **Technical Details**:
  - REST endpoints: POST /global/submit, GET /global/model

---

## MVP Phase 3: Defense Integration (Sprint 5-6)

### Epic 5: Multi-Layer Defense System

**PROTO-501: Implement Layer 1 - Cryptographic Verification**
- **Story Points**: 5
- **Priority**: Critical
- **Sprint**: 5
- **Description**: Integrate Merkle tree verification into aggregation flow
- **Acceptance Criteria**:
  - Verify all client commitments in galaxy tree
  - Verify galaxy roots in global tree
  - Detect tampering via root hash changes
  - Log verification events to console
- **Dependencies**: PROTO-301, PROTO-401

**PROTO-502: Implement Layer 2 - Statistical Detection**
- **Story Points**: 5
- **Priority**: Critical
- **Sprint**: 5
- **Description**: Integrate statistical analysis into pipeline
- **Acceptance Criteria**:
  - Run statistical analyzer on all received gradients
  - Apply at galaxy level only (simplify for MVP)
  - Log anomaly reports to console
  - Feed results to reputation system
- **Dependencies**: PROTO-302, PROTO-304

**PROTO-503: Implement Layer 3 - Robust Aggregation**
- **Story Points**: 5
- **Priority**: Critical
- **Sprint**: 6
- **Description**: Integrate Byzantine-robust aggregation
- **Acceptance Criteria**:
  - Apply Multi-Krum at galaxy level
  - Exclude flagged gradients from Layer 2
  - Combine statistical filtering with geometric filtering
- **Dependencies**: PROTO-303

**PROTO-504: Implement Layer 4 - Reputation System**
- **Story Points**: 5
- **Priority**: Critical
- **Sprint**: 6
- **Description**: Integrate adaptive reputation-based defense
- **Acceptance Criteria**:
  - Update reputations based on all layer outputs
  - Apply quarantine logic (threshold < 0.3)
  - Exclude quarantined clients from aggregation
  - Log quarantine events
- **Dependencies**: PROTO-304

**PROTO-505: Implement Defense Coordination Logic**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 6
- **Description**: Coordinate all defense layers
- **Acceptance Criteria**:
  - Sequential pipeline: Layer 1 → Layer 2 → Layer 3 → Layer 4
  - Pass results between layers
  - Log defense decisions
  - Handle layer failures gracefully
- **Technical Details**:
  - Simple sequential execution
  - Return aggregated gradient + defense report

---

## MVP Phase 4: System Integration (Sprint 6-7)

### Epic 6: Configuration and Basic Infrastructure

**PROTO-701: Implement System Configuration Manager**
- **Story Points**: 5
- **Priority**: High
- **Sprint**: 6
- **Description**: Centralized configuration management
- **Acceptance Criteria**:
  - Define system parameters in config.yaml:
    - Number of galaxies (G = 3)
    - Number of clients (n = 30)
    - Byzantine threshold (α = 0.3)
    - Reputation threshold (0.3)
    - Statistical thresholds
  - Load configuration at startup
  - Validate configuration constraints
- **Technical Details**:
  - Use PyYAML
  - Simple schema validation

**PROTO-702: Implement Basic Logging**
- **Story Points**: 5
- **Priority**: High
- **Sprint**: 6
- **Description**: Basic logging system
- **Acceptance Criteria**:
  - Console logging with timestamps
  - Log levels: INFO, WARNING, ERROR
  - Per-component loggers (client, galaxy, global)
  - Log key events: round start/end, detections, quarantine
- **Technical Details**:
  - Use Python logging module
  - Simple console handler

### Epic 7: Orchestration

**PROTO-1101: Implement Federated Learning Coordinator**
- **Story Points**: 13
- **Priority**: Critical
- **Sprint**: 7
- **Description**: Main orchestrator for FL rounds
- **Acceptance Criteria**:
  - Coordinate multi-round training (10 rounds for MVP)
  - Synchronize clients, galaxies, global aggregator
  - Handle round transitions
  - Fixed round parameters: 1 local epoch, batch size 32
- **Technical Details**:
  - Simple synchronous coordination
  - Sequential client updates

**PROTO-1102: Implement Round Manager**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 7
- **Description**: Manage individual FL round lifecycle
- **Acceptance Criteria**:
  - Round initialization
  - All clients participate (no sampling in MVP)
  - Gradient collection phase
  - Aggregation phase
  - Model distribution phase
  - Round finalization with metrics logging
- **Technical Details**:
  - Track round state and timing
  - Simple timeout handling (60s per phase)

**PROTO-1104: Implement Model Synchronization**
- **Story Points**: 5
- **Priority**: High
- **Sprint**: 7
- **Description**: Distribute global model to all clients
- **Acceptance Criteria**:
  - Broadcast model parameters to all clients
  - Simple sequential distribution
  - Verify model integrity (hash check)
- **Technical Details**:
  - REST API: GET /model
  - Return model state_dict as JSON

---

## MVP Phase 5: Utilities and Data (Sprint 7-8)

### Epic 8: Essential Utilities

**PROTO-1001: Implement Gradient Utilities**
- **Story Points**: 5
- **Priority**: High
- **Sprint**: 7
- **Description**: Helper functions for gradient operations
- **Acceptance Criteria**:
  - Gradient serialization/deserialization (to/from list)
  - Norm computation (L2)
  - Distance metrics (L2, cosine)
  - Gradient aggregation (average, weighted)
- **Technical Details**:
  - Support PyTorch tensors
  - Convert to numpy for operations

**PROTO-1002: Implement Statistical Utilities**
- **Story Points**: 5
- **Priority**: High
- **Sprint**: 7
- **Description**: Statistical computation helpers
- **Acceptance Criteria**:
  - Median, mean, std computation
  - Outlier detection (Z-score)
  - Cosine similarity
- **Technical Details**:
  - Use numpy/scipy

**PROTO-1003: Implement Cryptographic Utilities**
- **Story Points**: 3
- **Priority**: High
- **Sprint**: 7
- **Description**: Cryptographic helper functions (simplified)
- **Acceptance Criteria**:
  - SHA-256 hash function wrapper
  - Nonce generation (random bytes)
  - Timestamp generation
- **Technical Details**:
  - Use hashlib and secrets modules

### Epic 9: Galaxy Management (Simplified)

**PROTO-801: Implement Static Galaxy Assignment**
- **Story Points**: 5
- **Priority**: High
- **Sprint**: 8
- **Description**: Simple static assignment of clients to galaxies
- **Acceptance Criteria**:
  - Round-robin assignment of clients to G galaxies
  - Balance galaxy sizes
  - Fixed assignment for entire training
  - Return mapping: client_id → galaxy_id
- **Technical Details**:
  - Simple modulo assignment: galaxy_id = client_id % G

---

## MVP Phase 6: Data Persistence (Sprint 8)

### Epic 10: Basic Data Management

**PROTO-1201: Implement Simple Gradient Store**
- **Story Points**: 5
- **Priority**: Medium
- **Sprint**: 8
- **Description**: In-memory storage for current round gradients
- **Acceptance Criteria**:
  - Store gradients by client_id for current round
  - Clear after each round
  - Support retrieval by client_id
- **Technical Details**:
  - Simple dict structure
  - No persistent storage in MVP

**PROTO-1203: Implement Simple Reputation Store**
- **Story Points**: 5
- **Priority**: Medium
- **Sprint**: 8
- **Description**: Simple file-based reputation storage
- **Acceptance Criteria**:
  - Store reputation scores in JSON file
  - Load at startup
  - Save after each round
  - Track quarantined clients
- **Technical Details**:
  - Use reputation.json file
  - Simple dict structure

**PROTO-1205: Implement Model Checkpoint Manager**
- **Story Points**: 3
- **Priority**: Medium
- **Sprint**: 8
- **Description**: Save model checkpoints
- **Acceptance Criteria**:
  - Save model every 5 rounds
  - Save to models/round_{n}.pt
  - Keep last 3 checkpoints only
- **Technical Details**:
  - Use torch.save()

---

## MVP Phase 7: Simulation and Validation (Sprint 9-10)

### Epic 11: Client Simulation

**PROTO-1501: Implement Honest Client Simulator**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 9
- **Description**: Simulate multiple honest FL clients
- **Acceptance Criteria**:
  - Simulate 21 honest clients (70% of 30)
  - Each trains on local MNIST data partition
  - Use IID data distribution (random split)
  - Sequential simulation (no parallelism in MVP)
- **Technical Details**:
  - MNIST dataset split into 30 partitions
  - Each client gets 2000 samples

**PROTO-1502: Implement Byzantine Client Simulator**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 9
- **Description**: Simulate Byzantine attacks
- **Acceptance Criteria**:
  - Simulate 9 Byzantine clients (30% of 30)
  - Implement 2 attack types:
    1. Label flipping (flip labels 0↔9, 1↔8, etc.)
    2. Gradient poisoning (scale gradients by -1 or random values)
  - Mix with honest clients
- **Technical Details**:
  - Simple attack implementations
  - Configurable attack type

**PROTO-1503: Implement Simple Metrics Collector**
- **Story Points**: 5
- **Priority**: High
- **Sprint**: 9
- **Description**: Collect and log training metrics
- **Acceptance Criteria**:
  - Track per-round metrics:
    - Global model accuracy (on test set)
    - Number of flagged clients
    - Number of quarantined clients
    - Aggregation time
  - Log to console and CSV file
  - Plot basic metrics at end of training

---

## MVP Phase 8: End-to-End Integration (Sprint 10)

### Epic 12: Integration and Error Handling

**PROTO-1401: Implement Basic Error Handling**
- **Story Points**: 5
- **Priority**: High
- **Sprint**: 10
- **Description**: Handle common errors gracefully
- **Acceptance Criteria**:
  - Validate gradient shapes
  - Handle missing clients (skip if not responding)
  - Catch and log exceptions
  - Continue training on errors when possible
- **Technical Details**:
  - Try-catch blocks around key operations
  - Graceful degradation

**PROTO-1404: Implement Data Validation**
- **Story Points**: 3
- **Priority**: High
- **Sprint**: 10
- **Description**: Validate inputs
- **Acceptance Criteria**:
  - Validate gradient shapes match model
  - Validate message formats
  - Check for NaN/Inf values
  - Reject invalid submissions
- **Technical Details**:
  - Schema validation for API messages

**PROTO-1105: Implement End-to-End Test**
- **Story Points**: 8
- **Priority**: Critical
- **Sprint**: 10
- **Description**: Complete integration test
- **Acceptance Criteria**:
  - Run full FL training: 10 rounds, 30 clients, 3 galaxies
  - Include 30% Byzantine clients (label flipping)
  - Verify model converges (>90% accuracy)
  - Verify Byzantine clients detected (>80% detection rate)
  - Verify quarantine system works
  - Complete run in <10 minutes
- **Technical Details**:
  - Automated test script
  - Assert on final metrics

---

## MVP Summary

### Total Scope
- **Total Tasks**: 48 tasks (reduced from 106)
- **Total Story Points**: ~280 points (reduced from ~490)
- **Estimated Duration**: 10 sprints (20 weeks) with 2-3 developers
- **Target**: Working demo with 30 clients, 3 galaxies, Byzantine defense

### Sprint Breakdown

**Sprint 1-2 (Foundation)**: Merkle Tree infrastructure
- PROTO-101 to 106

**Sprint 3-4 (Core Components)**: Clients and Galaxy Aggregators
- PROTO-201, 202, 203, 205
- PROTO-301, 302, 303, 306

**Sprint 5 (Global + Defense Start)**: Global Aggregator and Defense Layers 1-2
- PROTO-401, 403, 404, 407
- PROTO-501, 502

**Sprint 6 (Defense Complete)**: Defense Layers 3-4 and Config
- PROTO-503, 504, 505
- PROTO-701, 702

**Sprint 7 (Orchestration)**: FL Coordinator and Utilities
- PROTO-1101, 1102, 1104
- PROTO-1001, 1002, 1003

**Sprint 8 (Data + Galaxy)**: Data persistence and Galaxy Management
- PROTO-801
- PROTO-1201, 1203, 1205

**Sprint 9 (Simulation)**: Client simulators and Metrics
- PROTO-1501, 1502, 1503

**Sprint 10 (Integration)**: Error handling and E2E testing
- PROTO-1401, 1404, 1105

### Key MVP Features Included
✅ Hierarchical architecture (clients → galaxies → global)
✅ Merkle tree verification (two-level)
✅ Statistical anomaly detection (3 metrics)
✅ Multi-Krum robust aggregation
✅ Reputation-based quarantine
✅ Byzantine attack simulation
✅ End-to-end federated learning
✅ Basic logging and metrics

### Features Deferred Post-MVP
❌ Public audit interface
❌ Authentication system
❌ Re-admission protocol
❌ Forensic analysis tools
❌ Dynamic galaxy rebalancing
❌ Advanced optimizations (compression, parallelization)
❌ Evidence archive system
❌ Network simulation
❌ Caching layer

### Success Criteria for MVP
1. ✅ Complete 10 rounds of federated learning
2. ✅ Achieve >90% accuracy on MNIST with 30% Byzantine clients
3. ✅ Detect >80% of Byzantine clients
4. ✅ Quarantine detected clients
5. ✅ Verify gradient integrity with Merkle proofs
6. ✅ Complete end-to-end run in <10 minutes
7. ✅ Generate metrics and visualizations

---

## Dependencies Graph (Critical Path)

```
Sprint 1-2:
PROTO-101 → PROTO-102 → PROTO-103
         → PROTO-104 → PROTO-105 → PROTO-106

Sprint 3-4:
PROTO-201, PROTO-202 (parallel)
PROTO-105 → PROTO-301
PROTO-302, PROTO-303 (parallel)

Sprint 5:
PROTO-106 → PROTO-401
PROTO-301 + PROTO-401 → PROTO-501
PROTO-302 + PROTO-304 → PROTO-502

Sprint 6:
PROTO-303 → PROTO-503
PROTO-304 → PROTO-504
PROTO-501 to 504 → PROTO-505

Sprint 7:
PROTO-505 → PROTO-1101 → PROTO-1102

Sprint 10:
All previous → PROTO-1105
```

---

## Risk Mitigation

**High Risk Items**:
1. **PROTO-302 (Statistical Analyzer)**: Complex algorithm
   - Mitigation: Start simple, add metrics incrementally
   
2. **PROTO-303 (Multi-Krum)**: O(n²) complexity
   - Mitigation: Start with small n (30 clients), optimize later
   
3. **PROTO-1101 (FL Coordinator)**: Complex orchestration
   - Mitigation: Synchronous, sequential approach for MVP

4. **PROTO-1105 (E2E Test)**: Integration issues
   - Mitigation: Incremental testing throughout sprints

**Medium Risk Items**:
- Communication layer reliability (PROTO-205, 306, 407)
- Merkle tree correctness (PROTO-101 to 106)
- Defense coordination (PROTO-505)

---

## Post-MVP Roadmap (Future Sprints)

### Phase 2: Production Hardening (Sprints 11-13)
- Add authentication and authorization
- Implement public audit interface
- Add comprehensive error handling
- Performance optimization
- Deployment automation

### Phase 3: Advanced Features (Sprints 14-16)
- Re-admission protocol
- Forensic analysis tools
- Dynamic galaxy rebalancing
- Advanced attack simulations
- Differential privacy integration

### Phase 4: Scale and Performance (Sprints 17-20)
- Support 1000+ clients
- Parallel processing
- Gradient compression
- Distributed aggregators
- Cloud deployment
