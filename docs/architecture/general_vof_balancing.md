# General VoF Load Balancing and Migration Contract

This document defines the architectural contract for balancing and live ownership migration in VoF-based solvers.

## Balancing Philosophy

- **Forecast-Driven Balancing:** Balancing is performed based on work-sampling observations from the last accepted state. The system proposes ownership-only partition changes to improve load distribution, provided the predicted gain meets a configurable threshold.
- **Independence from Physics:** Load balancing and migration do not advance the physical clock. Migration is treated as an offline ownership-only reorganization of the solver state.
- **Data Integrity:** Migration is an atomic operation. The physical state (phase inventories, velocity fields, pressure histories, and momentum) is preserved exactly across partition boundaries, ensuring that rebalanced solvers remain functionally identical to their predecessors.

## Migration Requirements

- **Consistency Preservation:** Migration must preserve all conserved quantities, including species inventories, phase occupation provenance, and cleanup momentum, without introducing numerical diffusion or source-term artifacts.
- **Derived Storage:** While physical inventories are migrated, derived storage (e.g., pressure-preconditioner matrices, geometric caches, local operator lookups) must be rebuilt post-migration to reflect the new ownership distribution.
- **Atomicity:** The solver state is published using an atomic swap. Any failure during the migration planning or execution phase preserves the accepted solver state and fields, ensuring that the solver can continue from the pre-migration state.

## Restrictions

- **Coupling Constraints:** Rebalancing is explicitly disallowed during ongoing representation-relaxation phases or when coupled-channel cleanup is active.
- **Scope:** This path supports only ownership-only migration. It does not introduce capabilities for Exact/Bounded arithmetic backends, reacting flow, or ALE (moving mesh) solvers.
- **Manual Control:** Rebalancing is an explicit operation; there is no automatic cadence. The solver coordinator must evaluate the amortization of migration overhead against the measured step-time savings.

For detailed interface and implementation documentation, see:
- [General AMR Architecture](../architecture/general_amr.md)
- [VoF Transport Contract](vof_transport.md)
