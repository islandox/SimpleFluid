# VoF Transport and State Ownership Contract

This document defines the architectural contract for conservative VoF transport and state ownership. It ensures that mass-compatible momentum, phase budgets, and population identities are preserved across mesh-adaptation and solver-advancement stages.

## Conservative Transport Principles

- **Phase Accounting:** The system enforces strict phase volume conservation. Source-free transport is defined by canonical face-ledger updates, ensuring algebraic cancellation of internal transfers across coarse/fine seams.
- **Positivity and Capacity:** All phase transfers must respect non-negativity and capacity constraints. Transfers are limited by the available inventory of the donor cell, ensuring that no inventory is created or destroyed numerically.
- **Arithmetic Integrity:** While floating-point arithmetic is the default, the system provides optional rational-path (`ExactVoF`) support for validated geometric robustness. The arithmetic mode is collective and determined at the operation entry boundary; backends do not support silent fallback.

## State Ownership and Snapshots

- **Immutable Snapshots:** Adaptive mesh hierarchies and geometry providers own immutable state. Numerical operators and field adapters work on these snapshots, ensuring that solver trials do not overwrite accepted physical states until a successful commit.
- **Replayability:** A successful transport step or geometry update must be reproducible from the snapshot state and the exact source-ledger events, independent of external solver history.
- **Atomicity:** The solver state, including inventory ledgers and field snapshots, is published atomically. A failed transport operation (e.g., convergence failure in the pressure solver) leaves the accepted hierarchy and accepted fields completely unchanged.

## Component Integration

The transport system integrates with the broader multiphysics solver through:
- **Face-Stream Mapping:** The transport contract decouples donor-cell admission, PLIC reconstruction candidates, and face-flux integration, enabling modular updates for new mesh topologies.
- **Consistency Checks:** Every transport step performs an independent local extensive balance check to detect roundoff-induced errors before source evaluation.

For specific implementation contracts, see:
- **Conservative Face Transport:** [conservative_face_vof.md](conservative_face_vof.md)
- **General Native Transport:** [general_vof_transport.md](general_vof_transport.md)
- **Rational Arithmetic Contracts:** [exact_vof.md](exact_vof.md)
