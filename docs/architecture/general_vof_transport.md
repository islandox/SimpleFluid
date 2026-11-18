# General-Native VoF Transport Contract

This document defines the architectural framework for general-native VoF transport, providing a unified interface for selecting transport families regardless of the underlying mesh backend.

## Transport Families

The architecture supports multiple transport families, each optimized for different physical regimes and topological requirements:

| Transport Family | Topological Context | Implementation Focus |
| --- | --- | --- |
| **Algebraic Face-Volume** | Fixed native meshes | Canonical face-based inventory exchange; efficient for source-free advection. |
| **Interface-Local Swept** | Moving mixed-domain meshes | Geometric face candidates with signed ledger updates for interface-sensitive motion. |
| **Geometric Swept-Tetrahedra** | High-fidelity moving domains | Complete swept-solid construction and exact positive recipient partitioning. |

Regardless of the selected backend, all families must adhere to the shared [VoF Transport Contract](vof_transport.md).

## Architectural Principles

- **Arithmetic Policy:** The arithmetic mode (`Ordinary`, `BoundedCertification`, or `Exact`) is a collective decision made at the operation entry boundary. Backends and transport families do not support silent fallback to lower-precision arithmetic.
- **Atomic Publication:** Every transport step, including inventory updates and face-stream ledgers, is published atomically. Failure in any step (e.g., pressure-solver convergence, occupancy violation) leaves the accepted physical state and hierarchy untouched.
- **Consistency Verification:** Before publication, the system executes an independent local extensive balance check. This check detects and mitigates roundoff-induced defects without resorting to inventory clipping or uncontrolled fraction renormalization.

## Integration Requirements

- **Phase-Resolution Policy:** Transport is subject to strict phase-resolution and support-geometry checks. If a cell fails these criteria, the system reports a `ResolutionRequired` failure, requiring an explicit regrid or reinitialization from the solver's adaptation controller.
- **Physical Acceptance Gates:** All transport operations are subject to global mass/species/energy conservation gates, which remain independent of the linear solver's convergence tolerances.

## Verification
The current verification status and the roadmap for upcoming capabilities are detailed in:
- [VoF Solver Contracts](../verification/vof_solver_contracts.md)
- [TODO.md](../../TODO.md#stage-7--general-vof-and-flow-qualification)
