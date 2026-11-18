# Coupled Operator Backend Contract

This document defines the architectural contract for the assembly and execution of coupled velocity-pressure operators within the multiphysics solver framework.

## Operator Backend Architecture

- **Block-Coupled Systems:** The solver supports the simultaneous assembly of velocity and pressure equations, forming a monolithic (but potentially sparse) operator system. This system ensures consistent pressure-velocity coupling without reliance on segregated iterations.
- **Backend Specialization:** The framework separates the numerical operator assembly (e.g., Assembled CSR vs. Block-Composite) from the iteration workspace policy (e.g., Cached vs. Streamed Products). This modularity allows for optimizing performance based on memory-bandwidth requirements.
- **Operator Authority:** The coupled operator backend acts as the authoritative source for the discretized system, managing the interaction between the momentum mass matrix, the divergence operator, and the pressure-continuity constraint.

## Algebraic and Boundary Constraints

- **Pressure Normalization:** The system enforces a pressure-gauge condition to handle the all-Neumann pressure constraint. In the absence of physical Dirichlet pressure boundaries, the solver automatically identifies a reference cell (usually the minimum GID) and enforces a zero-pressure reference condition.
- **Least-Squares Boundary Conditions:** Boundary treatment, particularly for slip-velocity and pressure-gradients, is integrated into the analytic operator construction. This ensures boundary consistency across all stages of the linearization.
- **Algebraic Consistency:** The construction of the Schur-complement system must remain consistent with the reconstructed face fluxes and phase-volume budgets established in the VoF transport stage.

## Integration and Performance

- **Memory Management:** The system employs rank-local, bounded-memory workspaces for linear operator construction. Backends (e.g., `StreamedProducts`) are optimized to limit memory overhead during the formation of Schur-complement terms by releasing intermediate product matrices immediately after accumulation.
- **Operator Validity:** Linear operators and their associated preconditioners are tied to the mesh and geometry epoch. Any change to the underlying mesh hierarchy, pressure-boundary conditions, or fluid material properties invalidates cached operators and triggers a refresh.
- **Linear Solver Reuse:** The system facilitates the reuse of operators and preconditioners across multiple nonlinear iterations, provided that the underlying nonlinear system’s linearization point has not shifted beyond acceptable bounds.

For verification and performance-measurement workflows, see:
- [Benchmarks](../benchmarks/coupled_operator/README.md)
- [VoF Solver Contracts](../verification/vof_solver_contracts.md)
