# Coupled Nonlinear Solver Contract

The coupled nonlinear solver provides an integrated framework for velocity-pressure coupling using Newton-Krylov methods. It enables the simultaneous solution of nonlinear flow equations, providing improved convergence stability compared to segregated schemes for complex physics.

## Coupling Architecture

- **Operator Decomposition:** The solver decomposes the multiphysics system into a coupled nonlinear residual function ($F(u, p) = 0$). It supports both analytic Newton and Picard-based linearization strategies.
- **Backend Independence:** While the system currently provides a NOX-based backend, the architectural framework is designed to abstract the nonlinear iteration from the specific linear solver backend.
- **Transactional Consistency:** Each nonlinear solve is managed as an atomic operation. Iterations are performed on a scratch workspace; the global solver state is only updated upon successful convergence and physical acceptance of the solution.

## Solver Requirements

- **Residual and Jacobian Construction:** Physics models (momentum, continuity, species transport) must provide consistent residual evaluations. The nonlinear solver relies on these models to construct the coupled Jacobian matrix or evaluate directional derivatives.
- **Numerical Gates:** All nonlinear solutions are subject to absolute physical residual gates (e.g., integrated continuity error). The solver does not rely solely on internal linear solver convergence; nonlinear convergence must satisfy strict physical conservation requirements.
- **Resource Management:** Nonlinear solver workspaces (including cached Jacobians, Schur-complement preconditioners) are rank-local and must conform to pre-allocated memory budgets.

## Integration Principles

- **State Atomicity:** Successful convergence publishes the solution to the core solver's state. If a nonlinear iteration exceeds the allowed budget or violates physical tolerances (e.g., EOS occupancy limits), the solver triggers a full rollback of the tentative iteration fields.
- **Physical Consistency:** The solver respects independent physical bounds. During nonlinear iterations, fields must remain within pre-defined physical ranges (e.g., positive mass/moles) to ensure stable Jacobian evaluation.
- **Linear Backend Interoperability:** The coupled framework supports pluggable linear backends (e.g., GMRES, BiCGStab) to accelerate the inner linear system, while maintaining a segregated scalar transport policy for passive species.

For detailed solver configuration and verification targets, see:
- [VoF Solver Contracts](../verification/vof_solver_contracts.md)
