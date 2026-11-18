# Steady Non-Orthogonal Diffusion Convergence Contract

This contract defines the convergence requirements for implicit non-orthogonal diffusion operators, ensuring consistency in the presence of tangential cross-diffusion corrections.

## Convergence Criteria

- **Dual-Gate Verification:** For meshes with tangential partition-face coupling, convergence requires satisfying both:
  1. **Update Tolerance:** The normalized infinite-norm of the correction field must be below the specified tolerance.
  2. **Residual Tolerance:** The global physical residual, excluding lagged partition-gradient corrections, must meet the specified $L_2$ tolerance after synchronizing gradients.
- **Physical Load Normalization:** The convergence residual is normalized against the fixed physical load (prescribed sources and boundary conditions). This ensures that the convergence criteria remain physically meaningful even for problems with near-zero homogeneous solutions.

## Architectural Requirements

- **Implicit Treatment:** The operator employs a matrix-RHS split where owned-cell gradients are assembled implicitly, and partition-face gradients are treated explicitly (lagged) and corrected through iterative sweeps.
- **Collective Validation:** Convergence controls, diffusivity fields, and correction-field presence must be rank-consistent. Inconsistent configurations are rejected collectively before assembly to prevent deadlocks or numerical divergence.
- **Iteration Management:** The framework uses an iterative correction loop. Linear solver failure or exhaustion of the correction iteration budget results in a convergence failure, leaving the system in a defined state.

## Integration Principles

- **Backward Compatibility:** Explicit and Hybrid correction modes retain their fixed-sweep contracts. The new convergence-based implicit contract is opt-in and provides a more robust closure for non-orthogonal domains.
- **Numerical Parity:** Convergence gates must be applied consistently across serial and MPI configurations, ensuring that decomposition invariance is maintained for the final converged solution.

For implementation, see the steady-state solve entry points in `src/solvers/`.
