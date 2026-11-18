# Region Accuracy Qualification Contract

This contract defines the standard set of numerical tests required to qualify multi-region FVM operators for spatial accuracy and decomposition invariance.

## Qualification Fixtures

- **Nonlinear Refinement:** Exercises the reaction-diffusion operator on compact multi-region meshes. It verifies spatial accuracy and interface error convergence under varying mesh refinement levels and interface non-orthogonality.
- **Periodic Advection:** Transports a scalar field through translated periodic identifications. It verifies amplitude and phase error convergence, ensuring that periodic interface seams do not introduce damping or inventory drift.
- **Decomposition Invariance:** Ensures that the physical operator (e.g., flux-divergence) is invariant under domain decomposition. It compares results between one, two, and eight-region representations of an identical physical grid to establish that region seams do not alter the flux calculation.

## Acceptance Criteria

- **Error Convergence:** All qualification fixtures require strictly decreasing global and interface-band errors across refinement levels, with convergence rates meeting the established thresholds for the selected discretization (e.g., first-order upwind vs. second-order operators).
- **Residual Convergence:** Nonlinear operators must converge to a relative residual below the specified solver tolerances (e.g., $2 \times 10^{-10}$), ensuring that interface corrections are fully resolved.
- **Numerical Invariance:** Operator actions (e.g., divergence, flux evaluation) must be identical to within double-precision floating-point tolerances ($\approx 10^{-12}$) across different regional decomposition configurations.

## Maintenance Policy

- **Regression Control:** Each qualification fixture must be reproducible using the current toolchain. Performance and scaling measurements should remain separate from the accuracy verification suite.
- **Evidence Management:** Qualification records (e.g., logs, manifests) must be maintained in the build artifacts as durable proof of the model's accuracy, not embedded within the architectural documentation.

For workflow details, see:
- [Region performance measurement contract](region_scaling.md)
