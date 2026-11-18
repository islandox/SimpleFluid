# General VoF Pressure Projection Contract

The pressure projection provides the mechanical coupling for the incompressible VoF solver, enforcing mass conservation and enforcing the divergence-free condition via a pressure-driven correction.

## Modeling Approach

- **Coupled Projection:** The system solves a coupled velocity-pressure system that minimizes the difference between the predicted and corrected momentum fields, subject to the mass continuity constraint ($Bq = b$).
- **Canonical Face Integration:** Pressure and surface-tension forces share a common face mobility operator. Physical geometry is incorporated directly into the operator construction, ensuring consistency between the pressure gradient and the surface-tension force.
- **Thermodynamic Consistency:** The projection maintains the physical work identity, ensuring that changes in kinetic energy are consistent with the work done by pressure and surface-tension forces, accounting for caloric EOS responses where applicable.

## Architectural Requirements

- **Incompressibility:** The solver enforces local continuity using a variable-density face coefficient, coupling the pressure-correction to the phase distribution reconstructed by the VoF model.
- **Energy Conservation:** The projection must satisfy a published energy identity, which relates the kinetic energy change to the pressure-work and surface-tension work. This identity is used as a numerical consistency gate for solver acceptance.
- **Solvers and Workspaces:** The projection is an operation that takes the current solver state (momenta, phase inventories) and produces a velocity correction. This operation must be atomic: failure to converge or violate mass/occupancy bounds triggers a rollback of the solver state.

## Integration Principles

- **Predictor-Corrector:** The pressure correction is applied to a predicted momentum field. The correction does not accumulate pressure discrepancies across stages; each step is self-contained.
- **Physical Accuracy:** Projection gates (e.g., divergence residuals) are evaluated based on physical units of the integrated volume flux.
- **Resource Management:** Workspace usage (Krylov solver storage, operator caches) is bounded and must fit within rank-local limits.

For verification targets and coupled solver qualification, see:
- [VoF Solver Contracts](../verification/vof_solver_contracts.md)
